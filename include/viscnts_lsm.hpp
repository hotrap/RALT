#ifndef VISCNTS_LSM_H__
#define VISCNTS_LSM_H__

#include <boost/fiber/buffered_channel.hpp>

#include "alloc.hpp"
#include "cache.hpp"
#include "common.hpp"
#include "fileenv.hpp"
#include "hash.hpp"
#include "key.hpp"
#include "memtable.hpp"
#include "splay.hpp"
#include "chunk.hpp"
#include "writebatch.hpp"
#include "iterators.hpp"
#include "sst.hpp"
#include "deletedrange.hpp"

/**
 * The viscnts.
 *
 * The global structure
 * - EstimateLSM
 *  - SuperVision: A pointer to the newest version of multiple versions. it's written by compact_thread and read by many threads
 *    - Level: largest_ and tree_. largest_ stores the data after major compaction, and tree_ is the levels.
 *      - Partition: SST file and deleted range.
 *        - ImmutableFile: It stores information such as ranges, file_id, size, file handle and so on.
 *          - FileBlock: There are two file blocks in one immutable file, data block and index block.
 *                        Data in file blocks are divided into two parts. The first part stores kvpairs.
 *                        The second part stores the offsets of the kvpairs.
 *            - Chunk: one Chunk is a buffer of size kChunkSize, it stores kvpairs or offsets of kvpairs.
 *  - UnorderedBuf: An append-only lock-free buffer, which is sorted and flushed when it's full.
 *
 * [Insert]
 *  First, append into the unordered buffer. Then the buffer is flushed and becomes a Level. Levels then compacted into bigger levels.
 * - EstimateLSM.append
 *  - UnsortedBufferPtrs.append
 *    - UnsortedBuffer.append
 * - (If unsorted buffer is full, then the buffer is appended to a queue to wait for flush,
 *    if the queue is too large, then it waits for the compact_thread) SuperVersion.flush_bufs
 *
 *
 * [Scan]
 *  Get the iterators on each level. It first seek on the Partitions by their ranges, then seek the ImmutableFile. First it seeks the index block.
 *  Then it seeks the data block. The index block typically stores the (kIndexChunkSize * i)-th key.
 *
 * [Compact and Decay]
 *  See EstimateLSM::SuperVision::compact
 *  If it has determined which levels to compact, it first get the list of Paritions that overlaps with others, and then get the iterator.
 *  Then use Compact::flush to write new SSTables.
 *  If it's necessary to decay, then use Compact::decay_first.
 *  - Compaction
 *    - SeqIteratorSet get the iterators on multiple files
 *    - SSTBuilder
 *      - WriteBatch
 *        - AppendFile
 *
 * [Delete Range]
 *  Record the deleted range on each Partitions (because ImmutableFile is immutable semantically, so we don't store deleted range in them).
 *  We must consider the deleted range when scanning and range counting.
 *  While scanning, if it has a deleted range, then we must compare the next key to the deleted range.
 *  While range counting, we must count the number of kvpairs that is deleted in [L, R]. Because we don't change the file itself.
 *  Since the file is immutable, so we don't actually store and compare Slices, instead, we store and compare their ranks in the file.
 *
 *  It requires that there is no Delete Range operations during Scans.
 *
 * [TODOs]
 *  Check the usage of IndSlice, avoid calling IndSlice(const IndSlice&)
 *  A more efficient memory allocator
 *  Try to fix the performance issue.
 *
 * [Notes]
 * - The bottleneck is sorting, because we use only one thread for compacting.
 *   We can use multiple threads to do sorting and writing to file.
 *   We can also divide flush and compact into different threads, or divide computing and writing threads.
 * - Concurrency.
 *   We support multithreading appending.
 *   We atomically append kv pairs to unsorted buffer, then we append the full buffer to the queue under a lock.
 *   We support scanning while compacting. This is done by refcounts in SuperVersion, and compact_thread() is copy-on-write.
 *
 * [Notes]
 * - Compare with RocksDB, why we are slow? RocksDB "Bulk Load of keys in Random Order" 80MB/sec, "Bulk Load of keys in Sequential Order" 370MB/sec.
 * RocksDB "... in Random Order" 400MB/sec if it is configured to first load all data in L0 with compaction switched off and using an unsorted vector
 * memtable. Our performance is ~93MB/sec when inserting 6e8 random keys of 3e8 different kinds. (write 4492*18777920B data) ~218MB/sec when inserting
 * 6e8 sequential keys. (write 5665*18777920B data) Scan performance ~200MB/sec
 * - First, RocksDB is multi-threaded. And the set-up is 12 cores, 24 vCPUs (This is the configuration in the old performance benchmark in 2017, 8
 * vCPUs are set in the latest). However we use 4 cores, 8 vCPUs. And we only use multithread in SuperVersion->flush_bufs() (This will be optimized in
 * the future). And we don't flush buffers while compaction.
 * - Second, the value size of RocksDB is 800B, which means it needs less comparision to flush a file block and to read a file block. Although
 *    RA and WA are big, the bandwidth of SSD can be utilized better.
 * - But, RocksDB "Random Read" is about ~1.3e5 ops/sec. But we are 1.2e6 ops/sec. This throughput is enough.
 *
 * - I add two threads to deal with buffers, one flushes buffers to the disk and the other compacts these buffer to the main tree. Buffers cannot be
 * seen until they are compacted. I optimize the file dividing method in _divide_file(), surprisingly it improves performance a lot. Why? Maybe it's
 * because we don't need to open/close file handles because there're only 817 files vs 5641 files (file handle limit: 1024) ? The speed is ~377MB/sec,
 * it costs 39s to flush 3e8 keys (817 * 18888736B).
 * */

namespace viscnts_lsm {

const static size_t kSSTable = 1 << 24;
const static size_t kRatio = 10;
// about kMemTable... on average, we expect the size of index block < kDataBlockSize

int SKeyCompFunc(const SKey& A, const SKey& B) {
  if (A.len() != B.len()) return A.len() < B.len() ? -1 : 1;
  return memcmp(A.data(), B.data(), A.len());
}

using KeyCompType = int(const SKey&, const SKey&);

template <typename KeyCompT>
class Compaction {
  // builder_ is used to build one file
  // files_ is used to get global file name
  // env_ is used to get global environment
  // flag_ means whether lst_value_ is valid
  // lst_value_ is the last key value pair appended to builder_
  // vec_newfiles_ stores the information of new files.
  // - the ranges of files
  // - the size of files
  // - the filename of files
  // - the file_id of files
  // file_id is used if we have cache.
  SSTBuilder builder_;
  FileName* files_;
  Env* env_;
  bool flag_;
  std::pair<IndSKey, SValue> lst_value_;
  std::mt19937 rndgen_;
  KeyCompT comp_;
  double real_size_;
  double lst_real_size_;
  double decay_prob_;  // = 0.5 on default

  void _begin_new_file() {
    builder_.reset();
    auto [filename, id] = files_->next_pair();
    builder_.new_file(std::make_unique<WriteBatch>(std::unique_ptr<AppendFile>(env_->openAppFile(filename))));
    vec_newfiles_.emplace_back();
    vec_newfiles_.back().filename = filename;
    vec_newfiles_.back().file_id = id;
  }

  void _end_new_file() {
    builder_.make_index();
    builder_.finish();
    vec_newfiles_.back().size = builder_.size();
    vec_newfiles_.back().range = std::move(builder_.range());
    vec_newfiles_.back().decay_size = real_size_ - lst_real_size_;
    lst_real_size_ = real_size_;
  }

 public:
  struct NewFileData {
    std::string filename;
    size_t file_id;
    size_t size;
    std::pair<IndSKey, IndSKey> range;
    double decay_size;
  };
  Compaction(FileName* files, Env* env, KeyCompT comp) : files_(files), env_(env), flag_(false), rndgen_(std::random_device()()), comp_(comp) {
    decay_prob_ = 0.5;
  }

  template <typename TIter, typename FilterFunc>
  auto flush(TIter& left, FilterFunc&& filter_func) {
    vec_newfiles_.clear();
    _begin_new_file();
    // null iterator
    if (!left.valid()) {
      return std::make_pair(vec_newfiles_, 0.0);
    }
    flag_ = false;
    real_size_ = 0;
    lst_real_size_ = 0;
    int CNT = 0;
    // read first kv.
    {
      auto L = left.read();
      lst_value_ = {L.first, L.second};
      left.next();
    }
    while (left.valid()) {
      CNT++;
      auto L = left.read();
      if (comp_(lst_value_.first.ref(), L.first) == 0) {
        lst_value_.second += L.second;
      } else {
        // only store those filter returning true.
        if (filter_func(lst_value_)) {
          real_size_ += _calc_decay_value(lst_value_);
          builder_.append(lst_value_);
          _divide_file(DataKey(L.first, L.second).size());
        }
        lst_value_ = {L.first, L.second};
      }
      left.next();
    }
    // store the last kv.
    {
      builder_.set_lstkey(lst_value_.first);
      if (filter_func(lst_value_)) {
        builder_.append(lst_value_);
        real_size_ += _calc_decay_value(lst_value_); 
      } 
    }
    _end_new_file();
    return std::make_pair(vec_newfiles_, real_size_);
  }

  template <typename TIter>
  auto decay1(TIter& iters) {
    return flush(iters, [this](auto& kv){
      kv.second.counts *= decay_prob_;
      if (kv.second.counts < 1) {
        std::uniform_real_distribution<> dis(0, 1.);
        if (dis(rndgen_) < kv.second.counts) {
          return false;
        }
        kv.second.counts = 1;
      }
      return true;
    });
  }

  template<typename TIter>
  auto flush(TIter& left) {
    return flush(left, [](auto&) { return true; });
  }

 private:
  template <typename T>
  double _calc_decay_value(const std::pair<T, SValue>& kv) {
    return kv.first.len() + kv.second.vlen;
  }
  void _divide_file(size_t size) {
    if (builder_.kv_size() + size > kSSTable) {
      builder_.set_lstkey(lst_value_.first);
      _end_new_file();
      _begin_new_file();
    }
  }
  std::vector<NewFileData> vec_newfiles_;
};

constexpr auto kLimitMin = 10;
constexpr auto kLimitMax = 20;
constexpr auto kMergeRatio = 0.1;
constexpr auto kUnmergedRatio = 0.1;
constexpr auto kUnsortedBufferSize = kSSTable;
constexpr auto kUnsortedBufferMaxQueue = 4;
constexpr auto kMaxFlushBufferQueueSize = 10;
constexpr auto kWaitCompactionSleepMilliSeconds = 100;

template <typename CompactFunc, typename DecayFunc>
struct DefaultCompactPolicy {

};

template <typename KeyCompT>
class EstimateLSM {
  struct Partition {
    ImmutableFile<KeyCompT> file_;
    DeletedRange deleted_ranges_;
    int global_range_counts_;
    double decay_size_;
    std::string filename_;

   public:
    Partition(Env* env, BaseAllocator* file_alloc, KeyCompT comp, const typename Compaction<KeyCompT>::NewFileData& data)
        : file_(data.file_id, data.size, std::unique_ptr<RandomAccessFile>(env->openRAFile(data.filename)), file_alloc, data.range, comp),
          deleted_ranges_(),
          decay_size_(data.decay_size),
          filename_(data.filename) {
      global_range_counts_ = file_.counts();
    }
    ~Partition() {
      logger("delete, ", filename_);

      file_.remove();
    }

    SSTIterator<KeyCompT> seek(const SKey& key) { return SSTIterator(&file_, key); }
    SSTIterator<KeyCompT> begin() { return SSTIterator(&file_); }
    bool overlap(const SKey& lkey, const SKey& rkey) const { return file_.range_overlap({lkey, rkey}); }
    auto range() const { return file_.range(); }
    size_t size() const { return file_.size(); }
    size_t data_size() const { return file_.data_size(); }
    size_t counts() const { return global_range_counts_; }
    size_t range_count(const std::pair<SKey, SKey>& range) {
      auto rank_pair = file_.rank_pair(range);
      // logger("q[rank_pair]=", rank_pair.first, ",", rank_pair.second);
      return deleted_ranges_.deleted_counts(rank_pair);
    }
    size_t range_data_size(const std::pair<SKey, SKey>& range) {
      auto rank_pair = file_.rank_pair(range);
      // return deleted_ranges_.deleted_data_size(rank_pair);
      // Estimate
      return deleted_ranges_.deleted_counts(rank_pair) * (file_.size() / (double) file_.counts());
    }
    void delete_range(const std::pair<SKey, SKey>& range) {
      auto rank_pair = file_.rank_pair(range);
      logger("delete_range[rank_pair]=", rank_pair.first, ",", rank_pair.second);
      deleted_ranges_.insert({rank_pair.first, rank_pair.second});
      global_range_counts_ = file_.counts() - deleted_ranges_.sum();
    }
    double decay_size() { return decay_size_; }
    const DeletedRange& deleted_ranges() { return deleted_ranges_; }
    const ImmutableFile<KeyCompT>& file() { return file_; }
  };
  struct Level {
    std::vector<std::shared_ptr<Partition>> head_;
    size_t size_;
    size_t data_size_;
    double decay_size_;
    class LevelIterator {
      typename std::vector<std::shared_ptr<Partition>>::const_iterator vec_it_;
      typename std::vector<std::shared_ptr<Partition>>::const_iterator vec_it_end_;
      SSTIterator<KeyCompT> iter_;
      std::shared_ptr<std::vector<std::shared_ptr<Partition>>> vec_ptr_;
      DeletedRange::Iterator del_ranges_iterator_;

     public:
      LevelIterator() {}
      LevelIterator(const std::vector<std::shared_ptr<Partition>>& vec, uint32_t id, SSTIterator<KeyCompT>&& iter, DeletedRange::Iterator&& del_iter)
          : vec_it_(id >= vec.size() ? vec.end() : vec.begin() + id + 1),
            vec_it_end_(vec.end()),
            iter_(std::move(iter)),
            del_ranges_iterator_(std::move(del_iter)) {
        if (iter_.valid()) _del_next();
      }
      LevelIterator(const std::vector<std::shared_ptr<Partition>>& vec, uint32_t id) {
        if (id < vec.size()) {
          del_ranges_iterator_ = DeletedRange::Iterator(vec[id]->deleted_ranges());
          vec_it_ = vec.begin() + id + 1;
          vec_it_end_ = vec.end();
          iter_ = SSTIterator(vec[id]->begin());
        } else {
          vec_it_ = vec.end();
          vec_it_end_ = vec.end();
        }
        if (iter_.valid()) _del_next();
      }
      LevelIterator(std::shared_ptr<std::vector<std::shared_ptr<Partition>>> vec_ptr, uint32_t id, SSTIterator<KeyCompT>&& iter,
                    DeletedRange::Iterator&& del_iter)
          : vec_it_(id >= vec_ptr->size() ? vec_ptr->end() : vec_ptr->begin() + id + 1),
            vec_it_end_(vec_ptr->end()),
            iter_(std::move(iter)),
            vec_ptr_(std::move(vec_ptr)),
            del_ranges_iterator_(std::move(del_iter)) {
        if (iter_.valid()) _del_next();
      }
      LevelIterator(std::shared_ptr<std::vector<std::shared_ptr<Partition>>> vec_ptr, uint32_t id) : vec_ptr_(std::move(vec_ptr)) {
        if (id < vec_ptr_->size()) {
          del_ranges_iterator_ = DeletedRange::Iterator((*vec_ptr_)[id]->deleted_ranges());
          vec_it_ = vec_ptr_->begin() + id + 1;
          vec_it_end_ = vec_ptr_->end();
          iter_ = SSTIterator((*vec_ptr_)[id]->begin());
        } else {
          vec_it_ = vec_ptr_->end();
          vec_it_end_ = vec_ptr_->end();
        }

        if (iter_.valid()) _del_next();
      }
      bool valid() { return iter_.valid(); }
      void next() {
        iter_.next();
        if (!iter_.valid() && vec_it_ != vec_it_end_) {
          iter_ = SSTIterator(&(*vec_it_)->file());
          vec_it_++;
        }
        if (iter_.valid()) _del_next();
      }
      void read(DataKey& kv) { return iter_.read(kv); }

     private:
      void _del_next() {
        while (del_ranges_iterator_.valid()) {
          auto id = iter_.rank();
          auto new_id = del_ranges_iterator_.jump(id);
          if (id != new_id) {
            iter_.jump(new_id);
            if (!iter_.valid() && vec_it_ != vec_it_end_) {
              iter_ = SSTIterator(&(*vec_it_)->file());
              del_ranges_iterator_ = DeletedRange::Iterator((*vec_it_)->deleted_ranges());
              vec_it_++;
              continue;
            }
          }
          return;
        }
      }
    };
    Level() : size_(0), data_size_(0), decay_size_(0) {}
    Level(size_t size, double decay_size) : size_(size), decay_size_(decay_size) {}
    size_t size() const { return size_; }
    double decay_size() const { return decay_size_; }
    LevelIterator seek(const SKey& key, KeyCompT comp) const {
      int l = 0, r = head_.size() - 1, where = -1;
      while (l <= r) {
        int mid = (l + r) >> 1;
        if (comp(key, head_[mid]->range().second) <= 0)
          where = mid, r = mid - 1;
        else
          l = mid + 1;
      }
      if (where == -1) return LevelIterator();
      auto sst_iter = head_[where]->seek(key);
      return LevelIterator(head_, where, std::move(sst_iter), DeletedRange::Iterator(sst_iter.rank(), head_[where]->deleted_ranges()));
    }
    LevelIterator seek_to_first() const { return LevelIterator(head_, 0); }
    bool overlap(const SKey& lkey, const SKey& rkey, KeyCompT comp) const {
      if (!head_.size()) return false;
      int l = 0, r = head_.size() - 1, where = -1;
      while (l <= r) {
        int mid = (l + r) >> 1;
        if (comp(lkey, head_[mid]->range().second) <= 0) {
          where = mid, r = mid - 1;
        } else {
          l = mid + 1;
        }
      }
      if (where == -1) return false;
      return head_[where]->overlap(lkey, rkey);
    }
    void append_par(std::shared_ptr<Partition> par) {
      logger("[append_par]: ", par->size());
      size_ += par->size();
      data_size_ += par->data_size();
      decay_size_ += par->decay_size();
      head_.push_back(std::move(par));
    }

    size_t range_count(const std::pair<SKey, SKey>& range, KeyCompT comp) {
      auto [where_l, where_r] = _get_range_in_head(range, comp);
      // logger_printf("where[%d, %d]", where_l, where_r);
      if (where_l == -1 || where_r == -1 || where_l > where_r) return 0;

      if (where_l == where_r) {
        return head_[where_l]->range_count(range);
      } else {
        size_t ans = 0;
        for (int i = where_l + 1; i < where_r; i++) ans += head_[i]->counts();
        ans += head_[where_l]->range_count(range);
        ans += head_[where_r]->range_count(range);
        return ans;
      }
    }

    size_t range_data_size(const std::pair<SKey, SKey>& range, KeyCompT comp) {
      auto [where_l, where_r] = _get_range_in_head(range, comp);
      // logger_printf("where[%d, %d]", where_l, where_r);
      if (where_l == -1 || where_r == -1 || where_l > where_r) return 0;

      if (where_l == where_r) {
        return head_[where_l]->range_data_size(range);
      } else {
        size_t ans = 0;
        for (int i = where_l + 1; i < where_r; i++) ans += head_[i]->counts();
        ans += head_[where_l]->range_data_size(range);
        ans += head_[where_r]->range_data_size(range);
        return ans;
      }
    }

    void delete_range(const std::pair<SKey, SKey>& range, KeyCompT comp) {
      auto [where_l, where_r] = _get_range_in_head(range, comp);
      if (where_l == -1 || where_r == -1 || where_l > where_r) return;
      if (where_l == where_r)
        head_[where_l]->delete_range(range);
      else {
        head_[where_l]->delete_range(range);
        head_[where_r]->delete_range(range);
        head_.erase(head_.begin() + where_l + 1, head_.begin() + where_r);
      }
      decay_size_ = 0;
      for (auto& a : head_) decay_size_ += a->decay_size();
    }

   private:
    std::pair<int, int> _get_range_in_head(const std::pair<SKey, SKey>& range, KeyCompT comp) {
      int l = 0, r = head_.size() - 1, where_l = -1, where_r = -1;
      while (l <= r) {
        int mid = (l + r) >> 1;
        if (comp(range.first, head_[mid]->range().second) <= 0)
          where_l = mid, r = mid - 1;
        else
          l = mid + 1;
      }

      l = 0, r = head_.size() - 1;
      while (l <= r) {
        int mid = (l + r) >> 1;
        if (comp(head_[mid]->range().first, range.second) <= 0)
          where_r = mid, l = mid + 1;
        else
          r = mid - 1;
      }
      return {where_l, where_r};
    }
  };
  class SuperVersion {
    std::vector<std::shared_ptr<Level>> tree_;
    std::shared_ptr<Level> largest_;
    std::atomic<uint32_t> ref_;
    double decay_size_overestimate_;
    KeyCompT comp_;

    std::mutex* del_mutex_;

    using LevelIteratorSetT = SeqIteratorSet<typename Level::LevelIterator, KeyCompT>;

   public:
    SuperVersion(std::mutex* mutex, KeyCompT comp)
        : largest_(std::make_shared<Level>()), ref_(1), decay_size_overestimate_(0), comp_(comp), del_mutex_(mutex) {}
    void ref() {
      ref_++;
      // logger("ref count becomes (", this, "): ", ref_);
    }
    void unref() {
      // logger("ref count (", this, "): ", ref_);
      if (!--ref_) {
        std::unique_lock lck(*del_mutex_);
        delete this;
      }
    }

    // return lowerbound.
    LevelIteratorSetT seek(const SKey& key) const {
      LevelIteratorSetT ret(comp_);
      if (largest_.get() != nullptr) ret.push(largest_->seek(key, comp_));
      for (auto& a : tree_) ret.push(a->seek(key, comp_));
      return ret;
    }

    LevelIteratorSetT seek_to_first() const {
      LevelIteratorSetT ret(comp_);
      if (largest_.get() != nullptr) ret.push(largest_->seek_to_first());
      for (auto& a : tree_) ret.push(a->seek_to_first());
      return ret;
    } 
    
    // Now I use iterators to check key existence... can be optimized.
    bool search_key(const SKey& key) const {
      auto check = [&](auto iter) -> bool {
        DataKey kv;
        if (!iter.valid()) return false;
        iter.read(kv);
        return comp_(kv.key(), key) == 0;
      };
      if (largest_.get() != nullptr) {
        if (check(largest_->seek(key, comp_))) return true;
      }
      for (auto& a : tree_) if (check(a->seek(key, comp_))) return true;
      return false;
    }

    std::vector<std::shared_ptr<Level>> flush_bufs(const std::vector<UnsortedBuffer*>& bufs, FileName* filename, Env* env, BaseAllocator* file_alloc,
                                                   KeyCompT comp) {
      if (bufs.size() == 0) return {};
      std::vector<std::shared_ptr<Level>> ret_vectors;
      auto flush_func = [](FileName* filename, Env* env, KeyCompT comp, UnsortedBuffer* buf, std::mutex& mu, BaseAllocator* file_alloc,
                           std::vector<std::shared_ptr<Level>>& ret_vectors) {
        Compaction worker(filename, env, comp);
        buf->sort(comp);
        auto iter = std::make_unique<UnsortedBuffer::Iterator>(*buf);
        auto [files, decay_size] = worker.flush(*iter);
        auto level = std::make_shared<Level>(0, decay_size);
        for (auto& a : files) level->append_par(std::make_shared<Partition>(env, file_alloc, comp, a));
        delete buf;
        std::unique_lock lck(mu);
        ret_vectors.push_back(std::move(level));
      };
      std::vector<std::thread> thread_pool;
      std::mutex thread_mutex;
      // We now expect the number of SSTs is small, e.g. 4
      for (uint32_t i = 1; i < bufs.size(); i++) {
        thread_pool.emplace_back(flush_func, filename, env, comp, bufs[i], std::ref(thread_mutex), file_alloc, std::ref(ret_vectors));
      }
      if (bufs.size() >= 1) {
        flush_func(filename, env, comp, bufs[0], thread_mutex, file_alloc, std::ref(ret_vectors));
      }
      // for (int i = 0; i < bufs.size(); ++i) flush_func(filename, env, comp, bufs[i], thread_mutex, file_alloc, ret);
      for (auto& a : thread_pool) a.join();
      return ret_vectors;
    }
    SuperVersion* push_new_buffers(const std::vector<std::shared_ptr<Level>>& vec) {
      auto ret = new SuperVersion(del_mutex_, comp_);
      ret->tree_ = tree_;
      // logger(tree_.size(), ", ", ret->tree_.size());
      ret->largest_ = largest_;
      ret->decay_size_overestimate_ = decay_size_overestimate_;
      ret->tree_.insert(ret->tree_.end(), vec.begin(), vec.end());
      for (auto&& x : vec) {
        ret->decay_size_overestimate_ += x->decay_size();
      }
      ret->_sort_levels();
      return ret;
    }
    SuperVersion* compact(FileName* filename, Env* env, BaseAllocator* file_alloc, KeyCompT comp, bool trigger_decay = false) const {
      auto Lsize = 0;
      for (auto& a : tree_) Lsize += a->size();
      // check if overlap with any partition
      auto check_func = [comp, this](const Partition& par) {
        for (auto& a : tree_)
          if (a->overlap(par.range().first, par.range().second, comp)) return true;
        if (largest_ && largest_->overlap(par.range().first, par.range().second, comp)) return true;
        return false;
      };
      // add partitions that is overlapped with other levels.
      // the remaining partitions are stored in a std::vector.
      std::vector<std::shared_ptr<Partition>> rest;
      auto add_level = [&rest, &check_func](const Level& level) {
        auto for_iter_ptr = std::make_shared<std::vector<std::shared_ptr<Partition>>>();
        for (auto& par : level.head_) {
          if (check_func(*par)) {
            // logger("fuck<", par->size(), ">");
            for_iter_ptr->push_back(par);
          } else {
            // logger("fuck_rest<", par->size(), ">");
            rest.push_back(par);
          }
        }
        return for_iter_ptr;
      };
      // decay
      if (trigger_decay) {
        logger("[addr, decay size, Lsize, largest_->size()]: ", this, ", ", decay_size_overestimate_, ", ", Lsize, ", ", largest_->size());
        auto iters = std::make_unique<LevelIteratorSetT>(comp);

        std::string __print_str = "[";
        for (auto& a : tree_) {
          __print_str += std::to_string(a->size() / (double)kSSTable) + ", ";
        }
        logger(__print_str + "]");

        // logger("Major Compaction tree_.size() = ", tree_.size());
        // push all iterators to necessary partitions to SeqIteratorSet.
        for (auto& a : tree_) iters->push(a->seek_to_first());
        if (largest_ && largest_->size() != 0) iters->push(largest_->seek_to_first());
        iters->build();

        // compaction...
        Compaction worker(filename, env, comp);
        auto [files, decay_size] = worker.decay1(*iters);
        auto ret = new SuperVersion(del_mutex_, comp_);
        // major compaction, so levels except largest_ become empty.
        
        // std::shared_ptr move... maybe no problem.
        {
          std::unique_lock lck(*del_mutex_);
          for (auto& a : files) {
            auto par = std::make_shared<Partition>(env, file_alloc, comp, a);
            ret->largest_->append_par(std::move(par));
          }   
          // calculate new current decay size
          ret->decay_size_overestimate_ = ret->_calc_current_decay_size();
          logger("[new decay size]: ", ret->decay_size_overestimate_);
          logger("[new largest_]: ", ret->largest_->size());
          ret->_sort_levels();
        }
         
        return ret;
      } else {
        // similar to universal compaction in rocksdb
        // if tree_[i]->size()/(\sum_{j=0..i-1}tree_[j]->size()) <= kMergeRatio
        // then merge them.
        // if kMergeRatio ~ 1./X, then it can be treated as (X+1)-tired compaction

        // if the number of tables >= kLimitMin, then begin to merge
        // if the number of tables >= kLimitMax, then increase kMergeRatio. (this works when the number of tables is small)

        // logger("tree_.size() = ", tree_.size());

        std::string __print_str = "[";
        for (auto& a : tree_) {
          __print_str += std::to_string(a->size() / (double)kSSTable) + ", ";
        }
        logger(__print_str + "]");

        if (tree_.size() >= kLimitMin) {
          auto _kRatio = kMergeRatio;
          double min_ratio = 1e30;
          int where = -1, _where = -1;
          size_t sum = tree_.back()->size();
          for (int i = tree_.size() - 2; i >= 0; --i) {
            if (tree_[i]->size() <= sum * _kRatio)
              where = i;
            else if (auto t = tree_[i]->size() / (double)sum; min_ratio > t) {
              _where = i;
              min_ratio = t;
            }
            sum += tree_[i]->size();
          }
          if (tree_.size() >= kLimitMax && where == -1) {
            where = _where;
            size_t sum = tree_.back()->size();
            for (int i = tree_.size() - 2; i >= 0; --i) {
              if (tree_[i]->size() <= sum * min_ratio) where = i;
              sum += tree_[i]->size();
            }
          }
          if (where == -1) return nullptr;

          auto iters = std::make_unique<LevelIteratorSetT>(comp);

          // logger("compact(): where = ", where);
          // push all iterators to necessary partitions to SeqIteratorSet.
          for (uint32_t i = where; i < tree_.size(); ++i) iters->push(typename Level::LevelIterator(add_level(*tree_[i]), 0));
          iters->build();
          std::sort(rest.begin(), rest.end(), [comp](const std::shared_ptr<Partition>& x, const std::shared_ptr<Partition>& y) {
            return comp(x->range().first, y->range().second) < 0;
          });

          // logger("compact...");
          // compaction...
          Compaction worker(filename, env, comp);
          auto [files, decay_size] = worker.flush(*iters);
          auto ret = new SuperVersion(del_mutex_, comp_);
          // minor compaction?
          // decay_size_overestimate = largest_ + remaining levels + new level

          {  
            std::unique_lock lck(*del_mutex_);
            ret->tree_ = tree_;
            ret->largest_ = largest_;
            for (int i = 0; i < where; i++) ret->decay_size_overestimate_ += tree_[i]->decay_size();
            auto rest_iter = rest.begin();
            auto level = std::make_shared<Level>();
            for (auto& a : files) {
              auto par = std::make_shared<Partition>(env, file_alloc, comp, a);
              while (rest_iter != rest.end() && comp((*rest_iter)->range().first, par->range().first) <= 0) {
                level->append_par(std::move(*rest_iter));
                rest_iter++;
              }
              level->append_par(std::move(par));
            }  
            while (rest_iter != rest.end()) level->append_par(std::move(*rest_iter++));
            ret->tree_.erase(ret->tree_.begin() + where, ret->tree_.end());
            ret->tree_.push_back(std::move(level));
            // calculate new current decay size
            ret->decay_size_overestimate_ = ret->_calc_current_decay_size();
            ret->_sort_levels();
            return ret;
          }

        }
        return nullptr;
      }
    }

    size_t range_count(const std::pair<SKey, SKey>& range, KeyCompT comp) {
      size_t ans = 0;
      if (largest_) ans += largest_->range_count(range, comp);
      for (auto& a : tree_) ans += a->range_count(range, comp);
      return ans;
    }

    size_t range_data_size(const std::pair<SKey, SKey>& range, KeyCompT comp) {
      size_t ans = 0;
      if (largest_) ans += largest_->range_data_size(range, comp);
      for (auto& a : tree_) ans += a->range_data_size(range, comp);
      return ans;
    }

    SuperVersion* delete_range(std::pair<SKey, SKey> range, KeyCompT comp) {
      auto ret = new SuperVersion(del_mutex_, comp_);
      ret->tree_ = tree_;
      ret->largest_ = largest_;
      ret->decay_size_overestimate_ = decay_size_overestimate_;
      if (ret->largest_) ret->largest_->delete_range(range, comp);
      for (auto& a : ret->tree_) a->delete_range(range, comp);
      return ret;
    }

    double get_current_decay_size() { return decay_size_overestimate_; }

   private:
    double _calc_current_decay_size() {
      double ret = 0;
      if (largest_) ret += largest_->decay_size();
      for (auto& a : tree_) ret += a->decay_size();
      return ret;
    }
    void _sort_levels() {
      std::sort(tree_.begin(), tree_.end(), [](const std::shared_ptr<Level>& x, const std::shared_ptr<Level>& y) { return x->size() > y->size(); });
    }
  };
  std::unique_ptr<Env> env_;
  std::unique_ptr<FileName> filename_;
  std::unique_ptr<BaseAllocator> file_alloc_;
  KeyCompT comp_;
  SuperVersion* sv_;
  std::thread compact_thread_;
  std::thread flush_thread_;
  std::atomic<bool> terminate_signal_;
  UnsortedBufferPtrs bufs_;
  std::mutex sv_mutex_;
  std::mutex sv_load_mutex_;
  // Only one thread can modify sv.
  std::mutex sv_modify_mutex_;
  std::vector<std::shared_ptr<Level>> flush_buf_vec_;
  std::mutex flush_buf_vec_mutex_;
  std::condition_variable signal_flush_to_compact_;
  double decay_size_overestimate_{0};
  size_t sv_tick_{0};
  // 0: idle, 1: working
  uint8_t flush_thread_state_{0};
  uint8_t compact_thread_state_{0};

  // Statistics.
  struct Statistics {
    std::vector<double> decay_size_in_time_;

  } stats;

 public:
  class SuperVersionIterator {
    using LevelIteratorSetTForScan = SeqIteratorSetForScan<typename Level::LevelIterator, KeyCompT>;
    LevelIteratorSetTForScan iter_;
    SuperVersion* sv_;

   public:
    SuperVersionIterator(LevelIteratorSetTForScan&& iter, SuperVersion* sv) : iter_(std::move(iter)), sv_(sv) { iter_.build(); }
    ~SuperVersionIterator() { sv_->unref(); }
    auto read() { return iter_.read(); }
    auto valid() { return iter_.valid(); }
    auto next() { return iter_.next(); }
  };
  EstimateLSM(std::unique_ptr<Env>&& env, std::unique_ptr<FileName>&& filename, std::unique_ptr<BaseAllocator>&& file_alloc,
              KeyCompT comp)
      : env_(std::move(env)),
        filename_(std::move(filename)),
        file_alloc_(std::move(file_alloc)),
        comp_(comp),
        sv_(new SuperVersion(&sv_mutex_, comp)),
        terminate_signal_(0),
        bufs_(kUnsortedBufferSize, kUnsortedBufferMaxQueue) {
    compact_thread_ = std::thread([this]() { compact_thread(); });
    flush_thread_ = std::thread([this]() { flush_thread(); });
  }
  ~EstimateLSM() {
    {
      terminate_signal_ = 1;
      std::unique_lock lck0(flush_buf_vec_mutex_);
      std::unique_lock lck1(sv_modify_mutex_);
      std::unique_lock lck2(sv_load_mutex_);
      bufs_.terminate();
      signal_flush_to_compact_.notify_one();  
    }
    compact_thread_.join();
    flush_thread_.join();
    sv_->unref();
  }
  void append(const SKey& key, const SValue& value) { bufs_.append_and_notify(key, value); }
  auto seek(const SKey& key) {
    auto sv = get_current_sv();
    return new SuperVersionIterator(sv->seek(key), sv);
  }

  auto seek_to_first() {
    auto sv = get_current_sv();
    return new SuperVersionIterator(sv->seek_to_first(), sv);
  }

  size_t range_count(const std::pair<SKey, SKey>& range) {
    if (comp_(range.first, range.second) > 0) return 0;
    auto sv = get_current_sv();
    auto ret = sv->range_count(range, comp_);
    sv->unref();
    return ret;
  }

  size_t range_data_size(const std::pair<SKey, SKey>& range) {
    if (comp_(range.first, range.second) > 0) return 0;
    auto sv = get_current_sv();
    auto ret = sv->range_data_size(range, comp_);
    sv->unref();
    return ret;
  }

  void all_flush() {
    bufs_.flush();
    wait();
  }

  void wait() {
    while (true) {
      {  
        std::unique_lock lck0(bufs_.get_mutex());
        std::unique_lock lck1(flush_buf_vec_mutex_);
        std::unique_lock lck2(sv_modify_mutex_);
        if (!bufs_.size() && flush_buf_vec_.empty() && compact_thread_state_ == 0 && flush_thread_state_ == 0) {
          return;
        }
      }
      if (terminate_signal_) return;
      std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(kWaitCompactionSleepMilliSeconds));
    }
  }

  void delete_range(std::pair<SKey, SKey> range) {
    if (comp_(range.first, range.second) > 0) return;
    std::unique_lock del_range_lck(sv_modify_mutex_);
    auto new_sv = sv_->delete_range(range, comp_);
    _update_superversion(new_sv);
  }

  void trigger_decay() {
    std::unique_lock del_range_lck(sv_modify_mutex_);
    auto new_sv = sv_->compact(filename_.get(), env_.get(), file_alloc_.get(), comp_, true);
    _update_superversion(new_sv);
  }

  double get_current_decay_size() {
    return decay_size_overestimate_;
  }

  bool search_key(const SKey& key) {
    auto sv = get_current_sv();
    return sv->search_key(key);
  }

 private:
  void flush_thread() {
    while (!terminate_signal_) {
      flush_thread_state_ = 0;
      auto buf_q_ = bufs_.wait_and_get();
      if (terminate_signal_) return;
      if (buf_q_.empty()) continue;
      flush_thread_state_ = 1;
      auto new_vec = sv_->flush_bufs(buf_q_, filename_.get(), env_.get(), file_alloc_.get(), comp_);
      while (new_vec.size() + flush_buf_vec_.size() > kMaxFlushBufferQueueSize) {
        logger("full");
        std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(kWaitCompactionSleepMilliSeconds));
        if (terminate_signal_) return;
      }
      {
        std::unique_lock lck(flush_buf_vec_mutex_);
        flush_buf_vec_.insert(flush_buf_vec_.end(), new_vec.begin(), new_vec.end());
        signal_flush_to_compact_.notify_one();  
      }
    }
  }
  void compact_thread() {
    while (!terminate_signal_) {
      compact_thread_state_ = 0;
      SuperVersion* new_sv;
      {
        std::unique_lock flush_lck(flush_buf_vec_mutex_);
        // wait buffers from flush_thread()
        signal_flush_to_compact_.wait(flush_lck, [&]() { return !flush_buf_vec_.empty() || terminate_signal_; });
        if (terminate_signal_) return;
        compact_thread_state_ = 1;
        sv_modify_mutex_.lock();
        new_sv = sv_->push_new_buffers(flush_buf_vec_);
        flush_buf_vec_.clear();
      }

      auto last_compacted_sv = new_sv;
      while (true) {
        auto new_compacted_sv = last_compacted_sv->compact(filename_.get(), env_.get(), file_alloc_.get(), comp_);
        if (new_compacted_sv == nullptr) {
          break;
        } else {
          last_compacted_sv->unref();
          last_compacted_sv = new_compacted_sv;
        }
      }
      _update_superversion(last_compacted_sv);
      sv_modify_mutex_.unlock();
    }
  }

  void _update_superversion(SuperVersion* new_sv) {
    if (new_sv != nullptr) {
      sv_load_mutex_.lock();
      auto old_sv = sv_;
      sv_ = new_sv;
      decay_size_overestimate_ = new_sv->get_current_decay_size();
      sv_tick_++;
      sv_load_mutex_.unlock();
      old_sv->unref();
    }
  }

  SuperVersion* get_current_sv() {
    sv_load_mutex_.lock();
    auto sv = sv_;
    sv->ref();
    sv_load_mutex_.unlock();
    return sv;
  }
};

// Viscnts, implement lsm tree and other things.

template <typename KeyCompT>
class VisCnts {
  std::unique_ptr<EstimateLSM<KeyCompT>> tree[2];
  KeyCompT comp_;
  double decay_limit_;
  std::mutex decay_m_;

 public:
  // Use different file path for two trees.
  VisCnts(KeyCompT comp, const std::string& path, double delta)
    : tree{std::make_unique<EstimateLSM<KeyCompT>>(std::unique_ptr<Env>(createDefaultEnv()), std::make_unique<FileName>(0, path), std::make_unique<DefaultAllocator>(), comp), 
          std::make_unique<EstimateLSM<KeyCompT>>(std::unique_ptr<Env>(createDefaultEnv()), std::make_unique<FileName>(0, path + "a"), std::make_unique<DefaultAllocator>(), comp)},
      comp_(comp), decay_limit_(delta) {}
  void access(int tier, const std::pair<SKey, SValue>& kv) { tree[tier]->append(kv.first, kv.second); check_decay(); }
  auto delete_range(int tier, const std::pair<SKey, SKey>& range) { return tree[tier]->delete_range(range); }
  auto seek_to_first(int tier) { return tree[tier]->seek_to_first(); }
  auto seek(int tier, const SKey& key) { return tree[tier]->seek(key); }
  auto weight_sum(int tier) { return tree[tier]->get_current_decay_size(); }
  auto range_data_size(int tier, const std::pair<SKey, SKey>& range) { return tree[tier]->range_data_size(range); }
  bool is_hot(int tier, const SKey& key) {
    return tree[tier]->search_key(key);
  }
  void transfer_range(int src_tier, int dst_tier, const std::pair<SKey, SKey>& range, std::pair<bool, bool> excluded) {
    if (src_tier == dst_tier) return;
    if (comp_(range.first, range.second) > 0) return;
    auto iter = tree[src_tier]->seek(range.first);
    if (excluded.first) {
      auto L = iter->read();
      if (comp_(L.first, range.first) == 0) {
        iter->next();
      }
    }
    int cnt = 0;
    while(iter->valid()) {
      auto L = iter->read();
      if ((excluded.second && (comp_(L.first, range.second) >= 0)) || (!excluded.second && comp_(L.first, range.second) > 0)) {
        break;
      }
      cnt++;
      tree[dst_tier]->append(L.first, L.second);
      iter->next();
    }
    printf("[%d]",cnt);
    delete iter;
  }
  void check_decay() {
    if (weight_sum(0) + weight_sum(1) > decay_limit_) {
      std::unique_lock lck(decay_m_);
      if (weight_sum(0) + weight_sum(1) > decay_limit_) {
        tree[0]->trigger_decay();
        tree[1]->trigger_decay();  
      }
    }
  }
  void flush() {
    for(auto& a : tree) a->all_flush();
  }
};

}  // namespace viscnts_lsm

#endif