#ifndef VISCNTS_LSM_H__
#define VISCNTS_LSM_H__

#include <pthread.h>

#include <future>
#include <memory>
#include <queue>

#include "alloc.hpp"
#include "bloomfilter.hpp"
#include "cache.hpp"
#include "chunk.hpp"
#include "common.hpp"
#include "compaction.hpp"
#include "deletedrange.hpp"
#include "fileenv.hpp"
#include "hash.hpp"
#include "iterators.hpp"
#include "key.hpp"
#include "kthest.hpp"
#include "memtable.hpp"
#include "options.h"
#include "splay.hpp"
#include "sst.hpp"
#include "writebatch.hpp"

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
 *  // Then it seeks the data block. The index block typically stores the (kIndexChunkSize * i)-th key. <-- I don't use this now.
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

namespace ralt {

constexpr auto kLimitMin = 5;
constexpr auto kLimitMax = 5;
constexpr auto kMergeRatio = 0.1;
constexpr auto kUnsortedBufferSize = 1ull << 24;
constexpr auto kUnsortedBufferMaxQueue = 1;
constexpr auto kMaxFlushBufferQueueSize = 10;
constexpr auto kWaitCompactionSleepMilliSeconds = 100;
constexpr auto kLevelMultiplier = 10;
constexpr auto kStepDecayLen = 10;
constexpr auto kExtraBufferMultiplier = 20;

constexpr size_t kEstPointNum = 1e4;
constexpr double kHotSetExceedLimit = 0.1;
constexpr double kPhyExceedLimit = 0.1;

template<typename T>
class atomic_shared_ptr {
 public:
  atomic_shared_ptr() = default;
  atomic_shared_ptr(const atomic_shared_ptr& p) {
    ptr_ = p.ptr_;
    ref_ = p.ref_;
    if (ptr_) {
      p.ref_->fetch_add(1, std::memory_order_relaxed);
    }
  }
  atomic_shared_ptr(atomic_shared_ptr&& p) {
    ptr_ = p.ptr_;
    ref_ = p.ref_;
    p.ptr_ = nullptr;
    p.ref_ = nullptr;
  }
  atomic_shared_ptr(T* p) : ptr_(p) {
    if (ptr_ != nullptr) {
      ref_ = new std::atomic<int>(1);
    }
  }
  ~atomic_shared_ptr() {
    if (ptr_) {
      release();
    }
  }
  atomic_shared_ptr& operator=(const atomic_shared_ptr& p) {
    if (p.ptr_) {
      p.ref_->fetch_add(1, std::memory_order_relaxed);
    }
    if (ptr_) {
      release();
    }
    ptr_ = p.ptr_;
    ref_ = p.ref_;
    return *this;
  }
  atomic_shared_ptr& operator=(atomic_shared_ptr&& p) {
    if (ptr_) {
      release();
    }
    ptr_ = p.ptr_;
    ref_ = p.ref_;
    p.ptr_ = nullptr;
    p.ref_ = nullptr;
    return *this;
  }
  T* operator->() {
    return ptr_;
  }
  const T* operator->() const {
    return ptr_;
  }
  
  T& operator*() {
    return *ptr_;
  }
  const T& operator*() const {
    return *ptr_;
  }
  operator bool() const {
    return ptr_ != nullptr;
  }
  void release() {
    if (ref_->fetch_sub(1, std::memory_order_relaxed) == 1) {
      delete ptr_;
      delete ref_;
      ptr_ = nullptr;
      ref_ = nullptr;
    }
  }

 private:
  std::atomic<int>* ref_{nullptr};
  T* ptr_{nullptr};
};

template<typename T, typename... Args>
atomic_shared_ptr<T> make_atomic_shared(Args&&... args) {
  return atomic_shared_ptr<T>(new T(std::forward<Args>(args)...));
}

template <typename KeyCompT, typename ValueT, typename IndexDataT>
class EstimateLSM {
  struct Partition {
    std::shared_ptr<const Options> options_;
    ImmutableFile<KeyCompT, ValueT, IndexDataT> file_;
    DeletedRange deleted_ranges_;
    int global_range_counts_;
    double avg_hot_size_;
    double global_hot_size_;
    size_t key_n_;
    size_t real_phy_size_{0};
    std::string filename_;
    IndSlice check_hot_buffer_;
    IndSlice check_stably_hot_buffer_;

   public:
    Partition(
        std::shared_ptr<const Options> options, Env* env,
        FileChunkCache* file_index_cache, FileChunkCache* file_key_cache,
        KeyCompT comp,
        typename Compaction<KeyCompT, ValueT, IndexDataT>::NewFileData&& data)
        : options_(std::move(options)),
          file_(
              data.file_id, data.size,
              std::unique_ptr<RandomAccessFile>(env->openRAFile(data.filename)),
              data.range, file_index_cache, file_key_cache, comp),
          deleted_ranges_(),
          global_hot_size_(data.hot_size),
          key_n_(data.key_n),
          real_phy_size_(data.real_phy_size),
          filename_(data.filename),
          check_hot_buffer_(std::move(data.check_hot_buffer)),
          check_stably_hot_buffer_(std::move(data.check_stably_hot_buffer)) {
      DB_ASSERT(options_);
      global_range_counts_ = file_.counts();
      avg_hot_size_ = data.hot_size / (double)file_.counts();
    }
    ~Partition() {
      logger("delete, ", filename_);
      file_.remove();
    }

    SSTIterator<KeyCompT, ValueT> seek(SKey key) const { return SSTIterator<KeyCompT, ValueT>(file_.seek(key)); }
    SSTIterator<KeyCompT, ValueT> begin() const { return SSTIterator<KeyCompT, ValueT>(file_.seek_to_first()); }
    bool overlap(SKey lkey, SKey rkey) const { return file_.range_overlap({lkey, rkey}); }
    auto range() const { return file_.range(); }
    size_t size() const { return file_.size(); }
    size_t data_size() const { return file_.data_size(); }
    size_t counts() const { return global_range_counts_; }
    size_t range_count(const std::pair<SKey, SKey>& range) {
      auto rank_pair = file_.rank_pair(range, {0, 0});
      // logger("q[rank_pair]=", rank_pair.first, ",", rank_pair.second);
      return deleted_ranges_.deleted_counts(rank_pair);
    }
    size_t range_data_size(const std::pair<SKey, SKey>& range) {
      // auto rank_pair = file_.rank_pair(range, {0, 0});
      // return deleted_ranges_.deleted_data_size(rank_pair);
      // Estimate
      // return deleted_ranges_.deleted_counts(rank_pair) * avg_hot_size_;
      return file_.estimate_range_hot_size(range);
    }
    void delete_range(const std::pair<SKey, SKey>& range, const std::pair<bool, bool> exclude_info) {
      auto rank_pair = file_.rank_pair(range, exclude_info);
      logger("delete_range[rank_pair]=", rank_pair.first, ",", rank_pair.second);
      deleted_ranges_.insert({rank_pair.first, rank_pair.second});
      global_range_counts_ = file_.counts() - deleted_ranges_.sum();
      global_hot_size_ = global_range_counts_ * avg_hot_size_;
    }
    double hot_size() const { return global_hot_size_; }
    const DeletedRange& deleted_ranges() const { return deleted_ranges_; }
    const ImmutableFile<KeyCompT, ValueT, IndexDataT>& file() const { return file_; }
    bool check_stably_hot(SKey key) const {
      BloomFilter bf(options_->bloom_bits);
      return bf.Find(key, check_stably_hot_buffer_.ref());
    }
    bool check_hot(SKey key) const {
      BloomFilter bf(options_->bloom_bits);
      return bf.Find(key, check_hot_buffer_.ref());
    }
    size_t get_key_n() const {
      return key_n_;
    }
    size_t get_real_phy_size() const {
      return real_phy_size_;
    }
    // void update_hot_size_by_tick_threshold(TickFilter<ValueT> tick_filter) {
    //   double new_hot_size = 0;
    //   size_t new_counts = 0;
    //   for(auto iter = begin(); iter.valid(); iter.next()) {
    //     BlockKey<SKey, ValueT> kv;
    //     iter.read(kv);
    //     if (tick_filter.check(kv.value())) {
    //       new_hot_size += kv.size();
    //       new_counts += 1;
    //     }
    //   }
    //   logger(global_hot_size_, ", ", new_hot_size);
    //   // FIXME: wrong counts
    //   global_range_counts_ = new_counts;
    //   avg_hot_size_ = new_counts == 0 ? new_hot_size / (double) new_counts : 0;
    //   global_hot_size_ = new_hot_size;
    // }
  };
  class Level {
    std::vector<atomic_shared_ptr<Partition>> head_;
    size_t size_{0};
    size_t data_size_{0};
    double hot_size_{0};
    size_t key_n_{0};
    size_t real_phy_size_{0};
  public:
    // iterator for level
    class LevelIterator {
      typename std::vector<atomic_shared_ptr<Partition>>::const_iterator vec_it_;
      typename std::vector<atomic_shared_ptr<Partition>>::const_iterator vec_it_end_;
      SSTIterator<KeyCompT, ValueT> iter_;
      atomic_shared_ptr<std::vector<atomic_shared_ptr<Partition>>> vec_ptr_;
      DeletedRange::Iterator del_ranges_iterator_;

     public:
      LevelIterator() {}
      LevelIterator(const std::vector<atomic_shared_ptr<Partition>>& vec, uint32_t id, SSTIterator<KeyCompT, ValueT>&& iter, DeletedRange::Iterator&& del_iter)
          : vec_it_(id >= vec.size() ? vec.end() : vec.begin() + id + 1),
            vec_it_end_(vec.end()),
            iter_(std::move(iter)),
            del_ranges_iterator_(std::move(del_iter)) {
        if (iter_.valid()) _del_next();
      }
      LevelIterator(const std::vector<atomic_shared_ptr<Partition>>& vec, uint32_t id) {
        if (id < vec.size()) {
          del_ranges_iterator_ = DeletedRange::Iterator(vec[id]->deleted_ranges());
          vec_it_ = vec.begin() + id + 1;
          vec_it_end_ = vec.end();
          iter_ = SSTIterator<KeyCompT, ValueT>(vec[id]->file().seek_to_first());
        } else {
          vec_it_ = vec.end();
          vec_it_end_ = vec.end();
        }
        if (iter_.valid()) _del_next();
      }
      LevelIterator(atomic_shared_ptr<std::vector<atomic_shared_ptr<Partition>>> vec_ptr, uint32_t id, SSTIterator<KeyCompT, ValueT>&& iter,
                    DeletedRange::Iterator&& del_iter)
          : vec_it_(id >= vec_ptr->size() ? vec_ptr->end() : vec_ptr->begin() + id + 1),
            vec_it_end_(vec_ptr->end()),
            iter_(std::move(iter)),
            vec_ptr_(std::move(vec_ptr)),
            del_ranges_iterator_(std::move(del_iter)) {
        if (iter_.valid()) _del_next();
      }
      LevelIterator(atomic_shared_ptr<std::vector<atomic_shared_ptr<Partition>>> vec_ptr, uint32_t id) : vec_ptr_(std::move(vec_ptr)) {
        if (id < vec_ptr_->size()) {
          del_ranges_iterator_ = DeletedRange::Iterator((*vec_ptr_)[id]->deleted_ranges());
          vec_it_ = vec_ptr_->begin() + id + 1;
          vec_it_end_ = vec_ptr_->end();
          iter_ = SSTIterator<KeyCompT, ValueT>((*vec_ptr_)[id]->file().seek_to_first());
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
          iter_ = SSTIterator<KeyCompT, ValueT>((*vec_it_)->file().seek_to_first());
          del_ranges_iterator_ = DeletedRange::Iterator((*vec_it_)->deleted_ranges());
          vec_it_++;
        }
        if (iter_.valid()) _del_next();
      }
      void read(BlockKey<SKey, ValueT>& kv) { return iter_.read(kv); }

     private:
      void _del_next() {
        while (del_ranges_iterator_.valid()) {
          auto id = iter_.rank();
          auto new_id = del_ranges_iterator_.jump(id);
          if (id != new_id) {
            iter_.jump(new_id);
            if (!iter_.valid() && vec_it_ != vec_it_end_) {
              iter_ = SSTIterator<KeyCompT, ValueT>((*vec_it_)->file().seek_to_first());
              del_ranges_iterator_ = DeletedRange::Iterator((*vec_it_)->deleted_ranges());
              vec_it_++;
              continue;
            }
          }
          return;
        }
      }
    };

    // Used in batch_seek
    // class LevelAsyncSeekHandle {
    //   public:
    //     LevelAsyncSeekHandle(SKey key, const Partition& part, const Level& level, int where, async_io::AsyncIOQueue& aio)
    //       : part_(part), level_(level), where_(where), sst_handle_(key, part.file(), aio) {}
    //     void init() {
    //       sst_handle_.init();
    //     }
    //     /* the chunk has been read. it returns the iterator when it has found result. */
    //     template<typename T>
    //     std::optional<LevelIterator> next(T&& aio_info) {
    //       auto result = sst_handle_.next(std::forward<T>(aio_info));
    //       if(!result) {
    //         return {};
    //       }
    //       auto del_range_iter = DeletedRange::Iterator(result.value().rank(), part_.deleted_ranges());
    //       return LevelIterator(level_.head_, where_, std::move(result.value()), std::move(del_range_iter));
    //     }
    //   private:
    //     const Partition& part_;
    //     const Level& level_;
    //     int where_;
    //     typename ImmutableFile<KeyCompT, ValueT>::ImmutableFileAsyncSeekHandle sst_handle_;
    // };

    Level() : size_(0), data_size_(0), hot_size_(0) {}
    size_t size() const { return size_; }
    double hot_size() const { return hot_size_; }
    /* seek the first key >= key. */
    LevelIterator seek(SKey key, KeyCompT comp) const {
      if (head_.size() == 1 && head_[0]->range().first.data() == nullptr) {
        assert(head_[0]->range().second.data() == nullptr);
        return LevelIterator();
      }
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
    
    bool check_hot(SKey key, KeyCompT comp) const {
      if (head_.size() == 1 && head_[0]->range().first.data() == nullptr) {
        assert(head_[0]->range().second.data() == nullptr);
        return false;
      }
      int l = 0, r = head_.size() - 1, where = -1;
      while (l <= r) {
        int mid = (l + r) >> 1;
        if (comp(key, head_[mid]->range().second) <= 0)
          where = mid, r = mid - 1;
        else
          l = mid + 1;
      }
      if (where == -1) return false;
      return head_[where]->check_hot(key);
    }
    
    bool check_stably_hot(SKey key, KeyCompT comp) const {
      if (head_.size() == 1 && head_[0]->range().first.data() == nullptr) {
        assert(head_[0]->range().second.data() == nullptr);
        return false;
      }
      int l = 0, r = head_.size() - 1, where = -1;
      while (l <= r) {
        int mid = (l + r) >> 1;
        if (comp(key, head_[mid]->range().second) <= 0)
          where = mid, r = mid - 1;
        else
          l = mid + 1;
      }
      if (where == -1) return false;
      return head_[where]->check_stably_hot(key);
    }
    size_t get_real_phy_size() const {
      return real_phy_size_;
    }
    // Get minimal overlapping SSTs.
    std::vector<atomic_shared_ptr<Partition>> get_min_overlap_pars(const Level& next_level, int cnt, KeyCompT comp) {
      int lp = 0, rp = 0, mn = 1e9, mni = 0;
      for (int i = cnt - 1; i < head_.size(); i++) {
        while(lp < next_level.head_.size() && comp(next_level.head_[lp]->range().second, head_[i - cnt + 1]->range().first) < 0) lp++;
        while(rp < next_level.head_.size() && comp(next_level.head_[rp]->range().first, head_[i]->range().second) <= 0) rp++;
        if (mn > rp - lp) {
          mn = rp - lp;
          mni = i - cnt + 1;
        }
      }
      auto ret = std::vector<atomic_shared_ptr<Partition>>(head_.begin() + mni, head_.begin() + std::min<size_t>(mni + cnt, head_.size()));
      for (auto& par : ret) {
        size_ -= par->size();
        data_size_ -= par->data_size();
        hot_size_ -= par->hot_size();
        key_n_ -= par->get_key_n();
        real_phy_size_ -= par->get_real_phy_size();
      }
      head_.erase(head_.begin() + mni, head_.begin() + std::min<size_t>(mni + cnt, head_.size()));
      return ret;
    }
    std::vector<atomic_shared_ptr<Partition>> get_overlapped_pars(const std::pair<SKey, SKey>& range) {
      int l = -1, r = -1;
      for (int i = 0; i < head_.size(); i++) {
        if (head_[i]->overlap(range.first, range.second)) {
          if (l == -1) {
            l = i;
          }
          r = i;
        }
      }
      if (l == -1) {
        return {};
      }
      auto ret = std::vector<atomic_shared_ptr<Partition>>(head_.begin() + l, head_.begin() + r + 1);
      for (auto& par : ret) {
        size_ -= par->size();
        data_size_ -= par->data_size();
        hot_size_ -= par->hot_size();
        key_n_ -= par->get_key_n();
        real_phy_size_ -= par->get_real_phy_size();
      }
      head_.erase(head_.begin() + l, head_.begin() + r + 1);
      return ret;
    }
    // Get Partitions from L to R. [L, R)
    std::vector<atomic_shared_ptr<Partition>> get_pars_range(int l, int r) {
      if (l >= r || l < 0) {
        return {};
      }
      auto ret = std::vector<atomic_shared_ptr<Partition>>(head_.begin() + l, head_.begin() + r);
      for (auto& par : ret) {
        size_ -= par->size();
        data_size_ -= par->data_size();
        hot_size_ -= par->hot_size();
        key_n_ -= par->get_key_n();
        real_phy_size_ -= par->get_real_phy_size();
      }
      head_.erase(head_.begin() + l, head_.begin() + r);
      return ret;
    }
    // Insert ordered partitions, ensure no overlaps.
    void insert_pars(const std::vector<atomic_shared_ptr<Partition>>& pars, KeyCompT comp) {
      if (pars.empty()) {
        return;
      }
      for (auto& par : pars) {
        size_ += par->size();
        data_size_ += par->data_size();
        hot_size_ += par->hot_size();
        key_n_ += par->get_key_n();
        real_phy_size_ += par->get_real_phy_size();
      }
      int pos = head_.size();
      for (int i = 0; i < head_.size(); i++) {
        if (comp(head_[i]->range().first, pars[0]->range().second) > 0) {
          pos = i;
          break;
        }
      }
      head_.insert(head_.begin() + pos, pars.begin(), pars.end());
    }
    // std::optional<LevelAsyncSeekHandle> get_async_seek_handle(SKey key, KeyCompT comp, async_io::AsyncIOQueue& aio) const {
    //   int l = 0, r = head_.size() - 1, where = -1;
    //   while (l <= r) {
    //     int mid = (l + r) >> 1;
    //     if (comp(key, head_[mid]->range().second) <= 0)
    //       where = mid, r = mid - 1;
    //     else
    //       l = mid + 1;
    //   }
    //   if (where == -1) return {};
    //   return LevelAsyncSeekHandle(key, *head_[where], *this, where, aio);
    // }
    /* return the begin. */
    LevelIterator seek_to_first() const { return LevelIterator(head_, 0); }
    bool overlap(SKey lkey, SKey rkey, KeyCompT comp) const {
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
    void append_par(atomic_shared_ptr<Partition> par) {
      logger("[append_par]: ", par->size());
      size_ += par->size();
      data_size_ += par->data_size();
      hot_size_ += par->hot_size();
      key_n_ += par->get_key_n();
      real_phy_size_ += par->get_real_phy_size();
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
        for (int i = where_l + 1; i < where_r; i++) ans += head_[i]->hot_size();
        ans += head_[where_l]->range_data_size(range);
        ans += head_[where_r]->range_data_size(range);
        return ans;
      }
    }

    // Dedicated
    void delete_range(const std::pair<SKey, SKey>& range, KeyCompT comp, std::pair<bool, bool> exclude_info) {
      auto [where_l, where_r] = _get_range_in_head(range, comp);
      if (where_l == -1 || where_r == -1 || where_l > where_r) return;
      if (where_l == where_r) {
        hot_size_ -= head_[where_l]->hot_size();
        head_[where_l]->delete_range(range, exclude_info);
        hot_size_ += head_[where_l]->hot_size();
      } else {
        hot_size_ -= head_[where_l]->hot_size();
        hot_size_ -= head_[where_r]->hot_size();
        head_[where_l]->delete_range(range, exclude_info);
        head_[where_r]->delete_range(range, exclude_info);
        hot_size_ += head_[where_l]->hot_size();
        hot_size_ += head_[where_r]->hot_size();
        for (int i = where_l + 1; i < where_r; i++) hot_size_ -= head_[i]->hot_size();
        head_.erase(head_.begin() + where_l + 1, head_.begin() + where_r);
      }
    }

    
    const std::vector<atomic_shared_ptr<Partition>>& get_pars() const {
      return head_;
    }

    size_t pars_cnt() const {
      return head_.size();
    }

    size_t get_key_n() const {
      return key_n_;
    }

    // void update_hot_size_by_tick_threshold(TickFilter<ValueT> tick_filter) {
    //   double new_hot_size = 0;
    //   for (auto& a : head_) {
    //     a->update_hot_size_by_tick_threshold(tick_filter);
    //     new_hot_size += a->hot_size();
    //   }
    //   logger(hot_size_, ", ", new_hot_size);
    //   hot_size_ = new_hot_size;
    // }

   private:
    std::pair<int, int> _get_range_in_head(const std::pair<SKey, SKey>& range, KeyCompT comp) {
      if (head_.size() == 1 && head_[0]->range().first.data() == nullptr) {
        assert(head_[0]->range().second.data() == nullptr);
        return {-1, -1};
      }
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
    std::shared_ptr<const Options> options_;
    std::vector<atomic_shared_ptr<Level>> tree_;
    std::atomic<uint32_t> ref_;
    double hot_size_overestimate_{0}, size_{0};
    size_t key_n_{0};
    KeyCompT comp_;
    size_t decay_step_{0};
    size_t real_phy_size_{0};

    using LevelIteratorSetT = SeqIteratorSet<typename Level::LevelIterator, KeyCompT, ValueT>;

   public:
    SuperVersion(std::shared_ptr<const Options> options, KeyCompT comp)
        : options_(std::move(options)), ref_(1), comp_(comp) {
      DB_ASSERT(options_);
    }
    SuperVersion(const SuperVersion& sv)
        : options_(sv.options_),
          ref_(1),
          hot_size_overestimate_(sv.hot_size_overestimate_),
          size_(sv.size_),
          key_n_(sv.key_n_),
          comp_(sv.comp_),
          decay_step_(sv.decay_step_),
          real_phy_size_(sv.real_phy_size_) {
      DB_ASSERT(options_);
      for (auto& level : sv.tree_) {
        tree_.push_back(make_atomic_shared<Level>(*level));
      }
    }
    void ref() {
      ref_++;
      // logger("ref count becomes (", this, "): ", ref_);
    }
    void unref() {
      // logger("ref count (", this, "): ", ref_);
      if (!--ref_) {
        delete this;
      }
    }

    std::string to_string() const {
      std::string str = "tree: [";
      for (auto& a : tree_) {
        str += "(" + std::to_string(a->pars_cnt()) + ", " + std::to_string(a->size() / (double)kSSTable) + "), ";
      }
      str += "], step: " + std::to_string(decay_step_) + ", hot_size: " + std::to_string(hot_size_overestimate_) + ", size: " + std::to_string(size_) + ", key n: " + std::to_string(key_n_) + ", real phy size: " + std::to_string(real_phy_size_);
      return str;
    }

    uint32_t get_ref_cnt() const {
      return ref_.load(std::memory_order_relaxed);
    }

    void set_decay_step(size_t decay_step) {
      decay_step_ = decay_step;
    }

    size_t get_decay_step() const {
      return decay_step_;
    }

    // return lowerbound.
    LevelIteratorSetT seek(SKey key) const {
      LevelIteratorSetT ret(comp_);
      for (auto& a : tree_) ret.push(a->seek(key, comp_));
      return ret;
    }

    // optimize is stably hot.
    // We seek from the largest level to the smallest level. 
    // Because it may have stably hot tag in the largest level
    // then we don't need to seek other levels. 
    bool is_stably_hot(SKey key) const {
      int cnt = 0;
      for (int i = 0; i < tree_.size(); i++) {
        if (tree_[i]->check_stably_hot(key, comp_)) {
          return true;
        }
        if (tree_[i]->check_hot(key, comp_)) {
          return true;
          // auto iter = tree_[i]->seek(key, comp_);
          // if (!iter.valid()) continue;
          // BlockKey<SKey, ValueT> kv;
          // iter.read(kv);
          // if (comp_(kv.key(), key) == 0) {
          //   if (kv.value().is_stable() || cnt) {
          //     return true;
          //   }
          //   cnt = 1;
          // }
          if (cnt == 1) {
            return true;
          } else {
            cnt += 1;
          }
        } else {
          // logger("fuck ", i, "/", tree_.size());
        }
      }
      return false;
      // auto iter = seek(key);
      // logger("fuck");
      // key.print();
      // bool cnt = 0;
      // for(auto& a : iter.get_iterators()) {
      //   BlockKey<SKey, ValueT> kv;
      //   a.read(kv);
      //   if (comp_(kv.key(), key) == 0) {
      //     if (kv.value().is_stable() || cnt) {
      //       return true;
      //     }
      //     cnt = 1;
      //   }
      // }
      // return false;
    }
    
    // return lowerbound.
    // seek concurrently, using std::async.
    // Too slow... (why?)
    LevelIteratorSetT concurrent_seek(SKey key) const {
      LevelIteratorSetT ret(comp_);
      std::vector<std::future<typename Level::LevelIterator>> futures;
      auto add_iter = [&](auto a) {
        return std::async([a, this, &key]() -> typename Level::LevelIterator {
          return a->seek(key, comp_);
        });
      };
      for (auto& a : tree_) futures.push_back(add_iter(a.get()));
      for (auto& a : futures) ret.push(a.get());
      return ret;
    }

    /* seek use async seek handles. Not so slow but slower. (why? Maybe because we are not using DIRECT_IO.)*/
    // LevelIteratorSetT batch_seek(SKey key) const {
    //   LevelIteratorSetT ret(comp_);
    //   std::vector<typename Level::LevelAsyncSeekHandle> handles;
    //   async_io::AsyncIOQueue aio(tree_.size() + 10);
    //   auto process_handle = [&](auto&& handle) {
    //     if (handle) {
    //       handle.value().init();
    //       handles.push_back(std::move(handle.value()));
    //       auto result = handles.back().next(handles.size() - 1);
    //       if (result) {
    //         ret.push(std::move(result.value()));
    //         return;
    //       }
    //     }
    //   };
    //   for (auto& a : tree_) {
    //     process_handle(a->get_async_seek_handle(key, comp_, aio));
    //   }
    //   size_t now_in_q = aio.size();
    //   size_t next = 0;
    //   if(aio.size()) {
    //     aio.submit();
    //   }
    //   while(aio.size()) {
    //     auto cur = aio.get_one().value();
    //     auto result = handles[cur].next(cur);
    //     if (result) {
    //       now_in_q--;
    //       ret.push(std::move(result.value()));
    //     } else {
    //       next++;
    //     }
    //     if (next == now_in_q) {
    //       aio.submit();
    //       next = 0;
    //     }
    //   }
    //   return ret;
    // }

    LevelIteratorSetT seek_to_first() const {
      LevelIteratorSetT ret(comp_);
      for (auto& a : tree_) ret.push(a->seek_to_first());
      return ret;
    } 
    
    // Now I use iterators to check key existence... can be optimized.
    bool search_key(SKey key) const {
      auto iter = seek(key);
      // logger("fuck");
      // key.print();
      for(auto& a : iter.get_iterators()) {
        BlockKey<SKey, ValueT> kv;
        a.read(kv);
        // kv.key().print();
        if (comp_(kv.key(), key) == 0) return true;
      }
      return false;
    }

    std::vector<atomic_shared_ptr<Level>> flush_bufs(const std::vector<UnsortedBuffer<KeyCompT, ValueT>*>& bufs, EstimateLSM<KeyCompT, ValueT, IndexDataT>& lsm) {
      if (bufs.size() == 0) return {};
      std::vector<atomic_shared_ptr<Level>> ret_vectors;
      auto flush_func = [this, &lsm](UnsortedBuffer<KeyCompT, ValueT>* buf,
                                     std::mutex& mu,
                                     std::vector<atomic_shared_ptr<Level>>&
                                         ret_vectors) {
        Compaction<KeyCompT, ValueT, IndexDataT> worker(
            options_, lsm.get_current_tick(), lsm.get_filename(), lsm.get_env(),
            lsm.get_comp());
        buf->sort(lsm.get_current_tick());
        auto iter = std::make_unique<typename UnsortedBuffer<KeyCompT, ValueT>::Iterator>(*buf);
        auto [files, hot_size] = worker.flush(*iter);
        auto level = make_atomic_shared<Level>();
        for (auto& a : files)
          level->append_par(make_atomic_shared<Partition>(
              options_, lsm.get_env(), lsm.get_file_index_cache(),
              lsm.get_file_key_cache(), lsm.get_comp(), std::move(a)));
        delete buf;
        lsm.update_write_bytes(worker.get_write_bytes());
        {
          std::unique_lock lck(mu);
          ret_vectors.push_back(std::move(level));
        }
      };
      std::vector<std::thread> thread_pool;
      std::mutex thread_mutex;
      // We now expect the number of SSTs is small, e.g. 4
      // for (uint32_t i = 1; i < bufs.size(); i++) {
        // thread_pool.emplace_back(flush_func, bufs[i], std::ref(thread_mutex), std::ref(ret_vectors));
      // }
      // if (bufs.size() >= 1) {
        // flush_func(bufs[0], thread_mutex, std::ref(ret_vectors));
      // }
      for (int i = 0; i < bufs.size(); ++i) flush_func(bufs[i], thread_mutex, std::ref(ret_vectors));
      // for (auto& a : thread_pool) a.join();
      return ret_vectors;
    }
    SuperVersion* push_new_buffers(const std::vector<atomic_shared_ptr<Level>>& vec) {
      auto ret = new SuperVersion(*this);
      ret->tree_.insert(ret->tree_.end(), vec.begin(), vec.end());
      ret->recalc_stats();
      ret->_sort_levels();
      return ret;
    }
    enum class JobType {
      kDecay = 0,
      kMajorCompaction = 1,
      kTieredCompaction = 2,
      kLeveledCompaction = 3,
      kStepDecay = 4,
    };

    SuperVersion* compact(EstimateLSM<KeyCompT, ValueT, IndexDataT>& lsm, JobType job_type) const {
      return compact(lsm, job_type, [](auto&&...){});
    }

    template<typename FuncT>
    SuperVersion* compact(EstimateLSM<KeyCompT, ValueT, IndexDataT>& lsm, JobType job_type, FuncT&& do_something, int tiered_where = -1) const {
      auto comp = lsm.get_comp();
      auto current_tick = lsm.get_current_tick();
      auto filename = lsm.get_filename();
      auto env = lsm.get_env();
      auto file_index_cache = lsm.get_file_index_cache();
      auto file_key_cache = lsm.get_file_key_cache();
      auto tick_filter = lsm.get_tick_filter();
      auto decay_tick_filter = lsm.get_decay_tick_filter();
      // check if overlap with any partition
      auto check_func = [comp, this](const Partition& par) {
        for (auto& a : tree_)
          if (a->overlap(par.range().first, par.range().second, comp)) return true;
        return false;
      };
      // add partitions that is overlapped with other levels.
      // the remaining partitions are stored in a std::vector.
      std::vector<atomic_shared_ptr<Partition>> rest;
      auto add_level = [&rest, &check_func](const Level& level) {
        auto for_iter_ptr = make_atomic_shared<std::vector<atomic_shared_ptr<Partition>>>();
        for (auto& par : level.get_pars()) {
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
      logger("current superversion. ", this->to_string());
      // decay
      if (job_type == JobType::kDecay) {
        logger("[decay]");
        auto iters = std::make_unique<LevelIteratorSetT>(comp);

        // logger("Major Compaction tree_.size() = ", tree_.size());
        // push all iterators to necessary partitions to SeqIteratorSet.
        for (auto& a : tree_) iters->push(a->seek_to_first());
        iters->build();

        // compaction...
        Compaction<KeyCompT, ValueT, IndexDataT> worker(options_, current_tick,
                                                        filename, env, comp);
        auto [files, hot_size] = worker.decay1(*iters, std::forward<FuncT>(do_something));
        auto ret = new SuperVersion(*this);
        // major compaction, so levels except largest_ become empty.
        
        auto level = make_atomic_shared<Level>();
        for (auto& a : files) {
          auto par =
              make_atomic_shared<Partition>(options_, env, file_index_cache,
                                            file_key_cache, comp, std::move(a));
          level->append_par(std::move(par));
        }   
        ret->tree_.push_back(std::move(level));
        // calculate new current decay size
        ret->recalc_stats();
        ret->_sort_levels();

        lsm.update_write_bytes(worker.get_write_bytes());
        return ret;
      } else if (job_type == JobType::kMajorCompaction) {
        logger("[major compaction]");
        logger("tick threshold", tick_filter.get_tick_threshold());
        auto iters = std::make_unique<LevelIteratorSetT>(comp);

        // logger("Major Compaction tree_.size() = ", tree_.size());
        // push all iterators to necessary partitions to SeqIteratorSet.
        for (auto& a : tree_) iters->push(a->seek_to_first());
        iters->build();

        // compaction...
        Compaction<KeyCompT, ValueT, IndexDataT> worker(options_, current_tick,
                                                        filename, env, comp);
        auto [files, hot_size] = worker.flush_with_filter(*iters, tick_filter, decay_tick_filter, std::forward<FuncT>(do_something));
        auto ret = new SuperVersion(options_, comp);
        // major compaction, so levels except largest_ become empty.
        
        auto level = make_atomic_shared<Level>();
        for (auto& a : files) {
          auto par =
              make_atomic_shared<Partition>(options_, env, file_index_cache,
                                            file_key_cache, comp, std::move(a));
          level->append_par(std::move(par));
        }   
        ret->tree_.push_back(std::move(level));
        // calculate new current decay size
        ret->recalc_stats();
        ret->_sort_levels();

        lsm.update_write_bytes(worker.get_write_bytes());
         
        return ret;
      } else if (job_type == JobType::kTieredCompaction) {
        // similar to universal compaction in rocksdb
        // if tree_[i]->size()/(\sum_{j=0..i-1}tree_[j]->size()) <= kMergeRatio
        // then merge them.
        // if kMergeRatio ~ 1./X, then it can be treated as (X+1)-tired compaction

        // if the number of tables >= kLimitMin, then begin to merge
        // if the number of tables >= kLimitMax, then increase kMergeRatio. (this works when the number of tables is small)

        if (tree_.size() >= kLimitMin || tiered_where != -1) {
          auto _kRatio = kMergeRatio;
          double min_ratio = 1e30;
          int where = -1, _where = -1;
          size_t sum = tree_.back()->size();
          if (tiered_where != -1) {
            where = tiered_where;
          } else {
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
          }
          logger("[tiered compaction]");

          auto iters = std::make_unique<LevelIteratorSetT>(comp);

          // logger("compact(): where = ", where);
          // push all iterators to necessary partitions to SeqIteratorSet.
          for (uint32_t i = where; i < tree_.size(); ++i) iters->push(typename Level::LevelIterator(add_level(*tree_[i]), 0));
          iters->build();
          std::sort(rest.begin(), rest.end(), [comp](const atomic_shared_ptr<Partition>& x, const atomic_shared_ptr<Partition>& y) {
            return comp(x->range().first, y->range().second) < 0;
          });

          // logger("compact...");
          // compaction...
          Compaction<KeyCompT, ValueT, IndexDataT> worker(
              options_, current_tick, filename, env, comp);
          // Don't use tick filter because it is partial merge.
          auto [files, hot_size] = worker.flush_with_filter(*iters, TickFilter<ValueT>(-114514), TickFilter<ValueT>(-114514), std::forward<FuncT>(do_something));
          auto ret = new SuperVersion(*this);
          lsm.update_write_bytes(worker.get_write_bytes());
          // minor compaction?
          // hot_size_overestimate = largest_ + remaining levels + new level

          auto rest_iter = rest.begin();
          auto level = make_atomic_shared<Level>();
          for (auto& a : files) {
            auto par = make_atomic_shared<Partition>(
                options_, env, file_index_cache, file_key_cache, comp,
                std::move(a));
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
          ret->recalc_stats();
          ret->_sort_levels();
          logger("[new superversion]", ret->to_string());
          return ret;

        }
        return nullptr;
      } else if (job_type == JobType::kLeveledCompaction) {
        
        if (tree_.size() < kLimitMin) {
          return nullptr;
        }

        double size_limit = tree_[0]->size();
        uint32_t mni = -1;
        for (int i = 1; i < tree_.size(); i++) {
          size_limit /= kLevelMultiplier;
          if (tree_[i]->size() > size_limit) {
            mni = i;
          }
        }

        if (mni == -1) {
          return nullptr;
        }

        // if (tree_[mni]->pars_cnt() <= 1) {
        //   return compact(lsm, JobType::kTieredCompaction, std::forward<FuncT>(do_something), mni);
        // }

        auto ret = new SuperVersion(*this);
        auto pars0 = ret->tree_[mni]->get_min_overlap_pars(*tree_[mni - 1], 1, comp);
        if (pars0.empty()) {
          logger("Failed!");
          ret->unref();
          return nullptr;
        }
        auto pars1 = ret->tree_[mni - 1]->get_overlapped_pars({pars0[0]->range().first, pars0.back()->range().second});
        auto iters = std::make_unique<LevelIteratorSetT>(comp);
        iters->push(typename Level::LevelIterator(pars0, 0));
        iters->push(typename Level::LevelIterator(pars1, 0));
        iters->build();

        Compaction<KeyCompT, ValueT, IndexDataT> worker(options_, current_tick,
                                                        filename, env, comp);
        auto [files, hot_size] = worker.flush_with_filter(*iters, TickFilter<ValueT>(-114514), TickFilter<ValueT>(-114514), std::forward<FuncT>(do_something));
        lsm.update_write_bytes(worker.get_write_bytes());

        std::vector<atomic_shared_ptr<Partition>> pars;
        for (auto& a : files) {
          auto par =
              make_atomic_shared<Partition>(options_, env, file_index_cache,
                                            file_key_cache, comp, std::move(a));
          pars.push_back(std::move(par));
        }
        
        logger("[leveled compaction]: ", mni, "(", pars0.size(), "), ", mni - 1, "(", pars1.size(), ") => ", mni - 1, "(", pars.size(), ")");
        ret->tree_[mni - 1]->insert_pars(pars, comp);

        ret->recalc_stats();
        ret->_sort_levels();
        logger("[new superversion]", ret->to_string());
        return ret;
      } else if (job_type == JobType::kStepDecay) {
        if (tree_.empty()) {
          return nullptr;
        }

        logger("physical size: ", lsm.get_physical_size_limit(), ", ", size_, ". hot size: ", hot_size_overestimate_, ", ", lsm.get_hot_size_limit());

        if (size_ < lsm.get_physical_size_limit() && hot_size_overestimate_ < lsm.get_hot_size_limit()) {
          return nullptr;
        }

        logger("[step decay] decay tick threshold: ", decay_tick_filter.get_tick_threshold(), ", hot tick threshold: ", tick_filter.get_tick_threshold());

        auto ret = new SuperVersion(*this);
        if (decay_step_ > ret->tree_[0]->pars_cnt()) {
          ret->set_decay_step(0);
          return ret;
        }

        size_t step = ret->get_decay_step();

        if (step >= ret->tree_[0]->pars_cnt()) {
          ret->set_decay_step(0);
          return ret;
        }
        
        // Find kStepDecayLen SSTs to compact.
        std::vector<int> compact_r(ret->tree_.size());
        std::vector<int> compact_l(ret->tree_.size());
        compact_l[0] = compact_r[0] = step;
        int sst_cnt = 0;
        for (int j = 1; j < ret->tree_.size(); j++) {
          auto range = ret->tree_[0]->get_pars()[compact_l[0]]->range();
          while (compact_l[j] < ret->tree_[j]->pars_cnt() && comp(ret->tree_[j]->get_pars()[compact_l[j]]->range().second, range.first) < 0) {
            compact_l[j] += 1;
          }
          compact_r[j] = compact_l[j];
        }
        while (compact_r[0] < ret->tree_[0]->pars_cnt() && sst_cnt <= kStepDecayLen) {
          auto range = ret->tree_[0]->get_pars()[compact_r[0]]->range();
          compact_r[0] += 1;
          sst_cnt += 1;
          for (int j = 1; j < ret->tree_.size(); j++) {
            while (compact_r[j] < ret->tree_[j]->pars_cnt() && comp(ret->tree_[j]->get_pars()[compact_r[j]]->range().first, range.second) <= 0
                  && sst_cnt <= kStepDecayLen) {
              compact_r[j] += 1;
              sst_cnt += 1;
            }
          }
        }

        // Push all iterators.
        std::vector<std::vector<atomic_shared_ptr<Partition>>> compact_data;
        for (int i = 0; i < ret->tree_.size(); i++) {
          auto v = ret->tree_[i]->get_pars_range(compact_l[i], compact_r[i]);
          if (v.empty()) continue;
          compact_data.push_back(std::move(v));
        }

        auto iters = std::make_unique<LevelIteratorSetT>(comp);
        for (auto& a : compact_data) {
          iters->push(typename Level::LevelIterator(a, 0));
        }
        iters->build();
        logger("[", step, ", ", compact_r[0], "]");

        // compaction...
        Compaction<KeyCompT, ValueT, IndexDataT> worker(options_, current_tick,
                                                        filename, env, comp);
        auto [files, hot_size] = worker.flush_with_filter(*iters, tick_filter, decay_tick_filter, std::forward<FuncT>(do_something));
        
        std::vector<atomic_shared_ptr<Partition>> result;
        std::vector<atomic_shared_ptr<Partition>> last_ssts;
        last_ssts.clear();
        for (auto& a : files) {
          auto par =
              make_atomic_shared<Partition>(options_, env, file_index_cache,
                                            file_key_cache, comp, std::move(a));
          if ((compact_r[0] < tree_[0]->pars_cnt() && comp(par->range().second, tree_[0]->get_pars()[compact_r[0]]->range().first) >= 0) || 
          (compact_l[0] >= 1 && comp(par->range().first, tree_[0]->get_pars()[compact_l[0] - 1]->range().second) <= 0)) {
            last_ssts.push_back(std::move(par));
          } else {
            result.push_back(std::move(par));
          }
        }   
        ret->tree_[0]->insert_pars(result, comp);
        ret->set_decay_step(step + result.size() >= ret->tree_[0]->pars_cnt() ? 0 : step + result.size());
        if (last_ssts.size()) {
          auto level = make_atomic_shared<Level>();
          level->insert_pars(last_ssts, comp);
          ret->tree_.push_back(std::move(level));
        }
        // calculate new current decay size
        ret->recalc_stats();
        ret->_sort_levels();
        lsm.update_write_bytes(worker.get_write_bytes());
        logger("[new superversion]", ret->to_string());
        return ret;
      } else {
        return nullptr;
      }
    }

    template<typename T>
    void scan_all_without_merge(T&& func) const {
      for (auto& level : tree_) {
        auto iter = level->seek_to_first();
        for (; iter.valid(); iter.next()) {
          BlockKey<SKey, ValueT> L;
          iter.read(L);
          func(L.value().get_score(options_),
               L.key().size() + sizeof(ValueT) + 4,
               L.value().get_hot_size() + L.key().len(), L.value());
        }
      }
    }

    
    template<typename T>
    void scan_all_with_merge(T&& func) const {
      LevelIteratorSetT _iter(comp_);
      for (auto& level : tree_) _iter.push(level->seek_to_first());
      SeqIteratorSetForScan iter(options_, std::move(_iter), 0,
                                 TickFilter<ValueT>(-114514));
      iter.build();
      for (; iter.valid(); iter.next()) {
        auto L = iter.read();
        // +4 for the offset bytes. in SST.
        func(L.second.get_score(*options_), L.first.size() + sizeof(ValueT) + 4,
             L.second.get_hot_size() + L.first.len(), L.second);
      }
    }

    size_t range_count(const std::pair<SKey, SKey>& range, KeyCompT comp) {
      size_t ans = 0;
      for (auto& level : tree_) ans += level->range_count(range, comp);
      return ans;
    }

    size_t range_data_size(const std::pair<SKey, SKey>& range, KeyCompT comp) {
      size_t ans = 0;
      for (auto& level : tree_) ans += level->range_data_size(range, comp);
      return ans;
    }

    SuperVersion* delete_range(std::pair<SKey, SKey> range, KeyCompT comp, std::pair<bool, bool> exclude_info) {
      auto ret = new SuperVersion(*this);
      for (auto& level : ret->tree_) level->delete_range(range, comp, exclude_info);
      ret->recalc_stats();
      return ret;
    }

    double get_current_hot_size() const { return hot_size_overestimate_; }

    size_t get_current_real_phy_size() const {
      return real_phy_size_;
    }

    double get_size() const { return size_; }

    size_t get_key_n() const { return key_n_; }

    void recalc_stats() {
      hot_size_overestimate_ = _calc_current_hot_size();
      size_ = _calc_current_size();
      key_n_ = _calc_key_n();
      real_phy_size_ = _calc_real_phy_size();
    }

   private:
    double _calc_current_hot_size() {
      double ret = 0;
      for (auto& a : tree_) ret += a->hot_size();
      return ret;
    }
    double _calc_current_size() {
      double ret = 0;
      for (auto& a : tree_) ret += a->size();
      return ret;
    }
    size_t _calc_key_n() {
      size_t ret = 0;
      for (auto& a : tree_) ret += a->get_key_n();
      return ret;
    }
    size_t _calc_real_phy_size() {
      size_t ret = 0;
      for (auto& a : tree_) ret += a->get_real_phy_size();
      return ret;
    }
    // Sort levels by size and remove empty levels.
    // Levels are from the largest to the smallest.
    void _sort_levels() {
      std::sort(tree_.begin(), tree_.end(), [](const atomic_shared_ptr<Level>& x, const atomic_shared_ptr<Level>& y) { return x->size() > y->size(); });
      for (int i = 0; i < tree_.size(); i++) if (tree_[i]->size() == 0) {
        tree_.erase(tree_.begin() + i, tree_.end());
        break;
      }
    }
  };
  // Arguments
  std::shared_ptr<const Options> options_;
  Env* env_;
  std::unique_ptr<FileName> filename_;
  KeyCompT comp_;
  size_t physical_size_limit_{0};
  uint64_t max_hot_size_limit_{0};
  uint64_t min_hot_size_limit_{0};
  uint64_t accessed_size_to_decr_counter_{0};

  SuperVersion* sv_;
  std::thread compact_thread_;
  Timer compact_thread_timer_;
  std::thread flush_thread_;
  Timer flush_thread_timer_;
  std::atomic<bool> terminate_signal_;
  UnsortedBufferPtrs<KeyCompT, ValueT> bufs_;
  std::mutex sv_mutex_;
  std::mutex sv_load_mutex_;
  // Only one thread can modify sv.
  std::mutex sv_modify_mutex_;
  std::vector<atomic_shared_ptr<Level>> flush_buf_vec_;
  std::mutex flush_buf_vec_mutex_;
  std::condition_variable signal_flush_to_compact_;
  double hot_size_overestimate_{0};
  size_t sv_tick_{0};
  // 0: idle, 1: working
  uint8_t flush_thread_state_{0};
  uint8_t compact_thread_state_{0};
  uint64_t hot_size_limit_{0};
  size_t phy_size_{0};
  size_t real_hot_size_{0};
  size_t real_phy_size_{0};
  size_t key_n_{0};
  size_t period_{0};
  size_t lst_decay_period_{0};
  size_t exp_tick_period_{0};
  size_t delta_c_{kCMax};
  size_t last_stable_hot_size_{0};

  // Used for tick
  std::atomic<size_t>& current_tick_;
  std::atomic<double> tick_threshold_{-114514};
  std::atomic<double> decay_tick_threshold_{-1919810};
  std::atomic<size_t> current_access_bytes_;

  // default cache
  std::unique_ptr<FileChunkCache> file_cache_;

  // Statistics
  std::mutex stat_mutex_;
  std::atomic<size_t> stat_write_bytes_{0};
  size_t stat_read_bytes_{0};
  size_t stat_compact_time_{0};
  size_t stat_decay_scan_time_{0};
  size_t stat_decay_write_time_{0};
  size_t stat_flush_time_{0};

  // Clock hand for simulating clock algorithm
  IndSKey clock_hand_;
  bool clock_hand_valid_{false};


  using LevelIteratorSetTForScan = SeqIteratorSetForScan<typename Level::LevelIterator, KeyCompT, ValueT>;
 public:
  class SuperVersionIterator {
    LevelIteratorSetTForScan iter_;
    SuperVersion* sv_;

   public:
    SuperVersionIterator(LevelIteratorSetTForScan&& iter, SuperVersion* sv) : iter_(std::move(iter)), sv_(sv) { 
      iter_.build(); 
    }
    ~SuperVersionIterator() { sv_->unref(); }
    auto read() { return iter_.read(); }
    auto valid() { return iter_.valid(); }
    auto next() { return iter_.next(); }
  };
  EstimateLSM(const Options& options, Env* env, size_t file_cache_size,
              std::unique_ptr<FileName>&& filename, KeyCompT comp,
              std::atomic<size_t>& current_tick, size_t initial_hot_size,
              size_t max_hot_size, size_t min_hot_size,
              size_t physical_size_limit,
              uint64_t accessed_size_to_decr_counter)
      : options_(std::make_shared<const Options>(options)),
        env_(env),
        filename_(std::move(filename)),
        comp_(comp),
        max_hot_size_limit_(max_hot_size),
        min_hot_size_limit_(min_hot_size),
        accessed_size_to_decr_counter_(accessed_size_to_decr_counter),
        sv_(new SuperVersion(options_, comp)),
        terminate_signal_(0),
        bufs_(options_, kUnsortedBufferSize, kUnsortedBufferMaxQueue, comp),
        file_cache_(std::make_unique<FileChunkCache>(file_cache_size)),
        current_tick_(current_tick),
        physical_size_limit_(physical_size_limit),
        hot_size_limit_(initial_hot_size) {
    compact_thread_ = std::thread([this]() { compact_thread(); });
    clockid_t compact_thread_clock_id;
    pthread_getcpuclockid(compact_thread_.native_handle(),
                          &compact_thread_clock_id);
    compact_thread_timer_ = Timer(compact_thread_clock_id);

    flush_thread_ = std::thread([this]() { flush_thread(); });
    clockid_t flush_thread_clock_id;
    pthread_getcpuclockid(flush_thread_.native_handle(),
                          &flush_thread_clock_id);
    flush_thread_timer_ = Timer(flush_thread_clock_id);
    logger_printf("LSM bloom filter BPK: %zu bits.", options_->bloom_bits);
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
  
  void append(SKey key, ValueT _value) {
    auto read_size = key.len() + _value.get_hot_size();
    auto access_bytes = current_access_bytes_.fetch_add(read_size, std::memory_order_relaxed);
    if (access_bytes % size_t(accessed_size_to_decr_counter_) + read_size >
        size_t(accessed_size_to_decr_counter_)) {
      period_ += 1;
    }
    uint64_t accessed_size_to_decr_tick =
        options_->tick_period_multiplier * hot_size_limit_;
    if (access_bytes % accessed_size_to_decr_tick + read_size >
        accessed_size_to_decr_tick) {
      exp_tick_period_ += 1;
    }
    ValueT value(exp_tick_period_, _value.get_hot_size(), delta_c_);
    bufs_.append_and_notify(key, value);
  }
  void append(SKey key, size_t vlen) {
    auto read_size = key.len() + vlen;
    auto access_bytes = current_access_bytes_.fetch_add(read_size, std::memory_order_relaxed);
    if (access_bytes % size_t(accessed_size_to_decr_counter_) + read_size >
        size_t(accessed_size_to_decr_counter_)) {
      period_ += 1;
    }
    uint64_t accessed_size_to_decr_tick =
        options_->tick_period_multiplier * hot_size_limit_;
    if (access_bytes % accessed_size_to_decr_tick + read_size >
        accessed_size_to_decr_tick) {
      exp_tick_period_ += 1;
    }
    ValueT value(exp_tick_period_, vlen, delta_c_);
    bufs_.append_and_notify(key, value);
  }
  auto seek(SKey key) {
    auto sv = get_current_sv();
    return std::make_unique<SuperVersionIterator>(
        LevelIteratorSetTForScan(options_, sv->seek(key), get_current_tick(),
                                 get_tick_filter()),
        sv);
  }

  auto seek_to_first() {
    auto sv = get_current_sv();
    return std::make_unique<SuperVersionIterator>(
        LevelIteratorSetTForScan(options_, sv->seek_to_first(),
                                 get_current_tick(), get_tick_filter()),
        sv);
  }

  auto batch_seek(SKey key) {
    auto sv = get_current_sv();
    return std::make_unique<SuperVersionIterator>(
        LevelIteratorSetTForScan(options_, sv->batch_seek(key),
                                 get_current_tick(), get_tick_filter()),
        sv);
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

  void set_physical_size(size_t physical_size) {
    physical_size_limit_ = physical_size;
  }

  size_t get_physical_size_limit() const {
    return physical_size_limit_;
  }

  size_t get_hot_size_limit() const {
    return hot_size_limit_;
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

  void delete_range(std::pair<SKey, SKey> range, std::pair<bool, bool> exclude_info) {
    if (comp_(range.first, range.second) > 0) return;
    std::unique_lock del_range_lck(sv_modify_mutex_);
    auto new_sv = sv_->delete_range(range, comp_, exclude_info);
    auto old_sv = _update_superversion(new_sv);
    if (old_sv) {
      old_sv->unref();
    }
  }

  void trigger_decay() {
    std::unique_lock del_range_lck(sv_modify_mutex_);
    auto new_sv = sv_->compact(*this, SuperVersion::JobType::kDecay);
    auto old_sv = _update_superversion(new_sv);
    if (old_sv) {
      old_sv->unref();
    }
  }

  double get_current_hot_size() {
    return hot_size_overestimate_;
  }

  size_t get_key_n() const {
    return key_n_;
  }

  
  double get_current_phy_size() const {
    return phy_size_;
  }

  void set_tick_threshold(size_t x) {
    tick_threshold_.store(x, std::memory_order_relaxed);
  }
  
  void set_decay_tick_threshold(size_t x) {
    decay_tick_threshold_.store(x, std::memory_order_relaxed);
  }

  bool search_key(SKey key) {
    auto sv = get_current_sv();
    auto ret = sv->search_key(key);
    sv->unref();
    return ret;
  }

  bool is_stably_hot(SKey key) {
    auto sv = get_current_sv();
    auto ret = sv->is_stably_hot(key);
    sv->unref();
    return ret;
  }

  double get_current_tick() const {
    return current_tick_.load(std::memory_order_relaxed);
  }

  TickFilter<ValueT> get_tick_filter() const {
    return TickFilter<ValueT>(tick_threshold_.load(std::memory_order_relaxed));
  }

  TickFilter<ValueT> get_decay_tick_filter() const {
    return TickFilter<ValueT>(decay_tick_threshold_.load(std::memory_order_relaxed));
  }

  /* used for fast decay. */
  void faster_decay() {
    bool is_set_unstable = lst_decay_period_ != period_;
    lst_decay_period_ = period_;
    sv_modify_mutex_.lock();
    logger("[tick threshold for hot]: ", tick_threshold_.load(), ", [for phy]: ", decay_tick_threshold_.load());
    while(true) {
      Timer sw;
      auto new_sv = sv_->compact(*this, SuperVersion::JobType::kStepDecay, [is_set_unstable](auto& key, auto& value) {
        if (is_set_unstable) {
          value.decrease_stable();
        }
      });
      stat_decay_write_time_ += sw.GetTimeInNanos();

      if (new_sv == nullptr) {
        break;
      }
      auto old_sv = _update_superversion(new_sv);
      while (old_sv && old_sv->get_ref_cnt() > 1) {
        logger("wait");
        sv_modify_mutex_.unlock();
        std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(kWaitCompactionSleepMilliSeconds));
        sv_modify_mutex_.lock();
      }
      if (old_sv) {
        old_sv->unref();
      }
      if (sv_->get_decay_step() == 0) {
        break;
      }
    }
    real_hot_size_ = sv_->get_current_hot_size();
    real_phy_size_ = sv_->get_current_real_phy_size();
    sv_modify_mutex_.unlock();
  }

  double get_tick_threshold() const {
    return tick_threshold_.load(std::memory_order_relaxed);
  }

  /* Used for stats */
  const FileChunkCache* get_cache() const {
    return file_cache_.get();
  }

  size_t get_write_bytes() const {
    return stat_write_bytes_;
  }

  size_t get_compact_time() const {
    return stat_compact_time_;
  }

  size_t get_flush_time() const {
    return stat_flush_time_;
  }

  size_t get_decay_scan_time() const {
    return stat_decay_scan_time_;
  }

  size_t get_decay_write_time() const {
    return stat_decay_write_time_;
  }

  size_t get_compact_thread_time() const {
    return compact_thread_timer_.GetTimeInNanos();
  }
  size_t get_flush_thread_time() const {
    return flush_thread_timer_.GetTimeInNanos();
  }

  // Only used for simulating clock cache.
  void clock_style_decay() {
    ssize_t total_hot_size = 0;
    ssize_t total_phy_size = 0;
    auto sv = get_current_sv();
    sv->scan_all_with_merge([&](double tick, size_t phy_size, size_t hot_size,
                                const ValueT& value) {
      if (value.get_score(*options_) > 0) {
        total_hot_size += hot_size;
        total_phy_size += phy_size;
      }
    });
    sv->unref();
    sv_modify_mutex_.lock();
    logger("[total phy]: ", total_phy_size, ", [total hot] ", total_hot_size, ", [hot limit] ", hot_size_limit_, ", [phy limit]", physical_size_limit_);
    while(true) {
      Timer sw;
      auto new_sv =
          sv_->compact(*this, SuperVersion::JobType::kStepDecay,
                       [&](auto& key, auto& value) {
                         if (value.get_score(*options_) > 0 &&
                             (total_phy_size > physical_size_limit_ ||
                              total_hot_size > hot_size_limit_) &&
                             (!clock_hand_valid_ ||
                              comp_(key.ref(), clock_hand_.ref()) > 0)) {
                           value.decrease_stable();
                           if (value.get_score(*options_) == 0) {
                             total_hot_size -= value.get_hot_size() + key.len();
                             total_phy_size -= key.size() + sizeof(ValueT) + 4;
                             if (total_phy_size <= physical_size_limit_ &&
                                 total_hot_size <= hot_size_limit_) {
                               clock_hand_ = key;
                               clock_hand_valid_ = true;
                             }
                           }
                         }
                       });
      stat_decay_write_time_ += sw.GetTimeInNanos();

      if (new_sv == nullptr) {
        break;
      }
      auto old_sv = _update_superversion(new_sv);
      while (old_sv && old_sv->get_ref_cnt() > 1) {
        logger("wait");
        sv_modify_mutex_.unlock();
        std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(kWaitCompactionSleepMilliSeconds));
        sv_modify_mutex_.lock();
      }
      if (old_sv) {
        old_sv->unref();
      }
      if (sv_->get_decay_step() == 0) {
        if (total_phy_size <= physical_size_limit_ && total_hot_size <= hot_size_limit_) {
          break;
        } else {
          clock_hand_valid_ = false;
        }
      }
    }
    real_hot_size_ = sv_->get_current_hot_size();
    real_phy_size_ = sv_->get_current_real_phy_size();
    sv_modify_mutex_.unlock();
  }

  void update_tick_threshold() {
    KthEst<double> est_hot(kEstPointNum, hot_size_limit_);
    KthEst<double> est_phy(kEstPointNum, physical_size_limit_);
    KthEst<double> est_cnt(kEstPointNum, key_n_ * 0.9);
    auto sv = get_current_sv();
    logger(sv->to_string());
    est_hot.pre_scan1(sv->get_current_hot_size() / sv->get_current_real_phy_size() * sv->get_size() * 1.1);
    est_phy.pre_scan1(sv->get_size());
    est_cnt.pre_scan1(sv->get_key_n());
    auto hot_tick_filter = get_tick_filter();
    size_t total_hot_size = 0;
    size_t total_n = 0, stable_n = 0, stable_hot_size = 0;
    Timer sw;
    sv->scan_all_with_merge([&](double tick, size_t phy_size, size_t hot_size, const ValueT& value) {
      est_phy.scan1(-tick, phy_size);
      est_hot.scan1(-tick, hot_size);
      est_cnt.scan1(-tick, 1);
      total_hot_size += hot_size;
      total_n += 1;
      if (value.is_stable()) {
        stable_n += 1;
        stable_hot_size += hot_size;
      }
    });
    last_stable_hot_size_ = stable_hot_size;
    uint64_t max_hot_size_limit_decr =
        (max_hot_size_limit_ - min_hot_size_limit_) / 50;
    uint64_t min_hot_size_limit =
        hot_size_limit_ >= max_hot_size_limit_decr
            ? hot_size_limit_ - max_hot_size_limit_decr
            : 0;
    min_hot_size_limit = std::max(min_hot_size_limit_, min_hot_size_limit);
    hot_size_limit_ = std::max(min_hot_size_limit,
                               std::min(max_hot_size_limit_, stable_hot_size));
    logger("total_hot_size: ", total_hot_size, ", total_n: ", total_n, ", stable_n: ", stable_n, ", stable_hot_size: ", stable_hot_size, ", period: ", period_, ", lst_decay_period: ", lst_decay_period_);
    sv->unref();
    est_hot.sort();
    est_phy.sort();
    tick_threshold_ = -est_hot.get_from_points(hot_size_limit_);
    decay_tick_threshold_ = std::max(-est_phy.get_from_points(physical_size_limit_), -est_hot.get_from_points(hot_size_limit_ * 2));
    stat_decay_scan_time_ += sw.GetTimeInNanos();
  }

  
  auto get_comp() const { return comp_; }

  auto get_filename() const { return filename_.get(); }

  auto get_env() const { return env_; }

  auto get_file_index_cache() const { return file_cache_.get(); }

  auto get_file_key_cache() const { return nullptr; }

  void update_write_bytes(size_t x) {
    stat_write_bytes_ += x;
  }

  bool check_decay_condition() {
    double hs_step = std::max(hot_size_limit_ * kHotSetExceedLimit, (max_hot_size_limit_ - min_hot_size_limit_) / 20.0);
    double phy_step = std::max<double>(kSSTable, physical_size_limit_ * kHotSetExceedLimit);
    return hot_size_overestimate_ > hot_size_limit_ + hs_step ||
        phy_size_ > physical_size_limit_ + phy_step || lst_decay_period_ != period_;
  }

  void set_hot_set_limit(size_t new_limit) {
    hot_size_limit_ = new_limit;
  }

  void set_phy_limit(size_t new_limit) {
    physical_size_limit_ = new_limit;
  }

  void set_all_limit(size_t new_hs_limit, size_t new_phy_limit) {
    hot_size_limit_ = new_hs_limit;
    physical_size_limit_ = new_phy_limit;
  }

  size_t get_phy_limit() const {
    return physical_size_limit_;
  }

  size_t get_hot_set_limit() const {
    return hot_size_limit_;
  }

  uint64_t get_min_hot_size_limit() { return min_hot_size_limit_; }
  void set_min_hot_size_limit(uint64_t min_hot_size_limit) {
    min_hot_size_limit_ = min_hot_size_limit;
  }

  uint64_t get_max_hot_size_limit() { return max_hot_size_limit_; }
  void set_max_hot_size_limit(uint64_t max_hot_size_limit) {
    max_hot_size_limit_ = max_hot_size_limit;
  }

  void set_proper_phy_limit() {
    if (phy_size_ == 0) {
      return;
    }
    physical_size_limit_ = std::max(10 * kSSTable, real_phy_size_);
  }

  size_t get_real_hs_size() const {
    return real_hot_size_;
  }

  size_t get_real_phy_size() const {
    return real_phy_size_;
  }


 private:
  void flush_thread() {
    while (true) {
      if (bufs_.size() == 0) {      
        flush_thread_state_ = 0;
      }
      if (terminate_signal_) {
        flush_thread_state_ = 0;
        return;
      }
      auto buf_q_ = bufs_.wait_and_get();
      if (buf_q_.empty()) continue;
      flush_thread_state_ = 1;
      Timer sw;
      auto new_vec = sv_->flush_bufs(buf_q_, *this);
      stat_flush_time_ += sw.GetTimeInNanos();
      while (new_vec.size() + flush_buf_vec_.size() > kMaxFlushBufferQueueSize) {
        logger("full");
        std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(kWaitCompactionSleepMilliSeconds));
        if (terminate_signal_) {            
          flush_thread_state_ = 0;
          return;
        }
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
      SuperVersion* new_sv;
      {
        std::unique_lock flush_lck(flush_buf_vec_mutex_);
        if (flush_buf_vec_.empty()) {      
          compact_thread_state_ = 0;
        }
        if (terminate_signal_) {      
          compact_thread_state_ = 0;
          return;
        }
        // wait buffers from flush_thread()
        signal_flush_to_compact_.wait(flush_lck, [&]() { return !flush_buf_vec_.empty() || terminate_signal_; });
        if (terminate_signal_) return;
        compact_thread_state_ = 1;
        sv_modify_mutex_.lock();
        new_sv = sv_->push_new_buffers(flush_buf_vec_);
        flush_buf_vec_.clear();
      }

      // Compact until there is no potential compaction.
      auto last_compacted_sv = new_sv;
      while (true) {
        SuperVersion* new_compacted_sv = nullptr;
        Timer sw;
        // new_compacted_sv = last_compacted_sv->compact(*this, SuperVersion::JobType::kTieredCompaction);
        new_compacted_sv = last_compacted_sv->compact(*this, SuperVersion::JobType::kLeveledCompaction);
        stat_compact_time_ += sw.GetTimeInNanos();
        if (new_compacted_sv == nullptr) {
          break;
        } else {
          last_compacted_sv->unref();
          last_compacted_sv = new_compacted_sv;
        }
      }
      auto old_sv = _update_superversion(last_compacted_sv);
      sv_modify_mutex_.unlock();
      // Wait to release old superversion
      while (old_sv && old_sv->get_ref_cnt() > 1 && !terminate_signal_) {
        logger("wait");
        std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(kWaitCompactionSleepMilliSeconds));
      }
      if (old_sv) {
        old_sv->unref();
      }
    }
  }

  SuperVersion* _update_superversion(SuperVersion* new_sv) {
    if (new_sv != nullptr) {
      sv_load_mutex_.lock();
      auto old_sv = sv_;
      sv_ = new_sv;
      hot_size_overestimate_ = new_sv->get_current_hot_size();
      key_n_ = new_sv->get_key_n();
      phy_size_ = new_sv->get_size();
      sv_tick_++;
      sv_load_mutex_.unlock();
      return old_sv;
    }
    return nullptr;
  }

  SuperVersion* get_current_sv() {
    sv_load_mutex_.lock();
    auto sv = sv_;
    sv->ref();
    sv_load_mutex_.unlock();
    return sv;
  }
};

enum class CachePolicyT {
  kUseDecay = 0,
  kUseTick,
  kUseFasterTick,
  kClockStyleDecay
};

// Viscnts, implement lsm tree and other things.

// We don't use tree[1] now.
template <typename KeyCompT, typename ValueT, typename IndexDataT, CachePolicyT cache_policy>
class alignas(128) VisCnts {
  std::unique_ptr<EstimateLSM<KeyCompT, ValueT, IndexDataT>> tree;
  std::atomic<size_t> current_tick_{0};
  std::atomic<size_t> stat_input_bytes_{0};
  KeyCompT comp_;
  std::mutex decay_m_;
  std::atomic<bool> is_updating_tick_threshold_{false};
  bool terminate_signal_{false};
  std::condition_variable decay_cv_;
  std::thread decay_thread_;
  Timer decay_thread_timer_;

  bool is_first_tick_update_{true};
  size_t decay_count_{0};

 public:
  using IteratorT = typename EstimateLSM<KeyCompT, ValueT, IndexDataT>::SuperVersionIterator;
  // Use different file path for two trees.
  VisCnts(const Options& options, KeyCompT comp, const std::string& path,
          size_t initial_hot_size, size_t max_hot_size, size_t min_hot_size,
          size_t physical_size, uint64_t accessed_size_to_decr_counter)
      : tree{std::make_unique<EstimateLSM<KeyCompT, ValueT, IndexDataT>>(
            options, createDefaultEnv(), kIndexCacheSize,
            std::make_unique<FileName>(0, path + "/a0"), comp, current_tick_,
            initial_hot_size, max_hot_size, min_hot_size, physical_size,
            accessed_size_to_decr_counter)},
        comp_(comp) {
    decay_thread_ = std::thread([&]() { decay_thread(); });
    clockid_t decay_cpu_clock_id;
    pthread_getcpuclockid(decay_thread_.native_handle(), &decay_cpu_clock_id);
    decay_thread_timer_ = Timer(decay_cpu_clock_id);
  }
  ~VisCnts() {
    terminate_signal_ = true;
    decay_cv_.notify_all(); 
    decay_thread_.join();
    
    auto stat0 = tree->get_cache()->get_stats();
    logger_printf("all. read bytes: %zu B, %lf GB", GetReadBytes(), GetReadBytes() / 1073741824.0);
    logger_printf("all. write bytes: %zu B, %lf GB", GetWriteBytes(), GetWriteBytes() / 1073741824.0);
    logger_printf("all. input bytes: %zu B, %lf GB", stat_input_bytes_.load(), stat_input_bytes_.load() / 1073741824.0);
    logger_printf("tier 0. hit: %zu, access: %zu", stat0.hit_count, stat0.access_count);
    logger_printf("tier 0. write bytes: %zu B, %lf GB", tree->get_write_bytes(), tree->get_write_bytes() / 1073741824.0);
    logger_printf("tier 0. compact time: %zu ns, %.6lf s", tree->get_compact_time(), tree->get_compact_time() / 1e9);
    logger_printf("tier 0. flush time: %zu ns, %.6lf s", tree->get_flush_time(), tree->get_flush_time() / 1e9);
    logger_printf("tier 0. decay scan time: %zu ns, %.6lf s", tree->get_decay_scan_time(), tree->get_decay_scan_time() / 1e9);
    logger_printf("tier 0. decay write time: %zu ns, %.6lf s", tree->get_decay_write_time(), tree->get_decay_write_time() / 1e9);
  }
  void access(SKey key, size_t vlen) { 
    if (cache_policy == CachePolicyT::kUseDecay) {
      // tree->append(key, ValueT(1, vlen));
    } else if (cache_policy == CachePolicyT::kUseTick || cache_policy == CachePolicyT::kUseFasterTick || cache_policy == CachePolicyT::kClockStyleDecay) {
      stat_input_bytes_.fetch_add(key.size() + sizeof(ValueT), std::memory_order_relaxed);
      tree->append(key, vlen);
    }
    check_decay(); 
  }
  auto delete_range(const std::pair<SKey, SKey>& range, std::pair<bool, bool> exclude_info) { 
    return tree->delete_range(range, exclude_info); 
  }
  auto seek_to_first() { return tree->seek_to_first(); }
  auto seek(SKey key) { return tree->seek(key); }
  auto weight_sum() { return tree->get_current_hot_size(); }
  auto range_data_size(const std::pair<SKey, SKey>& range) {
    return tree->range_data_size(range); 
  }
  bool is_hot(SKey key) {
    return tree->search_key(key);
  }
  bool is_stably_hot(SKey key) {
    return tree->is_stably_hot(key);
  }

  /* check if hot size exceeds the limit. If so, trigger decay.*/
  void check_decay() {
    if (cache_policy == CachePolicyT::kUseDecay) {
      // if (weight_sum() > hot_set_limit_) {
      //   std::unique_lock lck(decay_m_);
      //   if (weight_sum() > hot_set_limit_) {
      //     tree->trigger_decay();
      //   }
      // }  
    } else if (cache_policy == CachePolicyT::kUseTick || cache_policy == CachePolicyT::kUseFasterTick || cache_policy == CachePolicyT::kClockStyleDecay) {
      if (tree->check_decay_condition() && !is_updating_tick_threshold_.load(std::memory_order_relaxed)) {
        decay_cv_.notify_one(); 
      }
    }
  }
  void flush() {
    std::unique_lock lck(decay_m_);
    tree->all_flush();
    if (cache_policy == CachePolicyT::kUseTick || cache_policy == CachePolicyT::kUseFasterTick || cache_policy == CachePolicyT::kClockStyleDecay) {
      if (tree->check_decay_condition()) {
        update_tick_threshold();
      }
    }
  }
  void set_physical_size(size_t physical_size) {
    tree->set_physical_size(physical_size);
  }

  void update_tick_threshold() {
    if (cache_policy == CachePolicyT::kUseTick) {
      // auto hot_size = weight_sum();
      // KthEst<double> est(kEstPointNum, hot_set_limit_);
      // est.pre_scan1(hot_size);
      // auto append_all_to = [&](auto&& iter, auto&& scan_f) {
      //   for(; iter->valid(); iter->next()) {
      //     auto L = iter->read();
      //     // logger(L.second.get_score(), ", ", L.second.get_hot_size() + L.first.len());
      //     // We find the smallest tick so that sum(a.tick > tick) a.hot_size <= hot_limit.
      //     scan_f(-L.second.get_score(), L.second.get_hot_size() + L.first.len());
      //   }
      // };
      // StopWatch sw;
      // logger("first scan");
      // append_all_to(tree->seek_to_first(), [&] (auto&&... a) { est.scan1(a...); });
      // // logger("first scan end");
      // est.pre_scan2();
      // // logger("second scan");
      // append_all_to(tree->seek_to_first(), [&] (auto&&... a) { est.scan2(a...); });
      // // logger("second scan end");
      // double new_tick_threshold = est.get_interplot_kth();
      // tree->set_tick_threshold(-new_tick_threshold);
      // tree->set_decay_tick_threshold(-new_tick_threshold);
      // tree->update_tick_threshold();
      // logger("threshold: ", -new_tick_threshold, ", weight_sum: ", weight_sum(), "used time: ", sw.GetTimeInSeconds(), "s");
    } else if (cache_policy == CachePolicyT::kUseFasterTick) {
      // We sample some points and find the oldest 10% * SIZE records. These records are removed in major compaction (update_tick_threshold_and_update_est).
      // We sample points when we are doing compaction, and we get threshold from those points. 
      // The threshold is old but it ensures that we always remove >= 10% * SIZE records. 
      // But we have a constraint: The tick threshold must be able to be updated with new current tick.
      // For harmonic mean (class TickValue) it's impossible. 
      logger("first fast scan");
      StopWatch sw;
      tree->update_tick_threshold();
      tree->faster_decay();
      logger("weight_sum: ", weight_sum(), "used time: ", sw.GetTimeInSeconds(), "s");
    }
    decay_count_ += 1;
  }

  void set_new_hot_limit(size_t new_limit) {
    tree->set_hot_set_limit(new_limit);
    is_first_tick_update_ = true;
    decay_count_ = 0;
    check_decay();
  }

  void set_new_phy_limit(size_t new_limit) {
    tree->set_phy_limit(new_limit);
    is_first_tick_update_ = true;
    decay_count_ = 0;
    check_decay();
  }

  void set_all_limit(size_t new_hs_limit, size_t new_phy_limit) {
    tree->set_hot_set_limit(new_hs_limit);
    tree->set_phy_limit(new_phy_limit);
    is_first_tick_update_ = true;
    decay_count_ = 0;
    check_decay();
  }

  size_t get_phy_limit() const {
    return tree->get_phy_limit();
  }
  
  size_t get_hot_set_limit() const {
    return tree->get_hot_set_limit();
  }

  uint64_t get_min_hot_size_limit() { return tree->get_min_hot_size_limit(); }
  void set_min_hot_size_limit(uint64_t min_hot_size_limit) {
    tree->set_min_hot_size_limit(min_hot_size_limit);
  }

  uint64_t get_max_hot_size_limit() { return tree->get_max_hot_size_limit(); }
  void set_max_hot_size_limit(uint64_t max_hot_size_limit) {
    tree->set_max_hot_size_limit(max_hot_size_limit);
  }

  void set_proper_phy_limit() {
    tree->set_proper_phy_limit();
  }

  size_t get_real_hs_size() const {
    return tree->get_real_hs_size();
  }

  size_t get_real_phy_size() const {
    return tree->get_real_phy_size();
  }


  size_t decay_count() {
    return decay_count_;
  }

  bool HandleReadBytes(uint64_t *value) {
    *value = GetReadBytes();
    return true;
  }
  bool HandleWriteBytes(uint64_t *value) {
    *value = GetWriteBytes();
    return true;
  }
  bool HandleCompactionCPUNanos(uint64_t *value) {
    *value = tree->get_compact_time();
    return true;
  }
  bool HandleFlushCPUNanos(uint64_t *value) {
    *value = tree->get_flush_time();
    return true;
  }
  bool HandleDecayScanCPUNanos(uint64_t *value) {
    *value = tree->get_decay_scan_time();
    return true;
  }
  bool HandleDecayWriteCPUNanos(uint64_t *value) {
    *value = tree->get_decay_write_time();
    return true;
  }
  bool HandleCompactionThreadCPUNanos(uint64_t *value) {
    *value = tree->get_compact_thread_time();
    return true;
  }
  bool HandleFlushThreadCPUNanos(uint64_t *value) {
    *value = tree->get_flush_thread_time();
    return true;
  }
  bool HandleDecayThreadCPUNanos(uint64_t *value) {
    *value = decay_thread_timer_.GetTimeInNanos();
    return true;
  }

  private:
    void decay_thread() {
      while(!terminate_signal_) {
        std::unique_lock lck(decay_m_);
        decay_cv_.wait(lck, [&](){ return terminate_signal_ || tree->check_decay_condition(); });
        is_updating_tick_threshold_ = true;
        if (terminate_signal_) {
          return;
        }
        if (cache_policy == CachePolicyT::kUseTick || cache_policy == CachePolicyT::kUseFasterTick) {
          update_tick_threshold();
        } else if (cache_policy == CachePolicyT::kClockStyleDecay) {
          tree->clock_style_decay();
        }
        is_updating_tick_threshold_ = false;
      }
    }
};

}  // namespace ralt

#endif