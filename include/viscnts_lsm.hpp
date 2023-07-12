#ifndef VISCNTS_LSM_H__
#define VISCNTS_LSM_H__

#include <boost/fiber/buffered_channel.hpp>
#include <queue>

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
#include "kthest.hpp"
#include "compaction.hpp"

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

constexpr auto kLimitMin = 10;
constexpr auto kLimitMax = 20;
constexpr auto kMergeRatio = 0.1;
constexpr auto kUnmergedRatio = 0.1;
constexpr auto kUnsortedBufferSize = kSSTable;
constexpr auto kUnsortedBufferMaxQueue = 4;
constexpr auto kMaxFlushBufferQueueSize = 10;
constexpr auto kWaitCompactionSleepMilliSeconds = 100;

template <typename KeyCompT, typename ValueT>
class EstimateLSM {
  struct Partition {
    ImmutableFile<KeyCompT, ValueT> file_;
    DeletedRange deleted_ranges_;
    int global_range_counts_;
    double avg_hot_size_;
    double global_hot_size_;
    std::string filename_;

   public:
    Partition(Env* env, FileChunkCache* file_index_cache, FileChunkCache* file_key_cache, KeyCompT comp, const typename Compaction<KeyCompT, ValueT>::NewFileData& data)
        : file_(data.file_id, data.size, std::unique_ptr<RandomAccessFile>(env->openRAFile(data.filename)), data.range, file_index_cache, file_key_cache, comp),
          deleted_ranges_(),
          global_hot_size_(data.hot_size),
          filename_(data.filename) {
      global_range_counts_ = file_.counts();
      avg_hot_size_ = data.hot_size / (double) file_.counts();
    }
    ~Partition() {
      logger("delete, ", filename_);
      file_.remove();
    }

    SSTIterator<KeyCompT, ValueT> seek(SKey key) { return SSTIterator(&file_, key); }
    SSTIterator<KeyCompT, ValueT> begin() { return SSTIterator(&file_); }
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
      auto rank_pair = file_.rank_pair(range, {0, 0});
      // return deleted_ranges_.deleted_data_size(rank_pair);
      // Estimate
      return deleted_ranges_.deleted_counts(rank_pair) * avg_hot_size_;
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
    const ImmutableFile<KeyCompT, ValueT>& file() const { return file_; }
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
    std::vector<std::shared_ptr<Partition>> head_;
    size_t size_;
    size_t data_size_;
    double hot_size_;
  public:
    // iterator for level
    class LevelIterator {
      typename std::vector<std::shared_ptr<Partition>>::const_iterator vec_it_;
      typename std::vector<std::shared_ptr<Partition>>::const_iterator vec_it_end_;
      SSTIterator<KeyCompT, ValueT> iter_;
      std::shared_ptr<std::vector<std::shared_ptr<Partition>>> vec_ptr_;
      DeletedRange::Iterator del_ranges_iterator_;

     public:
      LevelIterator() {}
      LevelIterator(const std::vector<std::shared_ptr<Partition>>& vec, uint32_t id, SSTIterator<KeyCompT, ValueT>&& iter, DeletedRange::Iterator&& del_iter)
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
      LevelIterator(std::shared_ptr<std::vector<std::shared_ptr<Partition>>> vec_ptr, uint32_t id, SSTIterator<KeyCompT, ValueT>&& iter,
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

    // Used in batch_seek
    class LevelAsyncSeekHandle {
      public:
        LevelAsyncSeekHandle(SKey key, const Partition& part, const Level& level, int where, async_io::AsyncIOQueue& aio)
          : part_(part), level_(level), where_(where), sst_handle_(key, part.file(), aio) {}
        void init() {
          sst_handle_.init();
        }
        /* the chunk has been read. it returns the iterator when it has found result. */
        template<typename T>
        std::optional<LevelIterator> next(T&& aio_info) {
          auto result = sst_handle_.next(std::forward<T>(aio_info));
          if(!result) {
            return {};
          }
          auto del_range_iter = DeletedRange::Iterator(result.value().rank(), part_.deleted_ranges());
          return LevelIterator(level_.head_, where_, std::move(result.value()), std::move(del_range_iter));
        }
      private:
        const Partition& part_;
        const Level& level_;
        int where_;
        typename ImmutableFile<KeyCompT, ValueT>::ImmutableFileAsyncSeekHandle sst_handle_;
    };

    Level() : size_(0), data_size_(0), hot_size_(0) {}
    Level(size_t size, double hot_size) : size_(size), hot_size_(hot_size) {}
    size_t size() const { return size_; }
    double hot_size() const { return hot_size_; }
    /* seek the first key >= key. */
    LevelIterator seek(SKey key, KeyCompT comp) const {
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
    std::optional<LevelAsyncSeekHandle> get_async_seek_handle(SKey key, KeyCompT comp, async_io::AsyncIOQueue& aio) const {
      int l = 0, r = head_.size() - 1, where = -1;
      while (l <= r) {
        int mid = (l + r) >> 1;
        if (comp(key, head_[mid]->range().second) <= 0)
          where = mid, r = mid - 1;
        else
          l = mid + 1;
      }
      if (where == -1) return {};
      return LevelAsyncSeekHandle(key, *head_[where], *this, where, aio);
    }
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
    void append_par(std::shared_ptr<Partition> par) {
      logger("[append_par]: ", par->size());
      size_ += par->size();
      data_size_ += par->data_size();
      hot_size_ += par->hot_size();
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

    
    const std::vector<std::shared_ptr<Partition>>& get_pars() const {
      return head_;
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
    std::atomic<uint32_t> ref_;
    double hot_size_overestimate_;
    KeyCompT comp_;

    std::mutex* del_mutex_;

    using LevelIteratorSetT = SeqIteratorSet<typename Level::LevelIterator, KeyCompT, ValueT>;

   public:
    SuperVersion(std::mutex* mutex, KeyCompT comp)
        : ref_(1), hot_size_overestimate_(0), comp_(comp), del_mutex_(mutex) {}
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
    LevelIteratorSetT seek(SKey key) const {
      LevelIteratorSetT ret(comp_);
      for (auto& a : tree_) ret.push(a->seek(key, comp_));
      return ret;
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

    /* seek use async seek handles. Not so slow but slower. (why?)*/
    LevelIteratorSetT batch_seek(SKey key) const {
      LevelIteratorSetT ret(comp_);
      std::vector<typename Level::LevelAsyncSeekHandle> handles;
      async_io::AsyncIOQueue aio(tree_.size() + 10);
      auto process_handle = [&](auto&& handle) {
        if (handle) {
          handle.value().init();
          handles.push_back(std::move(handle.value()));
          auto result = handles.back().next(handles.size() - 1);
          if (result) {
            ret.push(std::move(result.value()));
            return;
          }
        }
      };
      for (auto& a : tree_) {
        process_handle(a->get_async_seek_handle(key, comp_, aio));
      }
      size_t now_in_q = aio.size();
      size_t next = 0;
      if(aio.size()) {
        aio.submit();
      }
      while(aio.size()) {
        auto cur = aio.get_one().value();
        auto result = handles[cur].next(cur);
        if (result) {
          now_in_q--;
          ret.push(std::move(result.value()));
        } else {
          next++;
        }
        if (next == now_in_q) {
          aio.submit();
          next = 0;
        }
      }
      return ret;
    }

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

    std::vector<std::shared_ptr<Level>> flush_bufs(const std::vector<UnsortedBuffer<ValueT>*>& bufs, FileName* filename, Env* env, 
                                                   FileChunkCache* file_index_cache, FileChunkCache* file_key_cache,
                                                   KeyCompT comp, double current_tick) {
      if (bufs.size() == 0) return {};
      std::vector<std::shared_ptr<Level>> ret_vectors;
      auto flush_func = [current_tick, file_index_cache, file_key_cache](FileName* filename, Env* env, KeyCompT comp, UnsortedBuffer<ValueT>* buf, std::mutex& mu,
                           std::vector<std::shared_ptr<Level>>& ret_vectors) {
        Compaction<KeyCompT, ValueT> worker(current_tick, filename, env, comp);
        buf->sort(comp);
        auto iter = std::make_unique<typename UnsortedBuffer<ValueT>::Iterator>(*buf);
        auto [files, hot_size] = worker.flush(*iter);
        auto level = std::make_shared<Level>(0, hot_size);
        for (auto& a : files) level->append_par(std::make_shared<Partition>(env, file_index_cache, file_key_cache, comp, a));
        delete buf;
        std::unique_lock lck(mu);
        ret_vectors.push_back(std::move(level));
      };
      std::vector<std::thread> thread_pool;
      std::mutex thread_mutex;
      // We now expect the number of SSTs is small, e.g. 4
      for (uint32_t i = 1; i < bufs.size(); i++) {
        thread_pool.emplace_back(flush_func, filename, env, comp, bufs[i], std::ref(thread_mutex), std::ref(ret_vectors));
      }
      if (bufs.size() >= 1) {
        flush_func(filename, env, comp, bufs[0], thread_mutex, std::ref(ret_vectors));
      }
      // for (int i = 0; i < bufs.size(); ++i) flush_func(filename, env, comp, bufs[i], thread_mutex, ret);
      for (auto& a : thread_pool) a.join();
      return ret_vectors;
    }
    SuperVersion* push_new_buffers(const std::vector<std::shared_ptr<Level>>& vec) {
      auto ret = new SuperVersion(del_mutex_, comp_);
      ret->tree_ = tree_;
      // logger(tree_.size(), ", ", ret->tree_.size());
      ret->hot_size_overestimate_ = hot_size_overestimate_;
      ret->tree_.insert(ret->tree_.end(), vec.begin(), vec.end());
      for (auto&& x : vec) {
        ret->hot_size_overestimate_ += x->hot_size();
      }
      ret->_sort_levels();
      return ret;
    }
    enum class JobType {
      kDecay = 0,
      kMajorCompaction = 1,
      kCompaction = 2,
    };
    SuperVersion* compact(FileName* filename, Env* env, FileChunkCache* file_index_cache, FileChunkCache* file_key_cache, KeyCompT comp, double current_tick, TickFilter<ValueT> tick_filter, JobType job_type) const {
      auto Lsize = 0;
      for (auto& a : tree_) Lsize += a->size();
      // check if overlap with any partition
      auto check_func = [comp, this](const Partition& par) {
        for (auto& a : tree_)
          if (a->overlap(par.range().first, par.range().second, comp)) return true;
        return false;
      };
      // add partitions that is overlapped with other levels.
      // the remaining partitions are stored in a std::vector.
      std::vector<std::shared_ptr<Partition>> rest;
      auto add_level = [&rest, &check_func](const Level& level) {
        auto for_iter_ptr = std::make_shared<std::vector<std::shared_ptr<Partition>>>();
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
      // print size
      {
        std::string __print_str = "[";
        for (auto& a : tree_) {
          __print_str += std::to_string(a->size() / (double)kSSTable) + ", ";
        }
        logger(__print_str + "]");  
      }
      
      // decay
      if (job_type == JobType::kDecay) {
        logger("[addr, decay size, Lsize]: ", this, ", ", hot_size_overestimate_, ", ", Lsize);
        auto iters = std::make_unique<LevelIteratorSetT>(comp);

        // logger("Major Compaction tree_.size() = ", tree_.size());
        // push all iterators to necessary partitions to SeqIteratorSet.
        for (auto& a : tree_) iters->push(a->seek_to_first());
        iters->build();

        // compaction...
        Compaction<KeyCompT, ValueT> worker(current_tick, filename, env, comp);
        auto [files, hot_size] = worker.decay1(*iters);
        auto ret = new SuperVersion(del_mutex_, comp_);
        // major compaction, so levels except largest_ become empty.
        
        // std::shared_ptr move... maybe no problem.
        {
          std::unique_lock lck(*del_mutex_);
          auto level = std::make_shared<Level>(0, hot_size);
          for (auto& a : files) {
            auto par = std::make_shared<Partition>(env, file_index_cache, file_key_cache, comp, a);
            level->append_par(std::move(par));
          }   
          ret->tree_.push_back(std::move(level));
          // calculate new current decay size
          ret->hot_size_overestimate_ = ret->_calc_current_hot_size();
          logger("[new decay size]: ", ret->hot_size_overestimate_);
          ret->_sort_levels();
        }
         
        return ret;
      } else if (job_type == JobType::kMajorCompaction) {
        logger("[addr, decay size, Lsize]: ", this, ", ", hot_size_overestimate_, ", ", Lsize);
        logger(tick_filter.get_tick_threshold());
        auto iters = std::make_unique<LevelIteratorSetT>(comp);

        // logger("Major Compaction tree_.size() = ", tree_.size());
        // push all iterators to necessary partitions to SeqIteratorSet.
        for (auto& a : tree_) iters->push(a->seek_to_first());
        iters->build();

        // compaction...
        Compaction<KeyCompT, ValueT> worker(current_tick, filename, env, comp);
        auto [files, hot_size] = worker.flush_with_filter(*iters, tick_filter);
        auto ret = new SuperVersion(del_mutex_, comp_);
        // major compaction, so levels except largest_ become empty.
        
        // std::shared_ptr move... maybe no problem.
        {
          std::unique_lock lck(*del_mutex_);
          auto level = std::make_shared<Level>(0, 0);
          for (auto& a : files) {
            auto par = std::make_shared<Partition>(env, file_index_cache, file_key_cache, comp, a);
            level->append_par(std::move(par));
          }   
          ret->tree_.push_back(std::move(level));
          // calculate new current decay size
          ret->hot_size_overestimate_ = ret->_calc_current_hot_size();
          logger("[new decay size]: ", ret->hot_size_overestimate_);
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
          Compaction<KeyCompT, ValueT> worker(current_tick, filename, env, comp);
          auto [files, hot_size] = worker.flush_with_filter(*iters, tick_filter);
          auto ret = new SuperVersion(del_mutex_, comp_);
          // minor compaction?
          // hot_size_overestimate = largest_ + remaining levels + new level

          {  
            std::unique_lock lck(*del_mutex_);
            ret->tree_ = tree_;
            for (int i = 0; i < where; i++) ret->hot_size_overestimate_ += tree_[i]->hot_size();
            auto rest_iter = rest.begin();
            auto level = std::make_shared<Level>();
            for (auto& a : files) {
              auto par = std::make_shared<Partition>(env, file_index_cache, file_key_cache, comp, a);
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
            ret->hot_size_overestimate_ = ret->_calc_current_hot_size();
            ret->_sort_levels();
            return ret;
          }

        }
        return nullptr;
      }
    }

    size_t range_count(const std::pair<SKey, SKey>& range, KeyCompT comp) {
      size_t ans = 0;
      for (auto& a : tree_) ans += a->range_count(range, comp);
      return ans;
    }

    size_t range_data_size(const std::pair<SKey, SKey>& range, KeyCompT comp) {
      size_t ans = 0;
      for (auto& a : tree_) ans += a->range_data_size(range, comp);
      return ans;
    }

    SuperVersion* delete_range(std::pair<SKey, SKey> range, KeyCompT comp, std::pair<bool, bool> exclude_info) {
      auto ret = new SuperVersion(del_mutex_, comp_);
      ret->tree_ = tree_;
      ret->hot_size_overestimate_ = hot_size_overestimate_;
      for (auto& a : ret->tree_) a->delete_range(range, comp, exclude_info);
      ret->hot_size_overestimate_ = ret->_calc_current_hot_size();
      return ret;
    }

    double get_current_hot_size() const { return hot_size_overestimate_; }

   private:
    double _calc_current_hot_size() {
      double ret = 0;
      for (auto& a : tree_) ret += a->hot_size();
      return ret;
    }
    void _sort_levels() {
      std::sort(tree_.begin(), tree_.end(), [](const std::shared_ptr<Level>& x, const std::shared_ptr<Level>& y) { return x->size() > y->size(); });
    }
  };
  Env* env_;
  std::unique_ptr<FileName> filename_;
  KeyCompT comp_;
  SuperVersion* sv_;
  std::thread compact_thread_;
  std::thread flush_thread_;
  std::atomic<bool> terminate_signal_;
  UnsortedBufferPtrs<ValueT> bufs_;
  std::mutex sv_mutex_;
  std::mutex sv_load_mutex_;
  // Only one thread can modify sv.
  std::mutex sv_modify_mutex_;
  std::vector<std::shared_ptr<Level>> flush_buf_vec_;
  std::mutex flush_buf_vec_mutex_;
  std::condition_variable signal_flush_to_compact_;
  double hot_size_overestimate_{0};
  size_t sv_tick_{0};
  // 0: idle, 1: working
  uint8_t flush_thread_state_{0};
  uint8_t compact_thread_state_{0};

  // Used for tick
  std::atomic<size_t>& current_tick_;
  std::atomic<double> tick_threshold_{-114514};

  // default cache
  std::unique_ptr<FileChunkCache> file_cache_;


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
  EstimateLSM(Env* env, size_t file_cache_size, std::unique_ptr<FileName>&& filename,
              KeyCompT comp, std::atomic<size_t>& current_tick)
      : env_(env),
        filename_(std::move(filename)),
        comp_(comp),
        sv_(new SuperVersion(&sv_mutex_, comp)),
        terminate_signal_(0),
        bufs_(kUnsortedBufferSize, kUnsortedBufferMaxQueue),
        file_cache_(std::make_unique<FileChunkCache>(file_cache_size)),
        current_tick_(current_tick) {
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
  void append(SKey key, ValueT value) { 
    bufs_.append_and_notify(key, value);
  }
  auto seek(SKey key) {
    auto sv = get_current_sv();
    return std::make_unique<SuperVersionIterator>(LevelIteratorSetTForScan(sv->seek(key), get_current_tick(), get_tick_filter()), sv);
  }

  auto seek_to_first() {
    auto sv = get_current_sv();
    return std::make_unique<SuperVersionIterator>(LevelIteratorSetTForScan(sv->seek_to_first(), get_current_tick(), get_tick_filter()), sv);
  }

  auto batch_seek(SKey key) {
    auto sv = get_current_sv();
    return std::make_unique<SuperVersionIterator>(LevelIteratorSetTForScan(sv->batch_seek(key), get_current_tick(), get_tick_filter()), sv);
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

  void delete_range(std::pair<SKey, SKey> range, std::pair<bool, bool> exclude_info) {
    if (comp_(range.first, range.second) > 0) return;
    std::unique_lock del_range_lck(sv_modify_mutex_);
    auto new_sv = sv_->delete_range(range, comp_, exclude_info);
    _update_superversion(new_sv);
  }

  void trigger_decay() {
    std::unique_lock del_range_lck(sv_modify_mutex_);
    auto new_sv = sv_->compact(filename_.get(), env_, file_cache_.get(), nullptr, comp_, get_current_tick(), get_tick_filter(), SuperVersion::JobType::kDecay);
    _update_superversion(new_sv);
  }

  double get_current_hot_size() {
    return hot_size_overestimate_;
  }

  bool search_key(SKey key) {
    auto sv = get_current_sv();
    return sv->search_key(key);
  }

  double get_current_tick() const {
    return current_tick_.load(std::memory_order_relaxed);
  }

  TickFilter<ValueT> get_tick_filter() const {
    return TickFilter<ValueT>(tick_threshold_.load(std::memory_order_relaxed));
  }

  void update_tick_threshold(double new_threshold) {
    tick_threshold_ = new_threshold;
    std::unique_lock del_range_lck(sv_modify_mutex_);
    logger(get_tick_filter().get_tick_threshold(), ", ", new_threshold);
    auto new_sv = sv_->compact(filename_.get(), env_, file_cache_.get(), nullptr, comp_, get_current_tick(), get_tick_filter(), SuperVersion::JobType::kMajorCompaction);
    _update_superversion(new_sv);
  }

  double get_tick_threshold() const {
    return tick_threshold_.load(std::memory_order_relaxed);
  }

  /* Used for stats */
  const FileChunkCache* get_cache() const {
    return file_cache_.get();
  }

 private:
  void flush_thread() {
    while (!terminate_signal_) {
      flush_thread_state_ = 0;
      auto buf_q_ = bufs_.wait_and_get();
      if (terminate_signal_) return;
      if (buf_q_.empty()) continue;
      flush_thread_state_ = 1;
      auto new_vec = sv_->flush_bufs(buf_q_, filename_.get(), env_, file_cache_.get(), nullptr, comp_, current_tick_);
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
        auto new_compacted_sv = last_compacted_sv->compact(filename_.get(), env_, file_cache_.get(), nullptr, comp_, get_current_tick(), TickFilter<ValueT>(-114514), SuperVersion::JobType::kCompaction);
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
      hot_size_overestimate_ = new_sv->get_current_hot_size();
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

enum class CachePolicyT {
  kUseDecay = 0,
  kUseTick
};

// Viscnts, implement lsm tree and other things.

constexpr size_t kEstPointNum = 1e4;
constexpr double kHotSetExceedLimit = 0.1;

template <typename KeyCompT, typename ValueT, CachePolicyT cache_policy>
class VisCnts {
  std::unique_ptr<EstimateLSM<KeyCompT, ValueT>> tree[2];
  KeyCompT comp_;
  size_t hot_set_limit_;
  std::mutex decay_m_;
  std::atomic<bool> is_updating_tick_threshold_{false};
  bool terminate_signal_{false};
  std::mutex decay_thread_m_;
  std::condition_variable decay_cv_;
  std::thread decay_thread_;
  std::atomic<size_t> current_tick_{0};

 public:
  using IteratorT = typename EstimateLSM<KeyCompT, ValueT>::SuperVersionIterator;
  // Use different file path for two trees.
  VisCnts(KeyCompT comp, const std::string& path, size_t delta)
    : tree{std::make_unique<EstimateLSM<KeyCompT, ValueT>>(createDefaultEnv(), kIndexCacheSize, std::make_unique<FileName>(0, path + "a0"), comp, current_tick_), 
          std::make_unique<EstimateLSM<KeyCompT, ValueT>>(createDefaultEnv(), kIndexCacheSize, std::make_unique<FileName>(0, path + "a1"), comp, current_tick_)},
      comp_(comp), hot_set_limit_(delta) {
        decay_thread_ = std::thread([&](){ decay_thread(); });
      }
  ~VisCnts() {
    {
      std::unique_lock lck(decay_thread_m_);
      terminate_signal_ = true;
      decay_cv_.notify_all();  
    }
    decay_thread_.join();
    
    auto stat0 = tree[0]->get_cache()->get_stats();
    logger_printf("tier 0. hit: %d, access: %d", stat0.hit_count, stat0.access_count);
    auto stat1 = tree[1]->get_cache()->get_stats();
    logger_printf("tier 1. hit: %d, access: %d", stat1.hit_count, stat1.access_count);
  }
  void access(int tier, SKey key, size_t vlen) { 
    if (cache_policy == CachePolicyT::kUseDecay) {
      tree[tier]->append(key, ValueT(1, vlen));
    } else if (cache_policy == CachePolicyT::kUseTick) {
      tree[tier]->append(key, ValueT(current_tick_.load(std::memory_order_relaxed), vlen));
      current_tick_.fetch_add(1, std::memory_order_relaxed);
    }
    check_decay(); 
  }
  auto delete_range(int tier, const std::pair<SKey, SKey>& range, std::pair<bool, bool> exclude_info) { 
    return tree[tier]->delete_range(range, exclude_info); 
  }
  auto seek_to_first(int tier) { return tree[tier]->seek_to_first(); }
  auto seek(int tier, SKey key) { return tree[tier]->seek(key); }
  auto weight_sum(int tier) { return tree[tier]->get_current_hot_size(); }
  auto range_data_size(int tier, const std::pair<SKey, SKey>& range) { 
    static int x = 0;
    x++;
    if (x%10000==0){
      logger(x);
    }
    return tree[tier]->range_data_size(range); 
  }
  bool is_hot(int tier, SKey key) {
    return tree[tier]->search_key(key);
  }
  void transfer_range(int src_tier, int dst_tier, const std::pair<SKey, SKey>& range, std::pair<bool, bool> exclude_info) {
    if (src_tier == dst_tier) return;
    if (comp_(range.first, range.second) > 0) return;
    std::unique_lock lck(decay_m_);
    auto iter = tree[src_tier]->seek(range.first);
    if (exclude_info.first) {
      while (iter->valid()) {
        auto L = iter->read();
        if (comp_(L.first, range.first) == 0) {
          iter->next();
        } else {
          break;
        }
      }
    }
    int cnt = 0;
    while(iter->valid()) {
      auto L = iter->read();
      if ((exclude_info.second && (comp_(L.first, range.second) >= 0)) || (!exclude_info.second && comp_(L.first, range.second) > 0)) {
        break;
      }
      cnt++;
      tree[dst_tier]->append(L.first, L.second);
      iter->next();
    }
    tree[src_tier]->delete_range(range, exclude_info);
  }

  /* check if hot size exceeds the limit. If so, trigger decay.*/
  void check_decay() {
    if (cache_policy == CachePolicyT::kUseDecay) {
      if (weight_sum(0) + weight_sum(1) > hot_set_limit_) {
        std::unique_lock lck(decay_m_);
        if (weight_sum(0) + weight_sum(1) > hot_set_limit_) {
          tree[0]->trigger_decay();
          tree[1]->trigger_decay();  
        }
      }  
    } else if (cache_policy == CachePolicyT::kUseTick) {
      if (weight_sum(0) + weight_sum(1) > hot_set_limit_ * (1 + kHotSetExceedLimit) && !is_updating_tick_threshold_.load(std::memory_order_relaxed)) {
        auto hot_size = weight_sum(0) + weight_sum(1);
        decay_cv_.notify_one(); 
      }
    }
  }
  void flush() {
    std::unique_lock lck(decay_m_);
    for(auto& a : tree) a->all_flush();
    if (cache_policy == CachePolicyT::kUseTick) {
      if (weight_sum(0) + weight_sum(1) > hot_set_limit_ * (1 + kHotSetExceedLimit)) {
        update_tick_threshold();
      }
    }
  }
  size_t get_hot_size(size_t tier) {
    return tree[tier]->get_current_hot_size();
  }

  void update_tick_threshold() {
    auto hot_size = weight_sum(0) + weight_sum(1);
    KthEst<double> est(kEstPointNum, hot_set_limit_);
    est.pre_scan1(hot_size);
    auto append_all_to = [&](auto&& iter, auto&& scan_f) {
      for(; iter->valid(); iter->next()) {
        auto L = iter->read();
        // logger(L.second.get_tick(), ", ", L.second.get_hot_size() + L.first.len());
        // We find the smallest tick so that sum(a.tick > tick) a.hot_size <= hot_limit.
        scan_f(-L.second.get_tick(), L.second.get_hot_size() + L.first.len());
      }
    };
    StopWatch sw;
    logger("first scan");
    append_all_to(tree[0]->seek_to_first(), [&] (auto&&... a) { est.scan1(a...); });
    append_all_to(tree[1]->seek_to_first(), [&] (auto&&... a) { est.scan1(a...); });
    // logger("first scan end");
    est.pre_scan2();
    // logger("second scan");
    append_all_to(tree[0]->seek_to_first(), [&] (auto&&... a) { est.scan2(a...); });
    append_all_to(tree[1]->seek_to_first(), [&] (auto&&... a) { est.scan2(a...); });
    // logger("second scan end");
    double new_tick_threshold = est.get_interplot_kth();
    tree[0]->update_tick_threshold(-new_tick_threshold);
    tree[1]->update_tick_threshold(-new_tick_threshold);
    logger("threshold: ", -new_tick_threshold, ", weight_sum0: ", weight_sum(0), ", weight_sum1: ", weight_sum(1), "used time: ", sw.GetTimeInSeconds(), "s");
  }

  private:
    void decay_thread() {
      while(!terminate_signal_) {
        std::unique_lock lck(decay_m_);
        decay_cv_.wait(lck, [&](){ return terminate_signal_ || weight_sum(0) + weight_sum(1) > hot_set_limit_ * (1 + kHotSetExceedLimit); });
        is_updating_tick_threshold_ = true;
        if (terminate_signal_) {
          return;
        }
        if (cache_policy == CachePolicyT::kUseTick) {
          update_tick_threshold();
        }
        is_updating_tick_threshold_ = false;
      }
    }
};

}  // namespace viscnts_lsm

#endif