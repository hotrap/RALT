#include "viscnts.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <queue>
#include <set>
#include <thread>

#include "alloc.hpp"
#include "cache.hpp"
#include "common.hpp"
#include "file.hpp"
#include "hash.hpp"
#include "key.hpp"
#include "memtable.hpp"
#include "splay.hpp"

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

static std::mutex logger_m_;
template <typename... Args>
void logger(Args&&... a) {
  std::unique_lock lck_(logger_m_);
  (std::cerr << ... << a) << std::endl;
}

template <typename... Args>
void logger_printf(const char* str, Args&&... a) {
  std::unique_lock lck_(logger_m_);
  printf(str, a...);
  fflush(stdout);
}

const static size_t kPageSize = 1 << 12;                // 4 KB
const static size_t kMagicNumber = 0x25a65facc3a23559;  // echo viscnts | sha1sum
const static size_t kSSTable = 1 << 20;
const static size_t kRatio = 10;
const static size_t kBatchSize = 1 << 20;
const static size_t kChunkSize = 1 << 12;  // 8 KB
const static size_t kIndexChunkSize = 1 << 12;
// about kMemTable... on average, we expect the size of index block < kDataBlockSize

// generate global filename
class FileName {
  std::atomic<size_t> file_ts_;
  std::filesystem::path path_;

 public:
  explicit FileName(size_t ts, std::string path) : file_ts_(ts), path_(path) { std::filesystem::create_directories(path); }
  std::string gen_filename(size_t id) { return "lainlainlainlain" + std::to_string(id) + ".data"; }
  std::string gen_path() { return (path_ / gen_filename(file_ts_)).string(); }
  std::string next() { return (path_ / gen_filename(++file_ts_)).string(); }
  std::pair<std::string, size_t> next_pair() {
    auto id = ++file_ts_;
    return {(path_ / gen_filename(id)).string(), id};
  }
};

// information of file blocks in SST files
struct FileBlockHandle {
  // The structure of a file block
  // [handles]
  // [kv pairs...]
  uint32_t offset;
  uint32_t size;    // The size of SST doesn't exceed 4GB
  uint32_t counts;  // number of keys in the fileblock
  FileBlockHandle() { offset = size = counts = 0; }
  explicit FileBlockHandle(uint32_t offset, uint32_t size, uint32_t counts) : offset(offset), size(size), counts(counts) {}
};

// this iterator is used in compaction
// or decay
class SeqIterator {
 public:
  virtual ~SeqIterator() = default;
  virtual bool valid() = 0;
  virtual void next() = 0;
  virtual std::pair<SKey, SValue> read() = 0;
  virtual SeqIterator* copy() = 0;
};

// manage a temporary chunk (several pages) in SST files
// I don't use cache here, because we don't need it? (I think we can cache index blocks)
class Chunk {
  uint8_t* data_;
  BaseAllocator* alloc_;

 public:
  Chunk() {
    data_ = nullptr;
    alloc_ = nullptr;
  }
  // read a chunk from file
  Chunk(uint32_t offset, BaseAllocator* alloc, RandomAccessFile* file_ptr) {
    data_ = nullptr;
    acquire(offset, alloc, file_ptr);
  }
  Chunk(const Chunk& c) {
    alloc_ = c.alloc_;
    data_ = alloc_->allocate(kChunkSize);
    memcpy(data_, c.data_, kChunkSize);
  }
  Chunk& operator=(const Chunk& c) {
    alloc_ = c.alloc_;
    if (!data_) data_ = alloc_->allocate(kChunkSize);
    memcpy(data_, c.data_, kChunkSize);
    return (*this);
  }
  Chunk(Chunk&& c) {
    data_ = c.data_;
    alloc_ = c.alloc_;
    c.data_ = nullptr;
  }
  Chunk& operator=(Chunk&& c) {
    if (data_) alloc_->release(data_);
    data_ = c.data_;
    alloc_ = c.alloc_;
    c.data_ = nullptr;
    return (*this);
  }
  ~Chunk() {
    if (data_) alloc_->release(data_);
  }
  uint8_t* data(uint32_t offset = 0) const { return data_ + offset; }

  // read a chunk from file, reuse the allocated data
  void acquire(uint32_t offset, BaseAllocator* alloc, RandomAccessFile* file_ptr) {
    if (!data_) data_ = alloc->allocate(kChunkSize), alloc_ = alloc;
    Slice result;
    auto err = file_ptr->read(offset, kChunkSize, data_, result);
    assert(result.data() == data_);
    if (err) {
      logger("error in Chunk::Chunk(): ", err);
      exit(-1);
      data_ = nullptr;
      return;
    }
    if (result.len() != kChunkSize) {
      // logger("acquire < kChunkSize");
      memset(data_ + result.len(), 0, kChunkSize - result.len());
    }
  }
};

// two types of fileblock, one stores (key size, key), the other stores (key size, key, value)
template <typename KV, typename KVComp>
class FileBlock {     // process blocks in a file
  uint32_t file_id_;  // file id
  FileBlockHandle handle_;
  BaseAllocator* alloc_;
  RandomAccessFile* file_ptr_;

  // [keys] [offsets]
  // serveral (now 2) kinds of file blocks
  // data block and index block
  // their values are different: data block stores SValue, index block stores FileBlockHandle.
  // attention: the size of value is fixed.

  uint32_t offset_index_;
  KVComp comp_;

  constexpr static auto kValuePerChunk = kChunkSize / sizeof(uint32_t);

  // get the chunk id and chunk offset from an offset.
  std::pair<uint32_t, uint32_t> _kv_offset(uint32_t offset) {
    assert(offset < handle_.offset + handle_.size);
    // we must ensure no keys cross two chunks in SSTBuilder.
    // offset is absolute.
    return {offset / kChunkSize, offset % kChunkSize};
  }

  // get the id-th offset. Since offsets are all uint32_t, the address can be calculated directly.
  std::pair<uint32_t, uint32_t> _pos_offset(uint32_t id) {
    assert(id < handle_.counts);
    static_assert(kChunkSize % sizeof(uint32_t) == 0);
    const auto the_offset = offset_index_ + sizeof(uint32_t) * id;
    return {the_offset / kChunkSize, the_offset % kChunkSize};
  }

 public:
  // Only seek, maintain two Chunks, one is of key(kv pairs), one is of value(offset).
  class SeekIterator {
   public:
    SeekIterator(FileBlock<KV, KVComp> block) : block_(block), current_value_id_(-1), current_key_id_(-1) {}
    void seek_and_read(uint32_t id, KV& key) {
      uint32_t _offset;
      // find offset
      auto [chunk_id, offset] = block_._pos_offset(id);
      if (current_value_id_ != chunk_id) block_.acquire(current_value_id_ = chunk_id, currenct_value_chunk_);
      _offset = block_.read_value(offset, currenct_value_chunk_);
      // read key
      auto [chunk_key_id, key_offset] = block_._kv_offset(_offset);
      if (current_key_id_ != chunk_key_id) block_.acquire(current_key_id_ = chunk_key_id, current_key_chunk_);
      block_.read_key(key_offset, current_key_chunk_, key);
    }

    auto seek_offset(uint32_t id) {
      // find offset
      auto [chunk_id, offset] = block_._pos_offset(id);
      if (current_value_id_ != chunk_id) block_.acquire(current_value_id_ = chunk_id, currenct_value_chunk_);
      auto ret = block_.read_value(offset, currenct_value_chunk_);
      return ret;
    }

   private:
    FileBlock<KV, KVComp> block_;
    Chunk currenct_value_chunk_, current_key_chunk_;
    uint32_t current_value_id_, current_key_id_;
  };

  // maintain one Chunk for key. Thus, we need the offset of the first key, and its id.
  class EnumIterator {
   public:
    EnumIterator() { id_ = block_.handle_.counts = 0; }
    EnumIterator(FileBlock<KV, KVComp> block, uint32_t offset, uint32_t id) : block_(block), id_(id), key_size_(0) {
      // logger_printf("EI[%d, %d]", id_, block_.counts());
      if (valid()) {
        auto [chunk_id, chunk_offset] = block_._kv_offset(offset);
        current_key_id_ = chunk_id;
        offset_ = chunk_offset;
        current_key_chunk_ = block_.acquire(current_key_id_);
      }
    }
    EnumIterator(const EnumIterator& it) noexcept { (*this) = (it); }

    EnumIterator(EnumIterator&& it) noexcept { (*this) = std::move(it); }

    EnumIterator& operator=(EnumIterator&& it) noexcept {
      block_ = it.block_;
      offset_ = it.offset_;
      current_key_id_ = it.current_key_id_;
      id_ = it.id_;
      current_key_chunk_ = std::move(it.current_key_chunk_);
      assert(it.current_key_chunk_.data() == nullptr);
      key_size_ = 0;
      return (*this);
    }

    EnumIterator& operator=(const EnumIterator& it) noexcept {
      block_ = it.block_;
      offset_ = it.offset_;
      current_key_id_ = it.current_key_id_;
      id_ = it.id_;
      current_key_chunk_ = it.current_key_chunk_;
      key_size_ = 0;
      return (*this);
    }

    void next() {
      // logger_printf("nextEI[%d, %d]", id_, block_.counts());
      assert(valid());
      // logger("[next]", id_, " ", block_.handle_.counts);
      if (!key_size_) _read_size();
      offset_ += key_size_;
      assert(offset_ <= kChunkSize);
      key_size_ = 0, id_++;
      if (valid() && (offset_ + sizeof(uint32_t) >= kChunkSize || block_.is_empty_key(offset_, current_key_chunk_))) {
        offset_ = 0, current_key_id_++;
        block_.acquire(current_key_id_, current_key_chunk_);
      }
    }
    auto read(KV& key) {
      assert(valid());
      block_.read_key(offset_, current_key_chunk_, key);
      key_size_ = key.size();
    }
    bool valid() { return id_ < block_.handle_.counts; }

    int rank() { return id_; }

    void jump(uint32_t new_id) {
      // logger_printf("jumpEI[%d, %d]", id_, new_id);
      if (new_id >= block_.handle_.counts) {
        id_ = block_.handle_.counts;
        return;
      }
      if (new_id - id_ < kIndexChunkSize) {
        while (id_ < new_id) next();
      } else {
        SeekIterator it = SeekIterator(block_);
        id_ = new_id;
        offset_ = it.seek_offset(id_);
        auto [chunk_id, chunk_offset] = block_._kv_offset(offset_);
        current_key_id_ = chunk_id;
        offset_ = chunk_offset;
        current_key_chunk_ = block_.acquire(current_key_id_);
      }
    }

   private:
    FileBlock<KV, KVComp> block_;
    Chunk current_key_chunk_;
    uint32_t current_key_id_, offset_, id_, key_size_;

    void _read_size() { key_size_ = block_.read_key_size(offset_, current_key_chunk_); }
  };

  FileBlock() {
    file_id_ = 0;
    alloc_ = nullptr;
    file_ptr_ = nullptr;
    offset_index_ = 0;
  }
  explicit FileBlock(uint32_t file_id, FileBlockHandle handle, BaseAllocator* alloc, RandomAccessFile* file_ptr, const KVComp& comp)
      : file_id_(file_id), handle_(handle), alloc_(alloc), file_ptr_(file_ptr), comp_(comp) {
    offset_index_ = handle_.offset + handle_.size - handle_.counts * sizeof(uint32_t);
    assert(offset_index_ % kChunkSize == 0);
  }

  explicit FileBlock(uint32_t file_id, FileBlockHandle handle, BaseAllocator* alloc, RandomAccessFile* file_ptr, uint32_t offset_index,
                     const KVComp& comp)
      : file_id_(file_id), handle_(handle), alloc_(alloc), file_ptr_(file_ptr), comp_(comp) {
    offset_index_ = offset_index;
  }

  Chunk acquire(size_t id) {
    assert(id * kChunkSize < handle_.offset + handle_.size);
    auto ret = Chunk(id * kChunkSize, alloc_, file_ptr_);
    return ret;
  }

  void acquire(size_t id, Chunk& c) {
    assert(id * kChunkSize < handle_.offset + handle_.size);
    c.acquire(id * kChunkSize, alloc_, file_ptr_);
  }

  // assume that input is valid, read_key_offset and read_value_offset
  template <typename T>
  void read_key(uint32_t offset, const Chunk& c, T& result) const {
    assert(offset < kChunkSize);
    result.read(c.data(offset));
  }

  uint32_t read_key_size(uint32_t offset, const Chunk& c) const { return KV::read_size(c.data(offset)); }

  uint32_t read_value(uint32_t offset, const Chunk& c) const {
    assert(offset < kChunkSize);
    return *reinterpret_cast<uint32_t*>(c.data(offset));
  }

  bool is_empty_key(uint32_t offset, const Chunk& c) const { return *reinterpret_cast<uint32_t*>(c.data(offset)) == 0; }

  // it's only used for IndexKey, i.e. BlockKey<uint32_t>, so that the type of kv.value() is uint32_t.
  uint32_t upper_offset(const SKey& key) const {
    int l = 0, r = handle_.counts - 1;
    uint32_t ret = -1;
    SeekIterator it = SeekIterator(*this);
    KV _key;
    while (l <= r) {
      auto mid = (l + r) >> 1;
      it.seek_and_read(mid, _key);
      // compare two keys
      if (comp_(key, _key.key()) <= 0) {
        r = mid - 1, ret = _key.value();
      } else
        l = mid + 1;
    }
    return ret;
  }

  // it's only used for IndexKey, i.e. BlockKey<uint32_t>, so that the type of kv.value() is uint32_t.
  // Find the biggest _key that _key <= key.
  int lower_offset(const SKey& key) const {
    int l = 0, r = handle_.counts - 1;
    int ret = -1;
    SeekIterator it = SeekIterator(*this);
    KV _key;
    while (l <= r) {
      auto mid = (l + r) >> 1;
      it.seek_and_read(mid, _key);
      // compare two keys
      if (comp_(_key.key(), key) <= 0) {
        l = mid + 1, ret = _key.value();
      } else
        r = mid - 1;
    }
    return ret;
  }

  // it calculates the smallest No. of the key that >= input key.
  uint32_t upper_key(const SKey& key, uint32_t L, uint32_t R) const {
    int l = L, r = std::min(R, handle_.counts - 1);
    uint32_t ret = r + 1;
    SeekIterator it = SeekIterator(*this);
    KV _key;
    while (l <= r) {
      auto mid = (l + r) >> 1;
      it.seek_and_read(mid, _key);
      // compare two keys
      if (comp_(key, _key.key()) <= 0) {
        r = mid - 1, ret = mid;
      } else
        l = mid + 1;
    }
    return ret;
  }

  // it calculates the smallest No. of the key that > input key.
  uint32_t upper_key_not_eq(const SKey& key, uint32_t L, uint32_t R) const {
    int l = L, r = std::min(R, handle_.counts - 1);
    uint32_t ret = r + 1;
    SeekIterator it = SeekIterator(*this);
    KV _key;
    while (l <= r) {
      auto mid = (l + r) >> 1;
      it.seek_and_read(mid, _key);
      // compare two keys
      if (comp_(key, _key.key()) < 0) {
        r = mid - 1, ret = mid;
      } else
        l = mid + 1;
    }
    return ret;
  }

  // seek with the offset storing the offset of the key.
  EnumIterator seek_with_id(uint32_t id) const {
    SeekIterator it = SeekIterator(*this);
    return EnumIterator(*this, it.seek_offset(id), id);
  }

  size_t counts() const { return handle_.counts; }

  size_t size() const { return handle_.size; }
};

template <typename Value>
class BlockKey {
 private:
  SKey key_;
  Value v_;

 public:
  BlockKey() : key_(), v_() {}
  BlockKey(SKey key, Value v) : key_(key), v_(v) {}
  const uint8_t* read(const uint8_t* from) {
    from = key_.read(from);
    v_ = *reinterpret_cast<const Value*>(from);
    return from + sizeof(Value);
  }
  size_t size() const { return key_.size() + sizeof(v_); }
  uint8_t* write(uint8_t* to) const {
    to = key_.write(to);
    *reinterpret_cast<Value*>(to) = v_;
    return to + sizeof(Value);
  }
  SKey key() const { return key_; }
  Value value() const { return v_; }
  static size_t read_size(uint8_t* from) {
    size_t ret = SKey::read_size(from);
    return ret + sizeof(Value);
  }
};

using DataKey = BlockKey<SValue>;
using IndexKey = BlockKey<uint32_t>;

using RefDataKey = BlockKey<SValue*>;

int SKeyCompFunc(const SKey& A, const SKey& B) {
  if (A.len() != B.len()) return A.len() < B.len() ? -1 : 1;
  return memcmp(A.data(), B.data(), A.len());
}

using KeyCompType = int(const SKey&, const SKey&);

// one SST
template <typename KeyCompT>
class ImmutableFile {
  // The structure of the file:
  // [A data block]
  // [A index block]
  // [offset of index block, size, counts]
  // [offset of data block, size, counts]
  // [kMagicNumber] (8B)
  // Now we don't consider crash consistency
  // leveldb drops the shared prefix of a key which is not restart point, here we don't consider the compression now

  // we assume the length of key is < 4096.
  // from metadata file
  uint32_t file_id_;
  uint32_t size_;  // the size of the file
  // range is stored in memory
  std::pair<IndSKey, IndSKey> range_;
  // filename is not stored in metadata file, it's generated by file_id and the filename of the metadata file
  // std::string filename_;
  // pointer to the file
  // I don't consider the deletion of this pointer now
  // 1000 (limitation of file handles) files of 64MB, then 64GB of data, it may be enough?
  std::unique_ptr<RandomAccessFile> file_ptr_;
  FileBlock<IndexKey, KeyCompT> index_block_;
  FileBlock<DataKey, KeyCompT> data_block_;
  // LRUCache pointer reference to the one in VisCnts (is deleted)
  BaseAllocator* alloc_;
  KeyCompT comp_;

 public:
  ImmutableFile(uint32_t file_id, uint32_t size, std::unique_ptr<RandomAccessFile>&& file_ptr, BaseAllocator* alloc,
                const std::pair<IndSKey, IndSKey>& range, const KeyCompT& comp)
      : file_id_(file_id), size_(size), range_(range), file_ptr_(std::move(file_ptr)), alloc_(alloc), comp_(comp) {
    // read index block
    Slice result(nullptr, 0);
    FileBlockHandle index_bh, data_bh;
    size_t mgn;
    auto ret = file_ptr_->read(size_ - sizeof(size_t), sizeof(size_t), (uint8_t*)(&mgn), result);
    assert(ret >= 0);
    assert(mgn == kMagicNumber);
    ret = file_ptr_->read(size_ - sizeof(size_t) - sizeof(FileBlockHandle) * 2, sizeof(FileBlockHandle), (uint8_t*)(&index_bh), result);
    assert(ret >= 0);
    ret = file_ptr_->read(size_ - sizeof(size_t) - sizeof(FileBlockHandle), sizeof(FileBlockHandle), (uint8_t*)(&data_bh), result);
    assert(ret >= 0);
    index_block_ = FileBlock<IndexKey, KeyCompT>(file_id, index_bh, alloc, file_ptr_.get(), comp_);
    data_block_ = FileBlock<DataKey, KeyCompT>(file_id, data_bh, alloc, file_ptr_.get(), comp_);
  }

  // seek the first key that >= key, but only seek index block.
  typename FileBlock<DataKey, KeyCompT>::EnumIterator estimate_seek(const SKey& key) {
    if (comp_(range_.second.ref(), key) < 0) return {};
    auto id = index_block_.upper_offset(key);
    if (id == -1) return {};
    return data_block_.seek_with_id(id);
  }

  // seek the first key that >= key
  typename FileBlock<DataKey, KeyCompT>::EnumIterator seek(const SKey& key) const {
    if (comp_(range_.second.ref(), key) < 0) return {};
    auto id = index_block_.lower_offset(key);
    if (id == -1) return data_block_.seek_with_id(0);
    id = data_block_.upper_key(key, id, id + kIndexChunkSize - 1);
    return data_block_.seek_with_id(id);
  }

  std::pair<int, int> rank_pair(const std::pair<SKey, SKey>& range) const {
    int retl = 0, retr = 0;
    auto& [L, R] = range;
    if (comp_(range_.second.ref(), L) < 0)
      retl = data_block_.counts();
    else if (comp_(L, range_.first.ref()) < 0)
      retl = 0;
    else {
      auto id = index_block_.lower_offset(L);
      if (id == -1)
        retl = 0;
      else
        retl = data_block_.upper_key(L, id, id + kIndexChunkSize - 1);
    }
    if (comp_(range_.second.ref(), R) <= 0)
      retr = data_block_.counts();
    else if (comp_(R, range_.first.ref()) < 0)
      retr = 0;
    else {
      auto id = index_block_.lower_offset(R);
      if (id == -1)
        retr = 0;
      else
        retr = data_block_.upper_key_not_eq(R, id, id + kIndexChunkSize - 1);
    }

    return {retl, retr};
  }

  // calculate number of elements in [L, R]
  int range_count(const std::pair<SKey, SKey>& range) {
    auto [retl, retr] = rank_pair(range);
    return retr - retl;
  }

  bool in_range(const SKey& key) {
    auto& [l, r] = range_;
    return comp_(l.ref(), key) <= 0 && comp_(key, r.ref()) <= 0;
  }
  std::pair<SKey, SKey> range() const { return {range_.first.ref(), range_.second.ref()}; }

  size_t size() const { return size_; }

  size_t data_size() const { return data_block_.size(); }

  size_t counts() const { return data_block_.counts(); }

  bool range_overlap(const std::pair<SKey, SKey>& range) const {
    auto& [l, r] = range_;
    return comp_(l.ref(), range.second) <= 0 && comp_(range.first, r.ref()) <= 0;
  }

  FileBlock<DataKey, KeyCompT> data_block() const { return data_block_; }

  void remove() {
    auto ret = file_ptr_->remove();
    assert(ret >= 0);
  }
};

// This write data in a buffer, and flush the buffer when it's full. It is used in SSTBuilder.
class WriteBatch {
  // Semantically there can only be one AppendFile for each SST file.
  std::unique_ptr<AppendFile> file_ptr_;
  const size_t buffer_size_;
  size_t used_size_;
  uint8_t* data_;

 public:
  explicit WriteBatch(std::unique_ptr<AppendFile>&& file, size_t size) : file_ptr_(std::move(file)), buffer_size_(size), used_size_(0) {
    data_ = new uint8_t[size];
  }
  ~WriteBatch() {
    flush();
    delete[] data_;
  }

  template <typename T>
  void append(const T& kv) noexcept {
    size_t cp_size = std::min(kv.len(), buffer_size_ - used_size_);
    memcpy(data_ + used_size_, kv.data(), cp_size);
    used_size_ += cp_size;
    if (cp_size != kv.len()) {
      flush();
      append(Slice(kv.data() + cp_size, kv.len() - cp_size));
    }
  }

  template <typename T>
  void append_other(const T& x) {
    if (used_size_ + sizeof(T) > buffer_size_) {
      append(Slice(const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(&x)), sizeof(T)));
    } else {
      *reinterpret_cast<T*>(data_ + used_size_) = x;
      used_size_ += sizeof(T);
    }
  }

  template <typename T>
  void append_key(const T& x) {
    if (__builtin_expect(used_size_ + x.size() > buffer_size_, 0)) {
      flush();
      assert(x.size() <= buffer_size_);
      x.write(data_);
      used_size_ = x.size();
    } else {
      x.write(data_ + used_size_);
      used_size_ += x.size();
    }
  }

  // reserve one slice so that we don't need use IndSKey to store temporary key, when merging the same key
  // RefDataKey is BlockKey<SValue*>, which means we can modify the value, and the address of value can be nullptr
  RefDataKey reserve_kv(const DataKey& kv, size_t vlen) {
    assert(kv.size() <= buffer_size_ && vlen < kv.size());
    auto size = kv.size();
    if (__builtin_expect(used_size_ + size > buffer_size_, 0)) {
      flush();
      kv.write(data_);
      used_size_ = size;
    } else {
      kv.write(data_ + used_size_);
      used_size_ += size;
    }
    SKey ret_key;
    ret_key.read(data_ + used_size_ - size);
    return RefDataKey(ret_key, reinterpret_cast<SValue*>(data_ + used_size_ - vlen));
  }

  void fill(uint8_t what, size_t len) {
    size_t fill_size = std::min(len, buffer_size_ - used_size_);
    memset(data_ + used_size_, what, fill_size);
    used_size_ += fill_size;
    if (fill_size != len) {
      flush();
      fill(what, len - fill_size);
    }
  }

  void flush() {
    if (used_size_) {
      auto ret = file_ptr_->write(Slice(data_, used_size_));
      assert(ret == 0);
    }
    used_size_ = 0;
  }
};

class DeletedRange {
 public:
  struct Node {
    std::pair<int, int> ranks;
    Node(std::pair<int, int> _rank = std::pair<int, int>()) : ranks(_rank) {}
  };

  DeletedRange() : nodes_(CompareNode()) {}

  void insert(std::pair<int, int> rank) {
    if (nodes_.empty()) {
      nodes_.insert(Node(rank));
      return;
    }
    auto L = nodes_.lower_bound(std::make_pair(0, rank.first));
    auto R = nodes_.lower_bound(std::make_pair(0, rank.second));
    if (L == nodes_.end() || (rank.second < L->ranks.first)) {
      nodes_.insert(Node(rank));
    } else {
      if (L->ranks.first <= rank.first) {
        rank.first = L->ranks.first;
      }
      if (R != nodes_.end() && R->ranks.first <= rank.second) {
        rank.second = R->ranks.second;
        R = std::next(R);
      }
      nodes_.erase(L, R);
      nodes_.insert(Node(rank));
    }
  }

  int deleted_counts(std::pair<int, int> rank) {
    if (nodes_.empty()) {
      return rank.second - rank.first;
    }
    auto L = nodes_.lower_bound(std::make_pair(0, rank.first));
    auto R = nodes_.lower_bound(std::make_pair(0, rank.second));
    if (L == nodes_.end()) {
      return rank.second - rank.first;
    } else {
      int sum = rank.second - rank.first;
      if (L->ranks.first <= rank.first) {
        if (L == R) return 0;
        sum -= L->ranks.second - rank.first;
        L = std::next(L);
      }
      if (R != nodes_.end() && R->ranks.first <= rank.second) {
        sum -= rank.second - R->ranks.first;
      }
      for (auto p = L; p != R; p++) {
        sum -= p->ranks.second - p->ranks.first;
      }
      return sum;
    }
  }

  int sum() {
    int ret = 0;
    for (auto& a : nodes_) ret += a.ranks.second - a.ranks.first;
    return ret;
  }

 private:
  class CompareNode {
   public:
    int operator()(const Node& x, const Node& y) const { return x.ranks.second < y.ranks.second; }
  };

  std::set<Node, CompareNode> nodes_;

 public:
  class Iterator {
    typename std::set<Node, CompareNode>::const_iterator iter_;
    typename std::set<Node, CompareNode>::const_iterator iter_end_;

   public:
    Iterator() {}
    Iterator(int k, const DeletedRange& range) : iter_(range.nodes_.lower_bound(Node(std::make_pair(0, k)))), iter_end_(range.nodes_.end()) {}
    Iterator(const DeletedRange& range) : iter_(range.nodes_.begin()), iter_end_(range.nodes_.end()) {}
    bool valid() { return iter_ != iter_end_; }
    int jump(int k) {
      while (iter_->ranks.first <= k) {
        k = iter_->ranks.second;
        iter_ = std::next(iter_);
        if (iter_ == iter_end_) return k;
      }
      return k;
    }
  };
};

// A set of iterators, use heap to manage but not segment tree because it can avoid comparisions opportunistically
template <typename Iterator, typename KVComp>
class SeqIteratorSet {
  std::vector<Iterator> iters_;
  std::vector<Iterator*> seg_tree_;
  std::vector<SKey> keys_;
  std::vector<SValue> values_;
  uint32_t size_;
  KVComp comp_;

 public:
  SeqIteratorSet(const KVComp& comp) : size_(0), comp_(comp) {}
  SeqIteratorSet(const SeqIteratorSet& ss) { (*this) = ss; }
  SeqIteratorSet(SeqIteratorSet&& ss) { (*this) = std::move(ss); }
  SeqIteratorSet& operator=(const SeqIteratorSet& ss) {
    for (auto& a : ss.iters_) iters_.emplace_back(a);
    comp_ = ss.comp_;
    build();
    return (*this);
  }
  SeqIteratorSet& operator=(SeqIteratorSet&& ss) {
    iters_ = std::move(ss.iters_);
    seg_tree_ = std::move(ss.seg_tree_);
    keys_ = std::move(ss.keys_);
    values_ = std::move(ss.values_);
    size_ = ss.size_;
    comp_ = ss.comp_;
    return (*this);
  }
  void build() {
    size_ = iters_.size();
    DataKey kv;
    seg_tree_.resize(size_ + 1, nullptr);
    keys_.resize(size_);
    values_.resize(size_);
    for (uint32_t i = 1; i <= iters_.size(); ++i) {
      seg_tree_[i] = &iters_[i - 1];
      seg_tree_[i]->read(kv);
      keys_[i - 1] = std::move(kv.key());
      values_[i - 1] = kv.value();
      for (uint32_t j = i; j > 1 && _min(j, j >> 1) == j; j >>= 1) std::swap(seg_tree_[j], seg_tree_[j >> 1]);
    }
  }
  bool valid() { return size_ >= 1; }
  void next() {
    // logger_printf("[S.Set, next, %d]", size_);
    seg_tree_[1]->next();
    if (!seg_tree_[1]->valid()) {
      if (size_ == 1) {
        size_ = 0;
        return;
      }
      seg_tree_[1] = seg_tree_[size_];
      size_--;
    }

    DataKey kv;
    seg_tree_[1]->read(kv);
    uint32_t id = seg_tree_[1] - iters_.data();
    keys_[id] = std::move(kv.key());
    values_[id] = kv.value();

    uint32_t x = 1;
    while ((x << 1 | 1) <= size_) {
      auto r = _min(x << 1, x << 1 | 1);
      if (_min(r, x) == x) return;
      std::swap(seg_tree_[x], seg_tree_[r]);
      x = r;
    }
    if ((x << 1) <= size_) {
      if (_min(x, x << 1) == (x << 1)) std::swap(seg_tree_[x], seg_tree_[x << 1]);
    }
  }
  std::pair<SKey, SValue> read() {
    int x = seg_tree_[1] - iters_.data();
    return {keys_[x], values_[x]};
  }
  void push(Iterator&& new_iter) {
    if (!new_iter.valid()) return;
    iters_.push_back(std::move(new_iter));
  }

  KVComp comp_func() { return comp_; }

 private:
  uint32_t _min(uint32_t x, uint32_t y) {
    uint32_t idx = seg_tree_[x] - iters_.data();
    uint32_t idy = seg_tree_[y] - iters_.data();
    return comp_(keys_[idx], keys_[idy]) < 0 ? x : y;
  }
};

template <typename Iterator, typename KVComp>
class SeqIteratorSetForScan {
  IndSKey current_key_;
  SValue current_value_;
  SeqIteratorSet<Iterator, KVComp> iter_;
  bool valid_;

 public:
  SeqIteratorSetForScan(SeqIteratorSet<Iterator, KVComp>&& iter) : iter_(std::move(iter)), valid_(true) {}
  void build() {
    iter_.build();
    next();
  }
  std::pair<SKey, SValue> read() { return {current_key_.ref(), current_value_}; }
  void next() {
    if (!iter_.valid()) {
      valid_ = false;
      return;
    }
    auto result = iter_.read();
    current_key_ = result.first;
    current_value_ = result.second;
    iter_.next();
    while (iter_.valid()) {
      result = iter_.read();
      if (iter_.comp_func()(result.first, current_key_.ref()) == 0) {
        current_value_ += result.second;
      } else
        break;
      iter_.next();
    }
  }

  bool valid() { return valid_; }
};

template <typename KeyCompT>
class SSTIterator {
 public:
  SSTIterator() : file_(nullptr) {}
  SSTIterator(const ImmutableFile<KeyCompT>* file) : file_(file), file_block_iter_(file->data_block(), 0, 0) {}
  SSTIterator(const ImmutableFile<KeyCompT>* file, const SKey& key) : file_(file), file_block_iter_(file->seek(key)) {}
  SSTIterator(SSTIterator&& it) noexcept {
    file_ = it.file_;
    file_block_iter_ = std::move(it.file_block_iter_);
    it.file_ = nullptr;
  }
  SSTIterator& operator=(SSTIterator&& it) noexcept {
    file_ = it.file_;
    file_block_iter_ = std::move(it.file_block_iter_);
    it.file_ = nullptr;
    return *this;
  }
  SSTIterator(const SSTIterator& it) : file_(it.file_), file_block_iter_(it.file_block_iter_) {}
  SSTIterator& operator=(const SSTIterator& it) {
    file_ = it.file_;
    file_block_iter_ = it.file_block_iter_;
    return *this;
  }
  bool valid() { return file_block_iter_.valid(); }
  // ensure it's valid.
  void next() { file_block_iter_.next(); }
  // remember, SKey is a reference to file block
  void read(DataKey& kv) { file_block_iter_.read(kv); }

  int rank() { return file_block_iter_.rank(); }

  void jump(int new_id) { file_block_iter_.jump(new_id); }

 private:
  const ImmutableFile<KeyCompT>* file_;
  typename FileBlock<DataKey, KeyCompT>::EnumIterator file_block_iter_;
};

class SSTBuilder {
  std::unique_ptr<WriteBatch> file_;

  void _align() {
    if (now_offset % kChunkSize != 0) {
      file_->fill(0, kChunkSize - now_offset % kChunkSize);
      now_offset += kChunkSize - now_offset % kChunkSize;
    }
  }

  void _append_align(size_t len) {
    if (len && (now_offset + len - 1) / kChunkSize != now_offset / kChunkSize) _align();
  }

 public:
  SSTBuilder(std::unique_ptr<WriteBatch>&& file = nullptr) : file_(std::move(file)), now_offset(0), lst_offset(0), counts(0), size_(0) {}
  void append(const DataKey& kv) {
    assert(kv.key().len() > 0);
    _append_align(kv.size());
    if (offsets.size() % (kIndexChunkSize / sizeof(uint32_t)) == 0) {
      index.emplace_back(kv.key(), offsets.size());
      if (offsets.size() == 0) first_key = kv.key();
    }
    lst_key = kv.key();
    offsets.push_back(now_offset);
    now_offset += kv.size();
    file_->append_key(kv);
    size_ += kv.size();
  }
  template <typename Value>
  void append(const std::pair<SKey, Value>& kv) {
    append(DataKey(kv.first, kv.second));
  }
  template <typename Value>
  void append(const std::pair<IndSKey, Value>& kv) {
    append(DataKey(kv.first.ref(), kv.second));
  }

  RefDataKey reserve_kv(const DataKey& kv) {
    assert(kv.key().len() > 0);
    _append_align(kv.size());
    if (offsets.size() % (kIndexChunkSize / sizeof(uint32_t)) == 0) {
      index.emplace_back(kv.key(), offsets.size());
      if (offsets.size() == 0) first_key = kv.key();
    }
    lst_key = kv.key();
    offsets.push_back(now_offset);
    now_offset += kv.size();
    size_ += kv.size();
    return file_->reserve_kv(kv, sizeof(decltype(kv.value())));
  }

  void make_index() {
    // if (offsets.size() % (kChunkSize / sizeof(uint32_t)) != 1) {
    //   index.emplace_back(lst_key, offsets.size());  // append last key into index block.
    // }
    // logger("SSTB: ", now_offset, " num: ", offsets.size());
    _align();
    // file_->check(offsets.size() * 32);
    // logger("SSTB(AFTER ALIGN): ", now_offset);
    // auto data_index_offset = now_offset;
    // append all the offsets in the data block
    for (const auto& a : offsets) {
      if (kChunkSize % sizeof(decltype(a))) _append_align(sizeof(decltype(a)));
      file_->append_other(a);
      now_offset += sizeof(decltype(a));
    }
    // logger("SSTB: ", now_offset, " num: ", offsets.size());
    auto data_bh = FileBlockHandle(0, now_offset, offsets.size());
    _align();
    lst_offset = now_offset;
    std::vector<uint32_t> v;
    // append keys in the index block
    for (const auto& a : index) {
      IndexKey index_key(a.first.ref(), a.second);
      _append_align(index_key.size());
      file_->append_key(index_key);
      v.push_back(now_offset);
      now_offset += index_key.size();
    }
    _align();
    // append all the offsets in the index block
    for (const auto& a : v) {
      if (kChunkSize % sizeof(decltype(a))) _append_align(sizeof(decltype(a)));
      file_->append_other(a);
      now_offset += sizeof(decltype(a));
    }
    // append two block handles.
    // logger("[INDEX SIZE]: ", index.size());
    file_->append_other(FileBlockHandle(lst_offset, now_offset - lst_offset, index.size()));  // write offset of index block
    file_->append_other(data_bh);
    now_offset += sizeof(FileBlockHandle) * 2;
  }
  void finish() {
    file_->append_other(kMagicNumber);
    now_offset += sizeof(size_t);
    file_->flush();
  }
  void reset() {
    now_offset = 0;
    lst_offset = 0;
    counts = 0;
    size_ = 0;
    index.clear();
    offsets.clear();
  }
  void new_file(std::unique_ptr<WriteBatch>&& file) {
    file_ = std::move(file);
    assert(file_ != nullptr);
  }

  size_t size() { return now_offset; }
  size_t kv_size() { return size_; }

  std::pair<IndSKey, IndSKey> range() {
    if (index.size())
      return {first_key, lst_key};
    else
      return {};
  }

 private:
  uint32_t now_offset, lst_offset, counts, size_;
  std::vector<std::pair<IndSKey, uint32_t>> index;
  std::vector<uint32_t> offsets;
  IndSKey lst_key, first_key;
};

template <typename KeyCompT>
class Compaction {
  // builder_ is used to build one file
  // files_ is used to get global file name
  // env_ is used to get global environment
  // flag_ means whether lst_value_ is valid
  // lst_value_ is the last value appended to builder_
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
    builder_.new_file(std::make_unique<WriteBatch>(std::unique_ptr<AppendFile>(env_->openAppFile(filename)), kBatchSize));
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
  Compaction(FileName* files, Env* env, const KeyCompT& comp) : files_(files), env_(env), flag_(false), rndgen_(std::random_device()()), comp_(comp) {
    decay_prob_ = 0.5;
  }

  template <typename TIter>
  auto flush(TIter& left) {
    vec_newfiles_.clear();
    _begin_new_file();
    flag_ = false;
    real_size_ = 0;
    lst_real_size_ = 0;
    int CNT = 0;
    RefDataKey lst_kv;
    while (left.valid()) {
      CNT++;
      auto L = left.read();
      // logger("read key: ", L.first.len(), ", ", (int)L.first.data()[0], " ", (int)L.first.data()[1], " ", (int)L.first.data()[2], " ",
      //      (int)L.first.data()[3], "--", L.second.counts);
      if (flag_ && comp_(lst_kv.key(), L.first) == 0) {
        *lst_kv.value() += L.second;
      } else {
        if (flag_) {
          real_size_ += _calc_decay_value(std::make_pair(lst_kv.key(), *lst_kv.value()));
          _divide_file(DataKey(L.first, L.second).size());
        }
        // logger("flush key: ", L.first.len(), ", ", (int)L.first.data()[0], " ", (int)L.first.data()[1], " ", (int)L.first.data()[2], " ",
        //    (int)L.first.data()[3], "--", L.second.counts);
        lst_kv = builder_.reserve_kv(DataKey(L.first, L.second));
        flag_ = true;
      }
      left.next();
    }
    // logger("flush(): ", CNT);
    if (flag_) real_size_ += _calc_decay_value(std::make_pair(lst_kv.key(), *lst_kv.value()));
    _end_new_file();
    return std::make_pair(vec_newfiles_, real_size_);
  }

  template <typename TIter>
  auto decay_first(TIter& iters) {
    real_size_ = 0;  // re-calculate size
    lst_real_size_ = 0;
    vec_newfiles_.clear();
    _begin_new_file();
    flag_ = false;
    int A = 0;
    while (iters.valid()) {
      A++;
      auto L = iters.read();
      if (flag_ && comp_(lst_value_.first.ref(), L.first) == 0) {
        lst_value_.second += L.second;
      } else {
        if (flag_ && _decay_kv(lst_value_)) {
          real_size_ += _calc_decay_value(lst_value_);
          _divide_file(DataKey(lst_value_.first.ref(), lst_value_.second).size());
          builder_.append(lst_value_);
        }
        lst_value_ = L, flag_ = true;
      }
      iters.next();
    }
    if (flag_) {
      if (_decay_kv(lst_value_)) {
        A++;
        real_size_ += _calc_decay_value(lst_value_);
        builder_.append(lst_value_);
      }
    }
    logger("[decay_first]: ", A);
    _end_new_file();
    return std::make_pair(vec_newfiles_, real_size_);
  }

 private:
  template <typename T>
  double _calc_decay_value(const std::pair<T, SValue>& kv) {
    return (kv.first.len() + kv.second.vlen) * std::min(kv.second.counts * .5, 1.);
  }
  template <typename T>
  bool _decay_kv(std::pair<T, SValue>& kv) {
    kv.second.counts *= decay_prob_;
    if (kv.second.counts < 1) {
      std::uniform_real_distribution<> dis(0, 1.);
      if (dis(rndgen_) < kv.second.counts) {
        return false;
      }
      kv.second.counts = 1;
    }
    return true;
  }
  void _divide_file(size_t size) {
    if (builder_.kv_size() + size > kSSTable) {
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
      return (rank_pair.second - rank_pair.first) / (double)file_.counts() * range_count(range) / (double)global_range_counts_;
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
    LevelIterator seek(const SKey& key, const KeyCompT& comp) const {
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
    bool overlap(const SKey& lkey, const SKey& rkey, const KeyCompT& comp) const {
      if (!head_.size()) return false;
      int l = 0, r = head_.size() - 1, where = -1;
      while (l <= r) {
        int mid = (l + r) >> 1;
        if (comp(lkey, head_[mid]->range().second) <= 0)
          where = mid, r = mid - 1;
        else
          l = mid + 1;
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

    size_t range_count(const std::pair<SKey, SKey>& range, const KeyCompT& comp) {
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

    size_t range_data_size(const std::pair<SKey, SKey>& range, const KeyCompT& comp) {
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

    void delete_range(const std::pair<SKey, SKey>& range, const KeyCompT& comp) {
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
    std::pair<int, int> _get_range_in_head(const std::pair<SKey, SKey>& range, const KeyCompT& comp) {
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
    double decay_limit_;

    std::mutex* del_mutex_;

    bool _decay_possible() const { return decay_size_overestimate_ >= decay_limit_; }

    using LevelIteratorSetT = SeqIteratorSet<typename Level::LevelIterator, KeyCompT>;

   public:
    SuperVersion(double decay_limit, std::mutex* mutex)
        : largest_(std::make_shared<Level>()), ref_(1), decay_size_overestimate_(0), decay_limit_(decay_limit), del_mutex_(mutex) {}
    void ref() {
      ref_++;
      // logger("ref count becomes (", this, "): ", ref_);
    }
    void unref() {
      // logger("ref count (", this, "): ", ref_);
      if (!--ref_) {
        // std::unique_lock lck(*del_mutex_);
        delete this;
      }
    }

    LevelIteratorSetT seek(const SKey& key, const KeyCompT& comp) const {
      LevelIteratorSetT ret(comp);
      if (largest_.get() != nullptr) ret.push(largest_->seek(key, comp));
      for (auto& a : tree_) ret.push(a->seek(key, comp));
      return ret;
    }

    LevelIteratorSetT seek_to_first(const KeyCompT& comp) const {
      LevelIteratorSetT ret(comp);
      if (largest_.get() != nullptr) ret.push(largest_->seek_to_first());
      for (auto& a : tree_) ret.push(a->seek_to_first());
      return ret;
    }

    std::vector<std::shared_ptr<Level>> flush_bufs(const std::vector<UnsortedBuffer*>& bufs, FileName* filename, Env* env, BaseAllocator* file_alloc,
                                                   const KeyCompT& comp) {
      if (bufs.size() == 0) return {};
      std::vector<std::shared_ptr<Level>> ret_vectors;
      auto flush_func = [](FileName* filename, Env* env, const KeyCompT& comp, UnsortedBuffer* buf, std::mutex& mu, BaseAllocator* file_alloc,
                           std::vector<std::shared_ptr<Level>>& ret_vectors) {
        Compaction worker(filename, env, comp);
        while (!buf->safe())
          ;
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
      auto ret = new SuperVersion(decay_limit_, del_mutex_);
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
    SuperVersion* compact(FileName* filename, Env* env, BaseAllocator* file_alloc, const KeyCompT& comp, bool trigger_decay = false) const {
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
      // If the unmerged data is too big, and estimated decay size is large enough.
      // all the partitions will be merged into the largest level.
      if ((Lsize >= kUnmergedRatio * largest_->size() && _decay_possible()) || trigger_decay) {
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
        auto [files, decay_size] = worker.decay_first(*iters);
        auto ret = new SuperVersion(decay_limit_, del_mutex_);
        // major compaction, so levels except largest_ become empty.

        // std::unique_lock lck(*del_mutex_);
        for (auto& a : files) {
          auto par = std::make_shared<Partition>(env, file_alloc, comp, a);
          ret->largest_->append_par(std::move(par));
        }

        // calculate new current decay size
        ret->decay_size_overestimate_ = ret->_calc_current_decay_size();
        logger("[new decay size]: ", ret->decay_size_overestimate_);
        logger("[new largest_]: ", ret->largest_->size());
        ret->_sort_levels();

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
          auto ret = new SuperVersion(decay_limit_, del_mutex_);
          // minor compaction?
          // decay_size_overestimate = largest_ + remaining levels + new level
          ret->tree_ = tree_;
          ret->largest_ = largest_;
          for (int i = 0; i < where; i++) ret->decay_size_overestimate_ += tree_[i]->decay_size();

          // logger("compact owari");

          auto rest_iter = rest.begin();
          auto level = std::make_shared<Level>();

          // std::unique_lock lck(*del_mutex_);
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
        return nullptr;
      }
    }

    size_t range_count(const std::pair<SKey, SKey>& range, const KeyCompT& comp) {
      size_t ans = 0;
      if (largest_) ans += largest_->range_count(range, comp);
      for (auto& a : tree_) ans += a->range_count(range, comp);
      return ans;
    }

    size_t range_data_size(const std::pair<SKey, SKey>& range, const KeyCompT& comp) {
      size_t ans = 0;
      if (largest_) ans += largest_->range_data_size(range, comp);
      for (auto& a : tree_) ans += a->range_data_size(range, comp);
      return ans;
    }

    SuperVersion* delete_range(std::pair<SKey, SKey> range, const KeyCompT& comp) {
      auto ret = new SuperVersion(decay_limit_, del_mutex_);
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
  double delta_;
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
  std::mutex sv_modify_mutex_;
  boost::fibers::buffered_channel<std::tuple<>>* notify_weight_change_;
  std::vector<std::shared_ptr<Level>> flush_buf_vec_;
  std::mutex flush_buf_vec_mutex_;
  std::condition_variable signal_flush_to_compact_;

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
  EstimateLSM(double delta, std::unique_ptr<Env>&& env, std::unique_ptr<FileName>&& filename, std::unique_ptr<BaseAllocator>&& file_alloc,
              const KeyCompT& comp, boost::fibers::buffered_channel<std::tuple<>>* notify_weight_change)
      : delta_(delta),
        env_(std::move(env)),
        filename_(std::move(filename)),
        file_alloc_(std::move(file_alloc)),
        comp_(comp),
        sv_(new SuperVersion(delta, &sv_mutex_)),
        terminate_signal_(0),
        bufs_(kUnsortedBufferSize, kUnsortedBufferMaxQueue),
        notify_weight_change_(notify_weight_change) {
    compact_thread_ = std::thread([this]() { compact_thread(); });
    flush_thread_ = std::thread([this]() { flush_thread(); });
  }
  ~EstimateLSM() {
    terminate_signal_ = 1;
    bufs_.notify_cv();
    signal_flush_to_compact_.notify_one();
    compact_thread_.join();
    flush_thread_.join();
    std::unique_lock lck(sv_load_mutex_);
    sv_->unref();
  }
  void append(const SKey& key, const SValue& value) { bufs_.append_and_notify(key, value); }
  auto seek(const SKey& key) {
    auto sv = get_current_sv();
    return new SuperVersionIterator(sv->seek(key, comp_), sv);
  }

  auto seek_to_first() {
    auto sv = get_current_sv();
    return new SuperVersionIterator(sv->seek_to_first(comp_), sv);
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
    while (bufs_.size())
      ;
  }

  void delete_range(std::pair<SKey, SKey> range) {
    if (comp_(range.first, range.second) > 0) return;
    std::unique_lock del_range_lck(sv_modify_mutex_);
    auto new_sv = sv_->delete_range(range, comp_);
    _update_superversion(new_sv);
    _update_channel();
  }

  void trigger_decay() {
    std::unique_lock del_range_lck(sv_modify_mutex_);
    auto new_sv = sv_->compact(filename_.get(), env_.get(), file_alloc_.get(), comp_, true);
    _update_superversion(new_sv);
    _update_channel();
  }

  double get_current_decay_size() {
    auto sv = get_current_sv();
    auto ret = sv->get_current_decay_size();
    sv->unref();
    return ret;
  }

 private:
  void flush_thread() {
    while (!terminate_signal_) {
      auto buf_q_ = terminate_signal_ ? bufs_.get() : bufs_.wait_and_get();
      if (buf_q_.empty() && terminate_signal_) return;
      if (buf_q_.empty()) continue;
      auto new_vec = sv_->flush_bufs(buf_q_, filename_.get(), env_.get(), file_alloc_.get(), comp_);
      while (new_vec.size() + flush_buf_vec_.size() > kMaxFlushBufferQueueSize) {
        logger("full");
        std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(kWaitCompactionSleepMilliSeconds));
        if (terminate_signal_) return;
      }
      std::unique_lock lck(flush_buf_vec_mutex_);
      flush_buf_vec_.insert(flush_buf_vec_.end(), new_vec.begin(), new_vec.end());
      signal_flush_to_compact_.notify_one();
    }
  }
  void compact_thread() {
    while (!terminate_signal_) {
      SuperVersion* new_sv;
      {
        std::unique_lock flush_lck(flush_buf_vec_mutex_);
        // wait buffers from flush_thread()
        while (flush_buf_vec_.empty() && !terminate_signal_) {
          signal_flush_to_compact_.wait(flush_lck);
        }
        if (terminate_signal_) return;
        sv_modify_mutex_.lock();
        new_sv = sv_->push_new_buffers(flush_buf_vec_);
        flush_buf_vec_.clear();
      }

      auto last_compacted_sv = new_sv;
      while (true) {
        auto new_compacted_sv = last_compacted_sv->compact(filename_.get(), env_.get(), file_alloc_.get(), comp_);
        if (new_compacted_sv == nullptr)
          break;
        else {
          last_compacted_sv->unref();
          last_compacted_sv = new_compacted_sv;
        }
      }
      _update_superversion(last_compacted_sv);
      sv_modify_mutex_.unlock();
      _update_channel();
    }
  }

  void _update_superversion(SuperVersion* new_sv) {
    if (new_sv != nullptr) {
      sv_load_mutex_.lock();
      auto old_sv = sv_;
      sv_ = new_sv;
      sv_load_mutex_.unlock();
      old_sv->unref();
    }
  }

  void _update_channel() {
    if (notify_weight_change_ != nullptr) {
      // It seems that it doesn't block for a long time because it blocks only when there is a writer.
      // Because it returns full when the channel is full.
      notify_weight_change_->try_push(std::tuple<>());
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
  EstimateLSM<KeyCompT> tree;

 public:
  VisCnts(const KeyCompT& comp, const std::string& path, double delta, [[maybe_unused]] bool createIfMissing,
          boost::fibers::buffered_channel<std::tuple<>>* notify_weight_change)
      : tree(delta, std::unique_ptr<Env>(createDefaultEnv()), std::make_unique<FileName>(0, path), std::make_unique<DefaultAllocator>(), comp,
             notify_weight_change) {}
  void access(const std::pair<SKey, SValue>& kv) { tree.append(kv.first, kv.second); }
  auto trigger_decay() { return tree.trigger_decay(); }
  auto delete_range(const std::pair<SKey, SKey>& range) { return tree.delete_range(range); }
  auto seek_to_first() { return tree.seek_to_first(); }
  auto seek(const SKey& key) { return tree.seek(key); }
  auto weight_sum() { return tree.get_current_decay_size(); }
  auto range_data_size(const std::pair<SKey, SKey>& range) { return tree.range_data_size(range); }
};

}  // namespace viscnts_lsm

#include "rocksdb/compaction_router.h"
#include "rocksdb/comparator.h"

struct SKeyComparatorFromRocksDB {
  const rocksdb::Comparator* ucmp;
  SKeyComparatorFromRocksDB() : ucmp(nullptr) {}
  SKeyComparatorFromRocksDB(const rocksdb::Comparator* _ucmp) : ucmp(_ucmp) {}
  int operator()(const viscnts_lsm::SKey& x, const viscnts_lsm::SKey& y) const {
    if (x.data() == nullptr || y.data() == nullptr) return x.len() - y.len();
    rocksdb::Slice rx(reinterpret_cast<const char*>(x.data()), x.len());
    rocksdb::Slice ry(reinterpret_cast<const char*>(y.data()), y.len());
    return ucmp->Compare(rx, ry);
  }
};

using VisCntsType = viscnts_lsm::VisCnts<SKeyComparatorFromRocksDB>;

struct HotRecInfoAndIter {
  viscnts_lsm::EstimateLSM<SKeyComparatorFromRocksDB>::SuperVersionIterator* iter;
  VisCntsType* vc;
  rocksdb::HotRecInfo result;
};

VisCnts::VisCnts(const rocksdb::Comparator* ucmp, const char* path, bool createIfMissing,
                 boost::fibers::buffered_channel<std::tuple<>>* notify_weight_change)
    : notify_weight_change_(notify_weight_change), weight_sum_(0) {
  vc_ = new VisCntsType(SKeyComparatorFromRocksDB(ucmp), path, 1e100, createIfMissing, notify_weight_change);
}

void VisCnts::Access(const rocksdb::Slice& key, size_t vlen, double weight) {
  auto vc = reinterpret_cast<VisCntsType*>(vc_);
  vc->access({viscnts_lsm::SKey(reinterpret_cast<const uint8_t*>(key.data()), key.size()), viscnts_lsm::SValue(weight, vlen)});
}

double VisCnts::WeightSum() {
  auto vc = reinterpret_cast<VisCntsType*>(vc_);
  return vc->weight_sum();
}

void VisCnts::Decay() {
  auto vc = reinterpret_cast<VisCntsType*>(vc_);
  vc->trigger_decay();
}

void VisCnts::RangeDel(const rocksdb::Slice& L, const rocksdb::Slice& R) {
  auto vc = reinterpret_cast<VisCntsType*>(vc_);
  vc->delete_range({viscnts_lsm::SKey(reinterpret_cast<const uint8_t*>(L.data()), L.size()),
                    viscnts_lsm::SKey(reinterpret_cast<const uint8_t*>(R.data()), R.size())});
}

size_t VisCnts::RangeHotSize(const rocksdb::Slice& L, const rocksdb::Slice& R) {
  auto vc = reinterpret_cast<VisCntsType*>(vc_);
  return vc->range_data_size({viscnts_lsm::SKey(reinterpret_cast<const uint8_t*>(L.data()), L.size()),
                              viscnts_lsm::SKey(reinterpret_cast<const uint8_t*>(R.data()), R.size())});
}

void VisCnts::add_weight(double delta) { std::abort(); }

VisCnts::~VisCnts() {
  auto vc = reinterpret_cast<VisCntsType*>(vc_);
  delete vc;
}

VisCnts::Iter::Iter(VisCnts* ac) {
  viscnts_lsm::logger("[new_iter]");
  auto vc = reinterpret_cast<VisCntsType*>(ac->vc_);
  auto ret = new HotRecInfoAndIter();
  ret->vc = vc;
  ret->iter = nullptr;
  iter_ = ret;
}

const rocksdb::HotRecInfo* VisCnts::Iter::SeekToFirst() {
  viscnts_lsm::logger("[seek to first]");
  auto ac = reinterpret_cast<HotRecInfoAndIter*>(iter_);
  if (!ac->iter) delete ac->iter;
  ac->iter = ac->vc->seek_to_first();
  if (!ac->iter->valid()) return nullptr;
  auto [key, value] = ac->iter->read();
  ac->result =
      rocksdb::HotRecInfo{.slice = rocksdb::Slice(reinterpret_cast<const char*>(key.data()), key.len()), .count = value.counts, .vlen = value.vlen};
  return &ac->result;
}

const rocksdb::HotRecInfo* VisCnts::Iter::Seek(const rocksdb::Slice& key) {
  viscnts_lsm::logger("[seek]");
  auto ac = reinterpret_cast<HotRecInfoAndIter*>(iter_);
  if (!ac->iter) delete ac->iter;
  ac->iter = ac->vc->seek(viscnts_lsm::SKey(reinterpret_cast<const uint8_t*>(key.data()), key.size()));
  if (!ac->iter->valid()) return nullptr;
  auto [rkey, rvalue] = ac->iter->read();
  ac->result = rocksdb::HotRecInfo{
      .slice = rocksdb::Slice(reinterpret_cast<const char*>(rkey.data()), rkey.len()), .count = rvalue.counts, .vlen = rvalue.vlen};
  return &ac->result;
}

const rocksdb::HotRecInfo* VisCnts::Iter::Next() {
  auto ac = reinterpret_cast<HotRecInfoAndIter*>(iter_);
  ac->iter->next();
  if (!ac->iter->valid()) return nullptr;
  auto [rkey, rvalue] = ac->iter->read();
  ac->result = rocksdb::HotRecInfo{
      .slice = rocksdb::Slice(reinterpret_cast<const char*>(rkey.data()), rkey.len()), .count = rvalue.counts, .vlen = rvalue.vlen};
  return &ac->result;
}

VisCnts::Iter::~Iter() {
  viscnts_lsm::logger("[delete iter]");
  auto ac = reinterpret_cast<HotRecInfoAndIter*>(iter_);
  if (ac->iter) delete ac->iter;
  delete ac;
}

/// testing function.

void test_files() {
  using namespace viscnts_lsm;
  int L = 1e6, FS = 20;
  uint8_t a[12];
  memset(a, 0, sizeof(a));

  std::unique_ptr<Env> env_(createDefaultEnv());
  std::vector<SSTBuilder> builders(FS);
  for (int i = 0; i < FS; i++)
    builders[i].new_file(
        std::make_unique<WriteBatch>(std::unique_ptr<AppendFile>(env_->openAppFile("/tmp/viscnts/test" + std::to_string(i))), kBatchSize));

  for (int i = 0; i < L; i++) {
    for (int j = 0; j < 12; j++) a[j] = i >> (j % 4) * 8 & 255;
    builders[abs(rand()) % FS].append({SKey(a, 12), SValue()});
  }
  int sum = 0;
  for (int i = 0; i < FS; ++i) sum += builders[i].size();
  logger_printf("[TEST] SIZE: [%d]\n", sum);
  for (int i = 0; i < FS; i++) {
    builders[i].make_index();
    builders[i].finish();
  }

  auto comp = +[](const SKey& a, const SKey& b) {
    auto ap = a.data(), bp = b.data();
    uint32_t x = ap[0] | ((uint32_t)ap[1] << 8) | ((uint32_t)ap[2] << 16) | ((uint32_t)ap[3] << 24);
    uint32_t y = bp[0] | ((uint32_t)bp[1] << 8) | ((uint32_t)bp[2] << 16) | ((uint32_t)bp[3] << 24);
    return (int)x - (int)y;
  };

  std::vector<ImmutableFile<KeyCompType*>> files;
  for (int i = 0; i < FS; ++i)
    files.push_back(ImmutableFile<KeyCompType*>(0, builders[i].size(),
                                                std::unique_ptr<RandomAccessFile>(env_->openRAFile("/tmp/viscnts/test" + std::to_string(i))),
                                                new DefaultAllocator(), {}, comp));
  auto iters = std::make_unique<SeqIteratorSet<SSTIterator<KeyCompType*>, KeyCompType*>>(comp);
  for (int i = 0; i < FS; ++i) {
    SSTIterator iter(&files[i]);
    iters->push(std::move(iter));
  }
  iters->build();
  for (int i = 0; i < L; i++) {
    assert(iters->valid());
    auto kv = iters->read();
    int x = 0;
    auto a = kv.first.data();
    for (int j = 0; j < 12; j++) {
      x |= a[j] << (j % 4) * 8;
      assert(j % 4 != 3 || x == i);
      if (j % 4 == 3) x = 0;
    }
    iters->next();
  }
  assert(!iters->valid());
  logger("test_file(): OK");
}

void test_unordered_buf() {
  using namespace viscnts_lsm;
  UnsortedBufferPtrs bufs(kUnsortedBufferSize, 100);
  int L = 1e7, TH = 10;
  std::atomic<int> signal = 0;
  std::vector<std::thread> v;
  std::vector<std::pair<IndSKey, SValue>> result;

  auto comp = +[](const SKey& a, const SKey& b) {
    auto ap = a.data(), bp = b.data();
    uint32_t x = ap[0] | ((uint32_t)ap[1] << 8) | ((uint32_t)ap[2] << 16) | ((uint32_t)ap[3] << 24);
    uint32_t y = bp[0] | ((uint32_t)bp[1] << 8) | ((uint32_t)bp[2] << 16) | ((uint32_t)bp[3] << 24);
    return (int)x - (int)y;
  };
  auto start = std::chrono::system_clock::now();
  auto th = std::thread(
      [comp](std::atomic<int>& signal, UnsortedBufferPtrs& bufs, std::vector<std::pair<IndSKey, SValue>>& result) {
        while (true) {
          auto buf_q_ = signal.load() ? bufs.get() : bufs.wait_and_get();
          using namespace std::chrono;

          if (!buf_q_.size()) {
            if (signal) break;
            continue;
          }
          for (auto& buf : buf_q_) {
            while (!buf->safe())
              ;
            buf->sort(comp);
            UnsortedBuffer::Iterator iter(*buf);
            while (iter.valid()) {
              result.emplace_back(iter.read());
              iter.next();
            }
            buf->clear();
          }
        }
      },
      std::ref(signal), std::ref(bufs), std::ref(result));
  for (int i = 0; i < TH; i++) {
    v.emplace_back(
        [i, L, TH](UnsortedBufferPtrs& bufs) {
          int l = L / TH * i, r = L / TH * (i + 1);
          std::vector<int> v(r - l);
          for (int i = l; i < r; i++) v[i - l] = i;
          std::shuffle(v.begin(), v.end(), std::mt19937(std::random_device()()));
          uint8_t a[12];
          memset(a, 0, sizeof(a));
          for (auto& i : v) {
            for (int j = 0; j < 12; j++) a[j] = i >> (j % 4) * 8 & 255;
            bufs.append_and_notify(SKey(a, 12), SValue());
          }
        },
        std::ref(bufs));
  }
  for (auto& a : v) a.join();
  logger("OK!");
  bufs.flush();
  signal = 1;
  bufs.notify_cv();
  th.join();

  auto end = std::chrono::system_clock::now();
  auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << double(dur.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den << std::endl;

  logger_printf("RESULT_SIZE: %d\n", result.size());
  assert(result.size() == (uint32_t)L);
  std::set<int> st;
  for (uint32_t i = 0; i < result.size(); ++i) {
    int x = 0;
    auto a = result[i].first.data();
    for (int j = 0; j < 12; j++) {
      x |= a[j] << (j % 4) * 8;
      if (j % 4 == 3) {
        assert(x >= 0 && x < L);
        assert(!st.count(x));
        st.insert(x);
        break;
      }
    }
  }
  logger("test_unordered_buf(): OK");
}

void test_lsm_store() {
  using namespace viscnts_lsm;

  auto start = std::chrono::system_clock::now();
  {
    EstimateLSM<KeyCompType*> tree(1e9, std::unique_ptr<Env>(createDefaultEnv()), std::make_unique<FileName>(0, "/tmp/viscnts/"),
                                   std::make_unique<DefaultAllocator>(), SKeyCompFunc, nullptr);
    int L = 1e7;
    uint8_t a[12];
    memset(a, 0, sizeof(a));
    for (int i = 0; i < L; i++) {
      for (int j = 0; j < 12; j++) a[j] = i >> (j % 4) * 8 & 255;
      tree.append(SKey(a, 12), SValue());
    }
  }

  auto end = std::chrono::system_clock::now();
  auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << double(dur.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den << std::endl;
  logger("test_lsm_store(): OK");
}

void test_lsm_store_and_scan() {
  using namespace viscnts_lsm;

  auto comp = +[](const SKey& a, const SKey& b) {
    auto ap = a.data(), bp = b.data();
    uint32_t x = ap[0] | ((uint32_t)ap[1] << 8) | ((uint32_t)ap[2] << 16) | ((uint32_t)ap[3] << 24);
    uint32_t y = bp[0] | ((uint32_t)bp[1] << 8) | ((uint32_t)bp[2] << 16) | ((uint32_t)bp[3] << 24);
    return (int)x - (int)y;
  };
  auto start = std::chrono::system_clock::now();
  {
    EstimateLSM tree(1e10, std::unique_ptr<Env>(createDefaultEnv()), std::make_unique<FileName>(0, "/tmp/viscnts/"),
                     std::make_unique<DefaultAllocator>(), comp, nullptr);
    int L = 3e8;
    std::vector<int> numbers(L);
    for (int i = 0; i < L; i++) numbers[i] = i;
    // std::shuffle(numbers.begin(), numbers.end(), std::mt19937(std::random_device()()));
    start = std::chrono::system_clock::now();
    std::vector<std::thread> threads;
    int TH = 1;
    for (int i = 0; i < TH; i++) {
      threads.emplace_back(
          [i, L, TH](std::vector<int>& numbers, EstimateLSM<KeyCompType*>& tree) {
            uint8_t a[16];
            int l = (L / TH + 1) * i, r = std::min((L / TH + 1) * (i + 1), (int)numbers.size());
            for (int i = l; i < r; ++i) {
              for (int j = 0; j < 16; j++) a[j] = numbers[i] >> (j % 4) * 8 & 255;
              tree.append(SKey(a, 16), SValue(1, 1));
            }
          },
          std::ref(numbers), std::ref(tree));
    }
    for (auto& a : threads) a.join();
    threads.clear();
    // std::shuffle(numbers.begin(), numbers.end(), std::mt19937(std::random_device()()));

    auto _numbers = numbers;
    // std::shuffle(_numbers.begin(), _numbers.end(), std::mt19937(std::random_device()()));

    for (int i = 0; i < TH; i++) {
      threads.emplace_back(
          [i, L, TH](std::vector<int>& numbers, EstimateLSM<KeyCompType*>& tree) {
            uint8_t a[16];
            int l = (L / TH + 1) * i, r = std::min((L / TH + 1) * (i + 1), (int)numbers.size());
            for (int i = l; i < r; ++i) {
              for (int j = 0; j < 16; j++) a[j] = numbers[i] >> (j % 4) * 8 & 255;
              tree.append(SKey(a, 16), SValue(numbers[i], 1));
            }
          },
          std::ref(_numbers), std::ref(tree));
    }
    for (auto& a : threads) a.join();
    tree.all_flush();
    auto end = std::chrono::system_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << double(dur.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(5000));
    start = std::chrono::system_clock::now();
    auto iter = std::unique_ptr<EstimateLSM<KeyCompType*>::SuperVersionIterator>(tree.seek_to_first());
    for (int i = 0; i < L; i++) {
      assert(iter->valid());
      auto kv = iter->read();
      int x = 0, y = 0;
      auto a = kv.first.data();
      for (int j = 0; j < 16; j++) {
        x |= a[j] << (j % 4) * 8;
        assert(j % 4 != 3 || x == i);
        if (j % 4 == 3) {
          if (j > 3) {
            assert(y == x);
          }
          y = x;
          x = 0;
        }
      }
      assert(kv.second.counts == i + 1);
      assert(kv.second.vlen == 1);
      iter->next();
    }
  }

  auto end = std::chrono::system_clock::now();
  auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << double(dur.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den << std::endl;
  logger("test_lsm_store_and_scan(): OK");
}

void test_random_scan_and_count() {
  using namespace viscnts_lsm;

  auto start = std::chrono::system_clock::now();
  {
    EstimateLSM<KeyCompType*> tree(1e10, std::unique_ptr<Env>(createDefaultEnv()), std::make_unique<FileName>(0, "/tmp/viscnts/"),
                                   std::make_unique<DefaultAllocator>(), SKeyCompFunc, nullptr);
    int L = 3e8, Q = 1e4;
    std::vector<int> numbers(L);
    auto comp2 = +[](int x, int y) {
      uint8_t a[12], b[12];
      for (int j = 0; j < 12; j++) a[j] = x >> (j % 4) * 8 & 255;
      for (int j = 0; j < 12; j++) b[j] = y >> (j % 4) * 8 & 255;
      return SKeyCompFunc(SKey(a, 12), SKey(b, 12)) < 0;
    };
    for (int i = 0; i < L; i++) numbers[i] = i;
    // std::sort(numbers.begin(), numbers.end(), comp2);
    std::shuffle(numbers.begin(), numbers.end(), std::mt19937(std::random_device()()));
    srand(std::random_device()());
    for (int i = 0; i < L / 2; i++) {
      uint8_t a[12];
      for (int j = 0; j < 12; j++) a[j] = numbers[i] >> (j % 4) * 8 & 255;
      tree.append(SKey(a, 12), SValue(1, 1));
    }
    tree.all_flush();

    auto end = std::chrono::system_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    logger("flush used time: ", double(dur.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den);
    std::this_thread::sleep_for(std::chrono::milliseconds(5000));
    auto numbers2 = std::vector<int>(numbers.begin(), numbers.begin() + L / 2);
    std::sort(numbers2.begin(), numbers2.end(), comp2);

    start = std::chrono::system_clock::now();
    for (int i = 0; i < Q; i++) {
      uint8_t a[12], b[12];
      int id = abs(rand()) % numbers.size();
      int x = numbers[id];
      for (int j = 0; j < 12; j++) a[j] = numbers[id] >> (j % 4) * 8 & 255;
      id = abs(rand()) % numbers.size();
      int y = numbers[id];
      for (int j = 0; j < 12; j++) b[j] = numbers[id] >> (j % 4) * 8 & 255;
      int ans = std::upper_bound(numbers2.begin(), numbers2.end(), x, comp2) - std::lower_bound(numbers2.begin(), numbers2.end(), y, comp2);
      ans = std::max(ans, 0);
      int output = tree.range_count({SKey(b, 12), SKey(a, 12)});
      assert(ans == output);
    }

    int QLEN = 1000;

    end = std::chrono::system_clock::now();
    dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    start = end;
    logger("range count used time: ", double(dur.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den);

    for (int i = 0; i < Q; i++) {
      uint8_t a[12];
      int id = abs(rand()) % numbers.size();
      for (int j = 0; j < 12; j++) a[j] = numbers[id] >> (j % 4) * 8 & 255;

      auto iter = std::unique_ptr<EstimateLSM<KeyCompType*>::SuperVersionIterator>(tree.seek(SKey(a, 12)));
      auto check_func = [](const uint8_t* a, int goal) {
        int x = 0, y = 0;
        for (int j = 0; j < 12; j++) {
          x |= a[j] << (j % 4) * 8;
          assert(j % 4 != 3 || x == goal);
          if (j % 4 == 3) {
            if (j > 3) {
              assert(y == x);
            }
            y = x;
            x = 0;
          }
        }
      };
      if (id < L / 2) {
        assert(iter->valid());
        auto kv = iter->read();
        auto a = kv.first.data();
        check_func(a, numbers[id]);
        iter->next();
      } else {
        // assert(iter->valid() || i == mx);
        // assert(!iter->valid() || i != mx);
      }
      int x = a[0] | ((uint32_t)a[1] << 8) | ((uint32_t)a[2] << 16) | ((uint32_t)a[3] << 24);

      int cnt = 0;
      auto it = std::upper_bound(numbers2.begin(), numbers2.end(), x, comp2);
      while (true) {
        if (++cnt > QLEN) break;
        assert((it != numbers2.end()) == (iter->valid()));
        if (it == numbers2.end()) break;
        auto a = iter->read().first.data();
        check_func(a, *it);
        it++;
        iter->next();
      }
    }
    end = std::chrono::system_clock::now();
    dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    logger("random scan used time: ", double(dur.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den);
  }
  logger("test_random_scan(): OK");
}

void test_lsm_decay() {
  using namespace viscnts_lsm;

  auto start = std::chrono::system_clock::now();
  {
    EstimateLSM<KeyCompType*> tree(1e7, std::unique_ptr<Env>(createDefaultEnv()), std::make_unique<FileName>(0, "/tmp/viscnts/"),
                                   std::make_unique<DefaultAllocator>(), SKeyCompFunc, nullptr);
    int L = 3e7;
    std::vector<int> numbers(L);
    // auto comp2 = +[](int x, int y) {
    //   uint8_t a[12], b[12];
    //   for (int j = 0; j < 12; j++) a[j] = x >> (j % 4) * 8 & 255;
    //   for (int j = 0; j < 12; j++) b[j] = y >> (j % 4) * 8 & 255;
    //   return SKeyCompFunc(SKey(a, 12), SKey(b, 12)) < 0;
    // };
    for (int i = 0; i < L; i++) numbers[i] = i;
    std::shuffle(numbers.begin(), numbers.end(), std::mt19937(std::random_device()()));
    srand(std::random_device()());
    for (int i = 0; i < L; i++) {
      uint8_t a[12];
      for (int j = 0; j < 12; j++) a[j] = numbers[i] >> (j % 4) * 8 & 255;
      tree.append(SKey(a, 12), SValue(1, 1));
    }
    uint8_t a[12];
    memset(a, 0, sizeof(a));
    auto iter = std::unique_ptr<EstimateLSM<KeyCompType*>::SuperVersionIterator>(tree.seek(SKey(a, 12)));
    double ans = 0;
    while (iter->valid()) {
      auto kv = iter->read();
      ans += (kv.second.vlen + 12) * std::min(kv.second.counts * 0.5, 1.);
      iter->next();
    }
    logger("decay size: ", ans);
    auto end = std::chrono::system_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    logger("decay used time: ", double(dur.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den);
  }
  logger("test_lsm_decay(): OK");
}

void test_delete_range() {
  using namespace viscnts_lsm;

  auto start = std::chrono::system_clock::now();
  {
    EstimateLSM<KeyCompType*> tree(1e10, std::unique_ptr<Env>(createDefaultEnv()), std::make_unique<FileName>(0, "/tmp/viscnts/"),
                                   std::make_unique<DefaultAllocator>(), SKeyCompFunc, nullptr);
    int L = 3e8, Q = 1e4;
    std::vector<int> numbers(L);
    auto comp2 = +[](int x, int y) {
      uint8_t a[12], b[12];
      for (int j = 0; j < 12; j++) a[j] = x >> (j % 4) * 8 & 255;
      for (int j = 0; j < 12; j++) b[j] = y >> (j % 4) * 8 & 255;
      return SKeyCompFunc(SKey(a, 12), SKey(b, 12)) < 0;
    };
    for (int i = 0; i < L; i++) numbers[i] = i;
    // std::sort(numbers.begin(), numbers.end(), comp2);
    std::shuffle(numbers.begin(), numbers.end(), std::mt19937(std::random_device()()));
    srand(std::random_device()());
    for (int i = 0; i < L / 2; i++) {
      uint8_t a[12];
      for (int j = 0; j < 12; j++) a[j] = numbers[i] >> (j % 4) * 8 & 255;
      tree.append(SKey(a, 12), SValue(1, 1));
    }
    tree.all_flush();

    auto end = std::chrono::system_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    logger("flush used time: ", double(dur.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den);
    std::this_thread::sleep_for(std::chrono::milliseconds(5000));
    auto numbers2 = std::vector<int>(numbers.begin(), numbers.begin() + L / 2);
    std::sort(numbers2.begin(), numbers2.end(), comp2);
    {
      int Qs = 1000;
      for (int i = 0; i < Qs; i++) {
        uint8_t a[12], b[12];
        int id = abs(rand()) % (numbers2.size() - std::min(1000, L / 4));
        int x = numbers2[id];
        for (int j = 0; j < 12; j++) a[j] = x >> (j % 4) * 8 & 255;
        id += std::min(1000, L / 4);
        int y = numbers2[id];
        for (int j = 0; j < 12; j++) b[j] = y >> (j % 4) * 8 & 255;
        auto L = std::lower_bound(numbers2.begin(), numbers2.end(), x, comp2);
        auto R = std::upper_bound(numbers2.begin(), numbers2.end(), y, comp2);
        if (L > R) {
          i--;
          continue;
        }
        numbers2.erase(L, R);
        logger("[x,y]=", x, ",", y);
        tree.delete_range({SKey(a, 12), SKey(b, 12)});
      }
    }

    start = std::chrono::system_clock::now();
    for (int i = 0; i < Q; i++) {
      uint8_t a[12], b[12];
      int id = abs(rand()) % numbers.size();
      int x = numbers[id];
      for (int j = 0; j < 12; j++) a[j] = numbers[id] >> (j % 4) * 8 & 255;
      id = abs(rand()) % numbers.size();
      int y = numbers[id];
      for (int j = 0; j < 12; j++) b[j] = numbers[id] >> (j % 4) * 8 & 255;
      int ans = std::upper_bound(numbers2.begin(), numbers2.end(), x, comp2) - std::lower_bound(numbers2.begin(), numbers2.end(), y, comp2);
      ans = std::max(ans, 0);
      int output = tree.range_count({SKey(b, 12), SKey(a, 12)});
      // logger("[ans,output]=", ans, ",", output);
      assert(ans == output);
    }

    int QLEN = 1000;

    end = std::chrono::system_clock::now();
    dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    start = end;
    logger("range count used time: ", double(dur.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den);

    for (int i = 0; i < Q; i++) {
      uint8_t a[12];
      int id = abs(rand()) % numbers.size();
      for (int j = 0; j < 12; j++) a[j] = numbers[id] >> (j % 4) * 8 & 255;

      auto iter = std::unique_ptr<EstimateLSM<KeyCompType*>::SuperVersionIterator>(tree.seek(SKey(a, 12)));
      auto check_func = [](const uint8_t* a, int goal) {
        int x = 0, y = 0;
        for (int j = 0; j < 12; j++) {
          x |= a[j] << (j % 4) * 8;
          // if (j % 4 == 3) {
          //   // logger("[j,x,goal]=",j,",",x,",",goal);
          // }
          assert(j % 4 != 3 || x == goal);
          if (j % 4 == 3) {
            if (j > 3) {
              assert(y == x);
            }
            y = x;
            x = 0;
          }
        }
      };
      int x = a[0] | ((uint32_t)a[1] << 8) | ((uint32_t)a[2] << 16) | ((uint32_t)a[3] << 24);

      int cnt = 0;
      auto it = std::lower_bound(numbers2.begin(), numbers2.end(), x, comp2);
      while (true) {
        if (++cnt > QLEN) break;
        assert((it != numbers2.end()) == (iter->valid()));
        if (it == numbers2.end()) break;
        auto a = iter->read().first.data();
        // logger("[it]:", *it);
        check_func(a, *it);
        it++;
        iter->next();
      }
    }
    end = std::chrono::system_clock::now();
    dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    logger("random scan used time: ", double(dur.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den);
  }
  logger("test_random_scan(): OK");
}
