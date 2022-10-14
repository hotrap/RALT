#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <thread>

#include "alloc.hpp"
#include "cache.hpp"
#include "common.hpp"
#include "file.hpp"
#include "hash.hpp"
#include "key.hpp"
#include "memtable.hpp"

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

const static size_t kPageSize = 4096;                   // 4 KB
const static size_t kMagicNumber = 0x25a65facc3a23559;  // echo viscnts | sha1sum
const static size_t kSSTable = 1 << 28;
const static size_t kRatio = 10;
const static size_t kBatchSize = 1 << 24;
const static size_t kChunkSize = 1 << 12;  // 8 KB
const static size_t kIndexChunkSize = 1 << 12;
// about kMemTable... on average, we expect the size of index block < kDataBlockSize

// generate global filename
class FileName {
  std::atomic<size_t> file_ts_;
  std::filesystem::path path_;

 public:
  explicit FileName(size_t ts, std::string path) : file_ts_(ts), path_(path) {}
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
// I don't use cache here, because we don't need it?
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
  KVComp* comp_;

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
      // logger("seek_and_read(): ", _offset);
      // read key
      auto [chunk_key_id, key_offset] = block_._kv_offset(_offset);
      if (current_key_id_ != chunk_key_id) block_.acquire(current_key_id_ = chunk_key_id, current_key_chunk_);
      block_.read_key(key_offset, current_key_chunk_, key);
    }

    auto seek_offset(uint32_t id) {
      // find offset
      auto [chunk_id, offset] = block_._pos_offset(id);
      // logger("seek_offset(): [chunk_id, offset] = ", chunk_id, ", ", offset);
      if (current_value_id_ != chunk_id) block_.acquire(current_value_id_ = chunk_id, currenct_value_chunk_);
      auto ret = block_.read_value(offset, currenct_value_chunk_);
      // logger("seek_offset(): return ", ret);
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
      auto [chunk_id, chunk_offset] = block_._kv_offset(offset);
      current_key_id_ = chunk_id;
      offset_ = chunk_offset;
      current_key_chunk_ = block_.acquire(current_key_id_);
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
      // logger("EnumIterator&!");
      // assert(0);
      return (*this);
    }

    void next() {
      assert(valid());
      if (!key_size_) _read_size();
      // assert(key_size_ == 32);
      offset_ += key_size_;
      assert(offset_ <= kChunkSize);
      key_size_ = 0, id_++;
      if (valid() && (offset_ + sizeof(uint32_t) >= kChunkSize || block_.is_empty_key(offset_, current_key_chunk_))) {
        offset_ = 0, current_key_id_++;
        block_.acquire(current_key_id_, current_key_chunk_);
      }
    }
    auto read(KV& key) {
      // for (int i = 0; i < 12; i++) printf("[%x]", *current_key_chunk_.data(i + 4));
      // printf("<%d>", offset_);
      // puts("<<<");
      // fflush(stdout);
      assert(valid());
      block_.read_key(offset_, current_key_chunk_, key);
      // for (int i = 0; i < 12; i++) printf("[%x]", *(key.key().data() + i));
      // puts("<<<");
      // fflush(stdout);
      key_size_ = key.size();
    }
    bool valid() { return id_ < block_.handle_.counts; }

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
  explicit FileBlock(uint32_t file_id, FileBlockHandle handle, BaseAllocator* alloc, RandomAccessFile* file_ptr, KVComp* comp)
      : file_id_(file_id), handle_(handle), alloc_(alloc), file_ptr_(file_ptr), comp_(comp) {
    // logger("init fileblock");
    offset_index_ = handle_.offset + handle_.size - handle_.counts * sizeof(uint32_t);
    assert(offset_index_ % kChunkSize == 0);
  }

  explicit FileBlock(uint32_t file_id, FileBlockHandle handle, BaseAllocator* alloc, RandomAccessFile* file_ptr, uint32_t offset_index, KVComp* comp)
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

  ssize_t exists(const SKey& key) {
    int l = 0, r = handle_.counts - 1;
    SeekIterator it = SeekIterator(*this);
    // logger("OK");
    KV _kv;
    while (l <= r) {
      auto mid = (l + r) >> 1;
      // compare two keys
      it.seek_and_read(mid, _kv);
      auto cmp = comp_(_kv.key(), key);
      if (!cmp) return 1;
      if (cmp < 0)
        l = mid + 1;
      else
        r = mid - 1;
    }
    // logger("OK2");
    return 0;
  }

  // it's only used for IndexKey, i.e. BlockKey<uint32_t>, so that the type of kv.value() is uint32_t.
  uint32_t upper_offset(const SKey& key) {
    int l = 0, r = handle_.counts - 1;
    uint32_t ret = -1;
    SeekIterator it = SeekIterator(*this);
    KV _key;
    while (l <= r) {
      auto mid = (l + r) >> 1;
      // logger("[l, r, mid] = ", l, ", ", r, ", ", mid);
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
  uint32_t lower_offset(const SKey& key) {
    int l = 0, r = handle_.counts - 1;
    uint32_t ret = -1;
    SeekIterator it = SeekIterator(*this);
    KV _key;
    while (l <= r) {
      auto mid = (l + r) >> 1;
      // logger("[l, r, mid] = ", l, ", ", r, ", ", mid);
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
  uint32_t upper_key(const SKey& key, uint32_t L, uint32_t R) {
    int l = L, r = std::min(R, handle_.counts - 1);
    uint32_t ret = -1;
    SeekIterator it = SeekIterator(*this);
    KV _key;
    while (l <= r) {
      auto mid = (l + r) >> 1;
      // logger("[l, r, mid] = ", l, ", ", r, ", ", mid);
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
  uint32_t upper_key_not_eq(const SKey& key, uint32_t L, uint32_t R) {
    int l = L, r = std::min(R, handle_.counts - 1);
    uint32_t ret = -1;
    SeekIterator it = SeekIterator(*this);
    KV _key;
    while (l <= r) {
      auto mid = (l + r) >> 1;
      // logger("[l, r, mid] = ", l, ", ", r, ", ", mid);
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
    // logger("seek_with_offset(): ", id);
    return EnumIterator(*this, it.seek_offset(id), id);
  }

  auto sub_fileblock(uint32_t offset_l, uint32_t offset_r) const {
    assert(offset_l >= offset_index_ && offset_r >= offset_index_);
    auto counts_l = (offset_l - offset_index_) / sizeof(uint32_t);
    auto counts_r = (offset_r - offset_index_) / sizeof(uint32_t);
    return FileBlock<KV, KVComp>(file_id_, FileBlockHandle(handle_.offset, offset_r - handle_.offset, counts_r - counts_l), alloc_, file_ptr_,
                                 offset_l, comp_);
  }

  size_t counts() const { return handle_.counts; }
};

template <typename Value>
class BlockKey {
 private:
  SKey key_;
  Value v_;

 public:
  BlockKey() = default;
  BlockKey(SKey key, Value v) : key_(key), v_(v) {}
  uint8_t* read(uint8_t* from) {
    from = key_.read(from);
    assert(key_.len() == 12);
    v_ = *reinterpret_cast<Value*>(from);
    return from + sizeof(Value);
  }
  size_t size() const { return key_.size() + sizeof(v_); }
  uint8_t* write(uint8_t* to) const {
    to = key_.write(to);
    assert(key_.len() == 12);
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

int SKeyCompFunc(const SKey& A, const SKey& B) {
  if (A.len() != B.len()) return A.len() < B.len() ? -1 : 1;
  return memcmp(A.data(), B.data(), A.len());
}

using KeyCompType = int(const SKey&, const SKey&);

// one SST
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
  FileBlock<IndexKey, KeyCompType> index_block_;
  FileBlock<DataKey, KeyCompType> data_block_;
  // LRUCache pointer reference to the one in VisCnts (is deleted)
  BaseAllocator* alloc_;
  KeyCompType* comp_;

 public:
  ImmutableFile(uint32_t file_id, uint32_t size, std::unique_ptr<RandomAccessFile>&& file_ptr, BaseAllocator* alloc,
                const std::pair<IndSKey, IndSKey>& range, KeyCompType* comp)
      : file_id_(file_id), size_(size), range_(range), file_ptr_(std::move(file_ptr)), alloc_(alloc), comp_(comp) {
    // read index block
    Slice result(nullptr, 0);
    FileBlockHandle index_bh, data_bh;
    size_t mgn;
    auto ret = file_ptr_->read(size_ - sizeof(size_t), sizeof(size_t), (uint8_t*)(&mgn), result);
    assert(ret >= 0);
    // logger("file size: ", size);
    // logger("magic number: ", mgn, " - ", kMagicNumber);
    assert(mgn == kMagicNumber);
    ret = file_ptr_->read(size_ - sizeof(size_t) - sizeof(FileBlockHandle) * 2, sizeof(FileBlockHandle), (uint8_t*)(&index_bh), result);
    assert(ret >= 0);
    ret = file_ptr_->read(size_ - sizeof(size_t) - sizeof(FileBlockHandle), sizeof(FileBlockHandle), (uint8_t*)(&data_bh), result);
    assert(ret >= 0);
    // logger_printf("index_block: [file_size=%d, counts=%d, offset=%d, size=%d]", size, index_bh.counts, index_bh.offset, index_bh.size);
    // logger_printf("data_block: [file_size=%d, counts=%d, offset=%d, size=%d]", size, data_bh.counts, data_bh.offset, data_bh.size);
    index_block_ = FileBlock<IndexKey, KeyCompType>(file_id, index_bh, alloc, file_ptr_.get(), comp_);
    data_block_ = FileBlock<DataKey, KeyCompType>(file_id, data_bh, alloc, file_ptr_.get(), comp_);

    // Chunk _c(0, alloc_, file_ptr_.get());
    // for (uint32_t i = 0; i < data_bh.counts; ++i) {
    //   if (i * 32 % kChunkSize == 0) data_block_.acquire(i / kChunkSize, _c);
    //   assert(*(uint32_t*)(_c.data(i * 32 % kChunkSize)) == 12);
    // }
    // Chunk _c(0, alloc_, file_ptr_.get());
    // for (int i = 0; i < 12; i++) printf("[%x]", *_c.data(i + 4));
    // logger("OK");
  }

  // ensure key is in range!!
  // I don't want to check range here...
  ssize_t exists(const SKey& key) {
    if (!in_range(key)) return 0;
    auto offset = index_block_.upper_offset(key);
    // offset_l is % kChunkSize = 0.
    // index keys are set every kChunkSize / sizeof(uint32_t) keys.
    auto ret = data_block_.sub_fileblock(offset / kChunkSize * kChunkSize, offset + sizeof(uint32_t)).exists(key);
    return ret;
  }

  FileBlock<DataKey, KeyCompType>::EnumIterator estimate_seek(const SKey& key) {
    if (comp_(range_.second.ref(), key) < 0) return {};
    auto id = index_block_.upper_offset(key);
    // logger("estimate_seek(): ", id);
    if (id == -1) return {};
    return data_block_.seek_with_id(id);
  }

  FileBlock<DataKey, KeyCompType>::EnumIterator seek(const SKey& key) {
    if (comp_(range_.second.ref(), key) < 0) return {};
    auto id = index_block_.lower_offset(key);
    // logger("seek(): ", id);
    if (id == -1) return data_block_.seek_with_id(0);
    id = data_block_.upper_key(key, id, id + kIndexChunkSize - 1);
    // logger("seek(accurate)(): ", id);
    return data_block_.seek_with_id(id);
  }

  size_t range_count(const SKey& L, const SKey& R) {
    size_t retl = 0, retr = 0;
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
    if (comp_(range_.second.ref(), R) < 0)
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

    return retr - retl;
  }

  bool in_range(const SKey& key) {
    auto& [l, r] = range_;
    return comp_(l.ref(), key) <= 0 && comp_(key, r.ref()) <= 0;
  }
  std::pair<SKey, SKey> range() const { return {range_.first.ref(), range_.second.ref()}; }

  size_t size() const { return size_; }

  size_t counts() const { return data_block_.counts(); }

  bool range_overlap(const std::pair<SKey, SKey>& range) const {
    auto& [l, r] = range_;
    return comp_(l.ref(), range.second) <= 0 && comp_(range.first, r.ref()) <= 0;
  }

  FileBlock<DataKey, KeyCompType> data_block() const { return data_block_; }

  void remove() {
    auto ret = file_ptr_->remove();
    assert(ret >= 0);
  }
};

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
    if (used_size_ + x.size() > buffer_size_) {
      auto ptr = new uint8_t[x.size()];
      x.write(ptr);
      append(Slice(ptr, x.size()));
      delete[] ptr;
    } else {
      x.write(data_ + used_size_);
      used_size_ += x.size();
    }
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

  void check(int C) {
    for (int i = 0; i < C; i += 32) assert(*(uint32_t*)(data_ + i) != 0);
  }
};

template <typename Iterator, typename KVComp>
class SeqIteratorSet {
  std::vector<Iterator> iters_;
  std::vector<Iterator*> seg_tree_;
  std::vector<SKey> keys_;
  std::vector<SValue> values_;
  uint32_t size_;
  KVComp* comp_;

 public:
  SeqIteratorSet(KVComp* comp) : size_(0), comp_(comp) {}
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

 private:
  uint32_t _min(uint32_t x, uint32_t y) {
    uint32_t idx = seg_tree_[x] - iters_.data();
    uint32_t idy = seg_tree_[y] - iters_.data();
    return comp_(keys_[idx], keys_[idy]) < 0 ? x : y;
  }
};

class SSTIterator {
 public:
  SSTIterator() : file_(nullptr) {}
  SSTIterator(ImmutableFile* file) : file_(file), file_block_iter_(file->data_block(), 0, 0) {}
  SSTIterator(ImmutableFile* file, const SKey& key) : file_(file), file_block_iter_(file->seek(key)) {}
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

 private:
  ImmutableFile* file_;
  FileBlock<DataKey, KeyCompType>::EnumIterator file_block_iter_;
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
    // kv.key().print();
    assert(kv.key().len() > 0);
    _append_align(kv.size());
    // assert(kv.size() == 32);
    if (offsets.size() % (kIndexChunkSize / sizeof(uint32_t)) == 0) {
      index.emplace_back(kv.key(), offsets.size());
      if (offsets.size() == 0) first_key = kv.key();
    } else
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
  void make_index() {
    // if (offsets.size() % (kChunkSize / sizeof(uint32_t)) != 1) {
    //   index.emplace_back(lst_key, offsets.size());  // append last key into index block.
    // }
    // logger("SSTB: ", now_offset, " num: ", offsets.size());
    _align();
    // file_->check(offsets.size() * 32);
    // logger("SSTB(AFTER ALIGN): ", now_offset);
    auto data_index_offset = now_offset;
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

class Compaction {
  // builder_ is used to build one file
  // files_ is used to get global file name
  // env_ is used to get global environment
  // flag_ means whether lst_value_ is valid
  // lst_value_ is the last value appended to builder_
  // vec_newfiles_ stores the information of new files.
  SSTBuilder builder_;
  FileName* files_;
  Env* env_;
  bool flag_;
  std::pair<IndSKey, SValue> lst_value_;
  std::mt19937 rndgen_;
  KeyCompType* comp_;

  void _begin_new_file() {
    builder_.reset();
    auto [filename, id] = files_->next_pair();
    // logger("begin_new_file(): ", filename);
    // logger("begin_new_file(): size: ", builder_.size());
    builder_.new_file(std::make_unique<WriteBatch>(std::unique_ptr<AppendFile>(env_->openAppFile(filename)), kBatchSize));
    vec_newfiles_.emplace_back();
    vec_newfiles_.back().filename = filename;
    vec_newfiles_.back().file_id = id;
  }

  void _end_new_file() {
    // logger("end_new_file(): kvsize: ", builder_.kv_size());
    // logger("end_new_file(): size: ", builder_.size());
    builder_.make_index();
    // logger("end_new_file(): size: ", builder_.size());
    builder_.finish();
    // logger("end_new_file(): size: ", builder_.size());
    vec_newfiles_.back().size = builder_.size();
    vec_newfiles_.back().range = builder_.range();
    // for(auto & a:vec_newfiles_)
    // printf("[%lld,%lld]", a.range.first.data(), a.range.second.data());fflush(stdout);
  }

 public:
  struct NewFileData {
    std::string filename;
    size_t file_id;
    size_t size;
    std::pair<IndSKey, IndSKey> range;
  };
  Compaction(FileName* files, Env* env, KeyCompType* comp) : files_(files), env_(env), flag_(false), rndgen_(std::random_device()()), comp_(comp) {
    decay_prob_ = 0.5;
  }

  template <typename TIter>
  auto flush(TIter& left) {
    vec_newfiles_.clear();
    _begin_new_file();
    flag_ = false;
    double real_size = 0;
    int CNT = 0;
    while (left.valid()) {
      CNT++;
      auto L = left.read();
      if (flag_ && comp_(lst_value_.first.ref(), L.first) == 0) {
        lst_value_.second += L.second;
      } else {
        if (flag_) {
          real_size += _calc_decay_value(lst_value_);
          builder_.append(lst_value_);
          _divide_file();
        }
        lst_value_ = L;
        flag_ = true;
      }
      left.next();
    }
    // logger("flush(): ", CNT);
    if (flag_) real_size += _calc_decay_value(lst_value_), builder_.append(lst_value_);
    _end_new_file();
    return std::make_pair(vec_newfiles_, real_size);
  }

  template <typename TIter>
  std::pair<std::vector<NewFileData>, double> decay_first(TIter&& iters) {
    double real_size_ = 0;  // re-calculate size
    vec_newfiles_.clear();
    _begin_new_file();
    flag_ = false;
    int A = 0;
    while (iters.valid()) {
      auto L = iters.read();
      if (flag_ && comp_(lst_value_.first.ref(), L.first) == 0) {
        lst_value_.second += L.second;
      } else {
        if (flag_ && _decay_kv(lst_value_)) {
          A++;
          real_size_ += _calc_decay_value(lst_value_);
          builder_.append(lst_value_);
          _divide_file();
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
    return {vec_newfiles_, real_size_};
  }

 private:
  double decay_prob_;  // = 0.5 on default
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
  void _divide_file() {
    if (builder_.kv_size() > kSSTable) {
      // logger("[divide file]");
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
constexpr auto kUnsortedBufferSize = 1 << 20;
constexpr auto kUnsortedBufferMaxQueue = 8;

class EstimateLSM {
  struct Partition {
    ImmutableFile file;
    Partition(Env* env, BaseAllocator* file_alloc, KeyCompType* comp, const Compaction::NewFileData& data)
        : file(data.file_id, data.size, std::unique_ptr<RandomAccessFile>(env->openRAFile(data.filename)), file_alloc, data.range, comp) {}
    ~Partition() { file.remove(); }

    SSTIterator seek(const SKey& key) {
      // assert(file.in_range(key));
      return SSTIterator(&file, key);
    }
    SSTIterator begin() { return SSTIterator(&file); }
    bool overlap(const SKey& lkey, const SKey& rkey) const { return file.range_overlap({lkey, rkey}); }
    auto range() const { return file.range(); }
    size_t size() const { return file.size(); }
    size_t counts() const { return file.counts(); }
    size_t range_count(const SKey& L, const SKey& R) { return file.range_count(L, R); }
  };
  struct Level {
    std::vector<std::shared_ptr<Partition>> head_;
    size_t size_;
    double decay_size_;
    class LevelIterator {
      std::vector<std::shared_ptr<Partition>>::const_iterator vec_it_;
      std::vector<std::shared_ptr<Partition>>::const_iterator vec_it_end_;
      SSTIterator iter_;
      std::shared_ptr<std::vector<std::shared_ptr<Partition>>> vec_ptr_;

     public:
      LevelIterator() {}
      LevelIterator(const std::vector<std::shared_ptr<Partition>>& vec, int id, SSTIterator&& iter)
          : vec_it_(vec.begin() + id + 1), vec_it_end_(vec.end()), iter_(std::move(iter)) {}
      LevelIterator(const std::vector<std::shared_ptr<Partition>>& vec, int id)
          : vec_it_(vec.begin() + id + 1), vec_it_end_(vec.end()), iter_(vec[id]->begin()) {}
      LevelIterator(std::shared_ptr<std::vector<std::shared_ptr<Partition>>> vec_ptr, int id, SSTIterator&& iter)
          : vec_it_(vec_ptr->begin() + id + 1), vec_it_end_(vec_ptr->end()), iter_(std::move(iter)), vec_ptr_(std::move(vec_ptr)) {}
      LevelIterator(std::shared_ptr<std::vector<std::shared_ptr<Partition>>> vec_ptr, int id)
          : vec_it_(vec_ptr->begin() + id + 1), vec_it_end_(vec_ptr->end()), iter_((*vec_ptr)[id]->begin()), vec_ptr_(std::move(vec_ptr)) {}
      bool valid() { return iter_.valid(); }
      void next() {
        iter_.next();
        if (!iter_.valid() && vec_it_ != vec_it_end_) {
          iter_ = SSTIterator(&(*vec_it_)->file);
          vec_it_++;
        }
      }
      void read(DataKey& kv) { return iter_.read(kv); }
    };
    Level() : size_(0), decay_size_(0) {}
    Level(size_t size, double decay_size) : size_(size), decay_size_(decay_size) {}
    size_t size() const { return size_; }
    double decay_size() const { return decay_size_; }
    LevelIterator seek(const SKey& key, KeyCompType* comp) const {
      int l = 0, r = head_.size() - 1, where = -1;
      while (l <= r) {
        int mid = (l + r) >> 1;
        if (comp(key, head_[mid]->range().second) <= 0)
          where = mid, r = mid - 1;
        else
          l = mid + 1;
      }
      // logger("Level::seek(): ", where);
      if (where == -1) return LevelIterator();
      return LevelIterator(head_, where, head_[where]->seek(key));
    }
    bool overlap(const SKey& lkey, const SKey& rkey, KeyCompType* comp) const {
      int l = 0, r = head_.size() - 1, where = -1;
      while (l <= r) {
        int mid = (l + r) >> 1;
        if (comp(lkey, head_[mid]->range().second) <= 0)
          where = mid, l = mid + 1;
        else
          r = mid - 1;
      }
      if (where == -1 && head_.size()) {
        return head_[0]->overlap(lkey, rkey);
      } else
        return head_[where]->overlap(lkey, rkey);
    }
    void append_par(std::shared_ptr<Partition> par) {
      size_ += par->size();
      head_.push_back(std::move(par));
    }
    void add_decay_size(double add) { decay_size_ += add; }

    size_t range_count(const SKey& L, const SKey& R, KeyCompType* comp) {
      int l = 0, r = head_.size() - 1, where_l = -1, where_r = -1;
      while (l <= r) {
        int mid = (l + r) >> 1;
        if (comp(L, head_[mid]->range().second) <= 0)
          where_l = mid, r = mid - 1;
        else
          l = mid + 1;
      }

      l = 0, r = head_.size() - 1;
      while (l <= r) {
        int mid = (l + r) >> 1;
        if (comp(head_[mid]->range().first, R) <= 0)
          where_r = mid, l = mid + 1;
        else
          r = mid - 1;
      }

      // logger_printf("where_l, where_r = %d, %d\n", where_l, where_r);
      if (where_l == -1 || where_r == -1) return 0;

      if (where_l == where_r) {
        return head_[where_l]->range_count(L, R);
      } else {
        size_t ans = 0;
        for (int i = where_l + 1; i < where_r; i++) ans += head_[i]->counts();
        ans += head_[where_l]->range_count(L, R);
        ans += head_[where_r]->range_count(L, R);
        return ans;
      }
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

   public:
    SuperVersion(double decay_limit, std::mutex* mutex)
        : largest_(std::make_shared<Level>()), ref_(1), decay_size_overestimate_(0), decay_limit_(decay_limit), del_mutex_(mutex) {}
    void ref() { ref_++; }
    void unref() {
      if (!--ref_) {
        // std::unique_lock lck(*del_mutex_);
        delete this;
      }
    }

    SeqIteratorSet<Level::LevelIterator, KeyCompType> seek(const SKey& key, KeyCompType* comp) const {
      SeqIteratorSet<Level::LevelIterator, KeyCompType> ret(comp);
      if (largest_.get() != nullptr) ret.push(largest_->seek(key, comp));
      for (auto& a : tree_) ret.push(a->seek(key, comp));
      return ret;
    }

    // in critical section
    SuperVersion* flush_bufs(const std::vector<UnsortedBuffer*>& bufs, FileName* filename, Env* env, BaseAllocator* file_alloc, KeyCompType* comp) {
      if (bufs.size() == 0) return nullptr;
      auto ret = new SuperVersion(decay_limit_, del_mutex_);
      ret->tree_ = tree_;
      ret->largest_ = largest_;
      ret->decay_size_overestimate_ = decay_size_overestimate_;
      for (auto& buf : bufs) {
        Compaction worker(filename, env, comp);
        buf->sort(comp);
        auto iter = std::make_unique<UnsortedBuffer::Iterator>(*buf);
        auto [files, decay_size] = worker.flush(*iter);
        auto level = std::make_shared<Level>(0, decay_size);
        // decay_size_overestimate is the sum of decay_size of all SSTs.
        ret->decay_size_overestimate_ += decay_size;
        for (auto& a : files) level->append_par(std::make_shared<Partition>(env, file_alloc, comp, a));
        ret->tree_.push_back(std::move(level));
        delete buf;
      }
      return ret;
    }
    SuperVersion* compact(FileName* filename, Env* env, BaseAllocator* file_alloc, KeyCompType* comp) const {
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
            for_iter_ptr->push_back(par);
          } else
            rest.push_back(par);
        }
        return for_iter_ptr;
      };
      // If the unmerged data is too big, and estimated decay size is large enough.
      // all the partitions will be merged into the largest level.
      if ((Lsize >= kUnmergedRatio * largest_->size() && _decay_possible())) {
        logger("[decay size, Lsize, largest_->size()]: ", decay_size_overestimate_, ", ", Lsize, ", ", largest_->size());
        auto iters = std::make_unique<SeqIteratorSet<Level::LevelIterator, KeyCompType>>(comp);

        // logger("Major Compaction tree_.size() = ", tree_.size());
        // push all iterators to necessary partitions to SeqIteratorSet.
        for (auto& a : tree_) iters->push(Level::LevelIterator(add_level(*a), 0));
        if (largest_ && largest_->size() != 0) iters->push(Level::LevelIterator(add_level(*largest_), 0));
        iters->build();

        // compaction...
        Compaction worker(filename, env, comp);
        auto [files, decay_size] = worker.decay_first(*iters);
        auto ret = new SuperVersion(decay_limit_, del_mutex_);
        // major compaction, so levels except largest_ become empty.
        ret->decay_size_overestimate_ = decay_size;
        logger("[new decay size]: ", ret->decay_size_overestimate_);

        auto rest_iter = rest.begin();

        // std::unique_lock lck(*del_mutex_);
        for (auto& a : files) {
          auto par = std::make_shared<Partition>(env, file_alloc, comp, a);
          while (rest_iter != rest.end() && comp((*rest_iter)->range().first, par->range().first) <= 0) {
            ret->largest_->append_par(std::move(*rest_iter));
            rest_iter++;
          }
          ret->largest_->append_par(std::move(par));
        }
        while (rest_iter != rest.end()) ret->largest_->append_par(std::move(*rest_iter++));

        return ret;
      } else {
        // similar to universal compaction in rocksdb
        // if tree_[i]->size()/(\sum_{j=0..i-1}tree_[j]->size()) <= kMergeRatio
        // then merge them.
        // if kMergeRatio ~ 1./X, then it can be treated as (X+1)-tired compaction

        // if the number of tables >= kLimitMin, then begin to merge
        // if the number of tables >= kLimitMax, then increase kMergeRatio. (this works when the number of tables is small)

        // logger("tree_.size() = ", tree_.size());

        if (tree_.size() >= kLimitMin) {
          auto _kRatio = kMergeRatio;
          double min_ratio = 1e30;
          int where = -1, _where = -1;
          size_t sum = tree_.back()->size();
          for (int i = tree_.size() - 2; i >= 0; --i) {
            if (tree_[i]->size() <= sum * _kRatio)
              where = i;
            else if (auto t = tree_[i]->size() / (double)sum; min_ratio < t) {
              _where = i;
              min_ratio = t;
            }
            sum += tree_[i]->size();
          }
          if (tree_.size() >= kLimitMax) {
            where = _where;
            size_t sum = tree_.back()->size();
            for (int i = tree_.size() - 2; i >= 0; --i) {
              if (tree_[i]->size() <= sum * min_ratio) where = i;
              sum += tree_[i]->size();
            }
          }
          if (where == -1) return nullptr;

          auto iters = std::make_unique<SeqIteratorSet<Level::LevelIterator, KeyCompType>>(comp);

          // logger("compact(): where = ", where);
          // push all iterators to necessary partitions to SeqIteratorSet.
          for (uint32_t i = where; i < tree_.size(); ++i) iters->push(Level::LevelIterator(add_level(*tree_[i]), 0));
          iters->build();

          // compaction...
          Compaction worker(filename, env, comp);
          auto [files, decay_size] = worker.flush(*iters);
          auto ret = new SuperVersion(decay_limit_, del_mutex_);
          // minor compaction?
          // decay_size_overestimate = largest_ + remaining levels + new level
          ret->tree_ = tree_;
          ret->largest_ = largest_;
          ret->decay_size_overestimate_ = largest_->decay_size() + decay_size;
          for (int i = 0; i < where; i++) ret->decay_size_overestimate_ += tree_[i]->decay_size();

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
          return ret;
        }
        return nullptr;
      }
    }

    size_t range_count(const SKey& L, const SKey& R, KeyCompType* comp) {
      size_t ans = 0;
      if (largest_) ans += largest_->range_count(L, R, comp);
      for (auto& a : tree_) ans += a->range_count(L, R, comp);
      return ans;
    }
  };
  double delta_;
  std::unique_ptr<Env> env_;
  std::unique_ptr<FileName> filename_;
  std::unique_ptr<BaseAllocator> file_alloc_;
  KeyCompType* comp_;
  SuperVersion* sv_;
  std::thread compact_thread_;
  bool terminate_signal_;
  UnsortedBufferPtrs bufs_;
  std::mutex sv_mutex_;
  std::mutex sv_load_mutex_;

 public:
  class SuperVersionIterator {
    SeqIteratorSet<Level::LevelIterator, KeyCompType> iter_;
    SuperVersion* sv_;

   public:
    SuperVersionIterator(SeqIteratorSet<Level::LevelIterator, KeyCompType>&& iter, SuperVersion* sv) : iter_(std::move(iter)), sv_(sv) {
      iter_.build();
    }
    ~SuperVersionIterator() { sv_->unref(); }
    auto read() { return iter_.read(); }
    auto valid() { return iter_.valid(); }
    auto next() { return iter_.next(); }
  };
  EstimateLSM(double delta, std::unique_ptr<Env>&& env, std::unique_ptr<FileName>&& filename, std::unique_ptr<BaseAllocator>&& file_alloc,
              KeyCompType* comp)
      : delta_(delta),
        env_(std::move(env)),
        filename_(std::move(filename)),
        file_alloc_(std::move(file_alloc)),
        comp_(comp),
        sv_(new SuperVersion(delta, &sv_mutex_)),
        terminate_signal_(0),
        bufs_(kUnsortedBufferSize, kUnsortedBufferMaxQueue) {
    compact_thread_ = std::thread([this]() { compact_thread(); });
  }
  ~EstimateLSM() {
    terminate_signal_ = 1;
    bufs_.notify_cv();
    compact_thread_.join();
    std::unique_lock lck(sv_load_mutex_);
    sv_->unref();
  }
  void append(const SKey& key, const SValue& value) { bufs_.append_and_notify(key, value); }
  auto seek(const SKey& key) {
    auto sv = get_current_sv();
    return new SuperVersionIterator(sv->seek(key, comp_), sv);
  }

  size_t range_count(const SKey& L, const SKey& R) {
    if (comp_(L, R) > 0) return 0;
    auto sv = get_current_sv();
    return sv->range_count(L, R, comp_);
  }

  void all_flush() {
    bufs_.flush();
    while (bufs_.size())
      ;
  }

 private:
  void compact_thread() {
    while (!terminate_signal_) {
      auto buf_q_ = terminate_signal_ ? bufs_.get() : bufs_.wait_and_get();
      if (buf_q_.empty() && terminate_signal_) return;
      if (buf_q_.empty()) continue;
      // logger("FLUSH");
      auto old_sv = sv_;
      auto new_sv = old_sv->flush_bufs(buf_q_, filename_.get(), env_.get(), file_alloc_.get(), comp_);
      if (new_sv != nullptr) {
        sv_load_mutex_.lock();
        sv_ = new_sv;
        sv_load_mutex_.unlock();
        old_sv->unref();
      }
      // logger("COMPACT");
      old_sv = sv_;
      new_sv = old_sv->compact(filename_.get(), env_.get(), file_alloc_.get(), comp_);
      if (new_sv != nullptr) {
        sv_load_mutex_.lock();
        sv_ = new_sv;
        sv_load_mutex_.unlock();
        old_sv->unref();
      }
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

class VisCnts {
  EstimateLSM tree;

 public:
  VisCnts(const std::string& path, double delta, bool createIfMissing)
      : tree(delta, std::unique_ptr<Env>(createDefaultEnv()), std::make_unique<FileName>(0, path), std::make_unique<DefaultAllocator>(),
             SKeyCompFunc) {}
  void access(const std::pair<SKey, SValue>& kv) { tree.append(kv.first, kv.second); }
  bool is_hot(const SKey& key) { return true; }
};

}  // namespace viscnts_lsm

void* VisCntsOpen(const char* path, double delta, bool createIfMissing) {
  auto ret = new viscnts_lsm::VisCnts(path, delta, createIfMissing);
  return ret;
}

int VisCntsAccess(void* ac, const char* key, size_t klen, size_t vlen) {
  auto vc = reinterpret_cast<viscnts_lsm::VisCnts*>(ac);
  vc->access({viscnts_lsm::SKey(const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(key)), klen), viscnts_lsm::SValue(1, vlen)});
  return 0;
}

bool VisCntsIsHot(void* ac, const char* key, size_t klen) {
  auto vc = reinterpret_cast<viscnts_lsm::VisCnts*>(ac);
  return vc->is_hot(viscnts_lsm::SKey(const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(key)), klen)) == 1;
}

int VisCntsClose(void* ac) {
  auto vc = reinterpret_cast<viscnts_lsm::VisCnts*>(ac);
  delete vc;
  return 0;
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

  // auto comp = +[](const SKey& a, const SKey& b) {
  //   int ret = memcmp(a.a_, b.a_, std::min(a.len_, b.len_));
  //   if (!ret) return (int)a.len_ - (int)b.len_;
  //   return ret;
  // };

  auto comp = +[](const SKey& a, const SKey& b) {
    auto ap = a.data(), bp = b.data();
    uint32_t x = ap[0] | ((uint32_t)ap[1] << 8) | ((uint32_t)ap[2] << 16) | ((uint32_t)ap[3] << 24);
    uint32_t y = bp[0] | ((uint32_t)bp[1] << 8) | ((uint32_t)bp[2] << 16) | ((uint32_t)bp[3] << 24);
    return (int)x - (int)y;
  };

  std::vector<ImmutableFile> files;
  for (int i = 0; i < FS; ++i)
    files.push_back(ImmutableFile(0, builders[i].size(), std::unique_ptr<RandomAccessFile>(env_->openRAFile("/tmp/viscnts/test" + std::to_string(i))),
                                  new DefaultAllocator(), {}, comp));
  auto iters = std::make_unique<SeqIteratorSet<SSTIterator, KeyCompType>>(comp);
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
          // logger("FUCK");
          auto buf_q_ = signal.load() ? bufs.get() : bufs.wait_and_get();
          using namespace std::chrono;

          if (!buf_q_.size()) {
            if (signal) break;
            continue;
          }
          for (auto& buf : buf_q_) {
            // while(!buf->safe());
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
  assert(result.size() == L);
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
    EstimateLSM tree(1e9, std::unique_ptr<Env>(createDefaultEnv()), std::make_unique<FileName>(0, "/tmp/viscnts/"),
                     std::make_unique<DefaultAllocator>(), SKeyCompFunc);
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
    EstimateLSM tree(1e9, std::unique_ptr<Env>(createDefaultEnv()), std::make_unique<FileName>(0, "/tmp/viscnts/"),
                     std::make_unique<DefaultAllocator>(), comp);
    int L = 1e7;
    uint8_t a[12];
    memset(a, 0, sizeof(a));
    std::vector<int> numbers(L);
    for (int i = 0; i < L; i++) numbers[i] = i;
    std::shuffle(numbers.begin(), numbers.end(), std::mt19937(std::random_device()()));
    for (int i = 0; i < L; i++) {
      for (int j = 0; j < 12; j++) a[j] = numbers[i] >> (j % 4) * 8 & 255;
      tree.append(SKey(a, 12), SValue(1, 1));
    }
    tree.all_flush();
    // memset(a, 0, sizeof(a));
    std::this_thread::sleep_for(std::chrono::milliseconds(5000));
    memset(a, 0, sizeof(a));
    auto iter = std::unique_ptr<EstimateLSM::SuperVersionIterator>(tree.seek(SKey(a, 12)));
    for (int i = 0; i < L; i++) {
      assert(iter->valid());
      auto kv = iter->read();
      int x = 0, y = 0;
      auto a = kv.first.data();
      for (int j = 0; j < 12; j++) {
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
      assert(kv.second.counts == 1);
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
    EstimateLSM tree(1e9, std::unique_ptr<Env>(createDefaultEnv()), std::make_unique<FileName>(0, "/tmp/viscnts/"),
                     std::make_unique<DefaultAllocator>(), SKeyCompFunc);
    int L = 1e8, Q = 1e4;
    std::vector<int> numbers(L);
    auto comp2 = +[](int x, int y) {
      uint8_t a[12], b[12];
      for (int j = 0; j < 12; j++) a[j] = x >> (j % 4) * 8 & 255;
      for (int j = 0; j < 12; j++) b[j] = y >> (j % 4) * 8 & 255;
      return SKeyCompFunc(SKey(a, 12), SKey(b, 12)) < 0;
    };
    for (int i = 0; i < L; i++) numbers[i] = i;
    std::shuffle(numbers.begin(), numbers.end(), std::mt19937(std::random_device()()));
    srand(std::random_device()());
    for (int i = 0; i < L / 2; i++) {
      uint8_t a[12];
      for (int j = 0; j < 12; j++) a[j] = numbers[i] >> (j % 4) * 8 & 255;
      tree.append(SKey(a, 12), SValue(1, 1));
      // if (rand() % std::max(1, L / 1000) == 0) {
      //   std::thread(
      //       [](EstimateLSM& tree) {
      //         uint8_t a[12];
      //         int id = abs(rand()) % 100000;
      //         for (int j = 0; j < 12; j++) a[j] = id >> (j % 4) * 8 & 255;
      //         auto iter = std::unique_ptr<EstimateLSM::SuperVersionIterator>(tree.seek(SKey(a, 12)));
      //       },
      //       std::ref(tree))
      //       .detach();
      // }
    }
    tree.all_flush();
    
    auto end = std::chrono::system_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    start = end;
    logger("flush used time: ", double(dur.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den);
    std::this_thread::sleep_for(std::chrono::milliseconds(5000));
    auto numbers2 = std::vector<int>(numbers.begin(), numbers.begin() + L / 2);
    std::sort(numbers2.begin(), numbers2.end(), comp2);

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
      int output = tree.range_count(SKey(b, 12), SKey(a, 12));
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

      auto iter = std::unique_ptr<EstimateLSM::SuperVersionIterator>(tree.seek(SKey(a, 12)));
      auto check_func = [](uint8_t* a, int goal) {
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
    EstimateLSM tree(1e7, std::unique_ptr<Env>(createDefaultEnv()), std::make_unique<FileName>(0, "/tmp/viscnts/"),
                     std::make_unique<DefaultAllocator>(), SKeyCompFunc);
    int L = 3e7, Q = 1e4;
    std::vector<int> numbers(L);
    auto comp2 = +[](int x, int y) {
      uint8_t a[12], b[12];
      for (int j = 0; j < 12; j++) a[j] = x >> (j % 4) * 8 & 255;
      for (int j = 0; j < 12; j++) b[j] = y >> (j % 4) * 8 & 255;
      return SKeyCompFunc(SKey(a, 12), SKey(b, 12)) < 0;
    };
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
    auto iter = std::unique_ptr<EstimateLSM::SuperVersionIterator>(tree.seek(SKey(a, 12)));
    double ans = 0;
    while(iter->valid()) {
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