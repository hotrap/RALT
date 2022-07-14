#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <thread>

#include "alloc.hpp"
#include "cache.hpp"
#include "common.hpp"
#include "file.hpp"
#include "hash.hpp"
#include "key.hpp"
#include "memtable.hpp"

namespace viscnts_lsm {

template <typename... Args>
void logger(Args&&... a) {
  static std::mutex m_;
  std::unique_lock lck_(m_);
  (std::cerr << ... << a) << std::endl;
}

const static size_t kDataBlockSize = 1 << 16;           // 64 KB
const static size_t kPageSize = 4096;                   // 4 KB
const static size_t kMagicNumber = 0x25a65facc3a23559;  // echo viscnts | sha1sum
const static size_t kMemTable = 1 << 26;                // 64 MB
const static size_t kSSTable = 1 << 26;
const static size_t kRatio = 10;
const static size_t kBatchSize = 1 << 16;
const static size_t kChunkSize = 1 << 13;  // 8 KB
// about kMemTable... on average, we expect the size of index block < kDataBlockSize

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

template <typename Value>
struct BlockValue {
  Value v;
  uint32_t offset;
};

class Chunk {
  LRUHandle* lru_handle_;
  LRUCache* cache_;
  uint8_t* data_;

 public:
  Chunk() {
    lru_handle_ = nullptr;
    cache_ = nullptr;
    data_ = nullptr;
  }
  Chunk(uint32_t file_id, uint32_t offset, LRUCache* cache, BaseAllocator* alloc, RandomAccessFile* file_ptr) : cache_(cache) {
    Slice result;
    size_t key = (size_t)file_id << 32 | offset;
    lru_handle_ = cache_->lookup(Slice(reinterpret_cast<uint8_t*>(&key), sizeof(size_t)), Hash8(reinterpret_cast<char*>(&key)));
    if (!lru_handle_->valid.load(std::memory_order_relaxed)) {
      auto ptr = alloc->allocate(kChunkSize);
      auto err = file_ptr->read(offset, kChunkSize, ptr, result);
      if (err) {
        logger("error in Chunk::Chunk(): ", err);
        data_ = nullptr;
        return;
      }
      if (!lru_handle_->valid.exchange(true, std::memory_order_relaxed)) {
        lru_handle_->data = result.data();
        lru_handle_->deleter = alloc;
      } else
        alloc->release(ptr);
      data_ = lru_handle_->data;
      assert(data_ != nullptr);
      // if (data_ != ptr) alloc_->release(ptr); // we don't use mmap
    } else {
      data_ = lru_handle_->data;
      assert(data_ != nullptr);
    }
  }
  Chunk(const Chunk&) = delete;
  Chunk& operator=(const Chunk&) = delete;
  Chunk(Chunk&& c) { (*this) = std::move(c); }
  Chunk& operator=(Chunk&& c) {
    if (lru_handle_ && cache_) cache_->release(lru_handle_);
    lru_handle_ = c.lru_handle_;
    data_ = c.data_;
    cache_ = c.cache_;
    c.lru_handle_ = nullptr;
    return (*this);
  }
  ~Chunk() {
    if (lru_handle_) cache_->release(lru_handle_);
  }
  uint8_t* data(uint32_t offset = 0) const { return data_ + offset; }
};

template <typename Value>
class FileBlock {     // process blocks in a file
  uint32_t file_id_;  // file id
  FileBlockHandle handle_;
  LRUCache* cache_;
  BaseAllocator* alloc_;
  RandomAccessFile* file_ptr_;

  // [keys] [(offset, value) pairs]
  // serveral kinds of file blocks
  // data block and index block
  // their values are different: data block stores SValue, index block stores FileBlockHandle.
  // attention: the size of value is fixed.

  uint32_t offset_index_;

  uint32_t lst_value_id_, lst_key_id_;

  // last key and value chunk
  // to speed up sequential accesses.

  constexpr static auto kValuePerChunk = kChunkSize / sizeof(Value);

  std::pair<uint32_t, uint32_t> _key_offset(uint32_t offset) {
    assert(offset < handle_.size);
    // ensure no key cross two chunks.
    // offset is absolute
    return {offset / kChunkSize, offset % kChunkSize};
  }

  std::pair<uint32_t, uint32_t> _value_offset(uint32_t id) {
    assert(id < handle_.counts);
    // calcuate the absolute offset
    const auto offset = offset_index_ + id * sizeof(Value);
    // calculate the chunk id
    // because of alignment
    const auto the_chunk = offset / kValuePerChunk;
    return {the_chunk * kChunkSize, offset + the_chunk * (sizeof(Value) - kChunkSize % sizeof(Value))};
  }

 public:
  class Iterator {
   public:
    Iterator(FileBlock<Value> block) : block_(block), current_value_id_(-1), current_key_id_(-1), offset_(0) {}
    Iterator(FileBlock<Value> block, uint32_t offset) : block_(block), current_value_id_(-1), current_key_id_(-1), offset_(offset) {}
    Iterator(const Iterator& it) {
      block_ = it.block_;
      offset_ = it.offset_;
      init();
    }
    Iterator& operator=(const Iterator& it) {
      block_ = it.block_;
      offset_ = it.offset_;
      init();
      return (*this);
    }
    auto seek_and_read(uint32_t id) {
      SKey _key;
      Value _value;
      auto [chunk_id, offset] = block_._value_offset(id);
      if (current_value_id_ != chunk_id) currenct_value_chunk_ = block_.acquire(current_value_id_ = chunk_id);
      block_.read_value_offset(offset % kChunkSize, currenct_value_chunk_, _value);
      // read key
      auto [chunk_key_id, key_offset] = block_._key_offset(_value.offset);
      if (current_key_id_ != chunk_key_id) current_key_chunk_ = block_.acquire(current_key_id_ = chunk_key_id);
      block_.read_key_offset(key_offset % kChunkSize, current_key_chunk_, _key);
      offset_ = offset;
      return std::make_pair(_key, _value);
    }
    void next() {
      offset_ += sizeof(Value);
      if (offset_ + sizeof(Value) > kChunkSize) {
        offset_ += kChunkSize - offset_ % kChunkSize, current_value_id_ += 1;
        currenct_value_chunk_ = block_.acquire(current_value_id_);
      }
    }
    auto read() {
      SKey _key;
      Value _value;
      block_.read_value_offset(offset_ % kChunkSize, currenct_value_chunk_, _value);
      auto [chunk_key_id, key_offset] = block_._key_offset(_value.offset);
      if (current_key_id_ != chunk_key_id) current_key_chunk_ = block_.acquire(current_key_id_ = chunk_key_id);
      block_.read_key_offset(key_offset % kChunkSize, current_key_chunk_, _key);
      return std::make_pair(_key, _value);
    }
    void init() {
      if (valid()) seek_and_read(offset_);
    }
    bool valid() { return offset_ < block_.handle_.offset + block_.handle_.size; }

   private:
    FileBlock<Value> block_;
    Chunk currenct_value_chunk_, current_key_chunk_;
    uint32_t current_value_id_, current_key_id_, offset_;
  };

  FileBlock() {
    file_id_ = 0;
    cache_ = nullptr;
    alloc_ = nullptr;
    file_ptr_ = nullptr;
    lst_value_id_ = lst_key_id_ = -1;
    offset_index_ = 0;
  }
  explicit FileBlock(uint32_t file_id, FileBlockHandle handle, LRUCache* cache, BaseAllocator* alloc, RandomAccessFile* file_ptr)
      : file_id_(file_id), handle_(handle), cache_(cache), alloc_(alloc), file_ptr_(file_ptr) {
    logger("init fileblock");
    lst_value_id_ = lst_key_id_ = -1;
    offset_index_ = handle_.offset + handle_.size - handle_.counts / kValuePerChunk * kChunkSize - handle_.counts % kValuePerChunk * sizeof(Value);
  }

  explicit FileBlock(uint32_t file_id, FileBlockHandle handle, LRUCache* cache, BaseAllocator* alloc, RandomAccessFile* file_ptr,
                     uint32_t offset_index)
      : file_id_(file_id), handle_(handle), cache_(cache), alloc_(alloc), file_ptr_(file_ptr) {
    logger("init fileblock by specified offset_index");
    lst_value_id_ = lst_key_id_ = -1;
    offset_index_ = offset_index;
  }

  Chunk acquire(size_t id) { return Chunk(file_id_, id * kChunkSize, cache_, alloc_, file_ptr_); }

  // assume that input is valid, read_key_offset and read_value_offset
  template <typename Key>
  void read_key_offset(uint32_t offset, const Chunk& c, Key& result) const {
    assert(offset < kChunkSize);
    result.read(c.data(offset));
  }

  void read_value_offset(uint32_t offset, const Chunk& c, Value& result) const {
    assert(offset < kChunkSize);
    result = *reinterpret_cast<Value*>(c.data(offset));
  }

  uint32_t end() { return handle_.offset + handle_.size; }
  uint32_t begin() { return offset_index_; }

  ssize_t exists(const SKey& key) {
    uint32_t l = 0, r = handle_.counts - 1;
    Iterator it = Iterator(*this);
    while (l < r) {
      auto mid = (l + r) >> 1;
      // compare two keys
      auto cmp = SKeyComparator()(it.seek_and_read(mid).first, key);
      if (!cmp) return 1;
      if (cmp < 0)
        l = mid;
      else
        r = mid - 1;
    }
    // concurrency bugs?
    // consider this later
    // lst_value_id_ = current_value_id;
    // lst_key_id_ = current_key_id;
    return 0;
  }

  // ensure key is in range!!
  Value upper(const SKey& key) {
    uint32_t l = 0, r = handle_.counts - 1;
    Value ret;
    Iterator it = Iterator(*this);
    while (l < r) {
      auto mid = (l + r) >> 1;
      auto kv = it.seek_and_read(mid);
      // compare two keys
      if (key <= kv.first) {
        l = mid, ret = kv.second;
      } else
        r = mid - 1;
    }
    return ret;
  }

  // ssize_t exists_in_lst(const SKey& key) {
  //   if (lst_value_id_ == -1) return 0;
  //   uint32_t l = 0, r = kChunkSize / sizeof(Value), current_value_id = lst_value_id_, current_key_id = lst_key_id_;
  //   Chunk currenct_value_chunk = acquire(lst_value_id_ * kChunkSize), current_key_chunk = acquire(lst_key_id_ * kChunkSize);
  //   while (l < r) {
  //     auto mid = (l + r) >> 1;
  //     SKey _key;
  //     Value _value;
  //     auto [chunk_id, offset] = _value_offset(mid + lst_value_id_ * kValuePerChunk);
  //     if (current_value_id != chunk_id) currenct_value_chunk = acquire(current_value_id = chunk_id);
  //     read_value_offset(offset, currenct_value_chunk, _value);
  //     // read key
  //     auto [chunk_key_id, key_offset] = _key_offset(_value.offset);
  //     if (current_key_id != chunk_key_id) current_key_chunk = acquire(current_key_id = chunk_key_id);
  //     read_key_offset(key_offset, current_key_chunk, _key);
  //     auto cmp = SKeyComparator()(_key, key);
  //     if (!cmp) return 1;
  //     if (cmp < 0)
  //       l = mid;
  //     else
  //       r = mid - 1;
  //   }
  //   return 0;
  // }

  auto sub_fileblock(uint32_t offset_l, uint32_t offset_r) {
    auto counts_l = (offset_l - offset_index_) / kChunkSize * kValuePerChunk + (offset_l - offset_index_) % kChunkSize / sizeof(Value);
    auto counts_r = (offset_r - offset_index_) / kChunkSize * kValuePerChunk + (offset_r - offset_index_) % kChunkSize / sizeof(Value);
    logger("sub fileblock: ", counts_r - counts_l, ", ", handle_.counts);
    return FileBlock<Value>(file_id_, FileBlockHandle(handle_.offset, offset_r - handle_.offset, counts_r - counts_l), cache_, alloc_, file_ptr_,
                            offset_l);
  }
};

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
  FileBlock<BlockValue<uint32_t>> index_block_;
  FileBlock<BlockValue<SValue>> data_block_;
  // LRUCache pointer reference to the one in VisCnts
  LRUCache* cache_;
  BaseAllocator* alloc_;
  // last Block range
  // initially block_range_ is SKey(nullptr, 1e9), SKey(nullptr, 1e9)
  std::pair<IndSKey, IndSKey> block_range_;

 public:
  ImmutableFile(uint32_t file_id, uint32_t size, std::unique_ptr<RandomAccessFile>&& file_ptr, LRUCache* cache, BaseAllocator* alloc,
                std::pair<IndSKey, IndSKey> range)
      : file_id_(file_id),
        size_(size),
        range_(range),
        file_ptr_(std::move(file_ptr)),
        cache_(cache),
        alloc_(alloc),
        block_range_(IndSKey(nullptr, 1e18), IndSKey(nullptr, 1e18)) {
    // read index block
    Slice result(nullptr, 0);
    FileBlockHandle index_bh, data_bh;
    size_t mgn;
    auto ret = file_ptr_->read(size_ - sizeof(size_t), sizeof(size_t), (uint8_t*)(&mgn), result);
    assert(ret >= 0);
    logger("file size: ", size);
    logger("magic number: ", mgn);
    assert(mgn == kMagicNumber);
    ret = file_ptr_->read(size_ - sizeof(size_t) - sizeof(FileBlockHandle) * 2, sizeof(FileBlockHandle), (uint8_t*)(&index_bh), result);
    assert(ret >= 0);
    // if (result.data() != (uint8_t*)(&index_bh)) index_bh = *(FileBlockHandle*)(result.data()); // we don't use mmap since mmap is shit due to TLB
    // flushes...
    ret = file_ptr_->read(size_ - sizeof(size_t) - sizeof(FileBlockHandle), sizeof(FileBlockHandle), (uint8_t*)(&data_bh), result);
    assert(ret >= 0);
    printf("[file_size=%d, counts=%d, offset=%d, size=%d]\n", size, index_bh.counts, index_bh.offset, index_bh.size);
    index_block_ = FileBlock<BlockValue<uint32_t>>(file_id, index_bh, cache, alloc, file_ptr_.get());
    data_block_ = FileBlock<BlockValue<SValue>>(file_id, data_bh, cache, alloc, file_ptr_.get());
  }

  // ensure key is in range!!
  // I don't want to check range here...
  ssize_t exists(const SKey& key) {
    auto handle = index_block_.upper(key);
    return data_block_.sub_fileblock(handle.offset, handle.offset + kChunkSize).exists(key);
  }
  bool in_range(const SKey& key) {
    auto& [l, r] = range_;
    return l <= key && key <= r;
  }

  std::pair<SKey, SKey> range() { return {range_.first.ref(), range_.second.ref()}; }

  uint32_t size() { return size_; }

  bool range_overlap(const std::pair<SKey, SKey>& range) {
    auto [l, r] = range_;
    return l <= range.second && range.first <= r;
  }

  auto sub_datablock(uint32_t offset_l, uint32_t offset_r) { return data_block_.sub_fileblock(offset_l, offset_r); }
  auto sub_datablock(uint32_t offset_l) { return data_block_.sub_fileblock(offset_l, data_block_.end()); }
  auto sub_datablock() { return data_block_.sub_fileblock(data_block_.begin(), data_block_.end()); }

 private:
  bool in_block_range(const SKey& key) {
    auto& [l, r] = block_range_;
    return l <= key && key <= r;
  }
};

// Semantically there can only be one AppendFile at the same time
class WriteBatch {
  std::unique_ptr<AppendFile> file_ptr_;
  const size_t buffer_size_;
  size_t used_size_;
  uint8_t* data_;

 public:
  explicit WriteBatch(std::unique_ptr<AppendFile>&& file, size_t size) : file_ptr_(std::move(file)), buffer_size_(size), used_size_(0) {
    data_ = new uint8_t[size];
  }
  ~WriteBatch() { delete[] data_; }

  template <typename T>
  void append(const T& kv) {
    // return;  // debug
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
    // return;  // debug
    if (used_size_ + sizeof(T) > buffer_size_) {
      T y = x;
      append(Slice(reinterpret_cast<uint8_t*>(&y), sizeof(T)));
    } else {
      *reinterpret_cast<T*>(data_ + used_size_) = x;
      used_size_ += sizeof(T);
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
    // return;  // debug
    file_ptr_->write(Slice(data_, used_size_));
    used_size_ = 0;
  }
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

class SeqIteratorSet : public SeqIterator {
  std::vector<std::unique_ptr<SeqIterator>> iters_;
  std::vector<std::pair<SKey, SValue>> tmp_kv_;

 public:
  SeqIteratorSet() = default;
  SeqIteratorSet(const SeqIteratorSet& ss) { (*this) = ss; }
  SeqIteratorSet& operator=(const SeqIteratorSet& ss) {
    tmp_kv_ = ss.tmp_kv_;
    for (auto& a : ss.iters_) iters_.push_back(std::unique_ptr<SeqIterator>(a->copy()));
    return (*this);
  }
  bool valid() override {
    for (auto& a : iters_)
      if (a->valid()) return true;
    return false;
  }
  void next() override {
    auto h = 0;
    SKey mn = tmp_kv_[0].first;
    for (uint32_t i = 1; i < tmp_kv_.size(); ++i)
      if (tmp_kv_[i].first <= mn) mn = tmp_kv_[h = i].first;
    iters_[h]->next();
    if (iters_[h]->valid())
      tmp_kv_[h] = iters_[h]->read();
    else
      tmp_kv_[h] = {SKey(nullptr, 1e18), SValue()};
  }
  std::pair<SKey, SValue> read() override {
    auto h = 0;
    SKey mn = tmp_kv_[0].first;
    for (uint32_t i = 1; i < tmp_kv_.size(); ++i)
      if (tmp_kv_[i].first <= mn) mn = tmp_kv_[h = i].first;
    return tmp_kv_[h];
  }
  void push(std::unique_ptr<SeqIterator>&& new_iter) {
    if (!new_iter->valid()) return;
    iters_.push_back(std::move(new_iter));
    tmp_kv_.push_back(new_iter->read());
  }
  SeqIterator* copy() override { return new SeqIteratorSet(*this); }
};

class MemTableIterator : public SeqIterator {
 public:
  MemTableIterator(MemTable::Node* node) : node_(node) {}
  bool valid() override { return node_ != nullptr; }
  // requirement: node_ is valid
  void next() override { node_ = node_->noBarrierGetNext(0); }
  std::pair<SKey, SValue> read() override { return {node_->key, node_->value}; }
  SeqIterator* copy() override { return new MemTableIterator(*this); }

 private:
  MemTable::Node* node_;
};

// This iterator assumes that (forgotten)
// wofo!!!
class SSTIterator : public SeqIterator {
 public:
  SSTIterator(ImmutableFile* file)
      : file_(file), kv_valid_(false), file_block_iter_(file->sub_datablock()) {
    file_block_iter_.init();
  }
  bool valid() override { return file_block_iter_.valid(); }
  // ensure it's valid.
  void next() override { file_block_iter_.next(), kv_valid_ = false; }
  // remember, SKey is a reference to file block
  std::pair<SKey, SValue> read() override {
    if (kv_valid_) {
      return kvpair_;
    }
    kv_valid_ = true;
    auto pair = file_block_iter_.read();
    return kvpair_ = {pair.first, pair.second.v};
  }
  SeqIterator* copy() override { return new SSTIterator(*this); }

 private:
  ImmutableFile* file_;
  bool kv_valid_;
  std::pair<SKey, SValue> kvpair_;
  FileBlock<BlockValue<SValue>>::Iterator file_block_iter_;
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
    if (len && (now_offset + len - 1) / kChunkSize != now_offset / kChunkSize) {
      file_->fill(0, kChunkSize - now_offset % kChunkSize);
      now_offset += kChunkSize - now_offset % kChunkSize;
    }
  }

  uint32_t _calc_offset(uint32_t id) {
    return id / (kChunkSize / sizeof(BlockValue<SValue>)) * kChunkSize + id % (kChunkSize / sizeof(BlockValue<SValue>)) * sizeof(BlockValue<SValue>);
  }

 public:
  SSTBuilder(std::unique_ptr<WriteBatch>&& file = nullptr) : file_(std::move(file)) {}
  template <typename T>
  void append(const std::pair<SKey, T>& kv) {
    _append_align(kv.first.size());
    counts++;
    if (counts == kChunkSize) {
      index.emplace_back(kv.first, _calc_offset(offsets.size() + 1));
      counts = 0;
    } else
      lst_key = kv.first;
    offsets.push_back({kv.second, now_offset});
    now_offset += kv.first.size();
    _write_key(kv.first);
  }
  void make_index() {
    if (counts) {
      index.emplace_back(lst_key, _calc_offset(offsets.size()));
      counts = 0;
    }
    _align();
    auto data_index_offset = now_offset;
    for (const auto& a : offsets) {
      _append_align(sizeof(decltype(a)));
      file_->append_other(a);
      now_offset += sizeof(decltype(a));
    }
    auto data_bh = FileBlockHandle(lst_offset, now_offset - lst_offset, offsets.size());
    _align();
    lst_offset = now_offset;
    std::vector<BlockValue<uint32_t>> v;
    for (const auto& a : index) {
      _append_align(a.first.size());
      _write_key(a.first);
      v.push_back({a.second + data_index_offset, now_offset});
      now_offset += a.first.size();
    }
    _align();
    for (const auto& a : v) {
      _append_align(sizeof(decltype(a)));
      file_->append_other(a);
      now_offset += sizeof(decltype(a));
    }
    file_->append_other(FileBlockHandle(lst_offset, now_offset - lst_offset, index.size()));  // write offset of index block
    file_->append_other(data_bh);
    now_offset += sizeof(FileBlockHandle) * 2;
  }
  template <typename T>
  void append(const std::pair<IndSKey, T>& kv) {
    append(std::make_pair(kv.first.ref(), kv.second));
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
    index.clear();
    offsets.clear();
  }
  void new_file(std::unique_ptr<WriteBatch>&& file) { file_ = std::move(file); }

  size_t size() { return now_offset; }

  std::pair<IndSKey, IndSKey> range() {
    if (index.size())
      return {index[0].first, index.back().first};
    else
      return {};
  }

 private:
  uint32_t now_offset, lst_offset, counts;
  std::vector<std::pair<IndSKey, uint32_t>> index;
  std::vector<BlockValue<SValue>> offsets;
  IndSKey lst_key;
  template <typename T>
  void _write_key(const T& a) {
    file_->append_other(a.len());
    file_->append(a);
  }
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

  void begin_new_file() {
    builder_.reset();
    auto [filename, id] = files_->next_pair();
    builder_.new_file(std::make_unique<WriteBatch>(std::unique_ptr<AppendFile>(env_->openAppFile(filename)), kBatchSize));
    vec_newfiles_.emplace_back();
    vec_newfiles_.back().filename = filename;
    vec_newfiles_.back().file_id = id;
  }

  void end_new_file() {
    builder_.make_index();
    builder_.finish();
    vec_newfiles_.back().size = builder_.size();
    vec_newfiles_.back().range = builder_.range();
  }

 public:
  struct NewFileData {
    std::string filename;
    size_t file_id;
    size_t size;
    std::pair<IndSKey, IndSKey> range;
  };
  Compaction(FileName* files, Env* env) : files_(files), env_(env), flag_(false), rndgen_(std::random_device()()) {}
  void compact_begin() {
    vec_newfiles_.clear();
    begin_new_file();
    flag_ = false;
  }
  template <typename TIter, typename UIter, std::enable_if_t<std::is_base_of_v<SeqIterator, TIter>, bool> = true,
            std::enable_if_t<std::is_base_of_v<SeqIterator, UIter>, bool> = true>
  void compact(TIter left, UIter right) {
    // compact from two different iterators
    // iterators can be MemTable & SST, or SST & SST, or MemTable & MemTable ?
    // return;  // debug
    auto L = left.valid() ? left.read() : std::pair<SKey, SValue>();
    auto R = right.valid() ? right.read() : std::pair<SKey, SValue>();

    while (left.valid() || right.valid()) {
      if (!right.valid() || (left.valid() && L.first <= R.first)) {
        if (flag_ && lst_value_.first == L.first) {
          lst_value_.second += L.second;
        } else {
          if (flag_) builder_.append(lst_value_);
          lst_value_ = L, flag_ = true;
        }
        left.next();
        if (left.valid()) L = left.read();
      } else {
        if (flag_ && lst_value_.first == R.first) {
          lst_value_.second += R.second;
        } else {
          if (flag_) builder_.append(lst_value_);
          lst_value_ = R, flag_ = true;
        }
        right.next();
        if (right.valid()) R = right.read();
      }
      _divide_file();
    }
  }

  // for the last level...
  template <typename TIter, typename UIter, std::enable_if_t<std::is_base_of_v<SeqIterator, TIter>, bool> = true,
            std::enable_if_t<std::is_base_of_v<SeqIterator, UIter>, bool> = true>
  double compact_last_level(TIter left, UIter right) {
    // compact from two different iterators
    // iterators can be MemTable & SST, or SST & SST, or MemTable & MemTable ?
    // return 0;  // debug
    std::pair<SKey, SValue> L = left.valid() ? left.read() : std::pair<SKey, SValue>();
    std::pair<SKey, SValue> R = right.valid() ? right.read() : std::pair<SKey, SValue>();
    double change = 0;

    while (left.valid() || right.valid()) {
      if (!right.valid() || (left.valid() && L.first <= R.first)) {
        if (flag_ && lst_value_.first == L.first) {
          lst_value_.second += L.second;
        } else {
          if (flag_) builder_.append(lst_value_), change += _calc_decay_value(lst_value_);
          lst_value_ = L, flag_ = true;
        }
        left.next();
        if (left.valid()) L = left.read();
      } else {
        if (flag_ && lst_value_.first == R.first) {
          change -= _calc_decay_value(R);
          lst_value_.second += R.second;
        } else {
          if (flag_) builder_.append(lst_value_), change += _calc_decay_value(lst_value_);
          lst_value_ = R, flag_ = true;
        }
        right.next();
        if (right.valid()) R = right.read();
      }
      _divide_file();
    }
    return change;
  }

  std::vector<NewFileData> compact_end() {
    if (flag_) builder_.append(lst_value_);
    // TODO: we can store shorter key in viscnts by checking the LCP
    // Now we begin to write index block
    end_new_file();
    return vec_newfiles_;
  }

  void flush_begin() {
    vec_newfiles_.clear();
    begin_new_file();
    flag_ = false;
  }

  template <typename TIter, std::enable_if_t<std::is_base_of_v<SeqIterator, TIter>, bool> = true>
  void flush(TIter left) {
    while (left.valid()) {
      // printf("A");
      auto L = left.read();
      if (flag_ && lst_value_.first == L.first) {
        lst_value_.second += L.second;
      } else {
        if (flag_) builder_.append(lst_value_);
        lst_value_ = L, flag_ = true;
      }
      left.next();
    }
  }

  std::vector<NewFileData> flush_end() {
    if (flag_) builder_.append(lst_value_);
    end_new_file();
    return vec_newfiles_;
  }

  std::pair<std::vector<NewFileData>, double> decay_first(SeqIteratorSet&& iters) {
    double real_size_ = 0;  // re-calculate size
    vec_newfiles_.clear();
    begin_new_file();
    flag_ = false;
    while (iters.valid()) {
      auto L = iters.read();
      if (flag_ && lst_value_.first == L.first) {
        lst_value_.second += L.second;
      } else {
        if (_decay_kv(lst_value_)) {
          real_size_ += _calc_decay_value(lst_value_);
          builder_.append(lst_value_);
          _divide_file();
        }
        lst_value_ = L, flag_ = true;
      }
      iters.next();
    }
    if (flag_) builder_.append(lst_value_);
    end_new_file();
    return {vec_newfiles_, real_size_};
  }

  std::pair<std::vector<NewFileData>, double> decay_second(SeqIteratorSet&& iters) {
    // maybe we can...
    double real_size_ = 0;  // re-calculate size
    vec_newfiles_.clear();
    begin_new_file();
    flag_ = false;

    while (iters.valid()) {
      lst_value_ = iters.read();
      iters.next();
      while (iters.valid() && iters.read().first == lst_value_.first) {
        auto c = iters.read();
        lst_value_.first = c.first;
        lst_value_.second += c.second;
        iters.next();
      }
      if (lst_value_.second.counts > 1) {
        lst_value_.second.counts *= 0.5;
        real_size_ += _calc_decay_value(lst_value_);
        builder_.append(lst_value_);
        _divide_file();
      } else {
        auto now_iter = std::unique_ptr<SeqIterator>(iters.copy());
        size_t num = 1, keep = 0;
        while (now_iter->valid()) {
          auto L = now_iter->read();
          now_iter->next();
          while (now_iter->valid() && iters.read().first == L.first) {
            auto c = now_iter->read();
            L.first = c.first;
            L.second += c.second;
            now_iter->next();
          }
          if (L.second.counts != lst_value_.second.counts) break;
          num++;
        }
        keep = num * decay_prob_;
        while (iters.valid()) {
          auto L = iters.read();
          iters.next();
          while (iters.valid() && iters.read().first == L.first) {
            auto c = iters.read();
            L.first = c.first;
            L.second += c.second;
            iters.next();
          }
          if (keep) {
            L.second.counts = 1;
            real_size_ += _calc_decay_value(L);
            builder_.append(L);
            _divide_file();
            keep--;
          } else {
            std::uniform_real_distribution<> dis(0, 1.);
            if (dis(rndgen_) < keep) {
              L.second.counts = 1;
              real_size_ += _calc_decay_value(L);
              builder_.append(L);
              _divide_file();
            }
          }
        }
      }
    }
    end_new_file();
    return {vec_newfiles_, real_size_};
  }

 private:
  std::mt19937 rndgen_;
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
    if (builder_.size() > kMemTable) {
      end_new_file();
      begin_new_file();
    }
  }
  std::vector<NewFileData> vec_newfiles_;
};

class LSMTree {
  class MemTables {
    struct Node {
      MemTable* mem;
      std::atomic<uint32_t> refcounts;
      std::atomic<bool> compacting_complete;
      Node() : mem(new MemTable), refcounts(0), compacting_complete(false) {}
      ~Node() { delete mem; }
    };
    std::mutex m_;

   public:
    Node nw_, od_;
    bool flag;
    bool compacting;
    MemTables() : flag(false), compacting(false) {}
    void append(const std::pair<SKey, SValue>& kv) {
      std::unique_lock<std::mutex> lck_(m_);
      if (compacting) {
        // printf("B");
        // logger(flag);
        flag ? od_.mem->append(kv.first, kv.second) : nw_.mem->append(kv.first, kv.second);
      } else {
        // logger("n", flag);
        // printf("C");
        compacting |= flag ? nw_.mem->size() >= kMemTable : od_.mem->size() >= kMemTable;
        // compacting old
        flag ? nw_.mem->append(kv.first, kv.second) : od_.mem->append(kv.first, kv.second);
        // swapping od and nw is done in compacting thread
        // as expected, the thread will wait until od.refcounts equals to zero.
      }
    }
    bool exists(const SKey& key) {
      (void)key;
      // TODO: fix potential bugs
      // debug
      // if (!od_.compacting_complete.load(std::memory_order_relaxed)) {
      //   od_.refcounts += 1;
      //   bool result = od_.mem->exists(key);
      //   od_.refcounts -= 1;
      //   if (result) return true;
      // }
      // if (!nw_.compacting_complete.load(std::memory_order_relaxed)) {
      //   nw_.refcounts += 1;
      //   bool result = nw_.mem->exists(key);
      //   nw_.refcounts -= 1;
      //   if (result) return true;
      // }
      return false;
    }
    std::mutex& get_mutex() { return m_; }
    void clear() {
      std::unique_lock<std::mutex> lck_(m_);
      flag ? nw_.mem->release() : od_.mem->release();
    }
  };

  class Immutables {
    std::mutex m_;

   public:
    struct Node {
      std::unique_ptr<ImmutableFile> file;
      std::atomic<uint32_t> refcounts;
      std::atomic<Node*> next;
      Node(std::unique_ptr<ImmutableFile>&& _file) : file(std::move(_file)), refcounts(0), next(nullptr) {}
    };
    Immutables() : head_(new Node(nullptr)) {}
    ~Immutables() {
      for (auto a = head_; a;) {
        auto b = a->next.load(std::memory_order_relaxed);
        delete a;
        a = b;
      }
    }
    Immutables(Immutables&& im) {
      head_ = im.head_;
      im.head_ = nullptr;
    }
    void ref(Node* node) { node->refcounts++; }
    void unref(Node* node) {
      if (--node->refcounts) {
        std::unique_lock<std::mutex> lck_(m_);
        delete node;
      }
    }
    void insert(std::vector<std::unique_ptr<ImmutableFile>>&& files, Node* left, Node* right) {
      std::unique_lock<std::mutex> lck_(m_);
      assert(head_ != nullptr);
      if (!head_->next) {
        std::reverse(files.begin(), files.end());
        for (auto& a : files) {
          logger("[insert file]");
          assert(a.get() != nullptr);
          auto node = new Node(std::move(a));
          node->next.store(head_->next, std::memory_order_relaxed);
          node->refcounts.fetch_add(1, std::memory_order_relaxed);
          head_->next = node;
        }

      } else {
        std::reverse(files.begin(), files.end());
        auto rright = right;
        for (auto& a : files) {
          assert(a.get() != nullptr);
          auto node = new Node(std::move(a));
          node->refcounts++;
          node->next.store(right, std::memory_order_relaxed);
          right = node;
        }
        // release old files
        for (auto a = left->next.load(std::memory_order_relaxed); a != rright;) {
          auto b = a->next.load(std::memory_order_relaxed);
          unref(a);
          a = b;
        }
        left->next.store(right, std::memory_order_release);
      }
    }

    void insert(std::unique_ptr<ImmutableFile>&& file, Node* left, Node* right) {
      std::vector<std::remove_reference_t<decltype(file)>> vec;
      vec.push_back(std::move(file));
      insert(std::move(vec), left, right);
    }

    Node* head() { return head_; }

    bool exists(const SKey& key) {
      // TODO: cache the file ranges
      for (Node* a = head_->next.load(std::memory_order_relaxed); a; a = a->next.load(std::memory_order_relaxed)) {
        if (a->file->exists(key)) return true;
      }
      return false;
    }

    size_t size() {
      size_t ret = 0;
      for (Node* a = head_->next.load(std::memory_order_relaxed); a; a = a->next.load(std::memory_order_relaxed)) {
        assert(a->file.get() != nullptr);
        ret += a->file->size();
      }
      return ret;
    }

    void clear() {
      for (Node* a = head_->next.load(std::memory_order_relaxed); a;) {
        auto b = a->next.load(std::memory_order_relaxed);
        unref(a);
        a = b;
      }
      head_->next = nullptr;
    }

   private:
    Node* head_;
  };
  class ImmutablesIterator : public SeqIterator {
    Immutables::Node* now_;
    SSTIterator iter_;

   public:
    ImmutablesIterator(Immutables::Node* begin) : now_(begin), iter_(begin->file.get()) {}
    bool valid() override { return now_ != nullptr; }
    void next() override {
      iter_.next();
      while (now_ && !iter_.valid()) {
        now_ = now_->next;
        if (!now_) return;
        iter_ = SSTIterator(now_->file.get());
      }
    }
    std::pair<SKey, SValue> read() override { return iter_.read(); }
    SeqIterator* copy() override { return new ImmutablesIterator(*this); }
  };
  std::vector<Immutables> tree_;
  MemTables mem_;
  std::unique_ptr<LRUCache> cache_;
  double estimated_size_;
  double delta_;
  std::unique_ptr<Env> env_;
  std::unique_ptr<FileName> filename_;
  std::unique_ptr<BaseAllocator> file_alloc_;
  std::thread compact_thread_;

  void _compact_result_insert(std::vector<Compaction::NewFileData>&& vec, Immutables::Node* left, Immutables::Node* right, int level) {
    std::vector<std::unique_ptr<ImmutableFile>> files;
    for (auto a : vec) {
      auto rafile = env_->openRAFile(a.filename);
      assert(rafile != nullptr);
      auto file =
          std::make_unique<ImmutableFile>(a.file_id, a.size, std::unique_ptr<RandomAccessFile>(rafile), cache_.get(), file_alloc_.get(), a.range);
      assert(file.get() != nullptr);
      files.push_back(std::move(file));
    }
    tree_[level].insert(std::move(files), left, right);
  }

  template <typename TableType, typename TableIterType>
  void _compact_tables(const TableType& table, const TableIterType& iter, uint32_t level) {
    logger("_compact_tables");
    Immutables::Node *left = nullptr, *right = nullptr;
    assert(tree_[level].head() != nullptr);
    for (auto a = tree_[level].head(); a; a = a->next.load(std::memory_order_relaxed)) {
      if (!a->file || !a->file->range_overlap(table->range())) {
        if (!left)
          left = a;
        else
          right = a;
      }
    }
    assert(left != nullptr);
    Compaction worker(filename_.get(), env_.get());
    if (!left || left->next.load(std::memory_order_relaxed) == right) {
      logger("flush");
      worker.flush_begin();
      worker.flush(iter);
      _compact_result_insert(worker.flush_end(), left, right, level);
    } else {
      logger("merge with sst");
      // return;  // debug
      worker.compact_begin();
      for (auto p = left->next.load(std::memory_order_relaxed); p != right; p = p->next.load(std::memory_order_relaxed)) {
        if (level == tree_.size() - 1)  // for the last level, we update the estimated size of viscnts.
          estimated_size_ += worker.compact_last_level(iter, SSTIterator(p->file.get()));
        else
          worker.compact(iter, SSTIterator(p->file.get()));
      }
      _compact_result_insert(worker.compact_end(), left, right, level);
    }
  }

  bool _check_decay() { return estimated_size_ > delta_; }

 public:
  LSMTree(std::unique_ptr<LRUCache>&& cache, double delta, std::unique_ptr<Env>&& env, std::unique_ptr<FileName>&& filename,
          std::unique_ptr<BaseAllocator>&& alloc)
      : cache_(std::move(cache)),
        estimated_size_(0),
        delta_(delta),
        env_(std::move(env)),
        filename_(std::move(filename)),
        file_alloc_(std::move(alloc)) {
    compact_thread_ = std::thread([this]() { compact_thread(); });
    compact_thread_.detach();
  }
  void append(const std::pair<SKey, SValue>& kv) { mem_.append(kv); }
  bool exists(const SKey& key) {
    if (mem_.exists(key)) return true;
    for (auto& a : tree_)
      if (a.exists(key)) return true;
    return false;
  }
  void decay_all() {
    SeqIteratorSet iter_set;
    iter_set.push(std::make_unique<MemTableIterator>(mem_.flag ? mem_.nw_.mem->head() : mem_.od_.mem->head()));
    for (auto& a : tree_) iter_set.push(std::make_unique<ImmutablesIterator>(ImmutablesIterator(a.head()->next)));
    Compaction worker(filename_.get(), env_.get());
    auto [vec, sz] = worker.decay_first(std::move(iter_set));  // we use this first;
    estimated_size_ = sz;                                      // this is the exact value of estimated_size_ // will it decay twice in succession?
    // we store the decay result in the last level to decrease write amp.
    _compact_result_insert(std::move(vec), tree_.back().head(), nullptr, tree_.size() - 1);
    // clear all the levels and memtables
    for (uint32_t i = 0; i < tree_.size() - 1; ++i) tree_[i].clear();
    mem_.clear();
  }
  void compact_thread() {
    logger("thread begin");
    // use leveling first...
    // although I think tiering is better.
    // we only create one compact_thread including flushing memtable, compacing, decay.
    using namespace std::chrono_literals;
    while (true) {
      std::this_thread::sleep_for(100ms);
      if (_check_decay()) {
        logger("begin decay");
        decay_all();
        continue;
      }
      if (mem_.compacting) {
        logger("flush memtable");
        if (mem_.od_.compacting_complete || mem_.nw_.compacting_complete) {
          // we must avoid moving nw_ to od_.
          // we use compacting_complete to mark the validality.
          // we use flag to mark which thread is processing now.
          // we use compacting to mark the whether there exists a table that needs a compaction.
          logger("mem_.od_: ", mem_.od_.compacting_complete, ", ", mem_.od_.refcounts);
          logger("mem_.nw_: ", mem_.nw_.compacting_complete, ", ", mem_.nw_.refcounts);
          logger("flag, compacting: ", mem_.flag, ", ", mem_.compacting);
          if (mem_.od_.compacting_complete && !mem_.od_.refcounts) {
            std::unique_lock<std::mutex> lck_(mem_.get_mutex());
            // release is to reset memtable
            mem_.od_.mem->release();
            // mem_.od_.mem = new MemTable();
            // mem_.nw_.mem = new MemTable();
            mem_.od_.compacting_complete = false;
            mem_.compacting = false;
            mem_.flag ^= 1;
          } else if (mem_.nw_.compacting_complete && !mem_.nw_.refcounts) {
            std::unique_lock<std::mutex> lck_(mem_.get_mutex());
            // release is to reset memtable
            mem_.nw_.mem->release();
            // mem_.od_.mem = new MemTable();
            // mem_.nw_.mem = new MemTable();
            mem_.nw_.compacting_complete = false;
            mem_.compacting = false;
            mem_.flag ^= 1;
          }

          logger("change memtable end");
          continue;
        }
        std::unique_lock<std::mutex> lck_(mem_.get_mutex());
        auto& table = mem_.flag ? mem_.nw_.mem : mem_.od_.mem;
        logger("memtable size: ", table->size());
        MemTableIterator iter(table->begin());
        if (tree_.size() == 0) {
          tree_.emplace_back();
        }
        _compact_tables(table, iter, 0);

        mem_.flag ? mem_.nw_.compacting_complete = true : mem_.od_.compacting_complete = true;
        logger("flush memtable end");
        // std::unique_lock<std::mutex> lck_(mem_.get_mutex());
        // mem_.od_ = std::move(mem.nw_)
        // mem_.od_.compacting = false;
      }
      // continue;  // debug
      size_t level_size = kSSTable * 2;
      for (uint32_t i = 0; i < tree_.size(); i++) {
        auto& a = tree_[i];
        if (a.size() > level_size) {
          logger("compact sst of ", i, "-th level");
          // choose one file to flush
          auto lucky_one = a.head()->next.load(std::memory_order_relaxed);
          if (i == tree_.size() - 1) {
            // new level
            tree_.emplace_back();
            tree_.back().insert(std::move(lucky_one->file), nullptr, nullptr);
            tree_[i].unref(lucky_one);
          } else {
            _compact_tables(lucky_one->file, SSTIterator(lucky_one->file.get()), i + 1);
          }
        }
        level_size *= kRatio;
      }
    }
  }
};

// Viscnts, implement lsm tree and other things.
class VisCnts {
  LSMTree tree;

 public:
  VisCnts(const std::string& path, double delta, bool createIfMissing)
      : tree(std::make_unique<LRUCache>(64), delta, std::unique_ptr<Env>(createDefaultEnv()), std::make_unique<FileName>(0, path),
             std::make_unique<DefaultAllocator>()) {}
  void access(const std::pair<SKey, SValue>& kv) { tree.append(kv); }
  bool is_hot(const SKey& key) { return tree.exists(key); }
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
