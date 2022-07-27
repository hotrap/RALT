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

// const static size_t kDataBlockSize = 1 << 16;           // 64 KB
const static size_t kPageSize = 4096;                   // 4 KB
const static size_t kMagicNumber = 0x25a65facc3a23559;  // echo viscnts | sha1sum
const static size_t kMemTable = 1 << 24;                // 1 MB
const static size_t kSSTable = 1 << 24;
const static size_t kRatio = 10;
const static size_t kBatchSize = 1 << 16;
const static size_t kChunkSize = 1 << 16;  // 8 KB
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

// if values are stored with offset then use this as values
template <typename Value>
struct BlockValue {
  Value v;
  uint32_t offset;
};

// manage a temporary chunk (several pages) in SST files
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
      // since this handle is not valid, we read the chunk from file, then try to cache it in memory.
      // if several threads read and cache the same chunk at the same time, then all but one of these threads should release the chunk.
      auto ptr = alloc->allocate(kChunkSize);
      auto err = file_ptr->read(offset, kChunkSize, ptr, result);
      if (err) {
        logger("error in Chunk::Chunk(): ", err);
        exit(-1);
        data_ = nullptr;
        return;
      }
      if (!lru_handle_->valid.exchange(true, std::memory_order_relaxed)) {
        // result.data() == ptr.
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
  Chunk(Chunk&& c) {
    lru_handle_ = c.lru_handle_;
    data_ = c.data_;
    cache_ = c.cache_;
    c.lru_handle_ = nullptr;
    c.data_ = nullptr;
    c.cache_ = nullptr;
  }
  Chunk& operator=(Chunk&& c) {
    if (lru_handle_ && cache_) cache_->release(lru_handle_);
    lru_handle_ = c.lru_handle_;
    data_ = c.data_;
    cache_ = c.cache_;
    c.lru_handle_ = nullptr;
    c.data_ = nullptr;
    c.cache_ = nullptr;
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
  // serveral (now 2) kinds of file blocks
  // data block and index block
  // their values are different: data block stores SValue, index block stores FileBlockHandle.
  // attention: the size of value is fixed.

  uint32_t offset_index_;

  uint32_t lst_value_id_, lst_key_id_;

  // last key and value chunk
  // to speed up sequential accesses.

  constexpr static auto kValuePerChunk = kChunkSize / sizeof(Value);

  std::pair<uint32_t, uint32_t> _key_offset(uint32_t offset) {
    assert(offset < handle_.offset + handle_.size);
    // we must ensure no keys cross two chunks in SSTBuilder.
    // offset is absolute.
    return {offset / kChunkSize, offset % kChunkSize};
  }

  std::pair<uint32_t, uint32_t> _value_offset(uint32_t id) {
    assert(id < handle_.counts);
    // calculate the absolute offset
    // const auto offset = offset_index_ + id * sizeof(Value);
    // calculate the chunk id
    // because of alignment
    // const auto the_offset = offset + id / kValuePerChunk * (kChunkSize - kValuePerChunk * sizeof(Value));
    // const auto the_chunk = offset_index_ / kChunkSize + id / kValuePerChunk;
    const auto rest = (kChunkSize - offset_index_ % kChunkSize) / sizeof(Value);
    const auto the_offset = id < rest ? id * sizeof(Value) + offset_index_
                                      : (id - rest) / kValuePerChunk * kChunkSize + (id - rest) % kValuePerChunk * sizeof(Value) + kChunkSize -
                                            offset_index_ % kChunkSize + offset_index_;
    //   assert(the_offset / kChunkSize == the_chunk);
    return {the_offset / kChunkSize, the_offset % kChunkSize};
  }

 public:
  class Iterator {
   public:
    Iterator(FileBlock<Value> block) : block_(block), current_value_id_(-1), current_key_id_(-1), id_(0) {}
    Iterator(FileBlock<Value> block, uint32_t id) : block_(block), current_value_id_(-1), current_key_id_(-1), id_(id) {}
    Iterator(const Iterator& it) {
      block_ = it.block_;
      offset_ = it.offset_;
      id_ = it.id_;
      current_value_id_ = -1;
      current_key_id_ = -1;
      init();
      logger("new fileblock iterator");
    }
    Iterator& operator=(const Iterator& it) {
      block_ = it.block_;
      offset_ = it.offset_;
      id_ = it.id_;
      current_value_id_ = -1;
      current_key_id_ = -1;
      init();
      return (*this);
    }
    auto seek_and_read(uint32_t id) {
      SKey _key;
      Value _value;
      // find value and offset
      auto [chunk_id, offset] = block_._value_offset(id);
      if (current_value_id_ != chunk_id) currenct_value_chunk_ = block_.acquire(current_value_id_ = chunk_id);
      block_.read_value_offset(offset, currenct_value_chunk_, _value);
      // read key
      auto [chunk_key_id, key_offset] = block_._key_offset(_value.offset);
      if (current_key_id_ != chunk_key_id) current_key_chunk_ = block_.acquire(current_key_id_ = chunk_key_id);
      block_.read_key_offset(key_offset, current_key_chunk_, _key);
      offset_ = offset;
      id_ = id;
      return std::make_pair(_key, _value);
    }
    void next() {
      assert(offset_ % kChunkSize % sizeof(Value) == 0);
      offset_ += sizeof(Value);
      id_++;
      if (offset_ + sizeof(Value) > kChunkSize) {
        offset_ = 0, current_value_id_ += 1;
        currenct_value_chunk_ = block_.acquire(current_value_id_);
      }
    }
    auto read() {
      SKey _key;
      Value _value;
      block_.read_value_offset(offset_, currenct_value_chunk_, _value);
      auto [chunk_key_id, key_offset] = block_._key_offset(_value.offset);
      if (current_key_id_ != chunk_key_id) current_key_chunk_ = block_.acquire(current_key_id_ = chunk_key_id);
      block_.read_key_offset(key_offset, current_key_chunk_, _key);
      return std::make_pair(_key, _value);
    }
    void init() {
      if (valid()) seek_and_read(id_);
    }
    bool valid() { return id_ < block_.handle_.counts; }

   private:
    FileBlock<Value> block_;
    Chunk currenct_value_chunk_, current_key_chunk_;
    uint32_t current_value_id_, current_key_id_, offset_, id_;
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
    // logger("init fileblock");
    lst_value_id_ = lst_key_id_ = -1;
    offset_index_ = handle_.offset + handle_.size - handle_.counts / kValuePerChunk * kChunkSize - handle_.counts % kValuePerChunk * sizeof(Value);
    assert(offset_index_ % kChunkSize == 0);
  }

  explicit FileBlock(uint32_t file_id, FileBlockHandle handle, LRUCache* cache, BaseAllocator* alloc, RandomAccessFile* file_ptr,
                     uint32_t offset_index)
      : file_id_(file_id), handle_(handle), cache_(cache), alloc_(alloc), file_ptr_(file_ptr) {
    // logger("init fileblock by specified offset_index");
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
    int l = 0, r = handle_.counts - 1;
    Iterator it = Iterator(*this);
    while (l <= r) {
      auto mid = (l + r) >> 1;
      // compare two keys
      auto cmp = SKeyComparator()(it.seek_and_read(mid).first, key);
      if (!cmp) return 1;
      if (cmp < 0)
        l = mid + 1;
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
    int l = 0, r = handle_.counts - 1;
    Value ret;
    Iterator it = Iterator(*this);
    while (l <= r) {
      auto mid = (l + r) >> 1;
      auto kv = it.seek_and_read(mid);
      // compare two keys
      if (key <= kv.first) {
        r = mid - 1, ret = kv.second;
      } else
        l = mid + 1;
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
    // logger("sub fileblock: ", counts_r - counts_l, ", ", handle_.counts, ", kVPC: ", kValuePerChunk, ", sz: ", sizeof(Value));
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

 public:
  ImmutableFile(uint32_t file_id, uint32_t size, std::unique_ptr<RandomAccessFile>&& file_ptr, LRUCache* cache, BaseAllocator* alloc,
                std::pair<IndSKey, IndSKey> range)
      : file_id_(file_id), size_(size), range_(range), file_ptr_(std::move(file_ptr)), cache_(cache), alloc_(alloc) {
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
    printf("index_block: [file_size=%d, counts=%d, offset=%d, size=%d]\n", size, index_bh.counts, index_bh.offset, index_bh.size);
    printf("data_block: [file_size=%d, counts=%d, offset=%d, size=%d]\n", size, data_bh.counts, data_bh.offset, data_bh.size);
    index_block_ = FileBlock<BlockValue<uint32_t>>(file_id, index_bh, cache, alloc, file_ptr_.get());
    data_block_ = FileBlock<BlockValue<SValue>>(file_id, data_bh, cache, alloc, file_ptr_.get());
  }

  // ensure key is in range!!
  // I don't want to check range here...
  ssize_t exists(const SKey& key) {
    if (!in_range(key)) return 0;
    auto handle = index_block_.upper(key);
    // return data_block_.exists(key);
    // logger("exists(): handle.v: ", handle.v);

    // offset_l is % kChunkSize = 0.
    auto ret = data_block_.sub_fileblock(handle.v / kChunkSize * kChunkSize, handle.v + sizeof(BlockValue<SValue>)).exists(key);
    assert(ret == data_block_.exists(key));
    return ret;
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

  void remove() {
    auto ret = file_ptr_->remove();
    assert(ret >= 0);
  }

 private:
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
  ~WriteBatch() { delete data_; }

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
    for (auto& a : ss.iters_) iters_.emplace_back(a->copy());
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
  SSTIterator(ImmutableFile* file) : file_(file), kv_valid_(false), file_block_iter_(file->sub_datablock()) { file_block_iter_.init(); }
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
    if (len && (now_offset + len - 1) / kChunkSize != now_offset / kChunkSize) _align();
  }

  template <typename T>
  uint32_t _calc_offset(uint32_t id) {
    return id / (kChunkSize / sizeof(T)) * kChunkSize + id % (kChunkSize / sizeof(T)) * sizeof(T);
  }

 public:
  SSTBuilder(std::unique_ptr<WriteBatch>&& file = nullptr) : file_(std::move(file)) {}
  template <typename T>
  void append(const std::pair<SKey, T>& kv) {
    _append_align(kv.first.size());
    if (!index.size() && !counts) first_key = kv.first;
    counts++;
    if (counts == kChunkSize / sizeof(BlockValue<T>)) {
      index.emplace_back(kv.first, _calc_offset<BlockValue<T>>(offsets.size()));
      // printf("[%d, %d, %d, %d, %d, %d, %d]", kv.first.data()[0], kv.first.data()[1], kv.first.data()[2], kv.first.data()[3], (int)offsets.size(),
      //        (int)sizeof(BlockValue<T>), _calc_offset<BlockValue<T>>(offsets.size()));
      counts = 0;
    } else
      lst_key = kv.first;
    offsets.push_back({kv.second, now_offset});
    now_offset += kv.first.size();
    _write_key(kv.first);
    size_ += kv.first.size() + sizeof(T);
  }
  void make_index() {
    if (counts) {
      index.emplace_back(lst_key, _calc_offset<BlockValue<SValue>>(offsets.size() - 1));
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
  std::vector<BlockValue<SValue>> offsets;
  IndSKey lst_key, first_key;
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

  void _begin_new_file() {
    builder_.reset();
    auto [filename, id] = files_->next_pair();
    logger("begin_new_file(): ", filename);
    logger("begin_new_file(): size: ", builder_.size());
    builder_.new_file(std::make_unique<WriteBatch>(std::unique_ptr<AppendFile>(env_->openAppFile(filename)), kBatchSize));
    vec_newfiles_.emplace_back();
    vec_newfiles_.back().filename = filename;
    vec_newfiles_.back().file_id = id;
  }

  void _end_new_file() {
    logger("end_new_file(): kvsize: ", builder_.kv_size());
    logger("end_new_file(): size: ", builder_.size());
    builder_.make_index();
    logger("end_new_file(): size: ", builder_.size());
    builder_.finish();
    logger("end_new_file(): size: ", builder_.size());
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
    _begin_new_file();
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
      // _divide_file();
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
      // _divide_file();
    }
    return change;
  }

  std::vector<NewFileData> compact_end() {
    if (flag_) builder_.append(lst_value_);
    // TODO: we can store shorter key in viscnts by checking the LCP
    // Now we begin to write index block
    _end_new_file();
    return vec_newfiles_;
  }

  void flush_begin() {
    vec_newfiles_.clear();
    _begin_new_file();
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
    _end_new_file();
    return vec_newfiles_;
  }

  std::pair<std::vector<NewFileData>, double> decay_first(SeqIteratorSet&& iters) {
    double real_size_ = 0;  // re-calculate size
    vec_newfiles_.clear();
    _begin_new_file();
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
    _end_new_file();
    return {vec_newfiles_, real_size_};
  }

  std::pair<std::vector<NewFileData>, double> decay_second(SeqIteratorSet&& iters) {
    // maybe we can...
    double real_size_ = 0;  // re-calculate size
    vec_newfiles_.clear();
    _begin_new_file();
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
    _end_new_file();
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
    if (builder_.kv_size() > kMemTable) {
      logger("[divide file]");
      _end_new_file();
      _begin_new_file();
    }
  }
  std::vector<NewFileData> vec_newfiles_;
};

class LSMTree {
  class MemTables {
    std::mutex m_;

   public:
    MemTable* mem_;
    bool compacting_;
    MemTables() : mem_(new MemTable) {}
    void append(const std::pair<SKey, SValue>& kv) {
      std::unique_lock<std::mutex> lck_(m_);
      compacting_ |= mem_->size() >= kMemTable;
      mem_->append(kv.first, kv.second);
    }
    bool exists(const SKey& key) {
      // TODO: fix potential bugs
      // maybe fixed? I don't know.
      // debug
      mem_->ref();
      auto mt = mem_;  // if mem_->ref success, then mt = old mem_.
      auto ret = mt->exists(key);
      mt->unref();
      return ret;
    }
    void new_memtable() {
      // std::unique_lock<std::mutex> lck_(m_);
      // must run in C.S. (satisfied in compact_thread)
      auto nw = new MemTable;
      auto od = mem_;
      mem_ = nw;
      od->unref();
      // consider:
      // mem_ = nw
      // mem_->ref() X
      // mem_->ref()
      // mem_ = nw
      // od->unref() O
    }
    std::mutex& get_mutex() { return m_; }
  };

  class Immutables {
    const static auto kMaxLevelNum = 100;
    std::mutex m_;

   public:
    struct Node : public RefCounts {
      std::unique_ptr<ImmutableFile> file;
      std::atomic<Node*> next;
      bool remove_flag_;
      Node(std::unique_ptr<ImmutableFile>&& _file) : file(std::move(_file)), next(nullptr), remove_flag_(false) {}
      ~Node() {
        if (remove_flag_) file->remove();
      }
      void remove() { remove_flag_ = true; }
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
    void insert(std::vector<std::unique_ptr<ImmutableFile>>&& files, Node* left, Node* right) {
      std::unique_lock<std::mutex> lck_(m_);
      assert(head_ != nullptr);
      if (!head_->next) {
        logger("insert if head_->next = nullptr");
        std::reverse(files.begin(), files.end());
        Node* rright = nullptr;
        for (auto&& a : files) {
          assert(a.get() != nullptr);
          auto node = new Node(std::move(a));
          node->next.store(rright, std::memory_order_relaxed);
          rright = node;
        }
        std::unique_lock lck_(tree_m_);
        head_->next = rright;
      } else {
        logger("insert otherwise, ", left, ", ", right);
        std::reverse(files.begin(), files.end());
        auto rright = right;
        for (auto&& a : files) {
          assert(a.get() != nullptr);
          auto node = new Node(std::move(a));
          node->next.store(right, std::memory_order_relaxed);
          right = node;
        }
        auto leftn = left->next.load(std::memory_order_relaxed);
        left->next.store(right, std::memory_order_release);
        // ensure not release nodes in use
        std::unique_lock lck_(tree_m_);
        logger("insert otherwise get lock");
        // release old files
        for (auto a = leftn; a != rright;) {
          auto b = a->next.load(std::memory_order_relaxed);
          a->remove();
          a->unref();
          a = b;
        }
        // debug
        int cnt = 0;
        for (auto a = left->next.load(std::memory_order_relaxed); a != rright; a = a->next.load(std::memory_order_relaxed)) {
          assert(a->file.get() != nullptr);
          cnt++;
        }
        assert(cnt == files.size());
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
      // or use some index tree-based structures
      std::vector<Node*> v;
      {
        // ensure necessary nodes are referred.
        std::shared_lock lck_(tree_m_);
        for (Node* a = head_->next.load(std::memory_order_relaxed); a; a = a->next.load(std::memory_order_relaxed)) {
          a->ref();
          v.push_back(a);
        }
      }

      bool ret = false;
      for (auto a : v) {
        if (a->file->exists(key)) ret = true;
        if (ret) break;
      }
      for (auto a : v) a->unref();
      return ret;
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
        a->unref();
        a = b;
      }
      head_->next = nullptr;
    }

    void remove(Node* prev) {
      if (!prev || !prev->next) return;
      std::unique_lock lck_(tree_m_);
      auto a = prev->next.load(std::memory_order_relaxed);
      prev->next = a->next.load(std::memory_order_relaxed);
      a->next = 0;
    }

   private:
    Node* head_;
    std::shared_mutex tree_m_;
  };
  class ImmutablesIterator : public SeqIterator {
    Immutables::Node* now_;
    Immutables::Node* end_;
    SSTIterator iter_;

   public:
    ImmutablesIterator(Immutables::Node* begin, Immutables::Node* end) : now_(begin), end_(end), iter_(begin->file.get()) {}
    bool valid() override { return now_ != end_; }
    void next() override {
      iter_.next();
      while (!iter_.valid()) {
        now_ = now_->next;
        if (now_ == end_) return;
        iter_ = SSTIterator(now_->file.get());
      }
    }
    std::pair<SKey, SValue> read() override { return iter_.read(); }
    SeqIterator* copy() override { return new ImmutablesIterator(*this); }
  };
  Immutables tree_[kMaxHeight];
  size_t tree_height_;
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
      if (level == tree_height_ - 1)  // for the last level, we update the estimated size of viscnts.
        estimated_size_ += worker.compact_last_level(iter, ImmutablesIterator(left->next.load(std::memory_order_relaxed), right));
      else
        worker.compact(iter, ImmutablesIterator(left->next.load(std::memory_order_relaxed), right));
      _compact_result_insert(worker.compact_end(), left, right, level);
    }
  }

  bool _check_decay() { return estimated_size_ > delta_; }

 public:
  LSMTree(std::unique_ptr<LRUCache>&& cache, double delta, std::unique_ptr<Env>&& env, std::unique_ptr<FileName>&& filename,
          std::unique_ptr<BaseAllocator>&& alloc)
      : tree_height_(0),cache_(std::move(cache)),
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
    for (int i = 0; i < tree_height_; i++)
      if (tree_[i].exists(key)) return true;
    return false;
  }
  void decay_all() {
    SeqIteratorSet iter_set;
    iter_set.push(std::make_unique<MemTableIterator>(mem_.mem_->head()));
    for (int i = 0; i < tree_height_; i++) iter_set.push(std::make_unique<ImmutablesIterator>(ImmutablesIterator(tree_[i].head()->next, nullptr)));
    Compaction worker(filename_.get(), env_.get());
    auto [vec, sz] = worker.decay_first(std::move(iter_set));  // we use this first;
    estimated_size_ = sz;                                      // this is the exact value of estimated_size_ // will it decay twice in succession?
    // we store the decay result in the last level to decrease write amp.
    _compact_result_insert(std::move(vec), tree_[tree_height_ - 1].head(), nullptr, tree_height_ - 1);
    // clear all the levels and memtables
    for (uint32_t i = 0; i < tree_height_ - 1; ++i) tree_[i].clear();
    mem_.new_memtable();
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
      if (mem_.compacting_) {
        logger("flush memtable");
        std::unique_lock<std::mutex> lck_(mem_.get_mutex());
        auto table = mem_.mem_;
        mem_.compacting_ = false;
        logger("memtable size: ", table->size());
        MemTableIterator iter(table->begin());
        if (tree_height_ == 0) {
          tree_height_ += 1;
        }
        _compact_tables(table, iter, 0);
        logger("flush memtable end");
        mem_.new_memtable();
        // std::unique_lock<std::mutex> lck_(mem_.get_mutex());
        // mem_.od_ = std::move(mem.nw_)
        // mem_.od_.compacting = false;
      }
      if (tree_height_) logger("level 0: ", tree_[0].size());
      // continue;  // debug
      // continue;  // debug
      size_t level_size = kSSTable * 2;
      for (uint32_t i = 0; i < tree_height_; i++) {
        auto& a = tree_[i];
        if (a.size() > level_size) {
          logger("compact sst of ", i, "-th level");
          // choose one file to compact
          auto lucky_one = a.head()->next.load(std::memory_order_relaxed);
          if (i == tree_height_ - 1) {
            // new level
            tree_height_++;
            tree_[tree_height_ - 1].head()->next = lucky_one;
            tree_[i].remove(tree_[i].head());
          } else {
            _compact_tables(lucky_one->file, SSTIterator(lucky_one->file.get()), i + 1);
            tree_[i].remove(tree_[i].head());
            lucky_one->remove();
            lucky_one->unref();
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
      : tree(std::make_unique<LRUCache>(1024), delta, std::unique_ptr<Env>(createDefaultEnv()), std::make_unique<FileName>(0, path),
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
