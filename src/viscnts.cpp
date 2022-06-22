#include <memory>

#include "alloc.hpp"
#include "cache.hpp"
#include "common.hpp"
#include "file.hpp"
#include "hash.hpp"
#include "key.hpp"
#include "memtable.hpp"

namespace viscnts_lsm {

const static size_t kDataBlockSize = 1 << 16;  // 64 KB
const static size_t kPageSize = 4096;          // 4 KB
const static size_t kMagicNumber = 0x25a65facc3a23559; // echo viscnts | sha1sum

class FileBlockHandle {
  // The structure of a file block
  // [handles]
  // [kv pairs...]
  uint32_t file_id_;
  uint32_t offset_;
  uint32_t size_;    // The size of SST doesn't exceed 4GB
  uint32_t counts_;  // number of keys in the fileblock
 public:
  FileBlockHandle() = default;
  explicit FileBlockHandle(uint32_t file_id, uint32_t offset, uint32_t size, uint32_t counts)
      : file_id_(file_id), offset_(offset), size_(size), counts_(counts) {}
  size_t size() const { return size_; }
  size_t offset() const { return offset_; }
  size_t counts() const { return counts_; }
  size_t file_id() const { return file_id_; }
};

class FileBlock {  // process blocks in a file
  FileBlockHandle handle_;
  uint8_t* data_;  // a reference to pointer in cache
  LRUCache* cache_;
  LRUHandle* lru_handle_;
  BaseAllocator* alloc_;
  RandomAccessFile* file_ptr_;

 public:
  FileBlock() = default;
  explicit FileBlock(FileBlockHandle handle, LRUCache* cache, BaseAllocator* alloc, RandomAccessFile* file_ptr)
      : handle_(handle), data_(nullptr), cache_(cache), alloc_(alloc), file_ptr_(file_ptr) {}

  ssize_t acquire() {
    // TODO: when file_ptr is MmapRAFile, we don't need to allocate buffer.
    Slice result(nullptr, 0);
    size_t key = (size_t)handle_.file_id() << 32 | handle_.offset();
    lru_handle_ = cache_->lookup(Slice(reinterpret_cast<uint8_t*>(&key), sizeof(size_t)), Hash8(reinterpret_cast<char*>(&key)));
    if (!lru_handle_->valid.load(std::memory_order_relaxed)) {
      auto ptr = alloc_->allocate(handle_.size());
      auto err = file_ptr_->read(handle_.offset(), handle_.size(), ptr, result);
      if (err) return err;
      if (lru_handle_->valid.exchange(true, std::memory_order_relaxed)) {
        lru_handle_->data = result.data();
        lru_handle_->deleter = alloc_;
      } else
        alloc_->release(ptr);
      data_ = result.data();
      if (data_ != ptr) alloc_->release(ptr);
    } else {
      data_ = result.data();
    }
    return 0;
  }

  void release() { cache_->release(lru_handle_); }

  int read_key(uint32_t id, SKey& result) const {
    if (id > handle_.counts()) return -1;
    uint32_t offset = reinterpret_cast<uint32_t*>(data_)[id];
    size_t klen = *reinterpret_cast<size_t*>(data_ + offset);
    result = SKey(data_ + offset + sizeof(klen), klen);
    return 0;
  }

  template <typename T>
  int read_value(uint32_t id, T& result) const {
    if (id > handle_.counts()) return -1;
    uint32_t offset = reinterpret_cast<uint32_t*>(data_)[id];
    size_t klen = *reinterpret_cast<size_t*>(data_ + offset);
    result = *reinterpret_cast<T*>(data_ + offset + klen + sizeof(size_t));
    return 0;
  }

  // assume that input is valid, read_key_offset and read_value_offset
  int read_key_offset(uint32_t offset, SKey& result) const {
    size_t klen = *reinterpret_cast<size_t*>(data_ + offset);
    result = SKey(data_ + offset + sizeof(klen), klen);
    return 0;
  }

  template <typename T>
  int read_value_offset(uint32_t offset, T& result) const {
    size_t klen = *reinterpret_cast<size_t*>(data_ + offset);
    result = *reinterpret_cast<T*>(data_ + offset + klen + sizeof(size_t));
    return 0;
  }

  uint32_t counts() { return handle_.counts(); }
  uint32_t offset() { return handle_.offset(); }
  ssize_t exists(const SKey& key) {
    uint32_t l = 0, r = handle_.counts() - 1;
    SKey _key(nullptr, 0);
    acquire();
    while (l < r) {
      auto mid = (l + r) >> 1;
      auto err = read_key(mid, _key);
      if (err) return err;
      auto cmp = SKeyComparator()(_key, key);
      if (!cmp) return 1;
      if (cmp < 0)
        l = mid;
      else
        r = mid - 1;
    }
    release();
    return 0;
  }
};

struct IndexBlockValue {
  uint32_t offset, size, counts;
};

// one SST
class ImmutableFile {
  // The structure of the file:
  // [Handles(offset) of keys of data blocks]
  //    -(uint32_t, 4Bytes)
  // [Data Blocks]...
  //    -(klen(size_t), [key], vlen(size_t), counts(double))
  // [Handles(offset) of index block]
  //    -([offset in index block(uint32_t), size_, counts_], 4Bytes)
  // [Block of the first key of data blocks] (=index block in leveldb)
  //    -(klen(size_t), [key], offset of corresponding data block(uint32_t))
  // Now we don't consider crash consistency
  // leveldb drops the shared prefix of a key which is not restart point, here we don't consider the compression now

  // we assume the length of key is < 4096.
  // from metadata file
  uint32_t level_;
  uint32_t file_id_;
  uint32_t size_;
  // range is stored in memory
  std::pair<SKey, SKey> range_;
  // filename is not stored in metadata file, it's generated by file_id and the filename of the metadata file
  std::string filename_;
  // pointer to the file
  // I don't consider the deletion of this pointer now
  // 1000 (limitation of file handles) files of 64MB, then 64GB of data, it may be enough?
  std::unique_ptr<RandomAccessFile> file_ptr_;
  FileBlock index_block_;
  // LRUCache pointer reference to the one in VisCnts
  LRUCache* cache_;
  BaseAllocator* alloc_;
  // last Block range
  // initially block_range_ is SKey(nullptr, 1e9), SKey(nullptr, 1e9)
  std::pair<SKey, SKey> block_range_;
  FileBlock lst_block_;

 public:
  /* the */
  ssize_t exists(const SKey& key) {
    if (in_block_range(key)) {
      return lst_block_.exists(key);
    }
    uint32_t l = 0, r = index_block_.counts() - 1;
    SKey _key(nullptr, 0);
    index_block_.acquire();
    while (l < r) {
      auto mid = (l + r) >> 1;
      auto err = index_block_.read_key(mid, _key);
      if (err) return err;
      auto cmp = SKeyComparator()(_key, key);
      if (!cmp) return 1;
      if (cmp < 0)
        l = mid;
      else
        r = mid - 1;
    }
    IndexBlockValue value;
    auto err = index_block_.read_value(l, value);
    // store the range of the current block
    if (err) return err;
    err = index_block_.read_key(l, _key);
    if (err) return err;
    block_range_.first = _key;
    if (l + 1 < index_block_.counts()) {
      err = index_block_.read_key(l + 1, _key);
      if (err) return err;
    } else
      block_range_.second = SKey(nullptr, 1e9);
    index_block_.release();
    lst_block_ = FileBlock(FileBlockHandle(file_id_, value.offset, value.size, value.counts), cache_, alloc_, file_ptr_.get());
    return lst_block_.exists(key);
  }
  bool in_range(const SKey& key) {
    auto& [l, r] = range_;
    return l <= key && key <= r;
  }

  uint32_t size() { return size_; }

  uint32_t get_block_id(uint32_t offset) {
    if (offset >= size_) return -1;
    uint32_t l = 0, r = index_block_.counts() - 1;
    SKey _key(nullptr, 0);
    IndexBlockValue value;
    index_block_.acquire();
    while (l < r) {
      auto mid = (l + r) >> 1;
      index_block_.read_value(mid, value);
      if (value.offset <= offset)
        l = mid;
      else
        r = mid - 1;
    }
    index_block_.release();
    return l;
  }

  uint32_t get_block_offset(uint32_t id) {
    if (id >= index_block_.counts()) return -1;
    index_block_.acquire();
    IndexBlockValue value;
    index_block_.read_value(id, value);
    index_block_.release();
    return value.offset;
  }

  // id must be valid
  FileBlock get_file_block(uint32_t id) {
    index_block_.acquire();
    IndexBlockValue value;
    index_block_.read_value(id, value);
    index_block_.release();
    return FileBlock(FileBlockHandle(file_id_, value.offset, value.size, value.counts), cache_, alloc_, file_ptr_.get());
  }

 private:
  bool in_block_range(const SKey& key) {
    auto& [l, r] = block_range_;
    return l <= key && key <= r;
  }
};

class WriteBatch {
  AppendFile* file_ptr_;
  uint8_t* data_;
  const size_t buffer_size_;
  size_t used_size_;

 public:
  explicit WriteBatch(AppendFile* file, size_t size) : file_ptr_(file), buffer_size_(size), used_size_(0) { data_ = new uint8_t[size]; }
  ~WriteBatch() { delete[] data_; }

  void append(const Slice& kv) {
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
      T y = x;
      append(Slice(reinterpret_cast<uint8_t*>(&y), 4));
    } else {
      *reinterpret_cast<T*>(data_ + used_size_) = x;
      used_size_ += sizeof(T);
    }
  }

  void flush() {
    file_ptr_->write(Slice(data_, used_size_));
    used_size_ = 0;
  }
};

// this iterator is used in compaction
class SeqIterator {
 public:
  virtual ~SeqIterator() = default;
  virtual bool valid() = 0;
  virtual void next() = 0;
  virtual std::pair<SKey, SValue> read() = 0;
};

class MemTableIterator : public SeqIterator {
 public:
  MemTableIterator(MemTable::Node* node) : node_(node) {}
  bool valid() { return node_ != nullptr; }
  // requirement: node_ is valid
  void next() { node_ = node_->noBarrierGetNext(0); }
  std::pair<SKey, SValue> read() { return std::make_pair(node_->key, node_->value); }

 private:
  MemTable::Node* node_;
};

class SSTIterator : public SeqIterator {
 public:
  SSTIterator(ImmutableFile* file, uint32_t offset) : file_(file), offset_(offset), kvpair_(SKey(nullptr, 0), SValue()) {
    if (file->size() > offset) {
      now_block_id_ = file_->get_block_id(offset_);
      // if now_block_id_ >= index_block_.counts(), then next_offset_ equals to ~0u.
      next_offset_ = file->get_block_offset(now_block_id_ + 1);
      now_ = file->get_file_block(now_block_id_);
      now_.acquire();
      SKey key(nullptr, 0);
      SValue value;
      now_.read_key_offset(offset_, key);
      now_.read_value_offset(offset_, value);
      offset_ += key.size() + sizeof(SValue);
      kvpair_ = std::make_pair(key, value);
      if (offset_ >= next_offset_) {
        now_.release();
        now_block_id_++;
        next_offset_ = file_->get_block_offset(now_block_id_ + 1);
        now_ = file_->get_file_block(now_block_id_);
        now_.acquire();
        offset_ = now_.offset();
      }
    }
  }
  bool valid() { return file_->size() > offset_; }
  void next() {
    SKey key(nullptr, 0);
    SValue value;
    now_.read_key_offset(offset_, key);
    now_.read_value_offset(offset_, value);
    offset_ += key.size() + sizeof(SValue);
    kvpair_ = std::make_pair(key, value);
    if (offset_ >= next_offset_) {
      now_.release();
      now_block_id_++;
      next_offset_ = file_->get_block_offset(now_block_id_ + 1);
      now_ = file_->get_file_block(now_block_id_);
      now_.acquire();
      offset_ = now_.offset();
    }
  }
  // remember, SKey is a reference to file block
  std::pair<SKey, SValue> read() { return kvpair_; }

 private:
  ImmutableFile* file_;
  uint32_t offset_;
  uint32_t now_block_id_;
  uint32_t next_offset_;
  FileBlock now_;
  std::pair<SKey, SValue> kvpair_;
};

class SSTBuilder {
  WriteBatch* file_;

 public:
  template <typename T>
  void append(const std::pair<SKey, T>& kv) {
    now_offset += kv.first.size() + sizeof(T);
    counts++;
    if (now_offset - lst_offset > kDataBlockSize) {
      index.emplace_back(kv.first.copy(), IndexBlockValue{now_offset, now_offset - lst_offset, counts});
      lst_offset = now_offset;
    }
    _write_kv(kv);
  }
  void make_index() {
    lst_offset = now_offset;
    for (const auto& a : index) {
      _write_kv(a);
      now_offset += a.first.size() + sizeof(IndexBlockValue);
    }
  }
  void finish() {
    uint32_t counts = index.size();
    file_->append_other(IndexBlockValue{now_offset, now_offset - lst_offset, counts});  // write offset of index block
    file_->append_other(kMagicNumber);
    file_->flush();
  }
  void reset() {
    now_offset = 0;
    lst_offset = 0;
    counts = 0;
    index.clear();
  }

 private:
  uint32_t now_offset, lst_offset, counts;
  std::vector<std::pair<SKey, IndexBlockValue>> index;
  template <typename T>
  void _write_kv(const std::pair<SKey, T>& a) {
    file_->append_other(a.first.len());
    file_->append(a.first);
    file_->append_other(a.second);
  }
};

class Compaction {
 public:
  void compact(SeqIterator* left, SeqIterator* right) {
    auto L = left->read();
    auto R = right->read();
    SSTBuilder builder;
    builder.reset();
    while (left->valid() || right->valid()) {
      if (!right->valid() || (L.first <= R.first))
        builder.append(L), left->next();
      else
        builder.append(R), right->next();
    }
    // TODO: we can choose shorter key in viscnts by checking the LCP
    // Now we begin to write index block
    builder.make_index();
    builder.finish();
  }
};

// Viscnts, implement lsm tree and other things.
class VisCnts {
 public:
};

}  // namespace viscnts_lsm