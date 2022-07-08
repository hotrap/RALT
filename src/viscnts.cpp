#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
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

const static size_t kDataBlockSize = 1 << 16;           // 64 KB
const static size_t kPageSize = 4096;                   // 4 KB
const static size_t kMagicNumber = 0x25a65facc3a23559;  // echo viscnts | sha1sum
const static size_t kMemTable = 1 << 28;                // 64 MB
const static size_t kSSTable = 1 << 28;
const static size_t kRatio = 10;
const static size_t kBatchSize = 1 << 16;
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
  FileBlockHandle() = default;
  explicit FileBlockHandle(uint32_t offset, uint32_t size, uint32_t counts) : offset(offset), size(size), counts(counts) {}
};

class FileBlock {     // process blocks in a file
  uint32_t file_id_;  // file id
  FileBlockHandle handle_;
  uint8_t* data_;  // a reference to pointer in cache
  LRUCache* cache_;
  LRUHandle* lru_handle_;
  BaseAllocator* alloc_;
  RandomAccessFile* file_ptr_;

 public:
  FileBlock() = default;
  explicit FileBlock(uint32_t file_id, FileBlockHandle handle, LRUCache* cache, BaseAllocator* alloc, RandomAccessFile* file_ptr)
      : file_id_(file_id), handle_(handle), data_(nullptr), cache_(cache), lru_handle_(nullptr), alloc_(alloc), file_ptr_(file_ptr) {}

  ssize_t acquire() {
    // TODO: when file_ptr is MmapRAFile, we don't need to allocate buffer.
    // <- We don't use mmap. since viscnts' accesses are sequential.
    Slice result(nullptr, 0);
    size_t key = (size_t)file_id_ << 32 | handle_.offset;
    lru_handle_ = cache_->lookup(Slice(reinterpret_cast<uint8_t*>(&key), sizeof(size_t)), Hash8(reinterpret_cast<char*>(&key)));
    if (!lru_handle_->valid.load(std::memory_order_relaxed)) {
      auto ptr = alloc_->allocate(handle_.size);
      auto err = file_ptr_->read(handle_.offset, handle_.size, ptr, result);
      if (err) return err;
      if (lru_handle_->valid.exchange(true, std::memory_order_relaxed)) {
        lru_handle_->data = result.data();
        lru_handle_->deleter = alloc_;
      } else
        alloc_->release(ptr);
      data_ = result.data();
      // if (data_ != ptr) alloc_->release(ptr); // we don't use mmap
    } else {
      data_ = result.data();
    }
    return 0;
  }

  void release() { cache_->release(lru_handle_); }

  int read_key(uint32_t id, SKey& result) const {
    if (id > handle_.counts) return -1;
    uint32_t offset = reinterpret_cast<uint32_t*>(data_)[id];
    size_t klen = *reinterpret_cast<size_t*>(data_ + offset);
    result = SKey(data_ + offset + sizeof(klen), klen);
    return 0;
  }

  template <typename T>
  int read_value(uint32_t id, T& result) const {
    if (id > handle_.counts) return -1;
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

  uint32_t counts() { return handle_.counts; }
  uint32_t offset() { return handle_.offset; }
  ssize_t exists(const SKey& key) {
    uint32_t l = 0, r = handle_.counts - 1;
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
  // [offset of index block, size, counts]
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
  FileBlock index_block_;
  // LRUCache pointer reference to the one in VisCnts
  LRUCache* cache_;
  BaseAllocator* alloc_;
  // last Block range
  // initially block_range_ is SKey(nullptr, 1e9), SKey(nullptr, 1e9)
  std::pair<IndSKey, IndSKey> block_range_;
  FileBlock lst_block_;

 public:
  ImmutableFile(uint32_t file_id, uint32_t size, std::unique_ptr<RandomAccessFile>&& file_ptr, LRUCache* cache, BaseAllocator* alloc,
                std::pair<IndSKey, IndSKey> range)
      : file_id_(file_id),
        size_(size),
        file_ptr_(std::move(file_ptr)),
        cache_(cache),
        alloc_(alloc),
        range_(range),
        block_range_(IndSKey(nullptr, 1e18), IndSKey(nullptr, 1e18)) {
    // read index block
    Slice result(nullptr, 0);
    FileBlockHandle index_bh;
    file_ptr->read(size_ - sizeof(size_t) - sizeof(FileBlockHandle), sizeof(FileBlockHandle), (uint8_t*)(&index_bh), result);
    if (result.data() != (uint8_t*)(&index_bh)) index_bh = *(FileBlockHandle*)(result.data());
    index_block_ = FileBlock(file_id, index_bh, cache, alloc, file_ptr.get());
  }
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
    FileBlockHandle value;
    auto err = index_block_.read_value(l, value);
    // store the range of the current block
    if (err) return err;
    err = index_block_.read_key(l, _key);
    if (err) return err;
    block_range_.first = IndSKey(_key);
    if (l + 1 < index_block_.counts()) {
      err = index_block_.read_key(l + 1, _key);
      if (err) return err;
    } else
      block_range_.second = std::move(IndSKey(nullptr, 1e18));
    index_block_.release();
    lst_block_ = FileBlock(file_id_, FileBlockHandle(value.offset, value.size, value.counts), cache_, alloc_, file_ptr_.get());
    return lst_block_.exists(key);
  }
  bool in_range(const SKey& key) {
    auto& [l, r] = range_;
    return l <= key && key <= r;
  }

  std::pair<SKey, SKey> range() { return {range_.first.ref(), range_.second.ref()}; }

  uint32_t size() { return size_; }

  uint32_t get_block_id(uint32_t offset) {
    if (offset >= size_) return -1;
    uint32_t l = 0, r = index_block_.counts() - 1;
    SKey _key(nullptr, 0);
    FileBlockHandle value;
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
    FileBlockHandle value;
    index_block_.read_value(id, value);
    index_block_.release();
    return value.offset;
  }

  uint32_t get_block_end_offset(uint32_t id) {
    if (id >= index_block_.counts()) return -1;
    index_block_.acquire();
    FileBlockHandle value;
    index_block_.read_value(id, value);
    index_block_.release();
    return value.offset + value.size;
  }

  // id must be valid
  FileBlock get_file_block(uint32_t id) {
    index_block_.acquire();
    FileBlockHandle value;
    index_block_.read_value(id, value);
    index_block_.release();
    return FileBlock(file_id_, FileBlockHandle(value.offset, value.size, value.counts), cache_, alloc_, file_ptr_.get());
  }

  bool range_overlap(const std::pair<SKey, SKey>& range) {
    auto [l, r] = range_;
    return l <= range.second && range.first <= r;
  }

 private:
  bool in_block_range(const SKey& key) {
    auto& [l, r] = block_range_;
    return l <= key && key <= r;
  }
};

// Semantically there can only be one AppendFile at the same time
class WriteBatch {
  std::unique_ptr<AppendFile> file_ptr_;
  uint8_t* data_;
  const size_t buffer_size_;
  size_t used_size_;

 public:
  explicit WriteBatch(std::unique_ptr<AppendFile>&& file, size_t size) : file_ptr_(std::move(file)), buffer_size_(size), used_size_(0) {
    data_ = new uint8_t[size];
  }
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
      append(Slice(reinterpret_cast<uint8_t*>(&y), sizeof(T)));
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
    for (int i = 1; i < tmp_kv_.size(); ++i)
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
    for (int i = 1; i < tmp_kv_.size(); ++i)
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
class SSTIterator : public SeqIterator {
 public:
  SSTIterator(ImmutableFile* file, uint32_t offset) : file_(file), offset_(offset), kvpair_(SKey(nullptr, 0), SValue()) {
    if (file->size() > offset) {
      now_block_id_ = file_->get_block_id(offset_);
      // if now_block_id_ >= index_block_.counts(), then next_offset_ equals to ~0u.
      next_offset_ = file->get_block_end_offset(now_block_id_);
      now_ = file->get_file_block(now_block_id_);
      now_.acquire();
      kvpair_valid_ = false;
    }
  }
  bool valid() override { return file_->size() > offset_; }
  void next() override {
    SKey key(nullptr, 0);
    if (!kvpair_valid_) {
      now_.read_key_offset(offset_, key);
      offset_ += key.size() + sizeof(SValue);
    } else
      offset_ += kvpair_.first.size() + sizeof(SValue);
    kvpair_valid_ = false;

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
  std::pair<SKey, SValue> read() override {
    if (!kvpair_valid_) {
      kvpair_valid_ = true;
      SKey key(nullptr, 0);
      SValue value;
      now_.read_key_offset(offset_, key);
      now_.read_value_offset(offset_, value);
      kvpair_ = {key, value};
    }
    return kvpair_;
  }
  SeqIterator* copy() override { return new SSTIterator(*this); }

 private:
  ImmutableFile* file_;
  uint32_t offset_;
  uint32_t now_block_id_;
  uint32_t next_offset_;
  FileBlock now_;
  std::pair<SKey, SValue> kvpair_;
  bool kvpair_valid_;
};

class SSTBuilder {
  std::unique_ptr<WriteBatch> file_;

 public:
  SSTBuilder(std::unique_ptr<WriteBatch>&& file = nullptr) : file_(std::move(file)) {}
  template <typename T>
  void append(const std::pair<SKey, T>& kv) {
    now_offset += kv.first.size() + sizeof(T);
    counts++;
    if (now_offset - lst_offset > kDataBlockSize) {
      index.emplace_back(IndSKey(kv.first), FileBlockHandle(now_offset, now_offset - lst_offset, counts));
      lst_offset = now_offset;
    }
    _write_kv(kv);
  }
  void make_index() {
    lst_offset = now_offset;
    for (const auto& a : index) {
      _write_kv(std::make_pair(a.first.ref(), a.second));
      now_offset += a.first.size() + sizeof(FileBlockHandle);
    }
  }
  void finish() {
    uint32_t counts = index.size();
    file_->append_other(FileBlockHandle(now_offset, now_offset - lst_offset, counts));  // write offset of index block
    file_->append_other(kMagicNumber);
    now_offset += sizeof(FileBlockHandle) + sizeof(size_t);
    file_->flush();
  }
  void reset() {
    now_offset = 0;
    lst_offset = 0;
    counts = 0;
    index.clear();
  }
  void new_file(std::unique_ptr<WriteBatch>&& file) { file_ = std::move(file); }

  size_t size() { return now_offset; }

  std::pair<IndSKey, IndSKey> range() { return {index[0].first, index.back().first}; }

 private:
  uint32_t now_offset, lst_offset, counts;
  std::vector<std::pair<IndSKey, FileBlockHandle>> index;
  template <typename T>
  void _write_kv(const std::pair<SKey, T>& a) {
    file_->append_other(a.first.len());
    file_->append(a.first);
    file_->append_other(a.second);
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
  std::pair<SKey, SValue> lst_value_;

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
    auto L = left.valid() ? left.read() : std::pair<SKey, SValue>();
    auto R = right.valid() ? right.read() : std::pair<SKey, SValue>();

    while (left.valid() || right.valid()) {
      if (!right.valid() || (L.first <= R.first)) {
        if (flag_ && lst_value_.first == L.first) {
          lst_value_.second += L.second;
          lst_value_.first = L.first;  // avoid copying SKey, because old SKey may be expired
        } else {
          if (flag_) builder_.append(lst_value_);
          lst_value_ = L, flag_ = true;
        }
        left.next();
        if (left.valid()) L = left.read();
      } else {
        if (flag_ && lst_value_.first == R.first) {
          lst_value_.second += R.second;
          lst_value_.first = R.first;
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
    auto L = left.valid() ? left.read() : std::pair<SKey, SValue>();
    auto R = right.valid() ? right.read() : std::pair<SKey, SValue>();
    double change = 0;

    while (left.valid() || right.valid()) {
      if (!right.valid() || (L.first <= R.first)) {
        if (flag_ && lst_value_.first == L.first) {
          lst_value_.second += L.second;
          lst_value_.first = L.first;  // avoid copying SKey, because old SKey may be expired
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
          lst_value_.first = R.first;
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
    builder_.make_index();
    builder_.finish();
    return vec_newfiles_;
  }

  void flush_begin() {
    vec_newfiles_.clear();
    begin_new_file();
    flag_ = false;
  }

  template <typename TIter, std::enable_if_t<std::is_base_of_v<SeqIterator, TIter>, bool> = true>
  void flush(TIter left) {
    if (!left.valid()) return;

    while (left.valid()) {
      auto L = left.read();
      if (flag_ && lst_value_.first == L.first) {
        lst_value_.second += L.second;
        lst_value_.first = L.first;  // avoid copying SKey, because old SKey may be expired
      } else {
        if (flag_) builder_.append(lst_value_);
        lst_value_ = L, flag_ = true;
      }
      left.next();
    }
  }

  std::vector<NewFileData> flush_end() {
    if (flag_) builder_.append(lst_value_);
    builder_.make_index();
    builder_.finish();
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
        lst_value_.first = L.first;  // avoid copying SKey, because old SKey may be expired
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
    builder_.make_index();
    builder_.finish();
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
    return {vec_newfiles_, real_size_};
  }

 private:
  std::mt19937 rndgen_;
  double decay_prob_;  // = 0.5 on default
  double _calc_decay_value(const std::pair<SKey, SValue>& kv) { return (kv.first.len() + kv.second.vlen) * std::min(kv.second.counts * .5, 1.); }
  bool _decay_kv(std::pair<SKey, SValue>& kv) {
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
      builder_.make_index();
      builder_.finish();
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
    bool compacting;
    bool flag;
    MemTables() : flag(false), compacting(false) {}
    void append(const std::pair<SKey, SValue>& kv) {
      std::unique_lock<std::mutex> lck_(m_);
      if (compacting) {
        flag ? od_.mem->append(kv.first, kv.second) : nw_.mem->append(kv.first, kv.second);
      } else {
        compacting |= flag ? nw_.mem->size() >= kMemTable : od_.mem->size() >= kMemTable;
        // compacting old
        flag ? nw_.mem->append(kv.first, kv.second) : od_.mem->append(kv.first, kv.second);
        // swapping od and nw is done in compacting thread
        // as expected, the thread will wait until od.refcounts equals to zero.
      }
    }
    bool exists(const SKey& key) {
      if (!od_.compacting_complete.load(std::memory_order_relaxed)) {
        od_.refcounts += 1;
        bool result = od_.mem->exists(key);
        od_.refcounts -= 1;
        if (result) return true;
      }
      if (!nw_.compacting_complete.load(std::memory_order_relaxed)) {
        nw_.refcounts += 1;
        bool result = nw_.mem->exists(key);
        nw_.refcounts -= 1;
        if (result) return true;
      }
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
      if (!head_) {
        std::reverse(files.begin(), files.end());
        for (auto&& a : files) {
          auto node = new Node(std::move(a));
          node->next.store(head_, std::memory_order_relaxed);
          head_ = node;
          node->refcounts.fetch_add(1, std::memory_order_relaxed);
        }

      } else {
        std::reverse(files.begin(), files.end());
        for (auto&& a : files) {
          auto node = new Node(std::move(a));
          node->refcounts++;
          node->next.store(right, std::memory_order_relaxed);
          right = node;
        }
        // release old files
        for (auto a = left->next.load(std::memory_order_relaxed); a != right;) {
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
    ImmutablesIterator(Immutables::Node* begin) : now_(begin), iter_(begin->file.get(), 0) {}
    bool valid() override { return now_ != nullptr; }
    void next() override {
      iter_.next();
      while (now_ && !iter_.valid()) {
        now_ = now_->next;
        if (!now_) return;
        iter_ = SSTIterator(now_->file.get(), 0);
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
      auto file = std::make_unique<ImmutableFile>(a.file_id, a.size, std::unique_ptr<RandomAccessFile>(env_->openRAFile(a.filename)), cache_.get(), file_alloc_.get(), a.range);
      files.push_back(std::move(file));
    }
    tree_[level].insert(std::move(files), left, right);
  }

  template <typename TableType, typename TableIterType>
  void _compact_tables(const TableType& table, const TableIterType& iter, int level) {
    Immutables::Node *left = nullptr, *right = nullptr;
    for (auto a = tree_[level].head(); a; a = a->next.load(std::memory_order_relaxed)) {
      if (!a->file || !a->file->range_overlap(table->range())) {
        if (!left)
          left = a;
        else
          right = a;
      }
    }
    Compaction worker(filename_.get(), env_.get());
    if (left->next == right) {
      worker.flush_begin();
      worker.flush(iter);
      _compact_result_insert(worker.flush_end(), left, right, level);
    } else {
      worker.compact_begin();
      for (auto p = left;; p = p->next) {
        if (level == tree_.size() - 1)  // for the last level, we update the estimated size of viscnts.
          estimated_size_ += worker.compact_last_level(iter, SSTIterator(p->file.get(), 0));
        else
          worker.compact(iter, SSTIterator(p->file.get(), 0));
        if (p == right) break;
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
    for (int i = 0; i < tree_.size() - 1; ++i) tree_[i].clear();
    mem_.clear();
  }
  void compact_thread() {
    // use leveling first...
    // although I think tiering is better.
    // we only create one compact_thread including flushing memtable, compacing, decay.
    using namespace std::chrono_literals;
    while (true) {
      std::this_thread::sleep_for(100ms);
      if (_check_decay()) {
        decay_all();
        continue;
      }
      if (mem_.compacting) {
        if (mem_.od_.compacting_complete || mem_.nw_.compacting_complete) {
          // we must avoid moving nw_ to od_.
          // we use compacting_complete to mark the validality.
          // we use flag to mark which thread is processing now.
          // we use compacting to mark the whether there exists a table that needs a compaction.
          if (mem_.od_.compacting_complete && !mem_.od_.refcounts) {
            std::unique_lock<std::mutex> lck_(mem_.get_mutex());
            // release is to reset memtable
            mem_.od_.mem->release();
            mem_.od_.compacting_complete = false;
            mem_.compacting = false;
            mem_.flag ^= 1;
          } else if (mem_.nw_.compacting_complete && !mem_.nw_.refcounts) {
            std::unique_lock<std::mutex> lck_(mem_.get_mutex());
            // release is to reset memtable
            mem_.nw_.mem->release();
            mem_.nw_.compacting_complete = false;
            mem_.compacting = false;
            mem_.flag ^= 1;
          }
          continue;
        }
        auto& table = mem_.flag ? mem_.nw_.mem : mem_.od_.mem;
        MemTableIterator iter(mem_.od_.mem->head());
        _compact_tables(table, iter, 0);

        mem_.flag ? mem_.nw_.compacting_complete = true : mem_.od_.compacting_complete = true;
        // std::unique_lock<std::mutex> lck_(mem_.get_mutex());
        // mem_.od_ = std::move(mem.nw_)
        // mem_.od_.compacting = false;
      }
      size_t level_size = kSSTable * 2;
      for (int i = 0; i < tree_.size(); i++) {
        auto& a = tree_[i];
        if (a.size() > level_size) {
          // choose one file to flush
          auto lucky_one = a.head()->next.load(std::memory_order_relaxed);
          if (i == tree_.size() - 1) {
            // new level
            tree_.emplace_back();
            tree_.back().insert(std::move(lucky_one->file), nullptr, nullptr);
            tree_[i].unref(lucky_one);
          } else {
            _compact_tables(lucky_one->file, SSTIterator(lucky_one->file.get(), 0), i + 1);
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
  return vc->is_hot(viscnts_lsm::SKey(const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(key)), klen));
}

int VisCntsClose(void* ac) {
  auto vc = reinterpret_cast<viscnts_lsm::VisCnts*>(ac);
  delete vc;
  return 0;
}
