#ifndef VISCNTS_SST_H__
#define VISCNTS_SST_H__

#include <vector>
#include "fileenv.hpp"
#include "fileblock.hpp"
#include "alloc.hpp"
#include "writebatch.hpp"

namespace viscnts_lsm {


const static size_t kMagicNumber = 0x25a65facc3a23559;  // echo viscnts | sha1sum
const static size_t kPageSize = 1 << 12;

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
                const std::pair<IndSKey, IndSKey>& range, KeyCompT comp)
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
    if (len && (now_offset + len - 1) / kChunkSize != now_offset / kChunkSize) {
      _align();
    }
  }

 public:
  SSTBuilder(std::unique_ptr<WriteBatch>&& file = nullptr) : 
    file_(std::move(file)), now_offset(0), lst_offset(0), counts(0), size_(0) {}
  void append(const DataKey& kv) {
    assert(kv.key().len() > 0);
    _append_align(kv.size());
    if (offsets.size() % (kIndexChunkSize / sizeof(uint32_t)) == 0) {
      index.emplace_back(kv.key(), offsets.size());
      if (offsets.size() == 0) first_key = kv.key();
    }
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
    offsets.push_back(now_offset);
    now_offset += kv.size();
    size_ += kv.size();
    return file_->reserve_kv(kv, sizeof(decltype(kv.value())));
  }

  template<typename T>
  void set_lstkey(const T& kv) {
    lst_key = kv.key();
  }

  void make_offsets(const std::vector<uint32_t>& offsets) {
    _align();
    for (const auto& a : offsets) {
      file_->append_other(a);
      now_offset += sizeof(uint32_t);
    }
  }

  void make_index() {
    // append all the offsets in the data block
    make_offsets(offsets);
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
    // append all the offsets in the index block
    make_offsets(v);
    // append two block handles.
    // write offset of index block
    file_->append_other(FileBlockHandle(lst_offset, now_offset - lst_offset, index.size()));  
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

}

#endif