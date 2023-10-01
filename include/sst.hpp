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

template <typename KeyCompT, typename ValueT>
class SSTIterator {
 public:
  using DataKey = BlockKey<SKey, ValueT>;
  SSTIterator() {}
  SSTIterator(typename FileBlock<DataKey, KeyCompT>::EnumIterator&& file_block_iter) 
    : file_block_iter_(std::move(file_block_iter)) {}
  SSTIterator(SSTIterator&& it) noexcept {
    file_block_iter_ = std::move(it.file_block_iter_);
  }
  SSTIterator& operator=(SSTIterator&& it) noexcept {
    file_block_iter_ = std::move(it.file_block_iter_);
    return *this;
  }
  SSTIterator(const SSTIterator& it) : file_block_iter_(it.file_block_iter_) {}
  SSTIterator& operator=(const SSTIterator& it) {
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
  typename FileBlock<DataKey, KeyCompT>::EnumIterator file_block_iter_;
};


// one SST
template <typename KeyCompT, typename ValueT, typename IndexDataT>
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
  std::unique_ptr<RandomAccessFile> file_ptr_;
  FileBlock<BlockKey<SKey, IndexDataT>, KeyCompT> index_block_;
  FileBlock<BlockKey<SKey, ValueT>, KeyCompT> data_block_;
  // LRUCache pointer reference to the one in VisCnts (is deleted)
  KeyCompT comp_;

  
  class FileOpen {
    public:
      FileOpen(RandomAccessFile* file) : file_(file) { fd_ = file_->get_fd(); }
      ~FileOpen() { file_->release_fd(fd_); }
      int get_fd() const { return fd_; } 
      
    private:
      RandomAccessFile* file_;
      int fd_;
  };

  FileOpen open_ra_file_;

  using DataKey = BlockKey<SKey, ValueT>;
  using IndexKey = BlockKey<SKey, IndexDataT>;

 public:
  // class ImmutableFileAsyncSeekHandle {
  //   public:
  //     class IndexBS {
  //       public:
  //         IndexBS(SKey key, KeyCompT comp, int L, int R) 
  //           : key_(key), comp_(comp), L_(L), R_(R) {}
  //         bool end() {
  //           return L_ > R_;
  //         }
  //         int get_mid() {
  //           return (L_ + R_) >> 1;
  //         }
  //         int get_answer() {
  //           return ANS_;
  //         }
  //         /* true if answer is updated. */
  //         bool gen_next(IndexKey candidate_key) {
  //           auto M = get_mid();
  //           // candidate_key.key().print();
  //           // logger(L_, ", ", R_, ", ", M, ", ", ANS_);
  //           if (comp_(candidate_key.key(), key_) <= 0) {
  //             L_ = M + 1;
  //             ANS_ = candidate_key.value();
  //             return true;
  //           } else {
  //             R_ = M - 1;
  //             return false;
  //           }
  //         }
  //       private:
  //         SKey key_;
  //         KeyCompT comp_;
  //         int L_{0}, R_{0};
  //         int ANS_{-1};
  //     };
  //     class DataBS {
  //       public:
  //         DataBS(SKey key, KeyCompT comp, int L, int R) 
  //           : key_(key), comp_(comp), L_(L), R_(R), ANS_(R + 1) {}
  //         bool end() {
  //           return L_ > R_;
  //         }
  //         int get_mid() {
  //           return (L_ + R_) >> 1;
  //         }
  //         int get_answer() {
  //           return ANS_;
  //         }
  //         /* true if answer is updated. */
  //         bool gen_next(DataKey candidate_key) {
  //           auto M = get_mid();
  //           // logger(L_, ", ", R_, ", ", M, ", ", ANS_);
  //           if (comp_(key_, candidate_key.key()) <= 0) {
  //             R_ = M - 1;
  //             ANS_ = M;
  //             return true;
  //           } else {
  //             L_ = M + 1;
  //             return false;
  //           }
  //         }
  //       private:
  //         SKey key_;
  //         KeyCompT comp_;
  //         int L_{0}, R_{0};
  //         int ANS_{0};
  //     };
  //     ImmutableFileAsyncSeekHandle(SKey key, const ImmutableFile& file, async_io::AsyncIOQueue& aio) 
  //       : key_(key), file_(file), aio_(aio) {}
  //     void init() {
  //       state_ = 0;
  //       index_async_seek_handle_ = std::make_unique<typename FileBlock<IndexKey, KeyCompT>::AsyncSeekHandle<IndexBS>>
  //       (file_.index_block_.get_async_seek_handle(aio_, IndexBS(key_, file_.comp_, 0, file_.index_block_.counts() - 1)));
  //       index_async_seek_handle_->init();
  //     }
  //     template<typename T>
  //     std::optional<SSTIterator<KeyCompT, ValueT>> next(T&& aio_info) {
  //       if (state_ == 0) {
  //         auto id = index_async_seek_handle_->search(std::forward<T>(aio_info));
  //         if (!id) {
  //           return {};
  //         }
  //         if (id.value().first == -1) {
  //           return SSTIterator<KeyCompT, ValueT>(file_.data_block_.get_enum_iterator_to_first());
  //         }
  //         id_from_index_ = id.value().first;
  //         // logger((uint64_t)(this), ":", id_from_index_);
  //         index_async_seek_handle_.reset();
  //         data_async_seek_handle_ = 
  //           std::make_unique<typename FileBlock<DataKey, KeyCompT>::AsyncSeekHandle<DataBS>>
  //           (file_.data_block_.template get_async_seek_handle<DataBS>(aio_, DataBS(key_, file_.comp_, id_from_index_, std::min<int>(file_.data_block_.counts() - 1, id_from_index_ + kIndexChunkSize - 1))));
  //         data_async_seek_handle_->init();
  //         state_ = 1;
  //       }
  //       if (state_ == 1) {
  //         auto id = data_async_seek_handle_->search(std::forward<T>(aio_info));
  //         if (!id) {
  //           return {};
  //         }
  //         state_ = 2;
  //         // logger((uint64_t)(this), ":", id.value().first, ", ", id.value().second);
  //         return SSTIterator<KeyCompT, ValueT>(file_.data_block_.get_enum_iterator(id.value().first, id.value().second));
  //       }
  //       return {};
  //     }
  //   private:
  //     int state_{0};
  //     SKey key_;
  //     int id_from_index_;
  //     const ImmutableFile& file_;
  //     async_io::AsyncIOQueue& aio_;
  //     std::unique_ptr<typename FileBlock<IndexKey, KeyCompT>::AsyncSeekHandle<typename ImmutableFile<KeyCompT, ValueT>::ImmutableFileAsyncSeekHandle::IndexBS>> index_async_seek_handle_;
  //     std::unique_ptr<typename FileBlock<DataKey, KeyCompT>::AsyncSeekHandle<DataBS>> data_async_seek_handle_;

  // };

  // Used in seek. We get a temporary fd before seeking, and release it after seeking.
  // We don't use it now.
  // class TempFileOpen {
  //   public:
  //     TempFileOpen(const ImmutableFile<KeyCompT, ValueT, IndexDataT>& file) : file_(file) { fd_ = file_.file_ptr_->get_fd(); }
  //     ~TempFileOpen() { file_.file_ptr_->release_fd(fd_); }
  //     int get_fd() const { return fd_; } 
      
  //   private:
  //     const ImmutableFile<KeyCompT, ValueT, IndexDataT>& file_;
  //     int fd_;
  // };


  ImmutableFile(uint32_t file_id, uint32_t size, std::unique_ptr<RandomAccessFile>&& file_ptr, 
                const std::pair<IndSKey, IndSKey>& range, FileChunkCache* file_index_cache, FileChunkCache* file_key_cache, 
                KeyCompT comp)
      : file_id_(file_id), size_(size), range_(range), file_ptr_(std::move(file_ptr)), comp_(comp), open_ra_file_(file_ptr_.get()) {
    // read index block
    int ra_fd = open_ra_file_.get_fd();
    FileBlockHandle index_bh, data_bh;
    size_t mgn;
    auto ret = file_ptr_->read(ra_fd, size_ - sizeof(size_t), sizeof(size_t), (uint8_t*)(&mgn));
    assert(ret >= 0);
    assert(mgn == kMagicNumber);
    ret = file_ptr_->read(ra_fd, size_ - sizeof(size_t) - sizeof(FileBlockHandle) * 2, sizeof(FileBlockHandle), (uint8_t*)(&index_bh));
    assert(ret >= 0);
    ret = file_ptr_->read(ra_fd, size_ - sizeof(size_t) - sizeof(FileBlockHandle), sizeof(FileBlockHandle), (uint8_t*)(&data_bh));
    assert(ret >= 0);
    index_block_ = FileBlock<IndexKey, KeyCompT>(file_id, index_bh, file_ptr_.get(), file_index_cache, comp_);
    data_block_ = FileBlock<DataKey, KeyCompT>(file_id, data_bh, file_ptr_.get(), file_key_cache, comp_);
  }

  // seek the first key that >= key, but only seek index block.
  typename FileBlock<DataKey, KeyCompT>::EnumIterator estimate_seek(SKey key) {
    if (comp_(range_.second.ref(), key) < 0) return {};
    auto id = index_block_.upper_offset(key, open_ra_file_.get_fd()).get_offset();
    if (id == -1) return {};
    return data_block_.seek_with_id(id, open_ra_file_.get_fd());
  }

  // seek the first key that >= key
  typename FileBlock<DataKey, KeyCompT>::EnumIterator seek(SKey key) const {
    if (comp_(range_.second.ref(), key) < 0) return {};
    auto id = index_block_.get_block_id_from_index(key, open_ra_file_.get_fd()).get_offset();
    if (id == -1) return data_block_.seek_with_id(0, open_ra_file_.get_fd());
    id = data_block_.lower_key(key, id, id + kIndexChunkCount - 1, open_ra_file_.get_fd());
    return data_block_.seek_with_id(id, open_ra_file_.get_fd());
  }

  
  typename FileBlock<DataKey, KeyCompT>::EnumIterator seek_to_first() const {
    return data_block_.seek_with_id(0, open_ra_file_.get_fd());
  }

  // ImmutableFileAsyncSeekHandle get_seek_handle(SKey key, async_io::AsyncIOQueue& aio) const {
  //   return ImmutableFileAsyncSeekHandle(key, *this, aio);
  // }

  std::pair<int, int> rank_pair(const std::pair<SKey, SKey>& range, const std::pair<bool, bool> exclude_info) const {
    int retl = 0, retr = 0;
    auto& [L, R] = range;
    if (comp_(range_.second.ref(), L) < 0)
      retl = data_block_.counts();
    else if (comp_(L, range_.first.ref()) < 0)
      retl = 0;
    else {
      auto id = index_block_.get_block_id_from_index(L, open_ra_file_.get_fd()).get_offset();
      if (id == -1) {
        retl = 0;
      } else {
        retl = !exclude_info.first 
              ? data_block_.lower_key(L, id, id + kIndexChunkCount - 1, open_ra_file_.get_fd())
              : data_block_.upper_key(L, id, id + kIndexChunkCount - 1, open_ra_file_.get_fd());
      }
    }
    if (comp_(range_.second.ref(), R) <= 0)
      retr = data_block_.counts();
    else if (comp_(R, range_.first.ref()) < 0)
      retr = 0;
    else {
      auto id = index_block_.get_block_id_from_index(R, open_ra_file_.get_fd()).get_offset();
      if (id == -1) {
        retr = 0;
      } else {
        retr = !exclude_info.second
              ? data_block_.upper_key(R, id, id + kIndexChunkCount - 1, open_ra_file_.get_fd())
              : data_block_.lower_key(R, id, id + kIndexChunkCount - 1, open_ra_file_.get_fd());
      }
    }

    return {retl, retr};
  }

  // calculate number of elements in [L, R]
  int range_count(const std::pair<SKey, SKey>& range) {
    auto [retl, retr] = rank_pair(range, {0, 0});
    return retr - retl;
  }

  bool in_range(SKey key) {
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

template<typename ValueT, typename IndexDataT>
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

  using DataKey = BlockKey<SKey, ValueT>;
  using IndexKey = BlockKey<SKey, IndexDataT>;

 public:
  SSTBuilder(std::unique_ptr<WriteBatch>&& file = nullptr) : 
    file_(std::move(file)), now_offset(0), lst_offset(0), counts(0), size_(0) {}
  void append(const DataKey& kv) {
    assert(kv.key().len() > 0);
    _append_align(kv.size());
    if (!offsets.size() || offsets.size() % kIndexChunkCount == 0) {
      // If it is the first element, then the second parameter (cumulative sum) should be empty.
      // Otherwise, we use the data of the last element.
      IndexDataT new_index_block(offsets.size(), index.size() ? index.back().second : IndexDataT());
      index.emplace_back(kv.key(), new_index_block);
      if (offsets.size() == 0) first_key = kv.key();
    }
    offsets.push_back(now_offset);
    index.back().second.add(kv.key(), kv.value());
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

  template<typename T>
  void set_lstkey(const T& key) {
    lst_key = key;
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

  size_t get_write_bytes() const {
    return file_ ? file_->get_stat_flushed_size() : 0;
  }

 private:
  uint32_t now_offset, lst_offset, counts, size_;
  std::vector<std::pair<IndSKey, IndexDataT>> index;
  std::vector<uint32_t> offsets;
  IndSKey lst_key, first_key;
};

}

#endif