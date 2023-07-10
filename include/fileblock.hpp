#ifndef VISCNTS_FILEBLOCK_H__
#define VISCNTS_FILEBLOCK_H__

#include "alloc.hpp"
#include "fileenv.hpp"
#include "chunk.hpp"
#include "asyncio.hpp"
#include "cache.hpp"

namespace viscnts_lsm {

// two types of fileblock, one stores (key size, key), the other stores (key size, key, value)
template <typename KV, typename KVComp>
class FileBlock {     // process blocks in a file
  uint32_t file_id_;  // file id
  FileBlockHandle handle_;
  RandomAccessFile* file_ptr_;
  FileChunkCache* file_cache_{nullptr};

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
    void seek_and_read(uint32_t id, KV& key, int ra_fd) {
      uint32_t _offset;
      // find offset
      auto [chunk_id, offset] = block_._pos_offset(id);
      if (current_value_id_ != chunk_id) block_.ra_acquire_with_cache(current_value_id_ = chunk_id, currenct_value_chunk_, current_value_chunk_ref_, ra_fd);
      _offset = block_.read_value(offset, current_value_chunk_ref_);
      // read key
      auto [chunk_key_id, key_offset] = block_._kv_offset(_offset);
      if (current_key_id_ != chunk_key_id) block_.ra_acquire_with_cache(current_key_id_ = chunk_key_id, current_key_chunk_, current_key_chunk_ref_, ra_fd);
      block_.read_key(key_offset, current_key_chunk_ref_, key);
    }

    auto seek_offset(uint32_t id, int ra_fd) {
      // find offset
      auto [chunk_id, offset] = block_._pos_offset(id);
      if (current_value_id_ != chunk_id) block_.ra_acquire_with_cache(current_value_id_ = chunk_id, currenct_value_chunk_, current_value_chunk_ref_, ra_fd);
      auto ret = block_.read_value(offset, current_value_chunk_ref_);
      return ret;
    }

   private:
    FileBlock<KV, KVComp> block_;
    RefChunk current_value_chunk_ref_, current_key_chunk_ref_;
    Chunk currenct_value_chunk_, current_key_chunk_;
    uint32_t current_value_id_, current_key_id_;
  };

  // Do something like coroutine..
  // SearchDataType: used to get binary search data.
  template<typename SearchDataType>
  class AsyncSeekHandle {
    public:
    AsyncSeekHandle(FileBlock<KV, KVComp> block, async_io::AsyncIOQueue& aio, const SearchDataType& bs_data) 
      : block_(block), current_value_id_(-1), current_key_id_(-1), aio_(aio), bs_data_(bs_data) {
        fd_ = block_.file_ptr_->get_fd();
      }
    
    ~AsyncSeekHandle() {
      if (fd_ >= 0) {
        block_.file_ptr_->release_fd(fd_);
      }
    }

    AsyncSeekHandle& operator=(AsyncSeekHandle&& handle) = delete;

    AsyncSeekHandle(AsyncSeekHandle&& handle) : aio_(handle.aio_), bs_data_(std::move(handle.bs_data_)) { 
      fd_ = handle.fd_;
      handle.fd_ = -1;
      block_ = std::move(handle.block_);
      currenct_value_chunk_ = std::move(handle.currenct_value_chunk_);
      current_key_chunk_ = std::move(handle.current_key_chunk_);
      current_value_id_ = handle.current_value_id_;
      tmp_offset_ = handle.tmp_offset_;
      ans_offset_ = handle.ans_offset_;
      candidate_ans_offset_ = handle.candidate_ans_offset_;
      state_ = handle.state_;
    }

    AsyncSeekHandle& operator=(const AsyncSeekHandle& handle) = delete;
    
    AsyncSeekHandle(const AsyncSeekHandle& handle) = delete;

    void init() {
      state_ = 0;
    }
    
    template<typename T>
    std::optional<std::pair<uint32_t, uint32_t>> search(T&& aio_info) {
      while(!bs_data_.end()) {
        if (state_ == 0) {
          auto mid = bs_data_.get_mid();
          if (find_offset_begin(mid, std::forward<T>(aio_info))) {
            state_ = 1;
          } else {
            // need wait.
            // wait for the next call.
            state_ = 1;
            return {};
          }
        }
        if (state_ == 1) {
          find_offset_end();
          state_ = 2;
        }
        if (state_ == 2) {
          if (read_key_begin(std::forward<T>(aio_info))) {
            state_ = 3;
          } else {
            // need wait.
            // wait for the next call.
            state_ = 3;
            return {};
          }
        }
        if (state_ == 3) {
          read_key_end();
          state_ = 0;
        }  
      }
      return std::make_pair(bs_data_.get_answer(), ans_offset_);
    }

   private:
    // Expect, call order: find_offset_begin, find_offset_end, read_key_begin, read_key_end.
    template<typename T>
    bool find_offset_begin(uint32_t id, T&& aio_info) {
      // find offset
      auto [chunk_id, offset] = block_._pos_offset(id);
      tmp_offset_ = offset;
      if (current_value_id_ != chunk_id) {
        // logger((uint64_t)(this), ", curr_value_id: ", current_value_id_);
        current_value_id_ = chunk_id;
        block_.async_acquire(fd_, chunk_id, currenct_value_chunk_, aio_, std::forward<T>(aio_info));
        return false;
      }
      // don't need to wait.
      return true;
    }

    void find_offset_end() {
      tmp_offset_ = block_.read_value(tmp_offset_, currenct_value_chunk_);
      candidate_ans_offset_ = tmp_offset_;
    }

    template<typename T>
    bool read_key_begin(T&& aio_info) {
      // read key
      auto [chunk_key_id, key_offset] = block_._kv_offset(tmp_offset_);
      tmp_offset_ = key_offset;
      if (current_key_id_ != chunk_key_id) {
        // logger("current_key_id", current_key_id_);
        current_key_id_ = chunk_key_id;
        block_.async_acquire(fd_, chunk_key_id, current_key_chunk_, aio_, std::forward<T>(aio_info));
        return false;
      }
      // don't need to wait.
      return true;
    }

    void read_key_end() {
      KV _key;
      block_.read_key(tmp_offset_, current_key_chunk_, _key);
      /* if the answer is updated, then we update the corresponding offset. */
      if(bs_data_.gen_next(_key)) {
        ans_offset_ = candidate_ans_offset_;
      }
    }
    
    int fd_{-1};
    FileBlock<KV, KVComp> block_;
    Chunk currenct_value_chunk_, current_key_chunk_;
    uint32_t current_value_id_, current_key_id_;
    async_io::AsyncIOQueue& aio_;
    uint32_t tmp_offset_{0};
    uint32_t ans_offset_{0};
    uint32_t candidate_ans_offset_{0};
    int state_{0};
    /* for binary search. */
    SearchDataType bs_data_;
  };

  // maintain one Chunk for key. Thus, we need the offset of the first key, and its id.
  class EnumIterator {
   public:
    EnumIterator() { id_ = block_.handle_.counts = 0; }
    EnumIterator(std::unique_ptr<SeqFile> seqfile, FileBlock<KV, KVComp> block, uint32_t offset, uint32_t id) 
      : seqfile_(std::move(seqfile)), block_(block), id_(id), key_size_(0) {
      // logger_printf("EI[%d, %d]", id_, block_.counts());
      if (valid()) {
        auto [chunk_id, chunk_offset] = block_._kv_offset(offset);
        current_key_id_ = chunk_id;
        offset_ = chunk_offset;
        seqfile_->seek(current_key_id_ * kChunkSize);
        block_.seq_acquire(current_key_chunk_, seqfile_.get());
      }
    }
    EnumIterator(const EnumIterator& it) = delete;

    EnumIterator& operator=(const EnumIterator& it) = delete;

    EnumIterator(EnumIterator&& it) noexcept { (*this) = std::move(it); }

    EnumIterator& operator=(EnumIterator&& it) noexcept {
      block_ = it.block_;
      offset_ = it.offset_;
      current_key_id_ = it.current_key_id_;
      id_ = it.id_;
      seqfile_ = std::move(it.seqfile_);
      current_key_chunk_ = std::move(it.current_key_chunk_);
      assert(it.current_key_chunk_.data() == nullptr);
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
        block_.seq_acquire(current_key_chunk_, seqfile_.get());
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
      if (new_id - id_ < kIndexChunkSize * 5) {
        while (id_ < new_id) next();
      } else {
        SeekIterator it = SeekIterator(block_);
        id_ = new_id;
        offset_ = it.seek_offset(id_, seqfile_->get_fd());
        auto [chunk_id, chunk_offset] = block_._kv_offset(offset_);
        current_key_id_ = chunk_id;
        offset_ = chunk_offset;
        seqfile_->seek(current_key_id_ * kChunkSize);
        block_.seq_acquire(current_key_chunk_, seqfile_.get());
      }
    }

   private:
    std::unique_ptr<SeqFile> seqfile_;
    FileBlock<KV, KVComp> block_;
    Chunk current_key_chunk_;
    uint32_t current_key_id_, offset_, id_, key_size_;

    void _read_size() { key_size_ = block_.read_key_size(offset_, current_key_chunk_); }
  };

  FileBlock() {
    file_id_ = 0;
    file_ptr_ = nullptr;
    offset_index_ = 0;
  }
  explicit FileBlock(uint32_t file_id, FileBlockHandle handle, RandomAccessFile* file_ptr, FileChunkCache* file_cache, const KVComp& comp)
      : file_id_(file_id), handle_(handle), file_ptr_(file_ptr), file_cache_(file_cache), comp_(comp) {
    offset_index_ = handle_.offset + handle_.size - handle_.counts * sizeof(uint32_t);
    assert(offset_index_ % kChunkSize == 0);
  }

  explicit FileBlock(uint32_t file_id, FileBlockHandle handle, RandomAccessFile* file_ptr, uint32_t offset_index,
                     const KVComp& comp)
      : file_id_(file_id), handle_(handle), file_ptr_(file_ptr), comp_(comp) {
    offset_index_ = offset_index;
  }

  void ra_acquire(size_t id, Chunk& c, int ra_fd) {
    assert(id * kChunkSize < handle_.offset + handle_.size);
    c.acquire(id * kChunkSize, file_ptr_, ra_fd);
  }

  void ra_acquire_with_cache(size_t id, Chunk& c, RefChunk& ref, int ra_fd) {
    assert(id * kChunkSize < handle_.offset + handle_.size);
    size_t key = id << 32 | file_id_;
    if (file_cache_) {
      auto result = file_cache_->try_get_cache(key);
      if (result.has_value()) {
        ref = std::move(result.value());
      } else {
        c.acquire(id * kChunkSize, file_ptr_, ra_fd);
        file_cache_->insert(key, c);
        ref = RefChunk(c, nullptr);
      }  
    } else {
      c.acquire(id * kChunkSize, file_ptr_, ra_fd);
    }
    
  }

  
  void seq_acquire(Chunk& c, SeqFile* seqfile) {
    c.acquire(seqfile);
  }

  template<typename T>
  void async_acquire(int ra_fd, size_t id, Chunk& c, async_io::AsyncIOQueue& aio, T&& aio_info) {
    assert(id * kChunkSize < handle_.offset + handle_.size);
    c.allocate();
    aio.read(ra_fd, c.data(), id * kChunkSize, kChunkSize, std::forward<T>(aio_info));
  }

  template<typename T>
  AsyncSeekHandle<T> get_async_seek_handle(async_io::AsyncIOQueue& aio, const T& bs_data) const {
    return AsyncSeekHandle<T>(*this, aio, bs_data);
  }

  // assume that input is valid, read_key_offset and read_value_offset
  template <typename ChunkT, typename T>
  void read_key(uint32_t offset, const ChunkT& c, T& result) const {
    assert(offset < kChunkSize);
    result.read(c.data(offset));
  }

  uint32_t read_key_size(uint32_t offset, const Chunk& c) const { return KV::read_size(c.data(offset)); }

  template<typename T>
  uint32_t read_value(uint32_t offset, const T& c) const {
    assert(offset < kChunkSize);
    return *reinterpret_cast<const uint32_t*>(c.data(offset));
  }

  bool is_empty_key(uint32_t offset, const Chunk& c) const { return *reinterpret_cast<uint32_t*>(c.data(offset)) == 0; }

  // it's only used for IndexKey, i.e. BlockKey<uint32_t>, so that the type of kv.value() is uint32_t.
  uint32_t upper_offset(SKey key, int ra_fd) const {
    int l = 0, r = handle_.counts - 1;
    uint32_t ret = -1;
    SeekIterator it = SeekIterator(*this);
    KV _key;
    while (l <= r) {
      auto mid = (l + r) >> 1;
      it.seek_and_read(mid, _key, ra_fd);
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
  int get_block_id_from_index(SKey key, int ra_fd) const {
    int l = 0, r = handle_.counts - 1;
    int ret = -1;
    SeekIterator it = SeekIterator(*this);
    KV _key;
    while (l <= r) {
      auto mid = (l + r) >> 1;
      it.seek_and_read(mid, _key, ra_fd);
      // compare two keys
      if (comp_(_key.key(), key) <= 0) {
        l = mid + 1, ret = _key.value();
      } else
        r = mid - 1;
    }
    return ret;
  }

  // it calculates the smallest No. of the key that >= input key.
  uint32_t lower_key(SKey key, uint32_t L, uint32_t R, int ra_fd) const {
    int l = L, r = std::min(R, handle_.counts - 1);
    uint32_t ret = r + 1;
    SeekIterator it = SeekIterator(*this);
    KV _key;
    while (l <= r) {
      auto mid = (l + r) >> 1;
      it.seek_and_read(mid, _key, ra_fd);
      // compare two keys
      if (comp_(key, _key.key()) <= 0) {
        r = mid - 1, ret = mid;
      } else
        l = mid + 1;
    }
    return ret;
  }

  // it calculates the smallest No. of the key that > input key.
  uint32_t upper_key(SKey key, uint32_t L, uint32_t R, int ra_fd) const {
    int l = L, r = std::min(R, handle_.counts - 1);
    uint32_t ret = r + 1;
    SeekIterator it = SeekIterator(*this);
    KV _key;
    while (l <= r) {
      auto mid = (l + r) >> 1;
      it.seek_and_read(mid, _key, ra_fd);
      // compare two keys
      if (comp_(key, _key.key()) < 0) {
        r = mid - 1, ret = mid;
      } else
        l = mid + 1;
    }
    return ret;
  }

  bool search_key(SKey key, int ra_fd) const {
    int l = 0, r = handle_.counts - 1;
    int ret = -1;
    SeekIterator it = SeekIterator(*this);
    KV _key;
    while (l <= r) {
      auto mid = (l + r) >> 1;
      it.seek_and_read(mid, _key, ra_fd);
      auto result = comp_(_key.key(), key);
      // compare two keys
      if (result <= 0) {
        if (result == 0) {
          return true;
        }
        l = mid + 1;
      } else
        r = mid - 1;
    }
    return false;

  }

  // seek with the offset storing the offset of the key.
  EnumIterator seek_with_id(uint32_t id, int ra_fd) const {
    SeekIterator it = SeekIterator(*this);
    return EnumIterator(std::unique_ptr<SeqFile>(file_ptr_->get_seqfile()), *this, it.seek_offset(id, ra_fd), id);
  }

  EnumIterator get_enum_iterator(uint32_t id, uint32_t offset) const {
    return EnumIterator(std::unique_ptr<SeqFile>(file_ptr_->get_seqfile()), *this, offset, id);
  }

  EnumIterator get_enum_iterator_to_first() const {
    return EnumIterator(std::unique_ptr<SeqFile>(file_ptr_->get_seqfile()), *this, 0, 0);
  }

  size_t counts() const { return handle_.counts; }

  size_t size() const { return handle_.size; }
};

}

#endif