#ifndef VISCNTS_FILEBLOCK_H__
#define VISCNTS_FILEBLOCK_H__

#include "alloc.hpp"
#include "fileenv.hpp"
#include "chunk.hpp"

namespace viscnts_lsm {

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
  uint32_t upper_offset(SKey key) const {
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
  int get_block_id_from_index(SKey key) const {
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
  uint32_t lower_key(SKey key, uint32_t L, uint32_t R) const {
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
  uint32_t upper_key(SKey key, uint32_t L, uint32_t R) const {
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

  bool search_key(SKey key) const {
    int l = 0, r = handle_.counts - 1;
    int ret = -1;
    SeekIterator it = SeekIterator(*this);
    KV _key;
    while (l <= r) {
      auto mid = (l + r) >> 1;
      it.seek_and_read(mid, _key);
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
  EnumIterator seek_with_id(uint32_t id) const {
    SeekIterator it = SeekIterator(*this);
    return EnumIterator(*this, it.seek_offset(id), id);
  }

  size_t counts() const { return handle_.counts; }

  size_t size() const { return handle_.size; }
};

}

#endif