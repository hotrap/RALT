#ifndef VISCNTS_FILEBLOCK_H__
#define VISCNTS_FILEBLOCK_H__

#include "alloc.hpp"
#include "fileenv.hpp"
#include "chunk.hpp"
// #include "asyncio.hpp"
#include "cache.hpp"
#include "bloomfilter.hpp"

namespace viscnts_lsm {

// two types of fileblock, one stores (key size, key), the other stores (key size, key, value)
template <typename KV, typename KVComp>
class FileBlock {     // process blocks in a file
  uint32_t file_id_;  // file id
  FileBlockHandle handle_;
  RandomAccessFile* file_ptr_;
  FileChunkCache* file_cache_{nullptr};

  // A block of size kChunkSize: [data data ...  NULL  offset offset ... offset]

  KVComp comp_;

  constexpr static auto kValuePerChunk = kChunkSize / sizeof(uint32_t);

  // get the chunk id and chunk offset from an offset.
  std::pair<uint32_t, uint32_t> _kv_offset(uint32_t offset) {
    assert(offset < handle_.offset + handle_.size);
    // we must ensure no keys cross two chunks in SSTBuilder.
    // offset is absolute.
    return {offset / kChunkSize, offset % kChunkSize};
  }

 public:
  class EnumIterator;
  // Only seek, maintain two Chunks, one is of key(kv pairs), one is of value(offset).
  class SeekIterator {
   public:
    SeekIterator(FileBlock<KV, KVComp> block) : block_(block), current_chunk_id_(-1) {}
    void seek_and_read(uint32_t chunk_id, uint32_t id, KV& key, int ra_fd) {
      read(chunk_id, seek_offset(chunk_id, id, ra_fd), key, ra_fd);
    }

    void read(uint32_t chunk_id, uint32_t offset, KV& key, int ra_fd) {
      read_chunk(chunk_id, ra_fd);
      block_.read_key(offset, current_chunk_ref_, key);
    }

    void read(uint32_t chunk_id, uint32_t offset, std::pair<IndSKey, typename KV::ValueType>& key, int ra_fd) {
      read_chunk(chunk_id, ra_fd);
      KV _key;
      block_.read_key(offset, current_chunk_ref_, _key);
      key = std::make_pair(_key.key(), _key.value());
    }

    auto seek_offset(uint32_t chunk_id, uint32_t id, int ra_fd) {
      // find offset
      read_chunk(chunk_id, ra_fd);
      auto key_n = *(uint32_t*)current_chunk_ref_.data(kChunkSize - sizeof(uint32_t));
      auto ret = *(uint32_t*)current_chunk_ref_.data(kChunkSize - ((key_n - id) * sizeof(uint32_t) + sizeof(uint32_t)));
      return ret;
    }

    void read_chunk(uint32_t chunk_id, int ra_fd) {
      if (current_chunk_id_ != chunk_id) block_.ra_acquire_with_cache(current_chunk_id_ = chunk_id, current_chunk_, current_chunk_ref_, ra_fd);
    }

    template<typename CompFn>
    std::optional<std::tuple<uint32_t, uint32_t, typename KV::ValueType>> get_maximum_proper_value(uint32_t chunk_id, SKey key, int ra_fd, CompFn comp_func) {
      read_chunk(chunk_id, ra_fd);
      auto key_n = *(uint32_t*)current_chunk_ref_.data(kChunkSize - sizeof(uint32_t));
      int l = 0, r = key_n - 1;
      std::optional<std::tuple<uint32_t, uint32_t, typename KV::ValueType>> ret;
      KV _key;
      while (l <= r) {
        int mid = (l + r) / 2;
        auto offset = seek_offset(chunk_id, mid, ra_fd);
        read(chunk_id, offset, _key, ra_fd);
        // DB_INFO("{}, {}, {} | {}, {}", l, r, mid, block_.comp_(key, _key.key()), comp_func(key, _key.key()));
        if (comp_func(key, _key.key())) {
          l = mid + 1;
        // DB_INFO("offset , mid = {}, {}, {}", offset, mid, BloomFilter::BloomHash(_key.key()));
          ret = std::make_tuple(mid, offset, _key.value());
        } else {
          r = mid - 1;
        }
      }
      return ret;
    }

    
    template<typename CompFn>
    std::optional<std::tuple<uint32_t, uint32_t, typename KV::ValueType>> get_minimum_proper_value(uint32_t chunk_id, SKey key, int ra_fd, CompFn comp_func) {
      read_chunk(chunk_id, ra_fd);
      auto key_n = *(uint32_t*)current_chunk_ref_.data(kChunkSize - sizeof(uint32_t));
      int l = 0, r = key_n - 1;
      std::optional<std::tuple<uint32_t, uint32_t, typename KV::ValueType>> ret;
      KV _key;
      while (l <= r) {
        int mid = (l + r) / 2;
        auto offset = seek_offset(chunk_id, mid, ra_fd);
        read(chunk_id, offset, _key, ra_fd);
        if (comp_func(key, _key.key()) || key_n - 1 == mid) {
          r = mid - 1;
        // DB_INFO("{}, offset , mid = {}, {}, {}", key_n, offset, mid, BloomFilter::BloomHash(_key.key()));
          ret = std::make_tuple(mid, offset, _key.value());
        } else {
          l = mid + 1;
        }
      }
      return ret;
    }

   private:
    FileBlock<KV, KVComp> block_;
    RefChunk current_chunk_ref_;
    Chunk current_chunk_;
    int current_chunk_id_{-1};

    friend class EnumIterator;
  };

  // maintain one Chunk for key. Thus, we need the offset of the first key, and its id.
  class EnumIterator {
   public:
    EnumIterator() { id_ = block_.handle_.counts = 0; }
    EnumIterator(std::unique_ptr<SeqFile> seqfile, FileBlock<KV, KVComp> block, uint32_t chunk_id, uint32_t offset, uint32_t id) 
      : seqfile_(std::move(seqfile)), block_(block), id_(id), key_size_(0) {
      if (valid()) {
        current_chunk_id_ = chunk_id;
        offset_ = offset;
        seqfile_->seek(current_chunk_id_ * kChunkSize);
        block_.seq_acquire(current_chunk_, seqfile_.get());
        current_chunk_end_offset_ = *(uint32_t*)(current_chunk_.data(kChunkSize - sizeof(uint32_t) * 2));
      }
    }
    EnumIterator(const EnumIterator& it) = delete;

    EnumIterator& operator=(const EnumIterator& it) = delete;

    EnumIterator(EnumIterator&& it) noexcept { (*this) = std::move(it); }

    EnumIterator& operator=(EnumIterator&& it) noexcept {
      block_ = it.block_;
      offset_ = it.offset_;
      current_chunk_id_ = it.current_chunk_id_;
      id_ = it.id_;
      seqfile_ = std::move(it.seqfile_);
      current_chunk_ = std::move(it.current_chunk_);
      current_chunk_end_offset_ = it.current_chunk_end_offset_;
      DB_ASSERT(it.current_chunk_.data() == nullptr);
      key_size_ = 0;
      return (*this);
    }

    EnumIterator(SeekIterator&& it, std::unique_ptr<SeqFile> seqfile, uint32_t offset, uint32_t id)
      : seqfile_(std::move(seqfile)) {
      block_ = it.block_;
      current_chunk_ = it.current_chunk_ref_.copy();
      current_chunk_id_ = it.current_chunk_id_;
      offset_ = offset;
      current_chunk_end_offset_ = *(uint32_t*)(current_chunk_.data(kChunkSize - sizeof(uint32_t) * 2));
      id_ = id;
      key_size_ = 0;
      seqfile_->seek((current_chunk_id_ + 1) * kChunkSize);
    }

    void next() {
      // logger_printf("nextEI[%d, %d]", id_, block_.counts());
      assert(valid());
      // logger("[next]", id_, " ", block_.handle_.counts, " ", offset_, " ", current_chunk_end_offset_, " ", *(uint32_t*)current_chunk_.data(kChunkSize - sizeof(uint32_t)));
      if (!key_size_) _read_size();
      offset_ += key_size_;
      assert(offset_ <= kChunkSize);
      key_size_ = 0, id_++;
      if (valid() && (offset_ > current_chunk_end_offset_)) {
        offset_ = 0, current_chunk_id_++;
        block_.seq_acquire(current_chunk_, seqfile_.get());
        current_chunk_end_offset_ = *(uint32_t*)(current_chunk_.data(kChunkSize - sizeof(uint32_t) * 2));
      }
    }
    auto read(KV& key) {
      assert(valid());
      block_.read_key(offset_, current_chunk_, key);
      key_size_ = key.serialize_size();
    }
    bool valid() { return id_ < block_.handle_.counts; }

    int rank() { return id_; }

   private:
    std::unique_ptr<SeqFile> seqfile_;
    FileBlock<KV, KVComp> block_;
    Chunk current_chunk_;
    uint32_t current_chunk_id_, offset_, id_, key_size_;
    uint32_t current_chunk_end_offset_;

    void _read_size() { key_size_ = block_.read_key_size(offset_, current_chunk_); }
  };

  FileBlock() {
    file_id_ = 0;
    file_ptr_ = nullptr;
  }
  explicit FileBlock(uint32_t file_id, FileBlockHandle handle, RandomAccessFile* file_ptr, FileChunkCache* file_cache, const KVComp& comp)
      : file_id_(file_id), handle_(handle), file_ptr_(file_ptr), file_cache_(file_cache), comp_(comp) {
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
      ref = RefChunk(c, nullptr);
    }
    
  }

  
  void seq_acquire(Chunk& c, SeqFile* seqfile) {
    c.acquire(seqfile);
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
  
  // return value: {id, offset, value}.
  template<typename CompFn>
  std::optional<std::tuple<uint32_t, uint32_t, typename KV::ValueType>> get_maximum_proper_value(SKey key, int ra_fd, CompFn comp_func) const {
    int l = handle_.offset / kChunkSize, r = handle_.offset / kChunkSize + handle_.size / kChunkSize - 1, result_id = 0;
    SeekIterator it = SeekIterator(*this);
    KV _key;
    while (l <= r) {
      auto mid = (l + r) >> 1;
      it.seek_and_read(mid, 0, _key, ra_fd);
      // compare two keys
      // true => the value is acceptable.
      // DB_INFO("{}, {}, {}, {}", l, r, mid, comp_(key, _key.key()));
      if (comp_func(key, _key.key())) {
        l = mid + 1, result_id = mid;
      } else {
        r = mid - 1;
      }
    }
    // DB_INFO("result_id = {}", result_id);
    return it.get_maximum_proper_value(result_id, key, ra_fd, comp_func);
  }

  
  template<typename CompFn>
  std::optional<std::tuple<uint32_t, uint32_t, typename KV::ValueType>> get_minimum_proper_value(SKey key, int ra_fd, CompFn comp_func) const {
    int l = handle_.offset / kChunkSize, r = handle_.offset / kChunkSize + handle_.size / kChunkSize - 1, result_id = r;
    SeekIterator it = SeekIterator(*this);
    KV _key;
    while (l <= r) {
      auto mid = (l + r) >> 1;
      it.seek_and_read(mid, 0, _key, ra_fd);
      // compare two keys
      // true => the value is acceptable.
      // DB_INFO("{}, {}, {}, {}", l, r, mid, comp_(key, _key.key()));
      if (comp_func(key, _key.key())) {
        r = mid - 1, result_id = mid;
      } else {
        l = mid + 1;
      }
    }
    // DB_INFO("result_id = {}", result_id);
    return it.get_minimum_proper_value(result_id, key, ra_fd, comp_func);
  }

  EnumIterator lower_in_chunk(SKey key, uint32_t chunk_offset, uint32_t id, int ra_fd) const {
    SeekIterator it = SeekIterator(*this);
    auto pos = it.get_maximum_proper_value(chunk_offset / kChunkSize, key, ra_fd, [&](auto my, auto it) {
      return comp_(it, my) < 0;
    });
    if (!pos) {
      return seek_chunk_begin(chunk_offset / kChunkSize, 0, id);
    }
    auto enum_it = EnumIterator(std::move(it), std::unique_ptr<SeqFile>(file_ptr_->get_seqfile())
                                        , std::get<1>(pos.value())
                                        , id + std::get<0>(pos.value()));
    if (enum_it.valid()) {
      enum_it.next();
    }
    return enum_it;
  }

  size_t get_prefix_size_sum(SKey key, int ra_fd, bool exclude) const {
    if (!exclude) {
      auto result = get_minimum_proper_value(key, ra_fd, [&](auto my, auto it) {
        return comp_(it, my) >= 0;
      });
      return result ? std::get<2>(result.value()).get_hot_size()[0] : 0;
    } else {
      auto result = get_maximum_proper_value(key, ra_fd, [&](auto my, auto it) {
        return comp_(it, my) < 0;
      });
      return result ? std::get<2>(result.value()).get_hot_size()[0] - std::get<2>(result.value()).get_this_hot_size() : 0;
    }
  }

  EnumIterator seek_chunk_begin(uint32_t chunk_id, uint32_t offset, uint32_t id) const {
    return EnumIterator(std::unique_ptr<SeqFile>(file_ptr_->get_seqfile()), *this, chunk_id, offset, id);
  }

  size_t counts() const { return handle_.counts; }

  size_t size() const { return handle_.size; }
};

}

#endif