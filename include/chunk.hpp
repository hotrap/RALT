#ifndef VISCNTS_CHUNK_H__
#define VISCNTS_CHUNK_H__


#include "key.hpp"
#include "alloc.hpp"
#include "fileenv.hpp"
#include "logging.hpp"

namespace viscnts_lsm {

const static size_t kChunkSize = 1 << 12;  // 4 KB
const static size_t kIndexChunkSize = 1 << 12;

// manage a temporary chunk (several pages) in SST files
// I don't use cache here, because we don't need it? (I think we can cache index blocks)
class Chunk {
  uint8_t* data_{nullptr};

 public:
  Chunk() {}

  Chunk(const Chunk& c) {
    data_ = BaseAllocator::align_alloc(kChunkSize, 4096);
    memcpy(data_, c.data_, kChunkSize);  
  }
  Chunk& operator=(const Chunk& c) {
    if (!data_) data_ = BaseAllocator::align_alloc(kChunkSize, 4096);
    memcpy(data_, c.data_, kChunkSize);
    return (*this);
  }
  Chunk(Chunk&& c) {
    data_ = c.data_;
    c.data_ = nullptr;
  }
  Chunk& operator=(Chunk&& c) {
    if (data_) BaseAllocator::align_release(data_);
    data_ = c.data_;
    c.data_ = nullptr;
    return (*this);
  }
  ~Chunk() {
    if (data_) {
      BaseAllocator::align_release(data_);
    }
  }
  uint8_t* data(uint32_t offset = 0) const { return data_ + offset; }

  // read a chunk from file, reuse the allocated data
  void acquire(uint32_t offset, RandomAccessFile* file_ptr, int ra_fd) {
    allocate();
    auto result = file_ptr->read(ra_fd, offset, kChunkSize, data_);
    if (result != kChunkSize) {
      // logger("acquire < kChunkSize");
      memset(data_ + result, 0, kChunkSize - result);
    }
  }

  
  void acquire(SeqFile* file_ptr) {
    allocate();
    auto result = file_ptr->read(kChunkSize, data_);
    if (result != kChunkSize) {
      // logger("acquire < kChunkSize");
      memset(data_ + result, 0, kChunkSize - result);
    }
  }

  /* allocate an empty chunk. */
  void allocate() {
    if (!data_) {
      data_ = BaseAllocator::align_alloc(kChunkSize, 4096);
    }
  }
};


}

#endif