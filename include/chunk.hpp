#ifndef VISCNTS_CHUNK_H__
#define VISCNTS_CHUNK_H__


#include "key.hpp"
#include "alloc.hpp"
#include "file.hpp"
#include "logging.hpp"

namespace viscnts_lsm {

const static size_t kChunkSize = 1 << 12;  // 8 KB
const static size_t kIndexChunkSize = 1 << 12;

// manage a temporary chunk (several pages) in SST files
// I don't use cache here, because we don't need it? (I think we can cache index blocks)
class Chunk {
  uint8_t* data_;
  BaseAllocator* alloc_;

 public:
  Chunk() {
    data_ = nullptr;
    alloc_ = nullptr;
  }
  // read a chunk from file
  Chunk(uint32_t offset, BaseAllocator* alloc, RandomAccessFile* file_ptr) {
    data_ = nullptr;
    acquire(offset, alloc, file_ptr);
  }
  Chunk(const Chunk& c) {
    alloc_ = c.alloc_;
    data_ = alloc_->allocate(kChunkSize);
    memcpy(data_, c.data_, kChunkSize);
  }
  Chunk& operator=(const Chunk& c) {
    alloc_ = c.alloc_;
    if (!data_) data_ = alloc_->allocate(kChunkSize);
    memcpy(data_, c.data_, kChunkSize);
    return (*this);
  }
  Chunk(Chunk&& c) {
    data_ = c.data_;
    alloc_ = c.alloc_;
    c.data_ = nullptr;
  }
  Chunk& operator=(Chunk&& c) {
    if (data_) alloc_->release(data_);
    data_ = c.data_;
    alloc_ = c.alloc_;
    c.data_ = nullptr;
    return (*this);
  }
  ~Chunk() {
    if (data_) {
      alloc_->release(data_);
    }
  }
  uint8_t* data(uint32_t offset = 0) const { return data_ + offset; }

  // read a chunk from file, reuse the allocated data
  void acquire(uint32_t offset, BaseAllocator* alloc, RandomAccessFile* file_ptr) {
    if (!data_) {
      data_ = alloc->allocate(kChunkSize);
      alloc_ = alloc;
    }
    Slice result;
    auto err = file_ptr->read(offset, kChunkSize, data_, result);
    assert(result.data() == data_);
    if (err) {
      logger("error in Chunk::Chunk(): ", err);
      exit(-1);
      data_ = nullptr;
      return;
    }
    if (result.len() != kChunkSize) {
      // logger("acquire < kChunkSize");
      memset(data_ + result.len(), 0, kChunkSize - result.len());
    }
  }
};


}

#endif