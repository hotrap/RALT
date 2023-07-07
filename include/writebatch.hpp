#ifndef VISCNTS_WRITEBATCH_H__
#define VISCNTS_WRITEBATCH_H__

#include "fileenv.hpp"
#include "key.hpp"

namespace viscnts_lsm {


// This write data in a buffer, and flush the buffer when it's full. It is used in SSTBuilder.
class WriteBatch {
  // Semantically there can only be one AppendFile for each SST file.
  std::unique_ptr<AppendFile> file_ptr_;
  const size_t buffer_size_;
  size_t used_size_;
  uint8_t* data_;

 public:
  const static size_t kBatchSize = 1 << 20;
  explicit WriteBatch(std::unique_ptr<AppendFile>&& file) : file_ptr_(std::move(file)), buffer_size_(kBatchSize), used_size_(0) {
    data_ = new uint8_t[kBatchSize];
  }
  ~WriteBatch() {
    flush();
    delete[] data_;
  }

  template <typename T>
  void append(const T& kv) noexcept {
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
      append(Slice(const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(&x)), sizeof(T)));
    } else {
      *reinterpret_cast<T*>(data_ + used_size_) = x;
      used_size_ += sizeof(T);
    }
  }

  template <typename T>
  void append_key(const T& x) {
    if (__builtin_expect(used_size_ + x.size() > buffer_size_, 0)) {
      flush();
      assert(x.size() <= buffer_size_);
      x.write(data_);
      used_size_ = x.size();
    } else {
      x.write(data_ + used_size_);
      used_size_ += x.size();
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
    if (used_size_) {
      auto ret = file_ptr_->write(Slice(data_, used_size_));
      (void)ret;
      assert(ret == 0);
    }
    used_size_ = 0;
  }
};

}

#endif