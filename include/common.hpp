#ifndef __COMMON_H___
#define __COMMON_H___
#include <stdint.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <vector>
#include <atomic>
#include <cstdio>

namespace viscnts_lsm {

class Slice {
 public:
  uint8_t* a_;
  size_t len_;
  Slice() : a_(nullptr), len_(0) {}
  explicit Slice(uint8_t* a, size_t len) : a_(a), len_(len) {}
  size_t size() const { return len_ + sizeof(size_t); }
  size_t len() const { return len_; }
  uint8_t* data() const { return a_; }
  bool operator==(const Slice& S) const { return S.len_ == len_ && memcmp(S.a_, a_, len_) == 0; }
  bool operator!=(const Slice& S) const { return S.len_ != len_ || memcmp(S.a_, a_, len_) != 0; }
  void read(uint8_t* from) {
    len_ = *reinterpret_cast<decltype(len_)*>(from);
    a_ = from + sizeof(len_);
  }
  uint8_t* write(uint8_t* to) const {
    *reinterpret_cast<decltype(len_)*>(to) = len_;
    to += sizeof(len_);
    memcpy(to, a_, len_);
    to += len_;
    return to;
  }
};
// Independent slice
class IndSlice {
 public:
  uint8_t* a_;
  size_t len_;
  IndSlice() : a_(nullptr), len_(0) {}
  explicit IndSlice(const uint8_t* a, size_t len) : a_(a == nullptr ? nullptr : new uint8_t[len]), len_(len) {
    if (a_) memcpy(a_, a, len);
  }
  explicit IndSlice(Slice s) : a_(s.data() == nullptr ? nullptr : new uint8_t[s.len()]), len_(s.len()) {
    if (a_) memcpy(a_, s.data(), s.len());
  }
  IndSlice(IndSlice&& s) : a_(s.a_), len_(s.len_) { s.a_ = nullptr, s.len_ = 0; }
  IndSlice(const IndSlice& s) : a_(s.data() == nullptr ? nullptr : new uint8_t[s.len()]), len_(s.len()) {
    if (a_) memcpy(a_, s.data(), s.len());
  }
  IndSlice& operator=(IndSlice&& s) {
    if (a_) delete a_;
    a_ = s.a_, len_ = s.len_, s.a_ = nullptr, s.len_ = 0;
    return (*this);
  }
  IndSlice& operator=(const Slice& s) {
    if (a_) delete a_;
    a_ = s.data() == nullptr ? nullptr : new uint8_t[s.len()];
    len_ = s.len();
    if (a_) memcpy(a_, s.data(), s.len());
    return (*this);
  }
  virtual ~IndSlice() { delete a_; }
  size_t size() const { return len_ + sizeof(size_t); }
  size_t len() const { return len_; }
  uint8_t* data() const { return a_; }
  Slice ref() const { return Slice(a_, len_); }
  void read(uint8_t* from) {
    len_ = *reinterpret_cast<decltype(len_)*>(from);
    if (a_) delete a_;
    a_ = new uint8_t[len_];
    memcpy(a_, from + sizeof(len_), len_);
  }
  uint8_t* write(uint8_t* to) const {
    *reinterpret_cast<decltype(len_)*>(to) = len_;
    to += sizeof(len_);
    memcpy(to, a_, len_);
    to += len_;
    return to;
  }
};

class RefCounts {
  std::atomic<uint32_t> __ref_counts;

 public:
  RefCounts() { __ref_counts = 1; }
  RefCounts& operator=(RefCounts&& r){
    __ref_counts.store(r.__ref_counts.load());
    return (*this);
  }
  virtual ~RefCounts() = default;
  void ref() { __ref_counts.fetch_add(1, std::memory_order_seq_cst); }
  void unref() {
    if (!--__ref_counts) { delete this; }
  }
};

}  // namespace viscnts_lsm

#endif