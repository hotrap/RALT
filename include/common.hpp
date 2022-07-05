#ifndef __COMMON_H___
#define __COMMON_H___
#include <stdint.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <vector>

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
};
// Independent slice
class IndSlice {
 public:
  uint8_t* a_;
  size_t len_;
  IndSlice() : a_(nullptr), len_(0) {}
  explicit IndSlice(const uint8_t* a, size_t len) : a_(a == nullptr ? nullptr : new uint8_t[len]), len_(len) { if(a_) memcpy(a_, a, len); }
  explicit IndSlice(Slice s) : a_(s.data() == nullptr ? nullptr : new uint8_t[s.len()]), len_(s.len()) { if(a_) memcpy(a_, s.data(), s.len()); }
  IndSlice(IndSlice&& s) : a_(s.a_), len_(s.len_) { s.a_ = nullptr; }
  IndSlice(const IndSlice& s) = default;
  IndSlice& operator=(IndSlice&& s) = default;
  ~IndSlice() { delete[] a_; }
  bool operator==(const Slice& S) const { return S.len_ == len_ && memcmp(S.a_, a_, len_) == 0; }
  bool operator!=(const Slice& S) const { return S.len_ != len_ || memcmp(S.a_, a_, len_) != 0; }
  size_t size() const { return len_ + sizeof(size_t); }
  size_t len() const { return len_; }
  uint8_t* data() const { return a_; }
  Slice ref() const { return Slice(a_, len_); }
};




}  // namespace viscnts_lsm

#endif