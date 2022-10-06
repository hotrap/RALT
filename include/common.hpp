#ifndef __COMMON_H___
#define __COMMON_H___
#include <stdint.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>

namespace viscnts_lsm {

class Slice {
  uint8_t* a_;
  uint32_t len_;

 public:
  Slice() : a_(nullptr), len_(0) {}
  explicit Slice(uint8_t* a, size_t len) : a_(a), len_(len) {}
  size_t size() const { return len_ + sizeof(len_); }
  size_t len() const { return len_; }
  uint8_t* data() const { return a_; }
  bool operator==(const Slice& S) const { return S.len_ == len_ && memcmp(S.a_, a_, len_) == 0; }
  bool operator!=(const Slice& S) const { return S.len_ != len_ || memcmp(S.a_, a_, len_) != 0; }
  uint8_t* read(uint8_t* from) {
    len_ = *reinterpret_cast<decltype(len_)*>(from);
    a_ = from + sizeof(len_);
    return from + sizeof(len_) + len_;
  }
  uint8_t* write(uint8_t* to) const {
    *reinterpret_cast<decltype(len_)*>(to) = len_;
    to += sizeof(len_);
    assert(a_ != nullptr);
    memcpy(to, a_, len_);
    to += len_;
    return to;
  }
  static size_t read_size(uint8_t* from) {
    auto ret = *reinterpret_cast<decltype(len_)*>(from);
    return ret + sizeof(len_);
  }
  void print() {
    printf("[len = %d]", len_);
    for (int i = 0; i < len_; i++) printf("[%x]", a_[i]);
    puts("");
    fflush(stdout);
  }
};
// Independent slice
class IndSlice {
  uint8_t* a_;
  uint32_t len_;

 public:
  IndSlice() : a_(nullptr), len_(0) {}
  explicit IndSlice(const uint8_t* a, size_t len) : a_(a == nullptr ? nullptr : new uint8_t[len]), len_(len) {
    // printf("IndSlice(%lld)!", a_);
    if (a_) memcpy(a_, a, len);
  }
  IndSlice(IndSlice&& s) noexcept : a_(s.a_), len_(s.len_) {
    // printf("IndSlice&&(%lld)!", a_);
    s.a_ = nullptr;
    s.len_ = 0;
  }
  IndSlice(const IndSlice& s) noexcept : a_(s.a_ == nullptr ? nullptr : new uint8_t[s.len_]), len_(s.len_) {
    // printf("IndSlice&(%lld)!", a_);
    if (a_) memcpy(a_, s.a_, s.len_);
  }
  IndSlice& operator=(IndSlice&& s) noexcept {
    // printf("IndSlice&&=(%lld)!", a_);
    // fflush(stdout);
    if (a_) delete a_;
    a_ = s.a_;
    len_ = s.len_;
    s.a_ = nullptr;
    s.len_ = 0;
    return (*this);
  }
  IndSlice& operator=(const IndSlice& s) noexcept {
    // printf("IndSlice&=(%lld)!", a_);
    // fflush(stdout);
    if (a_) delete a_;
    a_ = s.data() == nullptr ? nullptr : new uint8_t[s.len()];
    len_ = s.len();
    if (a_) memcpy(a_, s.data(), s.len());
    return (*this);
  }

  IndSlice(const Slice& s) noexcept : a_(s.data() == nullptr ? nullptr : new uint8_t[s.len()]), len_(s.len()) {
    if (a_) memcpy(a_, s.data(), s.len());
  }
  IndSlice& operator=(const Slice& s) noexcept {
    if (a_) delete a_;
    a_ = s.data() == nullptr ? nullptr : new uint8_t[s.len()];
    len_ = s.len();
    if (a_) memcpy(a_, s.data(), s.len());
    return (*this);
  }
  ~IndSlice() {
    // printf("~IndSlice(%lld)!", a_);
    // fflush(stdout);
    if (a_) delete a_;
  }
  size_t size() const { return len_ + sizeof(len_); }
  size_t len() const { return len_; }
  uint8_t* data() const { return a_; }
  Slice ref() const { return Slice(a_, len_); }
  uint8_t* read(uint8_t* from) {
    auto nlen = *reinterpret_cast<decltype(len_)*>(from);
    if (a_ && len_ != nlen) {
      delete a_;
      a_ = new uint8_t[len_];
    }
    len_ = nlen;
    memcpy(a_, from + sizeof(len_), len_);
    return from + sizeof(len_) + len_;
  }
  uint8_t* write(uint8_t* to) const {
    *reinterpret_cast<decltype(len_)*>(to) = len_;
    to += sizeof(len_);
    memcpy(to, a_, len_);
    to += len_;
    return to;
  }
  static size_t read_size(uint8_t* from) {
    auto ret = *reinterpret_cast<decltype(len_)*>(from);
    return ret + sizeof(len_);
  }
};

class RefCounts {
  std::atomic<uint32_t> __ref_counts;

 public:
  RefCounts() { __ref_counts = 1; }
  RefCounts& operator=(RefCounts&& r) {
    __ref_counts.store(r.__ref_counts.load());
    return (*this);
  }
  virtual ~RefCounts() = default;
  void ref() { __ref_counts.fetch_add(1, std::memory_order_seq_cst); }
  void unref() {
    if (!--__ref_counts) {
      delete this;
    }
  }
};

}  // namespace viscnts_lsm

#endif