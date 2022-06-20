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
  explicit Slice(uint8_t* a, size_t len) : a_(a), len_(len) {}
  size_t size() const { return len_ + sizeof(size_t); }
  uint8_t* data() const { return a_; }
  bool operator==(const Slice& S) const { return S.len_ == len_ && memcmp(S.a_, a_, len_) == 0; }
  bool operator!=(const Slice& S) const { return S.len_ != len_ || memcmp(S.a_, a_, len_) != 0; }
};

}  // namespace viscnts_lsm

#endif