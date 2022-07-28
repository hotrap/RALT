
#ifndef __KEY_H__
#define __KEY_H__
#include "common.hpp"

namespace viscnts_lsm {

using SKey = Slice;
using IndSKey = IndSlice;

inline int operator<=(const SKey& A, const SKey& B) {
  if (A.len() != B.len()) return A.len() < B.len() ? -1 : 1;
  return memcmp(A.data(), B.data(), A.len()) <= 0;
}

inline int operator<=(const IndSKey& A, const SKey& B) { return A.ref() <= B; }
inline int operator<=(const SKey& A, const IndSKey& B) { return A <= B.ref(); }
inline int operator<=(const IndSKey& A, const IndSKey& B) { return A.ref() <= B.ref(); }

inline int operator==(const IndSKey& A, const SKey& B) { return A.ref() == B; }
inline int operator==(const SKey& A, const IndSKey& B) { return A == B.ref(); }
inline int operator==(const IndSKey& A, const IndSKey& B) { return A.ref() == B.ref(); }

struct SValue {
  double counts;
  size_t vlen;
  SValue() = default;
  SValue(double _counts, size_t _vlen) : counts(_counts), vlen(_vlen) {}
};

inline SValue& operator+=(SValue& a, const SValue& v) {
  a.counts += v.counts;
  return a;
}

struct SKeyComparator {
  int operator()(const SKey& A, const SKey& B) const {
    if (A.len() != B.len()) return A.len() < B.len() ? -1 : 1;
    return memcmp(A.data(), B.data(), A.len());
  }
  int is_equal(const SKey& A, const SKey& B) const {
    if (A.len() != B.len()) return -1;
    return memcmp(A.data(), B.data(), A.len());
  }
};

}  // namespace viscnts_lsm

#endif