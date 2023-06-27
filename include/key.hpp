
#ifndef __KEY_H__
#define __KEY_H__
#include "common.hpp"

namespace viscnts_lsm {

using SKey = Slice;
using IndSKey = IndSlice;

// inline int operator<=(SKey A, SKey B) {
//   if (A.len() != B.len()) return A.len() < B.len() ? -1 : 1;
//   return memcmp(A.data(), B.data(), A.len()) <= 0;
// }

// inline int operator<=(const IndSKey& A, SKey B) { return A.ref() <= B; }
// inline int operator<=(SKey A, const IndSKey& B) { return A <= B.ref(); }
// inline int operator<=(const IndSKey& A, const IndSKey& B) { return A.ref() <= B.ref(); }

// inline int operator==(const IndSKey& A, SKey B) { return A.ref() == B; }
// inline int operator==(SKey A, const IndSKey& B) { return A == B.ref(); }
// inline int operator==(const IndSKey& A, const IndSKey& B) { return A.ref() == B.ref(); }

struct SValue {
  double counts;
  size_t vlen;
  SValue() : counts(0), vlen(0) {}
  SValue(double _counts, size_t _vlen) : counts(_counts), vlen(_vlen) {}
};

inline SValue& operator+=(SValue& a, const SValue& v) {
  a.counts += v.counts;
  return a;
}

struct SKeyComparator {
  int operator()(SKey A, SKey B) const {
    if (A.len() != B.len()) return A.len() < B.len() ? -1 : 1;
    return memcmp(A.data(), B.data(), A.len());
  }
  int is_equal(SKey A, SKey B) const {
    if (A.len() != B.len()) return -1;
    return memcmp(A.data(), B.data(), A.len());
  }
};

template <typename Value>
class BlockKey {
 private:
  SKey key_;
  Value v_;

 public:
  BlockKey() : key_(), v_() {}
  BlockKey(SKey key, Value v) : key_(key), v_(v) {}
  const uint8_t* read(const uint8_t* from) {
    from = key_.read(from);
    v_ = *reinterpret_cast<const Value*>(from);
    return from + sizeof(Value);
  }
  size_t size() const { return key_.size() + sizeof(v_); }
  uint8_t* write(uint8_t* to) const {
    to = key_.write(to);
    *reinterpret_cast<Value*>(to) = v_;
    return to + sizeof(Value);
  }
  SKey key() const { return key_; }
  Value value() const { return v_; }
  static size_t read_size(uint8_t* from) {
    size_t ret = SKey::read_size(from);
    return ret + sizeof(Value);
  }
};

using DataKey = BlockKey<SValue>;
using IndexKey = BlockKey<uint32_t>;
using RefDataKey = BlockKey<SValue*>;

}  // namespace viscnts_lsm

#endif