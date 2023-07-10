
#ifndef __KEY_H__
#define __KEY_H__
#include "common.hpp"
#include <random>

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
  double counts{0};
  size_t vlen{0};
  SValue() {}
  SValue(double _counts, size_t _vlen) : counts(_counts), vlen(_vlen) {}
  void merge(const SValue& v, double) {
    counts += v.counts;
  }
  size_t get_hot_size() const {
    return vlen;
  }
  double get_tick() const {
    return counts;
  }
  bool decay(double prob, std::mt19937_64& rgen) {
    counts *= prob; 
    if (counts < 1) {
      std::uniform_real_distribution<> dis(0, 1.);
      if (dis(rgen) < counts) {
        return false;
      }
      counts = 1;
    }
    return true;
  }
};

struct TickValue {
  double tick{0};
  size_t vlen{0};
  TickValue() {}
  TickValue(double _tick, size_t _vlen) : tick(_tick), vlen(_vlen) {}
  void merge(const TickValue& v, double cur_tick) {
    tick = cur_tick - 1 / (1 / (cur_tick - tick) + 1 / (cur_tick - v.tick));
  }
  size_t get_hot_size() const {
    return vlen;
  }
  double get_tick() const {
    return tick;
  }
  bool decay(double, std::mt19937_64&) {
    return true;
  }
};

struct LRUTickValue {
  double tick{0};
  size_t vlen{0};
  LRUTickValue() {}
  LRUTickValue(double _tick, size_t _vlen) : tick(_tick), vlen(_vlen) {}
  void merge(const LRUTickValue& v, double cur_tick) {
    tick = std::max(tick, v.tick);
  }
  size_t get_hot_size() const {
    return vlen;
  }
  double get_tick() const {
    return tick;
  }
  bool decay(double, std::mt19937_64&) {
    return true;
  }
};

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

template <typename KeyT, typename ValueT>
class BlockKey {
 private:
  KeyT key_;
  ValueT v_;

 public:
  BlockKey() : key_(), v_() {}
  BlockKey(KeyT key, ValueT v) : key_(key), v_(v) {}
  const uint8_t* read(const uint8_t* from) {
    from = key_.read(from);
    v_ = *reinterpret_cast<const ValueT*>(from);
    return from + sizeof(ValueT);
  }
  size_t size() const { return key_.size() + sizeof(v_); }
  uint8_t* write(uint8_t* to) const {
    to = key_.write(to);
    *reinterpret_cast<ValueT*>(to) = v_;
    return to + sizeof(ValueT);
  }
  KeyT key() const { return key_; }
  ValueT value() const { return v_; }
  static size_t read_size(uint8_t* from) {
    size_t ret = KeyT::read_size(from);
    return ret + sizeof(ValueT);
  }
};

// using DataKey = BlockKey<SKey, SValue>;
// using IndexKey = BlockKey<SKey, uint32_t>;
// using RefDataKey = BlockKey<SKey, SValue*>;

}  // namespace viscnts_lsm

#endif