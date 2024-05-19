
#ifndef __KEY_H__
#define __KEY_H__
#include "common.hpp"
#include <array>
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

// Dedicated.
class SValue {
  double counts_{0};
  size_t vlen_{0};
 public:
  SValue() {}
  SValue(double counts, size_t vlen, size_t _ = 0) : counts_(counts), vlen_(vlen << 1) {}
  void merge(const SValue& v, double) {
    counts_ += v.counts_;
    set_stable(1);
  }
  size_t get_hot_size() const {
    return vlen_ >> 1;
  }
  double get_score() const {
    return counts_;
  }
  double get_counts() const {
    return counts_;
  }
  bool decay(double prob, std::mt19937_64& rgen) {
    counts_ *= prob; 
    if (counts_ < 1) {
      std::uniform_real_distribution<> dis(0, 1.);
      if (dis(rgen) < counts_) {
        return false;
      }
      counts_ = 1;
    }
    return true;
  }
  int tag() const {
    return 0;
  }
  bool is_stable() const {
    return vlen_ & 1;
  }
  void set_stable(bool x) {
    vlen_ = (vlen_ >> 1) << 1 | x;
  }
};

class TickValue {
  double tick_{0};
  size_t vlen_{0};
 public:
  TickValue() {}
  TickValue(double tick, size_t vlen) : tick_(tick), vlen_(vlen << 1) {}
  void merge(const TickValue& v, double cur_tick) {
    tick_ = cur_tick - 1 / (1 / (cur_tick - tick_) + 1 / (cur_tick - v.tick_));
    set_stable(1);
  }
  size_t get_hot_size() const {
    return vlen_ >> 1;
  }
  double get_score() const {
    return tick_;
  }
  bool decay(double, std::mt19937_64&) {
    return true;
  }
  int tag() const {
    return 0;
  }
  bool is_stable() const {
    return vlen_ & 1;
  }
  void set_stable(bool x) {
    vlen_ = (vlen_ >> 1) << 1 | x;
  }
  size_t get_count() const {
    return 1;
  }
};

class LRUTickValue {
  double tick_{0};
  size_t vlen_{0};
 public:
  LRUTickValue() {}
  LRUTickValue(double tick, size_t vlen, unsigned int init_score = 0, bool init_tag = false) : tick_(tick), vlen_(vlen) {}
  void merge(const LRUTickValue& v, double cur_tick) {
    tick_ = std::max(tick_, v.tick_);
  }
  size_t get_hot_size() const {
    return vlen_;
  }
  double get_score() const {
    return tick_;
  }
  bool decay(double, std::mt19937_64&) {
    return true;
  }
  int tag() const {
    return 0;
  }
  bool is_stable() const {
    return 1;
  }
  void set_stable(bool x) {
  }
  size_t get_count() const {
    return 1;
  }
  void decrease_stable() {

  }
};


class ClockTickValue {
  int c_{0};
  size_t vlen_;
 public:
  ClockTickValue() {}
  ClockTickValue(double tick, size_t vlen, unsigned int init_score = 0, bool init_tag = false) : c_(1), vlen_(vlen) {}
  void merge(const ClockTickValue& v, double cur_tick) {
    c_ += v.c_;
  }
  size_t get_hot_size() const {
    return vlen_;
  }
  double get_score() const {
    return c_;
  }
  bool decay(double, std::mt19937_64&) {
    return true;
  }
  int tag() const {
    return 0;
  }
  bool is_stable() const {
    return 1;
  }
  void set_stable(bool x) {
  }
  size_t get_count() const {
    return 1;
  }
  void decrease_stable() {
    c_ = std::max(c_ - 1, 0);
  }
};


class ExpTickValue {
  double tick_{0};
  double score_{0};
  size_t vlen_{0};
 public:
  ExpTickValue() {}
  ExpTickValue(double tick, size_t vlen, unsigned int init_score, bool init_tag = false) : tick_(tick), score_(1), vlen_(vlen << 15 | init_score << 1 | init_tag) {}
  void merge(const ExpTickValue& v, double cur_tick) {
    set_counter(std::min<int>(10, get_counter() + v.get_counter()));
    vlen_ |= 1;
    if (tick_ < v.tick_) {
      score_ = pow(0.999, v.tick_ - tick_) * score_ + v.score_;
      tick_ = v.tick_;
    } else if (tick_ > v.tick_) {
      score_ = pow(0.999, tick_ - v.tick_) * v.score_ + score_;
    }
  }
  size_t get_hot_size() const {
    return vlen_ >> 15;
  }
  double get_score() const {
    return log(score_) + log(0.9998) * (-tick_) + (1e5) * is_stable();
  }
  bool decay(double, std::mt19937_64&) {
    return true;
  }
  int tag() const {
    return 0;
  }
  bool is_stable() const {
    return (vlen_ & 1) && get_counter() > 0;
  }
  void set_counter(int x) {
    vlen_ = (vlen_ >> 15) << 15 | (x << 1) | (vlen_ & 1);
  }
  int get_counter() const {
    return (vlen_ >> 1) & 32767;
  }
  void decrease_stable() {
    set_counter(std::max<int>(get_counter() - 1, 0));
  }
};

// Not used.
// class Tag2TickValue {
//   double tick_{0};
//   size_t vlen_{0};
//   public:
//   Tag2TickValue() {}
//   Tag2TickValue(double _tick, size_t _vlen, size_t tag) : tick_(_tick), vlen_(tag << 63 | _vlen) {}
//   void merge(const Tag2TickValue& v, double cur_tick) {
//     if (v.tag() != tag()) {
//       if (v.tag() < tag()) {
//         tick_ = v.tick_;
//         vlen_ = v.vlen_;
//       } else {
//         return;
//       }
//     }
//     tick_ = std::max(tick_, v.tick_);
//   }
//   size_t get_hot_size() const {
//     return vlen_ & ((1ull << 63) - 1);
//   }
//   double get_score() const {
//     return tick_;
//   }
//   bool decay(double, std::mt19937_64&) {
//     return true;
//   }
//   int tag() const {
//     return vlen_ >> 63 & 1;
//   }
// };

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

/* KeyT is a variable-length key, and ValueT is fixed size. */
template <typename KeyT, typename ValueT>
class BlockKey {
 private:
  KeyT key_;
  ValueT v_;

 public:
  using KeyType = KeyT;
  using ValueType = ValueT;

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

#pragma pack(4)
template<const int num_tier>
class IndexData {
  std::array<size_t, num_tier> hot_size_{};
  int offset_{0};
  public:
    IndexData() {}
    IndexData(int offset) : offset_(offset) {}

    /* offset is determined when creating this. */
    IndexData(int offset, const IndexData& data) {
      offset_ = offset;
      hot_size_ = data.hot_size_;
    }

    /*add key to this index block*/
    template<typename T>
    void add(const SKey& key, const T& value) {
      hot_size_[value.tag()] += value.get_hot_size() + key.len();
    }

    const std::array<size_t, num_tier>& get_hot_size() const {
      return hot_size_;
    }

    int get_offset() const {
      return offset_;
    }
};
#pragma pack()

// using DataKey = BlockKey<SKey, SValue>;
// using IndexKey = BlockKey<SKey, uint32_t>;
// using RefDataKey = BlockKey<SKey, SValue*>;

}  // namespace viscnts_lsm

#endif
