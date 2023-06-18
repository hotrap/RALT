#ifndef VISCNTS_ITERATORS_H__
#define VISCNTS_ITERATORS_H__

#include "key.hpp"

namespace viscnts_lsm {

// this iterator is used in compaction
// or decay
class SeqIterator {
 public:
  virtual ~SeqIterator() = default;
  virtual bool valid() = 0;
  virtual void next() = 0;
  virtual std::pair<SKey, SValue> read() = 0;
  virtual SeqIterator* copy() = 0;
};

// A set of iterators, use heap to manage but not segment tree because it can avoid comparisions opportunistically
template <typename Iterator, typename KVComp>
class SeqIteratorSet {
  std::vector<Iterator> iters_;
  std::vector<Iterator*> seg_tree_;
  std::vector<SKey> keys_;
  std::vector<SValue> values_;
  uint32_t size_;
  KVComp comp_;

 public:
  SeqIteratorSet(const KVComp& comp) : size_(0), comp_(comp) {}
  SeqIteratorSet(const SeqIteratorSet& ss) { (*this) = ss; }
  SeqIteratorSet(SeqIteratorSet&& ss) { (*this) = std::move(ss); }
  SeqIteratorSet& operator=(const SeqIteratorSet& ss) {
    for (auto& a : ss.iters_) iters_.emplace_back(a);
    comp_ = ss.comp_;
    build();
    return (*this);
  }
  SeqIteratorSet& operator=(SeqIteratorSet&& ss) {
    iters_ = std::move(ss.iters_);
    seg_tree_ = std::move(ss.seg_tree_);
    keys_ = std::move(ss.keys_);
    values_ = std::move(ss.values_);
    size_ = ss.size_;
    comp_ = ss.comp_;
    return (*this);
  }
  void build() {
    size_ = iters_.size();
    DataKey kv;
    seg_tree_.resize(size_ + 1, nullptr);
    keys_.resize(size_);
    values_.resize(size_);
    for (uint32_t i = 1; i <= iters_.size(); ++i) {
      seg_tree_[i] = &iters_[i - 1];
      seg_tree_[i]->read(kv);
      keys_[i - 1] = std::move(kv.key());
      values_[i - 1] = kv.value();
      for (uint32_t j = i; j > 1 && _min(j, j >> 1) == j; j >>= 1) std::swap(seg_tree_[j], seg_tree_[j >> 1]);
    }
  }
  bool valid() { return size_ >= 1; }
  void next() {
    // logger_printf("[S.Set, next, %d]", size_);
    seg_tree_[1]->next();
    if (!seg_tree_[1]->valid()) {
      if (size_ == 1) {
        size_ = 0;
        return;
      }
      seg_tree_[1] = seg_tree_[size_];
      size_--;
    }

    DataKey kv;
    seg_tree_[1]->read(kv);
    uint32_t id = seg_tree_[1] - iters_.data();
    keys_[id] = std::move(kv.key());
    values_[id] = kv.value();

    uint32_t x = 1;
    while ((x << 1 | 1) <= size_) {
      auto r = _min(x << 1, x << 1 | 1);
      if (_min(r, x) == x) return;
      std::swap(seg_tree_[x], seg_tree_[r]);
      x = r;
    }
    if ((x << 1) <= size_) {
      if (_min(x, x << 1) == (x << 1)) std::swap(seg_tree_[x], seg_tree_[x << 1]);
    }
  }
  std::pair<SKey, SValue> read() {
    int x = seg_tree_[1] - iters_.data();
    return {keys_[x], values_[x]};
  }
  void push(Iterator&& new_iter) {
    if (!new_iter.valid()) return;
    iters_.push_back(std::move(new_iter));
  }

  KVComp comp_func() { return comp_; }

 private:
  uint32_t _min(uint32_t x, uint32_t y) {
    uint32_t idx = seg_tree_[x] - iters_.data();
    uint32_t idy = seg_tree_[y] - iters_.data();
    return comp_(keys_[idx], keys_[idy]) < 0 ? x : y;
  }
};

template <typename Iterator, typename KVComp>
class SeqIteratorSetForScan {
  IndSKey current_key_;
  SValue current_value_;
  SeqIteratorSet<Iterator, KVComp> iter_;
  bool valid_;

 public:
  SeqIteratorSetForScan(SeqIteratorSet<Iterator, KVComp>&& iter) : iter_(std::move(iter)), valid_(true) {}
  void build() {
    iter_.build();
    next();
  }
  std::pair<SKey, SValue> read() { return {current_key_.ref(), current_value_}; }
  void next() {
    if (!iter_.valid()) {
      valid_ = false;
      return;
    }
    auto result = iter_.read();
    current_key_ = result.first;
    current_value_ = result.second;
    iter_.next();
    while (iter_.valid()) {
      result = iter_.read();
      if (iter_.comp_func()(result.first, current_key_.ref()) == 0) {
        current_value_ += result.second;
      } else
        break;
      iter_.next();
    }
  }

  bool valid() { return valid_; }
};

}


#endif