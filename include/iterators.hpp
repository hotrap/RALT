#ifndef VISCNTS_ITERATORS_H__
#define VISCNTS_ITERATORS_H__

#include "key.hpp"
#include "tickfilter.hpp"

namespace viscnts_lsm {

// A set of iterators, use heap to manage but not segment tree because it can avoid comparisions opportunistically
template <typename Iterator, typename KVCompT, typename ValueT>
class SeqIteratorSet {
  std::vector<Iterator> iters_;
  std::vector<Iterator*> seg_tree_;
  std::vector<SKey> keys_;
  std::vector<ValueT> values_;
  uint32_t size_;
  KVCompT comp_;

  using SeqIteratorSetT = SeqIteratorSet<Iterator, KVCompT, ValueT>;

 public:
  SeqIteratorSet(const KVCompT& comp) : size_(0), comp_(comp) {}
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
    BlockKey<SKey, ValueT> kv;
    seg_tree_.resize(size_ + 1, nullptr);
    keys_.resize(size_);
    values_.resize(size_);
    for (uint32_t i = 1; i <= size_; ++i) {
      seg_tree_[i] = &iters_[i - 1];
    }
    for (uint32_t i = 1; i <= size_; i++) {
      while (i < size_ && !seg_tree_[i]->valid()) {
        seg_tree_[i] = seg_tree_[size_];
        size_--;
      }
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
    while (!seg_tree_[1]->valid()) {
      if (size_ == 1) {
        size_ = 0;
        return;
      }
      seg_tree_[1] = seg_tree_[size_];
      size_--;
    }

    BlockKey<SKey, ValueT> kv;
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
  std::pair<SKey, ValueT> read() {
    int x = seg_tree_[1] - iters_.data();
    return {keys_[x], values_[x]};
  }
  void push(Iterator&& new_iter) {
    if (!new_iter.valid()) return;
    iters_.push_back(std::move(new_iter));
  }

  KVCompT comp_func() { return comp_; }

  std::vector<Iterator>& get_iterators() {
    return iters_;
  }

 private:
  uint32_t _min(uint32_t x, uint32_t y) {
    uint32_t idx = seg_tree_[x] - iters_.data();
    uint32_t idy = seg_tree_[y] - iters_.data();
    return comp_(keys_[idx], keys_[idy]) < 0 ? x : y;
  }
};

template <typename Iterator, typename KeyCompT, typename ValueT>
class SeqIteratorSetForScan {
  IndSKey current_key_;
  ValueT current_value_;
  SeqIteratorSet<Iterator, KeyCompT, ValueT> iter_;
  bool valid_;
  double current_tick_;
  TickFilter<ValueT> tick_filter_;

 public:
  SeqIteratorSetForScan(SeqIteratorSet<Iterator, KeyCompT, ValueT>&& iter, double current_tick, TickFilter<ValueT> tick_filter) 
    : iter_(std::move(iter)), valid_(true), current_tick_(current_tick), tick_filter_(tick_filter) {
      // logger(tick_filter_.get_tick_threshold());
    }
  void build() {
    iter_.build();
    next();
  }
  std::pair<SKey, ValueT> read() { return {current_key_.ref(), current_value_}; }
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
        current_value_.merge(result.second, current_tick_);
      } else {
        if (tick_filter_.check(current_value_)) {
          return;
        } else {
          current_key_ = result.first;
          current_value_ = result.second;
        }
      }  
      iter_.next();
    }
    if (!tick_filter_.check(current_value_)) {
      valid_ = false;
    }
  }

  bool valid() { return valid_; }
};

}


#endif