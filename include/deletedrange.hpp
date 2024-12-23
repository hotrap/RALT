#ifndef VISCNTS_DELETED_RANGE_H__
#define VISCNTS_DELETED_RANGE_H__

#include <set>

namespace ralt {
  
class DeletedRange {
 public:
  struct Node {
    std::pair<int, int> ranks;
    Node(std::pair<int, int> _rank = std::pair<int, int>()) : ranks(_rank) {}
  };

  DeletedRange() : nodes_(CompareNode()) {}

  void insert(std::pair<int, int> rank) {
    if (nodes_.empty()) {
      nodes_.insert(Node(rank));
      return;
    }
    auto L = nodes_.lower_bound(std::make_pair(0, rank.first));
    auto R = nodes_.lower_bound(std::make_pair(0, rank.second));
    if (L == nodes_.end() || (rank.second < L->ranks.first)) {
      nodes_.insert(Node(rank));
    } else {
      if (L->ranks.first <= rank.first) {
        rank.first = L->ranks.first;
      }
      if (R != nodes_.end() && R->ranks.first <= rank.second) {
        rank.second = R->ranks.second;
        R = std::next(R);
      }
      nodes_.erase(L, R);
      nodes_.insert(Node(rank));
    }
  }

  int deleted_counts(std::pair<int, int> rank) {
    if (nodes_.empty()) {
      return rank.second - rank.first;
    }
    auto L = nodes_.lower_bound(std::make_pair(0, rank.first));
    auto R = nodes_.lower_bound(std::make_pair(0, rank.second));
    if (L == nodes_.end()) {
      return rank.second - rank.first;
    } else {
      int sum = rank.second - rank.first;
      if (L->ranks.first <= rank.first) {
        if (L == R) return 0;
        sum -= L->ranks.second - rank.first;
        L = std::next(L);
      }
      if (R != nodes_.end() && R->ranks.first <= rank.second) {
        sum -= rank.second - R->ranks.first;
      }
      for (auto p = L; p != R; p++) {
        sum -= p->ranks.second - p->ranks.first;
      }
      return sum;
    }
  }

  int sum() {
    int ret = 0;
    for (auto& a : nodes_) ret += a.ranks.second - a.ranks.first;
    return ret;
  }

 private:
  class CompareNode {
   public:
    int operator()(const Node& x, const Node& y) const { return x.ranks.second < y.ranks.second; }
  };

  std::set<Node, CompareNode> nodes_;

 public:
  class Iterator {
    typename std::set<Node, CompareNode>::const_iterator iter_;
    typename std::set<Node, CompareNode>::const_iterator iter_end_;

   public:
    Iterator() {}
    Iterator(int k, const DeletedRange& range) : iter_(range.nodes_.lower_bound(Node(std::make_pair(0, k)))), iter_end_(range.nodes_.end()) {}
    Iterator(const DeletedRange& range) : iter_(range.nodes_.begin()), iter_end_(range.nodes_.end()) {}
    bool valid() { return iter_ != iter_end_; }
    int jump(int k) {
      while (iter_->ranks.first <= k) {
        k = iter_->ranks.second;
        iter_ = std::next(iter_);
        if (iter_ == iter_end_) return k;
      }
      return k;
    }
  };
};



}

#endif