#include <atomic>
#include <random>

#include "common.hpp"

namespace viscnts_lsm {

static std::mt19937 rndGen(time(0));
template <class Key, class Value, class Allocator, class Comparator>
class SkipList {
  const static int kMaxHeight = 12;

  static int rndHeight() { return __builtin_clz(std::max(1u, (unsigned int)rndGen() & ((1 << kMaxHeight) - 1))) - (31 - kMaxHeight); }

 public:
  explicit SkipList(Allocator *A_, const Comparator &C_) : alloc_(A_), comp_(C_) {
    height_ = 1;
    char *_head_mem = alloc_->allocate(sizeof(Node) + sizeof(std::atomic<Node *>) * (kMaxHeight - 1));
    head_ = reinterpret_cast<Node *>(_head_mem);
    prev_height_ = 0;
    prev_[0] = head_;
  }

  SkipList(const SkipList &) = delete;
  void operator=(const SkipList &) = delete;

  inline int GetMaxHeight() const { return height_.load(std::memory_order_relaxed); }

  class Node {
   public:
    explicit Node(const Key &k) : key(k) {}

    Node *getNext(int level) const {
      assert(level >= 0 && level < kMaxHeight);
      return next[level].load(std::memory_order_acquire);
    }

    void setNext(int level, Node *s) {
      assert(level >= 0 && level < kMaxHeight);
      next[level].store(s, std::memory_order_release);
    }

    void noBarrierSetNext(int level, Node *s) {
      assert(level >= 0 && level < kMaxHeight);
      next[level].store(s, std::memory_order_relaxed);
    }

    Node *noBarrierGetNext(int level) const {
      assert(level >= 0 && level < kMaxHeight);
      return next[level].load(std::memory_order_relaxed);
    }

    const Key key;
    Value value;

   private:
    std::atomic<Node *> next[1];
  };

  Node *newNode(const Key &k, int height);

  Node *queryLower(const Key &k) const;

  Node *queryEqual(const Key &k) const;

  Node *queryUpper(const Key &k, Node **prev) const;

  Node *queryUpper(const Key &k) const;

  Node *insert(const Key &k, const Value &v);

  Node *remove(const Key &k);

  Node *getHead() { return head_; }

  int compNode(Node *x, Node *y) {
    if (x == nullptr && y == nullptr)
      return 0;
    else if (x == nullptr)
      return 1;
    else if (y == nullptr)
      return -1;
    else
      return comp_(x->key, y->key);
  }

  int compWithNext(Node *node) {
    if (node->noBarrierGetNext(0) != nullptr) return comp_.is_equal(node->key, node->noBarrierGetNext(0)->key);
    return -1;
  }

  void print() {
    puts("");
    for (auto z = head_; z != nullptr; z = z->getNext(0)) {
      auto str = std::string(reinterpret_cast<char *>(z->key.a_), z->key.len_);
      printf("[%s,%d]", str.c_str(), z->key.is_del());
    }
    puts("");
  }

 private:
  Allocator *alloc_;
  Comparator comp_;
  std::atomic<int> height_;
  Node *prev_[kMaxHeight];
  int prev_height_;
  Node *head_;
};





}  // namespace viscnts_lsm