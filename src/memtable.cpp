#include "memtable.hpp"

namespace viscnts_lsm {



template <class Key, class SkipListValue, class Allocator, class Comparator>
typename SkipList<Key, SkipListValue, Allocator, Comparator>::Node *SkipList<Key, SkipListValue, Allocator, Comparator>::newNode(const Key &k,
                                                                                                                                 int height) {
  char *mem = alloc_->allocate(sizeof(Node) + sizeof(std::atomic<Node *>) * (height - 1));
  return new (mem) Node(k);
}
template <class Key, class SkipListValue, class Allocator, class Comparator>
typename SkipList<Key, SkipListValue, Allocator, Comparator>::Node *SkipList<Key, SkipListValue, Allocator, Comparator>::queryLower(
    const Key &k) const {
  // find the first one that node -> k >= k.
  Node *nw = head_;
  Node *lst = nullptr;
  int level = GetMaxHeight() - 1;
  while (true) {
    Node *next = nw->getNext(level);
    int cmp = (next == nullptr || next == lst) ? -1 : comp_(k, next->key);
    // cmp > 0 means k > next -> k
    // cmp = 0 means k = next -> k
    // cmp < 0 means k < next -> k
    if (cmp == 0 || (cmp < 0 && level == 0)) {
      return next;
    } else if (cmp < 0) {
      lst = next;
      level--;
    } else {
      nw = next;
    }
  }
}

template <class Key, class SkipListValue, class Allocator, class Comparator>
typename SkipList<Key, SkipListValue, Allocator, Comparator>::Node *SkipList<Key, SkipListValue, Allocator, Comparator>::queryEqual(
    const Key &k) const {
  // find the largest one that node -> k <= k.
  Node *nw = head_;
  Node *lst = nullptr;
  int level = GetMaxHeight() - 1;
  while (true) {
    Node *next = nw->noBarrierGetNext(level);
    int cmp = (next == nullptr || next == lst) ? -1 : comp_(k, next->key);
    // printf("[(nw)%s]", std::string(reinterpret_cast<char *>(nw->key.a_),
    // nw->key.len_).c_str()); printf("[(next)%s]",
    //        next == nullptr
    //            ? "nullptr"
    //            : std::string(reinterpret_cast<char *>(next->key.a_),
    //            next->key.len_).c_str());
    // fflush(stdout);
    if (cmp == 0) {
      nw = next;
    } else if (cmp < 0 && level == 0) {
      return nw != head_ && comp_.is_equal(k, nw->key) == 0 ? nw : nullptr;
    } else if (cmp < 0) {
      lst = next;
      level--;
    } else
      nw = next;
  }
}

template <class Key, class SkipListValue, class Allocator, class Comparator>
typename SkipList<Key, SkipListValue, Allocator, Comparator>::Node *SkipList<Key, SkipListValue, Allocator, Comparator>::queryUpper(
    const Key &k, Node **prev) const {
  // find the largest one that node -> k <= k.
  Node *nw = head_;
  Node *lst = nullptr;
  int level = GetMaxHeight() - 1;
  while (true) {
    Node *next = nw->noBarrierGetNext(level);
    int cmp = (next == nullptr || next == lst) ? -1 : comp_(k, next->key);
    if (cmp == 0) {
      nw = next;
    } else if (cmp < 0 && level == 0) {
      prev[level] = nw;
      return nw;
    } else if (cmp < 0) {
      lst = next;
      prev[level] = nw;
      level--;
    } else
      nw = next;
  }
}

template <class Key, class SkipListValue, class Allocator, class Comparator>
typename SkipList<Key, SkipListValue, Allocator, Comparator>::Node *SkipList<Key, SkipListValue, Allocator, Comparator>::queryUpper(
    const Key &k) const {
  // find the largest one that node -> k <= k.
  Node *nw = head_;
  Node *lst = nullptr;
  int level = GetMaxHeight() - 1;
  while (true) {
    Node *next = nw->noBarrierGetNext(level);
    int cmp = (next == nullptr || next == lst) ? -1 : comp_(k, next->key);
    if (cmp == 0) {
      nw = next;
    } else if (cmp < 0 && level == 0) {
      return nw;
    } else if (cmp < 0) {
      lst = next;
      level--;
    } else
      nw = next;
  }
}

template <class Key, class SkipListValue, class Allocator, class Comparator>
typename SkipList<Key, SkipListValue, Allocator, Comparator>::Node *SkipList<Key, SkipListValue, Allocator, Comparator>::insert(
    const Key &k, const SkipListValue &v) {
  auto lst = prev_[0]->noBarrierGetNext(0);
  auto nxt = lst == nullptr ? nullptr : lst->noBarrierGetNext(0);
  if (prev_height_ && lst != nullptr && comp_(lst->key, k) <= 0 && (nxt == nullptr || comp_(k, nxt->key) <= 0)) {
    for (int i = 0; i < prev_height_; i++) prev_[i] = lst;
  } else
    queryUpper(k, prev_);
  int height = rndHeight();
  prev_height_ = height;
  if (height > GetMaxHeight()) {
    for (int i = GetMaxHeight(); i < height; i++) prev_[i] = head_;
    height_.store(height, std::memory_order_relaxed);
  }

  Node *nw = newNode(k, height);
  nw->value = v;
  for (int i = 0; i < height; i++) {
    // we don't want nw -> next[0] = A -> next, nw -> next[1] = B -> next, B -> next = nw, A
    // -> next = nw; OoO on CPU.
    // Intel do LoadLoad, LoadStore barrier in default, but we need StoreStore barrier, in
    // other words, sfence.
    nw->noBarrierSetNext(i, prev_[i]->noBarrierGetNext(i));
    prev_[i]->setNext(i, nw);
  }

  return nw;
}

template <class Key, class SkipListValue, class Allocator, class Comparator>
typename SkipList<Key, SkipListValue, Allocator, Comparator>::Node *SkipList<Key, SkipListValue, Allocator, Comparator>::remove(const Key &k) {
  Node *prev_[kMaxHeight];
  int height = rndHeight();
  queryUpper(k, prev_);

  if (prev_[0] == head_ || prev_[0]->key.is_del() || comp_.is_equal(k, prev_[0]->key) != 0) return nullptr;

  if (height > GetMaxHeight()) {
    for (int i = GetMaxHeight(); i < height; i++) prev_[i] = head_;
    height_.store(height, std::memory_order_relaxed);
  }

  Node *nw = newNode(k, height);
  for (int i = 0; i < height; i++) {
    // we don't want nw -> next[0] = A -> next, nw -> next[1] = B -> next, B -> next = nw, A
    // -> next = nw; OoO on CPU.
    // Intel do LoadLoad, LoadStore barrier in default, but we need StoreStore barrier, in
    // other words, sfence.
    nw->noBarrierSetNext(i, prev_[i]->noBarrierGetNext(i));
    prev_[i]->setNext(i, nw);
  }

  return nw;
}

}  // namespace viscnts_lsm