#include <atomic>
#include <random>

#include "alloc.hpp"
#include "common.hpp"
#include "key.hpp"

namespace viscnts_lsm {

const static int kMaxHeight = 16;
static std::mt19937 rndGen(time(0));
template <class Key, class Value, class Allocator, class Comparator>
class SkipList {

  static int rndHeight() { return __builtin_clz(std::max(1u, (unsigned int)rndGen() & ((1 << kMaxHeight) - 1))) - (31 - kMaxHeight); }

 public:
  explicit SkipList(Allocator *A_, const Comparator &C_) : alloc_(A_), comp_(C_) {
    height_ = 1;
    uint8_t *_head_mem = alloc_->allocate(sizeof(Node) + sizeof(std::atomic<Node *>) * (kMaxHeight - 1));
    memset(_head_mem, 0, sizeof(Node) + sizeof(std::atomic<Node *>) * (kMaxHeight - 1));
    head_ = reinterpret_cast<Node *>(_head_mem);
    prev_height_ = 0;
    memset(prev_, 0, sizeof(prev_));
    prev_[0] = head_;
  }

  SkipList(const SkipList &) = delete;
  SkipList& operator=(const SkipList &) = delete;
  SkipList& operator=(SkipList&& list) {
    alloc_ = list.alloc_;
    comp_ = list.comp_;
    height_ = list.height_.load(std::memory_order_relaxed);
    prev_height_ = list.prev_height_;
    head_ = list.head_;
    list.head_ = nullptr;
    list.alloc_ = nullptr;
    memcpy(prev_, list.prev_, sizeof(list.prev_));
    memset(list.prev_, 0, sizeof(list.prev_));
    return (*this);
  }

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

class MemTable : public RefCounts {
  size_t size_;
  MemtableAllocator alloc_;
  SkipList<SKey, SValue, MemtableAllocator, SKeyComparator> list_;

 public:
  using Node = SkipList<SKey, SValue, MemtableAllocator, SKeyComparator>::Node;
  explicit MemTable() : size_(0), alloc_(), list_(&alloc_, SKeyComparator()) {}
  MemTable& operator=(MemTable&& m) {
    RefCounts::operator=(std::move(m));
    list_ = std::move(m.list_);
    alloc_ = m.alloc_;
    size_ = m.size_;
    m.size_ = 0;
    return (*this);
  }
  void append(const SKey &key, const SValue &value); 
  bool exists(const SKey &key);
  Node *find(const SKey &key); 
  Node *head() { return list_.getHead(); }
  Node *begin() { return list_.getHead()->noBarrierGetNext(0); }
  size_t size() { return size_; }
  std::pair<SKey, SKey> range() {
    auto mx = head();
    auto mn = head()->noBarrierGetNext(0);
    for(int level = kMaxHeight - 1; level >= 0; --level)
      while(auto c = mx->noBarrierGetNext(level)) mx = c;
    return std::make_pair(mn->key, mx->key);
  }
};

class UnsortedBuffer {
  std::atomic<size_t> used_size_;
  size_t buffer_size_;
  uint8_t* data_;
  std::vector<std::pair<SKey, SValue>> sorted_result_;
  public:
    UnsortedBuffer(size_t size) : used_size_(0), buffer_size_(size), data_(new uint8_t[buffer_size_]) {}
    ~UnsortedBuffer() { delete[] data_; }
    bool append(const SKey& key, const SValue& value) {
      auto size = key.size() + sizeof(SValue);
      auto pos = used_size_.fetch_add(size, std::memory_order_relaxed);
      if(pos > buffer_size_) return false;
      *reinterpret_cast<SValue*>(key.write(data_ + pos - size)) = value;
      return true;
    }
    template<typename KeyComp>
    void sort(KeyComp comp) {
      sorted_result_.clear();
      auto limit = std::min(used_size_.load(std::memory_order_relaxed), buffer_size_);
      uint32_t len;
      for(uint8_t* d = data_; d - data_ < limit && (len = *reinterpret_cast<uint32_t*>(d)) != 0; ) {
        sorted_result_.emplace_back(SKey(d + sizeof(uint32_t), len), *reinterpret_cast<SValue*>(d + sizeof(uint32_t) + len));
        d += len + sizeof(uint32_t) + sizeof(SValue);
      }
      std::sort(sorted_result_.begin(), sorted_result_.end(), comp);
    }
    void clear() {
      auto limit = std::min(used_size_.load(std::memory_order_relaxed), buffer_size_);
      memset(data_, 0, limit);
      used_size_ = 0;
      sorted_result_.clear();
    }
    class Iterator {
      std::vector<std::pair<SKey, SValue>>::iterator iter_;
      std::vector<std::pair<SKey, SValue>>::iterator iter_end_;
      public:
        Iterator(UnsortedBuffer& buf) : iter_(buf.sorted_result_.begin()), iter_end_(buf.sorted_result_.end()) {}
        void next() { iter_++; }
        bool valid() { return iter_ != iter_end_; }
        std::pair<SKey, SValue> read() { return *iter_; }
    };
};

}  // namespace viscnts_lsm