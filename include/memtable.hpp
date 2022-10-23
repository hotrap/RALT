#include <atomic>
#include <condition_variable>
#include <mutex>
#include <random>
#include <thread>

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
  SkipList &operator=(const SkipList &) = delete;
  SkipList &operator=(SkipList &&list) {
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
  MemTable &operator=(MemTable &&m) {
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
    for (int level = kMaxHeight - 1; level >= 0; --level)
      while (auto c = mx->noBarrierGetNext(level)) mx = c;
    return std::make_pair(mn->key, mx->key);
  }
};

class UnsortedBuffer {
  std::atomic<size_t> used_size_;
  std::atomic<size_t> ref_;
  size_t buffer_size_;
  uint8_t *data_;
  std::vector<std::pair<SKey, const uint8_t *>> sorted_result_;

 public:
  UnsortedBuffer(size_t size) : used_size_(0), ref_(0), buffer_size_(size), data_(new uint8_t[size]) { memset(data_, 0, size); }
  UnsortedBuffer(const UnsortedBuffer &buf) {
    used_size_ = buf.used_size_.load();
    buffer_size_ = buf.buffer_size_;
    data_ = new uint8_t[buffer_size_];
    sorted_result_ = buf.sorted_result_;
    ref_ = 0;
    memcpy(data_, buf.data_, buffer_size_);
  }
  UnsortedBuffer(UnsortedBuffer &&buf) {
    used_size_ = buf.used_size_.load();
    buffer_size_ = buf.buffer_size_;
    data_ = buf.data_;
    sorted_result_ = std::move(buf.sorted_result_);
    buf.data_ = nullptr;
    ref_ = 0;
  }
  ~UnsortedBuffer() {
    if (data_) delete[] data_;
  }
  bool append(const SKey &key, const SValue &value) {
    auto size = key.size() + sizeof(SValue);
    auto pos = used_size_.fetch_add(size, std::memory_order_relaxed);
    if (pos + size > buffer_size_) return false;
    ref_ += 1;
    *reinterpret_cast<SValue *>(key.write(data_ + pos)) = value;
    ref_ -= 1;
    return true;
  }

  template <typename KeyComp>
  void sort(KeyComp comp) {
    sorted_result_.clear();
    ssize_t limit = std::min(used_size_.load(std::memory_order_relaxed), buffer_size_);
    uint32_t len;
    auto comp_func = [comp](const std::pair<SKey, const uint8_t *> &x, const std::pair<SKey, const uint8_t *> &y) {
      return comp(x.first, y.first) < 0;
    };
    for (uint8_t *d = data_; d - data_ < limit && (len = *reinterpret_cast<uint32_t *>(d)) != 0;) {
      SKey key;
      const uint8_t *v = key.read(d);
      sorted_result_.emplace_back(std::pair<SKey, const uint8_t *>(key, v));
      d += len + sizeof(uint32_t) + sizeof(SValue);
    }
    // tim::timsort(sorted_result_.begin(), sorted_result_.end(), comp_func);
    std::sort(sorted_result_.begin(), sorted_result_.end(), comp_func);
  }
  void clear() {
    auto limit = std::min(used_size_.load(std::memory_order_relaxed), buffer_size_);
    memset(data_, 0, limit);
    used_size_ = 0;
    ref_ = 0;
  }
  size_t size() const { return used_size_; }
  bool safe() const { return ref_ == 0; }
  class Iterator {
    std::vector<std::pair<SKey, const uint8_t *>>::const_iterator iter_;
    std::vector<std::pair<SKey, const uint8_t *>>::const_iterator iter_end_;

   public:
    Iterator(const UnsortedBuffer &buf) : iter_(buf.sorted_result_.begin()), iter_end_(buf.sorted_result_.end()) {}
    void next() { iter_++; }
    bool valid() { return iter_ != iter_end_; }
    std::pair<SKey, SValue> read() {
      SValue value;
      value = *reinterpret_cast<const SValue *>(iter_->second);
      return {iter_->first, value};
    }
  };
};

class UnsortedBufferPtrs {
  constexpr static size_t kWaitSleepMilliSeconds = 10;
  std::mutex m_;
  std::condition_variable cv_;
  std::atomic<UnsortedBuffer *> buf;
  std::vector<UnsortedBuffer *> buf_q_;
  size_t buffer_size_;
  size_t max_q_size_;

 public:
  UnsortedBufferPtrs(size_t buffer_size, size_t max_q_size) : buffer_size_(buffer_size), max_q_size_(max_q_size) {
    buf = new UnsortedBuffer(buffer_size_);
  }
  ~UnsortedBufferPtrs() {
    delete buf.load();
    for (auto &a : buf_q_) delete a;
  }
  bool append(const SKey &key, const SValue &value) {
    auto bbuf = buf.load(std::memory_order_relaxed);
    if (bbuf->append(key, value)) return true;
    do {
      m_.lock();
      if (buf.load(std::memory_order_relaxed) != bbuf) {
        bbuf = buf.load(std::memory_order_relaxed);
        m_.unlock();
        continue;
      }
      while (buf.load(std::memory_order_relaxed) == bbuf && buf_q_.size() >= max_q_size_) {
        m_.unlock();
        cv_.notify_one();
        std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(kWaitSleepMilliSeconds));
        m_.lock();
      }
      if (buf.load(std::memory_order_relaxed) != bbuf) {
        bbuf = buf.load(std::memory_order_relaxed);
        m_.unlock();
        if (bbuf->append(key, value)) return true;
        continue;
      }
      // printf("[%llu]", buf.load()); fflush(stdout);
      buf_q_.push_back(buf);
      buf = bbuf = new UnsortedBuffer(buffer_size_);
      m_.unlock();
    } while (!bbuf->append(key, value));
    return false;
  }
  std::vector<UnsortedBuffer *> get() {
    if (buf_q_.size()) {
      std::unique_lock lck_(m_);
      auto ret = std::move(buf_q_);
      buf_q_.clear();
      return ret;
    }
    return std::vector<UnsortedBuffer *>();
  }

  UnsortedBuffer *this_buf() { return buf.load(std::memory_order_relaxed); }

  void notify_cv() { cv_.notify_all(); }

  std::vector<UnsortedBuffer *> wait_and_get() {
    if (buf_q_.size()) {
      std::unique_lock lck_(m_);
      auto ret = std::move(buf_q_);
      buf_q_.clear();
      return ret;
    }
    std::unique_lock lck_(m_);
    cv_.wait(lck_);
    auto ret = std::move(buf_q_);
    buf_q_.clear();
    return ret;
  }
  void append_and_notify(const SKey &key, const SValue &value) {
    if (!append(key, value)) {
      cv_.notify_one();
    }
  }

  void flush() {
    std::unique_lock lck_(m_);
    buf_q_.push_back(buf);
    buf = new UnsortedBuffer(buffer_size_);
    cv_.notify_one();
  }

  int size() {
    std::unique_lock lck_(m_);
    return buf_q_.size();
  }
};

}  // namespace viscnts_lsm
