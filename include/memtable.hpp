#ifndef VISCNTS_MEMTABLE_H__
#define VISCNTS_MEMTABLE_H__

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <random>
#include <thread>

#include "alloc.hpp"
#include "common.hpp"
#include "key.hpp"
#include "iterators.hpp"
#include <set>
#include <vector>
#include <unordered_map>

namespace ralt {

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

template<typename KeyCompT, typename ValueT>
class UnsortedBuffer {
  std::atomic<size_t> used_size_;
  std::atomic<uint32_t> working_count_;
  size_t buffer_size_;
  KeyCompT comp_;
  uint8_t *data_;
  std::vector<std::pair<SKey, uint8_t *>> sorted_result_;

  class Comparator {
    KeyCompT comp_;
    public:
      Comparator(KeyCompT comp) : comp_(comp) {}
      bool operator()(const std::pair<SKey, const uint8_t*>& x, const std::pair<SKey, const uint8_t*>& y) const {
        auto result = comp_(x.first, y.first);
        if (result == 0) {
          return x.second < y.second;
        } 
        return result < 0;
      }
  };

 public:
  UnsortedBuffer(size_t size, KeyCompT comp) : used_size_(0), working_count_(0), buffer_size_(size), comp_(comp), data_(new uint8_t[size]) { 
    memset(data_, 0, size);
  }
  UnsortedBuffer(const UnsortedBuffer &buf) = delete;
  UnsortedBuffer(UnsortedBuffer &&buf) {
    used_size_ = buf.used_size_.load();
    working_count_ = buf.working_count_.load();
    buffer_size_ = buf.buffer_size_;
    comp_ = std::move(buf.comp_);
    data_ = buf.data_;
    sorted_result_ = std::move(buf.sorted_result_);
    buf.data_ = nullptr;
  }
  ~UnsortedBuffer() {
    if (data_) delete[] data_;
  }
  bool append(const SKey &key, const ValueT &value) {
    auto size = key.size() + sizeof(ValueT);
    auto pos = used_size_.fetch_add(size, std::memory_order_relaxed);
    if (pos + size > buffer_size_) {
      used_size_.fetch_sub(size, std::memory_order_relaxed);
      return false;
    }
    working_count_ += 1;
    *reinterpret_cast<ValueT *>(key.write(data_ + pos)) = value;
    working_count_ -= 1;
    return true;
  }

  void sort(size_t current_tick) {
    while(working_count_.load(std::memory_order_relaxed) != 0);
    sorted_result_.clear();
    size_t limit = used_size_;
    auto comp_func = [this](const std::pair<SKey, uint8_t *> &x, const std::pair<SKey, uint8_t *> &y) {
      return comp_(x.first, y.first) < 0;
    };
    for (uint8_t *d = data_; d - data_ + sizeof(uint32_t) < limit && *reinterpret_cast<uint32_t *>(d) != 0;) {
      SKey key;
      uint8_t *v = key.read(d);
      sorted_result_.emplace_back(std::pair<SKey, uint8_t *>(key, v));
      d += key.size() + sizeof(ValueT);
    }
    // tim::timsort(sorted_result_.begin(), sorted_result_.end(), comp_func);
    std::sort(sorted_result_.begin(), sorted_result_.end(), comp_func);
    int d = 0;
    for (auto& a : sorted_result_) {
      if (d == 0 || comp_(sorted_result_[d - 1].first, a.first) != 0) {
        sorted_result_[d++] = a;
      } else {
        reinterpret_cast<ValueT*>(sorted_result_[d - 1].second)->merge(*reinterpret_cast<ValueT*>(a.second), current_tick);
      }
    }
    sorted_result_.resize(d);
  }

  void sort_with_mp(size_t current_tick) {
    sort(current_tick);
    return;
    // while(working_count_.load(std::memory_order_relaxed) != 0);
    // sorted_result_.clear();
    // SeqIteratorSet<MpIterator, KeyCompT, const uint8_t*> iters(comp_);
    // for (auto& a : mp_) if (a.size()) {
    //   iters.push(MpIterator(a.begin(), a.end()));
    // }
    // iters.build();
    // while(iters.valid()) {
    //   sorted_result_.push_back(iters.read());
    //   iters.next();
    // }
  }

  void clear() {
    auto limit = std::min(used_size_.load(std::memory_order_relaxed), buffer_size_);
    memset(data_, 0, limit);
    used_size_ = 0;
  }
  size_t size() const { return used_size_; }
  class Iterator {
    std::vector<std::pair<SKey, uint8_t *>>::const_iterator iter_;
    std::vector<std::pair<SKey, uint8_t *>>::const_iterator iter_end_;

   public:
    Iterator(const UnsortedBuffer &buf) : iter_(buf.sorted_result_.begin()), iter_end_(buf.sorted_result_.end()) {}
    void next() { iter_++; }
    bool valid() { return iter_ != iter_end_; }
    std::pair<SKey, ValueT> read() {
      ValueT value;
      value = *reinterpret_cast<const ValueT *>(iter_->second);
      return {iter_->first, value};
    }
  };
};

template<typename KeyCompT, typename ValueT>
class UnsortedBufferPtrs {
  constexpr static size_t kWaitSleepMilliSeconds = 10;
  std::mutex m_;
  std::condition_variable cv_;
  std::atomic<UnsortedBuffer<KeyCompT, ValueT> *> buf;
  std::vector<UnsortedBuffer<KeyCompT, ValueT> *> buf_q_;
  size_t buffer_size_;
  size_t max_q_size_;
  KeyCompT comp_;
  bool terminal_signal_{false};

 public:
  UnsortedBufferPtrs(size_t buffer_size, size_t max_q_size, KeyCompT comp) : buffer_size_(buffer_size), max_q_size_(max_q_size), comp_(comp) {
    buf = new UnsortedBuffer<KeyCompT, ValueT>(buffer_size_, comp_);
  }
  ~UnsortedBufferPtrs() {
    delete buf.load();
    for (auto &a : buf_q_) delete a;
  }
  bool append(const SKey &key, const ValueT &value) {
    // Maybe we should use write/read lock? but it's not beautiful.
    // We release the previous when the second is full instead of hazard pointers. Then it seems almost impossible that a thread is writing the
    // previous when we release it.

    // It seems unnecessary to use std::atomic<UnsortedBuffer*>
    auto bbuf = buf.load(std::memory_order_relaxed);
    if (bbuf->append(key, value)) return true;
    do {
      m_.lock();
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
        continue; // This almost never happens
      }
      buf_q_.push_back(buf);
      buf = bbuf = new UnsortedBuffer<KeyCompT, ValueT>(buffer_size_, comp_);
      m_.unlock();
      cv_.notify_one();
    } while (!bbuf->append(key, value));
    return true;
  }
  std::vector<UnsortedBuffer<KeyCompT, ValueT> *> get() {
    if (buf_q_.size()) {
      std::unique_lock lck_(m_);
      auto ret = std::move(buf_q_);
      buf_q_.clear();
      return ret;
    }
    return {};
  }

  UnsortedBuffer<KeyCompT, ValueT> *this_buf() { return buf.load(std::memory_order_relaxed); }

  void terminate() { terminal_signal_ = 1; cv_.notify_all(); }

  std::vector<UnsortedBuffer<KeyCompT, ValueT> *> wait_and_get() {
    std::unique_lock lck_(m_);
    cv_.wait(lck_, [&](){ return buf_q_.size() || terminal_signal_; });
    // It may return empty queue.
    // Because I must let it return to terminate the compact thread.
    auto ret = std::move(buf_q_);
    buf_q_.clear();
    return ret;
  }

  void append_and_notify(const SKey &key, const ValueT &value) {
    if (!append(key, value)) {
      cv_.notify_one();
    }
  }

  void flush() {
    std::unique_lock lck_(m_);
    if (buf.load()->size() > 0) {
      buf_q_.push_back(buf);
      buf = new UnsortedBuffer<KeyCompT, ValueT>(buffer_size_, comp_);  
    }
    cv_.notify_one();
  }

  int size() {
    return buf_q_.size();
  }

  std::mutex& get_mutex() {
    return m_;
  }

};

}  // namespace ralt

#endif
