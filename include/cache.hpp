#ifndef __CACHE_H___
#define __CACHE_H___
#include <atomic>
#include <memory>
#include <mutex>
#include <shared_mutex>

#include "alloc.hpp"
#include "common.hpp"
#include "key.hpp"

namespace viscnts_lsm {

struct LRUHandle {
  LRUHandle* nxt;
  LRUHandle* prv;
  LRUHandle* nxt_h;
  LRUHandle* prv_h;
  std::atomic<uint8_t*> data;
  BaseAllocator* deleter;
  std::atomic<uint32_t> refs;
  size_t klen;
  uint8_t key_data[1];
  LRUHandle() {
    nxt = prv = nxt_h = prv_h = nullptr;
    data = nullptr;
    deleter = nullptr;
    refs = 0;
    klen = 0;
  }
  static LRUHandle* genHandle(const Slice& key) {
    auto mem = new uint8_t[sizeof(LRUHandle) + key.len() - 1]();
    auto ret = new (mem) LRUHandle();
    memcpy(ret->key_data, key.data(), key.len());
    ret->klen = key.len();
    assert(ret->data == nullptr);
    assert(ret->deleter == nullptr);
    return ret;
  }
  Slice Key() { return Slice(key_data, klen); }
};

// use chain hashing
class HashTable {
 public:
  HashTable(size_t size) : size_(size), list_(size) {}
  ~HashTable() { 
    for(size_t i = 0; i < size_; ++i) if(list_[i].nxt_h) {
      for(auto ptr = list_[i].nxt_h, nxt_ptr = ptr->nxt_h; ptr; ptr = nxt_ptr, nxt_ptr = nxt_ptr ? nxt_ptr->nxt_h : nullptr) 
        delete ptr;
    }
  }
  HashTable(const HashTable&) = delete;
  HashTable(HashTable&&) = delete;
  HashTable& operator=(const HashTable&) = delete;
  HashTable& operator=(HashTable&&) = delete;
  LRUHandle* lookup(const Slice& key, uint32_t hash, bool allow_new) {
    auto* head = &list_[hash & (size_ - 1)], *tail = head;
    if(tail->nxt_h) tail = tail->nxt_h;
    else if(!allow_new) return nullptr;
    else return _insert(key, head);
    while (tail->nxt_h && tail->Key() != key) tail = tail->nxt_h;
    if (tail->Key() != key) {
      if (!allow_new) return nullptr;
      return _insert(key, head);
    }
    return tail;
  }
  bool erase(const Slice& key, uint32_t hash) {
    auto ptr = lookup(key, hash, false);
    if (ptr == nullptr) return false;
    erase(ptr);
    return true;
  }
  void erase(LRUHandle* ptr) {
    std::unique_lock lck_(m_);
    ptr->prv_h->nxt_h = ptr->nxt_h;
    if(ptr->nxt_h) ptr->nxt_h->prv_h = ptr->prv_h;
  }

 private:
  size_t size_;
  std::vector<LRUHandle> list_;
  std::mutex m_;

  LRUHandle* _insert(const Slice& key, LRUHandle* head) {
    std::unique_lock lck_(m_);
    auto tail = head;
    while (tail->nxt_h && tail->Key() != key) tail = tail->nxt_h;
    if(tail != head && tail->Key() == key) return tail;
    auto ret = LRUHandle::genHandle(key);
    ret->nxt_h = nullptr;
    ret->prv_h = tail;
    tail->nxt_h = ret;
    return ret;
  }
};

class LRUCache {
 public:
  const static size_t kTableSize = 1 << 20;
  LRUCache(size_t size);
  ~LRUCache();
  LRUHandle* lookup(const Slice& key, uint32_t hash);
  void release(LRUHandle* h);

 private:
  void ref(LRUHandle* h);
  void unref(LRUHandle* h);
  size_t limit_size_;
  size_t used_;
  std::shared_mutex mutex_;
  LRUHandle *lru_;
  HashTable table;
};

LRUCache::LRUCache(size_t size) : limit_size_(size), used_(0), table(LRUCache::kTableSize) {
  lru_ = new LRUHandle();
  lru_->nxt = lru_->prv = lru_;
}

LRUCache::~LRUCache() {
  delete lru_;
}

LRUHandle* LRUCache::lookup(const Slice& key, uint32_t hash) {
  std::unique_lock<std::shared_mutex> _(mutex_);
  auto ret = table.lookup(key, hash, true);
  ref(ret);
  return ret;
}

void LRUCache::release(LRUHandle* h) { unref(h); }

void LRUCache::ref(LRUHandle* h) {
  if (!h->refs && h->prv != nullptr) {  
    // std::unique_lock<std::shared_mutex> _(mutex_);
    assert(h->prv != nullptr);
    assert(h->nxt != nullptr);
    h->prv->nxt = h->nxt;
    h->nxt->prv = h->prv;
    used_--;
  }
  h->refs++;
}

void LRUCache::unref(LRUHandle* h) {
  assert((int)h->refs >= 0);
  std::unique_lock<std::shared_mutex> _(mutex_);
  if (!--h->refs) {  
    h->nxt = lru_->nxt;
    assert(lru_->nxt != nullptr);
    h->prv = lru_;
    lru_->nxt->prv = h;
    lru_->nxt = h;
    if (used_ < limit_size_) {
      used_++;
      return;
    } else {
      h = lru_->prv;
      lru_->prv = lru_->prv->prv;
      lru_->prv->nxt = lru_;
      assert(h != lru_);
    }
    table.erase(h);
    if(h->deleter) h->deleter->release(h->data); 
    delete h;
  }
}

}  // namespace viscnts_lsm

#endif