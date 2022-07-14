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
  uint8_t* data;
  BaseAllocator* deleter;
  std::atomic<bool> valid;
  bool in_use;
  std::atomic<uint32_t> refs;
  size_t klen;
  uint8_t key_data[1];
  static LRUHandle* genHandle(const Slice& key) {
    auto mem = new uint8_t[sizeof(LRUHandle) + key.len() - 1]();
    auto ret = new (mem) LRUHandle();
    memcpy(ret->key_data, key.data(), key.len());
    ret->klen = key.len();
    assert(ret->data == nullptr);
    return ret;
  }
  Slice Key() { return Slice(key_data, klen); }
};

// use chain hashing
class HashTable {
 public:
  HashTable(size_t size) : size_(size), list_(size) {}
  ~HashTable() { 
    for(size_t i = 0; i < size_; ++i) if(list_[i]) {
      for(auto ptr = list_[i], nxt_ptr = ptr->nxt_h; ptr; ptr = nxt_ptr, nxt_ptr = nxt_ptr ? nxt_ptr->nxt_h : nullptr) 
        ptr->deleter ? ptr->deleter->release(ptr->data), delete ptr : delete ptr;
    }
  }
  LRUHandle* lookup(const Slice& key, uint32_t hash, bool allow_new) {
    auto head = list_[hash % size_];
    if (!head) {
      if (!allow_new) return nullptr;
      auto ret = LRUHandle::genHandle(key);
      ret->nxt_h = nullptr;
      ret->prv_h = nullptr;
      list_[hash % size_] = ret;
      return ret;
    }
    while (head->nxt_h && head->Key() != key) head = head->nxt_h;
    if (head->Key() != key) {
      if (!allow_new) return nullptr;
      auto ret = LRUHandle::genHandle(key);
      ret->nxt_h = nullptr;
      ret->prv_h = head;
      head->nxt_h = ret;
      return ret;
    }
    return head;
  }
  bool erase(const Slice& key, uint32_t hash) {
    auto ptr = lookup(key, hash, false);
    if (ptr == nullptr) return false;
    erase(ptr);
    return true;
  }
  void erase(LRUHandle* ptr) {
    ptr->prv_h->nxt = ptr->nxt;
    ptr->nxt_h->prv = ptr->prv;
  }

 private:
  size_t size_;
  std::vector<LRUHandle*> list_;
};

class LRUCache {
 public:
  const static size_t TableSize = 1 << 20;
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
  LRUHandle *lru_, *in_use_;
  HashTable table;
};

LRUCache::LRUCache(size_t size) : limit_size_(size), used_(0), table(LRUCache::TableSize) {
  lru_ = new LRUHandle();
  in_use_ = new LRUHandle();
  lru_->nxt = lru_->prv = lru_;
  in_use_->nxt = in_use_->prv = in_use_;
}

LRUCache::~LRUCache() {
  delete lru_;
  delete in_use_;
}

LRUHandle* LRUCache::lookup(const Slice& key, uint32_t hash) {
  // std::unique_lock<std::shared_mutex> _(mutex_);
  auto ret = table.lookup(key, hash, true);
  ref(ret);
  return ret;
}

void LRUCache::release(LRUHandle* h) { unref(h); }

void LRUCache::ref(LRUHandle* h) {
  if (h->prv != nullptr) {
    h->prv->nxt = h->nxt;
    h->nxt->prv = h->prv;
  }
  h->refs++;
  if (!h->in_use) {
    h->in_use = true;
    h->nxt = in_use_->nxt;
    h->prv = in_use_;
    in_use_->nxt = h;
  }
}

void LRUCache::unref(LRUHandle* h) {
  if (--h->refs) {
    h->in_use = false;
    {
      std::unique_lock<std::shared_mutex> _(mutex_);
      h->prv->nxt = h->nxt;
      h->nxt->prv = h->prv;
      h->nxt = lru_->nxt;
      h->prv = lru_;
      lru_->nxt = h;
      if (used_ < limit_size_) {
        used_++;
        return;
      } else {
        h = lru_->prv;
      }
      table.erase(h);
    }
    if(h->deleter) h->deleter->release(h->data); 
    delete h;
  }
}

}  // namespace viscnts_lsm

#endif