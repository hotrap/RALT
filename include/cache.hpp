#ifndef __CACHE_H___
#define __CACHE_H___
#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <unordered_map>

#include "alloc.hpp"
#include "common.hpp"
#include "key.hpp"
#include "chunk.hpp"

namespace viscnts_lsm {

template<typename T>
class List {
  public:
    struct Handle {
      std::atomic<size_t> ref_counts{0};
      Handle* prv{nullptr};
      Handle* nxt{nullptr};
      T data;
      bool second_chance_{true};

      Handle(const T& data) : data(data) {}
    };

    ~List() {
      for (auto p = head_; ; ) {
        auto nxt = p->nxt;
        delete p;
        p = nxt;
        if (p == head_) {
          break;
        }
      }
    }

    List() {
      head_ = new Handle(T());
      tail_ = new Handle(T());
      head_->nxt = tail_;
      tail_->prv = head_;
      head_->prv = tail_;
      tail_->nxt = head_;
      head_->ref_counts = 114514;
      tail_->ref_counts = 114514;
    }

    Handle* insert(T data, Handle* nxt) {
      auto ptr = new Handle(data);
      size_++;
      insert(ptr, nxt);
      return ptr;
    }
    
    void insert(Handle* ptr, Handle* nxt) {
      ptr->nxt = nxt;
      ptr->prv = nxt->prv;
      nxt->prv = ptr;
      ptr->prv->nxt = ptr;
    }

    Handle* head() const {
      return head_;
    }

    Handle* tail() const {
      return tail_;
    }

    void move_to_head(Handle* ptr) {
      if (ptr->prv) {
        ptr->prv->nxt = ptr->nxt;
      }
      if (ptr->nxt) {
        ptr->nxt->prv = ptr->prv;
      }
      insert(ptr, head_->nxt);
    }

    void remove(Handle* ptr) {
      if (ptr == head_) {
        return;
      }
      size_--;
      if (ptr->prv) {
        ptr->prv->nxt = ptr->nxt;
      }
      if (ptr->nxt) {
        ptr->nxt->prv = ptr->prv;
      }
      delete ptr;
    }

    size_t size() const {
      return size_;
    }

  private:
    Handle* head_{nullptr};
    Handle* tail_{nullptr};
    size_t size_{0};
};

class RefChunk {
  public:
    RefChunk() {}
    RefChunk(const uint8_t* data, std::atomic<size_t>* ref_count) : data_(data), ref_count_(ref_count) {
      if(ref_count_) ref_count_->fetch_add(1, std::memory_order_relaxed);
    }
    RefChunk(const Chunk& c, std::atomic<size_t>* ref_count) : data_(c.data()), ref_count_(ref_count) {
      if(ref_count_) ref_count_->fetch_add(1, std::memory_order_relaxed);
    }
    RefChunk(const RefChunk&) = delete;
    RefChunk& operator=(const RefChunk&) = delete;
    RefChunk(RefChunk&& c) {
      if (ref_count_) {
        ref_count_->fetch_sub(1, std::memory_order_relaxed);
      }
      data_ = c.data_;
      ref_count_ = c.ref_count_;
      c.ref_count_ = nullptr;
    }
    RefChunk& operator=(RefChunk&& c) {
      if (ref_count_) {
        ref_count_->fetch_sub(1, std::memory_order_relaxed);
      }
      data_ = c.data_;
      ref_count_ = c.ref_count_;
      c.ref_count_ = nullptr;
      return *this;
    }
    ~RefChunk() { if(ref_count_) ref_count_->fetch_sub(1, std::memory_order_relaxed); }
    const uint8_t* data(uint32_t offset = 0) const { return data_ + offset; }
    Chunk copy() const {
      return Chunk(data_);
    }

  private:
    const uint8_t* data_{nullptr};
    std::atomic<size_t>* ref_count_{nullptr};
};

template<typename KeyT>
class LRUChunkCache {
  public:
    LRUChunkCache(size_t size_limit) : size_limit_(size_limit) {
      hand_ = lru_list_.tail();
    }
    std::optional<RefChunk> try_get_cache(KeyT key) {
      std::shared_lock lck(m_);
      auto hash_it = chunks_.find(key);
      access_count_++;
      if(hash_it == chunks_.end()) return {};
      hit_count_++;
      // Use clock algorithm
      hash_it->second.first->second_chance_ = true;
      // lru_list_.move_to_head(hash_it->second.first);
      return RefChunk(hash_it->second.second, &hash_it->second.first->ref_counts);
    }
    
    void insert(KeyT key, const Chunk& c) {
      Chunk copy_c(c);
      std::unique_lock lck(m_);
      if (chunks_.count(key)) {
        return;
      }
      if (lru_list_.size() >= size_limit_) {
        // size_limit must > maximum numbers of RefChunk. So that we can find a pointer to erase.
        auto a = hand_;
        do {
          a->second_chance_ = false;
          a = a->nxt; 
        } while (a != hand_ && (a->ref_counts || a->second_chance_));
        if (a == hand_) {
          do {
            a->second_chance_ = false;
            a = a->nxt; 
          } while (a != hand_ && (a->ref_counts || a->second_chance_));
          if (a == hand_) {
            logger("Cache exceeds limit.");
            exit(-1);
          }
        }
        hand_ = a->nxt;
        chunks_.erase(a->data);
        lru_list_.remove(a);
      }
      chunks_[key] = {lru_list_.insert(key, hand_), std::move(copy_c)};
    }

    // Get statistics.
    struct Statistics {
      size_t hit_count{0};
      size_t access_count{0};
    };

    Statistics get_stats() const {
      Statistics ret;
      ret.hit_count = hit_count_;
      ret.access_count = access_count_;
      return ret;
    }
  
  private:
    std::unordered_map<KeyT, std::pair<typename List<KeyT>::Handle*, Chunk>> chunks_;
    List<KeyT> lru_list_;
    typename List<KeyT>::Handle* hand_;
    std::shared_mutex m_;
    size_t size_limit_{0};
    std::atomic<size_t> hit_count_{0}, access_count_{0};
};

using FileChunkCache = LRUChunkCache<size_t>;

constexpr auto kIndexCacheSize = 4 * 1024;

FileChunkCache* GetDefaultIndexCache();

}  // namespace viscnts_lsm

#endif