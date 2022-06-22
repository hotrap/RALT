#ifndef __ALLOC_H___
#define __ALLOC_H___

#include "common.hpp"

class BaseAllocator {
 public:
  virtual uint8_t* allocate(size_t size) = 0;
  virtual void release(uint8_t*) = 0;
};

class MemtableAllocator : public BaseAllocator {
  static const int kBlockSize = 1e7;
  std::vector<uint8_t*> v;
  size_t nwsize;

 public:
  MemtableAllocator() {
    nwsize = 0;
    v.clear();
  }
  ~MemtableAllocator() {
    for (auto& a : v)
      if (a) {
        delete[] a;
      }
  }
  uint8_t* allocate(size_t size) {
    if (!v.size() || nwsize + size > kBlockSize) {
      v.push_back(new uint8_t[kBlockSize]);
      nwsize = size;
      return v.back();
    } else {
      auto ret = v.back() + nwsize;
      nwsize += size;
      return ret;
    }
  }
  // release all
  void release(uint8_t* ptr = nullptr) {
    nwsize = 0;
  }
};

#endif