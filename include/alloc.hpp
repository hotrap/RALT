#ifndef __ALLOC_H___
#define __ALLOC_H___

#include "common.hpp"

class BaseAllocator {
 public:
  virtual uint8_t* allocate(size_t size) = 0;
  virtual void release(uint8_t*) = 0;
};

class MemtableAllocator : public BaseAllocator {
  static const size_t kBlockSize = 1e7;
  std::vector<uint8_t*> v;
  size_t nwsize;
  size_t index;

 public:
  MemtableAllocator() : nwsize(0), index(0) {}
  ~MemtableAllocator() {
    for (auto& a : v) delete a;
  }
  uint8_t* allocate(size_t size) override {
    if (index == 0 || nwsize + size > kBlockSize) {
      if (index == v.size()) v.push_back(new uint8_t[size > kBlockSize ? size : kBlockSize]);
      else if(size > kBlockSize) delete v[index], v[index] = new uint8_t[size];
      nwsize = size;
      return v[index++];
    } else {
      auto ret = v[index - 1] + nwsize;
      nwsize += size;
      return ret;
    }
  }
  // release all
  void release(uint8_t* = nullptr) override {
    nwsize = 0;
    index = 0;
  }
};

class DefaultAllocator : public BaseAllocator {
 public:
  uint8_t* allocate(size_t size) override { return new uint8_t[size]; }
  void release(uint8_t* ptr) override { delete ptr; }
};


// class MemtableAllocator : public BaseAllocator {
//  public:
//   uint8_t* allocate(size_t size) override { return new uint8_t[size]; }
//   void release(uint8_t* = nullptr) override {}
// };


#endif