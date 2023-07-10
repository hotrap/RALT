#ifndef __ALLOC_H___
#define __ALLOC_H___

#include "common.hpp"

class BaseAllocator {
 public:
  static uint8_t* allocate(size_t size) { return new uint8_t[size]; }
  static uint8_t* align_alloc(size_t size, size_t alignment) { 
    void* ptr; 
    posix_memalign(&ptr, alignment, size); 
    return (uint8_t*) ptr; 
  }
  static void release(uint8_t* ptr) { delete[] ptr; }
  static void align_release(uint8_t* ptr) { free(ptr); }
};


#endif