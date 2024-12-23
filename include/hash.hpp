#ifndef __HASH_H___
#define __HASH_H___
#include <stdint.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>

namespace ralt {

size_t Hash(const char* data, size_t n, size_t seed);
size_t Hash8(const void* data, size_t seed);
size_t Hash8(size_t data, size_t seed);

}

#endif