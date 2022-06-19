#ifndef __HASH_H___
#define __HASH_H___
#include <stdint.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>

namespace viscnts_lsm {

uint32_t Hash(const char* data, size_t n, uint32_t seed);

}

#endif