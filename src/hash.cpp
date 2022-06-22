#include "hash.hpp"
namespace viscnts_lsm {

// copy from leveldb
uint32_t Hash(const char* data, size_t n, uint32_t seed) {
  // Similar to murmur hash
  const uint32_t m = 0xc6a4a793;
  const uint32_t r = 24;
  const char* limit = data + n;
  uint32_t h = seed ^ (n * m);

  // Pick up four bytes at a time
  while (data + 4 <= limit) {
    uint32_t w = *reinterpret_cast<const uint32_t*>(data);
    data += 4;
    h += w;
    h *= m;
    h ^= (h >> 16);
  }

  // Pick up remaining bytes
  switch (limit - data) {
    case 3:
      h += static_cast<unsigned char>(data[2]) << 16;
    case 2:
      h += static_cast<unsigned char>(data[1]) << 8;
    case 1:
      h += static_cast<unsigned char>(data[0]);
      h *= m;
      h ^= (h >> r);
      break;
  }
  return h;
}

// copy from leveldb
uint32_t Hash8(const char* data, uint32_t seed) {
  // Similar to murmur hash
  const uint32_t m = 0xc6a4a793;
  const uint32_t r = 24;
  uint32_t h = seed ^ (8 * m);
  uint32_t w = *reinterpret_cast<const uint32_t*>(data);
  uint32_t w2 = *reinterpret_cast<const uint32_t*>(data + 4);
  h += w;
  h *= m;
  h ^= (h >> 16);
  h += w2;
  h *= m;
  h ^= (h >> 16);
  return h;
}

uint32_t Hash8(const char* data) {
  return Hash8(data, 0x114514);
}

}  // namespace viscnts_lsm