#include <algorithm>
#include <vector>

#include "common.hpp"
#include "hash.hpp"
#include "key.hpp"

namespace viscnts_lsm {

static uint32_t BloomHash(const SKey& key) { return Hash(reinterpret_cast<const char*>(key.data()), key.size(), 0xbc9f1d34); }

class BloomFilter {
 private:
  size_t bits_per_key_;
  size_t k_;

 public:
  explicit BloomFilter(int bits_per_key) : bits_per_key_(bits_per_key) {
    k_ = static_cast<size_t>(bits_per_key * 0.69);
    k_ = std::max<size_t>(std::min<size_t>(30, k_), 1);
  }
  void create(const SKey* keys, int n, Slice out) {
    size_t bits = n * bits_per_key_;
    bits = std::max<size_t>(64, bits);
    size_t bytes = (bits + 7) / 8;
    bits = bytes * 8;
    assert(out.size() >= bits + sizeof(size_t));
    uint8_t* array = out.data();
    *reinterpret_cast<size_t*>(array) = k_;
    array += sizeof(size_t);
    for (int i = 0; i < n; i++) {
      uint32_t h = BloomHash(keys[i]);
      const uint32_t delta = (h >> 17) | (h << 15);
      uint32_t bitpos = h % bits, dpos = delta % bits;
      for (size_t j = 0; j < k_; j++) {
        array[bitpos / 8] |= (1 << (bitpos & 7));
        bitpos += dpos, bitpos >= bits ? bitpos -= bits : 0;
      }
    }
  }
  bool find(const SKey& key, const Slice& bloom_bits) {
    if (bloom_bits.size() < 4) return false;
    const size_t k = *(reinterpret_cast<size_t*>(bloom_bits.data()));
    if (k > 30) return true;
    size_t bits = bloom_bits.size() * 8 - sizeof(size_t) * 8;
    uint8_t* array = bloom_bits.data() + sizeof(size_t);
    uint32_t h = BloomHash(key);
    const uint32_t delta = (h >> 17) | (h << 15);
    uint32_t bitpos = h % bits, dpos = delta % bits;
    for (size_t j = 0; j < k_; j++) {
      if (!(array[bitpos / 8] & (1 << (bitpos & 7)))) return false;
      bitpos += dpos, bitpos >= bits ? bitpos -= bits : 0;
    }
    return true;
  }
};


}  // namespace viscnts_lsm