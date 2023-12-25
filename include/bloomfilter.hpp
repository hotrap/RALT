#ifndef BF_VISCNTS_LSM_H__
#define BF_VISCNTS_LSM_H__
#include <algorithm>
#include <vector>

#include "common.hpp"
#include "hash.hpp"
#include "key.hpp"

namespace viscnts_lsm {



constexpr auto kBloomFilterBitNum = 20;

class BloomFilter {
 private:
  size_t bits_per_key_;
  size_t k_;

 public:
  static size_t BloomHash(SKey key) { return Hash(reinterpret_cast<const char*>(key.data()), key.len(), 0xbc9f1d34); }
  explicit BloomFilter(int bits_per_key) : bits_per_key_(bits_per_key) {
    k_ = static_cast<size_t>(bits_per_key * 0.69); // 0.69 ~ ln(2)
    k_ = std::max<size_t>(std::min<size_t>(30, k_), 1);
  }
  IndSlice Create(int n) {
    size_t bits = n * bits_per_key_;
    bits = std::max<size_t>(64, bits);
    size_t bytes = (bits + 7) / 8;
    IndSlice out(bytes);
    return out;
  }

  void Add(const SKey& key, IndSlice& slice) {
    size_t bits = slice.len() * 8;
    uint8_t* array = slice.data();
    size_t h = BloomHash(key);
    // use the double-hashing in leveldb, i.e. h1 + i * h2
    const size_t delta = Hash8(h, 0x202312201805);
    size_t bitpos = h % bits, dpos = delta % bits;
    for (size_t j = 0; j < k_; j++) {
      array[bitpos / 8] |= (1 << (bitpos & 7));
      bitpos += dpos, bitpos >= bits ? bitpos -= bits : 0;
    }
  }
  
  void Add(size_t hash1, IndSlice& slice) {
    size_t bits = slice.len() * 8;
    uint8_t* array = slice.data();
    size_t h = hash1;
    // use the double-hashing in leveldb, i.e. h1 + i * h2
    const size_t delta = Hash8(h, 0x202312201805);
    size_t bitpos = h % bits, dpos = delta % bits;
    for (size_t j = 0; j < k_; j++) {
      array[bitpos / 8] |= (1 << (bitpos & 7));
      bitpos += dpos, bitpos >= bits ? bitpos -= bits : 0;
    }
  }

  bool Find(SKey key, const Slice& bloom_bits) {
    if (bloom_bits.len() * 8 < 64) return true;
    size_t bits = bloom_bits.len() * 8;
    const uint8_t* array = bloom_bits.data();
    size_t h = BloomHash(key);
    // use the double-hashing in leveldb, i.e. h_k = h1 + k * h2
    const size_t delta = Hash8(h, 0x202312201805);
    size_t bitpos = h % bits, dpos = delta % bits;
    for (size_t j = 0; j < k_; j++) {
      if (!(array[bitpos / 8] & (1 << (bitpos & 7)))) return false;
      bitpos += dpos, bitpos >= bits ? bitpos -= bits : 0;
    }
    return true;
  }
};


}  // namespace viscnts_lsm

#endif