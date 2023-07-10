#include "cache.hpp"

namespace viscnts_lsm {

FileChunkCache* GetDefaultIndexCache() {
  static FileChunkCache index_cache(kIndexCacheSize);
  return &index_cache;
}

}