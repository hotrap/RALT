#include "cache.hpp"

namespace ralt {

FileChunkCache* GetDefaultIndexCache() {
  static FileChunkCache index_cache(kIndexCacheSize);
  return &index_cache;
}

}