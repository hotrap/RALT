
#ifndef __KEY_H__
#define __KEY_H__
#include "common.hpp"

namespace viscnts_lsm {

struct KVPair {
  Slice key;
  double counts;
  size_t vlen;
};


}  // namespace viscnts_lsm

#endif