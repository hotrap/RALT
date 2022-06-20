
#ifndef __KEY_H__
#define __KEY_H__
#include "common.hpp"

namespace viscnts_lsm {

using SKey = Slice;


struct SValue {
  double counts;
  size_t vlen;
};

struct SKeyComparator {
    int operator()(const SKey& A, const SKey& B) const {
        if(A.size() != B.size()) return A.size() < B.size() ? -1 : 1;
        return memcmp(A.data(), B.data(), A.size());
    }
    int is_equal(const SKey& A, const SKey& B) const {
        if(A.size() != B.size()) return -1;
        return memcmp(A.data(), B.data(), A.size());
    }
};

}  // namespace viscnts_lsm

#endif