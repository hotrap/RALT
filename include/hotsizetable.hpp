#ifndef VISCNTS_HOTSIZE_TABLE_H__
#define VISCNTS_HOTSIZE_TABLE_H__

#include "compaction.hpp"

namespace viscnts_lsm {

template<typename KeyCompT, typename ValueT>
class HotSizeTable {
  std::vector<ImmutableFile<KeyCompT, ValueT>> files_;
  size_t hot_size_{0};

 public:
  
};


}


#endif