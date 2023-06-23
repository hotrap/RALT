#include "viscnts.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <queue>
#include <set>
#include <thread>

#include "viscnts_lsm.hpp"

#include "rocksdb/compaction_router.h"
#include "rocksdb/comparator.h"

struct SKeyComparatorFromRocksDB {
  const rocksdb::Comparator* ucmp;
  SKeyComparatorFromRocksDB() : ucmp(nullptr) {}
  SKeyComparatorFromRocksDB(const rocksdb::Comparator* _ucmp) : ucmp(_ucmp) {}
  int operator()(const viscnts_lsm::SKey& x, const viscnts_lsm::SKey& y) const {
    rocksdb::Slice rx(reinterpret_cast<const char*>(x.data()), x.len());
    rocksdb::Slice ry(reinterpret_cast<const char*>(y.data()), y.len());
    return ucmp->Compare(rx, ry);
  }
};

class VisCntsIter : public rocksdb::CompactionRouter::Iter {
  public:
    VisCntsIter(viscnts_lsm::EstimateLSM<SKeyComparatorFromRocksDB>::SuperVersionIterator* it) 
      : it_(it) {}
    ~VisCntsIter() {
      delete it_;
    }
    std::unique_ptr<rocksdb::Slice> next() {
      if (it_->valid()) {
        auto key = it_->read().first;
        return std::make_unique<rocksdb::Slice>(reinterpret_cast<const char*>(key.data()), key.len());
      }
      return nullptr;
    }
  private:
    viscnts_lsm::EstimateLSM<SKeyComparatorFromRocksDB>::SuperVersionIterator* it_;
};

using VisCntsType = viscnts_lsm::VisCnts<SKeyComparatorFromRocksDB>;

VisCnts VisCnts::New(
		const rocksdb::Comparator *ucmp, const char *dir,
		size_t max_hot_set_size) {
  return VisCnts(new VisCntsType(SKeyComparatorFromRocksDB(ucmp), dir, max_hot_set_size));
}

VisCnts::~VisCnts() {
  delete static_cast<VisCntsType*>(vc_);
}

void VisCnts::Access(size_t tier, rocksdb::Slice key, size_t vlen) {
  auto vc = static_cast<VisCntsType*>(vc_);
  vc->access(tier, {viscnts_lsm::SKey(reinterpret_cast<const uint8_t*>(key.data()), vlen), viscnts_lsm::SValue(1, vlen)});
}
bool VisCnts::IsHot(size_t tier, rocksdb::Slice key) {
  auto vc = static_cast<VisCntsType*>(vc_);
  return vc->is_hot(tier, viscnts_lsm::SKey(reinterpret_cast<const uint8_t*>(key.data()), key.size()));
}
void VisCnts::TransferRange(
  size_t target_tier, size_t source_tier, rocksdb::RangeBounds range
) {
  auto vc = static_cast<VisCntsType*>(vc_);
  auto lkey = viscnts_lsm::SKey(reinterpret_cast<const uint8_t*>(range.start.user_key.data()), range.start.user_key.size());
  auto rkey = viscnts_lsm::SKey(reinterpret_cast<const uint8_t*>(range.end.user_key.data()), range.end.user_key.size());
  vc->transfer_range(source_tier, target_tier, {lkey, rkey}, {range.start.excluded, range.end.excluded});
}
size_t VisCnts::RangeHotSize(
  size_t tier, rocksdb::RangeBounds range
) {
  auto vc = static_cast<VisCntsType*>(vc_);
  auto lkey = viscnts_lsm::SKey(reinterpret_cast<const uint8_t*>(range.start.user_key.data()), range.start.user_key.size());
  auto rkey = viscnts_lsm::SKey(reinterpret_cast<const uint8_t*>(range.end.user_key.data()), range.end.user_key.size());
  return vc->range_data_size(tier, {lkey, rkey});
}
std::unique_ptr<rocksdb::CompactionRouter::Iter> VisCnts::Begin(size_t tier) {
  auto vc = static_cast<VisCntsType*>(vc_);
  return std::unique_ptr<rocksdb::CompactionRouter::Iter>(new VisCntsIter(vc->seek_to_first(tier)));
}
std::unique_ptr<rocksdb::CompactionRouter::Iter> VisCnts::LowerBound(
  size_t tier, rocksdb::Slice key
) {
  auto vc = static_cast<VisCntsType*>(vc_);
  return std::unique_ptr<rocksdb::CompactionRouter::Iter>(new VisCntsIter(vc->seek(tier, viscnts_lsm::SKey(reinterpret_cast<const uint8_t*>(key.data()), key.size()))));
}

