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

using VisCntsType = viscnts_lsm::VisCnts<SKeyComparatorFromRocksDB, viscnts_lsm::LRUTickValue, viscnts_lsm::IndexData<1>, viscnts_lsm::CachePolicyT::kUseTick>;

class VisCntsIter : public rocksdb::TraitIterator<rocksdb::HotRecInfo> {
  public:
    VisCntsIter(std::unique_ptr<VisCntsType::IteratorT> it) 
      : it_(std::move(it)) {}
    ~VisCntsIter() {}
    std::unique_ptr<rocksdb::HotRecInfo> next() override {
      if (is_first_) {
        is_first_ = false;
      } else {
        it_->next();
      }
      if (it_->valid()) {
        auto key = it_->read().first;
        auto stable = it_->read().second.is_stable();
        rocksdb::HotRecInfo ret;
        ret.key = rocksdb::Slice(reinterpret_cast<const char*>(key.data()), key.len());
        ret.stable = stable;
        return std::make_unique<rocksdb::HotRecInfo>(ret);
      }
      return nullptr;
    }
  private:
    bool is_first_{true};
    std::unique_ptr<VisCntsType::IteratorT> it_;
};


class FastVisCntsIter : public FastIter<rocksdb::Slice> {
  public:
    FastVisCntsIter(std::unique_ptr<VisCntsType::IteratorT> it) 
      : it_(std::move(it)) {}
    ~FastVisCntsIter() {}
    std::optional<rocksdb::Slice> next() override {
      if (is_first_) {
        is_first_ = false;
      } else {
        it_->next();
      }
      if (it_->valid()) {
        auto key = it_->read().first;
        return rocksdb::Slice(reinterpret_cast<const char*>(key.data()), key.len());
      }
      return {};
    }
  private:
    bool is_first_{true};
    std::unique_ptr<VisCntsType::IteratorT> it_;
};

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
  vc->access(tier, viscnts_lsm::SKey(reinterpret_cast<const uint8_t*>(key.data()), key.size()), vlen);
}
bool VisCnts::IsHot(size_t tier, rocksdb::Slice key) {
  auto vc = static_cast<VisCntsType*>(vc_);
  // logger("is_hot");
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
rocksdb::CompactionRouter::Iter VisCnts::Begin(size_t tier) {
  auto vc = static_cast<VisCntsType*>(vc_);
  logger("Iter Begin");
  return rocksdb::CompactionRouter::Iter(std::make_unique<VisCntsIter>(vc->seek_to_first(tier)));
}
std::unique_ptr<FastIter<rocksdb::Slice>> VisCnts::FastBegin(size_t tier) {
  auto vc = static_cast<VisCntsType*>(vc_);
  return std::make_unique<FastVisCntsIter>(vc->seek_to_first(tier));
}
rocksdb::CompactionRouter::Iter VisCnts::LowerBound(
  size_t tier, rocksdb::Slice key
) {
  auto vc = static_cast<VisCntsType*>(vc_);
  // logger("Iter LowerBound");
  return rocksdb::CompactionRouter::Iter(std::make_unique<VisCntsIter>(vc->seek(tier, viscnts_lsm::SKey(reinterpret_cast<const uint8_t*>(key.data()), key.size()))));
}

size_t VisCnts::TierNum() {
  return 2;
}

void VisCnts::Flush() {
  auto vc = static_cast<VisCntsType*>(vc_);
  vc->flush();
}

size_t VisCnts::GetHotSize(size_t tier) {
  auto vc = static_cast<VisCntsType*>(vc_);
  return vc->weight_sum(tier);
}


void VisCnts::Access(rocksdb::Slice key, size_t vlen) {
  return Access(0, key, vlen);
}
bool VisCnts::IsHot(rocksdb::Slice key) {
  return IsHot(0, key);
}
size_t VisCnts::RangeHotSize(rocksdb::RangeBounds range) {
  return RangeHotSize(0, range);
}
rocksdb::CompactionRouter::Iter VisCnts::Begin() {
  return Begin(0);
}
rocksdb::CompactionRouter::Iter VisCnts::LowerBound(rocksdb::Slice key) {
  return LowerBound(0, key);
}