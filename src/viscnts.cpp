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
#include <unordered_map>

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

#ifdef USE_LRU
using VisCntsType = viscnts_lsm::VisCnts<SKeyComparatorFromRocksDB, viscnts_lsm::LRUTickValue, viscnts_lsm::IndexData<1>, viscnts_lsm::CachePolicyT::kUseFasterTick>;
#elif defined(USE_CLOCK)
using VisCntsType = viscnts_lsm::VisCnts<SKeyComparatorFromRocksDB, viscnts_lsm::ClockTickValue, viscnts_lsm::IndexData<1>, viscnts_lsm::CachePolicyT::kClockStyleDecay>;
#else
using VisCntsType = viscnts_lsm::VisCnts<SKeyComparatorFromRocksDB, viscnts_lsm::ExpTickValue, viscnts_lsm::IndexData<1>, viscnts_lsm::CachePolicyT::kUseFasterTick>;
#endif

class VisCntsIter : public rocksdb::TraitIterator<rocksdb::HotRecInfo> {
  public:
    VisCntsIter(std::unique_ptr<VisCntsType::IteratorT> it) 
      : it_(std::move(it)) {}
    ~VisCntsIter() {}
    rocksdb::optional<rocksdb::HotRecInfo> next() override {
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
        return rocksdb::optional<rocksdb::HotRecInfo>(std::move(ret));
      }
      return rocksdb::optional<rocksdb::HotRecInfo>();
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
		size_t init_hot_set_size, size_t max_hot_set_size, size_t min_hot_set_size, size_t max_physical_size) {
  return VisCnts(new VisCntsType(SKeyComparatorFromRocksDB(ucmp), dir, init_hot_set_size, max_hot_set_size, min_hot_set_size, max_physical_size));
}

VisCnts::~VisCnts() {
  delete static_cast<VisCntsType*>(vc_);
}

void VisCnts::Access(rocksdb::Slice key, size_t vlen) {
  auto vc = static_cast<VisCntsType*>(vc_);
  vc->access(viscnts_lsm::SKey(reinterpret_cast<const uint8_t*>(key.data()), key.size()), vlen);
}
bool VisCnts::IsHot(rocksdb::Slice key) {
  auto vc = static_cast<VisCntsType*>(vc_);
  // logger("is_hot");
  return vc->is_hot(viscnts_lsm::SKey(reinterpret_cast<const uint8_t*>(key.data()), key.size()));
}
bool VisCnts::IsStablyHot(rocksdb::Slice key) {
  auto vc = static_cast<VisCntsType*>(vc_);
  // logger("is_hot");
  return vc->is_stably_hot(viscnts_lsm::SKey(reinterpret_cast<const uint8_t*>(key.data()), key.size()));
}
size_t VisCnts::RangeHotSize(
  rocksdb::RangeBounds range
) {
  auto vc = static_cast<VisCntsType*>(vc_);
  auto lkey = viscnts_lsm::SKey(reinterpret_cast<const uint8_t*>(range.start.user_key.data()), range.start.user_key.size());
  auto rkey = viscnts_lsm::SKey(reinterpret_cast<const uint8_t*>(range.end.user_key.data()), range.end.user_key.size());
  return vc->range_data_size({lkey, rkey});
}
rocksdb::CompactionRouter::Iter VisCnts::Begin() {
  auto vc = static_cast<VisCntsType*>(vc_);
  logger("Iter Begin");
  return rocksdb::CompactionRouter::Iter(std::make_unique<VisCntsIter>(vc->seek_to_first()));
}
std::unique_ptr<FastIter<rocksdb::Slice>> VisCnts::FastBegin() {
  auto vc = static_cast<VisCntsType*>(vc_);
  return std::make_unique<FastVisCntsIter>(vc->seek_to_first());
}
rocksdb::CompactionRouter::Iter VisCnts::LowerBound(
  rocksdb::Slice key
) {
  auto vc = static_cast<VisCntsType*>(vc_);
  // logger("Iter LowerBound");
  return rocksdb::CompactionRouter::Iter(std::make_unique<VisCntsIter>(vc->seek(viscnts_lsm::SKey(reinterpret_cast<const uint8_t*>(key.data()), key.size()))));
}

size_t VisCnts::TierNum() {
  return 2;
}

void VisCnts::Flush() {
  auto vc = static_cast<VisCntsType*>(vc_);
  vc->flush();
}

size_t VisCnts::GetHotSize() {
  auto vc = static_cast<VisCntsType*>(vc_);
  return vc->weight_sum();
}

const std::string VisCnts::Properties::kReadBytes = "viscnts.read.bytes";
const std::string VisCnts::Properties::kWriteBytes = "viscnts.write.bytes";
const std::string VisCnts::Properties::kCompactionCPUNanos =
    "viscnts.compaction.cpu.nanos";
const std::string VisCnts::Properties::kFlushCPUNanos =
    "viscnts.flush.cpu.nanos";
const std::string VisCnts::Properties::kDecayScanCPUNanos =
    "viscnts.decay.scan.cpu.nanos";
const std::string VisCnts::Properties::kDecayWriteCPUNanos =
    "viscnts.decay.write.cpu.nanos";
const std::string VisCnts::Properties::kCompactionThreadCPUNanos =
    "viscnts.compaction.thread.cpu.nanos";
const std::string VisCnts::Properties::kFlushThreadCPUNanos =
    "viscnts.flush.thread.cpu.nanos";
const std::string VisCnts::Properties::kDecayThreadCPUNanos =
    "viscnts.decay.thread.cpu.nanos";

struct PropertyInfo {
  bool (VisCntsType::*handle_int)(uint64_t *value);
};
const std::unordered_map<std::string, PropertyInfo> ppt_name_to_info = {
    {VisCnts::Properties::kReadBytes,
     {.handle_int = &VisCntsType::HandleReadBytes}},
    {VisCnts::Properties::kWriteBytes,
     {.handle_int = &VisCntsType::HandleWriteBytes}},
    {VisCnts::Properties::kCompactionCPUNanos,
     {.handle_int = &VisCntsType::HandleCompactionCPUNanos}},
    {VisCnts::Properties::kFlushCPUNanos,
     {.handle_int = &VisCntsType::HandleFlushCPUNanos}},
    {VisCnts::Properties::kDecayScanCPUNanos,
     {.handle_int = &VisCntsType::HandleDecayScanCPUNanos}},
    {VisCnts::Properties::kDecayWriteCPUNanos,
     {.handle_int = &VisCntsType::HandleDecayWriteCPUNanos}},
    {VisCnts::Properties::kCompactionThreadCPUNanos,
     {.handle_int = &VisCntsType::HandleCompactionThreadCPUNanos}},
    {VisCnts::Properties::kFlushThreadCPUNanos,
     {.handle_int = &VisCntsType::HandleFlushThreadCPUNanos}},
    {VisCnts::Properties::kDecayThreadCPUNanos,
     {.handle_int = &VisCntsType::HandleDecayThreadCPUNanos}},
};
bool VisCnts::GetIntProperty(std::string_view property, uint64_t *value) {
  std::string p(property);
  auto it = ppt_name_to_info.find(p);
  if (it == ppt_name_to_info.end())
      return false;
  const PropertyInfo *property_info = &it->second;
  assert(property_info->handle_int != nullptr);
  auto vc = static_cast<VisCntsType *>(vc_);
  return (vc->*(property_info->handle_int))(value);
}

void VisCnts::SetHotSetSizeLimit(size_t new_limit) {
  auto vc = static_cast<VisCntsType*>(vc_);
  vc->set_new_hot_limit(new_limit);
}

void VisCnts::SetPhysicalSizeLimit(size_t new_limit) {
  auto vc = static_cast<VisCntsType*>(vc_);
  vc->set_new_phy_limit(new_limit);
}

void VisCnts::SetAllSizeLimit(size_t new_hs_limit, size_t new_phy_limit) {
  auto vc = static_cast<VisCntsType*>(vc_);
  vc->set_all_limit(new_hs_limit, new_phy_limit);
}

size_t VisCnts::GetPhySizeLimit() {
  auto vc = static_cast<VisCntsType*>(vc_);
  return vc->get_phy_limit();
}

size_t VisCnts::GetHotSetSizeLimit() {
  auto vc = static_cast<VisCntsType*>(vc_);
  return vc->get_hot_set_limit();
}

uint64_t VisCnts::GetMinHotSetSizeLimit() {
  auto vc = static_cast<VisCntsType *>(vc_);
  return vc->get_min_hot_size_limit();
}
void VisCnts::SetMinHotSetSizeLimit(uint64_t min_hot_size_limit) {
  auto vc = static_cast<VisCntsType *>(vc_);
  vc->set_min_hot_size_limit(min_hot_size_limit);
}

uint64_t VisCnts::GetMaxHotSetSizeLimit() {
  auto vc = static_cast<VisCntsType *>(vc_);
  return vc->get_max_hot_size_limit();
}
void VisCnts::SetMaxHotSetSizeLimit(uint64_t max_hot_size_limit) {
  auto vc = static_cast<VisCntsType *>(vc_);
  vc->set_max_hot_size_limit(max_hot_size_limit);
}

void VisCnts::SetProperPhysicalSizeLimit() {
  auto vc = static_cast<VisCntsType*>(vc_);
  vc->set_proper_phy_limit();
}

size_t VisCnts::DecayCount() {
  auto vc = static_cast<VisCntsType*>(vc_);
  return vc->decay_count();
}


size_t VisCnts::GetRealHotSetSize() {
  auto vc = static_cast<VisCntsType*>(vc_);
  return vc->get_real_hs_size();
}


size_t VisCnts::GetRealPhySize() {
  auto vc = static_cast<VisCntsType*>(vc_);
  return vc->get_real_phy_size();
}