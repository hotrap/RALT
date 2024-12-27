#include "ralt.h"

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

#include "rocksdb/comparator.h"
#include "rocksdb/ralt.h"

namespace ralt {

struct SKeyComparatorFromRocksDB {
  const rocksdb::Comparator* ucmp;
  SKeyComparatorFromRocksDB() : ucmp(nullptr) {}
  SKeyComparatorFromRocksDB(const rocksdb::Comparator* _ucmp) : ucmp(_ucmp) {}
  int operator()(const ralt::SKey& x, const ralt::SKey& y) const {
    rocksdb::Slice rx(reinterpret_cast<const char*>(x.data()), x.len());
    rocksdb::Slice ry(reinterpret_cast<const char*>(y.data()), y.len());
    return ucmp->Compare(rx, ry);
  }
};

#ifdef USE_LRU
using VisCntsType = ralt::VisCnts<SKeyComparatorFromRocksDB, ralt::LRUTickValue, ralt::IndexData<1>, ralt::CachePolicyT::kUseFasterTick>;
#elif defined(USE_CLOCK)
using VisCntsType = ralt::VisCnts<SKeyComparatorFromRocksDB, ralt::ClockTickValue, ralt::IndexData<1>, ralt::CachePolicyT::kClockStyleDecay>;
#else
using VisCntsType = ralt::VisCnts<SKeyComparatorFromRocksDB, ralt::ExpTickValue, ralt::IndexData<1>, ralt::CachePolicyT::kUseFasterTick>;
#endif

class VisCntsIter : public rocksdb::TraitIterator<rocksdb::HotRecInfo> {
  public:
    VisCntsIter(std::unique_ptr<VisCntsType::IteratorT> it) 
      : it_(std::move(it)) {}
    ~VisCntsIter() {}
    std::optional<rocksdb::HotRecInfo> next() override {
      if (is_first_) {
        is_first_ = false;
      } else {
        it_->next();
      }
      if (it_->valid()) {
        auto key = it_->read().first;
        rocksdb::HotRecInfo ret;
        ret = rocksdb::Slice(reinterpret_cast<const char *>(key.data()),
                             key.len());
        return std::optional<rocksdb::HotRecInfo>(std::move(ret));
      }
      return std::optional<rocksdb::HotRecInfo>();
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

RALT::RALT(const Options &options, const rocksdb::Comparator *ucmp,
           const char *dir, size_t init_hot_set_size, size_t max_hot_set_size,
           size_t min_hot_set_size, size_t max_physical_size,
           uint64_t accessed_size_to_decr_counter)
    : vc_(new VisCntsType(options, SKeyComparatorFromRocksDB(ucmp), dir,
                          init_hot_set_size, max_hot_set_size, min_hot_set_size,
                          max_physical_size, accessed_size_to_decr_counter)) {}
RALT::RALT(const rocksdb::Comparator *ucmp, const char *dir,
           uint64_t init_hot_set_size, uint64_t max_hot_set_size,
           uint64_t min_hot_set_size, uint64_t max_physical_size,
           uint64_t accessed_size_to_decr_counter)
    : RALT(Options(), ucmp, dir, init_hot_set_size, max_hot_set_size,
           min_hot_set_size, max_physical_size, accessed_size_to_decr_counter) {
}

RALT::~RALT() { delete static_cast<VisCntsType *>(vc_); }

void RALT::Access(rocksdb::Slice key, size_t vlen) {
  auto vc = static_cast<VisCntsType*>(vc_);
  vc->access(ralt::SKey(reinterpret_cast<const uint8_t*>(key.data()), key.size()), vlen);
}
uint64_t RALT::RangeHotSize(rocksdb::Slice smallest, rocksdb::Slice largest) {
  auto vc = static_cast<VisCntsType *>(vc_);
  auto lkey = ralt::SKey(
      reinterpret_cast<const uint8_t *>(smallest.data()), smallest.size());
  auto rkey = ralt::SKey(
      reinterpret_cast<const uint8_t *>(largest.data()), largest.size());
  return vc->range_data_size({lkey, rkey});
}
rocksdb::RALT::Iter RALT::LowerBound(rocksdb::Slice key) {
  auto vc = static_cast<VisCntsType *>(vc_);
  // logger("Iter LowerBound");
  return rocksdb::RALT::Iter(
      std::make_unique<VisCntsIter>(vc->seek(ralt::SKey(
          reinterpret_cast<const uint8_t *>(key.data()), key.size()))));
}
bool RALT::IsHot(rocksdb::Slice key) {
  auto vc = static_cast<VisCntsType*>(vc_);
  // logger("is_hot");
  return vc->is_stably_hot(ralt::SKey(
      reinterpret_cast<const uint8_t *>(key.data()), key.size()));
}
bool RALT::IsStablyHot(rocksdb::Slice key) {
  auto vc = static_cast<VisCntsType*>(vc_);
  // logger("is_hot");
  return vc->is_stably_hot(ralt::SKey(reinterpret_cast<const uint8_t*>(key.data()), key.size()));
}
rocksdb::RALT::Iter RALT::Begin() {
  auto vc = static_cast<VisCntsType*>(vc_);
  logger("Iter Begin");
  return rocksdb::RALT::Iter(
      std::make_unique<VisCntsIter>(vc->seek_to_first()));
}
std::unique_ptr<FastIter<rocksdb::Slice>> RALT::FastBegin() {
  auto vc = static_cast<VisCntsType*>(vc_);
  return std::make_unique<FastVisCntsIter>(vc->seek_to_first());
}

void RALT::Flush() {
  auto vc = static_cast<VisCntsType*>(vc_);
  vc->flush();
}

size_t RALT::GetHotSize() {
  auto vc = static_cast<VisCntsType*>(vc_);
  return vc->weight_sum();
}

const std::string RALT::Properties::kReadBytes = "ralt.read.bytes";
const std::string RALT::Properties::kWriteBytes = "ralt.write.bytes";
const std::string RALT::Properties::kCompactionCPUNanos =
    "ralt.compaction.cpu.nanos";
const std::string RALT::Properties::kFlushCPUNanos = "ralt.flush.cpu.nanos";
const std::string RALT::Properties::kDecayScanCPUNanos =
    "ralt.decay.scan.cpu.nanos";
const std::string RALT::Properties::kDecayWriteCPUNanos =
    "ralt.decay.write.cpu.nanos";
const std::string RALT::Properties::kCompactionThreadCPUNanos =
    "ralt.compaction.thread.cpu.nanos";
const std::string RALT::Properties::kFlushThreadCPUNanos =
    "ralt.flush.thread.cpu.nanos";
const std::string RALT::Properties::kDecayThreadCPUNanos =
    "ralt.decay.thread.cpu.nanos";

struct PropertyInfo {
  bool (VisCntsType::*handle_int)(uint64_t *value);
};
const std::unordered_map<std::string, PropertyInfo> ppt_name_to_info = {
    {RALT::Properties::kReadBytes,
     {.handle_int = &VisCntsType::HandleReadBytes}},
    {RALT::Properties::kWriteBytes,
     {.handle_int = &VisCntsType::HandleWriteBytes}},
    {RALT::Properties::kCompactionCPUNanos,
     {.handle_int = &VisCntsType::HandleCompactionCPUNanos}},
    {RALT::Properties::kFlushCPUNanos,
     {.handle_int = &VisCntsType::HandleFlushCPUNanos}},
    {RALT::Properties::kDecayScanCPUNanos,
     {.handle_int = &VisCntsType::HandleDecayScanCPUNanos}},
    {RALT::Properties::kDecayWriteCPUNanos,
     {.handle_int = &VisCntsType::HandleDecayWriteCPUNanos}},
    {RALT::Properties::kCompactionThreadCPUNanos,
     {.handle_int = &VisCntsType::HandleCompactionThreadCPUNanos}},
    {RALT::Properties::kFlushThreadCPUNanos,
     {.handle_int = &VisCntsType::HandleFlushThreadCPUNanos}},
    {RALT::Properties::kDecayThreadCPUNanos,
     {.handle_int = &VisCntsType::HandleDecayThreadCPUNanos}},
};
bool RALT::GetIntProperty(std::string_view property, uint64_t *value) {
  std::string p(property);
  auto it = ppt_name_to_info.find(p);
  if (it == ppt_name_to_info.end())
      return false;
  const PropertyInfo *property_info = &it->second;
  assert(property_info->handle_int != nullptr);
  auto vc = static_cast<VisCntsType *>(vc_);
  return (vc->*(property_info->handle_int))(value);
}

void RALT::SetHotSetSizeLimit(size_t new_limit) {
  auto vc = static_cast<VisCntsType*>(vc_);
  vc->set_new_hot_limit(new_limit);
}

void RALT::SetPhysicalSizeLimit(size_t new_limit) {
  auto vc = static_cast<VisCntsType*>(vc_);
  vc->set_new_phy_limit(new_limit);
}

void RALT::SetAllSizeLimit(size_t new_hs_limit, size_t new_phy_limit) {
  auto vc = static_cast<VisCntsType*>(vc_);
  vc->set_all_limit(new_hs_limit, new_phy_limit);
}

size_t RALT::GetPhySizeLimit() {
  auto vc = static_cast<VisCntsType*>(vc_);
  return vc->get_phy_limit();
}

size_t RALT::GetHotSetSizeLimit() {
  auto vc = static_cast<VisCntsType*>(vc_);
  return vc->get_hot_set_limit();
}

uint64_t RALT::GetMinHotSetSizeLimit() {
  auto vc = static_cast<VisCntsType *>(vc_);
  return vc->get_min_hot_size_limit();
}
void RALT::SetMinHotSetSizeLimit(uint64_t min_hot_size_limit) {
  auto vc = static_cast<VisCntsType *>(vc_);
  vc->set_min_hot_size_limit(min_hot_size_limit);
}

uint64_t RALT::GetMaxHotSetSizeLimit() {
  auto vc = static_cast<VisCntsType *>(vc_);
  return vc->get_max_hot_size_limit();
}
void RALT::SetMaxHotSetSizeLimit(uint64_t max_hot_size_limit) {
  auto vc = static_cast<VisCntsType *>(vc_);
  vc->set_max_hot_size_limit(max_hot_size_limit);
}

void RALT::SetProperPhysicalSizeLimit() {
  auto vc = static_cast<VisCntsType*>(vc_);
  vc->set_proper_phy_limit();
}

size_t RALT::DecayCount() {
  auto vc = static_cast<VisCntsType *>(vc_);
  return vc->decay_count();
}

size_t RALT::GetRealHotSetSize() {
  auto vc = static_cast<VisCntsType *>(vc_);
  return vc->get_real_hs_size();
}

size_t RALT::GetRealPhySize() {
  auto vc = static_cast<VisCntsType*>(vc_);
  return vc->get_real_phy_size();
}

}  // namespace ralt
