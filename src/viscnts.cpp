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
    if (x.data() == nullptr || y.data() == nullptr) return x.len() - y.len();
    rocksdb::Slice rx(reinterpret_cast<const char*>(x.data()), x.len());
    rocksdb::Slice ry(reinterpret_cast<const char*>(y.data()), y.len());
    return ucmp->Compare(rx, ry);
  }
};

using VisCntsType = viscnts_lsm::VisCnts<SKeyComparatorFromRocksDB>;

struct HotRecInfoAndIter {
  viscnts_lsm::EstimateLSM<SKeyComparatorFromRocksDB>::SuperVersionIterator* iter;
  VisCntsType* vc;
  rocksdb::HotRecInfo result;
};

VisCnts::VisCnts(const rocksdb::Comparator* ucmp, const char* path, bool createIfMissing,
                 boost::fibers::buffered_channel<std::tuple<>>* notify_weight_change)
    : notify_weight_change_(notify_weight_change), weight_sum_(0) {
  vc_ = new VisCntsType(SKeyComparatorFromRocksDB(ucmp), path, 1e100, createIfMissing, notify_weight_change);
}

void VisCnts::Access(const rocksdb::Slice& key, size_t vlen, double weight) {
  auto vc = reinterpret_cast<VisCntsType*>(vc_);
  vc->access({viscnts_lsm::SKey(reinterpret_cast<const uint8_t*>(key.data()), key.size()), viscnts_lsm::SValue(weight, vlen)});
}

double VisCnts::WeightSum() {
  auto vc = reinterpret_cast<VisCntsType*>(vc_);
  return vc->weight_sum();
}

void VisCnts::Decay() {
  auto vc = reinterpret_cast<VisCntsType*>(vc_);
  vc->trigger_decay();
}

void VisCnts::RangeDel(const rocksdb::Slice& L, const rocksdb::Slice& R) {
  auto vc = reinterpret_cast<VisCntsType*>(vc_);
  vc->delete_range({viscnts_lsm::SKey(reinterpret_cast<const uint8_t*>(L.data()), L.size()),
                    viscnts_lsm::SKey(reinterpret_cast<const uint8_t*>(R.data()), R.size())});
}

size_t VisCnts::RangeHotSize(const rocksdb::Slice& L, const rocksdb::Slice& R) {
  auto vc = reinterpret_cast<VisCntsType*>(vc_);
  return vc->range_data_size({viscnts_lsm::SKey(reinterpret_cast<const uint8_t*>(L.data()), L.size()),
                              viscnts_lsm::SKey(reinterpret_cast<const uint8_t*>(R.data()), R.size())});
}

void VisCnts::add_weight(double delta) { std::abort(); }

VisCnts::~VisCnts() {
  auto vc = reinterpret_cast<VisCntsType*>(vc_);
  delete vc;
}

VisCnts::Iter::Iter(VisCnts* ac) {
  viscnts_lsm::logger("[new_iter]");
  auto vc = reinterpret_cast<VisCntsType*>(ac->vc_);
  auto ret = new HotRecInfoAndIter();
  ret->vc = vc;
  ret->iter = nullptr;
  iter_ = ret;
}

const rocksdb::HotRecInfo* VisCnts::Iter::SeekToFirst() {
  viscnts_lsm::logger("[seek to first]");
  auto ac = reinterpret_cast<HotRecInfoAndIter*>(iter_);
  if (!ac->iter) delete ac->iter;
  ac->iter = ac->vc->seek_to_first();
  if (!ac->iter->valid()) return nullptr;
  auto [key, value] = ac->iter->read();
  ac->result =
      rocksdb::HotRecInfo{.slice = rocksdb::Slice(reinterpret_cast<const char*>(key.data()), key.len()), .count = value.counts, .vlen = value.vlen};
  return &ac->result;
}

const rocksdb::HotRecInfo* VisCnts::Iter::Seek(const rocksdb::Slice& key) {
  viscnts_lsm::logger("[seek]");
  auto ac = reinterpret_cast<HotRecInfoAndIter*>(iter_);
  if (!ac->iter) delete ac->iter;
  ac->iter = ac->vc->seek(viscnts_lsm::SKey(reinterpret_cast<const uint8_t*>(key.data()), key.size()));
  if (!ac->iter->valid()) return nullptr;
  auto [rkey, rvalue] = ac->iter->read();
  ac->result = rocksdb::HotRecInfo{
      .slice = rocksdb::Slice(reinterpret_cast<const char*>(rkey.data()), rkey.len()), .count = rvalue.counts, .vlen = rvalue.vlen};
  return &ac->result;
}

const rocksdb::HotRecInfo* VisCnts::Iter::Next() {
  auto ac = reinterpret_cast<HotRecInfoAndIter*>(iter_);
  ac->iter->next();
  if (!ac->iter->valid()) return nullptr;
  auto [rkey, rvalue] = ac->iter->read();
  ac->result = rocksdb::HotRecInfo{
      .slice = rocksdb::Slice(reinterpret_cast<const char*>(rkey.data()), rkey.len()), .count = rvalue.counts, .vlen = rvalue.vlen};
  return &ac->result;
}

VisCnts::Iter::~Iter() {
  viscnts_lsm::logger("[delete iter]");
  auto ac = reinterpret_cast<HotRecInfoAndIter*>(iter_);
  if (ac->iter) delete ac->iter;
  delete ac;
}