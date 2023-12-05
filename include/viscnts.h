#ifndef VISCNTS_N_
#define VISCNTS_N_

#include <optional>

#include "rocksdb/compaction_router.h"
#include "rocksdb/comparator.h"

template<typename T>
class FastIter {
public:
  virtual std::optional<T> next() = 0;
};

class VisCnts {
public:
  VisCnts(const VisCnts &) = delete;
  ~VisCnts();
  static VisCnts New(const rocksdb::Comparator *ucmp, const char *dir,
                     size_t max_hot_set_size, size_t max_physical_size);
  size_t TierNum();
  void Access(rocksdb::Slice key, size_t vlen);
  bool IsHot(rocksdb::Slice key);
  bool IsStablyHot(rocksdb::Slice key);
  size_t RangeHotSize(rocksdb::RangeBounds range);
  rocksdb::CompactionRouter::Iter Begin();
  std::unique_ptr<FastIter<rocksdb::Slice>> FastBegin();
  rocksdb::CompactionRouter::Iter LowerBound(rocksdb::Slice key);
  void Flush();
  size_t GetHotSize();

  void SetHotSetSizeLimit(size_t new_limit);
  void SetPhysicalSizeLimit(size_t new_limit);
  size_t DecayCount();

  struct Properties {
    static const std::string kCompactionCPUNanos;
    static const std::string kFlushCPUNanos;
    static const std::string kDecayScanCPUNanos;
    static const std::string kDecayWriteCPUNanos;
  };

  // If "property" is a valid integer property understood by this
  // implementation, fills "*value" with its current value and returns true.
  // Otherwise, returns false.
  bool GetIntProperty(std::string_view property, uint64_t *value);

private:
  VisCnts(void *vc) : vc_(vc) {}

  void *vc_;
};

#endif  // VISCNTS_N_