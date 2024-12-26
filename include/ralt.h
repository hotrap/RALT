#ifndef VISCNTS_N_
#define VISCNTS_N_

#include <optional>

#include "options.h"
#include "rocksdb/comparator.h"
#include "rocksdb/ralt.h"

namespace ralt {

template <typename T>
class FastIter {
 public:
  virtual std::optional<T> next() = 0;
};

class RALT : public rocksdb::RALT {
 public:
  RALT(const RALT &) = delete;
  ~RALT();
  RALT(const ralt::Options &options, const rocksdb::Comparator *ucmp,
       const char *dir, size_t init_hot_set_size, size_t max_hot_set_size,
       size_t min_hot_set_size, size_t max_physical_size,
       uint64_t accessed_size_to_decr_counter);
  const char *Name() const override { return "RALT-LSM"; }
  void Access(rocksdb::Slice key, size_t vlen) override;
  uint64_t RangeHotSize(rocksdb::Slice smallest,
                        rocksdb::Slice largest) override;
  rocksdb::RALT::Iter LowerBound(rocksdb::Slice key) override;
  bool IsHot(rocksdb::Slice key) override;

  bool IsStablyHot(rocksdb::Slice key);
  rocksdb::RALT::Iter Begin();
  std::unique_ptr<FastIter<rocksdb::Slice>> FastBegin();
  void Flush();
  size_t GetHotSize();

  void SetHotSetSizeLimit(size_t new_limit);
  void SetPhysicalSizeLimit(size_t new_limit);
  void SetAllSizeLimit(size_t new_hs_limit, size_t new_phy_limit);
  void SetProperPhysicalSizeLimit();
  size_t GetPhySizeLimit();
  size_t GetHotSetSizeLimit();
  uint64_t GetMinHotSetSizeLimit();
  void SetMinHotSetSizeLimit(uint64_t min_hot_size_limit);
  uint64_t GetMaxHotSetSizeLimit();
  void SetMaxHotSetSizeLimit(uint64_t max_hot_size_limit);
  size_t DecayCount();
  size_t GetRealHotSetSize();
  size_t GetRealPhySize();

  struct Properties {
    static const std::string kReadBytes;
    static const std::string kWriteBytes;
    static const std::string kCompactionCPUNanos;
    static const std::string kFlushCPUNanos;
    static const std::string kDecayScanCPUNanos;
    static const std::string kDecayWriteCPUNanos;
    static const std::string kCompactionThreadCPUNanos;
    static const std::string kFlushThreadCPUNanos;
    static const std::string kDecayThreadCPUNanos;
  };

  // If "property" is a valid integer property understood by this
  // implementation, fills "*value" with its current value and returns true.
  // Otherwise, returns false.
  bool GetIntProperty(std::string_view property, uint64_t *value);

 private:
  void *vc_;
};

}  // namespace ralt

#endif // VISCNTS_N_