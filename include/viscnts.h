#ifndef VISCNTS_N_
#define VISCNTS_N_

#include <boost/fiber/buffered_channel.hpp>
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
	VisCnts(const VisCnts&) = delete;
	~VisCnts();
	static VisCnts New(
		const rocksdb::Comparator *ucmp, const char *dir,
		size_t max_hot_set_size, size_t max_physical_size
	);
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
	size_t DecayCount();
private:
	VisCnts(void *vc) : vc_(vc) {}

	void* vc_;
};

#endif  // VISCNTS_N_