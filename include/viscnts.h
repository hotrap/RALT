#ifndef VISCNTS_N_
#define VISCNTS_N_

#include <boost/fiber/buffered_channel.hpp>

#include "rocksdb/compaction_router.h"
#include "rocksdb/comparator.h"

class VisCnts {
public:
	VisCnts(const VisCnts&) = delete;
	~VisCnts();
	static VisCnts New(
		const rocksdb::Comparator *ucmp, const char *dir,
		size_t max_hot_set_size
	);
	size_t TierNum();
	void Access(size_t tier, rocksdb::Slice key, size_t vlen);
	bool IsHot(size_t tier, rocksdb::Slice key);
	void TransferRange(
		size_t target_tier, size_t source_tier, rocksdb::RangeBounds range
	);
	size_t RangeHotSize(
		size_t tier, rocksdb::RangeBounds range
	);
	std::unique_ptr<rocksdb::CompactionRouter::Iter> Begin(size_t tier);
	std::unique_ptr<rocksdb::CompactionRouter::Iter> LowerBound(
		size_t tier, rocksdb::Slice key
	);
private:
	VisCnts(void *vc) : vc_(vc) {}

	void* vc_;
};

#endif  // VISCNTS_N_