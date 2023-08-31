#ifndef VISCNTS_N_
#define VISCNTS_N_

#include <boost/fiber/buffered_channel.hpp>

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
	rocksdb::CompactionRouter::Iter Begin(size_t tier);
	rocksdb::CompactionRouter::Iter LowerBound(
		size_t tier, rocksdb::Slice key
	);
	std::unique_ptr<FastIter<rocksdb::Slice>> FastBegin(size_t tier);
	// single tier
	void Access(rocksdb::Slice key, size_t vlen);
	bool IsHot(rocksdb::Slice key);
	size_t RangeHotSize(rocksdb::RangeBounds range);
	rocksdb::CompactionRouter::Iter Begin();
	rocksdb::CompactionRouter::Iter LowerBound(rocksdb::Slice key);
	void Flush();
	size_t GetHotSize(size_t tier);
private:
	VisCnts(void *vc) : vc_(vc) {}

	void* vc_;
};

#endif  // VISCNTS_N_