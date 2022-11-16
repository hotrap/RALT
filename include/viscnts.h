#ifndef VISCNTS_N_
#define VISCNTS_N_

#include "rocksdb/comparator.h"
#include "rocksdb/compaction_router.h"

#include <boost/fiber/buffered_channel.hpp>

class VisCnts {
public:
	class Iter {
	public:
		Iter(VisCnts* vc);
		~Iter();
		// The returned pointer will stay valid until the next call to one of
		// these functions
		const rocksdb::HotRecInfo* SeekToFirst();
		const rocksdb::HotRecInfo* Seek(const rocksdb::Slice* key);
		const rocksdb::HotRecInfo* Next();
	private:
		void* iter_;
		rocksdb::HotRecInfo cur_;
		friend class VisCnts;
	};

	VisCnts(const rocksdb::Comparator *ucmp, const char *path,
		bool createIfMissing,
		boost::fibers::buffered_channel<std::tuple<>>* notify_weight_change);
	VisCnts(const VisCnts&) = delete;
	~VisCnts();
	void Access(const rocksdb::Slice *key, size_t vlen, double weight);
	double WeightSum();
	void Decay();
	void RangeDel(const rocksdb::Slice* smallest,
		const rocksdb::Slice* largest);
private:
	void add_weight(double delta);

	void* vc_;
	boost::fibers::buffered_channel<std::tuple<>>* notify_weight_change_;

	std::mutex lock_;
	double weight_sum_;
};

#endif	// VISCNTS_N_