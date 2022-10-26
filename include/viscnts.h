#ifndef VISCNTS_N_
#define VISCNTS_N_

#include "rocksdb/comparator.h"

extern void *VisCntsOpen(const rocksdb::Comparator *ucmp, const char *path,
        bool createIfMissing);
extern double VisCntsAccess(void *ac, const rocksdb::Slice *key, size_t vlen,
                double weight);
double VisCntsDecay(void *ac);

extern void *VisCntsNewIter(void *ac);

// The returned pointer will stay valid until the next call to one of these
// functions
extern const rocksdb::HotRecInfo *VisCntsSeekToFirst(void *iter);
extern const rocksdb::HotRecInfo *VisCntsSeek(void *iter,
                const rocksdb::Slice *key);
extern const rocksdb::HotRecInfo *VisCntsNext(void *iter);

extern void VisCntsDelIter(void *iter);
extern double VisCntsRangeDel(void *ac, const rocksdb::Slice *smallest,
                const rocksdb::Slice *largest);
extern int VisCntsClose(void *ac);

#endif // VISCNTS_N_
