#include <bits/stdc++.h>

using namespace std;

extern void* VisCntsOpen(const char* path, double delta, bool createIfMissing);

extern int VisCntsAccess(void* ac, const char* key, size_t klen, size_t vlen);

extern bool VisCntsIsHot(void* ac, const char* key, size_t klen);

extern int VisCntsClose(void* ac);


void test_basic() {
    auto vc = VisCntsOpen("/tmp/viscnts/", 10, 1);
    VisCntsAccess(vc, "a", 1, 1);
    assert(VisCntsIsHot(vc, "a", 1));
    VisCntsClose(vc);
}

int main() {
    test_basic();
}