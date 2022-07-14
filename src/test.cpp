#include <bits/stdc++.h>

#include "memtable.hpp"

using namespace std;

extern void* VisCntsOpen(const char* path, double delta, bool createIfMissing);

extern int VisCntsAccess(void* ac, const char* key, size_t klen, size_t vlen);

extern bool VisCntsIsHot(void* ac, const char* key, size_t klen);

extern int VisCntsClose(void* ac);

void test_basic() {
  auto vc = VisCntsOpen("/tmp/viscnts/", 1 << 30, 1);
  VisCntsAccess(vc, "a", 1, 1);
  // assert(VisCntsIsHot(vc, "a", 1) == 1);
  // assert(VisCntsIsHot(vc, "b", 1) == 0);
  for (int i = 0; i < 10000000; ++i) {
    VisCntsAccess(vc, (char*)(&i), 4, 1);
    if (i % 1000000 == 0) printf("[ins: %d]", i), fflush(stdout);
  }
  // for(int i = 0; i < 10000000; ++i) {
  //     assert(VisCntsIsHot(vc,  (char*)(&i), 4) == 1);
  // }
  puts("Pass");
  VisCntsClose(vc);
}

void test_memtable() {
  using namespace viscnts_lsm;
  MemTable T;
  std::set<int> st;
  for (int i = 0, x = rand() % 1000000; i < 1000000; ++i, x = rand() % 1000000) T.append(SKey((uint8_t*)(&x), sizeof(int)), SValue()), st.insert(x);
  puts("insert complete");
  fflush(stdout);
  for (int i = 0; i < 1000000; ++i) assert(T.exists(SKey((uint8_t*)(&i), 4)) == st.count(i));
  puts("assert complete");
  fflush(stdout);
  T.release();
  st.clear();
  for (int i = 0, x = rand() % 1000000; i < 1000000; ++i, x = rand() % 1000000) T.append(SKey((uint8_t*)(&x), sizeof(int)), SValue()), st.insert(x);
  puts("insert2 complete");
  fflush(stdout);
  for (int i = 0; i < 1000000; ++i) assert(T.exists(SKey((uint8_t*)(&i), 4)) == st.count(i));
  puts("assert2 complete");
  fflush(stdout);
  puts("memtable pass");
  fflush(stdout);
}

int main() {
  test_basic();
  // test_memtable();
}