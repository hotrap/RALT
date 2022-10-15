#include <bits/stdc++.h>

#include "memtable.hpp"

using namespace std;

extern void* VisCntsOpen(const char* path, double delta, bool createIfMissing);

extern int VisCntsAccess(void* ac, const char* key, size_t klen, size_t vlen);

extern bool VisCntsIsHot(void* ac, const char* key, size_t klen);

extern int VisCntsClose(void* ac);

extern void test_files();
extern void test_unordered_buf();
extern void test_lsm_store();
extern void test_lsm_store_and_scan();
extern void test_random_scan_and_count();
extern void test_lsm_decay();
extern void test_splay();

int main() {
  // test_basic();
  // test_memtable();
  // test_decay();
  // test_ops();
  // test_files();
  // test_unordered_buf();
  // test_lsm_store();
  // test_lsm_store_and_scan();
  // test_random_scan_and_count();
  // test_lsm_decay();
  test_splay();
}