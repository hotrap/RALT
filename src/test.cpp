#include <bits/stdc++.h>

#include "memtable.hpp"

using namespace std;

extern void test_files();
extern void test_unordered_buf();
extern void test_lsm_store();
extern void test_lsm_store_and_scan();
extern void test_random_scan_and_count();
extern void test_lsm_decay();
extern void test_splay();
extern void test_delete_range();

int main() {
  test_files();
  // test_unordered_buf();
  test_lsm_store();
  test_lsm_store_and_scan();
  test_random_scan_and_count();
  test_lsm_decay();
  test_splay();
  test_delete_range();
}