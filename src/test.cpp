#include <bits/stdc++.h>

#include "memtable.hpp"

using namespace std;

extern void* VisCntsOpen(const char* path, double delta, bool createIfMissing);

extern int VisCntsAccess(void* ac, const char* key, size_t klen, size_t vlen);

extern bool VisCntsIsHot(void* ac, const char* key, size_t klen);

extern int VisCntsClose(void* ac);

class KeyTests {
 public:
  KeyTests() {}
  int gen(std::mt19937& rnd, int l, int r) {
    uniform_int_distribution<int> dis(l, r);
    int x = dis(rnd);
    return x;
  }
  template <typename Append, typename Exists>
  void test1(const Append& _append, const Exists& _exist, int R) {
    std::set<int> st;
    std::random_device rd;
    std::mt19937 rnd(19260817);
    for (int i = 0; i < R; i++) {
      int x = gen(rnd, 0, R);
      if (i & 1) {
        _append(x);
        st.insert(x);
      } else {
        if(_exist(x) != st.count(x)) {
          printf("failed: i = %d, x = %d, _exist = %d, st.count = %d\n", i, x, _exist(x), st.count(x));
          fflush(stdout);
          exit(-1);
        }
        // assert(_exist(x) == st.count(x));
      }
    }
  }

  std::mutex m_;
  template <typename Append, typename Exists>
  void test2(const Append& _append, const Exists& _exist, int L, int R) {
    std::set<int> st;
    std::random_device rd;
    std::mt19937 rnd(rd());
    for (int i = 0; i < R - L; i++) {
      int x = gen(rnd, L, R);
      if (i % 5 == 0) {
        st.insert(x);
        std::unique_lock lck_(m_);
        _append(x);
      } else {
        assert(_exist(x) == st.count(x));
      }
    }
  }

  template <typename Append>
  void test3(const Append& _append, int R) {
    std::random_device rd;
    std::mt19937 rnd(19260817);
    for (int i = 0; i < R; i++) {
      int x = gen(rnd, 0, R);
      _append(x);
    }
  }

  template <typename Append, typename Exists>
  void test4(const Append& _append, const Exists& _exist, int R) {
    for (int i = 0; i < R; i++) {
      _append(i);
    }
    int cnt = 0;
    for (int i = 0; i < R; i++) {
      cnt += _exist(i);
    }
    printf("%d", cnt);
  }
};

void test_basic() {
  auto vc = VisCntsOpen("/tmp/viscnts/", 1 << 30, 1);
  KeyTests A;
  A.test1([&](int x) { VisCntsAccess(vc, (char*)(&x), 4, 1); }, [&](int x) { return VisCntsIsHot(vc, (char*)(&x), 4); }, 1e7);
  A.test3([&](int x) { VisCntsAccess(vc, (char*)(&x), 4, 1); }, 1e6);
  puts("[Basic] Pass single thread");
  VisCntsClose(vc);
  vc = VisCntsOpen("/tmp/viscnts/", 1 << 30, 1);
  std::vector<std::thread> v;
  for (int i = 0; i < 20; ++i) {
    v.emplace_back([i, &A, &vc]() {
      A.test2([&](int x) { VisCntsAccess(vc, (char*)(&x), 4, 1); }, [&](int x) { return VisCntsIsHot(vc, (char*)(&x), 4); }, 5000000 * i + 1,
              5000000 * (i + 1));
    });
  }
  for (auto& a : v) a.join();
  puts("[Basic] Pass multi-thread #1");
  VisCntsClose(vc);
}

void test_memtable() {
  using namespace viscnts_lsm;
  MemTable* T = new MemTable;
  KeyTests A;
  A.test1([&](int x) { T->append(SKey((uint8_t*)&x, 4), SValue()); }, [&](int x) { return T->exists(SKey((uint8_t*)&x, 4)); }, 1e6);
  T->unref();
  T = new MemTable();
  A.test1([&](int x) { T->append(SKey((uint8_t*)&x, 4), SValue()); }, [&](int x) { return T->exists(SKey((uint8_t*)&x, 4)); }, 1e6);
  T->unref();
  puts("[Memtable] Pass single thread");
  T = new MemTable();
  std::vector<std::thread> v;
  for (int i = 0; i < 10; ++i) {
    v.emplace_back([i, &T, &A]() {
      A.test2([&](int x) { T->append(SKey((uint8_t*)&x, 4), SValue()); }, [&](int x) { return T->exists(SKey((uint8_t*)&x, 4)); }, 100000 * i + 1,
              100000 * (i + 1));
    });
  }
  for (auto& a : v) a.join();
  puts("[Memtable] Pass multi-thread #1");

}


void test_decay() {
  auto vc = VisCntsOpen("/tmp/viscnts", 1e6, 1);
  KeyTests A;
  A.test4([&](int x) { VisCntsAccess(vc, (char*)(&x), 4, 1); }, [&](int x) { return VisCntsIsHot(vc, (char*)(&x), 4); }, 5e6);
  VisCntsClose(vc);
  puts("[Decay] Pass single thread");
  vc = VisCntsOpen("/tmp/viscnts", 1e6, 1);
  std::vector<std::thread> v;
  for (int i = 0; i < 10; ++i) {
    v.emplace_back([i, &vc, &A]() {
      A.test4([&](int x) { VisCntsAccess(vc, (char*)(&x), 4, 1); }, [&](int x) { return VisCntsIsHot(vc, (char*)(&x), 4); }, 2e5);
    });
  }
  for (auto& a : v) a.join();
  VisCntsClose(vc);
  puts("[Decay] Pass multi-thread #1");
}

int main() {
  test_basic();
  test_memtable();
  test_decay();
}