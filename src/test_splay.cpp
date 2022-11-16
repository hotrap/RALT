#include <cassert>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "splay.hpp"

void test_splay() {
  using namespace viscnts_lsm;
  struct Data {
    std::string key;
    int val;
    int sum;
  };
  auto Compare = [](const Data& x, const Data& y) { return x.key < y.key ? -1 : (x.key == y.key ? 0 : 1); };

  auto Update = [](Data& x, const Data& lc, const Data& rc) { x.sum = lc.sum + rc.sum + x.val; };
  auto SoloUpdate = [](Data& x, const Data& lc) { x.sum = lc.sum + x.val; };
  auto EmptyUpdate = [](Data& x) { x.sum = x.val; };
  auto Union = [](Data& x, const Data& dup) {
    x.val += dup.val;
    x.sum += dup.val;
  };
  auto GenStr = []() {
    int len = rand() % 10 + 1;
    std::string ret;
    while (len--) ret += 'a' + rand() % 26;
    return ret;
  };
  auto GenData = [GenStr]() {
    Data ret;
    ret.key = GenStr();
    ret.val = rand();
    ret.sum = rand();
    return ret;
  };
  Splay<Data, decltype(Update), decltype(SoloUpdate), decltype(EmptyUpdate), decltype(Union), decltype(Compare)> tree(Update, SoloUpdate, EmptyUpdate,
                                                                                                                      Union, Compare);
  std::vector<Data> vec;
  int n = 1e4;
  for (int i = 0; i < n; i++) {
    auto d = GenData();
    vec.push_back(d);
    tree.insert(d);

    auto ask = GenStr();
    Data qstr{ask, 0, 0};
    auto output_key = tree.presum(qstr);
    auto output = output_key ? output_key->sum : 0;
    auto ans = 0;
    for (auto& a : vec)
      if (Compare(a, qstr) < 0) ans += a.val;
    assert(ans == output);

    ask = GenStr();
    auto upper_iter = tree.upper(Data{ask, 0, 0});
    int j = -1;
    for (int k = 0; k < vec.size(); ++k)
      if (Compare(Data{ask, 0, 0}, vec[k]) <= 0 && (j == -1 || Compare(vec[k], vec[j]) < 0)) j = k;
    if (j == -1) assert(upper_iter == nullptr);
    else assert(Compare(vec[j], upper_iter->key) == 0);
    
  }

  auto start = std::chrono::system_clock::now();
  n = 1e6;
  for (int i = 0; i < n; i++) {
    auto d = GenData();
    tree.insert(d);
  }
  auto end = std::chrono::system_clock::now();
  auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout <<  "used time: " << double(dur.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den << std::endl;
}