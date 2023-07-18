#include "viscnts.h"
#include "logging.hpp"
#include "cache.hpp"

#include <fstream>

class DefaultComparator : public rocksdb::Comparator {
  public:
    DefaultComparator() {}
    const char* Name() const override {
      return "default comparator";
    }
    int Compare(const rocksdb::Slice& a, const rocksdb::Slice& b) const override {
      auto l = std::min(a.size(), b.size());
      int result = std::memcmp(a.data(), b.data(), l);
      if (result == 0) {
        if (a.size() < b.size()) {
          return -1;
        } else if (a.size() > b.size()) {
          return 1;
        } else {
          return 0;
        }
      }
      return result;
    }
    void FindShortestSeparator(std::string* start, const rocksdb::Slice& limit) const override {}
    void FindShortSuccessor(std::string* key) const override {};
} default_comp;

rocksdb::Slice convert_to_slice(char* output, size_t s, size_t len) {
  std::memset(output, 0, len - 8);
  std::memcpy(output + len - 8, &s, 8);
  return rocksdb::Slice(output, len);
}

size_t convert_to_int(rocksdb::Slice input) {
  return *reinterpret_cast<const size_t*>(input.data() + input.size() - 8);
}

void input_all(VisCnts& vc, int tier, const std::vector<std::pair<size_t, size_t>>& data, int TH, int vlen) {
  std::vector<std::future<void>> handles;
  for (int i = 0; i < TH; i++) {
    int L = (data.size() / TH + 1) * i;
    int R = std::min<int>(L + (data.size() / TH + 1), data.size());
    handles.push_back(std::async([L, R, &data, &vc, tier, vlen]() {
      char a[100];
      for (int j = L; j < R; j++) {
        vc.Access(tier, convert_to_slice(a, data[j].first, data[j].second), vlen);
      }
    }));
  }
  for (auto& a : handles) a.get();
  vc.Flush();
}

// first is string (8 byte), the second is null char after that.
std::vector<std::pair<size_t, size_t>> gen_testdata(int N, std::mt19937_64& gen) {
  std::vector<std::pair<size_t, size_t>> data(N);
  for (auto& [a, alen] : data) {
    std::uniform_int_distribution<> len_dis(8, 20);
    a = gen();
    alen = len_dis(gen);
  }
  return data;
}

void sort_data(std::vector<std::pair<size_t, size_t>>& data) {
  std::sort(data.begin(), data.end(), [&](auto x, auto y) -> bool {
    char ax[30], ay[30];
    return default_comp.Compare(convert_to_slice(ax, x.first, x.second), convert_to_slice(ay, y.first, y.second)) < 0;
  });
}

auto get_lower_bound_in_data(std::vector<std::pair<size_t, size_t>>& data, size_t x, size_t len) {
  return std::lower_bound(data.begin(), data.end(), std::make_pair(x, len), [&](auto x, auto y) -> bool {
    char ax[30], ay[30];
    return default_comp.Compare(convert_to_slice(ax, x.first, x.second), convert_to_slice(ay, y.first, y.second)) < 0;
  });
}

template<typename T1, typename T2>
void check_scan_result(T1 iter, T2 ans_iter, T2 ans_end) {
  while (true) {
    auto result = iter.next();
    if (result) {
      auto num = convert_to_int(*result);
      if (ans_iter->first != num) {
        DB_INFO("{}, {}", num, ans_iter->first);
      }
      DB_ASSERT(ans_iter != ans_end);
      DB_ASSERT(ans_iter->first == num);
      ans_iter++;
    } else {
      DB_ASSERT(ans_iter == ans_end);
      break;
    }
  }
}

template<typename F>
void parallel_run(int TH, F&& func) {
  std::vector<std::future<void>> handles;
  for (int i = 0; i < TH; i++) handles.push_back(std::async([&](){ func(); }));
  for (auto& a : handles) a.get();
}

void clear() {
  sync();
  system("bash -c \"echo 3 >/proc/sys/vm/drop_caches\"");
}


void test_store_and_scan() {
  size_t max_hot_set_size = 1e18;
  size_t N = 1e8, TH = 4, vlen = 10, Q = 1e4, QLEN = 100;
  auto vc = VisCnts::New(&default_comp, "/mnt/sd/tmp/viscnts/", max_hot_set_size);
  std::mt19937_64 gen(0x202306241834);
  auto data = gen_testdata(N, gen);
  StopWatch sw;
  input_all(vc, 0, data, TH, vlen);
  DB_INFO("input end. Used: {} s", sw.GetTimeInSeconds());
  sw.Reset();
  sort_data(data);
  DB_INFO("sort end. Used: {} s", sw.GetTimeInSeconds());
  clear();
  sw.Reset();

  
  {
    auto iter = vc.Begin(0);
    auto ans_iter = data.begin();
    while (true) {
      auto result = iter.next();
      if (result) {
        DB_ASSERT(ans_iter != data.end());
        if(!(ans_iter->first == convert_to_int(*result))) {
          DB_INFO("{}, {}, {}", ans_iter - data.begin(), ans_iter->first, convert_to_int(*result));
        }
        DB_ASSERT(ans_iter->first == convert_to_int(*result));
        ans_iter++;
      } else {
        DB_ASSERT(ans_iter == data.end());
        break;
      }
    }
  }

  DB_INFO("scan end. Used: {} s", sw.GetTimeInSeconds());
  clear();
  sw.Reset();  
  
  {
    auto iter = vc.FastBegin(0);
    auto ans_iter = data.begin();
    while (true) {
      auto result = iter->next();
      if (result.has_value()) {
        DB_ASSERT(ans_iter != data.end());
        if(!(ans_iter->first == convert_to_int(result.value()))) {
          DB_INFO("{}, {}, {}", ans_iter - data.begin(), ans_iter->first, convert_to_int(result.value()));
        }
        DB_ASSERT(ans_iter->first == convert_to_int(result.value()));
        ans_iter++;
      } else {
        DB_ASSERT(ans_iter == data.end());
        break;
      }
    }
    DB_INFO("fast scan end. Used: {} s", sw.GetTimeInSeconds());
  }
  sw.Reset();
  parallel_run(TH, [&](){
    for (int i = 0; i < Q; i++) {
      char a[30];
      size_t qx = gen(), qx_len = 8;
      auto iter = vc.LowerBound(0, convert_to_slice(a, qx, qx_len));
      auto ans_iter = get_lower_bound_in_data(data, qx, qx_len);
      for (int j = 0; j < QLEN; j++) {
        auto result = iter.next();
        if (result) {
          DB_ASSERT(ans_iter != data.end());
          DB_ASSERT(ans_iter->first == convert_to_int(*result));
          ans_iter++;
        } else {
          DB_ASSERT(ans_iter == data.end());
          break;
        }
      }
    }
  });
  DB_INFO("random scan end. Used: {} s", sw.GetTimeInSeconds());
}

void test_decay_simple() {
  // all keys are distinct.
  size_t max_hot_set_size = 6e9;
  size_t N = 3e7, TH = 4, vlen = 100, Q = 1e4, QLEN = 100;
  auto vc = VisCnts::New(&default_comp, "/tmp/viscnts/", max_hot_set_size);
  std::mt19937_64 gen(0x202306241834);
  auto data = gen_testdata(N, gen);
  StopWatch sw;
  input_all(vc, 0, data, TH, vlen);
  input_all(vc, 1, data, TH, vlen);
  input_all(vc, 0, data, TH, vlen);
  input_all(vc, 1, data, TH, vlen);
  DB_INFO("input end. Used: {} s", sw.GetTimeInSeconds());
  std::thread th([&]() {
    input_all(vc, 0, data, TH, vlen);
  });
  
  auto iter = vc.Begin(0);
  size_t sum = 0;
  int cnt = 0;
  while (true) {
    auto result = iter.next();
    if (result) {
      cnt += 1;
      sum += vlen + result->size();
    } else {
      break;
    }
  }
  DB_INFO("{}, {}, {}", cnt, sum, max_hot_set_size);
  DB_ASSERT(sum <= max_hot_set_size * 1.1);
  th.join();
}

void test_transfer_range() {
  size_t max_hot_set_size = 1e18;
  size_t N = 1e7, TH = 4, vlen = 10, Q = 1e4, QLEN = 100;
  auto vc = VisCnts::New(&default_comp, "/tmp/viscnts/", max_hot_set_size);
  std::mt19937_64 gen(0x202306242118);
  auto data0 = gen_testdata(N, gen);
  auto data1 = gen_testdata(N, gen);
  StopWatch sw;
  input_all(vc, 0, data0, TH, vlen);
  input_all(vc, 1, data1, TH, vlen);
  DB_INFO("input end. Used: {} s", sw.GetTimeInSeconds());
  sw.Reset();
  rocksdb::RangeBounds range;
  char ax[30], ay[30];
  range.start.user_key = convert_to_slice(ax, 1e18, 10);
  range.start.excluded = false;
  range.end.user_key = convert_to_slice(ay, 1e9, 10);
  range.end.excluded = false;
  vc.Flush();
  DB_INFO("{}, {}", vc.RangeHotSize(1, range), vc.GetHotSize(1));
  vc.TransferRange(0, 1, range);
  vc.Flush();
  DB_INFO("{}, {}", vc.RangeHotSize(1, range), vc.GetHotSize(1));
  DB_INFO("transfer end. Used: {} s", sw.GetTimeInSeconds());
  sw.Reset();

  char a[30];
  auto data_ans0 = data0;
  auto data_ans1 = decltype(data1)();
  for (auto [a0, a0len] : data1) {
    if (default_comp.Compare(range.start.user_key, convert_to_slice(a, a0, a0len)) <= 0
        && default_comp.Compare(convert_to_slice(a, a0, a0len), range.end.user_key) <= 0) {
      data_ans0.push_back({a0, a0len});
    } else {
      data_ans1.push_back({a0, a0len});
    }
  }
  sort_data(data_ans0);
  sort_data(data_ans1);
  DB_INFO("sort end. Used: {} s. Result set size: {}, {}", sw.GetTimeInSeconds(), data_ans0.size(), data_ans1.size());
  sw.Reset();

  check_scan_result(vc.Begin(0), data_ans0.begin(), data_ans0.end());
  check_scan_result(vc.Begin(1), data_ans1.begin(), data_ans1.end());
  DB_INFO("scan end. Used: {} s", sw.GetTimeInSeconds());
}

void test_parallel() {
  size_t max_hot_set_size = 1e9;
  size_t N = 1e7, NBLOCK = 1e5, TH = 4, vlen = 10, Q = 1e4, QLEN = 100;
  auto vc = VisCnts::New(&default_comp, "/tmp/viscnts/", max_hot_set_size);
  std::vector<std::future<void>> handles;
  for (int i = 0; i < TH; i++) {
    handles.push_back(std::async([&, i]() {
      std::mt19937_64 gen(0x202306251128 + i);
      for (int j = 0; j < N; j += NBLOCK) {
        auto data = gen_testdata(NBLOCK, gen);
        int tier = i & 1;
        char a[100];
        for (int k = 0; k < NBLOCK; k++) {
          vc.Access(tier, convert_to_slice(a, data[k].first, data[k].second), vlen);
        }
        auto iter = vc.Begin(tier);
        size_t sum = 0;
        while (true) {
          auto result = iter.next();
          if (result) {
            sum += vlen + result->size();
          } else {
            break;
          }
        }
        DB_INFO("{}, {}, {}", i, j / NBLOCK, sum);
      }
    }));
  }
  for (auto& a : handles) a.get();
  std::mt19937_64 gen(0x202306251200);
}

void test_ishot_simple() {
  size_t max_hot_set_size = 1e18;
  size_t N = 1e7, TH = 4, vlen = 10, Q = 1e4, QLEN = 100;
  auto vc = VisCnts::New(&default_comp, "/tmp/viscnts/", max_hot_set_size);
  std::mt19937_64 gen(0x202306291601);
  auto data = gen_testdata(N, gen);
  auto data2 = gen_testdata(N, gen);
  StopWatch sw;
  input_all(vc, 0, data, TH, vlen);
  DB_INFO("input end. Used: {} s", sw.GetTimeInSeconds());
  sw.Reset();
  {
    std::vector<std::future<void>> h;
    for (int i = 0; i < TH; i++) {
      h.push_back(std::async([L = N / 100 * i, R = N / 100 * (i + 1), &data, &vc]() {
      for (int j = L; j < R; j++) {
        char a[30];
        DB_ASSERT(vc.IsHot(0, convert_to_slice(a, data[j].first, data[j].second)));
      }}));    
    }
  }
  
  DB_INFO("true query end. Used: {} s", sw.GetTimeInSeconds());
  sw.Reset();
  for (int i = 0; i < N / 100; i++) {
    char a[30];
    DB_ASSERT(!vc.IsHot(0, convert_to_slice(a, data2[i].first, data2[i].second)));
  }
  DB_INFO("false query end. Used: {} s", sw.GetTimeInSeconds());
}

void test_cache_efficiency() {
  
}

int main() {
  test_store_and_scan();
  // test_decay_simple();
  // test_transfer_range();
  // test_parallel();
  // test_ishot_simple();
}