#include "viscnts.h"
#include "logging.hpp"
#include "cache.hpp"

#include <fstream>
#include <set>
#include <future>

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
  std::memset(output, 'a', len - 8);
  std::memcpy(output + len - 8, &s, 8);
  return rocksdb::Slice(output, len);
}

size_t convert_to_int(rocksdb::Slice input) {
  return *reinterpret_cast<const size_t*>(input.data() + input.size() - 8);
}

void input_all(RALT &vc, const std::vector<std::pair<size_t, size_t>> &data,
               int TH, int vlen) {
  std::vector<std::future<void>> handles;
  for (int i = 0; i < TH; i++) {
    int L = (data.size() / TH + 1) * i;
    int R = std::min<int>(L + (data.size() / TH + 1), data.size());
    handles.push_back(std::async([L, R, &data, &vc, vlen]() {
      Timer timer;
      char a[100];
      for (int j = L; j < R; j++) {
        vc.Access(convert_to_slice(a, data[j].first, data[j].second), vlen);
      }
      DB_INFO("timer: {}", timer.GetTimeInNanos());
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
    auto result = iter->next();
    if (result.has_value()) {
      auto num = convert_to_int(result.value());
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
  size_t max_physical_size = 1e18;
  size_t N = 1e6, TH = 4, vlen = 10, Q = 1e4, QLEN = 100;
  RALT vc(&default_comp, "/mnt/sd/tmp/viscnts/", max_hot_set_size,
          max_hot_set_size, max_hot_set_size, max_physical_size);
  std::mt19937_64 gen(0x202306241834);
  auto data = gen_testdata(N, gen);
  DB_INFO("gen_testdata end.");
  StopWatch sw;
  input_all(vc, data, TH, vlen);
  DB_INFO("input end. Used: {} s", sw.GetTimeInSeconds());
  sw.Reset();
  sort_data(data);
  DB_INFO("sort end. Used: {} s", sw.GetTimeInSeconds());
  clear();
  sw.Reset();

  
  {
    auto iter = vc.Begin();
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
  }

  DB_INFO("scan end. Used: {} s", sw.GetTimeInSeconds());
  clear();
  sw.Reset();  
  
  {
    auto iter = vc.FastBegin();
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
      auto iter = vc.LowerBound(convert_to_slice(a, qx, qx_len));
      auto ans_iter = get_lower_bound_in_data(data, qx, qx_len);
      for (int j = 0; j < QLEN; j++) {
        auto result = iter->next();
        if (result.has_value()) {
          DB_ASSERT(ans_iter != data.end());
          DB_ASSERT(ans_iter->first == convert_to_int(result.value()));
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
  size_t max_hot_set_size = 1e9;
  size_t max_physical_size = 1e18;
  size_t N = 1e7, TH = 4, vlen = 100, Q = 1e4, QLEN = 100;
  RALT vc(&default_comp, "/tmp/viscnts/", max_hot_set_size, max_hot_set_size,
          max_hot_set_size, max_physical_size);
  std::mt19937_64 gen(0x202306241834);
  auto data = gen_testdata(N, gen);
  StopWatch sw;
  input_all(vc, data, TH, vlen);
  input_all(vc, data, TH, vlen);
  input_all(vc, data, TH, vlen);
  input_all(vc, data, TH, vlen);
  DB_INFO("input end. Used: {} s", sw.GetTimeInSeconds());
  std::thread th([&]() {
    input_all(vc, data, TH, vlen);
  });
  
  auto iter = vc.Begin();
  size_t sum = 0;
  int cnt = 0;
  while (true) {
    auto result = iter->next();
    if (result.has_value()) {
      cnt += 1;
      sum += vlen + result.value().size();
    } else {
      break;
    }
  }
  DB_INFO("{}, {}, {}", cnt, sum, max_hot_set_size, max_physical_size);
  DB_ASSERT(sum <= max_hot_set_size * 1.1);
  th.join();
}

void test_decay_hit_rate() {
  // all keys are distinct.
  size_t N = 1e7, TH = 8, vlen = 1000, Q = 1e4, QLEN = 100;
  size_t max_hot_set_cnt = N * 0.05;
  size_t max_hot_set_size = max_hot_set_cnt * vlen;
  size_t max_physical_size = 60 * max_hot_set_cnt;
  RALT vc(&default_comp, "viscnts/", N * 0.05 * vlen, N * 0.05 * vlen,
          N * 0.05 * vlen, max_physical_size);
  std::mt19937_64 gen(0x202311101830);
  auto data = gen_testdata(N, gen);
  auto hot_data = decltype(data)(data.begin(), data.begin() + max_hot_set_cnt);
  std::set<std::pair<size_t, size_t>> hots;
  for (auto& a : hot_data) hots.insert(a);
  auto cold_data = decltype(data)(data.begin() + max_hot_set_cnt, data.end());
  StopWatch sw;
  auto real_data = decltype(data)();
  for (int i = 0, cnt0 = 0, cnt1 = 0; i < N; i++) {
    std::uniform_real_distribution<> dis(0, 1);
    if (dis(gen) < 0.95) {
      std::uniform_int_distribution<> dis2(0, hot_data.size() - 1);
      real_data.push_back(hot_data[dis2(gen)]);
    } else {
      std::uniform_int_distribution<> dis2(0, cold_data.size() - 1);
      real_data.push_back(cold_data[dis2(gen)]);
    }
  }
  auto start = std::clock();
  Timer timer;
  input_all(vc, real_data, TH, vlen);
  DB_INFO("input end. Used: {} s", sw.GetTimeInSeconds());
  
  auto iter = vc.Begin();
  size_t sum = 0;
  int cnt = 0;
  while (true) {
    auto result = iter->next();
    if (result.has_value()) {
      cnt += hots.count(std::make_pair(convert_to_int(result.value()), result.value().size()));
      sum += 1;
    } else {
      break;
    }
  }
  DB_INFO("{}, {}, {}", cnt, sum, hots.size());
  iter = nullptr;
  vc.SetHotSetSizeLimit(max_hot_set_size * 0.5);
  vc.Flush();
  sum = cnt = 0;
  iter = vc.Begin();
  while (true) {
    auto result = iter->next();
    if (result.has_value()) {
      cnt += hots.count(std::make_pair(convert_to_int(result.value()), result.value().size()));
      sum += 1;
    } else {
      break;
    }
  }
  DB_INFO("{}", timer.GetTimeInNanos());
  DB_INFO("{}, {}, {}", cnt, sum, hots.size());
  DB_INFO("{}", std::clock() - start);
}

void test_parallel() {
  size_t max_hot_set_size = 1e9;
  size_t max_physical_size = 1e18;
  size_t N = 1e7, NBLOCK = 1e5, TH = 4, vlen = 10, Q = 1e4, QLEN = 100;
  RALT vc(&default_comp, "/tmp/viscnts/", max_hot_set_size, max_hot_set_size,
          max_hot_set_size, max_physical_size);
  std::vector<std::future<void>> handles;
  for (int i = 0; i < TH; i++) {
    handles.push_back(std::async([&, i]() {
      std::mt19937_64 gen(0x202306251128 + i);
      for (int j = 0; j < N; j += NBLOCK) {
        auto data = gen_testdata(NBLOCK, gen);
        char a[100];
        for (int k = 0; k < NBLOCK; k++) {
          vc.Access(convert_to_slice(a, data[k].first, data[k].second), vlen);
        }
        auto iter = vc.Begin();
        size_t sum = 0;
        while (true) {
          auto result = iter->next();
          if (result.has_value()) {
            sum += vlen + (result.value()).size();
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
  size_t max_physical_size = 1e18;
  size_t N = 1e7, TH = 4, vlen = 10, Q = 1e4, QLEN = 100;
  RALT vc(&default_comp, "/tmp/viscnts/", max_hot_set_size, max_hot_set_size,
          max_hot_set_size, max_physical_size);
  std::mt19937_64 gen(0x202306291601);
  auto data = gen_testdata(N, gen);
  auto data2 = gen_testdata(N, gen);
  StopWatch sw;
  input_all(vc,  data, TH, vlen);
  DB_INFO("input end. Used: {} s", sw.GetTimeInSeconds());
  sw.Reset();
  {
    std::vector<std::future<void>> h;
    for (int i = 0; i < TH; i++) {
      h.push_back(std::async([L = N / 100 * i, R = N / 100 * (i + 1), &data, &vc]() {
      for (int j = L; j < R; j++) {
        char a[30];
        DB_ASSERT(vc.IsHot(convert_to_slice(a, data[j].first, data[j].second)));
      }}));    
    }
  }
  
  DB_INFO("true query end. Used: {} s", sw.GetTimeInSeconds());
  sw.Reset();
  for (int i = 0; i < N / 100; i++) {
    char a[30];
    DB_ASSERT(!vc.IsHot(convert_to_slice(a, data2[i].first, data2[i].second)));
  }
  DB_INFO("false query end. Used: {} s", sw.GetTimeInSeconds());
}
void print_memory() {
  std::system(("ps -q " + std::to_string(getpid()) +
              " -o rss | tail -n 1")
                .c_str());
}

void test_cache_efficiency() {
  
}

void test_stable_hot() {
  size_t max_hot_set_size = 1e18;
  size_t max_physical_size = 1e18;
  size_t N = 1e6, TH = 16, vlen = 10, Q = 1e4, QLEN = 100;
  RALT vc(&default_comp, "/tmp/viscnts/", max_hot_set_size, max_hot_set_size,
          max_hot_set_size, max_physical_size);
  std::mt19937_64 gen(0x202306291601);
  auto data = gen_testdata(N, gen);
  auto data2 = gen_testdata(N, gen);
  auto data3 = gen_testdata(N, gen);
  StopWatch sw;
  input_all(vc, data, TH, vlen);
  input_all(vc, data2, TH, vlen);
  input_all(vc, data, TH, vlen);
  DB_INFO("input end. Used: {} s", sw.GetTimeInSeconds());
  print_memory();
  sw.Reset();
  {
    std::vector<std::future<void>> h;
    for (int i = 0; i < TH; i++) {
      h.push_back(std::async([L = N / TH * i, R = N / TH * (i + 1), &data, &vc]() {
        int sum = 0;
        for (int j = L; j < R; j++) {
          char a[30];
          sum += vc.IsStablyHot(convert_to_slice(a, data[j].first, data[j].second));
        }
        DB_INFO("{}/{}", sum, R - L);
      }));    
    }
    
  }
  print_memory();
  DB_INFO("true query end. Used: {} s", sw.GetTimeInSeconds());
  sw.Reset();
  {
    std::vector<std::future<void>> h;
    for (int i = 0; i < TH; i++) {
      h.push_back(std::async([L = N / TH * i, R = N / TH * (i + 1), &data2, &vc]() {
        int sum = 0;
        for (int j = L; j < R; j++) {
          char a[30];
          sum += !vc.IsStablyHot(convert_to_slice(a, data2[j].first, data2[j].second));
        }
        DB_INFO("{}/{}", sum, R - L);
      }));    
    }
  }
  print_memory();
  DB_INFO("false query end. Used: {} s", sw.GetTimeInSeconds());
  sw.Reset();
  {
    std::vector<std::future<void>> h;
    for (int i = 0; i < TH; i++) {
      h.push_back(std::async([L = N / TH * i, R = N / TH * (i + 1), &data3, &vc]() {
        int sum = 0;
        for (int j = L; j < R; j++) {
          char a[30];
          sum += !vc.IsStablyHot(convert_to_slice(a, data3[j].first, data3[j].second));
        }
        DB_INFO("{}/{}", sum, R - L);
      }));    
    }
  }
  print_memory();
  DB_INFO("false query end. Used: {} s", sw.GetTimeInSeconds());
}

void test_lowerbound() {
  size_t max_hot_set_size = 1e18;
  size_t max_physical_size = 1e18;
  size_t N = 1e7, TH = 4, vlen = 10;
  RALT vc(&default_comp, "/tmp/viscnts/", max_hot_set_size, max_hot_set_size,
          max_hot_set_size, max_physical_size);
  std::mt19937_64 gen(0x202309252052);
  auto data = gen_testdata(N, gen);
  StopWatch sw;
  input_all(vc, data, TH, vlen);
  DB_INFO("input end. Used: {} s", sw.GetTimeInSeconds());
  for (int i = 0; i < data.size(); i++) {
    char a[30];
    vc.LowerBound(convert_to_slice(a, data[i].first, data[i].second));
  }
}

void test_range_hot_size() {
  size_t max_hot_set_size = 1e18;
  size_t max_physical_size = 1e18;
  size_t N = 1e7, TH = 4, vlen = 10, Q = 500000;
  RALT vc(&default_comp, "/tmp/viscnts/", max_hot_set_size, max_hot_set_size,
          max_hot_set_size, max_physical_size);
  std::mt19937_64 gen(0x202309252052);
  auto data = gen_testdata(N, gen);
  StopWatch sw;
  input_all(vc, data, TH, vlen);
  sort_data(data);
  DB_INFO("input end. Used: {} s", sw.GetTimeInSeconds());
  size_t total_size = 0;
  // for (int i = 0; i < data.size(); i++) {
  //   char a[30], a2[30];
  //   rocksdb::RangeBounds range;
  //   range.start.user_key = convert_to_slice(a, data[0].first, data[0].second);
  //   range.start.excluded = false;
  //   range.end.user_key = convert_to_slice(a2, data[i].first, data[i].second);
  //   range.end.excluded = false;
  //   total_size += vlen + data[i].second;
  //   DB_ASSERT(vc.RangeHotSize(range) >= total_size);
  //   // DB_INFO("{}, {}", vc.RangeHotSize(range), total_size);
  // }
  std::vector<size_t> S(data.size());
  for (int i = 0; i < data.size(); i++) {
    S[i] = (i == 0 ? 0 : S[i - 1]) + data[i].second + vlen;
  }
  for (int i = 0; i < Q; i++) {
    char a[30], a2[30];
    rocksdb::RangeBounds range;
    size_t l = std::uniform_int_distribution<>(0, data.size() - 1)(gen);
    size_t r = std::uniform_int_distribution<>(l, data.size() - 1)(gen);
    size_t ans = S[r] - (l == 0 ? 0 : S[l - 1]);
    size_t out =
        vc.RangeHotSize(convert_to_slice(a, data[l].first, data[l].second),
                        convert_to_slice(a2, data[r].first, data[r].second));
    DB_INFO("{},{}",ans,out);
    DB_ASSERT(out >= ans);
    // DB_INFO("{}, {}", vc.RangeHotSize(range), total_size);
  }
}

int main() {
  // test_store_and_scan();
  // test_decay_simple();
  // test_decay_hit_rate();
  // test_transfer_range();
  // test_parallel();
  // test_ishot_simple();
  test_stable_hot();
  // test_lowerbound();
  // test_range_hot_size();
}
