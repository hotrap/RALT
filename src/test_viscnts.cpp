#include "viscnts.h"
#include "logging.hpp"

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

template<typename F>
void parallel_run(int TH, F&& func) {
  std::vector<std::future<void>> handles;
  for (int i = 0; i < TH; i++) handles.push_back(std::async([&](){ func(); }));
  for (auto& a : handles) a.get();
}


void test_store_and_scan() {
  size_t max_hot_set_size = 1e18;
  size_t N = 1e8, TH = 4, vlen = 10, Q = 1e4, QLEN = 100;
  auto vc = VisCnts::New(&default_comp, "/tmp/viscnts/", max_hot_set_size);
  std::mt19937_64 gen(0x202306241834);
  auto data = gen_testdata(N, gen);
  StopWatch sw;
  input_all(vc, 0, data, TH, vlen);
  DB_INFO("input end. Used: {} s", sw.GetTimeInSeconds());
  sw.Reset();
  sort_data(data);
  DB_INFO("sort end. Used: {} s", sw.GetTimeInSeconds());
  sw.Reset();
  auto iter = vc.Begin(0);
  auto ans_iter = data.begin();
  while (true) {
    auto result = iter->next();
    if (result) {
      DB_ASSERT(ans_iter != data.end());
      DB_ASSERT(ans_iter->first == convert_to_int(*result));
      ans_iter++;
    } else {
      DB_ASSERT(ans_iter == data.end());
      break;
    }
  }
  DB_INFO("scan end. Used: {} s", sw.GetTimeInSeconds());
  sw.Reset();
  parallel_run(TH, [&](){
    for (int i = 0; i < Q; i++) {
      char a[30];
      size_t qx = gen(), qx_len = 8;
      auto iter = vc.LowerBound(0, convert_to_slice(a, qx, qx_len));
      auto ans_iter = get_lower_bound_in_data(data, qx, qx_len);
      for (int j = 0; j < QLEN; j++) {
        auto result = iter->next();
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

int main() {
  test_store_and_scan();
}