#include <bits/stdc++.h>

#include "memtable.hpp"
#include "viscnts_lsm.hpp"

using namespace std;
/// testing function.

void test_files() {
  using namespace viscnts_lsm;
  int L = 1e6, FS = 20;
  uint8_t a[12];
  memset(a, 0, sizeof(a));

  std::unique_ptr<Env> env_(createDefaultEnv());
  std::vector<SSTBuilder> builders(FS);
  for (int i = 0; i < FS; i++)
    builders[i].new_file(
        std::make_unique<WriteBatch>(std::unique_ptr<AppendFile>(env_->openAppFile("/tmp/viscnts/test" + std::to_string(i)))));

  for (int i = 0; i < L; i++) {
    for (int j = 0; j < 12; j++) a[j] = i >> (j % 4) * 8 & 255;
    builders[abs(rand()) % FS].append({SKey(a, 12), SValue()});
  }
  int sum = 0;
  for (int i = 0; i < FS; ++i) sum += builders[i].size();
  logger_printf("[TEST] SIZE: [%d]\n", sum);
  for (int i = 0; i < FS; i++) {
    builders[i].make_index();
    builders[i].finish();
  }

  auto comp = +[](const SKey& a, const SKey& b) {
    auto ap = a.data(), bp = b.data();
    uint32_t x = ap[0] | ((uint32_t)ap[1] << 8) | ((uint32_t)ap[2] << 16) | ((uint32_t)ap[3] << 24);
    uint32_t y = bp[0] | ((uint32_t)bp[1] << 8) | ((uint32_t)bp[2] << 16) | ((uint32_t)bp[3] << 24);
    return (int)x - (int)y;
  };

  std::vector<ImmutableFile<KeyCompType*>> files;
  for (int i = 0; i < FS; ++i)
    files.push_back(ImmutableFile<KeyCompType*>(0, builders[i].size(),
                                                std::unique_ptr<RandomAccessFile>(env_->openRAFile("/tmp/viscnts/test" + std::to_string(i))),
                                                new DefaultAllocator(), {}, comp));
  auto iters = std::make_unique<SeqIteratorSet<SSTIterator<KeyCompType*>, KeyCompType*>>(comp);
  for (int i = 0; i < FS; ++i) {
    SSTIterator iter(&files[i]);
    iters->push(std::move(iter));
  }
  iters->build();
  for (int i = 0; i < L; i++) {
    DB_ASSERT(iters->valid());
    auto kv = iters->read();
    int x = 0;
    auto a = kv.first.data();
    for (int j = 0; j < 12; j++) {
      x |= a[j] << (j % 4) * 8;
      DB_ASSERT(j % 4 != 3 || x == i);
      if (j % 4 == 3) x = 0;
    }
    iters->next();
  }
  DB_ASSERT(!iters->valid());
  logger("test_file(): OK");
}

void test_unordered_buf() {
  using namespace viscnts_lsm;
  UnsortedBufferPtrs bufs(kUnsortedBufferSize, 100);
  int L = 1e7, TH = 10;
  std::atomic<int> signal = 0;
  std::vector<std::thread> v;
  std::vector<std::pair<IndSKey, SValue>> result;

  auto comp = +[](const SKey& a, const SKey& b) {
    auto ap = a.data(), bp = b.data();
    uint32_t x = ap[0] | ((uint32_t)ap[1] << 8) | ((uint32_t)ap[2] << 16) | ((uint32_t)ap[3] << 24);
    uint32_t y = bp[0] | ((uint32_t)bp[1] << 8) | ((uint32_t)bp[2] << 16) | ((uint32_t)bp[3] << 24);
    return (int)x - (int)y;
  };
  auto start = std::chrono::system_clock::now();
  auto th = std::thread(
      [comp](std::atomic<int>& signal, UnsortedBufferPtrs& bufs, std::vector<std::pair<IndSKey, SValue>>& result) {
        while (true) {
          auto buf_q_ = signal.load() ? bufs.get() : bufs.wait_and_get();
          using namespace std::chrono;

          if (!buf_q_.size()) {
            if (signal) break;
            continue;
          }
          for (auto& buf : buf_q_) {
            buf->sort(comp);
            UnsortedBuffer::Iterator iter(*buf);
            while (iter.valid()) {
              result.emplace_back(iter.read());
              iter.next();
            }
            buf->clear();
          }
        }
      },
      std::ref(signal), std::ref(bufs), std::ref(result));
  for (int i = 0; i < TH; i++) {
    v.emplace_back(
        [i, L, TH](UnsortedBufferPtrs& bufs) {
          int l = L / TH * i, r = L / TH * (i + 1);
          std::vector<int> v(r - l);
          for (int i = l; i < r; i++) v[i - l] = i;
          std::shuffle(v.begin(), v.end(), std::mt19937(std::random_device()()));
          uint8_t a[12];
          memset(a, 0, sizeof(a));
          for (auto& i : v) {
            for (int j = 0; j < 12; j++) a[j] = i >> (j % 4) * 8 & 255;
            bufs.append_and_notify(SKey(a, 12), SValue());
          }
        },
        std::ref(bufs));
  }
  for (auto& a : v) a.join();
  logger("OK!");
  bufs.flush();
  signal = 1;
  bufs.terminate();
  th.join();

  auto end = std::chrono::system_clock::now();
  auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << double(dur.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den << std::endl;

  logger_printf("RESULT_SIZE: %d\n", result.size());
  DB_ASSERT(result.size() == (uint32_t)L);
  std::set<int> st;
  for (uint32_t i = 0; i < result.size(); ++i) {
    int x = 0;
    auto a = result[i].first.data();
    for (int j = 0; j < 12; j++) {
      x |= a[j] << (j % 4) * 8;
      if (j % 4 == 3) {
        DB_ASSERT(x >= 0 && x < L);
        DB_ASSERT(!st.count(x));
        st.insert(x);
        break;
      }
    }
  }
  logger("test_unordered_buf(): OK");
}

void test_lsm_store() {
  using namespace viscnts_lsm;

  auto start = std::chrono::system_clock::now();
  {
    EstimateLSM<KeyCompType*> tree(std::unique_ptr<Env>(createDefaultEnv()), std::make_unique<FileName>(0, "/tmp/viscnts/"),
                                   std::make_unique<DefaultAllocator>(), SKeyCompFunc);
    int L = 1e7;
    uint8_t a[12];
    memset(a, 0, sizeof(a));
    for (int i = 0; i < L; i++) {
      for (int j = 0; j < 12; j++) a[j] = i >> (j % 4) * 8 & 255;
      tree.append(SKey(a, 12), SValue());
    }
  }

  auto end = std::chrono::system_clock::now();
  auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << double(dur.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den << std::endl;
  logger("test_lsm_store(): OK");
}

void test_lsm_store_and_scan() {
  using namespace viscnts_lsm;

  auto comp = +[](const SKey& a, const SKey& b) {
    auto ap = a.data(), bp = b.data();
    uint32_t x = ap[0] | ((uint32_t)ap[1] << 8) | ((uint32_t)ap[2] << 16) | ((uint32_t)ap[3] << 24);
    uint32_t y = bp[0] | ((uint32_t)bp[1] << 8) | ((uint32_t)bp[2] << 16) | ((uint32_t)bp[3] << 24);
    return (int)x - (int)y;
  };
  auto start = std::chrono::system_clock::now();
  {
    EstimateLSM<KeyCompType*> tree(std::unique_ptr<Env>(createDefaultEnv()), std::make_unique<FileName>(0, "/tmp/viscnts/"),
                                  std::make_unique<DefaultAllocator>(), comp);
    int L = 1e8;
    std::vector<int> numbers(L);
    for (int i = 0; i < L; i++) numbers[i] = i;
    std::shuffle(numbers.begin(), numbers.end(), std::mt19937(std::random_device()()));
    start = std::chrono::system_clock::now();
    std::vector<std::thread> threads;
    int TH = 4;
    for (int i = 0; i < TH; i++) {
      threads.emplace_back(
          [i, L, TH](std::vector<int>& numbers, EstimateLSM<KeyCompType*>& tree) {
            uint8_t a[16];
            int l = (L / TH + 1) * i, r = std::min((L / TH + 1) * (i + 1), (int)numbers.size());
            for (int i = l; i < r; ++i) {
              for (int j = 0; j < 16; j++) a[j] = numbers[i] >> (j % 4) * 8 & 255;
              tree.append(SKey(a, 16), SValue(1, 1));
            }
          },
          std::ref(numbers), std::ref(tree));
    }
    for (auto& a : threads) a.join();
    threads.clear();
    std::shuffle(numbers.begin(), numbers.end(), std::mt19937(std::random_device()()));

    auto _numbers = numbers;
    std::shuffle(_numbers.begin(), _numbers.end(), std::mt19937(std::random_device()()));

    for (int i = 0; i < TH; i++) {
      threads.emplace_back(
          [i, L, TH](std::vector<int>& numbers, EstimateLSM<KeyCompType*>& tree) {
            uint8_t a[16];
            int l = (L / TH + 1) * i, r = std::min((L / TH + 1) * (i + 1), (int)numbers.size());
            for (int i = l; i < r; ++i) {
              for (int j = 0; j < 16; j++) a[j] = numbers[i] >> (j % 4) * 8 & 255;
              tree.append(SKey(a, 16), SValue(numbers[i], 1));
            }
          },
          std::ref(_numbers), std::ref(tree));
    }
    for (auto& a : threads) a.join();
    tree.all_flush();
    auto end = std::chrono::system_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << double(dur.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den << std::endl;
    start = std::chrono::system_clock::now();
    auto iter = std::unique_ptr<EstimateLSM<KeyCompType*>::SuperVersionIterator>(tree.seek_to_first());
    for (int i = 0; i < L; i++) {
      DB_ASSERT(iter->valid());
      auto kv = iter->read();
      int x = 0, y = 0;
      auto a = kv.first.data();
      for (int j = 0; j < 16; j++) {
        x |= a[j] << (j % 4) * 8;
        DB_ASSERT(j % 4 != 3 || x == i);
        if (j % 4 == 3) {
          if (j > 3) {
            DB_ASSERT(y == x);
          }
          y = x;
          x = 0;
        }
      }
      DB_ASSERT(kv.second.counts == i + 1);
      DB_ASSERT(kv.second.vlen == 1);
      iter->next();
    }
  }

  auto end = std::chrono::system_clock::now();
  auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << double(dur.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den << std::endl;
  logger("test_lsm_store_and_scan(): OK");
}

void test_random_scan_and_count() {
  using namespace viscnts_lsm;

  auto start = std::chrono::system_clock::now();
  {
    EstimateLSM<KeyCompType*> tree(std::unique_ptr<Env>(createDefaultEnv()), std::make_unique<FileName>(0, "/tmp/viscnts/"),
                                   std::make_unique<DefaultAllocator>(), SKeyCompFunc);
    int L = 3e7, Q = 1e4;
    std::vector<int> numbers(L);
    auto comp2 = +[](int x, int y) {
      uint8_t a[12], b[12];
      for (int j = 0; j < 12; j++) a[j] = x >> (j % 4) * 8 & 255;
      for (int j = 0; j < 12; j++) b[j] = y >> (j % 4) * 8 & 255;
      return SKeyCompFunc(SKey(a, 12), SKey(b, 12)) < 0;
    };
    for (int i = 0; i < L; i++) numbers[i] = i;
    // std::sort(numbers.begin(), numbers.end(), comp2);
    std::shuffle(numbers.begin(), numbers.end(), std::mt19937(std::random_device()()));
    srand(std::random_device()());
    for (int i = 0; i < L / 2; i++) {
      uint8_t a[12];
      for (int j = 0; j < 12; j++) a[j] = numbers[i] >> (j % 4) * 8 & 255;
      tree.append(SKey(a, 12), SValue(1, 1));
    }
    tree.all_flush();

    auto end = std::chrono::system_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    logger("flush used time: ", double(dur.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den);
    std::this_thread::sleep_for(std::chrono::milliseconds(5000));
    auto numbers2 = std::vector<int>(numbers.begin(), numbers.begin() + L / 2);
    std::sort(numbers2.begin(), numbers2.end(), comp2);

    start = std::chrono::system_clock::now();
    for (int i = 0; i < Q; i++) {
      uint8_t a[12], b[12];
      int id = abs(rand()) % numbers.size();
      int x = numbers[id];
      for (int j = 0; j < 12; j++) a[j] = numbers[id] >> (j % 4) * 8 & 255;
      id = abs(rand()) % numbers.size();
      int y = numbers[id];
      for (int j = 0; j < 12; j++) b[j] = numbers[id] >> (j % 4) * 8 & 255;
      int ans = std::upper_bound(numbers2.begin(), numbers2.end(), x, comp2) - std::lower_bound(numbers2.begin(), numbers2.end(), y, comp2);
      ans = std::max(ans, 0);
      int output = tree.range_count({SKey(b, 12), SKey(a, 12)});
      DB_ASSERT(ans == output);
    }

    int QLEN = 1000;

    end = std::chrono::system_clock::now();
    dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    start = end;
    logger("range count used time: ", double(dur.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den);

    for (int i = 0; i < Q; i++) {
      uint8_t a[12];
      int id = abs(rand()) % numbers.size();
      for (int j = 0; j < 12; j++) a[j] = numbers[id] >> (j % 4) * 8 & 255;

      auto iter = std::unique_ptr<EstimateLSM<KeyCompType*>::SuperVersionIterator>(tree.seek(SKey(a, 12)));
      auto check_func = [](const uint8_t* a, int goal) {
        int x = 0, y = 0;
        for (int j = 0; j < 12; j++) {
          x |= a[j] << (j % 4) * 8;
          DB_ASSERT(j % 4 != 3 || x == goal);
          if (j % 4 == 3) {
            if (j > 3) {
              DB_ASSERT(y == x);
            }
            y = x;
            x = 0;
          }
        }
      };
      if (id < L / 2) {
        DB_ASSERT(iter->valid());
        auto kv = iter->read();
        auto a = kv.first.data();
        check_func(a, numbers[id]);
        iter->next();
      } else {
        // DB_ASSERT(iter->valid() || i == mx);
        // DB_ASSERT(!iter->valid() || i != mx);
      }
      int x = a[0] | ((uint32_t)a[1] << 8) | ((uint32_t)a[2] << 16) | ((uint32_t)a[3] << 24);

      int cnt = 0;
      auto it = std::upper_bound(numbers2.begin(), numbers2.end(), x, comp2);
      while (true) {
        if (++cnt > QLEN) break;
        DB_ASSERT((it != numbers2.end()) == (iter->valid()));
        if (it == numbers2.end()) break;
        auto a = iter->read().first.data();
        check_func(a, *it);
        it++;
        iter->next();
      }
    }
    end = std::chrono::system_clock::now();
    dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    logger("random scan used time: ", double(dur.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den);
  }
  logger("test_random_scan(): OK");
}

void test_lsm_decay() {
  using namespace viscnts_lsm;

  auto start = std::chrono::system_clock::now();
  {
    EstimateLSM<KeyCompType*> tree(std::unique_ptr<Env>(createDefaultEnv()), std::make_unique<FileName>(0, "/tmp/viscnts/"),
                                   std::make_unique<DefaultAllocator>(), SKeyCompFunc);
    int L = 3e7;
    std::vector<int> numbers(L);
    // auto comp2 = +[](int x, int y) {
    //   uint8_t a[12], b[12];
    //   for (int j = 0; j < 12; j++) a[j] = x >> (j % 4) * 8 & 255;
    //   for (int j = 0; j < 12; j++) b[j] = y >> (j % 4) * 8 & 255;
    //   return SKeyCompFunc(SKey(a, 12), SKey(b, 12)) < 0;
    // };
    for (int i = 0; i < L; i++) numbers[i] = i;
    std::shuffle(numbers.begin(), numbers.end(), std::mt19937(std::random_device()()));
    srand(std::random_device()());
    for (int i = 0; i < L; i++) {
      uint8_t a[12];
      for (int j = 0; j < 12; j++) a[j] = numbers[i] >> (j % 4) * 8 & 255;
      tree.append(SKey(a, 12), SValue(1, 1));
    }
    uint8_t a[12];
    memset(a, 0, sizeof(a));
    auto iter = std::unique_ptr<EstimateLSM<KeyCompType*>::SuperVersionIterator>(tree.seek(SKey(a, 12)));
    double ans = 0;
    while (iter->valid()) {
      auto kv = iter->read();
      ans += (kv.second.vlen + 12) * std::min(kv.second.counts * 0.5, 1.);
      iter->next();
    }
    logger("decay size: ", ans);
    auto end = std::chrono::system_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    logger("decay used time: ", double(dur.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den);
  }
  logger("test_lsm_decay(): OK");
}

void test_delete_range() {
  using namespace viscnts_lsm;

  auto start = std::chrono::system_clock::now();
  {
    EstimateLSM<KeyCompType*> tree(std::unique_ptr<Env>(createDefaultEnv()), std::make_unique<FileName>(0, "/tmp/viscnts/"),
                                   std::make_unique<DefaultAllocator>(), SKeyCompFunc);
    int L = 3e8, Q = 1e4;
    std::vector<int> numbers(L);
    auto comp2 = +[](int x, int y) {
      uint8_t a[12], b[12];
      for (int j = 0; j < 12; j++) a[j] = x >> (j % 4) * 8 & 255;
      for (int j = 0; j < 12; j++) b[j] = y >> (j % 4) * 8 & 255;
      return SKeyCompFunc(SKey(a, 12), SKey(b, 12)) < 0;
    };
    for (int i = 0; i < L; i++) numbers[i] = i;
    // std::sort(numbers.begin(), numbers.end(), comp2);
    std::shuffle(numbers.begin(), numbers.end(), std::mt19937(std::random_device()()));
    srand(std::random_device()());
    for (int i = 0; i < L / 2; i++) {
      uint8_t a[12];
      for (int j = 0; j < 12; j++) a[j] = numbers[i] >> (j % 4) * 8 & 255;
      tree.append(SKey(a, 12), SValue(1, 1));
    }
    tree.all_flush();

    auto end = std::chrono::system_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    logger("flush used time: ", double(dur.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den);
    std::this_thread::sleep_for(std::chrono::milliseconds(5000));
    auto numbers2 = std::vector<int>(numbers.begin(), numbers.begin() + L / 2);
    std::sort(numbers2.begin(), numbers2.end(), comp2);
    {
      int Qs = 1000;
      for (int i = 0; i < Qs; i++) {
        uint8_t a[12], b[12];
        int id = abs(rand()) % (numbers2.size() - std::min(1000, L / 4));
        int x = numbers2[id];
        for (int j = 0; j < 12; j++) a[j] = x >> (j % 4) * 8 & 255;
        id += std::min(1000, L / 4);
        int y = numbers2[id];
        for (int j = 0; j < 12; j++) b[j] = y >> (j % 4) * 8 & 255;
        auto L = std::lower_bound(numbers2.begin(), numbers2.end(), x, comp2);
        auto R = std::upper_bound(numbers2.begin(), numbers2.end(), y, comp2);
        if (L > R) {
          i--;
          continue;
        }
        numbers2.erase(L, R);
        logger("[x,y]=", x, ",", y);
        tree.delete_range({SKey(a, 12), SKey(b, 12)});
      }
    }

    start = std::chrono::system_clock::now();
    for (int i = 0; i < Q; i++) {
      uint8_t a[12], b[12];
      int id = abs(rand()) % numbers.size();
      int x = numbers[id];
      for (int j = 0; j < 12; j++) a[j] = numbers[id] >> (j % 4) * 8 & 255;
      id = abs(rand()) % numbers.size();
      int y = numbers[id];
      for (int j = 0; j < 12; j++) b[j] = numbers[id] >> (j % 4) * 8 & 255;
      int ans = std::upper_bound(numbers2.begin(), numbers2.end(), x, comp2) - std::lower_bound(numbers2.begin(), numbers2.end(), y, comp2);
      ans = std::max(ans, 0);
      int output = tree.range_count({SKey(b, 12), SKey(a, 12)});
      // logger("[ans,output]=", ans, ",", output);
      DB_ASSERT(ans == output);
    }

    int QLEN = 1000;

    end = std::chrono::system_clock::now();
    dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    start = end;
    logger("range count used time: ", double(dur.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den);

    for (int i = 0; i < Q; i++) {
      uint8_t a[12];
      int id = abs(rand()) % numbers.size();
      for (int j = 0; j < 12; j++) a[j] = numbers[id] >> (j % 4) * 8 & 255;

      auto iter = std::unique_ptr<EstimateLSM<KeyCompType*>::SuperVersionIterator>(tree.seek(SKey(a, 12)));
      auto check_func = [](const uint8_t* a, int goal) {
        int x = 0, y = 0;
        for (int j = 0; j < 12; j++) {
          x |= a[j] << (j % 4) * 8;
          // if (j % 4 == 3) {
          //   // logger("[j,x,goal]=",j,",",x,",",goal);
          // }
          DB_ASSERT(j % 4 != 3 || x == goal);
          if (j % 4 == 3) {
            if (j > 3) {
              DB_ASSERT(y == x);
            }
            y = x;
            x = 0;
          }
        }
      };
      int x = a[0] | ((uint32_t)a[1] << 8) | ((uint32_t)a[2] << 16) | ((uint32_t)a[3] << 24);

      int cnt = 0;
      auto it = std::lower_bound(numbers2.begin(), numbers2.end(), x, comp2);
      while (true) {
        if (++cnt > QLEN) break;
        DB_ASSERT((it != numbers2.end()) == (iter->valid()));
        if (it == numbers2.end()) break;
        auto a = iter->read().first.data();
        // logger("[it]:", *it);
        check_func(a, *it);
        it++;
        iter->next();
      }
    }
    end = std::chrono::system_clock::now();
    dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    logger("random scan used time: ", double(dur.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den);
  }
  logger("test_random_scan(): OK");
}


int main() {
  // test_files();
  // test_unordered_buf();
  // test_lsm_store();
  // test_lsm_store_and_scan();
  test_random_scan_and_count();
  // test_lsm_decay();
  // test_splay();
  // test_delete_range();
}