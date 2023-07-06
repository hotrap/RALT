#include <bits/stdc++.h>

#include "zipf.h"
using namespace std;
const int keylen = 16;
const int valuelen = 16;
const int sstsz = 1e5;
const int kpartition = 1e4;
const double kratio = 0.1, kdecay = sstsz * 10, kprob = 0.5;
int l0_min = 2, l0_max = 20, cnt_decay = 0;
mt19937_64 genrd(time(0));
struct SST {
  vector<pair<string, double>> data;
  double decay_size;
  SST() { decay_size = 0; }
  SST(const SST& s) : data(s.data), decay_size(s.decay_size) {}
  SST(SST&& s) { (*this) = std::move(s); }
  SST& operator=(SST&& s) {
    data = (std::move(s.data));
    decay_size = s.decay_size;
    return (*this);
  }
  int size(int keylen) const { return data.size() * (8 + 4 + 4 + keylen); }
  bool exists(const string& key) {
    auto L = lower_bound(data.begin(), data.end(), std::make_pair(key, -1e9));
    if (L != data.end() && L->first == key) return 1;
    return 0;
  }
  SST operator+(const SST& s) const {
    SST ret;
    auto l = data.begin(), r = s.data.begin();
    while (l != data.end() || r != s.data.end()) {
      vector<pair<string, double>>::const_iterator res;
      if (l == data.end() || (r != s.data.end() && l->first >= r->first)) {
        res = r;
        r++;
      } else {
        res = l;
        l++;
      }
      if (ret.data.size() && ret.data.back().first == res->first)
        ret.data.back().second += res->second;
      else
        ret.data.push_back(*res);
    }
    return ret;
  }
  auto range() const { return make_pair(data[0].first, data.back().first); }
  void print_range() const {
    auto p = range();
    auto l = p.first, r = p.second;
    for (auto& a : l) cout << (int)(a & 255) << ",";
    cout << ";";
    for (auto& a : r) cout << (int)(a & 255) << ",";
    cout << "\n";
  }
  void append(string s) { data.emplace_back(s, 1); }
  void clear() {
    decay_size = 0;
    data.clear();
  }
  int operator<(const SST& s) const { return range().first < s.range().first; }
  bool check() {
    for (int i = 1; i < data.size(); ++i)
      if (data[i - 1].first >= data[i].first) return false;
    return true;
  }
  void sortk() {
    sort(data.begin(), data.end());
    calc();
  }
  vector<SST> part(int div) {
    int sz = data.size() / div + 1;
    vector<SST> ret;
    for (int i = 0; i < div; ++i) {
      SST s;
      if (i * sz >= data.size()) break;
      // printf("<%d ,%d>", i*sz, min<int>((i + 1) * sz, data.size())); fflush(stdout);
      s.data = vector<pair<string, double>>(data.begin() + i * sz, data.begin() + min<int>((i + 1) * sz, data.size()));
      s.calc();
      ret.emplace_back(std::move(s));
    }
    return ret;
  }
  void decay(double prob) {
    SST ret;
    uniform_real_distribution<> dis(0., 1.);
    for (auto& a : data) {
      if (a.second * prob >= 1) {
        ret.data.emplace_back(std::move(a.first), a.second * (1 - prob));
      } else {
        if (dis(genrd) <= a.second * prob) ret.data.emplace_back(std::move(a.first), 1);
      }
    }
    data = std::move(ret.data);
    calc();
  }
  void calc() {
    decay_size = 0;
    for (auto& a : data) decay_size += (keylen + valuelen) * std::min(a.second * kprob, 1.);
  }
};
template <typename T>
bool cross(T a, T b) {
  if (a.second < b.first || b.second < a.first) return 0;
  return 1;
}

struct Level {
  vector<SST> pars;
  int sz;
  double decay_size;
  Level() {
    sz = 0;
    decay_size = 0;
  }
  Level(vector<SST>&& _pars) : pars(std::move(_pars)) {
    sz = 0;
    decay_size = 0;
    for (auto& a : pars) {
      sz += a.size(keylen);
      decay_size += a.decay_size;
    }
  }
  Level(const Level& s) : pars(s.pars), sz(s.sz), decay_size(s.decay_size) {}
  Level(Level&& s) { (*this) = std::move(s); }
  Level& operator=(Level&& s) {
    pars = (std::move(s.pars));
    decay_size = s.decay_size;
    sz = s.sz;
    return (*this);
  }
  bool exists(const std::string key) {
    int l = 0, r = pars.size() - 1, ans = -1;
    while (l <= r) {
      int mid = (l + r) >> 1;
      if (pars[mid].range().first <= key)
        l = mid + 1, ans = mid;
      else
        r = mid - 1;
    }
    if (ans != -1 && pars[ans].exists(key)) return 1;
    return 0;
  }
  void clear() {
    sz = 0;
    decay_size = 0;
    pars.clear();
  }
  int size() { return sz; }
  int is_overlap(const SST& s) const {
    auto L = lower_bound(pars.begin(), pars.end(), s, [](const SST& a, const SST& b) { return a.range().second < b.range().second; });
    if (L != pars.end() && cross(s.range(), L->range())) return 1;
    if (L == pars.begin() && L == pars.end()) return 0;
    if (L == pars.end()) L--;
    for (int i = 0; i < 3 && L != pars.begin(); i++, L--)
      if (cross(s.range(), L->range())) return 1;
    return 0;
  }
  void append(int keylen, SST&& s) {
    sz += s.size(keylen);
    decay_size += s.decay_size;
    pars.push_back(std::move(s));
  }
  SST concat() {
    SST ret;
    for (auto& a : pars) {
      ret.data.insert(ret.data.end(), make_move_iterator(a.data.begin()), make_move_iterator(a.data.end()));
    }
    return ret;
  }
  void decay(double prob) {
    SST new_sst;
    for (auto& a : pars) {
      a.decay(prob);
      new_sst.data.insert(new_sst.data.end(), make_move_iterator(a.data.begin()), make_move_iterator(a.data.end()));
    }
    pars = new_sst.part(new_sst.size(keylen) / kpartition + 1);
    sz = 0, decay_size = 0;
    for (auto& a : pars) {
      sz += a.size(keylen);
      decay_size += a.decay_size;
    }
  }
};

struct SV {
  vector<Level> tree;
  SST buf;
  Level lastl;
  void clear() {
    buf.clear();
    lastl.clear();
    tree.clear();
  }
  long long merge(vector<int> v) {
    vector<Level> _sv;
    sort(v.begin(), v.end());
    long long cost = 0;
    vector<Level> _merge;
    for (int i = 0, j = 0; i < tree.size(); ++i) {
      while (j < v.size() && v[j] < i) j++;
      if (v[j] == i)
        _merge.emplace_back(std::move(tree[i]));
      else
        _sv.emplace_back(std::move(tree[i]));
    }
    for (int i = 0; i < _merge.size(); ++i)
      for (auto& sst : _merge[i].pars) {
        int flag = 0;
        for (int j = 0; j < _merge.size(); ++j)
          if (i != j) flag |= _merge[j].is_overlap(sst);
        if (flag) cost += sst.size(keylen);
      }
    SST s;
    for (int i = 0; i < _merge.size(); ++i) s = s + _merge[i].concat();
    // printf("<%d>", s.size(keylen) / kpartition + 1); fflush(stdout);
    _sv.emplace_back(s.part(s.size(keylen) / kpartition + 1));
    sort(_sv.begin(), _sv.end(), [](const Level& a, const Level& b) { return a.sz > b.sz; });
    tree = std::move(_sv);

    return cost;
  }
  void print_sizes() {
    printf("[");
    for (int i = 0; i < tree.size(); ++i) printf("%d, ", tree[i].size());
    puts("]");
  }
  int size() {
    int ans = 0;
    for (auto& a : tree) ans += a.size();
    ans += lastl.size();
    return ans;
  }
  double decay_size() {
    double ans = 0;
    for (auto& a : tree) ans += a.decay_size;
    ans += lastl.decay_size;
    return ans;
  }
  bool exists(const std::string& key) {
    for (auto& a : tree)
      if (a.exists(key)) return 1;
    return lastl.exists(key);
  }
} sv;
vector<string> keys;
mt19937 gen(time(0));
string genkey1() {
  uniform_int_distribution<> dis(0, keys.size() - 1);
  return keys[dis(gen)];
}
string genkey2(double alpha) {
  uniform_real_distribution<> dis(0, 1.);
  double z = dis(genrd);
  while (z == 0 || z == 1) z = dis(genrd);
  return keys[zipf(alpha, keys.size() - 1, z)];
}

long long write4(int keylen, string key) {
  sv.buf.append(key);
  long long cost = 0;
  if (sv.buf.size(keylen) >= sstsz) {
    SST s = std::move(sv.buf);
    s.sortk();
    sv.buf.clear();
    if (!sv.tree.size()) {
      Level l;
      cost += s.size(keylen);
      l.append(keylen, std::move(s));
      sv.tree.push_back(l);
      return cost;
    }
    Level l;
    auto bufcost = s.size(keylen);
    cost += s.size(keylen);
    l.append(keylen, std::move(s));
    sv.tree.push_back(l);
    // printf("[%.6lf]", sv.decay_size() );
    if (sv.lastl.size() < sv.size() * (1 - kratio) && sv.decay_size() >= kdecay) {
      puts("major compaction");
      vector<int> v;
      printf("<%d>", sv.tree.size());
      sv.tree.push_back(std::move(sv.lastl));
      for (int i = 0; i < sv.tree.size(); ++i) v.push_back(i);
      cost += sv.merge(v);
      sv.lastl = std::move(sv.tree[0]);
      printf("<%d, cost=%lld, %.6lf>", (int)sv.tree.size(), cost, sv.lastl.decay_size);
      sv.tree.clear();
      while (sv.lastl.decay_size >= kdecay) cost += sv.lastl.size(), sv.lastl.decay(kprob), cnt_decay++;
      printf("{%lld}", cost);
      return cost;
    }
    if (sv.tree.size() >= l0_min && sv.tree.size() < l0_max) {
      int sum = sv.tree.back().size(), where = -1;
      for (int i = sv.tree.size() - 2; i >= 0; --i) {
        if (sv.tree[i].size() <= kratio * sum) where = i;
        sum += sv.tree[i].size();
      }
      if (where != -1) {
        vector<int> v;
        for (int i = where; i < sv.tree.size(); ++i) v.push_back(i);
        cost += sv.merge(v);
        cost -= bufcost;
      }

    } else {
      double _kratio = kratio;
      while (sv.tree.size() >= l0_max) {
        // printf("FUCK"); fflush(stdout);
        //			int nw_sz=std::accumulate(v.begin()+4,v.end(),0);
        //			v.erase(v.begin()+4,v.end());
        //			v.insert(upper_bound(v.begin(),v.end(),nw_sz,[](int x,int y){return x>y;}),nw_sz);
        //			cost+=nw_sz;
        _kratio += 0.05;
        int sum = sv.tree.back().size(), where = -1;
        for (int i = sv.tree.size() - 2; i >= 0; --i) {
          if (sv.tree[i].size() <= _kratio * sum) where = i;
          sum += sv.tree[i].size();
        }
        if (where != -1) {
          vector<int> v;
          for (int i = where; i < sv.tree.size(); ++i) v.push_back(i);
          cost += sv.merge(v);
          cost -= bufcost;
        }
      }
    }
    return cost;
  }
  return 0;
}
void test1(int op) {
  while (l0_max <= 20) {
    long long cost = 0;
    for (int i = 0; i < op; ++i) {
      cost += write4(keylen, genkey1());
      //		if(!sv.check())printf("[WA]");
      // if (i % 10000 == 0) printf("<cost:%lld>\n", cost);
    }
    printf("l0_max: %d, cost: %lld, wa: %.4lf\n", l0_max, cost, cost / ((keylen + 16.) * op));
    l0_max++;
    sv.clear();
  }
  l0_max = 20;
}

struct TrueVisCnts {
  map<string, double> st;
  double decay_size;
  TrueVisCnts() { decay_size = 0; }
  void append(const std::string key) {
    decay_size -= (keylen + valuelen) * std::min(1., st[key] * kprob);
    st[key]++;
    decay_size += (keylen + valuelen) * std::min(1., st[key] * kprob);
    if (decay_size >= kdecay) decay();
  }

  void decay() {
    // puts("true viscnts decay");
    decay_size = 0;
    uniform_real_distribution<> dis(0., 1.);
    for (auto it = st.begin(); it != st.end(); it++) {
      if (it->second * kprob >= 1)
        it->second *= kprob, decay_size += (keylen + valuelen) * std::min(1., it->second * kprob);
      else {
        auto nt = next(it);
        if (dis(genrd) < it->second * kprob) {
          it->second = 1;
          decay_size += (keylen + valuelen) * std::min(1., it->second * kprob);
        } else {
          st.erase(it);
          it = nt;
        }
      }
    }
  }
  void clear() {
    st.clear();
    decay_size = 0;
  }
};
int main() {
  int op = 1e7, n = 1e7;
  for (int i = 0; i < n; ++i) {
    keys.push_back(string((char*)(&i), sizeof(int)));
  }
  std::shuffle(keys.begin(), keys.end(), genrd);
  // sort(keys.begin(), keys.end());
  // test1(op);
  TrueVisCnts tvc;
  vector<double> vec = {0.9};
  vector<vector<double>> res;
  for (auto _prob : vec) {
    long long cost = 0;
    vector<double> vv;
    puts("new prob");
    tvc.clear();
    cnt_decay = 0;
    for (int i = 0; i < op; ++i) {
      // auto key = genkey1();
      auto key = genkey2(_prob);
      cost += write4(keylen, key);
      tvc.append(key);
      //		if(!sv.check())printf("[WA]");
      if (i % 100000 == 0) {
        long long hot = 0, sz = 0;
        for (auto& [key, counts] : tvc.st)
          if (counts >= 2) hot += sv.exists(key), sz++;
        // printf("<cost:%lld, %lld/%lld, %.6lf, cnt_decay: %d>\n", cost, hot, sz, hot / (double)sz, cnt_decay);
        vv.push_back(hot / (double)sz);
      }
    }
    printf("l0_max: %d, cost: %lld, wa: %.4lf, cnt_decay: %d\n", l0_max, cost, cost / ((keylen + 16.) * op), cnt_decay);
    sv.clear();
    res.push_back(std::move(vv));
  }
  for (auto& r : res) {
    puts("---");
    for (auto& v : r) printf("%.6lf, ", v);
  }
}