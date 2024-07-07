#include <bits/stdc++.h>
using namespace std;
struct record {
  double counter{0};
  int tag{0};
  record() {}
  record(double c, int t):counter(c), tag(t){}
};

std::map<int, record> mp;
int mp_limit=1e4;

int L=1e4, R=7e4, D=1e4, N=1e6, opN=1e6;
double delta_c=1.3, c_max=2;
mt19937_64 rgen(0x114514);

int cnt=0;

void evict() {
  int num_stable=0;
  bool minus1 = cnt >= R;
  if (cnt >= R) {
    cnt = 0;
  }
  std::map<int, record> mp2;
  for(auto s : mp) {
    if (minus1) {
      s.second.counter -= 1;
    }
    if (s.second.tag && s.second.counter > 0) {
      mp2.insert(s);
    }  
  }
  mp=mp2;
  mp_limit=mp.size()+D;
  std::cout << mp.size() << std::endl;
}

void insert(int key) {
  if (++cnt>=R || mp.size() >= mp_limit) {
    evict();
  }
  if (mp.find(key) != mp.end()) {
    auto& r = mp[key];
    r.tag = 1;
    r.counter = std::min(r.counter + delta_c, c_max);
  } else {
    mp[key] = record(delta_c, 0);
  }
}

int genhotspot(int N, double prob_hotdata, double prob_hotop) {
  uniform_int_distribution<> hotdatadis(0, N*prob_hotdata-1);
  uniform_int_distribution<> colddatadis(N*prob_hotdata, N-1);
  uniform_real_distribution<> hotopdis(0, 1);
  if(hotopdis(rgen) <= prob_hotop) {
    return hotdatadis(rgen);
  } else {
    return colddatadis(rgen);
  }
}

int main(int argc, char** argv) {

  std::vector<double> pvec={5./7};
  for (auto p:pvec) {
    mp.clear();
    mp_limit=L;
    for (int i=0;i<opN*100;i++) {
      insert(genhotspot(N, (p*R)/N, 1));
    }
    std::cout << p << " " << mp_limit - D << std::endl;  
  }
  
}