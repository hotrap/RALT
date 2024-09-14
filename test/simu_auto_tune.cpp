#include <bits/stdc++.h>
using namespace std;
struct record {
  int key;
  double counter{0};
  int tag{0};
  int lst_time{0};
  record() {}
  record(int key, double c, int t, int lst_time):key(key), counter(c), tag(t), lst_time(lst_time){}
};

const int N=1.1e6, R=N*0.07, L=N*0.005, D=0.05*R, opN=1.1e6;
const double delta_c=26, c_max=100;
mt19937_64 rgen(0x114514);

int cnt=0;

void evict(std::map<int, record>& mp, int& mp_limit) {
  int num_stable=0;
  bool minus1 = cnt >= R;
  if (cnt >= R) {
    cnt = 0;
  }
  int stable_n = 0;
  std::vector<record> unstables;
  for(auto it = mp.begin(); it != mp.end();it++) {
    if (minus1) {
      it->second.counter -= 10;
    }
    if ((!it->second.tag || it->second.counter <= 0)) {
      unstables.push_back(it->second);
    } else {
      stable_n ++;
    }
  
  }
  std::sort(unstables.begin(), unstables.end(), [&](auto r1, auto r2){ return r1.lst_time<r2.lst_time; });
  mp_limit=(stable_n+D)*1.1;
  for(auto r : unstables) {
    if (mp.size() <= mp_limit * 0.9) {
      break;
    }
    mp.erase(r.key);
  }
  // std::cout << mp.size() << std::endl;
}

void insert(int key, std::map<int, record>& mp, int& mp_limit) {
  if (++cnt%int(R*0.1) == 0 || mp.size() >= mp_limit) {
    evict(mp, mp_limit);
  }
  if (mp.find(key) != mp.end()) {
    auto& r = mp[key];
    r.tag = 1;
    r.counter = std::min(r.counter + delta_c, c_max);
    r.lst_time = cnt;
  } else {
    mp[key] = record(key, delta_c, 0, cnt);
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

double calc_prob_stable(int N, int D, std::function<int()> gen_input) {
  std::map<int, int> cnt;
  std::set<int> ans;
  std::set<std::pair<int, int>> st;
  for (int i = 0; i < N; i++) {
    // for (int j = 0; j < D; j++) {
    //   cnt[gen_input()]++;
    // }
    // std::vector<int> v;
    // for (auto a : cnt) if (a.second <= 1) {
    //   v.push_back(a.first);
    // }
    // for (auto a : v) {
    //   cnt.erase(a);
    // }
    // int x = gen_input();
    // if (cnt.count(x)) {
    //   st.erase({cnt[x], x});
    //   if (cnt[x] > i - D) ans.insert(x);
    //   cnt[x] = i;
    //   st.insert({cnt[x], x});
    // } else {
    //   cnt[x] = i;
    //   st.insert({cnt[x], x});
    // }
    // while (st.size() && st.begin()->first <= i - D) cnt.erase(st.begin()->second), st.erase(st.begin());
  }
  return ans.size();
}

double markov_simple(double p1, double p, double q, int N) {
  std::vector<double> pi(std::max(N + 1, 100)), pi2(std::max(N + 1, 100));
  pi[0] = 1;
  for (int i = 0; i <= 5000; i++) {
    for (int j = 0; j <= N; j++) {
      if (j == 0) {
        pi2[0] = q * pi[1] + (1 - p1) * pi[0];
      }
      else if (j == 1) {
        pi2[1] = q * pi[2] + (1 - p - q) * pi[1];
      }
      else if (j == 2) {
        pi2[2] = q * (N >= 3 ? pi[3] : 0) + (N >= 3 ? 1 - p - q : 1 - q) * pi[2] + p1 * pi[0] + p * pi[1];
      }
      else if (j < N) {
        pi2[j] = q * pi[j + 1] + (1 - p - q) * pi[j] + p * pi[j - 1];
      }
      else {
        pi2[N] = (1 - q) * pi[N] + p * pi[N - 1];
      }
    }
    for (int j = 0; j <= N; j++) {
      pi[j] = pi2[j];
      // std::cout << pi[j] << " ";
    }
    // std::cout << pi[0] << std::endl;
  }
  return pi[0];
}

double markov_simple_2(double p1, double p, double q, int delta, int minus, int N) {
  std::vector<double> pi(std::max(N + 1, 100)), pi2(std::max(N + 1, 100));
  pi[0] = 1;
  for (int i = 0; i <= 30*1e6/R; i++) {
    for (int j = 0; j <= N; j++) {
      pi2[j] += pi[j];
      if (j > 0) {
        pi2[std::max(0, std::min(N, j / delta * delta + delta) - minus)] += pi[j] * p;
        pi2[j] -= pi[j] * p;
      } else {
        pi2[std::max(0, std::min(N, j + 2 * delta) - minus)] += pi[j] * p1;
        pi2[j] -= pi[j] * p1;
      }
      if (j > 0) {
        pi2[std::max(0, j - minus)] += pi[j] * q;
        pi2[j] -= pi[j] * q;
      }
    }
    for (int j = 0; j <= N; j++) {
      pi[j] = pi2[j];
      pi2[j] = 0;
    }
  }
  return pi[0];
}


double markov_complex(double p1, double p, double p2, double q, int delta, int minus, int N, int& round) {
  std::vector<double> pi(std::max(N + 1, 100)), pi2(std::max(N + 1, 100));
  pi[0] = 1;
  for (int i = 0; i <= 30*1e6/R; i++) {
    for (int j = 0; j <= N; j++) {
      pi2[j] += pi[j];
      if (j > 0) {
        pi2[std::max(0, std::min(N, j + delta) - minus)] += pi[j] * p;
        pi2[j] -= pi[j] * p;
        pi2[std::max(0, std::min(N, j + delta * 2) - minus)] += pi[j] * p2;
        pi2[j] -= pi[j] * p2;
      } else {
        pi2[std::max(0, std::min(N, j + 2 * delta) - minus)] += pi[j] * p1;
        pi2[j] -= pi[j] * p1;
      }
      if (j > 0) {
        pi2[std::max(0, j - minus)] += pi[j] * q;
        pi2[j] -= pi[j] * q;
      }
    }
    // if (pi[0] - pi2[0] <= 1e-3) {
    //   round = i;
    //   break;
    // }
    for (int j = 0; j <= N; j++) {
      pi[j] = pi2[j];
      pi2[j] = 0;
    }
    // std::cout << pi[0] << std::endl;
  }
  // for (int j = 0; j <= N; j++) {
  //   std::cout << pi[j] << " ";
  // }
  return pi[0];
}

double theory_calc_markov(int cmax, double p1, double p, double q) {
  return (q / p1) * pow(q / (p + q), cmax - 1) / ((q / p1) * pow(q / (p + q), cmax - 1) + (1 - pow(q / (p + q), cmax)) / (1 - q / (p + q)));
}

double theory_calc_markov_simple(double pk, int N) {
  return exp(2*D * pk) / (exp(pk * (2*D+N*R))-exp(N*R*pk)+1);
}


double theory_calc_markov_simple_2(int cmax, int deltac, double p1, double p, double q) {
  int r = cmax % deltac;
  int k = cmax / deltac;
  double v1 = 1 - (1 - p - q) - p * ((1 - pow(q / (p + q), deltac)) / (1 - q / (p + q)) - 1);
  double v2 = pow((p + q) / q, deltac - 1);
  double v3 = (1 - pow(q / (p + q), delta_c)) / (1 - q / (p + q));
  double sum = 1;
  double x1 = p1 / q * v2, x2 = 0, x3;
  sum += x1 * v3; // x1
  x2 = x1 * v1 * (1 / q) * v2; // x2
  sum += x2 * v3;
  x3 = (v1 * x2 - p1) * (1 / q) * v2; // x3
  sum += x3 * v3;
  for (int i = 4; i <= k + 1; i++) {
    x1 = x2, x2 = x3;
    x3 = (v1 * x2 - p * x1) * (1 / q) * pow((p + q) / q, i == k + 1 ? r - 1 : deltac - 1);
    sum += x3 * (i == k + 1 ? (1 - pow(q / (p + q), r)) / (1 - q / (p + q)) : v3);
  }
  std::cout << "["<<sum<<"]";
  
  return 1 / sum;
}

double get_real(int opN, double pk) {
  std::map<int, record> mp;
  int mp_limit=1e4;
  mp.clear();
  mp_limit=L;
  for (int i=0;i<opN;i++) {
    insert(genhotspot(N, 1/pk/N, 1), mp, mp_limit);
  }
  size_t ans=0;
  for(auto s : mp) {
    ans += s.second.tag == 1;
  }
  return ans;
}

void print_proper_delta_c() {
  for (int pp = 1; pp <= 100; pp++) {
    int round;
    double pk = 1 / (0.01 * pp * N);
    double p1 = (1 - exp(-R * pk)) * (1 - exp(-2 * D * pk));
    double p = pk * R * exp(-R * pk);
    double p2 = 1 - exp(-R * pk) - pk * R * exp(-R * pk);
    double q = exp(-R * pk);
    
    std::cout << std::endl << pp << ": ";
    for (int cmax = 10; cmax <= 500; cmax += 10) {
      auto maxp = markov_complex(p1, p, p2, q, cmax, 10, cmax, round);
      
      // for (int i = 1; i<=cmax;i++) {
      //   if (markov_complex(p1, p, p2, q, i, 10, cmax, round) * 0.95 <= maxp) {
          // std::cout << i << "/" << cmax << ", ";
          // std::cout << std::setprecision(3) << i * (1 - q) << ", ";
          // std::cout << i << "/" << c_max << ": " << maxp << ", " << (1 - q) * i << std::endl;
          // break;
        // }
      // }
      std::cout << markov_complex(p1, p, p2, q, 20 / (1 - q), 10, cmax, round) / maxp << " ";
    }
      
  }
  
}

int main(int argc, char** argv) {
  print_proper_delta_c();
  return 0;

  // std::vector<double> pvec={5./7};
  // for (auto p:pvec) {
  //   mp.clear();
  //   mp_limit=L;
  //   for (int i=0;i<opN*100;i++) {
  //     insert(genhotspot(N, (p*R)/N, 1));
  //   }
  //   std::cout << p << " " << mp_limit - D << std::endl;  
  // }

  std::cout << get_real(R, 1 / (0.05 * N))
            << "[" << R * 2 * D * pow(1 / (0.05 * N), 2) * (0.05 * N) << "]"
            << "[" << (1 - exp(-R * 1 / (0.05 * N))) * (1 - exp(-2 * D * 1 / (0.05 * N))) * (0.05 * N) << "]"
            << "/" << N * 0.05 << std::endl;
  int cmax = 50;
  /*
  for (int i = 1; i<=50;i++) {
    int round;
    double pk = 1 / (0.01*i*N);
    double p1 = (1 - exp(-R * pk)) * (1 - exp(-2 * D * pk));
    double p = pk * R * exp(-R * pk);
    double p2 = 1 - exp(-R * pk) - pk * R * exp(-R * pk);
    double q = exp(-R * pk);
    cout << pk << ", " << p + p2 << ", " << q << ", " << q / p1 << ", " << q / (p + q + p2);
    double ans1 = (1 - markov_complex(p1, p, p2, q, delta_c, 10, c_max, round)) * (1 / pk);
    double ans2 = get_real(N * 2, pk);
      std::cout<<i<<":"<< ans1 <<", "<< D
      << ", " << ans2 << " " 
      << std::endl;
      // std::cout << (1-theory_calc_markov_simple(1 / (0.01 * i * N), 4))*(0.01*i*N) <<", "<< D<< " ";
  }
  */
  for (int i = 1; i<=c_max;i++) {
    int round;
    double pk = 1 / (0.2*N);
    double p1 = (1 - exp(-R * pk)) * (1 - exp(-2 * D * pk));
    double p = pk * R * exp(-R * pk);
    double p2 = 1 - exp(-R * pk) - pk * R * exp(-R * pk);
    double q = exp(-R * pk);
    double ans1 = (1 - markov_complex(p1, p, p2, q, i, 10, c_max, round)) * (1 / pk);
    double ans2 = (1 - markov_simple_2(p1, p + p2, q, i, 10, c_max)) * (1 / pk);
    // double ans2 = get_real(N * 2, pk);
      std::cout<<i<<":"<< ans1 <<", " << ans2 << "|";
      std::cout << (1 - theory_calc_markov_simple(pk, c_max/10-1)) * (1 / pk) 
                << ", " << (1 - theory_calc_markov(c_max/10-1, p1, p + p2, q)) * (1 / pk) 
                << ", " << (1 - theory_calc_markov_simple_2(c_max/10-1,max(i/10,1), p1, p + p2, q)) * (1 / pk) 
                << std::endl;
  }
}