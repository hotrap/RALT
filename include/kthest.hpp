#ifndef VISCNTS_KTHEST_H__
#define VISCNTS_KTHEST_H__

#include <memory>
#include <set>
#include <random>

namespace viscnts_lsm {

template<typename Key>
class KthEst {
  public:

    /* point_num should > 2 */
    KthEst(size_t point_num, size_t size_limit) : point_num_(point_num), size_limit_(size_limit) {

    }

    void pre_scan1(size_t total_size) {
      std::mt19937_64 rgen((std::random_device())());
      std::uniform_int_distribution<> dis(0, total_size - 1);
      point_size_.resize(point_num_);
      for(int i = 0; i < point_num_; i++) {
        point_size_[i] = dis(rgen);
      }
      std::sort(point_size_.begin(), point_size_.end());
      point_data_.resize(point_num_);
      scan1_size_sum_ = 0;
      scan1_point_num_ = 0;
      first_interval_key_size_ = 0;
      scan2_flag_ = false;
    }
    
    void scan1(Key key, size_t key_size) {
      scan1_size_sum_ += key_size;
      while(scan1_point_num_ < point_num_ && point_size_[scan1_point_num_] < scan1_size_sum_) {
        point_data_[scan1_point_num_] = {key, 0};
        scan1_point_num_ += 1;
      }
    }
    
    void pre_scan2() {
      std::sort(point_data_.begin(), point_data_.end(), [](auto x, auto y) {
        return x.first < y.first;
      });
      point_data_.erase(std::unique(point_data_.begin(), point_data_.end(), [](auto x, auto y) {
        return x.first == y.first;
      }), point_data_.end());
    }

    void scan2(Key key, size_t key_size) {
      if (!scan2_flag_) {
        scan2_flag_ = true;
        min_key_ = key;
        max_key_ = key;
      } else {
        min_key_ = std::min(min_key_, key);
        max_key_ = std::max(max_key_, key);
      }
      auto t = std::upper_bound(point_data_.begin(), point_data_.end(), std::make_pair(key, 0), [](auto x, auto y) {
        return x.first < y.first;
      });
      if(t != point_data_.begin()) {
        std::prev(t)->second += key_size;
      } else {
        first_interval_key_size_ += key_size;
      }
    }


    Key get_kth() {
      Key ret = max_key_;
      size_t sum = first_interval_key_size_;
      for(auto& a : point_data_) {
        sum += a.second;
        ret = a.first;
        // logger(a.first, ", ", a.second, ", ", sum);
        if (sum > size_limit_) {
          break;
        }
      }
      return ret;
    }

    Key get_interplot_kth() {
      size_t sum = first_interval_key_size_;
      if (sum > size_limit_) {
        Key L = min_key_;
        Key R = point_data_[0].first;
        return L + (R - L) / double(first_interval_key_size_) * double(size_limit_ - sum);
      }
      for(int i = 0; i < point_data_.size(); i++) {
        if (sum + point_data_[i].second > size_limit_) {
          Key L = point_data_[i].first;
          Key R = i == point_data_.size() - 1 ? max_key_ : point_data_[i + 1].first;
          return L + (R - L) / double(point_data_[i].second) * double(size_limit_ - sum);
        }
        sum += point_data_[i].second;
      }
      return max_key_;
    }

  private:
    std::vector<size_t> point_size_;
    std::vector<std::pair<Key, size_t>> point_data_;
    size_t point_num_;
    size_t size_limit_;
    size_t scan1_size_sum_{0}, scan1_point_num_{0}, first_interval_key_size_{0};
    Key min_key_, max_key_;
    bool scan2_flag_{false};

};

}


#endif