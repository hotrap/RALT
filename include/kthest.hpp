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
      logger(total_size);
      std::uniform_int_distribution<size_t> dis(0, total_size - 1);
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
      logger("scan1 size: ", scan1_size_sum_, ", scan1 points: ", scan1_point_num_);
      std::sort(point_data_.begin(), point_data_.begin() + scan1_point_num_, [](auto x, auto y) {
        return x.first < y.first;
      });
      scan1_point_num_ = std::unique(point_data_.begin(), point_data_.begin() + scan1_point_num_, [](auto x, auto y) {
        return x.first == y.first;
      }) - point_data_.begin();
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
      auto t = std::upper_bound(point_data_.begin(), point_data_.begin() + scan1_point_num_, std::make_pair(key, 0), [](auto x, auto y) {
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
      for(int i = 0; i < scan1_point_num_; i++) {
        sum += point_data_[i].second;
        ret = point_data_[i].first;
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
        logger(min_key_, ", ", first_interval_key_size_, ", ", sum);
        return L + (R - L) / double(first_interval_key_size_) * double(size_limit_ - sum);
      }
      for(int i = 0; i < scan1_point_num_; i++) {
        if (sum + point_data_[i].second > size_limit_) {
          Key L = point_data_[i].first;
          Key R = i == scan1_point_num_ - 1 ? max_key_ : point_data_[i + 1].first;
          logger(point_data_[i].first, ", ", point_data_[i].second, ", ", sum);
          logger(max_key_, "<-maxkey", L, ", ", R, ", ", double(size_limit_ - sum), ", ", double(point_data_[i].second));
          return L + (R - L) / double(point_data_[i].second) * double(size_limit_ - sum);
        }
        sum += point_data_[i].second;
      }
      return max_key_;
    }

    void sort() {
      logger("scan1 size: ", scan1_size_sum_);
      std::sort(point_data_.begin(), point_data_.begin() + scan1_point_num_, [](auto x, auto y) {
        return x.first < y.first;
      });
    }

    Key get_from_points(double size_limit) {
      double frac = size_limit / (double) scan1_size_sum_;
      logger(frac, size_limit, scan1_size_sum_);
      size_t i;
      if (scan1_size_sum_ == 0) {
        i = 0;
      } else {
        i = std::max<size_t>(
            0, std::min<size_t>(
                  scan1_point_num_ - 1,
                  int(scan1_point_num_ * frac)));
      }
      return point_data_[i].first;
    }

    
    Key get_from_lst_points(size_t now_size) {
      logger(scan1_point_num_, ", ", scan1_size_sum_, ", ", now_size);
      std::sort(point_data_.begin(), point_data_.begin() + scan1_point_num_, [](auto x, auto y) {
        return x.first < y.first;
      });
      // A = lst size
      // B = additional size
      // since the points are from the lst, we should use: (B - A) / A.
      double frac = 0.9;
      logger(frac);
      size_t i;
      if (scan1_size_sum_ == 0) {
        i = 0;
      } else {
        i = std::max<size_t>(
            0, std::min<size_t>(
                  scan1_point_num_ - 1,
                  int(scan1_point_num_ * frac)));
      }
      return point_data_[i].first;
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