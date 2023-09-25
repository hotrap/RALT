#ifndef VISCNTS_COMPACTION_H__
#define VISCNTS_COMPACTION_H__

#include "sst.hpp"
#include "fileenv.hpp"
#include "writebatch.hpp"
#include "tickfilter.hpp"
#include <random>

namespace viscnts_lsm {

constexpr size_t kSSTable = 1ull << 24;
template <typename KeyCompT, typename ValueT, typename IndexDataT>
class Compaction {
  // builder_ is used to build one file
  // files_ is used to get global file name
  // env_ is used to get global environment
  // flag_ means whether lst_value_ is valid
  // lst_value_ is the last key value pair appended to builder_
  // vec_newfiles_ stores the information of new files.
  // - the ranges of files
  // - the size of files
  // - the filename of files
  // - the file_id of files
  // file_id is used if we have cache.
  SSTBuilder<ValueT, IndexDataT> builder_;
  FileName* files_;
  Env* env_;
  bool flag_;
  std::pair<IndSKey, ValueT> lst_value_;
  std::mt19937_64 rndgen_;
  KeyCompT comp_;
  double real_size_;
  double lst_real_size_;
  double decay_prob_;  // = 0.5 on default

  void _begin_new_file() {
    builder_.reset();
    auto [filename, id] = files_->next_pair();
    builder_.new_file(std::make_unique<WriteBatch>(std::unique_ptr<AppendFile>(env_->openAppFile(filename))));
    vec_newfiles_.emplace_back();
    vec_newfiles_.back().filename = filename;
    vec_newfiles_.back().file_id = id;
  }

  void _end_new_file() {
    builder_.make_index();
    builder_.finish();
    vec_newfiles_.back().size = builder_.size();
    vec_newfiles_.back().range = std::move(builder_.range());
    vec_newfiles_.back().hot_size = real_size_ - lst_real_size_;
    lst_real_size_ = real_size_;
  }

 public:
  struct NewFileData {
    std::string filename;
    size_t file_id;
    size_t size;
    std::pair<IndSKey, IndSKey> range;
    double hot_size;
  };
  Compaction(double current_tick, FileName* files, Env* env, KeyCompT comp) 
    : current_tick_(current_tick), files_(files), env_(env), flag_(false), rndgen_(std::random_device()()), comp_(comp) {
    decay_prob_ = 0.5;
  }

  template <typename TIter, typename FilterFunc, typename OtherFunc>
  auto flush(TIter& left, FilterFunc&& filter_func, OtherFunc&& other_func) {
    vec_newfiles_.clear();
    // null iterator
    if (!left.valid()) {
      return std::make_pair(vec_newfiles_, 0.0);
    }
    _begin_new_file();
    flag_ = false;
    real_size_ = 0;
    lst_real_size_ = 0;
    int CNT = 0;
    // read first kv.
    {
      auto L = left.read();
      lst_value_ = {L.first, L.second};
      left.next();
    }
    while (left.valid()) {
      CNT++;
      auto L = left.read();
      if (comp_(lst_value_.first.ref(), L.first) == 0) {
        lst_value_.second.merge(L.second, current_tick_);
      } else {
        // only store those filter returning true.
        if (filter_func(lst_value_)) {
          // It maybe the last key.      
          builder_.set_lstkey(lst_value_.first);
          other_func(lst_value_.first, lst_value_.second);
          real_size_ += _calc_hot_size(lst_value_);
          builder_.append(lst_value_);
          _divide_file(L.first.size() + L.second.get_hot_size());
        }
        lst_value_ = {L.first, L.second};
      }
      left.next();
    }
    // store the last kv.
    {
      if (filter_func(lst_value_)) {
        // It is the last key.
        builder_.set_lstkey(lst_value_.first);
        other_func(lst_value_.first, lst_value_.second);
        builder_.append(lst_value_);
        real_size_ += _calc_hot_size(lst_value_); 
      } 
    }
    _end_new_file();
    return std::make_pair(vec_newfiles_, real_size_);
  }

  template <typename TIter, typename FuncT>
  auto decay1(TIter& iters, FuncT&& func) {
    return flush(iters, [this](auto& kv){
      return kv.second.decay(decay_prob_, rndgen_);
    }, std::forward<FuncT>(func));
  }

  template<typename TIter, typename FuncT>
  auto flush(TIter& left, FuncT&& func) {
    return flush(left, [](auto&) { return true; }, std::forward<FuncT>(func));
  }

  template<typename TIter>
  auto flush(TIter& left) {
    return flush(left, [](auto&) { return true; }, [](auto&&...){});
  }

  template<typename TIter, typename FuncT>
  auto flush_with_filter(TIter& left, TickFilter<ValueT> tick_filter, FuncT&& func) {
    return flush(left, [tick_filter](auto& kv) { return tick_filter.check(kv.second); }, std::forward<FuncT>(func));
  }

  size_t get_write_bytes() const {
    return builder_.get_write_bytes();
  }


 private:
  template <typename T>
  double _calc_hot_size(const std::pair<T, ValueT>& kv) {
    return kv.first.len() + kv.second.get_hot_size();
  }
  void _divide_file(size_t size) {
    if (builder_.kv_size() + size > kSSTable) {
      builder_.set_lstkey(lst_value_.first);
      _end_new_file();
      _begin_new_file();
    }
  }
  std::vector<NewFileData> vec_newfiles_;
  double current_tick_;
};


}







#endif