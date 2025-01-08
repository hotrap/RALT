#ifndef VISCNTS_TICKFILTER_H__
#define VISCNTS_TICKFILTER_H__

#include "key.hpp"

namespace ralt {

template<typename Value>
class TickFilter {};

template<>
class TickFilter<SValue> {
  public:
    TickFilter(double tick_threshold) : tick_threshold_(tick_threshold) {}
    bool operator()(SValue v) const { return true; }
    double get_tick_threshold() const { return tick_threshold_; }

  private:
    double tick_threshold_;
};

template<>
class TickFilter<TickValue> {
  public:
    TickFilter(double tick_threshold) : tick_threshold_(tick_threshold) {}
    bool operator()(TickValue v) const {
      return v.get_score() > tick_threshold_;
    }
    double get_tick_threshold() const { return tick_threshold_; }

  private:
    double tick_threshold_;
};


template<>
class TickFilter<LRUTickValue> {
  public:
    TickFilter(double tick_threshold) : tick_threshold_(tick_threshold) {}
    bool operator()(LRUTickValue v) const {
      return v.get_score() > tick_threshold_;
    }
    double get_tick_threshold() const { return tick_threshold_; }

  private:
    double tick_threshold_;
};


template<>
class TickFilter<ExpTickValue> {
  public:
    TickFilter(double tick_threshold) : tick_threshold_(tick_threshold) {}
    bool operator()(const Options &options, ExpTickValue v) const {
      if (tick_threshold_ == 0) {
        return v.is_stable();
      } else {
        return v.get_score(options) > tick_threshold_;
      }
    }
    double get_tick_threshold() const { return tick_threshold_; }

  private:
    double tick_threshold_;
};


template<>
class TickFilter<ClockTickValue> {
  public:
    TickFilter(double tick_threshold) {}
    bool operator()(ClockTickValue v) const { return v.get_score() > 0; }
    double get_tick_threshold() const { return 0; }

  private:
};

}



#endif