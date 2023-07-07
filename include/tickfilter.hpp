#ifndef VISCNTS_TICKFILTER_H__
#define VISCNTS_TICKFILTER_H__

#include "key.hpp"

namespace viscnts_lsm {

template<typename Value>
class TickFilter {};

template<>
class TickFilter<SValue> {
  public:
    TickFilter(double tick_threshold) : tick_threshold_(tick_threshold) {}
    bool check(SValue v) const { return true; }
    double get_tick_threshold() const { return tick_threshold_; }

  private:
    double tick_threshold_;
};

template<>
class TickFilter<TickValue> {
  public:
    TickFilter(double tick_threshold) : tick_threshold_(tick_threshold) {}
    bool check(TickValue v) const { return v.tick > tick_threshold_; }
    double get_tick_threshold() const { return tick_threshold_; }

  private:
    double tick_threshold_;
};

}



#endif