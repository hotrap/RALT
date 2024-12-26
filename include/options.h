#ifndef RALT_OPTIONS_H_
#define RALT_OPTIONS_H_

#include <cstddef>

namespace ralt {

struct Options {
  size_t bloom_bits = 14;
  double exp_smoothing_factor = 0.999;
  // Increment ticks after accessing (tick_period_multiplier *
  // hot set size limit) of data
  double tick_period_multiplier = 0.001;
};

} // namespace ralt

#endif // RALT_OPTIONS_H_
