#ifndef VISCNTS_LOGGING_H__
#define VISCNTS_LOGGING_H__

#include <mutex>
#include <iostream>

namespace viscnts_lsm {

template <typename... Args>
void logger(Args&&... a) {
  static std::mutex logger_m_;
  std::unique_lock lck_(logger_m_);
  (std::cerr << ... << a) << std::endl;
}

template <typename... Args>
void logger_printf(const char* str, Args&&... a) {
  auto len = snprintf(nullptr, 0, str, a...);
  auto dest_str = new char[len + 1];
  snprintf(dest_str, len + 1, str, a...);
  logger(dest_str);
  delete[] dest_str;
}

}

#endif