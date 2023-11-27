#ifndef VISCNTS_LOGGING_H__
#define VISCNTS_LOGGING_H__

#include <mutex>
#include <iostream>
#include <fmt/core.h>
#include <fmt/format.h>
#include <chrono>
#include <ctime>
#include <iomanip>

#include <cassert>

template <typename... Args>
void __logger(Args&&... a) {
  auto now = std::chrono::system_clock::now();
  std::time_t now_c = std::chrono::system_clock::to_time_t(now);
  std::tm *parts = std::localtime(&now_c);
  static std::mutex logger_m_;
  std::unique_lock lck_(logger_m_);
  std::cerr << std::put_time(parts, "[%Y-%m-%d %H:%M:%S.") << std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000 << "]";
  (std::cerr << ... << a) << std::endl;
}

#define logger(...) __logger(__FILE__, "@",  __LINE__, ":", __VA_ARGS__)

template <typename... Args>
void logger_printf(const char* str, Args&&... a) {
  auto len = snprintf(nullptr, 0, str, a...);
  auto dest_str = new char[len + 1];
  snprintf(dest_str, len + 1, str, a...);
  logger(dest_str);
  delete[] dest_str;
}



#define __LOG_ERR 3
#define __LOG_WARNING 4
#define __LOG_NOTICE 5
#define __LOG_INFO 6
#define __LOG_DEBUG 7
#define DB_WARNING(...) \
  __LOG(__LOG_WARNING, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__);
#define DB_NOTICE(...) \
  __LOG(__LOG_NOTICE, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__);
#define DB_INFO(...) \
  __LOG(__LOG_INFO, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__);
#define DB_DEBUG(...) \
  __LOG(__LOG_DEBUG, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__);

#define DB_ERR(...)                                                  \
  do {                                                               \
    __LOG(__LOG_ERR, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__); \
    std::abort();                                                    \
  } while (0)
#define DEFAULT_LOG_FILE NULL

#ifndef MY_NDEBUG
#define DEFAULT_LOG_LEVEL __LOG_DEBUG
#define DB_ASSERT(assertion)                                   \
  ({                                                           \
    if (!(assertion)) {                                        \
      DB_ERR("Internal Error: Assertion failed: " #assertion); \
      std::abort();                                            \
    }                                                          \
  })
#else
#define DEFAULT_LOG_LEVEL __LOG_DEBUG
#define DB_ASSERT(assertion) \
  do {                       \
  } while (0)
#endif

template <typename... Args>
void __LOG(int level, const char *file, const char *func, int line,
    const std::string &format_string, Args &&...args) {
  static const char *__loglevel_str[] = {
      "[emerg]",
      "[alert]",
      "[crit]",
      "[err]",
      "[warn]",
      "[notice]",
      "[info]",
      "[debug]",
  };
  static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
  if (level > DEFAULT_LOG_LEVEL)
    return;
  pthread_mutex_lock(&lock);
  fmt::print("{}[{}@{}:{}]: ", __loglevel_str[level], func, file, line);
  fmt::print(fmt::runtime(format_string), args...);
  fmt::print("\n");
  fflush(stdout);
  pthread_mutex_unlock(&lock);
}

class StopWatch {
 public:
  StopWatch() { start_ = std::chrono::system_clock::now(); }
  double GetTimeInSeconds() const {
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end - start_;
    return diff.count();
  }
  size_t GetTimeInNanos() {
    auto end = std::chrono::system_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_).count();
  }
  void Reset() { start_ = std::chrono::system_clock::now(); }

 private:
  std::chrono::time_point<std::chrono::system_clock> start_;
};

#endif