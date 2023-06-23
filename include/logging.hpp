#ifndef VISCNTS_LOGGING_H__
#define VISCNTS_LOGGING_H__

#include <mutex>
#include <iostream>
#include <fmt/core.h>
#include <fmt/format.h>

#include <cassert>

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

}

#endif