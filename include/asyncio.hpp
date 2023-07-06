#ifndef SAKURA_ASYNC_IO_H__
#define SAKURA_ASYNC_IO_H__

#include "liburing.h"
#include <string>
#include <algorithm>
#include <memory>

#include "logging.hpp"

namespace async_io {

class IOUring {
public:
  explicit IOUring(size_t queue_size) {
    if (auto s = io_uring_queue_init(queue_size, &ring_, 0); s < 0) {
      throw std::runtime_error("error initializing io_uring: " + std::to_string(s));
    }
  }

  IOUring(const IOUring &) = delete;
  IOUring &operator=(const IOUring &) = delete;

  ~IOUring() { io_uring_queue_exit(&ring_); }

  struct io_uring *get() {
    return &ring_;
  }

  void readv(int fd, void* mem, size_t offset, size_t size, void* info) {
    struct io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
    if (!sqe) {
      throw std::runtime_error("error cannot get sqe");
    }
    io_uring_prep_read(sqe, fd, mem, size, offset);
    io_uring_sqe_set_data(sqe, info);
  }

  void submit() {
    io_uring_submit(&ring_);
  }

  void* get_one() {
    struct io_uring_cqe* cqe;
    int ret = io_uring_wait_cqe(&ring_, &cqe);
    if (ret < 0) {
      throw std::runtime_error("error calling io_uring_wait_cqe: " + std::to_string(ret));
    }
    if (cqe->res < 0) {
      throw std::runtime_error("error io_uring_wait_cqe result code: " + std::to_string(cqe->res));
    }
    auto data = io_uring_cqe_get_data(cqe);
    io_uring_cqe_seen(&ring_, cqe);
    return data;
  }


private:
  struct io_uring ring_;
};

class AsyncIOQueue {
  public:
    AsyncIOQueue(size_t queue_size) : queue_size_(queue_size), iouring_(queue_size) {}

    void read(int fd, void* mem, size_t offset, size_t size, uint64_t info) {
      in_queue_size_ += 1;
      iouring_.readv(fd, mem, offset, size, reinterpret_cast<void*>(info));  
    }
    
    void read(int fd, void* mem, size_t offset, size_t size) {
      in_queue_size_ += 1;
      iouring_.readv(fd, mem, offset, size, nullptr);  
    }

    void submit() {
      iouring_.submit();
    }

    std::optional<uint64_t> get_one() {
      if (!in_queue_size_) {
        return {};
      }
      auto ret = iouring_.get_one();
      in_queue_size_ -= 1;
      return reinterpret_cast<uint64_t>(ret);
    }

    size_t size() {
      return in_queue_size_;
    }

  private:
    IOUring iouring_;
    size_t queue_size_;
    size_t in_queue_size_{0};
};

}


#endif