#include "file.hpp"

#include <fcntl.h>
#include <sys/mman.h>

#include <atomic>

namespace viscnts_lsm {

const static size_t MmapLimits = 1000;
const static size_t FileHandleLimits = 1000;

class Limiter {
 public:
  Limiter(uint32_t max_ac) : max_acquires_(max_ac) {}
  Limiter(const Limiter&) = delete;
  Limiter operator=(const Limiter&) = delete;
  bool acquire() {
    auto old = max_acquires_.fetch_sub(1, std::memory_order_relaxed);
    if (old > 0) return true;
    max_acquires_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  void release() { max_acquires_.fetch_add(1, std::memory_order_relaxed); }

 private:
  std::atomic<uint32_t> max_acquires_;
};

// If the program is IO bound, then we consider to use io_uring, because viscnts is stored on CD.

class PosixSeqFile : public SeqFile {
 public:
  PosixSeqFile(int fd) : fd_(fd) {}
  ssize_t read(size_t n, uint8_t* data, Slice& result) override {
    while (true) {
      ::ssize_t read_size = ::read(fd_, data, n);
      if (read_size < 0) {
        // Read error because some signals occurred during the system call, e.g. alarm()
        if (errno == EINTR) continue;
        return errno;
      }
      result = Slice(data, read_size);
      break;
    }
  }
  ssize_t seek(size_t offset) override {
    if (auto ret = ::lseek(fd_, offset, SEEK_CUR); ret < 0) return errno;
    return 0;
  }

 private:
  int fd_;
};

class PosixRandomAccessFile : public RandomAccessFile {
 public:
  PosixRandomAccessFile(int fd, std::string fname, Limiter* fd_limit) : fd_(fd), use_fd_(fd_limit->acquire()), fname_(fname), fd_limit_(fd_limit) {}
  ~PosixRandomAccessFile() {
    if (use_fd_) fd_limit_->release();
  }
  ssize_t read(size_t offset, size_t n, uint8_t* data, Slice& result) override {
    int xfd;
    if (!use_fd_) {
      xfd = ::open(fname_.c_str(), O_RDONLY);
    } else
      xfd = fd_;
    auto rd_n = ::pread(xfd, data, n, offset);
    if (rd_n < 0) return errno;
    result = Slice(data, rd_n < 0 ? 0 : rd_n);
    if (!use_fd_) ::close(xfd);
    return rd_n;
  }

 private:
  int fd_;
  bool use_fd_;
  std::string fname_;
  Limiter* fd_limit_;
};

class MmapRandomAccessFile : public RandomAccessFile {
 public:
  MmapRandomAccessFile(void* base, size_t length, Limiter* fd_limit) : base_(base), legnth_(length) fd_limit_(fd_limit) {}
  ~MmapRandomAccessFile() {
    ::munmap(base_, length_);
    fd_limit_->release();
  }
  ssize_t read(size_t offset, size_t n, uint8_t* data, Slice& result) override {
    if (offset + n > length_) {
      result = Slice(data, 0);
      return -EINVAL;
    }
    result = Slice(base_ + offset, n);
    return 0;
  }

 private:
  void* base_;
  size_t length_;
  Limiter* fd_limit_;
};

// PosixAppendFile is only used when flush SST (compaction result) and MemTable.
class PosixAppendFile : public AppendFile {
  public:
    PosixAppendFile(int fd) : fd_(fd) {}
    ~PosixAppendFile() {
      ::close(fd_);
    }
    ssize_t write(const Slice& data) override {
      auto ret = ::write(fd_, data.data(), data.len());
      if(ret < 0) return errno;
      return 0;
    }
    ssize_t sync() override {
      return ::sync(fd_);
    }
  private:
    int fd_;
};

class DefaultEnv : public Env {
 public:
  DefaultEnv() : _mmap_limit_(MmapLimits), _fd_limit_(FileHandleLimits) {}
  ssize_t openRAFile(std::string filename, RandomAccessFile*& result) override {
    auto fd = ::open(filename.c_str(), O_RDONLY);
    if (fd < 0) return errno;
    if (!_mmap_limit_.acquire()) {
      result = new PosixRandomAccessFile(fd, filename, &_fd_limit_);
      return 0;
    }
    struct ::stat file_stat;
    if (::stat(filename.c_str(), &file_stat) != 0) {
      _mmap_limit_.release();
      return errno;
    }
    void* mmap_base = ::mmap(nullptr, file_stat.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (mmap_base == MAP_FAILED) return errno;
    result = new MmapRandomAccessFile(mmap_base, file_stat.st_size, &_mmap_limit_);
    ::close(fd);
  }
  ssize_t openSeqFile(std::string filename, SeqFile*& result) override {
    auto fd = ::open(filename.c_str(), O_RDONLY);
    if (fd < 0) return errno;
    result = new PosixSeqFile(fd);
    return 0;
  }
  ssize_t openAppFile(std::string filename, AppendFile*& result) override {
    auto fd = ::open(filename.c_str(), O_APPEND | O_WRONLY | O_CREAT, 0644);
    if (fd < 0) return errno;
    
  }

 private:
  Limiter _mmap_limit_;
  Limiter _fd_limit_;
};

Env* createDefaultEnv() { return new DefaultEnv(); }

}  // namespace viscnts_lsm
