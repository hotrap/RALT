#include "fileenv.hpp"
#include "logging.hpp"

#include <fcntl.h>
#include <sys/mman.h>

#include <atomic>

namespace viscnts_lsm {

static std::atomic<size_t> global_read_bytes{0}, global_write_bytes{0};

size_t GetWriteBytes() {
  return global_write_bytes.load();
}

size_t GetReadBytes() {
  return global_read_bytes.load();
}

class PosixSeqFile : public SeqFile {
 public:
  PosixSeqFile(int fd) : fd_(fd) {}
  
  ~PosixSeqFile() { ::close(fd_); global_read_bytes.fetch_add(read_bytes_, std::memory_order_relaxed); }

  ssize_t read(size_t n, uint8_t* data) override {
    ::ssize_t read_size;
    read_bytes_ += n;
    while (true) {
      read_size = ::read(fd_, data, n);
      if (read_size < 0) {
        // Read error because some signals occurred during the system call, e.g. alarm()
        if (errno == EINTR) continue;
        return errno;
      }
      break;
    }
    return read_size;
  }
  ssize_t seek(size_t offset) override {
    if (auto ret = ::lseek(fd_, offset, SEEK_SET); ret < 0) return errno;
    return 0;
  }

  int get_fd() const override {
    return fd_;
  }

 private:
  int fd_;
  size_t read_bytes_{0};
};

class PosixRandomAccessFile : public RandomAccessFile {
 public:
  PosixRandomAccessFile(std::string fname) : fname_(fname) {}
  
  ~PosixRandomAccessFile() {}

  ssize_t read(int fd, size_t offset, size_t n, uint8_t* data) const override {
    auto rd_n = ::pread(fd, data, n, offset);
    global_read_bytes.fetch_add(n, std::memory_order_relaxed);
    if (rd_n < 0) {
      logger("Error: ", errno);
      exit(-1);
    }
    return rd_n;
  }

  ssize_t remove() const override {
    return ::remove(fname_.c_str());
  }

  int get_fd() const override {
    int fd = ::open(fname_.c_str(), O_RDONLY);
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED|POSIX_FADV_RANDOM);
    // logger(fd, ", ", fname_);
    if (fd < 0) {
      logger("Error: ", errno);
      exit(-1);
    }
    return fd;
  }
  void release_fd(int fd) const override {
    // logger(fd, ", ", fname_);
    ::close(fd);
  }

  SeqFile* get_seqfile() const override {
    auto fd = ::open(fname_.c_str(), O_RDONLY);
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED|POSIX_FADV_SEQUENTIAL);
    if (fd < 0) {
      logger("Error: ", errno);
      exit(-1);
    }
    return new PosixSeqFile(fd);
  }

 private:
  std::string fname_;
};

// PosixAppendFile is only used when flush SST (compaction result) and MemTable.
class PosixAppendFile : public AppendFile {
  public:
    PosixAppendFile(int fd) : fd_(fd) {}
    ~PosixAppendFile() {
      ::close(fd_);
      global_write_bytes.fetch_add(write_bytes_, std::memory_order_relaxed);
    }
    ssize_t write(const Slice& data) override {
      auto ret = ::write(fd_, data.data(), data.len());
      write_bytes_ += data.len();
      if(ret < 0) {
        logger("Error: ", errno);
        exit(-1);
      }
      return 0;
    }
    ssize_t sync() override {
      return ::fsync(fd_);
    }
    int get_fd() override {
      return fd_;
    }
  private:
    int fd_;
    size_t write_bytes_{0};
};

class DefaultEnv : public Env {
 public:
  DefaultEnv() {}
  RandomAccessFile* openRAFile(std::string filename) override {
    auto fd = ::open(filename.c_str(), O_RDONLY);
    if (fd < 0) return nullptr;
    auto result = new PosixRandomAccessFile(filename);
    ::close(fd);
    return result;
  }
  SeqFile* openSeqFile(std::string filename) override {
    auto fd = ::open(filename.c_str(), O_RDONLY);
    if (fd < 0) return nullptr;
    return new PosixSeqFile(fd);
  }
  AppendFile* openAppFile(std::string filename) override {
    auto fd = ::open(filename.c_str(), O_APPEND | O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return nullptr;
    return new PosixAppendFile(fd);
  }
};

Env* createDefaultEnv() { 
  static DefaultEnv env;
  return &env;
}

}  // namespace viscnts_lsm
