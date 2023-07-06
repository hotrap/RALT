#ifndef __FILE_CACHE_H__
#define __FILE_CACHE_H__
#include <string>
#include <memory>
#include <filesystem>

#include "common.hpp"
namespace viscnts_lsm {

class SeqFile {
 public:
  virtual ssize_t read(size_t n, uint8_t* data) = 0;
  virtual ssize_t seek(size_t offset) = 0;
  virtual int get_fd() const = 0;
  virtual ~SeqFile() = default;
};

class RandomAccessFile {
 public:
  virtual ssize_t read(int fd, size_t offset, size_t n, uint8_t* data) const = 0;
  virtual ssize_t remove() const = 0;
  virtual int get_fd() const = 0;
  virtual void release_fd(int fd) const = 0;
  virtual SeqFile* get_seqfile() const = 0;
  virtual ~RandomAccessFile() = default;
};


class AppendFile {
 public:
  virtual ssize_t write(const Slice& data) = 0;
  virtual ssize_t sync() = 0;
  virtual int get_fd() = 0;
  virtual ~AppendFile() = default;
};

class Env {
  public:
    virtual ~Env() = default;
    virtual RandomAccessFile* openRAFile(std::string filename) = 0;
    virtual SeqFile* openSeqFile(std::string filename) = 0;
    virtual AppendFile* openAppFile(std::string filename) = 0;
};


Env* createDefaultEnv();

// generate global filename
class FileName {
  std::atomic<size_t> file_ts_;
  std::filesystem::path path_;

 public:
  explicit FileName(size_t ts, std::string path) : file_ts_(ts), path_(path) { std::filesystem::create_directories(path); }
  std::string gen_filename(size_t id) { return "lainlainlainlain" + std::to_string(id) + ".data"; }
  std::string gen_path() { return (path_ / gen_filename(file_ts_)).string(); }
  std::string next() { return (path_ / gen_filename(++file_ts_)).string(); }
  std::pair<std::string, size_t> next_pair() {
    auto id = ++file_ts_;
    return {(path_ / gen_filename(id)).string(), id};
  }
};

// information of file blocks in SST files
struct FileBlockHandle {
  // The structure of a file block
  // [handles]
  // [kv pairs...]
  uint32_t offset;
  uint32_t size;    // The size of SST doesn't exceed 4GB
  uint32_t counts;  // number of keys in the fileblock
  FileBlockHandle() { offset = size = counts = 0; }
  explicit FileBlockHandle(uint32_t offset, uint32_t size, uint32_t counts) : offset(offset), size(size), counts(counts) {}
};

}  // namespace viscnts_lsm

#endif
