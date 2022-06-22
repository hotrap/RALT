#ifndef __FILE_CACHE_H__
#define __FILE_CACHE_H__
#include <string>

#include "common.hpp"
namespace viscnts_lsm {

class SeqFile {
 public:
  virtual ssize_t read(size_t n, uint8_t* data, Slice& result) = 0;
  virtual ssize_t seek(size_t offset) = 0;
  virtual ~SeqFile() = default;
};

class RandomAccessFile {
 public:
  virtual ssize_t read(size_t offset, size_t n, uint8_t* data, Slice& result) = 0;
  virtual ~RandomAccessFile() = default;
};


class AppendFile {
 public:
  virtual ssize_t write(const Slice& data) = 0;
  virtual ssize_t sync() = 0;
  virtual ~AppendFile() = default;
};

class Env {
  public:
    virtual ~Env() = default;
    virtual ssize_t openRAFile(std::string filename, RandomAccessFile*& result) = 0;
    virtual ssize_t openSeqFile(std::string filename, SeqFile*& result) = 0;
    virtual ssize_t openAppFile(std::string filename, AppendFile*& result) = 0;
};

Env* createDefaultEnv();

}  // namespace viscnts_lsm

#endif
