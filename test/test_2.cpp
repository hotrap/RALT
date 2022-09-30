#include <assert.h>
#include <bits/stdc++.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "liburing.h"
using namespace std;
const char* kDir = "/mnt/sd/tmp/viscnts/ffffff";
const int kBlockSize = 32768;
uint8_t* global_ptr;
static int get_file_size(int fd, off_t* size) {
  struct stat st;

  if (fstat(fd, &st) < 0) return -1;
  if (S_ISREG(st.st_mode)) {
    *size = st.st_size;
    return 0;
  } else if (S_ISBLK(st.st_mode)) {
    unsigned long long bytes;

    if (ioctl(fd, BLKGETSIZE64, &bytes) != 0) return -1;

    *size = bytes;
    return 0;
  }

  return -1;
}
mt19937_64 genrd(time(0));
struct KV {
  int klen;
  int vlen;
  double value;
  uint8_t key[16];
  KV() {
    klen = 0;
    vlen = 0;
    value = 0;
    memset(key, 0, sizeof(key));
  }
};
void ranKV(uint8_t* ptr) {
  static long long x = 0;

  KV* kv = (KV*)ptr;
  kv->klen = 16;
  *(uint64_t*)(kv->key) = genrd();
  *(uint64_t*)(kv->key + sizeof(uint64_t)) = genrd();
  // KV* kv = (KV*)ptr;
  // kv->klen = 16;
  // int y = x++;
  // for (int i = 15; i >= 0; --i) kv->key[i] = y & 255, y >>= 8;
  // *(uint64_t*)(kv->key + sizeof(uint64_t)) = x++;
}

template <typename T>
int KVComp(const T& a, const T& b) {
  // assert(a.klen == 16 && b.klen == 16);
  if (a.klen == b.klen && a.klen == 16) return memcmp(a.key, b.key, a.klen);
  // else printf("[%d]", a.klen);
  return a.klen < b.klen ? -1 : 1;
}
int genfile(int id, size_t size) {
  auto ptr = global_ptr;
  std::vector<KV> v(size);
  for (size_t i = 0; i < size; i++) {
    ranKV((uint8_t*)(v.data() + i));
    assert(v[i].klen == 16);
  }
  stable_sort(v.begin(), v.end(), [&](const KV& x, const KV& y) { return KVComp(x, y); });
  ptr = (uint8_t*)v.data();
  auto fd = fopen((kDir + to_string(id)).c_str(), "wb");
  fwrite(ptr, size * sizeof(KV), 1, fd);
  fclose(fd);
  return ::open((kDir + to_string(id)).c_str(), O_RDONLY);
}

int openfile(int id) { return ::open((kDir + to_string(id)).c_str(), O_RDONLY); }

class Scan {
 public:
  virtual bool valid() = 0;
  virtual void next() = 0;
  virtual KV read() = 0;
  virtual ~Scan() = default;
};

template <typename Iterator>
class BitTree {
  // using DataType = std::pair<Iterator, KV>;
  Iterator** tree;
  Iterator* iters;
  //  maxKV;
  int size;
  int xsize;
  int _get_sz(int size) {
    int L = 1;
    while (size > L) L <<= 1;
    return L;
  }
  Iterator* _min(int x, int y) {
    if (tree[x] == nullptr || tree[y] == nullptr) return tree[x] ? tree[x] : tree[y];
    return KVComp(tree[x]->read(), tree[y]->read()) < 0 ? tree[x] : tree[y];
  }

  void _upd(int where) {
    where += size;
    for (where >>= 1; where; where >>= 1) {
      // if(tree[where] == )
      tree[where] = _min(where << 1, where << 1 | 1);
      // printf("[%d,%d,%d]",tree[where << 1]->second.klen, tree[where << 1 | 1]->second.klen,tree[where]->second.klen);
    }
  }

 public:
  BitTree(int _size) : size(_size + 1), tree(new Iterator*[_size + 2]), iters(new Iterator[_size + 2]), xsize(0) {
    for (int i = 1; i <= size; ++i) tree[i] = nullptr;
  }
  ~BitTree() {
    delete[] iters;
    delete[] tree;
  }
  void set(int where, Iterator&& iter) {
    iters[where] = std::move(iter);
    tree[++xsize] = &iters[where];
    for (int x = xsize; x > 1 && KVComp(tree[x >> 1]->read(), tree[x]->read()) > 0; x >>= 1) std::swap(tree[x >> 1], tree[x]);
    // tree[1] = &iters[where];
    // _upd(where);
    // printf("[%d,%d]", tree[1]->second.klen, iters[where].second.klen);
  }
  void next() {
    auto* iter = tree[1];
    iter->next();
    auto& seg_tree_ = tree;
    auto& size_ = size;
    if (!seg_tree_[1]->valid()) {
      seg_tree_[1] = nullptr;
      int x = 1;
      while ((x << 1) <= xsize) {
        auto r = _min(x << 1, x << 1 | 1);
        tree[x] = r;
        x = tree[x << 1] == r ? x << 1 : x << 1 | 1;
        tree[x] = nullptr;
      }
      return;
    }
    int x = 1;
    while ((x << 1) <= xsize) {
      auto r = _min(x << 1, x << 1 | 1);
      if (r == nullptr || KVComp(tree[x]->read(), r->read()) <= 0) break;
      if (r == tree[x << 1]) {
        swap(tree[x << 1], tree[x]);
        x = x << 1;
      } else {
        swap(tree[x << 1 | 1], tree[x]);
        x = x << 1 | 1;
      }
    }
    // int y = x;
    // for (x >>= 1; x; y = x, x >>= 1) {
    //   auto r = _min(x << 1, x << 1 | 1);
    //   ;
    //   seg_tree_[x] = r;
    //   // if(seg_tree_[x] != seg_tree_[y]) break;
    // }
    // int x = 1;
    // while(x < size) {
    //   auto r = _min(x << 1, x << 1 | 1);
    //   if(seg_tree_[x] == r)
    // }
    // _upd(iter - iters);
  }
  KV read() { return tree[1]->read(); }
  bool valid() { return size >= 1 && tree[1] != nullptr && tree[1]->valid(); }
};

class FirstScan : public Scan {
  struct Iterator : public Scan {
    int fd;
    uint8_t* data;
    int64_t size, ptr, rdsize;
    Iterator() { data = nullptr; }
    Iterator(int fd_) : fd(fd_) {
      get_file_size(fd, &size);
      data = new uint8_t[kBlockSize];
      rdsize = ::read(fd, data, kBlockSize);
      ptr = 0;
      // printf("<%d>", size);
    }
    Iterator& operator=(Iterator&& iter) {
      fd = iter.fd;
      data = iter.data;
      size = iter.size;
      ptr = iter.ptr;
      rdsize = iter.rdsize;
      iter.data = nullptr;
      return (*this);
    }
    ~Iterator() { delete[] data; }
    bool valid() override { return size > 0; }
    void next() override {
      ptr += sizeof(KV);
      if (ptr + sizeof(KV) > rdsize) {
        size -= rdsize, ptr = 0;
        // rdsize = min<int64_t>(kBlockSize, size);
        rdsize = ::read(fd, data, kBlockSize);
      }
    }
    KV read() override {
      // printf("<%d>", *(uint64_t*)(((KV*)(data + ptr))->key + sizeof(uint64_t)));
      return *(KV*)(data + ptr);
    }
  };
  BitTree<Iterator> bits;

 public:
  FirstScan(vector<int> fds) : bits(fds.size()) {
    for (uint32_t i = 0; i < fds.size(); ++i) bits.set(i, Iterator(fds[i]));
  }
  void next() override { bits.next(); }
  KV read() override { return bits.read(); }
  bool valid() override { return bits.valid(); }
};

// const int kQueueSize = 64;
class SecondScan : public Scan {
  struct io_uring ring;
  struct file_info {
    int id;
    int fd;
    off_t offset;
    struct iovec iov;
  };
  bool* complete;
  file_info* fi;
  struct Iterator : public Scan {
    int fd, id;
    uint8_t *data, *data0;
    int64_t size, ptr, rdsize, offset;
    SecondScan* master;
    Iterator() { data = data0 = nullptr; }
    Iterator(int fd_, int id_, SecondScan* master_) : fd(fd_), id(id_), master(master_) {
      get_file_size(fd, &size);
      data = new uint8_t[kBlockSize];
      data0 = new uint8_t[kBlockSize];
      offset = 0;

      rdsize = ::read(fd, data, kBlockSize);
      master->_prefetch(fd, id, data0, kBlockSize, kBlockSize);
      ptr = 0;
    }
    Iterator& operator=(Iterator&& iter) {
      fd = iter.fd;
      id = iter.id;
      offset = iter.offset;
      master = iter.master;
      data = iter.data;
      data0 = iter.data0;
      size = iter.size;
      ptr = iter.ptr;
      rdsize = iter.rdsize;
      iter.data = nullptr;
      iter.data0 = nullptr;
      return (*this);
    }
    ~Iterator() {
      delete[] data;
      delete[] data0;
    }
    bool valid() override { return size > 0; }
    void next() override {
      ptr += sizeof(KV);
      if (ptr + sizeof(KV) > rdsize) {
        size -= rdsize, ptr = 0, offset += rdsize;
        if (!size) return;
        // printf("<%d,%d,%d>",id,size,rdsize);fflush(stdout);
        rdsize = min(size, master->_get(id));
        // printf("<(new)%d>",rdsize);fflush(stdout);
        swap(data, data0);
        if (rdsize != size) master->_prefetch(fd, id, data0, offset + kBlockSize, kBlockSize);
      }
    }
    KV read() override { return *(KV*)(data + ptr); }
  };
  BitTree<Iterator> bits;
  void _prefetch(int fd, int id, uint8_t* data, int offset, int size) {
    struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
    fi[id].id = id;
    fi[id].iov.iov_base = data;
    fi[id].iov.iov_len = size;
    fi[id].offset = offset;
    fi[id].fd = fd;
    complete[id] = false;
    io_uring_prep_readv(sqe, fd, &fi[id].iov, 1, offset);
    io_uring_sqe_set_data(sqe, &fi[id]);
    io_uring_submit(&ring);
  }
  void _wait_cqe() {
    struct io_uring_cqe* cqe;
    int ret = io_uring_wait_cqe(&ring, &cqe);
    if (ret < 0) {
      perror("io_uring_wait_cqe");
      exit(1);
      return;
    }
    struct file_info* fi = (file_info*)io_uring_cqe_get_data(cqe);
    if (cqe->res < 0, 0) {
      if (cqe->res == -EAGAIN) {
        _prefetch(fi->fd, fi->id, (uint8_t*)fi->iov.iov_base, fi->offset, fi->iov.iov_len);
        io_uring_cqe_seen(&ring, cqe);
      } else {
        printf("cqe error: %s\n", strerror(cqe->res));
      }
      fprintf(stderr, "Async readv failed.\n");
      exit(1);
      return;
    }
    complete[fi->id] = true;
    io_uring_cqe_seen(&ring, cqe);
  }
  int64_t _get(int id) {
    while (!complete[id]) _wait_cqe();
    return fi[id].iov.iov_len;
  }

 public:
  SecondScan(vector<int> fds) : bits(fds.size()) {
    complete = new bool[fds.size()];
    fi = new file_info[fds.size()];
    io_uring_queue_init(fds.size(), &ring, 0);
    memset(complete, 0, fds.size());
    for (uint32_t i = 0; i < fds.size(); ++i) bits.set(i, Iterator(fds[i], i, this));
  }
  ~SecondScan() {
    delete[] complete;
    delete[] fi;
  }
  void next() override { bits.next(); }
  KV read() override { return bits.read(); }
  bool valid() override { return bits.valid(); }
};

template <typename T>
void mytest(vector<int> v, int L, int BEGIN = 0) {
  auto start = chrono::system_clock::now();
  T fs(v);
  for (int i = BEGIN; i < v.size() * L + BEGIN; ++i) {
    if (!fs.valid()) puts("WA");
    auto p = fs.read();
    int x = 0;
    // for (int i = 0; i < 16; ++i) x = (x << 8) | p.key[i];
    // printf("[%d,%d]",x,i);fflush(stdout);
    // assert(x == i);
    // printf("[FUCK,%lld]",*(uint64_t*)(p.key + sizeof(uint64_t)));fflush(stdout);
    // assert(*(uint64_t*)(p.key + sizeof(uint64_t)) == i);
    fs.next();
  }
  if (fs.valid()) puts("WA");
  auto end = chrono::system_clock::now();
  auto dur = chrono::duration_cast<chrono::microseconds>(end - start);
  cout << double(dur.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den << std::endl;
}

void clear() {
  sync();

  std::ofstream ofs("/proc/sys/vm/drop_caches");
  ofs << "3" << std::endl;
}

int main() {
  vector<int> v;

  unsigned long long M = 2 * 32 * (1 << 20);
  int cnt = 0;
  global_ptr = new uint8_t[M * 32ull];
  memset(global_ptr, 0, M * 32ull);

  for (int X = 1; X <= 128; X <<= 1) {
    unsigned long long L = M / X;
    for (auto& a : v) ::close(a);
    v.clear();
    for (int i = 0; i < X; ++i) v.push_back(genfile(i, L));
    printf("gen complete for %d, ", X);
    fflush(stdout);
    assert(sizeof(KV) == 32);
    clear();
    sleep(2);

    for (int i = 0; i < 1; ++i) {
      mytest<FirstScan>(v, L, M * cnt);
      for (auto a : v) lseek(a, 0, SEEK_SET);
      clear();
    }

    sleep(2);
    for (int i = 0; i < 1; ++i) {
      mytest<SecondScan>(v, L, M * cnt);
      for (auto a : v) lseek(a, 0, SEEK_SET);
      clear();
    }
    cnt++;
  }

  delete[] global_ptr;

  // for(auto a : v) lseek(a, 0, SEEK_SET);
  // mytest<SecondScan>(v, L);

  // printf("FUCK2");fflush(stdout);
}