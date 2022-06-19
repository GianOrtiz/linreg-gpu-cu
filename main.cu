#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <queue>
#include <string>
#include <istream>
#include <ostream>
#include <fstream>
#include <iostream>
#include <thread>
#include <array>
#include <mutex>
#include <optional>

#define cudaCheck(ans)                     \
  {                                        \
    gpu_assert((ans), __FILE__, __LINE__); \
  }
inline void gpu_assert(cudaError_t code, const char *file, int line, bool abort = true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "GPUassert: %s %s:%d\n", cudaGetErrorString(code), file, line);
    if (abort)
      exit(code);
  }
}

template <class T, size_t TElemCount>
class circular_buffer
{
public:
  // circular_buffer() = default;
  explicit circular_buffer()
  {
    buf_ = {};
  }

  void put(T item)
  {
    // std::lock_guard<std::recursive_mutex> lock(mutex_);

    buf_[head_] = item;

    if (full_)
    {
      tail_ = (tail_ + 1) % TElemCount;
    }

    head_ = (head_ + 1) % TElemCount;

    full_ = head_ == tail_;
  }

  // __device__ void gpu_put(T *item)
  // {
  //   std::lock_guard<std::recursive_mutex> lock(mutex_);

  //   cudaCheck(cudaMemCpy(&buf_[head_], item, sizeof(T), cudaMemcpyHostToDevice));

  //   if (full_)
  //   {
  //     tail_ = (tail_ + 1) % TElemCount;
  //   }

  //   head_ = (head_ + 1) % TElemCount;

  //   full_ = head_ == tail_;
  // }

  T get()
  {
    // std::lock_guard<std::recursive_mutex> lock(mutex_);

    // if (empty())
    // {
    //   return std::nullopt;
    // }

    // Read data and advance the tail (we now have a free space)
    auto val = buf_[tail_];
    full_ = false;
    tail_ = (tail_ + 1) % TElemCount;

    return val;
  }

  void reset()
  {
    // std::lock_guard<std::recursive_mutex> lock(mutex_);
    head_ = tail_;
    full_ = false;
  }

  bool empty()
  {
    // Can have a race condition in a multi-threaded application
    // std::lock_guard<std::recursive_mutex> lock(mutex_);
    // if head and tail are equal, we are empty
    return (!full_ && (head_ == tail_));
  }

  bool full()
  {
    // If tail is ahead the head by 1, we are full
    return full_;
  }

  size_t capacity()
  {
    return TElemCount;
  }

  size_t size()
  {
    // A lock is needed in size ot prevent a race condition, because head_, tail_, and full_
    // can be updated between executing lines within this function in a multi-threaded
    // application
    // std::lock_guard<std::recursive_mutex> lock(mutex_);

    size_t size = TElemCount;

    if (!full_)
    {
      if (head_ >= tail_)
      {
        size = head_ - tail_;
      }
      else
      {
        size = TElemCount + head_ - tail_;
      }
    }

    return size;
  }

private:
  // mutable std::recursive_mutex mutex_;
  mutable std::array<T, TElemCount> buf_;
  mutable size_t head_ = 0;
  mutable size_t tail_ = 0;
  mutable bool full_ = 0;
};

const int WINDOW = 100;

const uint BUFFER_SIZE = 10000;
class Block
{
  uint index;

public:
  Block() = default;
  std::array<float, BUFFER_SIZE> data;
  Block(std::array<float, BUFFER_SIZE> data, uint index) : data(data), index(index){};
};

class Reader
{
  static const size_t BUFFER_COUNT = 100;

public:
  bool done = false;
  uint read_until = 0;
  circular_buffer<Block, Reader::BUFFER_COUNT> value_buffer;
  explicit Reader(){};
  // explicit Reader() = default;
  float *read(std::string filename)
  {
    std::ifstream in_file(filename, std::ifstream::binary);
    if (!in_file)
    {
      std::cerr << "Failed opening file" << std::endl;
      exit(1);
    }

    // Stop eating new lines in binary mode!!!
    in_file.unsetf(std::ios::skipws);

    while (!in_file.eof())
    {
      while (value_buffer.full())
      {
      }
      std::array<float, BUFFER_SIZE> read_buffer;
      in_file.read((char *)read_buffer.data(), BUFFER_SIZE * sizeof(float));
      value_buffer.put(Block(read_buffer, read_until));
      read_until += BUFFER_SIZE;
    }
    done = true;
    in_file.close();
  }
};

class DataScheduler
{
  static const size_t BUFFER_COUNT = 100;
  static const size_t OUT_BUFFER_COUNT = 100;
  Reader *reader;
  uint processed_until = 0;
  // circular_buffer<Block, DataScheduler::BUFFER_COUNT> *device_buffer;

public:
  bool done = false;
  // must write to buffer in the given order(ascending block index)
  circular_buffer<Block, DataScheduler::OUT_BUFFER_COUNT> out_buffer;
  explicit DataScheduler(Reader *reader) : reader(reader){};

  void loop()
  {
    while (!reader->done)
    {
      while (reader->value_buffer.empty())
      {
      }

      auto block = reader->value_buffer.get();
      // device_buffer->gpu_put(&block);
    }
    done = true;
  }
};

class Writer
{
  static const size_t BUFFER_COUNT = 100;

public:
  bool done = false;
  DataScheduler *scheduler;
  Writer(DataScheduler *scheduler) : scheduler(scheduler){};
  float *write(std::string filename)
  {
    std::ofstream out_file(filename, std::ofstream::binary);
    if (!out_file)
    {
      std::cerr << "Failed opening file" << std::endl;
      exit(1);
    }

    while (!scheduler->done)
    {
      while (scheduler->out_buffer.empty())
        ;
      out_file.write((char *)scheduler->out_buffer.get().data.data(), BUFFER_SIZE * sizeof(float));
    }
    done = true;
    out_file.close();
  }
};

int main()
{
  std::cout << "Initializing Reader..." << std::endl;
  Reader reader;
  std::cout << "Initializing Scheduler..." << std::endl;
  DataScheduler scheduler(&reader);
  std::cout << "Initializing Writer..." << std::endl;
  Writer writer(&scheduler);

  std::thread reader_thread(&Reader::read, &reader, "in.bin");
  std::thread scheduler_thread(&DataScheduler::loop, &scheduler);
  std::thread writer_thread(&Writer::write, &writer, "out.bin");

  reader_thread.join();
  scheduler_thread.join();
  writer_thread.join();
}
