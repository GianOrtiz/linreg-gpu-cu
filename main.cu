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
#include <ctime>
#include <chrono>

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

#define KERNEL_BLOCK_SIZE 16

template <class T, size_t TElemCount>
class circular_buffer
{
public:
  explicit circular_buffer()
  {
    buf_ = {};
  }

  void put(T item)
  {
    std::lock_guard<std::recursive_mutex> lock(mutex_);

    buf_[head_] = item;

    if (full_)
    {
      tail_ = (tail_ + 1) % TElemCount;
    }

    head_ = (head_ + 1) % TElemCount;

    full_ = head_ == tail_;
  }

  T get()
  {
    std::lock_guard<std::recursive_mutex> lock(mutex_);

    // Read data and advance the tail (we now have a free space)
    auto val = buf_[tail_];
    full_ = false;
    tail_ = (tail_ + 1) % TElemCount;

    return val;
  }

  void reset()
  {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    head_ = tail_;
    full_ = false;
  }

  bool empty()
  {
    // Can have a race condition in a multi-threaded application
    std::lock_guard<std::recursive_mutex> lock(mutex_);
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
    std::lock_guard<std::recursive_mutex> lock(mutex_);

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
  mutable std::recursive_mutex mutex_;
  mutable std::array<T, TElemCount> buf_;
  mutable size_t head_ = 0;
  mutable size_t tail_ = 0;
  mutable bool full_ = 0;
};

const int WINDOW = 100;

const uint BUFFER_SIZE = KERNEL_BLOCK_SIZE * 64;
const uint CHUNK_SIZE = BUFFER_SIZE + WINDOW;
class Chunk
{
public:
  uint index;
  Chunk() = default;
  float data[CHUNK_SIZE];
  Chunk(std::array<float, CHUNK_SIZE> data, uint index) : index(index)
  {
    std::copy(data.begin(), data.end(), this->data);
  };
};

__global__ void kernel_matrix_mult(Chunk *out, Chunk *chunk)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (chunk->index == 0 && index < WINDOW)
  {
    return;
  }

  float tmp_sum = 0;
  // each thread computes one element of the block sub-matrix
  for (int i = WINDOW; i > 0; i--)
  {
    tmp_sum += chunk->data[index - i + WINDOW];
  }
  out->data[index] = tmp_sum;
  out->index = chunk->index;
}

// reads a one-dimensional CSV file
class Reader
{
  static const size_t BUFFER_COUNT = 100;

public:
  bool done = false;
  uint read_until = 0;
  circular_buffer<Chunk *, Reader::BUFFER_COUNT> value_buffer;
  Chunk *previous_chunk;
  explicit Reader(){};

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

    auto start = std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now());
    int read_size = 0;

    std::string line{};
    // read the first line(column headers)
    std::getline(in_file, line);
    while (!in_file.eof())
    {
      while (value_buffer.full())
      {
      }
      std::array<float, BUFFER_SIZE> read_buffer;
      size_t i = 0;
      while (std::getline(in_file, line) && i < BUFFER_SIZE)
        read_buffer[i++] = std::stof(line);
      std::array<float, CHUNK_SIZE> chunk_read_buffer;
      std::copy(read_buffer.begin(), read_buffer.end(), &chunk_read_buffer.at(WINDOW));
      if (previous_chunk != nullptr)
      {
        std::copy(std::end(previous_chunk->data) - WINDOW, std::end(previous_chunk->data), chunk_read_buffer.begin());
      }

      Chunk *chunk = new Chunk(chunk_read_buffer, read_until);
      value_buffer.put(chunk);
      read_until += BUFFER_SIZE;
      previous_chunk = chunk;
      read_size += BUFFER_SIZE * sizeof(float);
    }
    auto end = std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now());
    auto time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    // read_size in bytes, time_diff in milliseconds
    float mean = ((float)read_size / (float)time_diff);
    std::printf("READER:\nTempo: %dms\nLido: %d bytes\nMédia: %f bytes/ms\n\n", time_diff, read_size, mean);
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

public:
  bool done = false;
  // must write to buffer in the given order(ascending block index)
  circular_buffer<Chunk, DataScheduler::OUT_BUFFER_COUNT> out_buffer;
  explicit DataScheduler(Reader *reader) : reader(reader){};

  void loop()
  {
    int current_stream = 0;
    while (!reader->done || !reader->value_buffer.empty())
    {
      while (reader->value_buffer.empty())
      {
      }

      // We use streams to synchronize executions callbacks in the GPU.
      Chunk *chunk = reader->value_buffer.get();
      Chunk *device_chunk;
      Chunk *device_output_chunk;

      cudaCheck(cudaMalloc(&device_chunk, sizeof(Chunk)));
      cudaCheck(cudaMalloc(&device_output_chunk, sizeof(Chunk)));
      cudaCheck(cudaMemcpy(device_chunk, chunk, sizeof(Chunk), cudaMemcpyHostToDevice));
      dim3 threads_per_block(KERNEL_BLOCK_SIZE);
      dim3 blocks_per_grid(BUFFER_SIZE / KERNEL_BLOCK_SIZE);

      printf("using %d threads per block\n", threads_per_block.x * threads_per_block.y);
      printf("using %d blocks per grid\n", blocks_per_grid.x * blocks_per_grid.y);

      kernel_matrix_mult<<<blocks_per_grid, threads_per_block>>>(device_output_chunk, device_chunk);

      Chunk output_chunk;
      // Copy result from device memory to the host memory
      cudaCheck(cudaMemcpy(&output_chunk, device_output_chunk, sizeof(Chunk), cudaMemcpyDeviceToHost));

      this->out_buffer.put(output_chunk);
      free(chunk);

      // Free arrays in device memory
      cudaCheck(cudaFree(device_chunk));
      cudaCheck(cudaFree(device_output_chunk));
    }
    done = true;
  }
};

// writes to a one-dimensional CSV file
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

    auto start = std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now());
    int read_size = 0;
    // write column header
    out_file << "x" << std::endl;
    while (!scheduler->done || !scheduler->out_buffer.empty())
    {
      while (scheduler->out_buffer.empty())
      {
        if (scheduler->done)
        {
          goto loop_end;
        }
      }
      Chunk chunk = scheduler->out_buffer.get();
      for (size_t i = 0; i < BUFFER_SIZE; i++)
      {
        out_file << std::to_string(chunk.data[i]) << std::endl;
      }
      read_size += BUFFER_SIZE * sizeof(float);
    }
  loop_end:
    auto end = std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now());
    auto time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    // read_size in bytes, time_diff in milliseconds
    float mean = ((float)read_size / (float)time_diff);
    std::printf("WRITER:\nTempo: %dms\nLido: %d bytes\nMédia: %f bytes/ms\n\n", time_diff, read_size, mean);
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

  std::thread reader_thread(&Reader::read, &reader, "in.csv");
  std::thread scheduler_thread(&DataScheduler::loop, &scheduler);
  std::thread writer_thread(&Writer::write, &writer, "out.csv");

  reader_thread.join();
  scheduler_thread.join();
  writer_thread.join();
}
