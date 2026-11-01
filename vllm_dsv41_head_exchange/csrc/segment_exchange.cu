// SPDX-License-Identifier: Apache-2.0
// Adapted from antgroup/sglang commit 380e2d5; see NOTICE.md and LICENSE.
// Modified: standalone C ABI, PyTorch symmetric buffers, fixed 8-head segments,
// timeout diagnostics, and argument validation. Only world=2 is GPU-tested.
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

__device__ __forceinline__ unsigned load_acquire(const unsigned* p) {
  unsigned value;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
               : "=r"(value)
               : "l"(p)
               : "memory");
  return value;
}

__device__ __forceinline__ void arrive_relaxed(unsigned* p) {
  asm volatile("red.relaxed.sys.global.add.u32 [%0], 1;" ::"l"(p) : "memory");
}

__device__ __forceinline__ void arrive_release(unsigned* p) {
  asm volatile("red.release.sys.global.add.u32 [%0], 1;" ::"l"(p) : "memory");
}

__device__ __forceinline__ unsigned load_relaxed(const unsigned* p) {
  unsigned value;
  asm volatile("ld.relaxed.sys.global.u32 %0, [%1];"
               : "=r"(value)
               : "l"(p)
               : "memory");
  return value;
}

__device__ __forceinline__ unsigned long long now_ns() {
  unsigned long long value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}

// One 128-byte cache-line-isolated arrival counter and epoch per CTA. This
// follows the two-phase arrival-window protocol in antgroup/sglang's
// distributed/communicator.cuh at commit 380e2d5. Threads [0, world) each
// signal one peer; only the local-rank thread waits on the summed arrivals.
__device__ __forceinline__ void peer_phase(const uint64_t* peers,
                                           uint64_t flag_offset, int world,
                                           int rank, int phase,
                                           unsigned epoch) {
  if (threadIdx.x < world) {
    auto* peer = reinterpret_cast<unsigned*>(peers[threadIdx.x] + flag_offset +
                                             blockIdx.x * 128);
    if (phase == 0)
      arrive_relaxed(peer);
    else
      arrive_release(peer);
    if (threadIdx.x == rank) {
      const auto deadline = now_ns() + 10000000000ull;
      const unsigned current = epoch + phase * world;
      unsigned polls = 0;
      while ((phase == 0 ? load_relaxed(peer) : load_acquire(peer)) - current <
             world) {
        if ((++polls & 1023) == 0 && now_ns() > deadline) {
          printf(
              "segment_exchange barrier timeout rank=%d block=%u phase=%d "
              "epoch=%u\n",
              rank, blockIdx.x, phase, epoch);
          asm volatile("trap;");
        }
      }
    }
  }
}

template <int World>
__global__ void exchange_kernel(const uint8_t* src, const uint64_t* peers,
                                uint64_t flag_offset, int64_t src_peer_stride,
                                int64_t src_row_stride, int64_t dst_rank_stride,
                                int64_t dst_row_stride, int rows,
                                int valid_rows, int rank) {
  constexpr int world = World;
  auto* local =
      reinterpret_cast<unsigned*>(peers[rank] + flag_offset + blockIdx.x * 128);
  const unsigned epoch =
      threadIdx.x == rank ? atomicAdd(local + 1, 2 * world) : 0;
  peer_phase(peers, flag_offset, world, rank, 0, epoch);
  __syncthreads();
  constexpr int unroll = 4;
  constexpr int vectors_per_tile = 32 * unroll;
  constexpr int tiles_per_segment = 8192 / sizeof(uint4) / vectors_per_tile;
  const int warp = blockIdx.x + gridDim.x * (threadIdx.x / 32);
  const int warps = gridDim.x * (blockDim.x / 32);
  const int lane = threadIdx.x % 32;
  const int total_tiles = rows * world * tiles_per_segment;
  for (int tile = warp; tile < total_tiles; tile += warps) {
    const int peer = (rank + 1 + tile % world) % world;
    const int rest = tile / world;
    const int row = rest / tiles_per_segment;
    const int base = (rest % tiles_per_segment) * vectors_per_tile + lane;
    const bool valid = peer * rows + row < valid_rows;
    const auto* input = reinterpret_cast<const uint4*>(
        src + peer * src_peer_stride + row * src_row_stride);
    auto* output = reinterpret_cast<uint4*>(
        peers[peer] + rank * dst_rank_stride + row * dst_row_stride);
    uint4 values[unroll];
#pragma unroll
    for (int j = 0; j < unroll; ++j)
      values[j] = valid ? input[base + j * 32] : make_uint4(0, 0, 0, 0);
#pragma unroll
    for (int j = 0; j < unroll; ++j) output[base + j * 32] = values[j];
  }
  // CTA synchronization orders every payload writer before the peer-signaling
  // threads. Their system-release operations publish those writes transitively.
  __syncthreads();
  peer_phase(peers, flag_offset, world, rank, 1, epoch);
}

extern "C" int launch_segment_exchange(
    const void* src, const void* peer_array, uint64_t flag_offset,
    int64_t src_peer_stride, int64_t src_row_stride, int64_t dst_rank_stride,
    int64_t dst_row_stride, int rows, int valid_rows, int segment_bytes,
    int world, int rank, int blocks, int threads, void* stream) {
  if ((world != 2 && world != 4 && world != 8) || rank < 0 || rank >= world ||
      blocks < 1 || blocks > 32 || threads < 32 || threads > 1024 ||
      threads % 32 || segment_bytes != 8192 || rows < 1 || rows > (1 << 24) ||
      valid_rows < 0 || valid_rows > rows * world || !src || !peer_array ||
      reinterpret_cast<uintptr_t>(src) % 16 || flag_offset % 128 ||
      src_peer_stride < 0 || src_peer_stride % 16 || src_row_stride < 0 ||
      src_row_stride % 16 || dst_rank_stride < 0 || dst_rank_stride % 16 ||
      dst_row_stride < 0 || dst_row_stride % 16)
    return -1;
  if (flag_offset < (world - 1) * static_cast<uint64_t>(dst_rank_stride) +
                        (rows - 1) * static_cast<uint64_t>(dst_row_stride) +
                        segment_bytes)
    return -1;
#define LAUNCH(W)                                                           \
  exchange_kernel<W>                                                        \
      <<<blocks, threads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(     \
          static_cast<const uint8_t*>(src),                                 \
          static_cast<const uint64_t*>(peer_array), flag_offset,            \
          src_peer_stride, src_row_stride, dst_rank_stride, dst_row_stride, \
          rows, valid_rows, rank)
  if (world == 2) {
    LAUNCH(2);
  } else if (world == 4) {
    LAUNCH(4);
  } else {
    LAUNCH(8);
  }
#undef LAUNCH
  return static_cast<int>(cudaGetLastError());
}
