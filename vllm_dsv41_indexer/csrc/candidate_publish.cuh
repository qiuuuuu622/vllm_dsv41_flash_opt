// SPDX-License-Identifier: Apache-2.0
// Local H20 adaptation: source-side CUB radix sort and valid-position count.
// The source's nonnegative block IDs MUST be unique. No score quantization.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_reduce.cuh>
#include <climits>

namespace sglang {

__global__ void publish_candidate_blocks(const int32_t* blocks,
                                         const int32_t* lengths,
                                         int32_t* sorted, int32_t* counts) {
  constexpr int kThreads = 256, kItems = 8, kBlocks = kThreads * kItems;
  using Sort = cub::BlockRadixSort<int32_t, kThreads, kItems>;
  using Reduce = cub::BlockReduce<int32_t, kThreads>;
  __shared__ union {
    typename Sort::TempStorage sort;
    typename Reduce::TempStorage reduce;
  } temp;
  const int row = blockIdx.x;
  const int length = max(lengths[row], 0);
  const int last = length / 8 + (length % 8 != 0);
  int32_t keys[kItems];
#pragma unroll
  for (int i = 0; i < kItems; ++i) {
    const int b = blocks[row * kBlocks + threadIdx.x * kItems + i];
    keys[i] = b >= 0 && b < last ? b : INT_MAX;
  }
  // For <=65535 blocks, valid IDs fit below 0xffff and INT_MAX has low
  // bits 0xffff. Sorting those 16 bits gives exactly the same order as a
  // 32-bit sort, while halving radix passes. Larger ranges take the full sort.
  Sort(temp.sort).Sort(keys, 0, last <= 65535 ? 16 : 32);
  int sum = 0;
#pragma unroll
  for (int i = 0; i < kItems; ++i) {
    const int b = keys[i];
    sorted[row * kBlocks + threadIdx.x * kItems + i] = b;
    if (b != INT_MAX) sum += min(8, length - b * 8);
  }
  __syncthreads();
  const int count = Reduce(temp.reduce).Sum(sum);
  if (threadIdx.x == 0) counts[row] = count;
}

struct CandidatePublish {
  static void run(tvm::ffi::TensorView blocks, tvm::ffi::TensorView lengths,
                  tvm::ffi::TensorView sorted, tvm::ffi::TensorView counts) {
    using namespace host;
    auto rows = SymbolicSize{"rows"};
    auto device = SymbolicDevice{};
    device.set_options<kDLGPU>();
    TensorMatcher({rows, 2048})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(blocks);
    TensorMatcher({rows, 2048})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(sorted);
    TensorMatcher({rows}).with_dtype<int32_t>().with_device(device).verify(
        lengths);
    TensorMatcher({rows}).with_dtype<int32_t>().with_device(device).verify(
        counts);
    if (!rows.unwrap()) return;
    LaunchKernel(rows.unwrap(), 256, device.unwrap())(
        publish_candidate_blocks,
        static_cast<const int32_t*>(blocks.data_ptr()),
        static_cast<const int32_t*>(lengths.data_ptr()),
        static_cast<int32_t*>(sorted.data_ptr()),
        static_cast<int32_t*>(counts.data_ptr()));
  }
};

}  // namespace sglang
