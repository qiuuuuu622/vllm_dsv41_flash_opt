# Source attribution

This experimental CUDA transport adapts the final-layout peer push, warp-tiled
vector copy, and two-phase per-CTA arrival protocol from antgroup/sglang commit
`380e2d5bda9dbff32d924765c5a6c1c54bb9ad47` (Apache-2.0):

- <https://github.com/antgroup/sglang/blob/380e2d5bda9dbff32d924765c5a6c1c54bb9ad47/python/sglang/kernels/jit/csrc/distributed/segment_a2a.cuh>
- <https://github.com/antgroup/sglang/blob/380e2d5bda9dbff32d924765c5a6c1c54bb9ad47/python/sglang/kernels/jit/include/sgl_kernel/distributed/communicator.cuh>

Changes in this experiment: standalone C ABI instead of TVM/sgl_kernel FFI;
PyTorch symmetric-memory buffers; fixed 8192-byte head segments; two independent
Q/O signal planes; 128-byte semaphore spacing; explicit timeout diagnostics;
specializations for world sizes 2/4/8 (only 2 tested); no PDL integration.
The historical prototype source is archived under results/head_exchange/20261101.

The upstream Apache-2.0 license is provided in `LICENSE` in this directory.
