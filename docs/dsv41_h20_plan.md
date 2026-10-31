# DSv4.1 H20 optimization implementation plan

Baseline: vLLM `v0.32.0`, commit `5df63a7203cdb9328d824a27b682e77c7b9928be`.
Target: one node, TP8 + EP8, DP1/PCP1/DCP1, NVIDIA H20 96 GB.

The four opt-in paths are independent. None is a measured H20 speedup yet.
The portable package does not import vLLM or Triton at import time, so its
numerical references, layouts and communication model can run on macOS.

| Flag | Existing mechanism | Proposed mechanism | Integration |
| --- | --- | --- | --- |
| `compact_indexer` | Score all visible keys, then mask candidates | Read and score candidate positions directly from packed MXFP4 K | Existing indexer prefill/decode consumers; source layers unchanged |
| `small_m_gemm` | General MXFP8 backend for all M | Group-32 FP8 quantization and deterministic split-K GEMM for small M | Opt-in Marlin/Humming MXFP8 linear adapters; original backend fallback |
| `compressor_shard` | Every rank projects all rows | Project a padded contiguous row shard, all-gather projected rows | V4.1 compressor projection; large row counts (including mixed batches) |
| `head_exchange` | Each rank attends all rows with heads padded to 64 | All-to-all Q heads to row owners; inverse exchange outputs | V4.1 FlashMLA decode, replicated KV only |

## Test contracts before implementation

- Candidate scoring: packed E2M1 values/E8M0 scales, head weights, candidate
  block IDs and visible lengths -> scores/top-k logical positions. Compare
  with independently computed full-context masked scores; test invalid,
  duplicate and reordered candidates, ties, causal tails and paged caches.
- Split-K GEMM: row-wise group-32 activation quantization and FP8 weights ->
  BF16 output. Compare split accumulation against dequantized FP64/FP32
  multiplication; report quantization error separately from kernel error.
- Compressor: replicated `[M,K]` input and `[N,K]` weights -> `[M,N]` on every
  rank. Test uneven M, empty rows and rank order; compare with a dense reference.
- Head exchange: head-sharded `[M,H/P,D]` tensors -> row-sharded `[ceil(M/P),H,D]`
  tensors and back. Test inverse identity, nondivisible M, sinks and attention
  results; use transport-independent packing functions in runtime and tests.
- Communication model: alpha/bandwidth estimates use explicit supplied
  assumptions, never CPU copy timing. Test byte counts and break-even formulas.

## Validation boundaries

macOS tests can prove reference arithmetic and layout behavior, not CUDA
compilation, GPU race freedom, NCCL capture safety, model accuracy or speedup.
The CPU Gloo benchmark additionally exercises real eight-process collectives
through the runtime transport adapters; these timings are not NVLink timings.
GPU tests and single-/multi-GPU benchmark entry points must be provided, with
unsupported environments skipped explicitly. No full model weights are required
for the kernel suites. End-to-end serving/model evaluation remains a separate gate.

The implementation must not claim bitwise equivalence when FP8 quantization,
split-K summation or FlashMLA scheduling changes. Unsupported layouts fall back
before issuing collectives; an error after collective entry is not swallowed.
