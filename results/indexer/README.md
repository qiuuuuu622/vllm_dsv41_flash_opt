# Indexer evidence

`20261101/` preserves the original experiments, including their raw JSON,
summary CSV, model configuration and exact Python source archive. The original
Ant Python loader requires the complete Ant checkout at commit
`6e6d5d9e13915adb01db0856b15d6908eaaa2c48`; that whole serving tree is not vendored
in this repository. These are historical results, not renamed release runs.

`release/` contains new runs of the committed module after removing the
experimental SGLang namespace loader and unused cluster plans:

- `matrix_final.json` and `matrix_reverse_final.json`: 12 decode shapes in
  each order, eager and CUDA Graph, 20 warmups and 30 samples of 10 calls.
- `verify6_final.json`: six verify shapes; the original DeepGEMM baseline
  expands six rows into N=1 rather than claiming a fused N=6 implementation.
- `stress_final.json`: four empty/partial/duplicate/invalid-candidate cases,
  also using wider scale exponents. Finite visible scores are assumed.
- `logical_tp8_final.json`: eight TP rank contracts on **one physical GPU**,
  each with a differently permuted local KV page table. Real kernels run;
  physical eight-GPU execution is not simulated or claimed.
- `tp2_final.json`: real two-process/two-H20 verification. Local physical page
  mappings differ; logical outputs agree. It tests consistency, not scaling.
- Matching logs retain the commands' outputs. `comparison.csv` averages the
  two runs' P50 values; it does not pool or mix single- and four-consumer costs.

The final 34 benchmark cases compare four distinct Q/head-weight pairs each:
136 comparisons, max score nRMSE 7.233e-8, min top-k set recall 1.0 in this run.
The prototype recorded one equal-score boundary substitution, so universal
bitwise equivalence is not claimed. New graph checks change candidates/lengths,
exercise empty/partial windows and duplicate IDs, test exact ties and poison
unread score tails with NaNs. No full model or CP-sharded cache is tested.

`portable_tests.xml` records 11 passing CPU contract/source-integrity checks.
`evidence.json` pins release code and results by SHA256. Baseline Python source
hashes are checked against the official vLLM wheel before each benchmark.

The optimized single-consumer function includes candidate publication, scores,
top-k and output mapping. The four-consumer variant includes one publication
and four distinct queries/weights. Source candidate selection, Q/K projections,
cache insertion, scheduler work and full-model execution are outside timing.
GPUs were not clock-locked. All speedups here are single-GPU function results.

To regenerate the release CSV from the repository root:

```bash
.venv/bin/python -m benchmarks.kernels.dsv41_indexer.summarize \
  results/indexer/release/matrix_final.json \
  results/indexer/release/matrix_reverse_final.json \
  --output results/indexer/release/comparison.csv
```
