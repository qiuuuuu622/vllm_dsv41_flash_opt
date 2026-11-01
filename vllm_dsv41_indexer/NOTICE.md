# Source attribution

The FP8 candidate scorer is adapted from `antgroup/sglang` commit
`6e6d5d9e13915adb01db0856b15d6908eaaa2c48`, specifically
`python/sglang/kernels/ops/attention/dsv4/sm90_fp4_indexer_wgmma.py`.
The original project is licensed under Apache-2.0. Its license is included in
`third_party/ant/LICENSE`.

Changes to the scorer: vLLM physical page-table lookup; FP8 E4M3 payloads instead
of packed FP4; FP8 shared-memory/MMA shapes; FP32 key scales, head weighting and
score reduction. No new activation or cache quantization is introduced.

`third_party/ant/` contains the unchanged `topk_v2.cuh` and its transitive
SGLang header dependencies from the same commit. `sources.json` records their
original paths and SHA256 values. Upstream attributions to vLLM and FlashInfer
inside these headers are preserved. The build enables deterministic ties and
disables unused cluster paths: this module's score width is fixed at 16,384.
Only the register-resident compact top-k dispatcher is reachable.

`csrc/candidate_publish.cuh` is a new CUB radix-sort/position-count adapter,
inspired by Ant's source-side publication and consumer reuse mechanism. It is
not an unchanged Ant kernel. `prepare.py` is the generic sort/scan duplicate
fallback. These adaptations are licensed under Apache-2.0.

No SGLang Python package namespace is replaced by the released module.
