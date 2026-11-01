# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint-free CPU/CUDA arithmetic benchmarks and explicit communication model."""

import argparse
import json
import platform
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

from vllm_dsv41_h20.communication import (
    LinkModel,
    head_exchange_bytes,
    projected_gather_bytes,
)
from vllm_dsv41_h20.layout import (
    pack_heads_to_rows,
    pack_rows_to_heads,
    sharded_projection_reference,
    simulate_all_to_all,
    unpack_heads_to_rows,
    unpack_rows_to_heads,
)
from vllm_dsv41_h20.reference import (
    canonical_candidates,
    compact_scores_reference,
    compact_topk,
    unpack_mxfp4,
)


def errors(actual, expected):
    delta = actual.float() - expected.float()
    return {
        "max_abs": delta.abs().max().item(),
        "normalized_rmse": (
            delta.square().mean().sqrt()
            / expected.float().square().mean().sqrt().clamp_min(1e-20)
        ).item(),
    }


def median_ms(fn, iterations=5, cuda=False):
    fn()
    if cuda:
        torch.accelerator.synchronize()
    times = []
    for _ in range(iterations):
        if cuda:
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            fn()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
        else:
            start = time.perf_counter()
            fn()
            times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def cpu_results():
    torch.manual_seed(20261031)
    torch.set_num_threads(1)
    rows, heads, dim, length, block = 8, 32, 128, 8192, 128
    q = torch.randn(rows, heads, dim)
    keys, weights = torch.randn(length, dim), torch.rand(rows, heads)
    candidates = torch.tensor([[0, 3, 31, 63]]).repeat(rows, 1)
    positions = canonical_candidates(candidates, block)
    visible = torch.tensor([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8192])
    dense = (torch.einsum("rhd,ld->rhl", q, keys).relu() * weights[..., None]).sum(1)
    selected = torch.zeros_like(dense, dtype=torch.bool)
    selected.scatter_(1, positions, True)
    selected &= torch.arange(length)[None] < visible[:, None]
    dense.masked_fill_(~selected, -float("inf"))
    compact = compact_scores_reference(q, keys, weights, positions, visible)
    dense_positions = torch.arange(length).expand(rows, -1)
    reference_topk = compact_topk(dense, dense_positions, 256)
    compact_indices = compact_topk(compact, positions, 256)
    finite = torch.isfinite(compact)
    indexer = {
        "shape_R_H_D_L": [rows, heads, dim, length],
        "candidate_capacity": positions.shape[1],
        "dense_score_elements": rows * length,
        "compact_score_elements": positions.numel(),
        "score_capacity_ratio": length / positions.shape[1],
        "topk_exact_match": bool(torch.equal(reference_topk, compact_indices)),
        "score_error": errors(compact[finite], dense.gather(1, positions)[finite]),
        "cpu_compact_reference_ms": median_ms(
            lambda: compact_scores_reference(q, keys, weights, positions, visible)
        ),
    }
    x, w = torch.randn(2049, 256).bfloat16(), torch.randn(128, 256).bfloat16()
    shard_result = sharded_projection_reference(x, w, 8)
    projection = {
        "M_N_K": [2049, 128, 256],
        "world": 8,
        "error": errors(shard_result, x.float() @ w.float().T),
        "cpu_serial_shards_ms": median_ms(
            lambda: sharded_projection_reference(x, w, 8)
        ),
    }
    q = torch.randn(129, 64, 64).bfloat16()
    sends = [pack_heads_to_rows(q[:, i * 8 : (i + 1) * 8], 8) for i in range(8)]
    row_q = [unpack_heads_to_rows(x) for x in simulate_all_to_all(sends)]
    receives = simulate_all_to_all([pack_rows_to_heads(x, 8) for x in row_q])
    restored = torch.cat([unpack_rows_to_heads(x, 129) for x in receives], dim=1)
    return {
        "indexer": indexer,
        "compressor": projection,
        "head_exchange": {"M_H_D": list(q.shape), "error": errors(restored, q)},
        "timing_scope": (
            "CPU reference arithmetic / serial logical ranks. "
            "Not H20 or NVLink latency."
        ),
    }


def communication_results():
    result = []
    for bandwidth in (50, 100, 200):
        for latency in (2, 5, 10):
            model = LinkModel(bandwidth, latency)
            gather = projected_gather_bytes(16384, 1024)
            exchange = head_exchange_bytes(128, 8, 512)
            # Include the tiny per-head FP32 sink all-gather performed by the adapter.
            head_us = 2 * model.all_to_all_us(exchange) + model.all_gather_us(64 * 4)
            result.append(
                {
                    "assumed_effective_GB_per_s": bandwidth,
                    "assumed_startup_us_per_round": latency,
                    "compressor_full_gather_bytes": gather,
                    "compressor_remote_send_bytes_per_rank": gather * 7 // 8,
                    "compressor_exposed_comm_us": model.all_gather_us(gather),
                    "head_each_exchange_total_send_bytes_per_rank": exchange,
                    "head_total_comm_including_sinks_us": head_us,
                }
            )
    return {
        "scope": "Analytical pairwise/ring model; assumptions, not measurements.",
        "compressor_example_M_K_N": [16384, 5120, 1024],
        "compressor_dense_FLOPs_per_rank": 2 * 16384 * 5120 * 1024,
        "compressor_shard_FLOPs_per_rank": 2 * 2048 * 5120 * 1024,
        "head_example_M_Hlocal_D": [128, 8, 512],
        "head_old_padded_row_heads_per_rank": 128 * 64,
        "head_new_row_heads_per_rank": 16 * 64,
        "scenarios": result,
        "break_even_rule": (
            "measured original compute - local compute - packing/planner overhead "
            "must exceed exposed communication"
        ),
    }


def cuda_results():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError(
            "CUDA benchmark requires SM90 (H20 or H100); no fabricated result"
        )
    from vllm_dsv41_h20.kernels import compact_mxfp4_scores

    torch.manual_seed(20261031)
    q = torch.randint(0, 256, (8, 32, 64), device="cuda", dtype=torch.uint8)
    qs = torch.full((8, 32, 4), 125, device="cuda", dtype=torch.uint8)
    keys = torch.randint(0, 256, (8192, 64), device="cuda", dtype=torch.uint8)
    ks = torch.full((8192, 4), 125, device="cuda", dtype=torch.uint8)
    weights = torch.rand(8, 32, device="cuda")
    positions = canonical_candidates(
        torch.tensor([[0, 3, 31, 63]] * 8, device="cuda"), 128
    )
    starts = torch.zeros(8, device="cuda", dtype=torch.int32)
    ends = torch.full_like(starts, 8192)

    def score():
        return compact_mxfp4_scores(q, qs, keys, ks, weights, positions, starts, ends)

    expected_scores = compact_scores_reference(
        unpack_mxfp4(q, qs), unpack_mxfp4(keys, ks), weights, positions, ends
    )
    actual_scores = score()
    torch.testing.assert_close(actual_scores, expected_scores, rtol=2e-5, atol=2e-3)
    return {
        "device": torch.cuda.get_device_name(),
        "compact_score_error": errors(actual_scores, expected_scores),
        "compact_score_only_ms": median_ms(score, cuda=True),
        "compact_score_and_topk_ms": median_ms(
            lambda: compact_topk(score(), positions, 256), cuda=True
        ),
        "scope": (
            "Synthetic kernel latency. No upstream backend baseline, "
            "no serving speedup claim."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["cpu", "communication", "cuda"], default="cpu"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "seed": 20261031,
        "mode": args.mode,
        "baseline": "vllm v0.32.0 / 5df63a7203cdb9328d824a27b682e77c7b9928be",
        "results": {
            "cpu": cpu_results,
            "communication": communication_results,
            "cuda": cuda_results,
        }[args.mode](),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
