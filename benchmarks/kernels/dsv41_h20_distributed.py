# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NCCL GPU / Gloo CPU transport smoke test. Launch with torchrun; no weights."""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist

from vllm_dsv41_h20.distributed import heads_to_rows, project_rows, rows_to_heads
from vllm_dsv41_h20.layout import RowPartition


class Group:
    """Small standalone implementation of the group interface used by the hooks."""

    def __init__(self):
        self.world_size = dist.get_world_size()
        self.rank_in_group = dist.get_rank()
        self.device_group = dist.group.WORLD

    def all_gather(self, x, dim=0):
        assert dim == 0
        result = torch.empty(
            (self.world_size * x.shape[0], *x.shape[1:]), device=x.device, dtype=x.dtype
        )
        dist.all_gather_into_tensor(result, x.contiguous())
        return result


def slowest_rank_median_ms(fn, device):
    for _ in range(3):
        fn()
    if device == "cuda":
        torch.accelerator.synchronize()
    measurements = []
    for _ in range(10):
        dist.barrier()
        if device == "cuda":
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            fn()
            end.record()
            end.synchronize()
            elapsed_ms = start.elapsed_time(end)
        else:
            begin = time.perf_counter()
            fn()
            elapsed_ms = (time.perf_counter() - begin) * 1000
        elapsed = torch.tensor(elapsed_ms, device=device)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        measurements.append(elapsed.item())
    return statistics.median(measurements)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=2049)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    device = args.device
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Real NCCL testing requires multiple CUDA GPUs")
        torch.accelerator.set_device_index(int(os.environ["LOCAL_RANK"]))
    torch.set_num_threads(1)
    dist.init_process_group("nccl" if device == "cuda" else "gloo")
    try:
        group = Group()
        if group.world_size < 2 or args.rows < 1:
            raise ValueError("at least two ranks and one row required")
        torch.manual_seed(191)
        # All ranks get identical replicated hidden states and compressor weights.
        x = torch.randn(args.rows, 512, device=device, dtype=torch.bfloat16)
        w = torch.randn(1024, 512, device=device, dtype=torch.bfloat16)

        def dense_projection():
            if device == "cuda":
                return torch.mm(x, w.T, out_dtype=torch.float32)
            return x.float() @ w.float().T

        expected = dense_projection()
        projected = project_rows(x, w, group)
        torch.testing.assert_close(projected, expected, rtol=2e-4, atol=1e-3)
        torch.manual_seed(719 + group.rank_in_group)
        q = torch.randn(args.rows, 8, 512, device=device, dtype=torch.bfloat16)
        full_q = group.all_gather(q).view(group.world_size, args.rows, 8, 512)
        full_q = full_q.permute(1, 0, 2, 3).reshape(
            args.rows, group.world_size * 8, 512
        )
        row_q = heads_to_rows(q, group)
        plan = RowPartition(args.rows, group.world_size)
        torch.testing.assert_close(
            row_q, plan.shard(full_q, group.rank_in_group), rtol=0, atol=0
        )
        restored = rows_to_heads(row_q, args.rows, group)
        torch.testing.assert_close(restored, q, rtol=0, atol=0)
        dense_ms = slowest_rank_median_ms(dense_projection, device)
        sharded_ms = slowest_rank_median_ms(lambda: project_rows(x, w, group), device)
        exchange_ms = slowest_rank_median_ms(
            lambda: rows_to_heads(heads_to_rows(q, group), args.rows, group), device
        )
        if group.rank_in_group == 0:
            rendered = json.dumps(
                {
                    "device": torch.cuda.get_device_name()
                    if device == "cuda"
                    else "CPU",
                    "backend": dist.get_backend(),
                    "world_size": group.world_size,
                    "rows": args.rows,
                    "projection_M_N_K": [args.rows, 1024, 512],
                    "projection_max_abs_error": (projected - expected)
                    .abs()
                    .max()
                    .item(),
                    "projection_dense_ms": dense_ms,
                    "projection_shard_gather_ms": sharded_ms,
                    "projection_speedup_measured": dense_ms / sharded_ms,
                    "heads_roundtrip_exact": True,
                    "two_exchanges_and_packing_ms": exchange_ms,
                    "scope": (
                        "Real process-group transport. CPU/Gloo is not NVLink. "
                        "Not FlashMLA or model-level validation."
                    ),
                },
                indent=2,
            )
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(rendered + "\n")
            print(rendered)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
