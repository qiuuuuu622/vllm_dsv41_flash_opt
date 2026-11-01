# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Function A/B for TP2/4/8, always with eight useful input heads per GPU."""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from vllm_dsv41_head_exchange.workspace import FusedWorkspace, NcclWorkspace

from .baseline import (
    BASELINE,
    FLASHMLA,
    check_reference,
    load_flashmla,
    measure,
    packed_cache,
    stats,
)


def warm(fn, count):
    for _ in range(count):
        fn()
    torch.accelerator.synchronize()
    dist.barrier()


def graph_measure(fn, repeat, inner):
    warm(fn, 30)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(inner):
            fn()
    warm(graph.replay, 10)
    result = measure(graph.replay, repeat, 1)
    return stats([value / inner for value in result["samples_us"]])


def run_case(args, fm, rows, ratio, rank, workspace_cls=NcclWorkspace):
    torch.manual_seed(20261101 + 100 * ratio + rows)
    swa, si, sd = packed_cache(rows, 128, 128, 64)
    extra, ei, ed = None, None, None
    if ratio:
        extra, ei, ed = packed_cache(rows, 512, args.context // ratio, 64 // ratio)
    sl = torch.full((rows,), 128, dtype=torch.int32, device="cuda")
    el = torch.full((rows,), 512, dtype=torch.int32, device="cuda") if ratio else None
    if args.vary_lengths:
        r = torch.arange(rows, device="cuda")
        sl = torch.where(r % 4 == 0, 0, torch.minimum(r * 19, sl)).int()
        si.masked_fill_(
            torch.arange(128, device="cuda")[None, None, :] >= sl[:, None, None], -1
        )
        if ratio:
            el = torch.where(r % 4 == 0, 0, torch.minimum(r * 79, el)).int()
            ei.masked_fill_(
                torch.arange(512, device="cuda")[None, None, :] >= el[:, None, None], -1
            )
    torch.manual_seed(317 + rank)
    q = torch.zeros(rows, 1, 64, 512, dtype=torch.bfloat16, device="cuda")
    q[:, :, :8] = torch.randn(rows, 1, 8, 512, dtype=torch.bfloat16, device="cuda")
    sink = torch.full((64,), -torch.inf, dtype=torch.float32, device="cuda")
    sink[:8] = torch.linspace(-2, 3, 8, device="cuda") + rank * 0.25
    output = torch.empty_like(q)
    metadata = fm.get_mla_metadata()[0]
    ws = workspace_cls(q, sink)
    row_si, row_sl = ws.metadata(si, -1), ws.metadata(sl)
    row_ei, row_el = ws.metadata(ei, -1), ws.metadata(el)
    row_metadata = fm.get_mla_metadata()[0]

    def attention(
        query, sinks, swa_indices, swa_lengths, extra_indices, extra_lengths, md, out
    ):
        return fm.flash_mla_with_kvcache(
            q=query,
            k_cache=swa,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=512,
            tile_scheduler_metadata=md,
            is_fp8_kvcache=True,
            indices=swa_indices,
            topk_length=swa_lengths,
            softmax_scale=512**-0.5,
            attn_sink=sinks,
            extra_k_cache=extra,
            extra_indices_in_kvcache=extra_indices,
            extra_topk_length=extra_lengths,
            out=out,
        )

    def baseline():
        return attention(q, sink, si, sl, ei, el, metadata, output)

    def exchanged():
        row_q = ws.forward_q(q)
        attention(
            row_q, ws.sink, row_si, row_sl, row_ei, row_el, row_metadata, ws.row_o
        )
        return ws.backward_o()

    def row_compute():
        return attention(
            ws.row_q, ws.sink, row_si, row_sl, row_ei, row_el, row_metadata, ws.row_o
        )

    def transport_only():
        ws.forward_q(q)
        ws.backward_o()

    baseline()
    exchanged()
    torch.accelerator.synchronize()
    baseline_error = check_reference(q, sink, output, sd, ed, sl, el)
    optimized_error = check_reference(q, sink, ws.output, sd, ed, sl, el)
    original = output[:, 0, :8].float()
    optimized = ws.output[:, 0, :8].float()
    diff = optimized - original
    torch.testing.assert_close(optimized, original, rtol=0.02, atol=0.002)
    nrmse = (diff.norm() / original.norm().clamp_min(1e-12)).item()
    assert nrmse < 0.01
    delta = {
        "max_abs": diff.abs().max().item(),
        "nrmse": nrmse,
        "bitwise_equal": torch.equal(output[:, 0, :8], ws.output[:, 0, :8]),
    }
    del sd, ed, original, optimized, diff
    timings = {}
    for name, fn in [
        ("baseline", baseline),
        ("exchanged_total", exchanged),
        ("row_attention_only", row_compute),
        ("exchange_and_packing_only", transport_only),
    ]:
        warm(fn, 100)
        eager = measure(fn, args.repeat, args.inner)
        graph = graph_measure(fn, args.repeat, args.inner)
        timings[name] = {"eager": eager, "graph": graph}
    return {
        "rank": rank,
        "workspace_class": type(ws).__name__,
        "exchange_blocks": getattr(ws, "blocks", None),
        "exchange_threads": getattr(ws, "threads", None),
        "rows": rows,
        "compress_ratio": ratio,
        "context": args.context,
        "vary_lengths": args.vary_lengths,
        "local_input_heads": 8,
        "row_shard_effective_heads": ws.active,
        "padded_heads": 64,
        "local_output_rows": ws.chunk,
        "baseline_error": baseline_error,
        "exchange_error": optimized_error,
        "exchange_vs_baseline": delta,
        "timings": timings,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--transport", choices=["nccl", "fused"], default="fused")
    p.add_argument("--rows", nargs="+", type=int, default=[8, 32, 64, 128, 256, 512])
    p.add_argument("--ratios", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--context", type=int, default=32768)
    p.add_argument("--repeat", type=int, default=30)
    p.add_argument("--inner", type=int, default=20)
    p.add_argument("--vary-lengths", action="store_true")
    p.add_argument("--interface", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    workspace_cls = FusedWorkspace if args.transport == "fused" else NcclWorkspace
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    world = dist.get_world_size()
    assert world in (2, 4, 8)
    fm, extension = load_flashmla(args.interface)
    result = {
        "status": "running",
        "transport_variant": args.transport,
        "baseline_commit": BASELINE,
        "flashmla_commit": FLASHMLA,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "torch": torch.__version__,
        "extension": extension,
        "physical_ranks": world,
        "input_heads_per_rank": 8,
        "exchanged_heads_per_rank": 8 * world,
        "scope": (
            "Function experiment, no production hooks. Persistent workspace; "
            "sink gathering and sparse metadata preparation outside timing. "
            "Warm cache/planner. Total includes both exchanges, layout writes, "
            "FlashMLA and output restoration. TP2/4 use eight heads per input "
            "rank, TP8 uses the full 64-head layout. No full-model accuracy or "
            "serving throughput claim."
        ),
        "measurement": {
            "repeat": args.repeat,
            "inner": args.inner,
            "warmup_calls": 100,
        },
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for ratio in args.ratios:
            for rows in args.rows:
                local = run_case(args, fm, rows, ratio, rank, workspace_cls)
                gathered = [None] * world
                dist.all_gather_object(gathered, local)
                if rank == 0:
                    item = {"rows": rows, "compress_ratio": ratio, "ranks": gathered}
                    result["cases"].append(item)
                    slow = {
                        name: max(
                            r["timings"][name]["graph"]["p50_us"] for r in gathered
                        )
                        for name in gathered[0]["timings"]
                    }
                    print(
                        json.dumps(
                            {
                                "rows": rows,
                                "ratio": ratio,
                                "graph_p50_us": slow,
                                "speedup": slow["baseline"] / slow["exchanged_total"],
                                "max_nrmse": max(
                                    r["exchange_vs_baseline"]["nrmse"] for r in gathered
                                ),
                            }
                        ),
                        flush=True,
                    )
                    args.output.write_text(json.dumps(result, indent=2) + "\n")
                torch.accelerator.empty_cache()
        result["status"] = "completed"
    except BaseException as exc:
        result["status"] = "failed"
        result["error"] = repr(exc)
        raise
    finally:
        if rank == 0:
            args.output.write_text(json.dumps(result, indent=2) + "\n")
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
