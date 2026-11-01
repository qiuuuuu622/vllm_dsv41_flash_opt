# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSV4.1 H20 baseline: TP2/4/8, eight useful heads per rank.

Loads the unmodified FlashMLA interface pinned by vLLM v0.32.0 and the
official v0.32.0 compiled extension. No model, server, or head exchange.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

BASELINE = "5df63a7203cdb9328d824a27b682e77c7b9928be"
FLASHMLA = "0eee43b12f034b657133cf2afca6a72ebb6efccf"


def load_flashmla(interface):
    package = importlib.metadata.distribution("vllm")
    assert package.version == "0.32.0", package.version
    files = list(Path(package.locate_file("vllm")).glob("_flashmla_C*.so"))
    assert len(files) == 1, files
    shipped = Path(
        package.locate_file("vllm/third_party/flashmla/flash_mla_interface.py")
    )
    assert (
        shipped.read_text().replace("import vllm._flashmla_C\n", "")
        == interface.read_text()
    ), "Wheel interface differs from v0.32.0's pinned FlashMLA source"
    torch.ops.load_library(str(files[0]))
    spec = importlib.util.spec_from_file_location("baseline_flashmla", interface)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, str(files[0])


def packed_cache(rows, width, span, block):
    """Independent V4 cache encoder; only selected tokens are populated.

    The H20 path uses 448 E4M3 NoPE + 64 BF16 RoPE + 8 UE8M0 bytes,
    with separate data/scales regions per page and a 576-byte page alignment.
    Each query has a disjoint request cache; extra indices scatter across span.
    """
    assert span % block == 0 and span >= width
    offset = torch.randint(span, (rows, 1), device="cuda")
    stride = 2 * torch.randint(max(span // 2, 1), (rows, 1), device="cuda") + 1
    if span == width:
        offset.zero_()
        stride.fill_(1)
    positions = (offset + stride * torch.arange(width, device="cuda")) % span
    slots = positions + torch.arange(rows, device="cuda")[:, None] * span
    # span is a power of two and stride is odd, so positions do not repeat.
    kv = torch.randn(rows * width, 512, device="cuda", dtype=torch.bfloat16)
    tiles = kv[:, :448].float().reshape(-1, 7, 64)
    exponent = torch.ceil(torch.log2((tiles.abs().amax(-1) / 448).clamp_min(1e-4)))
    scale = torch.exp2(exponent)
    quant = (tiles / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    decoded = torch.cat(
        [(quant.float() * scale.unsqueeze(-1)).flatten(1), kv[:, 448:].float()],
        dim=-1,
    ).view(rows, width, 512)
    nblocks = rows * span // block
    page_stride = math.ceil(block * 584 / 576) * 576
    storage = torch.zeros(nblocks, page_stride, device="cuda", dtype=torch.uint8)
    data = storage[:, : block * 576].view(nblocks, block, 576)
    scales = storage[:, block * 576 : block * 584].view(nblocks, block, 8)
    page, inner = slots.flatten() // block, slots.flatten() % block
    data[page, inner, :448] = quant.view(torch.uint8).reshape(-1, 448)
    data[page, inner, 448:] = kv[:, 448:].contiguous().view(torch.uint8)
    scales[page, inner, :7] = (exponent + 127).to(torch.uint8)
    cache = storage[:, : block * 584].view(nblocks, block, 1, 584)
    return cache, slots.to(torch.int32).unsqueeze(1), decoded


def stats(values):
    return {
        "p50_us": float(np.median(values)),
        "p95_us": float(np.percentile(values, 95)),
        "min_us": float(np.min(values)),
        "samples_us": values,
    }


def measure(fn, repeat, inner):
    samples = []
    for _ in range(repeat):
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / inner)
    return stats(samples)


def check_reference(q, sink, output, swa, extra, swa_lens, extra_lens):
    keys = swa if extra is None else torch.cat([swa, extra], dim=1)
    visible = torch.arange(128, device="cuda")[None, :] < swa_lens[:, None]
    if extra is not None:
        visible = torch.cat(
            [visible, torch.arange(512, device="cuda")[None, :] < extra_lens[:, None]],
            dim=1,
        )
    scores = torch.einsum("mhd,mtd->mht", q[:, 0, :8].float(), keys) * 512**-0.5
    scores.masked_fill_(~visible[:, None, :], -torch.inf)
    sinks = sink[:8].view(1, 8, 1).expand(q.shape[0], 8, 1)
    probabilities = torch.softmax(torch.cat([scores, sinks], dim=-1), dim=-1)[..., :-1]
    reference = torch.einsum("mht,mtd->mhd", probabilities, keys)
    actual = output[:, 0, :8].float()
    diff = actual - reference
    nrmse = (diff.norm() / reference.norm().clamp_min(1e-12)).item()
    torch.testing.assert_close(actual, reference, rtol=0.02, atol=0.002)
    assert nrmse < 0.01, nrmse
    assert torch.isfinite(actual).all()
    return {
        "max_abs_error_vs_dequantized_fp32": diff.abs().max().item(),
        "normalized_rmse_vs_dequantized_fp32": nrmse,
        "allclose_rtol": 0.02,
        "allclose_atol": 0.002,
    }


def run_case(args, fm, rows, ratio, rank):
    torch.manual_seed(20261101 + 100 * ratio + rows)
    swa, si, sd = packed_cache(rows, 128, 128, 64)
    extra, ei, ed = (None, None, None)
    if ratio:
        extra, ei, ed = packed_cache(rows, 512, args.context // ratio, 64 // ratio)
    sl = torch.full((rows,), 128, dtype=torch.int32, device="cuda")
    el = torch.full((rows,), 512, dtype=torch.int32, device="cuda") if ratio else None
    if args.vary_lengths:
        row = torch.arange(rows, device="cuda")
        sl = torch.where(row % 4 == 0, 0, torch.minimum(row * 19, sl)).int()
        si.masked_fill_(
            torch.arange(128, device="cuda")[None, None, :] >= sl[:, None, None], -1
        )
        if ratio:
            el = torch.where(row % 4 == 0, 0, torch.minimum(row * 79, el)).int()
            ei.masked_fill_(
                torch.arange(512, device="cuda")[None, None, :] >= el[:, None, None], -1
            )
    torch.manual_seed(317 + rank)
    q = torch.zeros(rows, 1, 64, 512, device="cuda", dtype=torch.bfloat16)
    q[:, :, :8] = torch.randn(rows, 1, 8, 512, device="cuda", dtype=torch.bfloat16)
    sink = torch.full((64,), -float("inf"), device="cuda", dtype=torch.float32)
    sink[:8] = torch.linspace(-2, 3, 8, device="cuda") + rank * 0.25
    output = torch.empty_like(q)
    metadata = fm.get_mla_metadata()[0]

    def call(md=metadata):
        return fm.flash_mla_with_kvcache(
            q=q,
            k_cache=swa,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=512,
            tile_scheduler_metadata=md,
            is_fp8_kvcache=True,
            indices=si,
            topk_length=sl,
            softmax_scale=512**-0.5,
            attn_sink=sink,
            extra_k_cache=extra,
            extra_indices_in_kvcache=ei,
            extra_topk_length=el,
            out=output,
        )

    call()
    torch.accelerator.synchronize()
    correctness = check_reference(q, sink, output, sd, ed, sl, el)
    del sd, ed
    for _ in range(args.warmup):
        call()
    torch.accelerator.synchronize()
    warm_until = time.monotonic() + args.warmup_seconds
    while time.monotonic() < warm_until:
        for _ in range(100):
            call()
        torch.accelerator.synchronize()
    active_bytes = torch.accelerator.memory_allocated()
    eager = measure(call, args.repeat, args.inner)
    planner = measure(lambda: call(fm.get_mla_metadata()[0]), args.repeat, args.inner)
    graph = torch.cuda.CUDAGraph()
    dist.barrier()
    with torch.cuda.graph(graph):
        for _ in range(args.inner):
            call()
    graph_timing = measure(graph.replay, args.repeat, 1)
    graph_timing = stats([x / args.inner for x in graph_timing["samples_us"]])
    result = {
        "rank": rank,
        "rows": rows,
        "compress_ratio": ratio,
        "context_tokens": args.context if ratio else 128,
        "extra_cache_tokens_per_request": args.context // ratio if ratio else 0,
        "swa_tokens": 128,
        "extra_topk": 512 if ratio else 0,
        "local_effective_heads": 8,
        "padded_heads": 64,
        "head_dim": 512,
        "kv_record_bytes": 584,
        "swa_page_tokens": 64,
        "extra_page_tokens": 64 // ratio if ratio else None,
        "vary_lengths": args.vary_lengths,
        "active_tensor_bytes_before_timing": active_bytes,
        "correctness": correctness,
        "eager_reused_scheduler": eager,
        "eager_fresh_scheduler": planner,
        "graph_reused_scheduler_per_call": graph_timing,
    }
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, nargs="+", default=[1, 8, 32, 64, 128, 256, 512])
    p.add_argument("--ratios", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--context", type=int, default=32768)
    p.add_argument("--repeat", type=int, default=30)
    p.add_argument("--inner", type=int, default=20)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--warmup-seconds", type=float, default=1.0)
    p.add_argument("--vary-lengths", action="store_true")
    p.add_argument("--interface", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    assert args.context >= 1024 and args.context & (args.context - 1) == 0
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    assert dist.get_world_size() in (2, 4, 8)
    assert torch.cuda.get_device_capability() == (9, 0)
    assert "H20" in torch.cuda.get_device_name()
    fm, library = load_flashmla(args.interface)
    result = {
        "status": "running",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "baseline_commit": BASELINE,
        "flashmla_commit": FLASHMLA,
        "interface_sha256": hashlib.sha256(args.interface.read_bytes()).hexdigest(),
        "extension": library,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "nccl": torch.cuda.nccl.version(),
        "gpu": torch.cuda.get_device_name(),
        "physical_ranks": dist.get_world_size(),
        "local_effective_heads": 8,
        "padded_heads": 64,
        "scope": (
            "Eight useful input heads per GPU padded to 64. TP2/4 are subproblems. "
            "Synthetic KV/Q/sinks; reference uses the same quantized KV. "
            "Warm caches/planner; eager includes host launch gaps. "
            "Input preparation, indexer and output projection excluded. "
            "No model accuracy or serving throughput claim."
        ),
        "measurement": {
            "repeat": args.repeat,
            "inner": args.inner,
            "warmup": args.warmup,
            "warmup_seconds": args.warmup_seconds,
        },
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for ratio in args.ratios:
            for rows in args.rows:
                local = run_case(args, fm, rows, ratio, rank)
                gathered = [None] * dist.get_world_size()
                dist.all_gather_object(gathered, local)
                if rank == 0:
                    result["cases"].append(
                        {"rows": rows, "compress_ratio": ratio, "ranks": gathered}
                    )
                    args.output.write_text(json.dumps(result, indent=2) + "\n")
                    print(
                        json.dumps(
                            {
                                "rows": rows,
                                "ratio": ratio,
                                "worst_rank_graph_p50_us": max(
                                    x["graph_reused_scheduler_per_call"]["p50_us"]
                                    for x in gathered
                                ),
                                "worst_rank_nrmse": max(
                                    x["correctness"][
                                        "normalized_rmse_vs_dequantized_fp32"
                                    ]
                                    for x in gathered
                                ),
                            }
                        ),
                        flush=True,
                    )
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
