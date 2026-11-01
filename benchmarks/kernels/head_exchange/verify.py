# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact byte-layout, changing-input, graph-replay and lagging-rank tests."""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from vllm_dsv41_head_exchange.workspace import FusedWorkspace


def check_case(rows, iterations, mode, rank):
    q = torch.zeros(rows, 1, 64, 512, device="cuda", dtype=torch.bfloat16)
    sink = torch.arange(64, device="cuda", dtype=torch.float32) + rank * 100
    world = dist.get_world_size()
    ws = FusedWorkspace(q, sink)
    active = 8 * world
    expected_sink = torch.cat(
        [torch.arange(8, device="cuda") + r * 100 for r in range(world)]
    ).float()
    assert torch.equal(ws.sink[:active], expected_sink)
    assert torch.isneginf(ws.sink[active:]).all()
    index = torch.arange(rows * 8 * 512, device="cuda").view(rows, 8, 512)
    local_rows = torch.arange(ws.chunk, device="cuda") + rank * ws.chunk
    h = torch.arange(active, device="cuda")
    d = torch.arange(512, device="cuda")
    row_index = (
        local_rows[:, None, None] * (8 * 512)
        + (h[None, :, None] % 8) * 512
        + d[None, None, :]
    )
    row_index = row_index + 13 * (h[None, :, None] // 8)
    expected_row = torch.zeros_like(ws.row_q)
    checks = torch.empty(iterations, 3, device="cuda", dtype=torch.bool)

    def exchange():
        ws.forward_q(q)
        ws.row_o.copy_(ws.row_q)
        ws.backward_o()

    exchange()
    torch.accelerator.synchronize()
    graph = None
    if mode == "graph":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            exchange()
    for step in range(iterations):
        # Include arbitrary BF16 bit patterns (NaNs/infinities included). Compare
        # integer views so NaN equality cannot hide or falsely report corruption.
        def encode(value):
            return ((value * 257) % 65536 - 32768).to(torch.int16).view(torch.bfloat16)

        q[:, 0, :8].copy_(encode(index + 13 * rank + 17 * step))
        expected_row[:, 0, :active] = encode(row_index + 17 * step)
        expected_row.masked_fill_(local_rows[:, None, None, None] >= rows, 0)
        if step % 13 == 0 and rank == (step // 13) % world:
            time.sleep(0.002)
        if graph is None:
            exchange()
        else:
            graph.replay()
        checks[step, 0] = torch.all(
            ws.row_q.view(torch.int16) == expected_row.view(torch.int16)
        )
        checks[step, 1] = torch.all(
            ws.output[:, 0, :8].view(torch.int16) == q[:, 0, :8].view(torch.int16)
        )
        checks[step, 2] = torch.all(ws.output[:, 0, 8:] == 0)
    torch.accelerator.synchronize()
    passed = checks.cpu().all().item()
    assert passed, {
        "rows": rows,
        "rank": rank,
        "mode": mode,
        "failed": (~checks).nonzero().cpu().tolist(),
    }
    dist.barrier()
    return {
        "rows": rows,
        "mode": mode,
        "iterations": iterations,
        "rank": rank,
        "blocks": ws.blocks,
        "exact": True,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, nargs="+", default=[1, 7, 9, 32, 127, 129, 512])
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    # Configuration disagreement must fail collectively before any peer kernel.
    for mismatched_rows in (True, False):
        q = torch.zeros(
            1 + rank if mismatched_rows else 1,
            1,
            64,
            512,
            device="cuda",
            dtype=torch.bfloat16,
        )
        sink = torch.zeros(64, device="cuda", dtype=torch.float32)
        try:
            FusedWorkspace(q, sink, blocks=1 if mismatched_rows else 1 + rank)
        except ValueError as exc:
            assert "Incompatible head-exchange ranks" in str(exc)
        else:
            raise AssertionError("Mismatched rank configurations were accepted")
    result = {"status": "running", "ranks": dist.get_world_size(), "cases": []}
    result["configuration_mismatch_rejected"] = True
    try:
        for rows in args.rows:
            for mode in ["eager", "graph"]:
                case = check_case(rows, args.iterations, mode, rank)
                gathered = [None] * dist.get_world_size()
                dist.all_gather_object(gathered, case)
                if rank == 0:
                    result["cases"].extend(gathered)
                    print(
                        json.dumps(
                            {
                                "rows": rows,
                                "mode": mode,
                                "iterations_per_rank": args.iterations,
                                "exact": True,
                            }
                        ),
                        flush=True,
                    )
                    args.output.write_text(json.dumps(result, indent=2) + "\n")
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
