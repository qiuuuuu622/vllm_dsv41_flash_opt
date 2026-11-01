# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check real kernels with permuted rank-local KV pages and changing graph inputs.

--logical-tp8 exercises eight independent rank contracts on one physical GPU;
--distributed exercises the actual torchrun group. Neither claims TP8 hardware
validation when only one or two physical GPUs are present.
"""

import argparse
import hashlib
import os

import torch
import torch.distributed as dist

from vllm_dsv41_indexer import IndexerParallel
from vllm_dsv41_indexer.workspace import CandidateState, CompactIndexer

from .common import check_vllm, save
from .inputs import make_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logical-tp8", action="store_true")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--output", default="build/indexer/verify.json")
    args = parser.parse_args()
    if args.distributed and args.logical_tp8:
        parser.error("Choose physical distributed ranks or logical TP8 simulation")
    check_vllm()
    physical_rank = int(os.environ.get("LOCAL_RANK", 0)) if args.distributed else 0
    torch.accelerator.set_device_index(physical_rank)
    if args.distributed:
        dist.init_process_group("nccl")
    world = dist.get_world_size() if args.distributed else 1
    contracts = (
        [IndexerParallel(8, rank) for rank in range(8)]
        if args.logical_tp8
        else [
            IndexerParallel.from_group(dist.group.WORLD)
            if args.distributed
            else IndexerParallel()
        ]
    )
    case = dict(
        rows=6,
        compressed_length=65536,
        rows_per_request=6,
        page_size=64,
        candidate_blocks=2048,
    )
    q, _, weights, cache, table, req, lens, blocks = make_inputs(case)
    logical_cache = cache[table.flatten().long()].clone()
    q_original = q.clone()
    block_original = blocks.clone()
    lengths_original = lens.clone()
    checks, reference = [], None
    for contract in contracts:
        generator = torch.Generator(device="cuda").manual_seed(900 + contract.tp_rank)
        perm = torch.randperm(cache.shape[0], generator=generator, device="cuda")
        local_cache = torch.empty_like(cache)
        local_cache[perm] = logical_cache
        local_table = perm.int().view_as(table)
        state = CandidateState(6, q.device, parallel=contract)
        op = CompactIndexer(state, q, weights, local_cache, local_table, req)

        def call(state=state, op=op):
            state.publish(blocks, lens)
            op.forward()

        call()
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        for step in range(3):
            q.copy_(q_original)
            blocks.copy_(block_original)
            lens.copy_(lengths_original)
            if step == 1:
                # Empty, partial and duplicate/invalid candidates under graph.
                lens[0] = 0
                lens[1] = 517
                blocks[:, 1] = blocks[:, 0]
                blocks[:, 2] = -1
            elif step == 2:
                # Exact ties should choose lowest logical positions.
                q.zero_()
            graph.replay()
            selected = op.output.cpu()
            sorted_blocks = state.blocks.cpu().tolist()
            counts = state.counts.cpu().tolist()
            host_blocks, host_lens = blocks.cpu().tolist(), lens.cpu().tolist()
            for row in range(6):
                valid_blocks = sorted(
                    {b for b in host_blocks[row] if 0 <= b < (host_lens[row] + 7) // 8}
                )
                positions = [
                    b * 8 + i
                    for b in valid_blocks
                    for i in range(8)
                    if b * 8 + i < host_lens[row]
                ]
                assert sorted_blocks[row] == valid_blocks + [2147483647] * (
                    2048 - len(valid_blocks)
                )
                assert counts[row] == len(positions)
                actual = [x for x in selected[row].tolist() if x >= 0]
                assert len(set(actual)) == len(actual) == min(512, len(positions))
                assert set(actual) <= set(positions)
                if step == 2:
                    assert set(actual) == set(positions[:512])
            digest = hashlib.sha256(
                selected.sort(1).values.numpy().tobytes()
            ).hexdigest()
            if step == 0:
                if reference is None:
                    reference = digest
                assert digest == reference, (
                    "Logical results changed with TP rank's physical pages"
                )
            checks.append(
                dict(tp_rank=contract.tp_rank, step=step, output_sha256=digest)
            )
        q.copy_(q_original)
        blocks.copy_(block_original)
        lens.copy_(lengths_original)

    # Fused top-k must ignore poisoned score tails and preserve exact ties.
    state = CandidateState(6, q.device)
    state.blocks.copy_(
        torch.arange(2048, device=q.device, dtype=torch.int32)[None, :] * 2
    )
    state.counts.copy_(
        torch.tensor([0, 1, 511, 512, 513, 16384], device=q.device, dtype=torch.int32)
    )
    scores = torch.full((6, 16384), float("nan"), device=q.device)
    output = torch.empty((6, 512), device=q.device, dtype=torch.int32)
    for row, n in enumerate(state.counts.cpu().tolist()):
        scores[row, :n] = torch.arange(n, device=q.device) % 4 - 2
    state.native.topk(scores, state.counts, state.blocks, output, 8, state.plan, None)
    for row, n in enumerate(state.counts.cpu().tolist()):
        idx = torch.argsort(scores[row, :n], descending=True, stable=True)[:512]
        expected = set((idx // 8 * 16 + idx % 8).cpu().tolist())
        assert {x for x in output[row].cpu().tolist() if x >= 0} == expected

    payload = dict(
        status="passed",
        physical_world=world,
        logical_tp_size=8 if args.logical_tp8 else world,
        logical_simulation=args.logical_tp8,
        checks=checks,
        poisoned_tail_and_ties="passed",
        reference=reference,
    )
    if args.distributed:
        results = [None] * world
        dist.all_gather_object(results, payload)
        assert len({x["reference"] for x in results}) == 1
        if dist.get_rank() == 0:
            save(args.output, dict(status="passed", ranks=results))
        dist.destroy_process_group()
    else:
        save(args.output, payload)
    print(f"Passed indexer checks on physical rank {physical_rank}", flush=True)


if __name__ == "__main__":
    main()
