# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math


class VllmSelector:
    """Exact auto top-k C ops for SM90 with fixed topk=512, aligned FP32 logits.

    Mirrors pinned indexer_topk.py; owns the same 1 MiB workspace rather than
    constructing a model/WorkspaceManager. No torch.topk performance baseline.
    """

    def __init__(self, device, rows):
        import torch

        self.workspace = torch.empty(1024 * 1024, device=device, dtype=torch.uint8)
        self.backend = "cooperative" if rows <= 64 else "persistent"

    def __call__(self, scores, lengths, output, max_len):
        import torch

        op = getattr(torch.ops._C, self.backend + "_topk")
        op(scores, lengths, output, self.workspace, 512, max_len)


def make_inputs(case, *, stress=False):
    import torch

    r, length, verify = (
        case["rows"],
        case["compressed_length"],
        case["rows_per_request"],
    )
    requests, page = (r // verify, case["page_size"])
    pages = math.ceil(length / page)
    generator = torch.Generator(device="cuda").manual_seed(20261101)
    raw = torch.empty((requests * pages, page * 132), device="cuda", dtype=torch.uint8)
    for start in range(0, raw.shape[0], 1024):
        stop = min(start + 1024, raw.shape[0])
        payload = torch.randn(
            (stop - start, page * 128), generator=generator, device="cuda"
        ).to(torch.float8_e4m3fn)
        raw[start:stop, : page * 128] = payload.view(torch.uint8)
        exps = torch.randint(
            -4 if stress else -1,
            5 if stress else 2,
            (stop - start, page),
            generator=generator,
            device="cuda",
        )
        scale = torch.exp2(exps.float())
        raw[start:stop, page * 128 :] = scale.view(torch.uint8)
    cache = raw.view(requests * pages, page, 1, 132)
    block_table = (
        torch.randperm(requests * pages, device="cuda", generator=generator)
        .int()
        .view(requests, pages)
    )
    q = torch.randn((r, 32, 128), device="cuda", generator=generator).to(
        torch.float8_e4m3fn
    )
    weights = torch.randn((r, 32), device="cuda", generator=generator) / math.sqrt(
        128 * 32
    )
    req = torch.arange(requests, device="cuda").repeat_interleave(verify).int()
    offsets = torch.arange(verify, device="cuda").repeat(requests)
    lens = (length - verify + 1 + offsets).clamp_min(0).int()
    blocks = torch.stack(
        [
            torch.randperm(pages * page // 8, device="cuda", generator=generator)[
                : case["candidate_blocks"]
            ]
            for _ in range(r)
        ]
    ).int()
    if stress:
        lens[0] = 0
        if r > 1:
            lens[1] = min(length, 517)
        blocks[:, 1] = blocks[:, 0]
        blocks[:, 2] = -1
    blocks[:, -1] = (lens - 1) // 8
    return (q, None, weights, cache, block_table, req, lens, blocks)
