# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Adapters included in end-to-end consumer timing; no full-cache relayout."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _unpack_q(Q, S, OUT, N: tl.constexpr, TILE: tl.constexpr):
    i = tl.program_id(0) * TILE + tl.arange(0, TILE)
    b = tl.load(Q + i // 2, i < N, 0).to(tl.int32)
    code = (b >> ((i % 2) * 4)) & 15
    e, m = (code >> 1) & 3, (code & 1).to(tl.float32)
    value = tl.where(e == 0, m * 0.5, (1 + m * 0.5) * tl.exp2(e.to(tl.float32) - 1))
    value = tl.where(code & 8 != 0, -value, value)
    exponent = tl.load(S + i // 32, i < N, 127).to(tl.int32)
    tl.store(OUT + i, value * tl.exp2(exponent.to(tl.float32) - 127), i < N)


@triton.jit
def _canonical(
    BLOCKS,
    LENS,
    SORTED,
    COUNTS,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0)
    i = tl.arange(0, TILE)
    length = tl.load(LENS + row)
    b = tl.load(BLOCKS + row * K + i, i < K, -1)
    sentinel = 2147483647
    valid = (i < K) & (b >= 0) & (b < tl.cdiv(length, BLOCK))
    b = tl.sort(tl.where(valid, b, sentinel))
    previous = tl.gather(b, tl.maximum(i - 1, 0), 0)
    b = tl.sort(tl.where((i > 0) & (b == previous), sentinel, b))
    n = tl.where(b != sentinel, tl.minimum(tl.maximum(length - b * BLOCK, 0), BLOCK), 0)
    tl.store(SORTED + row * K + i, b, i < K)
    tl.store(COUNTS + row, tl.sum(n, 0))


@triton.jit
def _map_topk(
    SELECTED,
    SCORES,
    BLOCKS,
    COUNTS,
    OUT,
    K: tl.constexpr,
    CANDIDATES: tl.constexpr,
    BLOCK: tl.constexpr,
    WIDTH: tl.constexpr,
    STRIDE: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0)
    i = tl.arange(0, TILE)
    col = tl.load(SELECTED + row * K + i, i < K, -1)
    count = tl.load(COUNTS + row)
    valid = (i < K) & (col >= 0) & (col < count) & (col < WIDTH)
    score = tl.load(SCORES + row * STRIDE + col, valid, -float("inf"))
    valid &= score > -float("inf")
    b = tl.load(BLOCKS + row * CANDIDATES + col // BLOCK, valid, 0)
    tl.store(OUT + row * K + i, tl.where(valid, b * BLOCK + col % BLOCK, -1), i < K)


class CandidateWorkspace:
    def __init__(
        self, rows, candidates, device, topk=512, block=8, q_dtype=torch.bfloat16
    ):
        self.block, self.topk, self.candidates = block, topk, candidates
        self.q = torch.empty((rows, 32, 128), device=device, dtype=q_dtype)
        self.blocks = torch.empty((rows, candidates), device=device, dtype=torch.int32)
        self.counts = torch.empty(rows, device=device, dtype=torch.int32)
        self.selected = torch.empty((rows, topk), device=device, dtype=torch.int32)
        self.output = torch.empty_like(self.selected)
        # Prefix-only scores: all consumers MUST obey counts, never torch.topk
        # on the entire allocation (the invisible suffix is uninitialized).
        self.scores = torch.empty(
            (rows, candidates * block), device=device, dtype=torch.float32
        )

    def prepare(self, q_packed, q_scales, blocks, lengths):
        _unpack_q[(triton.cdiv(self.q.numel(), 256),)](
            q_packed, q_scales, self.q, self.q.numel(), 256
        )
        self.prepare_candidates(blocks, lengths)

    def prepare_candidates(self, blocks, lengths):
        _canonical[(self.q.shape[0],)](
            blocks,
            lengths,
            self.blocks,
            self.counts,
            self.candidates,
            self.block,
            triton.next_power_of_2(self.candidates),
        )

    def map_output(self):
        _map_topk[(self.q.shape[0],)](
            self.selected,
            self.scores,
            self.blocks,
            self.counts,
            self.output,
            self.topk,
            self.candidates,
            self.block,
            self.scores.shape[1],
            self.scores.stride(0),
            triton.next_power_of_2(self.topk),
        )
        return self.output
