# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental Triton kernels. Imported only by explicitly selected CUDA paths."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _e2m1(byte, odd):
    code = (byte >> (odd * 4)) & 15
    mag = code & 7
    value = tl.where(
        mag < 4, mag * 0.5, tl.where(mag < 6, mag - 2.0, (mag - 4.0) * 2.0)
    )
    return tl.where((code & 8) != 0, -value, value)


@triton.jit
def _compact_mxfp4_scores(
    Q,
    QS,
    K,
    KS,
    Weights,
    Positions,
    Starts,
    Ends,
    Tables,
    Output,
    C: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    Q0: tl.constexpr,
    Q1: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    KS0: tl.constexpr,
    W0: tl.constexpr,
    W1: tl.constexpr,
    T0: tl.constexpr,
    PAGE: tl.constexpr,
    NEXT: tl.constexpr,
    PAGED: tl.constexpr,
    BH: tl.constexpr,
    BC: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.program_id(1) * BC + tl.arange(0, BC)
    pos = tl.load(Positions + row * C + columns, columns < C, -1).to(tl.int64)
    end = tl.load(Ends + row)
    start = 0 if PAGED else tl.load(Starts + row)
    valid = (columns < C) & (pos >= 0) & (pos < end - start)
    if PAGED:
        physical = tl.load(Tables + row // NEXT * T0 + pos // PAGE, valid, 0).to(
            tl.int64
        )
        base = physical * K0 + pos % PAGE * K1
    else:
        base = (start + pos) * K0
    heads = tl.arange(0, BH)
    dims = tl.arange(0, D)
    qb = tl.load(
        Q + row * Q0 + heads[:, None] * Q1 + dims[None, :] // 2, heads[:, None] < H, 0
    )
    qs = tl.load(
        QS + row * S0 + heads[:, None] * S1 + dims[None, :] // 32,
        heads[:, None] < H,
        127,
    ).to(tl.float32)
    q = _e2m1(qb, dims[None, :] % 2) * tl.where(
        qs == 255, float("nan"), tl.exp2(qs - 127.0)
    )
    kb = tl.load(K + base[None, :] + dims[:, None] // 2, valid[None, :], 0)
    if PAGED:
        ks = tl.load(
            K + base[None, :] + D // 2 + dims[:, None] // 32, valid[None, :], 127
        )
    else:
        ks = tl.load(
            KS + (start + pos[None, :]) * KS0 + dims[:, None] // 32, valid[None, :], 127
        )
    k = _e2m1(kb, dims[:, None] % 2) * tl.where(
        ks == 255, float("nan"), tl.exp2(ks.to(tl.float32) - 127.0)
    )
    dots = tl.dot(q, k, input_precision="tf32x3")
    weights = tl.load(Weights + row * W0 + heads * W1, heads < H, 0).to(tl.float32)
    scores = tl.sum(tl.maximum(dots, 0.0) * weights[:, None], axis=0)
    tl.store(
        Output + row * C + columns, tl.where(valid, scores, -float("inf")), columns < C
    )


def compact_mxfp4_scores(
    q,
    q_scales,
    keys,
    key_scales,
    weights,
    positions,
    starts,
    ends,
    *,
    block_table=None,
    page_size=0,
    next_n=1,
):
    """Scores compact positions; q/keys are packed uint8, scales raw E8M0 bytes."""
    rows, heads, packed_d = q.shape
    if packed_d != 64 or q.dtype != torch.uint8:
        raise ValueError("the first CUDA implementation supports MXFP4 D=128")
    scores = torch.empty(positions.shape, device=q.device, dtype=torch.float32)
    if not rows or not positions.shape[1]:
        return scores
    paged = block_table is not None
    _compact_mxfp4_scores[(rows, triton.cdiv(positions.shape[1], 32))](
        q,
        q_scales,
        keys,
        key_scales if not paged else keys,
        weights,
        positions,
        starts,
        ends,
        block_table if paged else starts,
        scores,
        positions.shape[1],
        heads,
        128,
        q.stride(0),
        q.stride(1),
        q_scales.stride(0),
        q_scales.stride(1),
        keys.stride(0),
        keys.stride(1) if paged else 0,
        key_scales.stride(0) if not paged else 0,
        weights.stride(0),
        weights.stride(1),
        block_table.stride(0) if paged else 0,
        page_size,
        next_n,
        paged,
        max(16, triton.next_power_of_2(heads)),
        32,
        num_warps=4,
    )
    return scores
