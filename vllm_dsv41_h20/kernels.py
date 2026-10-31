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


@triton.jit
def _quant_group32(X, A, Scale, K: tl.constexpr):
    row, group = tl.program_id(0), tl.program_id(1)
    cols = group * 32 + tl.arange(0, 32)
    values = tl.load(X + row * K + cols).to(tl.float32)
    maximum = tl.maximum(tl.max(tl.abs(values), 0), 1e-4)
    scale = tl.exp2(tl.ceil(tl.log2(maximum / 448.0)))
    tl.store(A + row * K + cols, tl.minimum(tl.maximum(values / scale, -448.0), 448.0))
    tl.store(Scale + row * (K // 32) + group, scale)


@triton.jit
def _block32_split_k(
    A,
    AS,
    W,
    WS,
    Partial,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SPLITS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    ns, split = tl.program_id(0), tl.program_id(1)
    mi, ni, ki = tl.arange(0, BM), ns * BN + tl.arange(0, BN), tl.arange(0, 32)
    groups: tl.constexpr = K // 32
    per_split: tl.constexpr = triton.cdiv(groups, SPLITS)
    acc = tl.full((BM, BN), 0, tl.float32)
    for g in range(split * per_split, tl.minimum((split + 1) * per_split, groups)):
        a = tl.load(A + mi[:, None] * K + g * 32 + ki[None, :], mi[:, None] < M, 0)
        w = tl.load(W + ni[None, :] * K + g * 32 + ki[:, None], ni[None, :] < N, 0)
        sa = tl.load(AS + mi * groups + g, mi < M, 0)
        sw = tl.load(WS + ni * groups + g, ni < N, 127).to(tl.float32)
        product = tl.dot(a, w)
        acc += product * sa[:, None] * tl.exp2(sw[None, :] - 127.0)
    tl.store(
        Partial + split * M * N + mi[:, None] * N + ni[None, :],
        acc,
        (mi[:, None] < M) & (ni[None, :] < N),
    )


@triton.jit
def _finish_split(
    Partial,
    Bias,
    Output,
    SIZE: tl.constexpr,
    N: tl.constexpr,
    SPLITS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pos = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    result = tl.full((BLOCK,), 0, tl.float32)
    for split in tl.static_range(SPLITS):
        result += tl.load(Partial + split * SIZE + pos, pos < SIZE, 0)
    if HAS_BIAS:
        result += tl.load(Bias + pos % N, pos < SIZE, 0).to(tl.float32)
    tl.store(Output + pos, result, pos < SIZE)


def small_m_block32_gemm(x, weight, scale_bytes, *, split_k=4, bias=None):
    """Group-32 FP8 activation, FP8 weight, ordered FP32 splits, BF16 output."""
    m, k = x.shape
    n = weight.shape[0]
    if split_k not in (1, 2, 4, 8):
        raise ValueError("split_k must be 1, 2, 4 or 8")
    if weight.dtype != torch.float8_e4m3fn or scale_bytes.dtype != torch.uint8:
        raise ValueError("weight must be E4M3 and scales raw E8M0 bytes")
    if scale_bytes.shape != (n, k // 32) or n < 1:
        raise ValueError("invalid weight scale shape")
    if not (0 < m <= 16 and k % 32 == 0 and weight.shape[1] == k):
        raise ValueError("small-M path requires 1<=M<=16, K divisible by 32")
    if (
        not x.is_contiguous()
        or not weight.is_contiguous()
        or not scale_bytes.is_contiguous()
    ):
        raise ValueError("small-M path requires contiguous inputs")
    a = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty((m, k // 32), device=x.device, dtype=torch.float32)
    _quant_group32[(m, k // 32)](x, a, scales, k, num_warps=1)
    partial = torch.empty((split_k, m, n), device=x.device, dtype=torch.float32)
    _block32_split_k[(triton.cdiv(n, 64), split_k)](
        a,
        scales,
        weight,
        scale_bytes,
        partial,
        m,
        n,
        k,
        split_k,
        16,
        64,
        num_warps=4,
    )
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    _finish_split[(triton.cdiv(m * n, 256),)](
        partial,
        bias if bias is not None else out,
        out,
        m * n,
        n,
        split_k,
        bias is not None,
        256,
    )
    return out
