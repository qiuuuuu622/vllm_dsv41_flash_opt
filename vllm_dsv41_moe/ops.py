# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in finalize kernels preserving vLLM 0.32.0 contracts.

Ant-inspired two-dimensional launch and shared-output fusion. GEMM, activation,
quantization, routing, communication and weight layouts are not changed here.
The caller must supply shared output at the same reduction stage as routed
output. These functions never move shared output across a collective.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _round_to_output(x, DTYPE: tl.constexpr):
    if DTYPE == tl.bfloat16:  # noqa: SIM300 - Triton constexpr dtype.
        # Make the materialized BF16 boundary explicit: scaling followed by
        # addition must not be contracted across either original store/load.
        return tl.inline_asm_elementwise(
            "{ .reg .b16 h; cvt.rn.bf16.f32 h, $1; cvt.f32.bf16 $0, h; }",
            constraints="=f,f",
            args=[x],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    return x.to(DTYPE).to(tl.float32)


@triton.jit
def _indexed(
    X,
    W,
    O,  # noqa: E741 - Output pointer, matching the measured kernel.
    IDS,
    MAP,
    VALID,
    SHARED,
    M: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    HAS_IDS: tl.constexpr,
    HAS_MAP: tl.constexpr,
    HAS_VALID: tl.constexpr,
    HAS_SHARED: tl.constexpr,
    SHARED_STRIDE: tl.constexpr,
    SCALE: tl.constexpr,
    BM: tl.constexpr,
    BH: tl.constexpr,
):
    rows = tl.program_id(0).to(tl.int64) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BH + tl.arange(0, BH)
    row_ok = rows < M
    active = row_ok
    if HAS_VALID:
        active = active & (rows < tl.load(VALID))
    any_valid = tl.zeros((BM,), tl.int32)
    acc = tl.zeros((BM, BH), tl.float32)
    for k in tl.static_range(K):
        take = active
        if HAS_IDS:
            expert = tl.load(IDS + rows * K + k, row_ok, other=-1)
            valid_id = expert >= 0
            any_valid += valid_id.to(tl.int32)
            take = take & valid_id
            if HAS_MAP:
                local = tl.load(
                    MAP + tl.maximum(expert, 0), row_ok & valid_id, other=-1
                )
                take = take & (local >= 0)
        weight = tl.load(W + rows * K + k, take, other=0).to(tl.float32)
        x = tl.load(
            X + ((rows * K + k) * H)[:, None] + cols[None, :],
            take[:, None] & (cols < H)[None, :],
            other=0,
        ).to(tl.float32)
        acc += x * weight[:, None]
    if HAS_IDS:
        active = active & (any_valid > 0)
    offset = rows[:, None] * H + cols[None, :]
    mask = row_ok[:, None] & (cols < H)[None, :]
    routed = _round_to_output(acc, O.dtype.element_ty)
    if HAS_SHARED:
        # Original indexed finalize preserves skipped output rows; a subsequent
        # torch.add still adds shared to those rows. Preserve both operations.
        old = tl.load(O + offset, mask & ~active[:, None], other=0)
        routed = tl.where(active[:, None], routed, old.to(tl.float32))
        routed = _round_to_output(routed * SCALE, O.dtype.element_ty)
        shared = tl.load(
            SHARED + rows[:, None] * SHARED_STRIDE + cols[None, :], mask, other=0
        )
        result = routed.to(tl.float32) + shared.to(tl.float32)
        tl.store(O + offset, result.to(O.dtype.element_ty), mask)
    else:
        tl.store(O + offset, routed, mask & active[:, None])


@triton.jit
def _grouped(
    X,
    W,
    O,  # noqa: E741 - Output pointer, matching the measured kernel.
    INVERSE,
    VALID,
    SHARED,
    M: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    CAPACITY: tl.constexpr,
    HAS_VALID: tl.constexpr,
    HAS_SHARED: tl.constexpr,
    SHARED_STRIDE: tl.constexpr,
    SCALE: tl.constexpr,
    BM: tl.constexpr,
    BH: tl.constexpr,
):
    rows = tl.program_id(0).to(tl.int64) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BH + tl.arange(0, BH)
    valid_rows = CAPACITY
    if HAS_VALID:
        valid_rows = tl.load(VALID)
    acc = tl.zeros((BM, BH), tl.float32)
    for k in tl.static_range(K):
        source = tl.load(INVERSE + rows * K + k, rows < M, other=CAPACITY).to(tl.int64)
        take = (rows < M) & (source >= 0) & (source < valid_rows)
        weight = tl.load(W + rows * K + k, take, other=0).to(tl.float32)
        x = tl.load(
            X + source[:, None] * H + cols[None, :],
            take[:, None] & (cols < H)[None, :],
            other=0,
        ).to(tl.float32)
        acc += x * weight[:, None]
    mask = (rows < M)[:, None] & (cols < H)[None, :]
    routed = _round_to_output(acc, O.dtype.element_ty)
    if HAS_SHARED:
        routed = _round_to_output(routed * SCALE, O.dtype.element_ty)
        shared = tl.load(
            SHARED + rows[:, None] * SHARED_STRIDE + cols[None, :], mask, other=0
        )
        routed = (routed.to(tl.float32) + shared.to(tl.float32)).to(O.dtype.element_ty)
    tl.store(O + rows[:, None] * H + cols[None, :], routed, mask)


def _check(inputs, weights, output, shared):
    assert inputs.is_cuda and inputs.is_contiguous()
    assert weights.is_contiguous() and output.is_contiguous()
    assert inputs.dtype == output.dtype
    assert inputs.device == weights.device == output.device
    assert inputs.dtype in (torch.float16, torch.bfloat16, torch.float32)
    if shared is not None:
        assert shared.shape == output.shape and shared.dtype == output.dtype
        assert shared.device == output.device and shared.stride(-1) == 1
        assert shared.data_ptr() != output.data_ptr()


def moe_fused_mul_sum(
    inputs,
    topk_weights,
    outputs=None,
    topk_ids=None,
    expert_map=None,
    num_valid_tokens=None,
    *,
    shared=None,
    routed_scaling_factor=1.0,
):
    m, k, h = inputs.shape
    if outputs is None:
        assert shared is None, "Fused shared addition requires initialized output."
        outputs = torch.empty((m, h), dtype=inputs.dtype, device=inputs.device)
    _check(inputs, topk_weights, outputs, shared)
    assert routed_scaling_factor == 1.0 or (
        shared is not None and inputs.dtype == torch.bfloat16
    )
    assert outputs.shape == (m, h) and topk_weights.shape == (m, k)
    assert expert_map is None or topk_ids is not None
    if m == 0:
        return outputs
    bm = 1 if m <= 4 else 2 if m <= 128 else 4
    _indexed[(triton.cdiv(m, bm), triton.cdiv(h, 512))](
        inputs,
        topk_weights,
        outputs,
        topk_ids,
        expert_map,
        num_valid_tokens,
        shared,
        m,
        h,
        k,
        topk_ids is not None,
        expert_map is not None,
        num_valid_tokens is not None,
        shared is not None,
        shared.stride(0) if shared is not None else h,
        routed_scaling_factor,
        bm,
        512,
        num_warps=4,
        num_stages=4,
    )
    return outputs


def moe_unpermute(
    out,
    permuted_hidden_states,
    topk_weights,
    inv_permuted_idx,
    expert_first_token_offset=None,
    *,
    shared=None,
    routed_scaling_factor=1.0,
):
    m, h = out.shape
    k = topk_weights.shape[1]
    _check(permuted_hidden_states, topk_weights, out, shared)
    assert routed_scaling_factor == 1.0 or (
        shared is not None and out.dtype == torch.bfloat16
    )
    assert topk_weights.shape == (m, k) and inv_permuted_idx.numel() == m * k
    if m == 0:
        return
    valid = (
        expert_first_token_offset[-1:]
        if expert_first_token_offset is not None
        else None
    )
    bm = 1 if m <= 4 else 2 if m <= 128 else 4
    _grouped[(triton.cdiv(m, bm), triton.cdiv(h, 512))](
        permuted_hidden_states,
        topk_weights,
        out,
        inv_permuted_idx,
        valid,
        shared,
        m,
        h,
        k,
        permuted_hidden_states.numel() // h,
        valid is not None,
        shared is not None,
        shared.stride(0) if shared is not None else h,
        routed_scaling_factor,
        bm,
        512,
        num_warps=4,
        num_stages=4,
    )
