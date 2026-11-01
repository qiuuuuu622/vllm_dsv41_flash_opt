# Actual H20 baseline is FP8; derived FP8 WGMMA/cache adapter, not native Ant FP4.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Derived from antgroup/sglang 6e6d5d9e13915adb01db0856b15d6908eaaa2c48.
# See third_party/ant/LICENSE and NOTICE.md.
# Changes: FP8 page-table loads, FP8 WGMMA, FP32 scales/weights/reduction.
# Candidate state is prepared by workspace.py.

import torch
from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared,
    mbarrier,
    tma,
    warpgroup_mma,
)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

from vllm.triton_utils import gl, gluon, triton

BLOCK_K = gl.constexpr(64)


HEADS = 32


_CANDIDATE_GRID_CTAS = 4096


@gluon.jit
def _row_head_sum(s):
    # One row's [keys, 32] heads; same tree as _head_sum.
    x = gl.reshape(s, [BLOCK_K, 2, 2, 4, 2])  # key, a1, a0, q, e
    x = gl.permute(x, [0, 1, 3, 4, 2])
    x0, x1 = gl.split(x)
    x0 = gl.permute(x0, [0, 2, 3, 1])
    x1 = gl.permute(x1, [0, 2, 3, 1])
    h0, h2 = gl.split(x0)
    h1, h3 = gl.split(x1)
    partial = ((h0 + h1) + h2) + h3  # [keys, q, e]
    even, odd = gl.split(partial)
    even = gl.sum(gl.sum(gl.reshape(even, [BLOCK_K, 2, 2]), axis=1), axis=1)
    odd = gl.sum(gl.sum(gl.reshape(odd, [BLOCK_K, 2, 2]), axis=1), axis=1)
    return even + odd


@gluon.jit
def _candidate_slots(
    slots_ptr,
    blocks_ptr,
    req,
    tile,
    count,
    stride_slots,
    CACHE_PAGE_SIZE: gl.constexpr,
    CANDIDATE_BLOCK_SIZE: gl.constexpr,
    key_l: gl.constexpr,
):
    # Compact column c holds logical position blocks[c // size] * size + c % size.
    cols = tile * BLOCK_K + gl.arange(0, BLOCK_K, layout=key_l)
    valid = cols < count
    block = gl.load(blocks_ptr + cols // CANDIDATE_BLOCK_SIZE, mask=valid, other=0)
    logical = block.to(gl.int64) * CANDIDATE_BLOCK_SIZE + cols % CANDIDATE_BLOCK_SIZE
    page = gl.load(
        slots_ptr + req * stride_slots + logical // CACHE_PAGE_SIZE, mask=valid, other=0
    ).to(gl.int64)
    return page * CACHE_PAGE_SIZE + logical % CACHE_PAGE_SIZE, valid


@gluon.jit
def _tile_payload(table_ptr, slot, valid, page_size, row_stride, L3: gl.constexpr):
    page_base = (slot // page_size) * row_stride
    off = slot % page_size
    blk = gl.arange(0, 4, layout=gl.SliceLayout(0, gl.SliceLayout(2, L3)))
    byte = gl.arange(0, 32, layout=gl.SliceLayout(0, gl.SliceLayout(1, L3)))
    cols = gl.expand_dims(gl.expand_dims(blk * 32, 0), 2) + gl.expand_dims(
        gl.expand_dims(byte, 0), 1
    )
    row = gl.expand_dims(gl.expand_dims(page_base + off * 128, 1), 2)
    payload = gl.load(
        table_ptr + row + cols,
        mask=gl.expand_dims(gl.expand_dims(valid, 1), 2),
        other=0,
    )
    scale_ptr = (table_ptr + page_base + page_size * 128 + off * 4).cast(
        gl.pointer_type(gl.float32)
    )
    scales = gl.load(scale_ptr, mask=valid, other=1.0)
    return payload, scales


@gluon.jit
def _decode_tile(payload, scales):
    # Preserve FP8 values; apply FP32 key scales AFTER the tensor-core dot.
    return gl.reshape(payload, [BLOCK_K, 128]).to(gl.float8e4nv, bitcast=True)


@gluon.jit
def _candidate_wgmma_kernel(
    q_desc,  # [B * HEADS, 128] FP8, blocks of one row
    w_ptr,  # [B, HEADS] FP32
    slots_ptr,  # [num_reqs, context] request-to-token table
    req_ptr,  # [B] request IDs
    blocks_ptr,  # [B, K] sorted logical candidate blocks
    counts_ptr,  # [B] int64 valid compact columns per row
    table_ptr,
    out_ptr,  # [B, K * block size] fp32 compact logits
    L,
    page_size,
    row_stride,
    stride_wb,
    stride_slots,
    stride_blocks,
    stride_out,
    CACHE_PAGE_SIZE: gl.constexpr,
    CANDIDATE_BLOCK_SIZE: gl.constexpr,
):
    # One row per program: its candidates are its own, so K tiles are not
    # shared; wgmma (N = 32 heads) replaces the per-row register MMA.
    mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, 32, 32]
    )
    L3: gl.constexpr = gl.BlockedLayout([1, 1, 16], [8, 4, 1], [4, 1, 1], [2, 1, 0])
    key_l: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, L3))
    k_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [BLOCK_K, 128], gl.float8e4nv
    )
    q_smem = gl.allocate_shared_memory(gl.float8e4nv, [32, 128], q_desc.layout)
    k_smem = gl.allocate_shared_memory(gl.float8e4nv, [BLOCK_K, 128], k_layout)
    bar = mbarrier.allocate_mbarrier()

    row = gl.program_id(0)
    lane = gl.program_id(1)
    stride = gl.num_programs(1)
    req = gl.load(req_ptr + row).to(gl.int64)
    count = gl.load(counts_ptr + row)
    count = gl.minimum(gl.maximum(count, 0), L).to(gl.int32)
    blocks_row = blocks_ptr + row.to(gl.int64) * stride_blocks
    mbarrier.init(bar, count=1)
    mbarrier.expect(bar, 32 * 128)
    tma.async_copy_global_to_shared(q_desc, [row * 32, 0], bar, q_smem)
    mbarrier.wait(bar, 0)
    mbarrier.invalidate(bar)

    cols = gl.arange(0, 32, layout=gl.SliceLayout(0, mma_layout))
    w = gl.load(w_ptr + row * stride_wb + cols)
    acc = gl.zeros([BLOCK_K, 32], gl.float32, layout=mma_layout)
    w = gl.broadcast(gl.expand_dims(w, 0), acc)[0]
    slot, valid = _candidate_slots(
        slots_ptr,
        blocks_row,
        req,
        lane,
        count,
        stride_slots,
        CACHE_PAGE_SIZE,
        CANDIDATE_BLOCK_SIZE,
        key_l,
    )
    payload, scales = _tile_payload(table_ptr, slot, valid, page_size, row_stride, L3)
    slot, valid = _candidate_slots(
        slots_ptr,
        blocks_row,
        req,
        lane + stride,
        count,
        stride_slots,
        CACHE_PAGE_SIZE,
        CANDIDATE_BLOCK_SIZE,
        key_l,
    )
    for tile in range(lane, gl.cdiv(count, BLOCK_K), stride):
        key_scale = scales
        k_tile = _decode_tile(payload, scales)
        payload, scales = _tile_payload(
            table_ptr, slot, valid, page_size, row_stride, L3
        )
        slot, valid = _candidate_slots(
            slots_ptr,
            blocks_row,
            req,
            tile + 2 * stride,
            count,
            stride_slots,
            CACHE_PAGE_SIZE,
            CANDIDATE_BLOCK_SIZE,
            key_l,
        )
        gl.barrier()  # the previous tile's wgmma has retired
        k_smem.store(k_tile)
        fence_async_shared()
        gl.barrier()
        acc = warpgroup_mma(k_smem, q_smem.permute((1, 0)), acc, use_acc=False)
        key_scale = gl.convert_layout(key_scale, gl.SliceLayout(1, acc.type.layout))
        acc = acc * gl.expand_dims(key_scale, 1)
        logit = _row_head_sum(gl.maximum(acc, 0.0) * w.to(gl.float32))
        LL: gl.constexpr = logit.type.layout
        keys = tile * BLOCK_K + gl.arange(0, BLOCK_K, layout=LL)
        logit = gl.where(keys < count, logit, float("-inf"))
        gl.store(
            out_ptr + row.to(gl.int64) * stride_out + keys,
            logit,
            mask=keys < stride_out,
        )


def candidate_logits(
    q: torch.Tensor,
    weights: torch.Tensor,
    slots: torch.Tensor,
    req_indices: torch.Tensor,
    candidate_blocks: torch.Tensor,
    counts: torch.Tensor,
    table: torch.Tensor,
    page_size: int,
    out: torch.Tensor,
    *,
    cache_page_size: int,
    candidate_block_size: int,
    fp32_scores: bool = True,
) -> None:
    """Prefix-only compact logits of candidate consumers, one row per program."""
    if not fp32_scores:
        raise ValueError("Only the validated FP32 score epilogue is supported")
    B = q.shape[0]
    q_layout = gl.NVMMASharedLayout.get_default_for([32, 128], gl.float8e4nv)
    q_desc = TensorDescriptor.from_tensor(q.view(B * HEADS, 128), [32, 128], q_layout)
    lanes = min(triton.cdiv(out.shape[1], 64), triton.cdiv(_CANDIDATE_GRID_CTAS, B))
    _candidate_wgmma_kernel[(B, lanes)](
        q_desc,
        weights,
        slots,
        req_indices,
        candidate_blocks,
        counts,
        table,
        out,
        out.shape[1],
        page_size,
        table.stride(0),
        weights.stride(0),
        slots.stride(0),
        candidate_blocks.stride(0),
        out.stride(0),
        CACHE_PAGE_SIZE=cache_page_size,
        CANDIDATE_BLOCK_SIZE=candidate_block_size,
        num_warps=4,
    )
