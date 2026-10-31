# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode adapter: exchange real heads for rows, then restore the TP layout."""

import torch

from .config import eligible_tensor, record_path
from .distributed import heads_to_rows, rows_to_heads
from .layout import RowPartition


def try_head_exchange(
    layer,
    q,
    kv_cache,
    swa_cache,
    swa_metadata,
    swa_indices,
    swa_lens,
    topk_indices,
    topk_lens,
    original_schedule,
    output,
):
    if not (
        layer._h20_tp8
        and layer.n_heads == 64
        and layer.n_local_heads == 8
        and q.shape[0] >= layer._h20_options.head_min_rows
        and q.dtype == torch.bfloat16
        and eligible_tensor(q, "head_exchange")
    ):
        return False
    from vllm.distributed import get_tp_group
    from vllm.v1.attention.ops.flashmla import (
        flash_mla_with_kvcache,
        get_mla_metadata,
    )

    group = get_tp_group()
    record_path("head_exchange")
    plan = RowPartition(q.shape[0], group.world_size)
    rank = group.rank_in_group
    row_q = heads_to_rows(q[:, : layer.n_local_heads].contiguous(), group)
    sinks = group.all_gather(layer.attn_sink[: layer.n_local_heads].contiguous(), dim=0)

    # A new upstream scheduler object denotes a new decode step. Never reuse
    # its full-row schedule for the sharded batch or across decode steps.
    schedules = getattr(swa_metadata, "_h20_row_schedules", None)
    if schedules is None:
        schedules = swa_metadata._h20_row_schedules = {}
    entry = schedules.get(layer.compress_ratio)
    if entry is None or entry[0] is not original_schedule:
        entry = (original_schedule, get_mla_metadata()[0])
        schedules[layer.compress_ratio] = entry

    row_output = torch.empty_like(row_q)
    flash_mla_with_kvcache(
        q=row_q.unsqueeze(1),
        k_cache=swa_cache,
        block_table=None,
        head_dim_v=512,
        tile_scheduler_metadata=entry[1],
        cache_seqlens=None,
        is_fp8_kvcache=True,
        indices=plan.shard(swa_indices, rank, -1),
        topk_length=plan.shard(swa_lens, rank),
        softmax_scale=layer.scale,
        attn_sink=sinks,
        extra_k_cache=kv_cache,
        extra_indices_in_kvcache=(
            plan.shard(topk_indices, rank, -1) if topk_indices is not None else None
        ),
        extra_topk_length=(
            plan.shard(topk_lens, rank) if topk_lens is not None else None
        ),
        out=row_output.unsqueeze(1),
    )
    restored = rows_to_heads(row_output, q.shape[0], group)
    output.zero_()
    output[:, : layer.n_local_heads].copy_(restored)
    return True
