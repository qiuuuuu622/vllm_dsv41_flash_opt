# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compact consumer adapter; retains the upstream candidate source and KV format."""

import torch

from .config import eligible_tensor, get_options, record_path
from .reference import canonical_candidates, compact_topk


def try_compact_scores(
    q,
    q_scale,
    keys,
    key_scales,
    weights,
    candidates,
    block_size,
    starts,
    ends,
    topk,
    *,
    block_table=None,
    page_size=0,
    next_n=1,
):
    """Return logical top-k positions, or None before launching unsupported work."""
    if not eligible_tensor(q, "compact_indexer"):
        return None
    if q_scale is None or candidates is None or q.dtype != torch.uint8:
        return None
    if candidates.ndim != 2:
        return None
    if q.ndim != 3 or q.shape[-1] != 64 or not 1 <= q.shape[1] <= 64:
        return None
    rows, heads, _ = q.shape
    width = candidates.shape[1] * block_size
    if width * rows > get_options().indexer_max_scores or block_size < 1:
        return None
    if candidates.shape[0] != rows or q_scale.shape != (rows, heads, 4):
        return None
    if starts.numel() != rows or ends.numel() != rows:
        return None
    if weights.shape != (rows, heads) or weights.stride(-1) != 1:
        return None
    if keys.dtype != torch.uint8 or q_scale.element_size() != 1:
        return None
    if block_table is None:
        if key_scales is None or key_scales.dtype != torch.uint8:
            return None
        if (
            keys.ndim != 2
            or keys.shape[1] != 64
            or key_scales.shape != (keys.shape[0], 4)
            or key_scales.stride(-1) != 1
        ):
            return None
    else:
        if keys.ndim != 4 or keys.shape[2:] != (1, 68):
            return None
        if page_size != keys.shape[1] or next_n < 1 or rows % next_n:
            return None
        if block_table.shape[0] < rows // next_n or block_table.stride(-1) != 1:
            return None
    if q.stride(-1) != 1 or q_scale.stride(-1) != 1 or keys.stride(-1) != 1:
        return None
    from .kernels import compact_mxfp4_scores

    record_path("compact_indexer")
    positions = canonical_candidates(candidates, block_size)
    scores = compact_mxfp4_scores(
        q,
        q_scale.view(torch.uint8),
        keys,
        key_scales,
        weights,
        positions,
        starts.contiguous(),
        ends.contiguous(),
        block_table=block_table,
        page_size=page_size,
        next_n=next_n,
    )
    return compact_topk(scores, positions, topk)
