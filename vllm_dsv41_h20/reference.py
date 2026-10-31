# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference numerics and device-independent candidate/top-k utilities."""

import torch

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def unpack_mxfp4(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    if packed.dtype != torch.uint8 or packed.shape[-1] % 16:
        raise ValueError("packed E2M1 requires uint8 and groups of 16 bytes")
    if scales.shape != (*packed.shape[:-1], packed.shape[-1] // 16):
        raise ValueError("one E8M0 scale per 32 values")
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).long()
    values = torch.tensor(E2M1, device=packed.device)[codes & 7]
    values = torch.where(codes >= 8, -values, values)
    scale = torch.exp2(scales.to(torch.float32) - 127)
    scale = torch.where(scales == 255, float("nan"), scale)
    return values * scale.repeat_interleave(32, dim=-1)


def canonical_candidates(blocks: torch.Tensor, block_size: int) -> torch.Tensor:
    """Sorted logical positions; invalid and duplicate blocks become -1."""
    if blocks.ndim != 2 or block_size < 1:
        raise ValueError("candidate blocks must be [rows, blocks], block size positive")
    ordered = torch.sort(blocks.to(torch.int64), dim=-1).values
    valid = ordered >= 0
    if ordered.shape[1] > 1:
        valid[:, 1:] &= ordered[:, 1:] != ordered[:, :-1]
    offsets = torch.arange(block_size, device=blocks.device)
    positions = ordered[..., None] * block_size + offsets
    return torch.where(valid[..., None], positions, -1).flatten(1)


def compact_scores_reference(q, keys, weights, positions, visible_lens):
    """One request: ReLU(Q K^T), weighted across heads, causal mask."""
    valid = (positions >= 0) & (positions < visible_lens[:, None])
    valid &= positions < keys.shape[0]
    safe = positions.clamp(0, max(0, keys.shape[0] - 1))
    if not keys.shape[0]:
        return weights.new_full(positions.shape, -float("inf"))
    chosen = keys[safe].float()
    dots = torch.einsum("rhd,rcd->rhc", q.float(), chosen)
    scores = (dots.relu() * weights.float()[..., None]).sum(dim=1)
    return scores.masked_fill(~valid, -float("inf"))


def compact_topk(scores, positions, k: int):
    """Position-stable ties; all invalid/non-finite scores return sentinel -1."""
    if k < 1 or scores.shape != positions.shape:
        raise ValueError("positive k and equal score/position shapes required")
    clean = torch.where(torch.isfinite(scores), scores, -float("inf"))
    # canonical_candidates is ordered by position. Stable ranking preserves ties.
    selected = torch.argsort(clean, dim=-1, descending=True, stable=True)
    take = min(k, scores.shape[1])
    selected = selected[:, :take]
    result = positions.gather(1, selected)
    result = result.masked_fill(~torch.isfinite(clean.gather(1, selected)), -1)
    out = torch.full((scores.shape[0], k), -1, dtype=torch.int32, device=scores.device)
    out[:, :take] = result.to(torch.int32)
    return out


def quantize_group32_reference(x):
    if x.shape[-1] % 32:
        raise ValueError("K must be divisible by 32")
    groups = x.float().reshape(*x.shape[:-1], -1, 32)
    maximum = groups.abs().amax(dim=-1).clamp_min(1e-4)
    scales = torch.exp2(torch.ceil(torch.log2(maximum / 448.0)))
    values = (groups / scales[..., None]).clamp(-448, 448)
    return values.reshape(x.shape).to(torch.float8_e4m3fn), scales


def block32_gemm_reference(a, a_scale, w, w_scale, split_k=1):
    """Fixed-order split accumulation; weights use per-row group-32 scales."""
    if split_k < 1 or a.shape[1] != w.shape[1] or a.shape[1] % 32:
        raise ValueError("invalid split-K GEMM dimensions")
    ad = a.float() * a_scale.float().repeat_interleave(32, -1)
    wd = w.float() * w_scale.float().repeat_interleave(32, -1)
    groups = a.shape[1] // 32
    span = (groups + split_k - 1) // split_k
    out = torch.zeros((a.shape[0], w.shape[0]), device=a.device)
    for part in range(split_k):
        lo, hi = part * span * 32, min((part + 1) * span * 32, a.shape[1])
        if lo < hi:
            out += ad[:, lo:hi] @ wd[:, lo:hi].T
    return out
