# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.triton_utils import tl, triton


@triton.jit
def _canonical_scan(
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
    length = tl.maximum(tl.load(LENS + row), 0)
    b = tl.load(BLOCKS + row * K + i, i < K, -1)
    sentinel = 2147483647
    b = tl.sort(
        tl.where((i < K) & (b >= 0) & (b < tl.cdiv(length, BLOCK)), b, sentinel)
    )
    previous = tl.gather(b, tl.maximum(i - 1, 0), 0)
    unique = (b != sentinel) & ((i == 0) | (b != previous))
    rank = tl.cumsum(unique.to(tl.int32)) - 1
    num_unique = tl.sum(unique.to(tl.int32), 0)
    tl.store(SORTED + row * K + i, sentinel, (i < K) & (i >= num_unique))
    tl.store(SORTED + row * K + rank, b, unique)
    n = tl.where(unique, tl.minimum(tl.maximum(length - b * BLOCK, 0), BLOCK), 0)
    tl.store(COUNTS + row, tl.sum(n, 0))
