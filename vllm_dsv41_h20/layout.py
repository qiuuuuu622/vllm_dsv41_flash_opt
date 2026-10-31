# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rank-uniform layouts used by real collectives and single-device simulation."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RowPartition:
    rows: int
    world_size: int

    def __post_init__(self):
        if self.rows < 0 or self.world_size < 1:
            raise ValueError("rows must be nonnegative and world_size positive")

    @property
    def chunk(self) -> int:
        return (self.rows + self.world_size - 1) // self.world_size

    def bounds(self, rank: int) -> tuple[int, int]:
        if not 0 <= rank < self.world_size:
            raise ValueError("rank outside partition")
        start = rank * self.chunk
        return min(start, self.rows), min(start + self.chunk, self.rows)

    def shard(self, tensor: torch.Tensor, rank: int, fill_value=0) -> torch.Tensor:
        if tensor.shape[0] != self.rows:
            raise ValueError("tensor does not match partition rows")
        lo, hi = self.bounds(rank)
        result = tensor.new_full((self.chunk, *tensor.shape[1:]), fill_value)
        result[: hi - lo].copy_(tensor[lo:hi])
        return result


def pack_heads_to_rows(q: torch.Tensor, world_size: int) -> torch.Tensor:
    """Pack `[M,h,D]` into destination-major `[P,ceil(M/P),h,D]`."""
    if q.ndim != 3:
        raise ValueError("Q must be [rows, local_heads, width]")
    plan = RowPartition(q.shape[0], world_size)
    padded = q.new_zeros((plan.chunk * world_size, *q.shape[1:]))
    padded[: q.shape[0]].copy_(q)
    return padded.reshape(world_size, plan.chunk, *q.shape[1:])


def unpack_heads_to_rows(received: torch.Tensor) -> torch.Tensor:
    """Source-major receive buffer -> local rows with globally ordered heads."""
    world, rows, heads, width = received.shape
    return received.permute(1, 0, 2, 3).reshape(rows, world * heads, width)


def pack_rows_to_heads(output: torch.Tensor, world_size: int) -> torch.Tensor:
    rows, heads, width = output.shape
    if heads % world_size:
        raise ValueError("heads must be divisible by world size")
    return (
        output.reshape(rows, world_size, heads // world_size, width)
        .permute(1, 0, 2, 3)
        .contiguous()
    )


def unpack_rows_to_heads(received: torch.Tensor, rows: int) -> torch.Tensor:
    world, chunk, heads, width = received.shape
    if not 0 <= rows <= world * chunk:
        raise ValueError("invalid unpadded row count")
    return received.reshape(world * chunk, heads, width)[:rows].contiguous()


def simulate_all_to_all(sent: list[torch.Tensor]) -> list[torch.Tensor]:
    """Logical exchange only. This is NOT a communication timing model."""
    world = len(sent)
    if not world or any(x.shape != sent[0].shape or x.shape[0] != world for x in sent):
        raise ValueError("one equally shaped destination-major buffer per rank")
    return [
        torch.stack([sent[src][dst] for src in range(world)]) for dst in range(world)
    ]


def sharded_projection_reference(x, weight, world_size: int):
    plan = RowPartition(x.shape[0], world_size)
    shards = [plan.shard(x, r).float() @ weight.float().T for r in range(world_size)]
    return torch.cat(shards)[: plan.rows]
