# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Byte-layout contract shared by the CUDA transport and portable checks."""

from dataclasses import dataclass


@dataclass(frozen=True)
class HeadRowLayout:
    """Keep eight input heads per rank, including for the TP2 subproblem."""

    rows: int
    world: int

    def __post_init__(self):
        if self.world not in (2, 4, 8) or not 0 < self.rows <= 1 << 24:
            raise ValueError("Expected world=2/4/8 and 1 <= rows <= 2**24")

    @property
    def chunk(self):
        return (self.rows + self.world - 1) // self.world

    @property
    def active_heads(self):
        return 8 * self.world

    @property
    def row_bytes(self):
        return 64 * 512 * 2

    @property
    def segment_bytes(self):
        return 8 * 512 * 2

    def strides(self, direction):
        """Return source peer/row and destination sender/row byte strides."""
        if direction == "q":
            return (
                self.chunk * self.row_bytes,
                self.row_bytes,
                self.segment_bytes,
                self.row_bytes,
            )
        if direction == "o":
            return (
                self.segment_bytes,
                self.row_bytes,
                self.chunk * self.row_bytes,
                self.row_bytes,
            )
        raise ValueError(f"Unknown exchange direction: {direction}")

    def row_range(self, rank):
        if not 0 <= rank < self.world:
            raise ValueError("Rank outside process group")
        start = min(rank * self.chunk, self.rows)
        return start, min((rank + 1) * self.chunk, self.rows)

    def shard(self, tensor, rank, invalid=0):
        """Slice replicated row metadata, padding the final shards if needed.

        Call again after changing metadata when a padded copy was necessary.
        This method does not maintain a cache of request-dependent metadata.
        """
        if tensor is None:
            return None
        if tensor.shape[0] != self.rows:
            raise ValueError("Metadata must have one entry per global query row")
        start, stop = self.row_range(rank)
        if stop - start == self.chunk:
            return tensor[start:stop]
        out = tensor.new_full((self.chunk, *tensor.shape[1:]), invalid)
        if stop > start:
            out[: stop - start].copy_(tensor[start:stop])
        return out
