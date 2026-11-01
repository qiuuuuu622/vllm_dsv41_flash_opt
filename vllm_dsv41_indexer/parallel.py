# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parallel contract for the replicated vLLM V4.1 indexer."""

from dataclasses import dataclass


@dataclass(frozen=True)
class IndexerParallel:
    """TP ranks each keep all index heads and their own physical cache mapping.

    EP does not partition the indexer. CP-sharded keys are unsupported: every
    rank must hold the complete logical context addressed by its local table.
    """

    tp_size: int = 1
    tp_rank: int = 0
    pcp_size: int = 1
    dcp_size: int = 1

    def __post_init__(self):
        if self.tp_size not in (1, 2, 4, 8) or not 0 <= self.tp_rank < self.tp_size:
            raise ValueError("Expected TP1/2/4/8 and a rank inside that TP group")
        if self.pcp_size != 1 or self.dcp_size != 1:
            raise ValueError("Compact FP8 indexer requires PCP1 and DCP1")

    @property
    def local_index_heads(self):
        return 32

    def check_query_shape(self, shape):
        if len(shape) != 3 or shape[0] <= 0 or tuple(shape[1:]) != (32, 128):
            raise ValueError("Each TP rank requires Q [rows, 32, 128]")

    @classmethod
    def from_group(cls, tp_group, *, pcp_size=1, dcp_size=1):
        """Read the supplied TP group's local rank; no collective is issued."""
        import torch.distributed as dist

        return cls(
            dist.get_world_size(tp_group),
            dist.get_rank(tp_group),
            pcp_size,
            dcp_size,
        )
