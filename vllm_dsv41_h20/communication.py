# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Alpha/bandwidth estimates, explicitly not measurements of NVLink."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LinkModel:
    effective_gbps: float  # Decimal GB/s (bytes, not bits); supplied assumption.
    startup_us: float
    world_size: int = 8

    def __post_init__(self):
        if self.effective_gbps <= 0 or self.startup_us < 0 or self.world_size < 1:
            raise ValueError("invalid communication assumptions")

    def all_gather_us(self, full_bytes: int) -> float:
        """Ring model: (P-1) rounds, (P-1)/P of full result per rank."""
        if full_bytes < 0:
            raise ValueError("bytes must be nonnegative")
        p = self.world_size
        return (p - 1) * self.startup_us + full_bytes * (p - 1) / p / (
            self.effective_gbps * 1e3
        )

    def all_to_all_us(self, send_bytes_per_rank: int) -> float:
        """Pairwise model; excludes local rank's self-copy."""
        return self.all_gather_us(send_bytes_per_rank)


def projected_gather_bytes(rows: int, output_width: int, world_size=8) -> int:
    if rows < 0 or output_width < 1 or world_size < 1:
        raise ValueError("invalid projection dimensions")
    padded = ((rows + world_size - 1) // world_size) * world_size
    return padded * output_width * 4  # FP32 compressor output


def head_exchange_bytes(rows: int, local_heads: int, width: int, world_size=8) -> int:
    if rows < 0 or local_heads < 1 or width < 1 or world_size < 1:
        raise ValueError("invalid attention dimensions")
    padded = ((rows + world_size - 1) // world_size) * world_size
    return padded * local_heads * width * 2  # one BF16 exchange, per rank


def break_even_us(original_us: float, local_us: float, extra_us=0.0) -> float:
    """Maximum exposed communication time before an optimization loses."""
    return original_us - local_us - extra_us
