# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Process-static options. Configure before constructing model workers."""

import os
from dataclasses import dataclass
from functools import cache, lru_cache

FEATURES = frozenset(
    {"compact_indexer", "small_m_gemm", "compressor_shard", "head_exchange"}
)


@dataclass(frozen=True)
class H20Options:
    features: frozenset[str] = frozenset()
    small_m_max: int = 16
    split_k: int = 4
    compressor_min_rows: int = 2048
    head_min_rows: int = 128
    indexer_max_scores: int = 16 * 1024 * 1024

    @classmethod
    def from_env(cls, env=None):
        env = os.environ if env is None else env
        names = frozenset(
            n.strip() for n in env.get("VLLM_DSV41_H20_OPT", "").split(",") if n.strip()
        )
        unknown = names - FEATURES
        if unknown:
            raise ValueError(f"Unknown H20 optimizations: {sorted(unknown)}")
        values = {}
        for field, suffix in (
            ("small_m_max", "SMALL_M_MAX"),
            ("split_k", "SPLIT_K"),
            ("compressor_min_rows", "COMPRESSOR_MIN_ROWS"),
            ("head_min_rows", "HEAD_MIN_ROWS"),
            ("indexer_max_scores", "INDEXER_MAX_SCORES"),
        ):
            value = env.get(f"VLLM_DSV41_H20_{suffix}")
            if value is not None:
                values[field] = int(value)
                if values[field] < 1:
                    raise ValueError(f"{suffix} must be positive")
        if values.get("split_k", 4) not in (1, 2, 4, 8):
            raise ValueError("SPLIT_K must be 1, 2, 4 or 8")
        if values.get("small_m_max", 16) > 16:
            raise ValueError("SMALL_M_MAX must be at most 16")
        return cls(features=names, **values)

    def enabled(self, name: str) -> bool:
        return name in self.features


@lru_cache(maxsize=1)
def get_options() -> H20Options:
    return H20Options.from_env()


@cache
def is_h20(device_index: int) -> bool:
    import torch

    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability(device_index) == (9, 0)
        and "H20" in torch.cuda.get_device_name(device_index)
    )


def eligible_tensor(tensor, feature: str) -> bool:
    return (
        get_options().enabled(feature)
        and tensor.device.type == "cuda"
        and is_h20(tensor.device.index or 0)
    )


def supported_topology(parallel) -> bool:
    return (
        parallel.tensor_parallel_size == 8
        and parallel.pipeline_parallel_size == 1
        and parallel.data_parallel_size == 1
        and parallel.prefill_context_parallel_size == 1
        and parallel.decode_context_parallel_size == 1
    )


def validate_model_config(config):
    """Experimental hooks require eager execution until graph validation exists."""
    if get_options().features and not config.model_config.enforce_eager:
        raise ValueError("VLLM_DSV41_H20_OPT currently requires --enforce-eager")


@cache
def record_path(name: str):
    """One activation log per process/feature, rather than logging every token."""
    from vllm.logger import init_logger

    init_logger(__name__).info("Experimental DSv4.1 H20 path active: %s", name)
