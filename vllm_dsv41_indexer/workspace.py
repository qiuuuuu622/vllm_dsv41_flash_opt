# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rank-local FP8 candidate state and consumer, with graph-stable workspaces.

Call publish after source candidate generation on every decode step, and share
that state only among consumers with identical row order, lengths and ratio=1.
Tensor contents may change; storage/shape/device must remain stable during graph
replay. Each workspace is used serially on one stream. Outputs are borrowed.
"""

import torch

from .native import load_kernels
from .parallel import IndexerParallel


def _tensor(tensor, shape, dtype, device, name):
    if tuple(tensor.shape) != tuple(shape) or tensor.dtype != dtype:
        raise ValueError(f"{name} must have shape {shape} and dtype {dtype}")
    if tensor.device != device or not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous on {device}")


class CandidateState:
    """Publish candidate IDs once per step for same-ratio local consumers.

    source_unique=True is valid for vLLM select_candidate_blocks outputs.
    Arbitrary external lists must use the default duplicate-removal path.
    No state or physical page IDs are shared across TP ranks.
    """

    def __init__(self, rows, device, *, parallel=None, compress_ratio=1):
        self.parallel = parallel or IndexerParallel()
        self.parallel.check_query_shape((rows, 32, 128))
        if compress_ratio != 1:
            raise ValueError("This module supports ratio-1 candidate consumers only")
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("CandidateState requires a CUDA device")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.accelerator.current_device_index())
        if self.device.index != torch.accelerator.current_device_index():
            raise ValueError("Set the rank's current CUDA device before construction")
        self.rows = rows
        self.native = load_kernels()
        self.blocks = torch.empty((rows, 2048), device=self.device, dtype=torch.int32)
        self.counts = torch.empty(rows, device=self.device, dtype=torch.int32)
        # Compact top-k never reads a cluster plan. Keep a correctly sized
        # zeroed buffer for the unchanged Ant entry point's metadata contract.
        self.plan = torch.zeros((rows + 1, 2), device=self.device, dtype=torch.int32)
        self._published = False

    def publish(self, blocks, lengths, *, source_unique=False):
        _tensor(blocks, (self.rows, 2048), torch.int32, self.device, "blocks")
        _tensor(lengths, (self.rows,), torch.int32, self.device, "lengths")
        if source_unique:
            self.native.publish(blocks, lengths, self.blocks, self.counts)
        else:
            from .prepare import _canonical_scan

            _canonical_scan[(self.rows,)](
                blocks,
                lengths,
                self.blocks,
                self.counts,
                2048,
                8,
                2048,
                num_warps=8,
            )
        self._published = True


class CompactIndexer:
    """Bind one consumer to rank-local Q/weights/cache and request page tables.

    Q is E4M3 [rows,32,128]; FP32 weights already include query scales. Cache is
    the vLLM FP8 byte ABI [pages,64,1,132]: within each page, 64*128 FP8 payload
    bytes followed by 64 FP32 scales. Finite visible scores are required.
    Request IDs and page-table entries must address the provided local cache.
    """

    def __init__(self, state, q, weights, cache, block_table, request_indices):
        self.state = state
        state.parallel.check_query_shape(q.shape)
        _tensor(q, (state.rows, 32, 128), torch.float8_e4m3fn, state.device, "Q")
        _tensor(weights, (state.rows, 32), torch.float32, state.device, "weights")
        if cache.ndim != 4 or cache.shape[0] <= 0:
            raise ValueError("Cache must have shape [pages,64,1,132]")
        _tensor(cache, (cache.shape[0], 64, 1, 132), torch.uint8, state.device, "cache")
        if block_table.ndim != 2 or min(block_table.shape) <= 0:
            raise ValueError("Page table must be [requests,max_pages]")
        _tensor(block_table, block_table.shape, torch.int32, state.device, "page table")
        _tensor(
            request_indices, (state.rows,), torch.int32, state.device, "request IDs"
        )
        self.q, self.weights, self.cache = q, weights, cache
        self.table, self.requests = block_table, request_indices
        self.scores = torch.empty(
            (state.rows, 16384), device=state.device, dtype=torch.float32
        )
        self.output = torch.empty(
            (state.rows, 512), device=state.device, dtype=torch.int32
        )

    def forward(self):
        """Return unordered logical top-k positions; pad underfilled rows with -1."""
        if not self.state._published:
            raise RuntimeError("Publish candidates before scoring a consumer")
        from .score import candidate_logits

        candidate_logits(
            self.q,
            self.weights,
            self.table,
            self.requests,
            self.state.blocks,
            self.state.counts,
            self.cache,
            64,
            self.scores,
            cache_page_size=64,
            candidate_block_size=8,
            fp32_scores=True,
        )
        self.state.native.topk(
            self.scores,
            self.state.counts,
            self.state.blocks,
            self.output,
            8,
            self.state.plan,
            None,
        )
        return self.output
