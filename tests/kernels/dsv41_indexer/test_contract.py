# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard TP8's replicated-index-head contract and preserved upstream kernels.

The cheapest check for accidental head partitioning is a CPU contract test;
actual CUDA/cache mapping checks live in the standalone GPU verifier.
"""

import hashlib
import json
from pathlib import Path

import pytest

from vllm_dsv41_indexer import IndexerParallel


@pytest.mark.parametrize("world", [1, 2, 4, 8])
def test_every_tp_rank_retains_all_32_index_heads(world):
    for rank in range(world):
        contract = IndexerParallel(world, rank)
        assert contract.local_index_heads == 32
        contract.check_query_shape((6, 32, 128))
        # Main-attention eight-head sharding must never reach this indexer.
        for heads in (8, 32 // 8, 64):
            with pytest.raises(ValueError):
                contract.check_query_shape((6, heads, 128))


@pytest.mark.parametrize("pcp,dcp", [(8, 1), (1, 8), (2, 2)])
def test_reject_cp_sharded_cache_contract(pcp, dcp):
    with pytest.raises(ValueError, match="PCP1 and DCP1"):
        IndexerParallel(8, 0, pcp, dcp)


@pytest.mark.parametrize("world,rank", [(8, 8), (8, -1), (3, 0)])
def test_reject_rank_outside_supported_tp_group(world, rank):
    with pytest.raises(ValueError):
        IndexerParallel(world, rank)


def test_vendored_topk_and_dependencies_match_pinned_ant_sources():
    root = Path(__file__).resolve().parents[3] / "vllm_dsv41_indexer/third_party/ant"
    manifest = json.loads((root / "sources.json").read_text())
    for file in manifest["files"]:
        assert (
            hashlib.sha256((root / file["path"]).read_bytes()).hexdigest()
            == file["sha256"]
        )
