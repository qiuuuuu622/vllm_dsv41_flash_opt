# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Portable contracts, isolated from vLLM's CUDA-dependent root conftest."""

from types import SimpleNamespace

import pytest
import torch

from vllm_dsv41_h20.communication import (
    LinkModel,
    break_even_us,
    head_exchange_bytes,
    projected_gather_bytes,
)
from vllm_dsv41_h20.config import H20Options, supported_topology, validate_model_config
from vllm_dsv41_h20.indexer import try_compact_scores
from vllm_dsv41_h20.layout import (
    RowPartition,
    pack_heads_to_rows,
    pack_rows_to_heads,
    sharded_projection_reference,
    simulate_all_to_all,
    unpack_heads_to_rows,
    unpack_rows_to_heads,
)
from vllm_dsv41_h20.reference import (
    E2M1,
    block32_gemm_reference,
    canonical_candidates,
    compact_scores_reference,
    compact_topk,
    quantize_group32_reference,
    unpack_mxfp4,
)


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(20261031)
    torch.set_num_threads(1)


def test_mxfp4_all_nibbles_sign_and_scale():
    raw = torch.tensor([i | ((15 - i) << 4) for i in range(16)], dtype=torch.uint8)
    expected = []
    for i in range(16):
        for code in (i, 15 - i):
            expected.append(E2M1[code & 7] * (-1 if code & 8 else 1))
    for exponent in (124, 127, 130):
        actual = unpack_mxfp4(raw, torch.tensor([exponent], dtype=torch.uint8))
        torch.testing.assert_close(
            actual, torch.tensor(expected) * 2 ** (exponent - 127)
        )
    assert unpack_mxfp4(raw, torch.tensor([255], dtype=torch.uint8)).isnan().all()


@pytest.mark.parametrize("rows,length,block,k", [(1, 17, 4, 9), (7, 65, 8, 20)])
def test_compact_matches_independent_full_mask_with_causal_tail(rows, length, block, k):
    q, keys = torch.randn(rows, 3, 32), torch.randn(length, 32)
    weights = torch.rand(rows, 3)
    candidates = torch.tensor([[2, 0, 2, -1, 7, 90]]).repeat(rows, 1)
    positions = canonical_candidates(candidates, block)
    visible = torch.linspace(0, length, rows).long()
    compact = compact_scores_reference(q, keys, weights, positions, visible)
    dense = (torch.einsum("rhd,ld->rhl", q, keys).relu() * weights[..., None]).sum(1)
    mask = torch.zeros(rows, length, dtype=torch.bool)
    for row in range(rows):
        for b in candidates[row].tolist():
            if b >= 0:
                for pos in range(b * block, min((b + 1) * block, visible[row].item())):
                    mask[row, pos] = True
    dense.masked_fill_(~mask, -float("inf"))
    # Compare scores separately: same index selection alone can hide numerical bugs.
    for row in range(rows):
        for col, pos in enumerate(positions[row].tolist()):
            expected = dense[row, pos] if 0 <= pos < length else -float("inf")
            torch.testing.assert_close(compact[row, col], torch.as_tensor(expected))
    dense_positions = torch.arange(length).expand(rows, -1)
    torch.testing.assert_close(
        compact_topk(compact, positions, k),
        compact_topk(dense, dense_positions, k),
        rtol=0,
        atol=0,
    )


def test_compact_ties_duplicates_empty_and_nonfinite():
    positions = canonical_candidates(torch.tensor([[3, 1, -1, 1]]), 2)
    scores = torch.where(positions >= 0, 2.0, -float("inf"))
    assert compact_topk(scores, positions, 6).tolist() == [[2, 3, 6, 7, -1, -1]]
    assert compact_topk(torch.empty(2, 0), torch.empty(2, 0), 3).tolist() == [
        [-1, -1, -1],
        [-1, -1, -1],
    ]
    scores = torch.tensor([[float("nan"), float("inf"), 1.0]])
    assert compact_topk(scores, torch.arange(3).view(1, -1), 3).tolist() == [
        [2, -1, -1]
    ]


@pytest.mark.parametrize("m,k,n", [(1, 32, 7), (3, 160, 65), (16, 512, 128)])
@pytest.mark.parametrize("splits", [1, 2, 4, 8])
def test_group32_split_k_against_independent_float64(m, k, n, splits):
    a, sa = quantize_group32_reference(torch.randn(m, k).bfloat16())
    w, sw = quantize_group32_reference(torch.randn(n, k).bfloat16())
    expected = (a.double() * sa.double().repeat_interleave(32, -1)) @ (
        w.double() * sw.double().repeat_interleave(32, -1)
    ).T
    actual = block32_gemm_reference(a, sa, w, sw, split_k=splits)
    torch.testing.assert_close(actual.double(), expected, rtol=2e-5, atol=3e-5)


@pytest.mark.parametrize("rows", [0, 1, 7, 8, 9, 129])
def test_projection_gather_preserves_uneven_rows(rows):
    x, w = torch.randn(rows, 64).bfloat16(), torch.randn(19, 64).bfloat16()
    actual = sharded_projection_reference(x, w, 8)
    torch.testing.assert_close(actual, x.float() @ w.float().T, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("rows", [0, 1, 7, 8, 9, 129])
def test_head_exchange_preserves_rank_and_head_order(rows):
    global_q = torch.arange(rows * 64 * 4).reshape(rows, 64, 4)
    sent = [pack_heads_to_rows(global_q[:, r * 8 : (r + 1) * 8], 8) for r in range(8)]
    received = [unpack_heads_to_rows(x) for x in simulate_all_to_all(sent)]
    plan = RowPartition(rows, 8)
    for rank, x in enumerate(received):
        torch.testing.assert_close(x, plan.shard(global_q, rank), rtol=0, atol=0)
    returned = simulate_all_to_all([pack_rows_to_heads(x, 8) for x in received])
    for rank, x in enumerate(returned):
        torch.testing.assert_close(
            unpack_rows_to_heads(x, rows),
            global_q[:, rank * 8 : (rank + 1) * 8],
            rtol=0,
            atol=0,
        )


def attention_reference(q, kv, indices, sinks):
    chosen = kv[indices.clamp_min(0)].double()
    logits = torch.einsum("mhd,mld->mhl", q.double(), chosen) / q.shape[-1] ** 0.5
    logits.masked_fill_(indices[:, None] < 0, -float("inf"))
    sink_logits = sinks.double()[None, :, None].expand(q.shape[0], -1, 1)
    probabilities = torch.softmax(torch.cat([logits, sink_logits], -1), -1)[..., :-1]
    return torch.einsum("mhl,mld->mhd", probabilities, chosen)


def test_attention_exchange_with_distinct_sinks_and_per_row_sparse_indices():
    rows, world, dim = 11, 8, 16
    q, kv = torch.randn(rows, 64, dim), torch.randn(23, dim)
    indices = torch.randint(0, 23, (rows, 9))
    indices[:, -2:] = -1
    sinks = torch.randn(64) * 3
    baseline = attention_reference(q, kv, indices, sinks)
    sent = [pack_heads_to_rows(q[:, r * 8 : (r + 1) * 8], world) for r in range(world)]
    row_q = [unpack_heads_to_rows(x) for x in simulate_all_to_all(sent)]
    plan = RowPartition(rows, world)
    outs = [
        attention_reference(x, kv, plan.shard(indices, rank, -1), sinks)
        for rank, x in enumerate(row_q)
    ]
    restored = simulate_all_to_all([pack_rows_to_heads(x, world) for x in outs])
    for rank, x in enumerate(restored):
        torch.testing.assert_close(
            unpack_rows_to_heads(x, rows),
            baseline[:, rank * 8 : (rank + 1) * 8],
            rtol=1e-12,
            atol=1e-12,
        )


def test_communication_units_and_padded_bytes():
    link = LinkModel(effective_gbps=100, startup_us=5, world_size=8)
    assert projected_gather_bytes(9, 1024) == 16 * 1024 * 4
    assert head_exchange_bytes(9, 8, 512) == 16 * 8 * 512 * 2
    assert link.all_gather_us(800_000) == pytest.approx(42.0)
    assert link.all_to_all_us(800_000) == pytest.approx(42.0)
    assert LinkModel(100, 5, 1).all_gather_us(800_000) == 0
    assert break_even_us(120, 50, 10) == 60
    with pytest.raises(ValueError):
        LinkModel(0, 0)


@pytest.mark.parametrize(
    "env",
    [
        {"VLLM_DSV41_H20_OPT": "typo"},
        {"VLLM_DSV41_H20_SPLIT_K": "3"},
        {"VLLM_DSV41_H20_SMALL_M_MAX": "17"},
        {"VLLM_DSV41_H20_HEAD_MIN_ROWS": "0"},
    ],
)
def test_bad_options_fail_at_construction(env):
    with pytest.raises(ValueError):
        H20Options.from_env(env)


def test_disabled_and_unsupported_paths_do_not_import_cuda(monkeypatch):
    from vllm_dsv41_h20 import config

    assert not H20Options.from_env({}).features
    monkeypatch.setattr(config, "get_options", lambda: H20Options())
    validate_model_config(
        SimpleNamespace(model_config=SimpleNamespace(enforce_eager=False))
    )
    assert (
        try_compact_scores(
            torch.zeros(1), None, None, None, None, None, 1, None, None, 1
        )
        is None
    )
    monkeypatch.setattr(
        config, "get_options", lambda: H20Options(frozenset({"head_exchange"}))
    )
    with pytest.raises(ValueError, match="enforce-eager"):
        validate_model_config(
            SimpleNamespace(model_config=SimpleNamespace(enforce_eager=False))
        )


def test_only_supported_collective_topology():
    parallel = SimpleNamespace(
        tensor_parallel_size=8,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
    )
    assert supported_topology(parallel)
    parallel.prefill_context_parallel_size = 8
    assert not supported_topology(parallel)


def test_small_m_retention_survives_backend_weight_repacking(monkeypatch):
    from vllm_dsv41_h20 import linear

    layer = torch.nn.Module()
    layer.prefix = "model.layers.0.attn.wq_b"
    layer.register_buffer("weight", torch.randn(8, 64).to(torch.float8_e4m3fn))
    layer.register_buffer("weight_scale", torch.full((8, 2), 127, dtype=torch.uint8))
    original = layer.weight.float().clone()
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            enforce_eager=True,
            hf_config=SimpleNamespace(architectures=["DeepseekV41ForCausalLM"]),
        )
    )
    monkeypatch.setattr(linear, "eligible_tensor", lambda *_: True)
    linear.retain_small_m_weights(layer, config)
    layer.weight = torch.zeros_like(layer.weight)
    torch.testing.assert_close(layer._h20_weight.float(), original)
    assert "_h20_weight" not in layer.state_dict()
    assert linear.try_small_m_linear(layer, torch.ones(17, 64).bfloat16(), None) is None
