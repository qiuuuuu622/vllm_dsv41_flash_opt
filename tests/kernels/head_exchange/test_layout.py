# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU oracle for the byte-address contract, including eight independent ranks.

No GPU timing is simulated. The test guards rank/head permutations, uneven row
padding, and inverse placement, which would silently corrupt attention outputs.
"""

import pytest
import torch

from vllm_dsv41_head_exchange.layout import HeadRowLayout


def push_segments(inputs, plan, direction):
    """Execute the CUDA ABI's byte strides with CPU copies as a layout oracle."""
    q_direction = direction == "q"
    output_rows = plan.chunk if q_direction else plan.world * plan.chunk
    outputs = [x.new_zeros(output_rows, 1, 64, 512) for x in inputs]
    src_peer, src_row, dst_rank, dst_row = plan.strides(direction)
    valid_rows = plan.rows if q_direction else plan.world * plan.chunk
    for sender, tensor in enumerate(inputs):
        src = tensor.view(torch.uint8).flatten()
        for receiver, out in enumerate(outputs):
            dst = out.view(torch.uint8).flatten()
            for row in range(plan.chunk):
                start = sender * dst_rank + row * dst_row
                if receiver * plan.chunk + row < valid_rows:
                    offset = receiver * src_peer + row * src_row
                    dst[start : start + plan.segment_bytes].copy_(
                        src[offset : offset + plan.segment_bytes]
                    )
    return outputs


@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("rows", [1, 7, 8, 9, 17, 129])
def test_raw_bits_reach_the_correct_row_and_head_owner(world, rows):
    plan = HeadRowLayout(rows, world)
    gen = torch.Generator().manual_seed(rows * 10 + world)
    inputs = [
        torch.randint(
            -32768, 32768, (rows, 1, 64, 512), dtype=torch.int16, generator=gen
        )
        for _ in range(world)
    ]
    row_q = push_segments(inputs, plan, "q")
    for receiver, out in enumerate(row_q):
        expected = torch.zeros_like(out)
        start, stop = plan.row_range(receiver)
        if stop > start:
            expected[: stop - start, 0, : plan.active_heads] = torch.cat(
                [x[start:stop, 0, :8] for x in inputs], dim=1
            )
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
    restored = push_segments(row_q, plan, "o")
    for rank, out in enumerate(restored):
        torch.testing.assert_close(
            out[:rows, :, :8], inputs[rank][:, :, :8], rtol=0, atol=0
        )
        assert torch.count_nonzero(out[:, :, 8:]) == 0
        assert torch.count_nonzero(out[rows:]) == 0


@pytest.mark.parametrize("world", [2, 8])
def test_tp_attention_with_sinks_and_empty_windows_matches_unexchanged(world):
    torch.manual_seed(42)
    rows, keys = 9, 13
    plan = HeadRowLayout(rows, world)
    q = [torch.randn(rows, 1, 64, 512, dtype=torch.bfloat16) for _ in range(world)]
    row_q = push_segments(q, plan, "q")
    kv = torch.randn(rows, keys, 512, dtype=torch.float64)
    visible = torch.arange(keys)[None, :] < (torch.arange(rows) % keys)[:, None]
    sinks = torch.randn(world, 8, dtype=torch.float64)

    def attention(query, cache, mask, sink):
        scores = torch.einsum("mhd,mtd->mht", query.double(), cache) / (512**0.5)
        scores.masked_fill_(~mask[:, None, :], -torch.inf)
        p = torch.softmax(
            torch.cat([scores, sink.expand(scores.shape[0], -1, 1)], -1), -1
        )
        return torch.einsum("mht,mtd->mhd", p[..., :-1], cache).to(torch.bfloat16)

    # Return independent attention outputs, not just a Q roundtrip.
    row_o = [torch.zeros_like(x) for x in row_q]
    for rank in range(world):
        start, stop = plan.row_range(rank)
        n = stop - start
        if n:
            row_o[rank][:n, 0, : plan.active_heads] = attention(
                row_q[rank][:n, 0, : plan.active_heads],
                kv[start:stop],
                visible[start:stop],
                sinks.flatten()[:, None],
            )
    restored = push_segments(row_o, plan, "o")
    for rank in range(world):
        expected = attention(q[rank][:, 0, :8], kv, visible, sinks[rank, :, None])
        torch.testing.assert_close(
            restored[rank][:rows, 0, :8], expected, rtol=0, atol=0
        )


def test_partial_metadata_is_refreshed_including_empty_rank():
    plan = HeadRowLayout(9, 8)
    indices = torch.arange(27).view(9, 3)
    torch.testing.assert_close(
        plan.shard(indices, 4, -1), torch.tensor([[24, 25, 26], [-1, -1, -1]])
    )
    assert (plan.shard(indices, 7, -1) == -1).all()
    indices.add_(100)
    assert plan.shard(indices, 4, -1)[0, 0] == 124


@pytest.mark.parametrize("rows,world", [(0, 8), (10, 3), (1 << 25, 8)])
def test_reject_unsupported_exchange_configuration(rows, world):
    with pytest.raises(ValueError):
        HeadRowLayout(rows, world)
