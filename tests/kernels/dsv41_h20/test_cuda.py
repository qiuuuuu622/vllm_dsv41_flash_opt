# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""No checkpoint required. These tests must run on SM90 before deployment."""

import pytest
import torch

from vllm_dsv41_h20.reference import (
    block32_gemm_reference,
    canonical_candidates,
    compact_scores_reference,
    quantize_group32_reference,
    unpack_mxfp4,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="requires an SM90 CUDA GPU; Mac is reference-only",
)


@pytest.mark.parametrize("paged", [False, True])
def test_compact_cuda_scores_with_multiple_requests_and_physical_pages(paged):
    from vllm_dsv41_h20.kernels import compact_mxfp4_scores

    torch.manual_seed(87)
    device = "cuda"
    rows, heads, length, page = 6, 32, 32, 8
    q = torch.randint(0, 256, (rows, heads, 64), dtype=torch.uint8, device=device)
    qs = torch.randint(123, 129, (rows, heads, 4), dtype=torch.uint8, device=device)
    keys = torch.randint(0, 256, (2, length, 64), dtype=torch.uint8, device=device)
    ks = torch.randint(123, 129, (2, length, 4), dtype=torch.uint8, device=device)
    weights = torch.rand(rows, heads, device=device)
    positions = canonical_candidates(
        torch.tensor([[2, 0, 2, -1, 9]] * rows, device=device), 8
    )
    lengths = torch.tensor([0, 19, 32, 4, 23, 31], device=device, dtype=torch.int32)
    starts = torch.tensor(
        [0, 0, 0, length, length, length], device=device, dtype=torch.int32
    )
    expected = torch.cat(
        [
            compact_scores_reference(
                unpack_mxfp4(q[r : r + 3], qs[r : r + 3]),
                unpack_mxfp4(keys[i], ks[i]),
                weights[r : r + 3],
                positions[r : r + 3],
                lengths[r : r + 3],
            )
            for i, r in enumerate((0, 3))
        ]
    )
    if paged:
        table = torch.tensor(
            [[5, 1, 7, 3], [6, 4, 2, 0]], device=device, dtype=torch.int32
        )
        # Extra padding between physical pages exercises non-contiguous cache strides.
        storage = torch.zeros(8, page * 68 + 256, device=device, dtype=torch.uint8)
        cache = storage.as_strided((8, page, 1, 68), (storage.stride(0), 68, 68, 1))
        for request in range(2):
            for logical in range(4):
                block = table[request, logical].item()
                cache[block, :, 0, :64] = keys[
                    request, logical * page : (logical + 1) * page
                ]
                cache[block, :, 0, 64:] = ks[
                    request, logical * page : (logical + 1) * page
                ]
        actual = compact_mxfp4_scores(
            q,
            qs,
            cache,
            None,
            weights,
            positions,
            torch.zeros_like(starts),
            lengths,
            block_table=table,
            page_size=page,
            next_n=3,
        )
    else:
        actual = compact_mxfp4_scores(
            q,
            qs,
            keys.flatten(0, 1),
            ks.flatten(0, 1),
            weights,
            positions,
            starts,
            starts + lengths,
        )
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-3)


@pytest.mark.parametrize("m,k,n", [(1, 160, 65), (8, 512, 128), (16, 5120, 1280)])
@pytest.mark.parametrize("split_k", [1, 4, 8])
def test_small_m_cuda_against_dequantized_reference(m, k, n, split_k):
    from vllm_dsv41_h20.kernels import small_m_block32_gemm

    torch.manual_seed(17)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    weight, sw = quantize_group32_reference(torch.randn(n, k, device="cuda"))
    a, sa = quantize_group32_reference(x)
    scale_bytes = (sw.log2() + 127).to(torch.uint8)
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    expected = (block32_gemm_reference(a, sa, weight, sw) + bias.float()).bfloat16()
    actual = small_m_block32_gemm(x, weight, scale_bytes, split_k=split_k, bias=bias)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=0.0625)
