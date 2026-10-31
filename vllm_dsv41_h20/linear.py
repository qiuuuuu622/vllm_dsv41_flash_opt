# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in adapter for unmodified, group-32 MXFP8 attention linear weights."""

import torch

from .config import eligible_tensor, get_options, record_path, validate_model_config


def retain_small_m_weights(layer, config):
    """Save logical weights before backend repacking; disabled costs no memory."""
    if not eligible_tensor(layer.weight, "small_m_gemm"):
        return
    architectures = config.model_config.hf_config.architectures or []
    if "DeepseekV41ForCausalLM" not in architectures:
        return
    validate_model_config(config)
    # BMM output projections have another layout. MoE kernels are not linear hooks.
    if getattr(layer, "bmm_batch_size", None) is not None:
        return
    if ".attn." not in getattr(layer, "prefix", ""):
        return
    w, s = layer.weight, layer.weight_scale
    if w.ndim != 2 or w.dtype != torch.float8_e4m3fn or w.shape[1] % 32:
        return
    if s.element_size() != 1 or s.shape != (w.shape[0], w.shape[1] // 32):
        return
    layer.register_buffer(
        "_h20_weight", w.detach().clone().contiguous(), persistent=False
    )
    layer.register_buffer(
        "_h20_scale_bytes",
        s.detach().view(torch.uint8).clone().contiguous(),
        persistent=False,
    )


def try_small_m_linear(layer, x, bias):
    if not hasattr(layer, "_h20_weight") or not isinstance(x, torch.Tensor):
        return None
    if x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_contiguous():
        return None
    if not 0 < x.shape[0] <= get_options().small_m_max:
        return None
    from .kernels import small_m_block32_gemm

    record_path("small_m_gemm")
    return small_m_block32_gemm(
        x,
        layer._h20_weight,
        layer._h20_scale_bytes,
        split_k=get_options().split_k,
        bias=bias,
    )
