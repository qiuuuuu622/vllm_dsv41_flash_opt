# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scoped, opt-in H20/vLLM 0.32.0 Humming finalizer replacement.

Shared fusion remains an explicit function-level experiment in aligned_ops;
this adapter does not move shared experts or scaling across EP collectives.
Use in a single-threaded benchmark/worker setup, not around concurrent requests.
"""

import contextlib
import importlib.metadata

import torch

from . import ops


@contextlib.contextmanager
def use_aligned_humming_finalize(max_tokens=128):
    """Use tiled finalizers for measured small batches; restore on exit.

    Large batches keep upstream kernels because the standalone tiled
    replacement regressed slightly there in the two H20 measurements.
    """
    if importlib.metadata.version("vllm") != "0.32.0":
        raise RuntimeError("This experimental adapter targets vLLM 0.32.0.")
    if importlib.metadata.version("humming-kernels") != "0.1.17":
        raise RuntimeError("This experimental adapter targets Humming 0.1.17.")
    if "H20" not in torch.cuda.get_device_name():
        raise RuntimeError("The dispatch threshold was measured on H20 only.")
    from vllm.model_executor.layers.fused_moe.experts import fused_humming_moe as hm

    original_sum, original_unpermute = hm.moe_fused_mul_sum, hm.moe_unpermute

    def indexed(*args, **kwargs):
        inputs = kwargs.get("inputs", args[0] if args else None)
        implementation = (
            ops.moe_fused_mul_sum if inputs.shape[0] <= max_tokens else original_sum
        )
        return implementation(*args, **kwargs)

    def grouped(*args, **kwargs):
        output = kwargs.get("out", args[0] if args else None)
        implementation = (
            ops.moe_unpermute if output.shape[0] <= max_tokens else original_unpermute
        )
        return implementation(*args, **kwargs)

    hm.moe_fused_mul_sum, hm.moe_unpermute = indexed, grouped
    try:
        yield
    finally:
        hm.moe_fused_mul_sum, hm.moe_unpermute = original_sum, original_unpermute
