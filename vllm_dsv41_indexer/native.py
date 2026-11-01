# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile only the vendored compact top-k and publisher; no SGLang imports."""

import hashlib
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def load_kernels():
    """Compile/load SM90 kernels outside graph capture, cached by source hash."""
    import torch
    from tvm_ffi.cpp import load_inline

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("The compact FP8 indexer requires SM90 CUDA")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Initialize the indexer before CUDA Graph capture")
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.cuh")) + sorted(root.rglob("*.h")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    source = """
#include <tvm/ffi/function.h>
#include <deepseek_v4/topk_v2.cuh>
#include <candidate_publish.cuh>
TVM_FFI_DLL_EXPORT_TYPED_FUNC(publish, sglang::CandidatePublish::run);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(topk, sglang::TopKKernel<true>::transform_paged);
"""
    flags = [
        "-std=c++20",
        "-O3",
        "--expt-relaxed-constexpr",
        "-gencode=arch=compute_90a,code=sm_90a",
        "-DSGL_CUDA_ARCH=900",
        "-DSGL_TOPK_DET_TIES=1",
        "-DSGL_TOPK_V2_MAX_C8_OCC2=0",
        "-DSGL_TOPK_V2_MAX_C16_OCC1=0",
    ]
    # At <=16384 scores the unchanged Ant dispatcher uses register top-k,
    # never a cluster. Disabling unused clusters removes occupancy probes and
    # plan kernels without changing compact top-k selection or arithmetic.
    digest.update((source + repr(flags)).encode())
    return load_inline(
        "dsv41_compact_" + digest.hexdigest()[:16],
        cuda_sources=source,
        extra_cuda_cflags=flags,
        extra_include_paths=[
            str(root / "csrc"),
            str(root / "third_party/ant/csrc"),
            str(root / "third_party/ant/include"),
        ],
    )


if __name__ == "__main__":
    load_kernels()
    print("Compact FP8 indexer CUDA kernels ready")
