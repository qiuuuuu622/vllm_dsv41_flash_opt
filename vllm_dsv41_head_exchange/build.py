# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicitly compile the standalone SM90 CUDA transport before graph capture."""

import argparse
import hashlib
import os
import shutil
import subprocess
from pathlib import Path

SOURCE = Path(__file__).parent / "csrc/segment_exchange.cu"


def library_path():
    override = os.environ.get("DSV41_HEAD_EXCHANGE_LIBRARY")
    if override:
        return Path(override).expanduser().resolve()
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()[:16]
    return (
        Path.home()
        / ".cache/vllm-dsv41-head-exchange"
        / digest
        / "libsegment_exchange.so"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nvcc", default=shutil.which("nvcc"))
    args = parser.parse_args()
    if not args.nvcc:
        parser.error("nvcc is required; pass --nvcc /usr/local/cuda/bin/nvcc")
    nvcc = Path(args.nvcc).resolve()
    output = library_path()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f".{os.getpid()}.so")
    try:
        subprocess.run(
            [
                str(nvcc),
                "-O3",
                "-std=c++17",
                "-arch=sm_90a",
                "--shared",
                "-Xcompiler",
                "-fPIC",
                "-cudart",
                "shared",
                "-Xlinker",
                "-rpath",
                "-Xlinker",
                str(nvcc.parent.parent / "lib64"),
                str(SOURCE),
                "-o",
                str(temporary),
            ],
            check=True,
        )
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(output)


if __name__ == "__main__":
    main()
