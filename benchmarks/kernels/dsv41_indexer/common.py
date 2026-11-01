# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measurement helpers. Imported only by explicitly requested GPU runs."""

import hashlib
import json
import statistics
import sys
from importlib.metadata import distribution
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def check_sources():
    return json.loads((ROOT / "results/indexer/20261101/provenance.json").read_text())


def check_vllm():
    # A checkout's vllm/ would otherwise shadow the installed baseline wheel.
    # Import the actual installed package, never replace an already loaded one.
    installed = Path(distribution("vllm").locate_file("vllm")).resolve()
    sys.path.insert(0, str(installed.parent))
    try:
        import vllm
    finally:
        sys.path.pop(0)
    if Path(vllm.__file__).resolve().parent != installed:
        raise RuntimeError("Start a fresh process without a different vllm import")

    assert vllm.__version__.split("+")[0] == "0.32.0", vllm.__version__
    root = Path(vllm.__file__).resolve().parent.parent
    manifest = check_sources()
    # A version string alone is insufficient: reject edited baseline wrappers.
    for item in manifest["files"]:
        if item["path"].startswith("snapshots/vllm_"):
            actual = root / item["source"]
            assert actual.is_file(), actual
            assert hashlib.sha256(actual.read_bytes()).hexdigest() == item["sha256"], (
                f"Baseline differs: {actual}"
            )
    return str(root)


def device_info():
    import subprocess

    import torch

    p = torch.cuda.get_device_properties(torch.accelerator.current_device_index())
    assert p.major == 9 and p.minor == 0 and "H20" in p.name, p
    try:
        topology = subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True)
    except (OSError, subprocess.CalledProcessError):
        topology = "unavailable"
    return dict(
        name=p.name,
        sm_count=p.multi_processor_count,
        topology=topology,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        memory_bytes=p.total_memory,
    )


def measure(fn, *, repeats=30, inner=10, graph=False, distributed=False):
    import torch
    import torch.distributed as dist

    for _ in range(20):
        fn()
    torch.accelerator.synchronize()
    if distributed:
        dist.barrier()
    call = fn
    if graph:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(inner):
                fn()
        call = g.replay
    samples = []
    for _ in range(repeats):
        if distributed:
            dist.barrier()
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        if graph:
            call()
        else:
            for _ in range(inner):
                call()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / inner)
    return dict(
        p50_us=statistics.median(samples),
        p95_us=sorted(samples)[int(0.95 * (len(samples) - 1))],
        samples_us=samples,
    )


def save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
