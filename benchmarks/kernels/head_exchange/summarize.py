# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare measured function latency; never infer TP8 numbers from TP2."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fused", type=Path, required=True)
    parser.add_argument("--nccl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fused, nccl = [json.loads(p.read_text()) for p in (args.fused, args.nccl)]
    for key in ("physical_ranks", "input_heads_per_rank", "exchanged_heads_per_rank"):
        if fused[key] != nccl[key]:
            raise ValueError(f"Cannot compare different {key}")
    if fused["status"] != "completed" or nccl["status"] != "completed":
        raise ValueError("Both measurements must be completed")
    original = {(c["rows"], c["compress_ratio"]): c for c in nccl["cases"]}
    rows = []

    def us(case, path, mode):
        return max(r["timings"][path][mode]["p50_us"] for r in case["ranks"])

    for case in fused["cases"]:
        old = original[(case["rows"], case["compress_ratio"])]
        if case["ranks"][0]["context"] != old["ranks"][0]["context"]:
            raise ValueError("Context lengths differ")
        for mode in ("eager", "graph"):
            baseline = us(case, "baseline", mode)
            optimized = us(case, "exchanged_total", mode)
            rows.append(
                {
                    "physical_ranks": fused["physical_ranks"],
                    "rows": case["rows"],
                    "ratio": case["compress_ratio"],
                    "mode": mode,
                    "baseline_us": baseline,
                    "nccl_total_us": us(old, "exchanged_total", mode),
                    "fused_total_us": optimized,
                    "nccl_transport_us": us(old, "exchange_and_packing_only", mode),
                    "fused_transport_us": us(case, "exchange_and_packing_only", mode),
                    "speedup": baseline / optimized,
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
