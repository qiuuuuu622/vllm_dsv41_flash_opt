# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize completed runs without mixing single-consumer and four-layer costs."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    groups = defaultdict(list)
    for filename in args.inputs:
        data = json.loads(Path(filename).read_text())
        if data["status"] != "completed":
            raise ValueError(f"Incomplete benchmark: {filename}")
        for case in data["cases"]:
            shape = case["case"]
            for mode, measurements in case["timings"].items():
                for name, timing in measurements.items():
                    key = (
                        shape["rows"],
                        shape["compressed_length"],
                        shape["rows_per_request"],
                        mode,
                        name,
                    )
                    groups[key].append(timing["p50_us"])
    rows = []
    for (n, context, verify, mode, name), samples in sorted(groups.items()):
        baseline = (
            "four_vllm_consumers" if name.startswith("four_") else "vllm_original"
        )
        original = statistics.mean(groups[(n, context, verify, mode, baseline)])
        latency = statistics.mean(samples)
        rows.append(
            dict(
                query_rows=n,
                requests=n // verify,
                context=context,
                verify_rows=verify,
                mode=mode,
                variant=name,
                run_count=len(samples),
                mean_run_p50_us=latency,
                speedup_vs_matching_baseline=original / latency,
            )
        )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
