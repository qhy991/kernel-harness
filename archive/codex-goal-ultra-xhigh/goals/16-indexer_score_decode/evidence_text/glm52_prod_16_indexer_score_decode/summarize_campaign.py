#!/usr/bin/env python3
"""Recompute paired summaries from one immutable campaign directory."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()

    rows = []
    grouped = defaultdict(list)
    for path in sorted((run_dir / "results").glob("*.json")):
        payload = json.loads(path.read_text())
        candidate = payload.get("candidate")
        if candidate is None:
            continue
        stem = path.stem
        kind = "cutedsl" if "cutedsl" in stem else "identity"
        mode = "cuda_graph" if stem.startswith("graph_") else "eager"
        batch = payload["workload"]["params"]["batch"]
        paired = candidate["paired_speedups"]
        recomputed = statistics.median(paired)
        if abs(recomputed - candidate["speedup"]) > 1e-12:
            raise AssertionError(f"{path}: paired median does not recompute")
        row = {
            "file": str(path.relative_to(run_dir)),
            "mode": mode,
            "batch": batch,
            "candidate": kind,
            "reference_median_ms": payload["reference"]["median_ms"],
            "candidate_median_ms": candidate["median_ms"],
            "paired_median_speedup": candidate["speedup"],
            "paired_p10_speedup": candidate["paired_p10_speedup"],
            "paired_p90_speedup": candidate["paired_p90_speedup"],
            "pairs": len(paired),
            "passes_3pct": candidate["speedup"] >= 1.03,
            "correctness": all(
                value == "pass"
                for key, value in payload["correctness_details"].items()
                if "candidate" in key
            ),
            "source_runner_sha256": payload["source_provenance"]["runner_sha256"],
            "candidate_sha256": candidate["sha256"],
        }
        rows.append(row)
        grouped[(mode, batch, kind)].append(row)

    aggregates = []
    for (mode, batch, kind), items in sorted(grouped.items()):
        speedups = [item["paired_median_speedup"] for item in items]
        aggregates.append(
            {
                "mode": mode,
                "batch": batch,
                "candidate": kind,
                "series": len(items),
                "series_speedups": speedups,
                "median_series_speedup": statistics.median(speedups),
                "min_series_speedup": min(speedups),
                "max_series_speedup": max(speedups),
                "all_series_pass_3pct": all(value >= 1.03 for value in speedups),
                "all_correct": all(item["correctness"] for item in items),
            }
        )

    payload = {"rows": rows, "aggregates": aggregates}
    (run_dir / "paired_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    with (run_dir / "paired_summary.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    lines = [
        "# Paired indexer score/top-k summary",
        "",
        "All speedups are medians of same-process interleaved per-pair "
        "`reference_ms / candidate_ms` ratios.",
        "",
        "| Mode | M | Candidate | Series speedups | Median | Min | 3% in every series | Correct |",
        "|---|---:|---|---|---:|---:|---|---|",
    ]
    for item in aggregates:
        speedups = ", ".join(f"{value:.6f}x" for value in item["series_speedups"])
        lines.append(
            f"| {item['mode']} | {item['batch']} | {item['candidate']} | "
            f"{speedups} | {item['median_series_speedup']:.6f}x | "
            f"{item['min_series_speedup']:.6f}x | "
            f"{item['all_series_pass_3pct']} | {item['all_correct']} |"
        )
    (run_dir / "paired_summary.md").write_text("\n".join(lines) + "\n")
    print(run_dir / "paired_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
