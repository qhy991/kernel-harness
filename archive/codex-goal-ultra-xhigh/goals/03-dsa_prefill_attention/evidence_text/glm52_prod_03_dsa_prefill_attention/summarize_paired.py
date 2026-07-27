#!/usr/bin/env python3
"""Build the paired benchmark summary from immutable raw JSON outputs."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load_group(prefix: str, count: int) -> list[dict]:
    return [json.loads((HERE / f"{prefix}_{i:02d}.json").read_text()) for i in range(1, count + 1)]


def order_stat(values: list[float], fraction: float, upper: bool = False) -> float:
    ordered = sorted(values)
    index = int(fraction * len(ordered)) - (1 if upper else 0)
    return ordered[min(len(ordered) - 1, max(0, index))]


def summarize(label: str, files: list[dict]) -> dict:
    paired = [value for data in files for value in data["candidate"]["paired_speedups"]]
    return {
        "label": label,
        "runs": len(files),
        "pairs": len(paired),
        "reference_medians_ms": [data["reference"]["median_ms"] for data in files],
        "candidate_medians_ms": [data["candidate"]["median_ms"] for data in files],
        "run_paired_medians": [data["candidate"]["speedup"] for data in files],
        "pooled_paired_p10": order_stat(paired, 0.1),
        "pooled_paired_median": statistics.median(paired),
        "pooled_paired_p90": order_stat(paired, 0.9, upper=True),
        "pooled_min": min(paired),
        "pooled_max": max(paired),
        "passes_3pct_gate": statistics.median(paired) >= 1.03,
    }


def main() -> None:
    rows = [
        summarize(
            "isolated stock-vs-stock noise floor",
            load_group("isolated_baseline", 3),
        ),
        summarize(
            "isolated source-trial PDL-off vs stock",
            load_group("source_trial_pdl_off", 3),
        ),
    ]
    (HERE / "paired_summary.json").write_text(
        json.dumps({"schema_version": 1, "results": rows}, indent=2) + "\n"
    )
    with (HERE / "paired_summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "label",
                "runs",
                "pairs",
                "pooled_p10",
                "pooled_median",
                "pooled_p90",
                "passes_3pct_gate",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["label"],
                    row["runs"],
                    row["pairs"],
                    row["pooled_paired_p10"],
                    row["pooled_paired_median"],
                    row["pooled_paired_p90"],
                    row["passes_3pct_gate"],
                ]
            )
    lines = [
        "# Paired M4096 results",
        "",
        "All rows use the isolated SGLang worktree and interleaved CUDA-event pairs.",
        "The pooled percentiles use the runner's discrete order-statistic convention.",
        "",
        "| Comparison | Runs × pairs | p10 | paired p50 | p90 | 3% gate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {runs} × {pairs_per_run} | {p10:.6f}× | {p50:.6f}× | "
            "{p90:.6f}× | {gate} |".format(
                label=row["label"],
                runs=row["runs"],
                pairs_per_run=row["pairs"] // row["runs"],
                p10=row["pooled_paired_p10"],
                p50=row["pooled_paired_median"],
                p90=row["pooled_paired_p90"],
                gate="PASS" if row["passes_3pct_gate"] else "FAIL",
            )
        )
    lines.extend(
        [
            "",
            "The PDL-off trial is a 0.32% pooled median regression and is rejected.",
            "The earlier `baseline_*.json` and `pdl_off_*.json` files are retained but",
            "excluded from the headline because their runner default resolved a separate",
            "same-SHA SGLang checkout instead of the explicitly isolated worktree.",
        ]
    )
    (HERE / "paired_summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
