#!/usr/bin/env python3
"""Summarize independent paired indexer-score campaign series."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(fraction * len(ordered)) - 1))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_dir", type=Path)
    args = parser.parse_args()
    campaign = args.campaign_dir.resolve()

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    failures: list[str] = []
    for path in sorted((campaign / "paired").glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except Exception as exc:
            failures.append(f"{path.name}: {exc}")
            continue
        if payload.get("candidate") is None:
            failures.append(f"{path.name}: no candidate result")
            continue
        payload["_path"] = str(path.relative_to(campaign))
        groups[payload["workload"]["name"]].append(payload)

    rows: list[dict[str, Any]] = []
    for task, series in sorted(groups.items()):
        ref_samples = [
            value
            for payload in series
            for value in payload["reference"]["samples_ms"]
        ]
        cand_samples = [
            value
            for payload in series
            for value in payload["candidate"]["samples_ms"]
        ]
        ratios = [
            value
            for payload in series
            for value in payload["candidate"]["paired_speedups"]
        ]
        per_series_speedups = [
            payload["candidate"]["speedup"] for payload in series
        ]
        metadata = series[0]["runtime_metadata"]
        median_speedup = statistics.median(ratios)
        rows.append(
            {
                "workload": task,
                "family": series[0]["workload"]["family"],
                "series": len(series),
                "samples_per_side": len(ref_samples),
                "reference_median_ms": statistics.median(ref_samples),
                "candidate_median_ms": statistics.median(cand_samples),
                "paired_median_speedup": median_speedup,
                "paired_p10_speedup": percentile(ratios, 0.10),
                "paired_p90_speedup": percentile(ratios, 0.90),
                "series_median_speedups": per_series_speedups,
                "all_series_correct": all(
                    payload["candidate"].get("correctness") == "PASS"
                    for payload in series
                ),
                "passes_3pct_gate": median_speedup >= 1.03,
                "no_series_regresses": min(per_series_speedups) >= 1.0,
                "chunked": metadata["chunked"],
                "stock_chunk_rows": metadata["stock_chunk_rows"],
                "balanced_candidate_max_rows": metadata[
                    "balanced_candidate_max_rows"
                ],
                "q_count": metadata["q_count"],
                "k_count": metadata["k_count"],
                "files": [payload["_path"] for payload in series],
            }
        )

    result = {
        "campaign_dir": str(campaign),
        "rows": rows,
        "parse_failures": failures,
    }
    (campaign / "paired_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )

    lines = [
        "# Paired campaign summary",
        "",
        "| Workload | Series | samples/side | chunks | ref p50 ms | cand p50 ms | paired p50 speedup | paired p10–p90 | correct | >=3% | no series regression |",
        "|---|---:|---:|---|---:|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        chunk_text = "+".join(str(value) for value in row["stock_chunk_rows"])
        if row["balanced_candidate_max_rows"] is not None:
            chunk_text += f" -> max {row['balanced_candidate_max_rows']}"
        lines.append(
            "| {workload} | {series} | {samples_per_side} | {chunks} | "
            "{reference_median_ms:.6f} | {candidate_median_ms:.6f} | "
            "{paired_median_speedup:.5f}x | {paired_p10_speedup:.5f}–"
            "{paired_p90_speedup:.5f}x | {correct} | {gate} | {no_regress} |".format(
                chunks=chunk_text,
                correct="PASS" if row["all_series_correct"] else "FAIL",
                gate="PASS" if row["passes_3pct_gate"] else "FAIL",
                no_regress="PASS" if row["no_series_regresses"] else "FAIL",
                **row,
            )
        )
    if failures:
        lines.extend(["", "Parse failures:", ""])
        lines.extend(f"- {failure}" for failure in failures)
    (campaign / "paired_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
