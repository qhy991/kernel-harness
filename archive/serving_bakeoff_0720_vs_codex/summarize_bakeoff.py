#!/usr/bin/env python3
"""Summarize serving bake-off JSON results into CSV + BAKEOFF.md."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REGISTRY = HERE / "registry.yaml"
RESULTS = HERE / "results"
SUMMARY_CSV = RESULTS / "bakeoff_summary.csv"
BAKEOFF_MD = HERE / "BAKEOFF.md"


def _load_result(case_id: str) -> dict:
    path = RESULTS / f"{case_id}.json"
    if not path.exists():
        return {"missing": True}
    return json.loads(path.read_text())


def main() -> None:
    reg = yaml.safe_load(REGISTRY.read_text())
    rows: list[dict] = []

    for case in reg["cases"]:
        status = case.get("status", "run")
        row = {
            "id": case["id"],
            "op": case["op"],
            "m": case.get("m", ""),
            "workload": case.get("workload", ""),
            "variant": case.get("variant", ""),
            "status": status,
            "candidate": case.get("candidate") or "",
            "paired_p50_speedup": "",
            "passes_3pct_gate": "",
            "ref_p50_ms": "",
            "cand_p50_ms": "",
            "correct": "",
            "error": "",
            "notes": case.get("notes", ""),
        }

        if status != "run":
            row["correct"] = "n/a"
            rows.append(row)
            continue

        result = _load_result(case["id"])
        if result.get("missing"):
            row["error"] = "missing_result_json"
            row["correct"] = "unknown"
            rows.append(row)
            continue
        if result.get("bakeoff_error"):
            row["error"] = f"exit_{result.get('exit_code')}"
            row["correct"] = "fail"
            # Keep a short hint from the log tail.
            log = result.get("log") or ""
            hint = ""
            for line in reversed(log.splitlines()):
                if line.strip():
                    hint = line.strip()[:240]
                    break
            if hint:
                row["notes"] = f"{row['notes']} | {hint}".strip(" |")
            rows.append(row)
            continue

        cand = result.get("candidate") or {}
        ref = result.get("reference") or {}
        row["paired_p50_speedup"] = (
            f"{cand['speedup']:.6f}" if "speedup" in cand else ""
        )
        row["passes_3pct_gate"] = str(cand.get("passes_3pct_median_gate", ""))
        row["ref_p50_ms"] = (
            f"{ref['median_ms']:.6f}" if "median_ms" in ref else ""
        )
        row["cand_p50_ms"] = (
            f"{cand['median_ms']:.6f}" if "median_ms" in cand else ""
        )
        # Reaching a result JSON without bakeoff_error means correctness passed
        # (runner asserts before timing).
        row["correct"] = "pass"
        rows.append(row)

    RESULTS.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "id",
        "op",
        "m",
        "workload",
        "variant",
        "status",
        "candidate",
        "paired_p50_speedup",
        "passes_3pct_gate",
        "ref_p50_ms",
        "cand_p50_ms",
        "correct",
        "error",
        "notes",
    ]
    with SUMMARY_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        "# Serving bake-off: archive-0720 vs Codex",
        "",
        f"Generated: `{now}`",
        "",
        "Protocol: `serving_native` interleaved paired A/B (eager CUDA events); "
        f"warmup={reg.get('warmup', 5)}, repeat={reg.get('repeat', 40)}.",
        "",
        "Speedup = stock_ref_p50 / candidate_p50 (>1 is faster). Gate: paired p50 ≥ 1.03×.",
        "",
        "## Runnable results",
        "",
        "| id | op | M | variant | paired p50 | gate | correct | error |",
        "|---|---|---:|---|---:|---|---|---|",
    ]
    for row in rows:
        if row["status"] != "run":
            continue
        lines.append(
            "| `{id}` | {op} | {m} | {variant} | {sp} | {gate} | {correct} | {err} |".format(
                id=row["id"],
                op=row["op"],
                m=row["m"],
                variant=row["variant"],
                sp=row["paired_p50_speedup"] or "—",
                gate=row["passes_3pct_gate"] or "—",
                correct=row["correct"],
                err=row["error"] or "",
            )
        )

    lines += [
        "",
        "## Non-run annotations (0720 under production ABI)",
        "",
        "| id | op | status | notes |",
        "|---|---|---|---|",
    ]
    for row in rows:
        if row["status"] == "run":
            continue
        lines.append(
            f"| `{row['id']}` | {row['op']} | `{row['status']}` | {row['notes']} |"
        )

    lines += [
        "",
        f"Machine-readable: [`results/bakeoff_summary.csv`](results/bakeoff_summary.csv).",
        "",
    ]
    BAKEOFF_MD.write_text("\n".join(lines))
    print(f"wrote {SUMMARY_CSV}")
    print(f"wrote {BAKEOFF_MD}")


if __name__ == "__main__":
    main()
