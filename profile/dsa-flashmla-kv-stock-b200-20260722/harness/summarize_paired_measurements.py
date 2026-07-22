#!/usr/bin/env python3
"""Summarize the goal22 paired eager and CUDA Graph measurement artifacts.

The benchmark producers already define paired p50 as the median of the per-pair
speedups.  This script validates and reports that stored value; it deliberately
does not substitute the ratio of the independently summarized medians.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


PROFILE_DIR = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = PROFILE_DIR / "analysis"
JSON_OUTPUT = ANALYSIS_DIR / "paired_measurements_summary.json"
MARKDOWN_OUTPUT = ANALYSIS_DIR / "paired_measurements_summary.md"
GATE_SPEEDUP = 1.03

MODES = ("eager", "cuda_graph")
BUCKETS = ("m16", "m32")
VARIANTS = ("control", "candidate")
SESSIONS = (1, 2, 3)

RAW_PREFIX = {
    ("eager", "control"): "paired_control",
    ("eager", "candidate"): "paired_combine32",
    ("cuda_graph", "control"): "graph_control",
    ("cuda_graph", "candidate"): "graph_combine32",
}

VARIANT_DESCRIPTION = {
    "control": "stock-pybind-tensor compiler/build control",
    "candidate": "combine32-m16-tensor specialization candidate",
}


def _load(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _expected_task(bucket: str) -> str:
    return f"dsa_flashmla_kv_decode_{bucket}"


def _measurement_row(
    mode: str, bucket: str, variant: str, session: int
) -> dict[str, Any]:
    prefix = RAW_PREFIX[(mode, variant)]
    path = ANALYSIS_DIR / f"{prefix}_{bucket}_r{session}.json"
    data, digest = _load(path)
    task = data.get("task", data.get("workload", {}).get("name"))
    if task != _expected_task(bucket):
        raise ValueError(f"{path.name}: unexpected task {task!r}")

    reference_median_ms = data["reference"]["median_ms"]
    candidate_median_ms = data["candidate"]["median_ms"]
    pair_speedups = [sample["speedup"] for sample in data["paired_samples"]]
    if len(pair_speedups) != data["repeat"]:
        raise ValueError(
            f"{path.name}: {len(pair_speedups)} pairs != repeat={data['repeat']}"
        )

    if mode == "eager":
        paired_median_speedup = data["candidate"]["speedup"]
        passes_gate = data["candidate"]["passes_3pct_median_gate"]
        expected_mode = "eager_cuda_event"
        if data["execution_mode"] != expected_mode:
            raise ValueError(
                f"{path.name}: {data['execution_mode']!r} != {expected_mode!r}"
            )
        correctness_pass = None
    else:
        paired_median_speedup = data["paired_median_speedup"]
        passes_gate = data["passes_3pct_gate"]
        expected_mode = "real_cuda_graph_replay"
        if data["mode"] != expected_mode:
            raise ValueError(f"{path.name}: {data['mode']!r} != {expected_mode!r}")
        correctness = data["correctness"]
        correctness_pass = bool(
            correctness["initial_exact_dtype_and_tolerance"]
            and correctness["mutated_inputs_match"]
            and not correctness["outputs_alias"]
            and correctness["mutated_candidate_change_max_abs"] > 0.0
            and correctness["mutated_reference_change_max_abs"] > 0.0
        )

    computed_pair_median = statistics.median(pair_speedups)
    if paired_median_speedup != computed_pair_median:
        raise ValueError(
            f"{path.name}: stored paired median {paired_median_speedup!r} "
            f"!= recomputed {computed_pair_median!r}"
        )
    if passes_gate != (paired_median_speedup >= GATE_SPEEDUP):
        raise ValueError(
            f"{path.name}: stored gate {passes_gate!r} disagrees with "
            f"speedup {paired_median_speedup!r}"
        )

    evidence = data["candidate_evidence"]
    return {
        "mode": mode,
        "bucket": bucket,
        "variant": variant,
        "variant_description": VARIANT_DESCRIPTION[variant],
        "session": session,
        "raw_file": str(path.relative_to(PROFILE_DIR)),
        "raw_sha256": digest,
        "task": task,
        "warmup": data["warmup"],
        "repeat": data["repeat"],
        "reference_median_ms": reference_median_ms,
        "candidate_median_ms": candidate_median_ms,
        "ratio_of_medians": reference_median_ms / candidate_median_ms,
        "paired_median_speedup": paired_median_speedup,
        "passes_3pct_gate": passes_gate,
        "graph_correctness_pass": correctness_pass,
        "candidate_label": evidence["label"],
        "candidate_extension_sha256": evidence["extension_sha256"],
        "candidate_source_commit": evidence.get("source_commit"),
    }


def _group_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for mode in MODES:
        for bucket in BUCKETS:
            for variant in VARIANTS:
                selected = [
                    row
                    for row in rows
                    if row["mode"] == mode
                    and row["bucket"] == bucket
                    and row["variant"] == variant
                ]
                speedups = [row["paired_median_speedup"] for row in selected]
                pass_count = sum(row["passes_3pct_gate"] for row in selected)
                groups.append(
                    {
                        "mode": mode,
                        "bucket": bucket,
                        "variant": variant,
                        "session_count": len(selected),
                        "session_paired_median_speedups": speedups,
                        "median_session_paired_speedup": statistics.median(speedups),
                        "minimum_session_paired_speedup": min(speedups),
                        "maximum_session_paired_speedup": max(speedups),
                        "sessions_passing_3pct_gate": pass_count,
                        "multiple_sessions_pass_3pct_gate": pass_count > 1,
                        "all_sessions_pass_3pct_gate": pass_count == len(selected),
                    }
                )
    return groups


def _compiler_control_comparison(
    groups: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for mode in MODES:
        for bucket in BUCKETS:
            control = next(
                group
                for group in groups
                if group["mode"] == mode
                and group["bucket"] == bucket
                and group["variant"] == "control"
            )
            candidate = next(
                group
                for group in groups
                if group["mode"] == mode
                and group["bucket"] == bucket
                and group["variant"] == "candidate"
            )
            control_speedup = control["median_session_paired_speedup"]
            candidate_speedup = candidate["median_session_paired_speedup"]
            result.append(
                {
                    "mode": mode,
                    "bucket": bucket,
                    "control_median_session_paired_speedup": control_speedup,
                    "candidate_median_session_paired_speedup": candidate_speedup,
                    "candidate_minus_control_speedup": candidate_speedup
                    - control_speedup,
                    "control_sessions_passing_3pct_gate": control[
                        "sessions_passing_3pct_gate"
                    ],
                    "candidate_sessions_passing_3pct_gate": candidate[
                        "sessions_passing_3pct_gate"
                    ],
                }
            )
    return result


def _baseline_context() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for bucket in BUCKETS:
        for session in SESSIONS:
            path = ANALYSIS_DIR / f"baseline_stock_{bucket}_r{session}.json"
            data, digest = _load(path)
            result.append(
                {
                    "bucket": bucket,
                    "session": session,
                    "raw_file": str(path.relative_to(PROFILE_DIR)),
                    "raw_sha256": digest,
                    "reference_median_ms": data["reference"]["median_ms"],
                    "warmup": data["warmup"],
                    "repeat": data["repeat"],
                }
            )
    return result


def _build_summary() -> dict[str, Any]:
    rows = [
        _measurement_row(mode, bucket, variant, session)
        for mode in MODES
        for bucket in BUCKETS
        for variant in VARIANTS
        for session in SESSIONS
    ]
    groups = _group_rows(rows)
    baselines = _baseline_context()
    candidate_groups = [group for group in groups if group["variant"] == "candidate"]
    graph_rows = [row for row in rows if row["mode"] == "cuda_graph"]
    baseline_drift: dict[str, float] = {}
    for bucket in BUCKETS:
        selected = [row for row in baselines if row["bucket"] == bucket]
        baseline_drift[bucket] = (
            selected[-1]["reference_median_ms"] / selected[0]["reference_median_ms"]
            - 1.0
        )

    return {
        "schema_version": 1,
        "generated_by": str(Path(__file__).relative_to(PROFILE_DIR)),
        "method": {
            "gate_speedup": GATE_SPEEDUP,
            "paired_speedup_definition": (
                "median of the raw per-pair reference_ms/candidate_ms speedups"
            ),
            "warning": (
                "paired_median_speedup is not the ratio of the separately "
                "reported reference and candidate medians"
            ),
            "compiler_control_comparison": (
                "descriptive difference between three-session medians; the "
                "control and candidate sessions are not paired to each other"
            ),
        },
        "measurements": rows,
        "groups": groups,
        "compiler_control_comparison": _compiler_control_comparison(groups),
        "unpaired_stock_baseline_context": {
            "acceptance_use": "context only; excluded from paired gates",
            "measurements": baselines,
            "r1_to_r3_relative_change": baseline_drift,
        },
        "overall_observations": {
            "candidate_groups_with_multiple_3pct_sessions": sum(
                group["multiple_sessions_pass_3pct_gate"]
                for group in candidate_groups
            ),
            "candidate_m16_eager_sessions_passing_3pct_gate": next(
                group["sessions_passing_3pct_gate"]
                for group in candidate_groups
                if group["mode"] == "eager" and group["bucket"] == "m16"
            ),
            "candidate_graph_sessions_passing_3pct_gate": sum(
                row["passes_3pct_gate"]
                for row in rows
                if row["mode"] == "cuda_graph" and row["variant"] == "candidate"
            ),
            "all_graph_correctness_checks_pass": all(
                row["graph_correctness_pass"] for row in graph_rows
            ),
        },
    }


def _fmt(value: float) -> str:
    """Use Python's shortest round-trippable representation."""
    return repr(value)


def _render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Paired eager and CUDA Graph measurement summary",
        "",
        "This file is generated from the raw JSON artifacts by:",
        "",
        "```bash",
        ".venv/bin/python profile/dsa-flashmla-kv-stock-b200-20260722/harness/summarize_paired_measurements.py",
        "```",
        "",
        "`paired_median_speedup` is the producer-defined median of the 100 raw "
        "pair-wise speedups. It is not `reference_median / candidate_median`; "
        "the machine-readable JSON preserves both quantities at full precision. "
        "The per-session 3% gate is `paired_median_speedup >= 1.03`.",
        "",
    ]

    for mode, heading in (("eager", "Paired eager"), ("cuda_graph", "Real CUDA Graph replay")):
        lines.extend(
            [
                f"## {heading}",
                "",
                "| Bucket | Variant | Session | Raw file | Reference p50 (ms) | Candidate p50 (ms) | Paired p50 speedup | 3% gate | Graph correctness |",
                "|---|---|---:|---|---:|---:|---:|---|---|",
            ]
        )
        for row in summary["measurements"]:
            if row["mode"] != mode:
                continue
            correctness = row["graph_correctness_pass"]
            correctness_text = "n/a" if correctness is None else ("PASS" if correctness else "FAIL")
            lines.append(
                "| {bucket} | {variant} | {session} | `{raw}` | {reference} | "
                "{candidate} | {speedup} | {gate} | {correctness} |".format(
                    bucket=row["bucket"].upper(),
                    variant=row["variant"],
                    session=row["session"],
                    raw=row["raw_file"],
                    reference=_fmt(row["reference_median_ms"]),
                    candidate=_fmt(row["candidate_median_ms"]),
                    speedup=_fmt(row["paired_median_speedup"]),
                    gate="PASS" if row["passes_3pct_gate"] else "FAIL",
                    correctness=correctness_text,
                )
            )
        lines.append("")

    lines.extend(
        [
            "## Compiler/build control comparison",
            "",
            "This is descriptive: it subtracts the median of the three control "
            "session speedups from the median of the three candidate session "
            "speedups. The control and candidate sessions were not paired to "
            "each other, so this delta is not itself an acceptance metric.",
            "",
            "| Mode | Bucket | Control median session speedup | Candidate median session speedup | Candidate - control | Control passes | Candidate passes |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["compiler_control_comparison"]:
        lines.append(
            "| {mode} | {bucket} | {control} | {candidate} | {delta} | {control_pass}/3 | {candidate_pass}/3 |".format(
                mode=row["mode"],
                bucket=row["bucket"].upper(),
                control=_fmt(row["control_median_session_paired_speedup"]),
                candidate=_fmt(row["candidate_median_session_paired_speedup"]),
                delta=_fmt(row["candidate_minus_control_speedup"]),
                control_pass=row["control_sessions_passing_3pct_gate"],
                candidate_pass=row["candidate_sessions_passing_3pct_gate"],
            )
        )

    lines.extend(
        [
            "",
            "## Unpaired stock baseline context",
            "",
            "These warmup=5, repeat=50 runs expose cold-to-warm drift and are "
            "not used for acceptance.",
            "",
            "| Bucket | Session | Raw file | Reference p50 (ms) |",
            "|---|---:|---|---:|",
        ]
    )
    for row in summary["unpaired_stock_baseline_context"]["measurements"]:
        lines.append(
            f"| {row['bucket'].upper()} | {row['session']} | `{row['raw_file']}` | {_fmt(row['reference_median_ms'])} |"
        )

    observations = summary["overall_observations"]
    drift = summary["unpaired_stock_baseline_context"]["r1_to_r3_relative_change"]
    lines.extend(
        [
            "",
            "## Evidence outcome and anomalies",
            "",
            f"- The specialization candidate clears 1.03 in only {observations['candidate_m16_eager_sessions_passing_3pct_gate']}/3 M16 eager sessions. M16 eager r2/r3 do not reproduce it.",
            f"- The specialization candidate clears 1.03 in {observations['candidate_graph_sessions_passing_3pct_gate']}/6 real CUDA Graph sessions. Every graph correctness/mutation/anti-alias check passes: `{str(observations['all_graph_correctness_checks_pass']).lower()}`.",
            "- No candidate mode/bucket group has more than one session above the 3% threshold. The only favorable candidate row is `analysis/paired_combine32_m16_r1.json`; graph replay reverses that apparent win.",
            "- The compiler/build control itself moves around unity. Its three-session medians and the candidate-minus-control deltas above bound the apparent build/toolchain and timing variation.",
            f"- The unpaired stock p50 changes from r1 to r3 by {_fmt(drift['m16'])} for M16 and {_fmt(drift['m32'])} for M32. This cold/warm drift is why those files are context only.",
            "- `analysis/paired_combine32_m32_r2.json` has a slightly lower candidate median than reference median but a paired speedup below 1.0. This is not corruption: median(pair-wise ratios) and ratio(independent medians) are different statistics.",
            "",
        ]
    )
    return "\n".join(lines)


def _check_or_write(path: Path, content: str, check: bool) -> None:
    if check:
        existing = path.read_text()
        if existing != content:
            raise SystemExit(f"stale generated summary: {path}")
    else:
        path.write_text(content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check", action="store_true", help="verify committed summaries are current"
    )
    args = parser.parse_args()

    summary = _build_summary()
    json_text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    markdown_text = _render_markdown(summary)
    _check_or_write(JSON_OUTPUT, json_text, args.check)
    _check_or_write(MARKDOWN_OUTPUT, markdown_text, args.check)
    action = "verified" if args.check else "wrote"
    print(f"{action}: {JSON_OUTPUT}")
    print(f"{action}: {MARKDOWN_OUTPUT}")


if __name__ == "__main__":
    main()
