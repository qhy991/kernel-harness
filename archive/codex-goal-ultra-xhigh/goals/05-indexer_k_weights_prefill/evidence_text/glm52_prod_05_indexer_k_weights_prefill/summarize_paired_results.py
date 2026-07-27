#!/usr/bin/env python3
"""Summarize goal-05 serving-native measurements without mixing GPUs.

The serving-native runner records two different kinds of numbers:

* ``candidate.speedup`` is the median of interleaved, per-pair
  ``reference_ms / candidate_ms`` ratios and is the comparison metric.
* ``reference.median_ms`` and ``candidate.median_ms`` are marginal latency
  medians.  They are useful context, but their quotient is not a paired
  speedup and unpaired baseline files must not be used for comparisons.

This script preserves that distinction in JSON, CSV, and Markdown.  It also
attaches the physical GPU printed by each campaign wrapper.  Missing result or
profiler files are reported instead of making the summary fail, which permits
safe use while a campaign is still being collected.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


SUMMARY_NAMES = {
    "paired_results_summary.json",
    "paired_results_summary.csv",
}


def _campaign_specs(evidence_dir: Path) -> dict[str, dict[str, Any]]:
    profile_dir = evidence_dir.parents[1] / (
        "profile/indexer-wk-weights-prefill-m4096-20260722"
    )
    first_results = [
        *(f"isolated_baseline_{run}.json" for run in ("01", "02", "03")),
        *(f"region_baseline_{run}.json" for run in ("01", "02", "03")),
        "isolated_reference_control.json",
        "region_reference_control.json",
        "isolated_cutedsl_sweep.json",
        *(f"isolated_flashinfer_{backend}_sweep.json" for backend in (
            "cutlass", "tgv", "cublaslt", "cudnn", "auto"
        )),
        *(f"isolated_best_{run}.json" for run in ("01", "02", "03")),
        *(f"region_best_{run}.json" for run in ("01", "02", "03")),
    ]
    schedule_results = [
        *(f"isolated_torch_mm_{run}.json" for run in ("01", "02", "03")),
        *(f"region_torch_mm_{run}.json" for run in ("01", "02", "03")),
        *(f"region_k_first_{run}.json" for run in ("01", "02", "03")),
    ]
    post_revert_results = [
        "post_revert_isolated_stock.json",
        "post_revert_region_stock.json",
    ]
    exact_results = [
        *(f"exact_bf16_wq/isolated_stock_{run}.json" for run in ("01", "02", "03")),
        *(f"exact_bf16_wq/region_stock_{run}.json" for run in ("01", "02", "03")),
        "exact_bf16_wq/isolated_identity.json",
        "exact_bf16_wq/region_identity.json",
        *(f"exact_bf16_wq/isolated_torch_mm_{run}.json" for run in ("01", "02", "03")),
        *(f"exact_bf16_wq/region_torch_mm_{run}.json" for run in ("01", "02", "03")),
        *(f"exact_bf16_wq/region_tgv_{run}.json" for run in ("01", "02", "03")),
    ]
    specs: dict[str, dict[str, Any]] = {
        "superseded_fp8_wq_backend_campaign": {
            "wrapper_log": evidence_dir / "wrapper_single_gpu_campaign.log",
            "status_file": evidence_dir / "sweep_status.txt",
            "expected_results": first_results,
            "patterns": [
                re.compile(r"^(?:isolated|region)_baseline_\d+$"),
                re.compile(r"^(?:isolated|region)_reference_control$"),
                re.compile(r"^isolated_(?:cutedsl|flashinfer_.+)_sweep$"),
                re.compile(r"^(?:isolated|region)_best_\d+$"),
            ],
            "expected_profiler_artifacts": [
                evidence_dir / "runtime_abi_trace_stock.json",
                evidence_dir / "runtime_abi_trace_best.json",
                profile_dir / "reports/nsys-stock.nsys-rep",
                profile_dir / "reports/nsys-best.nsys-rep",
                profile_dir / "analysis/nsys-stock_cuda_gpu_trace.csv",
                profile_dir / "analysis/nsys-best_cuda_gpu_trace.csv",
            ],
        },
        "superseded_fp8_wq_schedule_ncu_campaign": {
            "wrapper_log": evidence_dir / "wrapper_schedule_and_ncu_campaign.log",
            "status_file": evidence_dir / "schedule_campaign_status.txt",
            "expected_results": schedule_results,
            "patterns": [
                re.compile(r"^(?:isolated|region)_torch_mm_\d+$"),
                re.compile(r"^region_k_first_\d+$"),
            ],
            "expected_profiler_artifacts": [
                evidence_dir / "runtime_abi_trace_schedule_stock.json",
                evidence_dir / "runtime_abi_trace_k_first.json",
                evidence_dir / "runtime_abi_trace_ncu_stock.json",
                evidence_dir / "runtime_abi_trace_ncu_source_stock.json",
                profile_dir / "reports/nsys-schedule-stock.nsys-rep",
                profile_dir / "reports/nsys-k-first.nsys-rep",
                profile_dir / "reports/full-stock-wk-m4096.ncu-rep",
                profile_dir / "reports/source-stock-wk-m4096.ncu-rep",
                profile_dir / "reports/full-stock-indexer-post-m4096.ncu-rep",
                profile_dir / "reports/source-stock-indexer-post-m4096.ncu-rep",
                profile_dir
                / "analysis/nsys-schedule-stock_cuda_gpu_trace_nvtx-name_base.csv",
                profile_dir
                / "analysis/nsys-k-first_cuda_gpu_trace_nvtx-name_base.csv",
            ],
        },
        "superseded_fp8_wq_post_revert_smoke": {
            "wrapper_log": evidence_dir / "wrapper_post_revert_smoke.log",
            "status_file": evidence_dir / "post_revert_check_env.txt",
            "expected_results": post_revert_results,
            "patterns": [re.compile(r"^post_revert_(?:isolated|region)_stock$")],
            "expected_profiler_artifacts": [],
        },
        "exact_bf16_wq_campaign": {
            "wrapper_log": evidence_dir / "wrapper_exact_bf16_wq_campaign_attempt3.log",
            "status_file": evidence_dir / "exact_bf16_wq/campaign_status.txt",
            "expected_results": exact_results,
            "patterns": [],
            "expected_profiler_artifacts": [
                evidence_dir / "exact_bf16_wq/runtime_abi_trace_stock.json",
                evidence_dir / "exact_bf16_wq/runtime_abi_trace_torch_mm.json",
                profile_dir / "reports/nsys-exact-bf16-wq-stock.nsys-rep",
                profile_dir / "reports/nsys-exact-bf16-wq-torch-mm.nsys-rep",
                profile_dir
                / "analysis/nsys-exact-bf16-wq-stock_cuda_gpu_trace_nvtx-name_base.csv",
                profile_dir
                / "analysis/nsys-exact-bf16-wq-torch-mm_cuda_gpu_trace_nvtx-name_base.csv",
            ],
        },
        "exact_single_stream_campaign": {
            "wrapper_log": evidence_dir
            / "wrapper_exact_single_stream_campaign_attempt1.log",
            "status_file": evidence_dir / "exact_single_stream/campaign_status.txt",
            "expected_results": [
                f"exact_single_stream/region_single_stream_{run}.json"
                for run in ("01", "02", "03")
            ],
            "patterns": [],
            "expected_profiler_artifacts": [
                evidence_dir / "exact_single_stream/runtime_abi_trace_stock.json",
                evidence_dir
                / "exact_single_stream/runtime_abi_trace_single-stream.json",
                profile_dir / "reports/nsys-exact-single-stream-stock.nsys-rep",
                profile_dir
                / "reports/nsys-exact-single-stream-single-stream.nsys-rep",
                profile_dir
                / "analysis/nsys-exact-single-stream-stock_cuda_gpu_trace_nvtx-name_base.csv",
                profile_dir
                / "analysis/nsys-exact-single-stream-single-stream_cuda_gpu_trace_nvtx-name_base.csv",
            ],
        },
    }
    hardened_root = evidence_dir / "hardened_runs"
    hardened_dirs = (
        sorted(path for path in hardened_root.iterdir() if path.is_dir())
        if hardened_root.is_dir()
        else []
    )
    if hardened_dirs:
        hardened_dir = hardened_dirs[-1]
        prefix = str(hardened_dir.relative_to(evidence_dir))
        hardened_results = [
            *(f"{prefix}/results/isolated_stock_{run}.json" for run in ("01", "02", "03")),
            *(f"{prefix}/results/region_stock_{run}.json" for run in ("01", "02", "03")),
            f"{prefix}/results/isolated_identity.json",
            f"{prefix}/results/region_identity.json",
            *(f"{prefix}/results/isolated_tgv_{run}.json" for run in ("01", "02", "03")),
            *(f"{prefix}/results/region_tgv_{run}.json" for run in ("01", "02", "03")),
            *(f"{prefix}/results/isolated_torch_mm_{run}.json" for run in ("01", "02", "03")),
            *(f"{prefix}/results/region_torch_mm_{run}.json" for run in ("01", "02", "03")),
            *(f"{prefix}/results/region_single_stream_{run}.json" for run in ("01", "02", "03")),
        ]
        specs = {
            "immutable_hardened_campaign": {
                "wrapper_log": hardened_dir / "environment.txt",
                "status_file": hardened_dir / "status.txt",
                "expected_results": hardened_results,
                "patterns": [],
                "expected_profiler_artifacts": [
                    hardened_dir / "validation.json",
                    hardened_dir / "source_manifest.sha256",
                    hardened_dir / "profiles/nsys-stock.nsys-rep",
                    hardened_dir / "profiles/nsys-torch-mm.nsys-rep",
                    hardened_dir / "profiles/nsys-single-stream.nsys-rep",
                    hardened_dir / "profiles/abi-stock.json",
                    hardened_dir / "profiles/abi-torch-mm.json",
                    hardened_dir / "profiles/abi-single-stream.json",
                ],
            },
            **specs,
        }
    return specs


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _parse_gpu(wrapper_log: Path) -> dict[str, Any] | None:
    if not wrapper_log.is_file():
        return None
    contents = wrapper_log.read_text(errors="replace")
    match = re.search(
        r"allocated physical GPU\s+(\d+)\s+\((GPU-[^)]+)\)\s+"
        r"as logical GPU\s+(\d+)",
        contents,
    )
    if match is not None:
        return {
            "physical_ordinal": int(match.group(1)),
            "uuid": match.group(2),
            "logical_ordinal": int(match.group(3)),
        }
    ordinal = re.search(r"^selected_physical_gpu=(\d+)$", contents, re.MULTILINE)
    identity = re.search(
        r"^(\d+),\s*(GPU-[^,]+),\s*NVIDIA B200,", contents, re.MULTILINE
    )
    if ordinal is None or identity is None or ordinal.group(1) != identity.group(1):
        return None
    return {
        "physical_ordinal": int(ordinal.group(1)),
        "uuid": identity.group(2),
        "logical_ordinal": 0,
    }


def _read_status(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _is_result(payload: Any) -> bool:
    return (
        isinstance(payload, dict)
        and isinstance(payload.get("workload"), dict)
        and isinstance(payload.get("reference"), dict)
        and "candidate" in payload
        and "median_ms" in payload["reference"]
    )


def _series_id(stem: str) -> str:
    return re.sub(r"_(?:01|02|03)$", "", stem)


def _candidate_label(candidate: dict[str, Any] | None) -> str:
    if candidate is None:
        return "stock_reference_only"
    metadata = candidate.get("metadata")
    if isinstance(metadata, dict) and metadata.get("backend"):
        return str(metadata["backend"])
    path = candidate.get("path")
    return Path(path).stem if path else "candidate"


def _find_campaign(path: Path, specs: dict[str, dict[str, Any]]) -> str:
    if "hardened_runs" in path.parts:
        return "immutable_hardened_campaign"
    if "exact_bf16_wq" in path.parts:
        return "exact_bf16_wq_campaign"
    if "exact_single_stream" in path.parts:
        return "exact_single_stream_campaign"
    stem = path.stem
    for campaign_id, spec in specs.items():
        if any(pattern.fullmatch(stem) for pattern in spec["patterns"]):
            return campaign_id
    return "unclassified"


def _row(
    path: Path,
    payload: dict[str, Any],
    campaign_id: str,
    gpu: dict[str, Any] | None,
) -> dict[str, Any]:
    workload = payload["workload"]
    candidate = payload.get("candidate")
    candidate_dict = candidate if isinstance(candidate, dict) else None
    reference = payload["reference"]
    params = workload.get("params", {})
    if campaign_id.startswith("superseded_fp8_wq"):
        evidence_scope = (
            "production_exact_isolated_bf16_projection"
            if workload.get("family") == "bf16_linear"
            else "superseded_wrong_fp8_wq_and_rope_region"
        )
    elif campaign_id == "immutable_hardened_campaign":
        evidence_scope = "authoritative_fixed_model_immutable"
    elif campaign_id == "exact_bf16_wq_campaign":
        evidence_scope = "fixed_model_bf16_wq_exact_config"
    elif campaign_id == "exact_single_stream_campaign":
        evidence_scope = "fixed_model_preliminary_single_stream_with_linear_adapter"
    else:
        evidence_scope = "unclassified"
    return {
        "source_file": path.name,
        "series_id": _series_id(path.stem),
        "campaign_id": campaign_id,
        "evidence_scope": evidence_scope,
        "gpu": gpu,
        "workload_name": workload.get("name"),
        "workload_family": workload.get("family"),
        "source_symbol": workload.get("source_symbol"),
        "m": params.get("m") if isinstance(params, dict) else None,
        "execution_mode": payload.get("execution_mode"),
        "timing_contract": payload.get("timing_contract"),
        "reference_policy": payload.get("reference_policy"),
        "measurement_kind": "paired_candidate" if candidate_dict else "reference_only",
        "candidate_label": _candidate_label(candidate_dict),
        "candidate_path": candidate_dict.get("path") if candidate_dict else None,
        "candidate_metadata": candidate_dict.get("metadata") if candidate_dict else None,
        "marginal_latency_medians_ms": {
            "reference": reference.get("median_ms"),
            "candidate": candidate_dict.get("median_ms") if candidate_dict else None,
        },
        "reference_latency_ms": {
            "median": reference.get("median_ms"),
            "min": reference.get("min_ms"),
            "p95": reference.get("p95_ms"),
            "sample_count": len(payload.get("reference_samples_ms", [])),
        },
        "candidate_latency_ms": (
            {
                "median": candidate_dict.get("median_ms"),
                "min": candidate_dict.get("min_ms"),
                "p95": candidate_dict.get("p95_ms"),
                "sample_count": len(candidate_dict.get("samples_ms", [])),
            }
            if candidate_dict
            else None
        ),
        "paired_speedup": (
            {
                "median": candidate_dict.get("speedup"),
                "p10": candidate_dict.get("paired_p10_speedup"),
                "p90": candidate_dict.get("paired_p90_speedup"),
                "sample_count": len(candidate_dict.get("paired_speedups", [])),
                "passes_3pct_median_gate": candidate_dict.get(
                    "passes_3pct_median_gate"
                ),
            }
            if candidate_dict
            else None
        ),
    }


def _median_or_none(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["campaign_id"], row["series_id"])].append(row)

    aggregates = []
    for (campaign_id, series_id), group in sorted(grouped.items()):
        group.sort(key=lambda item: item["source_file"])
        paired_medians = [
            float(row["paired_speedup"]["median"])
            for row in group
            if row["paired_speedup"] is not None
            and row["paired_speedup"]["median"] is not None
        ]
        paired_p10s = [
            float(row["paired_speedup"]["p10"])
            for row in group
            if row["paired_speedup"] is not None
            and row["paired_speedup"]["p10"] is not None
        ]
        paired_p90s = [
            float(row["paired_speedup"]["p90"])
            for row in group
            if row["paired_speedup"] is not None
            and row["paired_speedup"]["p90"] is not None
        ]
        gates = [
            row["paired_speedup"]["passes_3pct_median_gate"]
            for row in group
            if row["paired_speedup"] is not None
            and row["paired_speedup"]["passes_3pct_median_gate"] is not None
        ]
        reference_medians = [
            float(row["marginal_latency_medians_ms"]["reference"])
            for row in group
            if row["marginal_latency_medians_ms"]["reference"] is not None
        ]
        candidate_medians = [
            float(row["marginal_latency_medians_ms"]["candidate"])
            for row in group
            if row["marginal_latency_medians_ms"]["candidate"] is not None
        ]
        aggregates.append(
            {
                "campaign_id": campaign_id,
                "evidence_scope": group[0]["evidence_scope"],
                "series_id": series_id,
                "gpu": group[0]["gpu"],
                "workload_name": group[0]["workload_name"],
                "measurement_kind": group[0]["measurement_kind"],
                "candidate_label": group[0]["candidate_label"],
                "run_count": len(group),
                "source_files": [row["source_file"] for row in group],
                "paired_speedup": (
                    {
                        "per_run_medians": paired_medians,
                        "across_run_median_of_recorded_medians": _median_or_none(
                            paired_medians
                        ),
                        "per_run_p10": paired_p10s,
                        "per_run_p90": paired_p90s,
                        "passes_3pct_median_gate_per_run": gates,
                        "all_runs_pass_3pct_gate": all(gates) if gates else None,
                    }
                    if paired_medians
                    else None
                ),
                "marginal_latency_medians_ms": {
                    "reference_per_run": reference_medians,
                    "candidate_per_run": candidate_medians,
                },
            }
        )
    return aggregates


def _campaign_summary(
    specs: dict[str, dict[str, Any]], evidence_dir: Path, root: Path
) -> list[dict[str, Any]]:
    summaries = []
    for campaign_id, spec in specs.items():
        expected_results = [evidence_dir / name for name in spec["expected_results"]]
        expected_profiles = list(spec["expected_profiler_artifacts"])
        summaries.append(
            {
                "campaign_id": campaign_id,
                "gpu": _parse_gpu(spec["wrapper_log"]),
                "wrapper_log": _relative(spec["wrapper_log"], root),
                "status_file": _relative(spec["status_file"], root),
                "status_lines": _read_status(spec["status_file"]),
                "expected_result_count": len(expected_results),
                "present_result_count": sum(path.is_file() for path in expected_results),
                "missing_result_files": [
                    _relative(path, root) for path in expected_results if not path.is_file()
                ],
                "missing_profiler_artifacts": [
                    _relative(path, root) for path in expected_profiles if not path.is_file()
                ],
            }
        )
    return summaries


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _write_csv(path: Path, aggregates: list[dict[str, Any]]) -> None:
    fieldnames = [
        "campaign_id",
        "evidence_scope",
        "physical_gpu",
        "gpu_uuid",
        "series_id",
        "workload_name",
        "measurement_kind",
        "candidate_label",
        "run_count",
        "reference_marginal_medians_ms",
        "candidate_marginal_medians_ms",
        "paired_speedup_medians",
        "paired_speedup_across_run_median",
        "paired_p10_per_run",
        "paired_p90_per_run",
        "passes_3pct_gate_per_run",
        "all_runs_pass_3pct_gate",
        "source_files",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for item in aggregates:
            gpu = item["gpu"] or {}
            paired = item["paired_speedup"] or {}
            marginal = item["marginal_latency_medians_ms"]
            writer.writerow(
                {
                    "campaign_id": item["campaign_id"],
                    "evidence_scope": item["evidence_scope"],
                    "physical_gpu": gpu.get("physical_ordinal"),
                    "gpu_uuid": gpu.get("uuid"),
                    "series_id": item["series_id"],
                    "workload_name": item["workload_name"],
                    "measurement_kind": item["measurement_kind"],
                    "candidate_label": item["candidate_label"],
                    "run_count": item["run_count"],
                    "reference_marginal_medians_ms": ";".join(
                        str(value) for value in marginal["reference_per_run"]
                    ),
                    "candidate_marginal_medians_ms": ";".join(
                        str(value) for value in marginal["candidate_per_run"]
                    ),
                    "paired_speedup_medians": ";".join(
                        str(value) for value in paired.get("per_run_medians", [])
                    ),
                    "paired_speedup_across_run_median": paired.get(
                        "across_run_median_of_recorded_medians"
                    ),
                    "paired_p10_per_run": ";".join(
                        str(value) for value in paired.get("per_run_p10", [])
                    ),
                    "paired_p90_per_run": ";".join(
                        str(value) for value in paired.get("per_run_p90", [])
                    ),
                    "passes_3pct_gate_per_run": ";".join(
                        str(value).lower()
                        for value in paired.get("passes_3pct_median_gate_per_run", [])
                    ),
                    "all_runs_pass_3pct_gate": paired.get(
                        "all_runs_pass_3pct_gate"
                    ),
                    "source_files": ";".join(item["source_files"]),
                }
            )


def _write_markdown(
    path: Path,
    campaigns: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
    unclassified: list[str],
    hardened_lane: dict[str, Any],
    qk_ncu_lane: dict[str, Any],
    tp4_lane: dict[str, Any],
) -> None:
    lines = [
        "# Paired result summary",
        "",
        "`candidate.speedup` is the runner-recorded median of interleaved "
        "per-pair `reference_ms / candidate_ms` ratios. The latency medians below "
        "are marginal medians and are not divided to manufacture a speedup. "
        "Reference-only baseline files are descriptive only. Results from "
        "different campaign GPUs are never combined into a comparison. The "
        "`immutable_hardened_campaign` is authoritative. The "
        "three `superseded_fp8_wq_*` campaigns used a non-production FP8 wq_b "
        "and generic RoPE in their prepare/store-subregion rows; only their isolated BF16 "
        "projection rows remain valid.",
        "",
        "## Campaign provenance",
        "",
        "| Campaign | Physical GPU | UUID | Results | Missing results | Missing profiler artifacts |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for campaign in campaigns:
        gpu = campaign["gpu"] or {}
        lines.append(
            "| {campaign_id} | {physical} | {uuid} | {present}/{expected} | "
            "{missing_results} | {missing_profiles} |".format(
                campaign_id=campaign["campaign_id"],
                physical=_fmt(gpu.get("physical_ordinal")),
                uuid=_fmt(gpu.get("uuid")),
                present=campaign["present_result_count"],
                expected=campaign["expected_result_count"],
                missing_results=len(campaign["missing_result_files"]),
                missing_profiles=len(campaign["missing_profiler_artifacts"]),
            )
        )

    lines.extend(
        [
            "",
            "## Recorded series",
            "",
            "| Campaign / GPU | Evidence scope | Series | Candidate | Runs | Reference marginal medians (ms) | Candidate marginal medians (ms) | Paired median speedups | Across-run median | All runs pass 1.03x |",
            "|---|---|---|---|---:|---|---|---|---:|---:|",
        ]
    )
    for item in aggregates:
        gpu = item["gpu"] or {}
        marginal = item["marginal_latency_medians_ms"]
        paired = item["paired_speedup"] or {}
        reference = ", ".join(
            _fmt(value) for value in marginal["reference_per_run"]
        )

        candidate = ", ".join(
            _fmt(value) for value in marginal["candidate_per_run"]
        ) or "-"
        paired_values = ", ".join(
            _fmt(value) for value in paired.get("per_run_medians", [])
        ) or "-"
        lines.append(
            "| {campaign} / {gpu} | {scope} | {series} | {candidate_label} | {runs} | "
            "{reference} | {candidate} | {paired_values} | {paired_median} | "
            "{gate} |".format(
                campaign=item["campaign_id"],
                gpu=_fmt(gpu.get("physical_ordinal")),
                scope=item["evidence_scope"],
                series=item["series_id"],
                candidate_label=item["candidate_label"],
                runs=item["run_count"],
                reference=reference,
                candidate=candidate,
                paired_values=paired_values,
                paired_median=_fmt(
                    paired.get("across_run_median_of_recorded_medians")
                ),
                gate=_fmt(paired.get("all_runs_pass_3pct_gate")),
            )
        )

    lines.extend(
        [
            "",
            "## Separate validation lanes",
            "",
            f"- Immutable hardened same-GPU rerun: {hardened_lane['status']}"
            + (
                f" (`{hardened_lane['run_dir']}`)"
                if hardened_lane.get("run_dir")
                else ""
            ),
            f"- Exact fixed-model Q/K NCU: {qk_ncu_lane['status']}.",
            f"- Corrected live TP4/DP4/EP4 trace: {tp4_lane['status']}"
            + (
                f" (`{tp4_lane['attempt_artifact']}`)"
                if tp4_lane.get("attempt_artifact")
                else ""
            )
            + "; this is not the TP8 gate.",
        ]
    )

    missing_results = [
        name for campaign in campaigns for name in campaign["missing_result_files"]
    ]
    missing_profiles = [
        name
        for campaign in campaigns
        for name in campaign["missing_profiler_artifacts"]
    ]
    lines.extend(["", "## Missing or unclassified artifacts", ""])
    if not missing_results and not missing_profiles and not unclassified:
        lines.append(
            "All artifacts expected by the completed single-GPU campaign specs are "
            "present. Q/K NCU and TP4 remain separate validation lanes."
        )
    else:
        for label, names in (
            ("Missing serving-native result", missing_results),
            ("Missing profiler artifact", missing_profiles),
            ("Unclassified serving-native result", unclassified),
        ):
            for name in names:
                lines.append(f"- {label}: `{name}`")
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="goal evidence directory (default: directory containing this script)",
    )
    args = parser.parse_args()
    evidence_dir = args.evidence_dir.resolve()
    root = evidence_dir.parents[1]
    specs = _campaign_specs(evidence_dir)
    gpu_by_campaign = {
        campaign_id: _parse_gpu(spec["wrapper_log"])
        for campaign_id, spec in specs.items()
    }

    rows = []
    invalid_json: list[str] = []
    for path in sorted(evidence_dir.rglob("*.json")):
        if path.name in SUMMARY_NAMES:
            continue
        if any("failed_attempt" in part for part in path.parts):
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            invalid_json.append(path.name)
            continue
        if not _is_result(payload):
            continue
        campaign_id = _find_campaign(path, specs)
        rows.append(
            _row(path, payload, campaign_id, gpu_by_campaign.get(campaign_id))
        )

    rows.sort(key=lambda item: (item["campaign_id"], item["source_file"]))
    aggregates = _aggregate(rows)
    campaigns = _campaign_summary(specs, evidence_dir, root)
    unclassified = [
        row["source_file"] for row in rows if row["campaign_id"] == "unclassified"
    ]
    hardened_lane: dict[str, Any] = {"status": "pending", "run_dir": None}
    hardened_root = evidence_dir / "hardened_runs"
    hardened_run_dirs = (
        sorted(
            (path for path in hardened_root.iterdir() if path.is_dir()),
            key=lambda path: (path.stat().st_mtime_ns, path.name),
        )
        if hardened_root.is_dir()
        else []
    )
    if hardened_run_dirs:
        hardened_run_dir = hardened_run_dirs[-1]
        hardened_lane["run_dir"] = _relative(hardened_run_dir, root)
        validation_path = hardened_run_dir / "validation.json"
        try:
            validation = json.loads(validation_path.read_text())
        except (OSError, json.JSONDecodeError):
            hardened_lane["status"] = "incomplete"
        else:
            hardened_lane["status"] = (
                "validated"
                if validation.get("status") == "PASS"
                else "failed validation"
            )
    qk_ncu_lane: dict[str, Any] = {
        "status": "blocked after three scheduler exit-75 attempts",
        "attempt_artifacts": [
            _relative(
                evidence_dir / f"wrapper_exact_ncu_campaign_attempt{attempt}.log",
                root,
            )
            for attempt in (1, 2, 3)
        ],
    }
    tp4_lane: dict[str, Any] = {"status": "pending"}
    tp4_blockers = sorted((evidence_dir / "tp4_live").glob("*_scheduler_blocker.json"))
    if tp4_blockers:
        blocker_path = tp4_blockers[-1]
        try:
            blocker = json.loads(blocker_path.read_text())
        except (OSError, json.JSONDecodeError):
            tp4_lane = {
                "status": "incomplete scheduler-blocker artifact",
                "attempt_artifact": _relative(blocker_path, root),
            }
        else:
            blocker_valid = (
                blocker.get("terminal_exit_code") == 75
                and blocker.get("diagnostic_executed") is False
                and blocker.get("run_directory_created") is False
                and isinstance(blocker.get("attempts"), int)
                and blocker["attempts"] > 0
            )
            tp4_lane = {
                "status": (
                    "blocked by shared four-GPU scheduler after "
                    f"{blocker.get('attempts')} exit-75 attempts"
                    if blocker_valid
                    else "invalid scheduler-blocker artifact"
                ),
                "attempt_artifact": _relative(blocker_path, root),
                "diagnostic_executed": blocker.get("diagnostic_executed"),
                "terminal_exit_code": blocker.get("terminal_exit_code"),
            }
    summary = {
        "schema_version": 1,
        "metric_semantics": {
            "comparison_metric": (
                "candidate.speedup: median of interleaved per-pair "
                "reference_ms / candidate_ms ratios"
            ),
            "marginal_latency_warning": (
                "reference and candidate latency medians are descriptive marginal "
                "medians; their quotient is not reported as a paired speedup"
            ),
            "cross_campaign_warning": (
                "campaigns used different locked physical GPUs; compare only paired "
                "ratios recorded within each result file"
            ),
        },
        "campaigns": campaigns,
        "result_file_count": len(rows),
        "invalid_json_files": invalid_json,
        "unclassified_result_files": unclassified,
        "separate_validation_lanes": {
            "hardened_same_gpu": hardened_lane,
            "exact_fixed_model_qk_ncu": qk_ncu_lane,
            "corrected_tp4_dp4_ep4_live_trace": tp4_lane,
        },
        "series": aggregates,
        "results": rows,
    }

    json_path = evidence_dir / "paired_results_summary.json"
    csv_path = evidence_dir / "paired_results_summary.csv"
    markdown_path = evidence_dir / "paired_results_summary.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _write_csv(csv_path, aggregates)
    _write_markdown(
        markdown_path,
        campaigns,
        aggregates,
        unclassified,
        hardened_lane,
        qk_ncu_lane,
        tp4_lane,
    )
    print(json.dumps({
        "result_files": len(rows),
        "series": len(aggregates),
        "outputs": [str(json_path), str(csv_path), str(markdown_path)],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
