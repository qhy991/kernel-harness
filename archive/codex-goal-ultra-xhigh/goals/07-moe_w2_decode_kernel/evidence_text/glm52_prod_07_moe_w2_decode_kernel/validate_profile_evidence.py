#!/usr/bin/env python3
"""Fail-closed audit for one selected production-W2 profiling attempt."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import importlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any


EVIDENCE = Path(__file__).resolve().parent
ROOT = EVIDENCE.parents[1]
TARGET_KERNEL = "sm100_fp8_fp4_gemm_1d1d"
ALIGNMENTS = (16, 32, 64, 96, 128)
BUCKETS = {
    "m16": "moe_w2_grouped_decode_m16",
    "m32": "moe_w2_grouped_decode_m32",
    "m16_current_source_m5": "moe_w2_grouped_decode_m16_current_source_m5",
    "m32_current_source_m9": "moe_w2_grouped_decode_m32_current_source_m9",
}
KINDS = {
    "nsys": ("nsys", ".nsys-rep", "nsys_metadata", "nsys_cuda_gpu_kern_sum"),
    "ncu_full": ("full", ".ncu-rep", "ncu_full_metadata", "ncu_full_details"),
    "ncu_source": ("source", ".ncu-rep", "ncu_source_metadata", "ncu_source_details"),
}
NSYS_TRACES = {
    "api": "nsys_cuda_api_trace",
    "gpu": "nsys_cuda_gpu_trace",
    "kernel_exec": "nsys_cuda_kern_exec_trace",
}
REQUIRED_FULL_METRICS = {
    "launch_occupancy": (
        "launch__grid_size",
        "launch__block_size",
        "launch__cluster_size",
        "launch__waves_per_multiprocessor",
        "launch__registers_per_thread",
        "launch__shared_mem_per_block",
        "launch__shared_mem_per_block_dynamic",
        "launch__occupancy_limit_registers",
        "launch__occupancy_limit_shared_mem",
        "launch__occupancy_limit_warps",
        "sm__maximum_warps_per_active_cycle_pct",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
    ),
    "block_balance": (
        "sm__cycles_active.avg",
        "sm__cycles_active.max",
        "sm__cycles_active.min",
        "sm__cycles_active.sum",
    ),
    "scheduler_stalls": (
        "smsp__warps_eligible.avg.per_cycle_active",
        "smsp__issue_active.avg.pct_of_peak_sustained_active",
        "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
        "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
        "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
        "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
        "smsp__pcsamp_sample_count",
        "smsp__pcsamp_warps_issue_stalled_long_scoreboard",
        "smsp__pcsamp_warps_issue_stalled_barrier",
        "smsp__pcsamp_warps_issue_stalled_selected",
    ),
    "tensor_core": (
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
        "sm__pipe_tensor_subpipe_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed",
    ),
    "pm_timeline": (
        "pmsampling:smsp__warps_issue_stalled_long_scoreboard.avg",
        "pmsampling:smsp__warps_issue_stalled_short_scoreboard.avg",
        "pmsampling:smsp__warps_issue_stalled_wait.avg",
        "pmsampling:smsp__warps_issue_stalled_barrier.avg",
    ),
    "memory_access": (
        "dram__bytes_read.sum",
        "dram__bytes_read.sum.pct_of_peak_sustained_elapsed",
        "dram__bytes_write.sum",
        "dram__bytes_write.sum.pct_of_peak_sustained_elapsed",
        "l1tex__t_sector_hit_rate.pct",
        "lts__t_sector_hit_rate.pct",
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
        "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
        "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio",
        "smsp__sass_inst_executed_op_local_ld.sum",
        "smsp__sass_inst_executed_op_local_st.sum",
    ),
}

if str(EVIDENCE) not in sys.path:
    sys.path.insert(0, str(EVIDENCE))
import summarize_paired as campaign  # noqa: E402


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_directory(path: Path, expected_parent: Path) -> Path:
    expect(path.parent.resolve() == expected_parent.resolve(), f"unsafe directory path: {path}")
    expect(path.is_dir() and not path.is_symlink(), f"missing or linked directory: {path}")
    expect(path.resolve() == path, f"non-canonical directory path: {path}")
    return path


def require_artifact(path: Path, expected_parent: Path) -> Path:
    expect(path.parent.resolve() == expected_parent.resolve(), f"artifact escaped attempt: {path}")
    expect(path.is_file() and not path.is_symlink(), f"missing or linked artifact: {path}")
    expect(path.stat().st_size > 0, f"empty artifact: {path}")
    expect(path.resolve() == path, f"non-canonical artifact path: {path}")
    return path


def atomic_write_new_or_check(path: Path, text: str) -> None:
    if path.exists():
        expect(path.is_file() and not path.is_symlink(), f"unsafe existing summary: {path}")
        expect(path.read_text() == text, f"existing validated summary drifted: {path}")
        return
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_ncu_report() -> Any:
    candidates = [
        Path("/opt/nvidia/nsight-compute/2026.1.1/extras/python"),
        Path("/opt/nvidia/nsight-compute/2025.3.1/extras/python"),
        Path("/opt/nvidia/nsight-compute/2025.1.1/extras/python"),
    ]
    for candidate in candidates:
        if (candidate / "ncu_report.py").is_file():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            break
    try:
        return importlib.import_module("ncu_report")
    except Exception as exc:  # pragma: no cover - host installation failure
        raise RuntimeError("Nsight Compute Python bindings are required for strict audit") from exc


def one_target_action(ncu_report: Any, report_path: Path) -> Any:
    report = ncu_report.load_report(report_path)
    actions = []
    total_actions = 0
    for range_index in range(report.num_ranges()):
        report_range = report.range_by_idx(range_index)
        for action_index in range(report_range.num_actions()):
            action = report_range.action_by_idx(action_index)
            total_actions += 1
            if TARGET_KERNEL in action.name():
                actions.append(action)
    expect(len(actions) == 1, f"{report_path}: expected exactly one target action; got {len(actions)}")
    return actions[0], total_actions


def audit_full_report(ncu_report: Any, report_path: Path) -> dict[str, Any]:
    action, total_actions = one_target_action(ncu_report, report_path)
    names = set(action.metric_names())
    families = {
        "timing": [name for name in names if name.startswith("gpu__time_duration")],
        "launch": [name for name in names if name.startswith("launch__")],
        "compute": [name for name in names if name.startswith(("sm__", "gpu__compute"))],
        "memory": [name for name in names if name.startswith(("dram__", "l1tex__", "lts__"))],
        "scheduler": [name for name in names if name.startswith("smsp__")],
        "pm_sampling": [name for name in names if name.startswith("pmsampling:")],
        "tensor": [name for name in names if "tensor" in name],
        "eligible_warps": [name for name in names if "warps_eligible" in name],
        "local_memory": [name for name in names if "op_local" in name],
        "global_store": [name for name in names if "global_st" in name],
    }
    missing = sorted(key for key, values in families.items() if not values)
    expect(not missing, f"{report_path}: missing full-profile metric families: {missing}")
    expect("launch__registers_per_thread" in names, f"{report_path}: register metric missing")
    expect(any("waves_per_multiprocessor" in name for name in names), f"{report_path}: CTA-wave metric missing")
    required_values: dict[str, dict[str, float | int]] = {}
    for dimension, metric_names in REQUIRED_FULL_METRICS.items():
        required_values[dimension] = {}
        for name in metric_names:
            expect(name in names, f"{report_path}: required {dimension} metric missing: {name}")
            metric = action.metric_by_name(name)
            expect(metric is not None, f"{report_path}: required metric unavailable: {name}")
            value = metric.value()
            expect(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value)),
                f"{report_path}: required metric is not finite numeric: {name}={value!r}",
            )
            required_values[dimension][name] = value
    return {
        "kernel": action.name(),
        "total_actions": total_actions,
        "target_actions": 1,
        "metric_count": len(names),
        "metric_family_counts": {key: len(values) for key, values in sorted(families.items())},
        "required_metric_values": required_values,
    }


def audit_source_report(ncu_report: Any, report_path: Path) -> dict[str, Any]:
    action, total_actions = one_target_action(ncu_report, report_path)
    names = set(action.metric_names())
    source_names = sorted(name for name in names if name.startswith("smsp__pcsamp_"))
    stall_names = [name for name in source_names if "warps_issue_stalled" in name]
    expect("smsp__pcsamp_sample_count" in names, f"{report_path}: SourceCounters sample count missing")
    expect(stall_names, f"{report_path}: SourceCounters stall metrics missing")
    correlated_instances = 0
    correlated_lines: set[tuple[str, int]] = set()
    for name in stall_names:
        metric = action.metric_by_name(name)
        if metric is None or metric.num_instances() == 0 or not metric.has_correlation_ids():
            continue
        correlation_ids = metric.correlation_ids()
        for index in range(metric.num_instances()):
            source = action.source_info(correlation_ids.as_uint64(index))
            if source is not None and source.file_name() and int(source.line()) > 0:
                correlated_instances += 1
                correlated_lines.add((source.file_name(), int(source.line())))
    expect(correlated_instances > 0, f"{report_path}: no SourceCounters PC mapped to a source line")
    return {
        "kernel": action.name(),
        "total_actions": total_actions,
        "target_actions": 1,
        "metric_count": len(names),
        "source_counter_metrics": len(source_names),
        "correlated_instances": correlated_instances,
        "correlated_source_lines": len(correlated_lines),
    }


def canonical_nsys_field(field: str) -> str:
    """Normalize an Nsys CSV field while retaining semantic punctuation."""

    return re.sub(r"\s+\([^)]*\)$", "", field.strip()).casefold()


def read_nsys_csv(path: Path, required_fields: set[str]) -> tuple[list[dict[str, str]], list[str]]:
    """Read Nsys CSV after any human preamble without altering the raw extract."""

    lines = path.read_text(encoding="utf-8", errors="strict").splitlines(keepends=True)
    required = {canonical_nsys_field(field) for field in required_fields}
    header_index = None
    for index, line in enumerate(lines):
        try:
            fields = next(csv.reader([line]))
        except csv.Error:
            continue
        canonical = [canonical_nsys_field(field) for field in fields]
        if required.issubset(canonical):
            header_index = index
            break
    expect(header_index is not None, f"{path}: Nsys CSV header missing after optional preamble")
    reader = csv.DictReader(io.StringIO("".join(lines[header_index:])))
    raw_fields = reader.fieldnames or []
    canonical_fields = [canonical_nsys_field(field) for field in raw_fields]
    expect(len(canonical_fields) == len(set(canonical_fields)), f"{path}: duplicate canonical CSV fields")
    field_map = dict(zip(raw_fields, canonical_fields, strict=True))
    rows = [
        {field_map[field]: (value or "").strip() for field, value in row.items() if field is not None}
        for row in reader
        if any((value or "").strip() for value in row.values())
    ]
    expect(rows, f"{path}: Nsys CSV has no data rows")
    return rows, canonical_fields


def nsys_number(path: Path, row: dict[str, str], field: str, *, optional: bool = False) -> int | float | None:
    value = row.get(canonical_nsys_field(field), "").replace(",", "").strip()
    if not value and optional:
        return None
    expect(bool(value), f"{path}: missing numeric field {field}")
    try:
        number = float(value)
    except ValueError as exc:
        raise RuntimeError(f"{path}: invalid numeric field {field}={value!r}") from exc
    expect(math.isfinite(number), f"{path}: non-finite numeric field {field}={value!r}")
    return int(number) if number.is_integer() else number


def audit_nsys_extract(path: Path) -> dict[str, Any]:
    rows, fields = read_nsys_csv(path, {"instances"})
    name_field = next((field for field in fields if field in {"name", "kernel name"}), None)
    expect(name_field is not None, f"{path}: malformed Nsys kernel summary")
    targets = [row for row in rows if TARGET_KERNEL in row.get(name_field, "")]
    expect(len(targets) == 1, f"{path}: expected one aggregated target row; got {len(targets)}")
    occurrences = nsys_number(path, targets[0], "instances")
    expect(occurrences == 1, f"{path}: expected one target occurrence; got {occurrences}")
    return {"target_rows": 1, "target_occurrences": occurrences}


def audit_nsys_trace_exports(
    api_path: Path, gpu_path: Path, kernel_exec_path: Path
) -> dict[str, Any]:
    api_rows, _ = read_nsys_csv(api_path, {"start", "duration", "name", "corrid"})
    gpu_rows, _ = read_nsys_csv(
        gpu_path,
        {
            "start",
            "duration",
            "corrid",
            "grdx",
            "grdy",
            "grdz",
            "blkx",
            "blky",
            "blkz",
            "reg/trd",
            "stcsmem",
            "dymsmem",
            "device",
            "strm",
            "name",
        },
    )
    exec_rows, _ = read_nsys_csv(
        kernel_exec_path,
        {"api start", "api dur", "kernel start", "kernel dur", "total dur", "devid", "api function", "kernel name"},
    )
    gpu_targets = [row for row in gpu_rows if TARGET_KERNEL in row["name"]]
    exec_targets = [row for row in exec_rows if TARGET_KERNEL in row["kernel name"]]
    expect(len(gpu_targets) == 1, f"{gpu_path}: expected one target GPU trace row; got {len(gpu_targets)}")
    expect(
        len(exec_targets) == 1,
        f"{kernel_exec_path}: expected one target kernel-exec row; got {len(exec_targets)}",
    )
    gpu = gpu_targets[0]
    execution = exec_targets[0]
    corrid = nsys_number(gpu_path, gpu, "corrid")
    api_targets = [
        row
        for row in api_rows
        if nsys_number(api_path, row, "corrid") == corrid
    ]
    expect(len(api_targets) == 1, f"{api_path}: target CorrID {corrid} did not map to one API row")
    api = api_targets[0]
    expect(api["name"] == "cuLaunchKernelEx", f"{api_path}: unexpected target launch API {api['name']!r}")

    api_start = nsys_number(api_path, api, "start")
    api_duration = nsys_number(api_path, api, "duration")
    kernel_start = nsys_number(gpu_path, gpu, "start")
    kernel_duration = nsys_number(gpu_path, gpu, "duration")
    expect(nsys_number(kernel_exec_path, execution, "api start") == api_start, "Nsys API start mismatch")
    expect(nsys_number(kernel_exec_path, execution, "api dur") == api_duration, "Nsys API duration mismatch")
    expect(nsys_number(kernel_exec_path, execution, "kernel start") == kernel_start, "Nsys kernel start mismatch")
    expect(
        nsys_number(kernel_exec_path, execution, "kernel dur") == kernel_duration,
        "Nsys kernel duration mismatch",
    )
    expect(execution["api function"] == "cuLaunchKernelEx", "Nsys kernel-exec API mismatch")
    expect("B200" in gpu["device"], f"{gpu_path}: target did not execute on B200")
    expect(nsys_number(gpu_path, gpu, "grdx") == 148, f"{gpu_path}: target grid is not 148 CTAs")
    expect(nsys_number(gpu_path, gpu, "blkx") == 256, f"{gpu_path}: target block is not 256 threads")
    expect(nsys_number(gpu_path, gpu, "reg/trd") > 0, f"{gpu_path}: register count missing")

    synchronizations = [row for row in api_rows if row["name"] == "cudaDeviceSynchronize"]
    profiler_starts = [row for row in api_rows if row["name"] == "cuProfilerStart"]
    expect(len(profiler_starts) == 1, f"{api_path}: expected one cuProfilerStart")
    expect(synchronizations, f"{api_path}: expected a post-launch cudaDeviceSynchronize")
    expect(
        all(nsys_number(api_path, row, "start") >= api_start for row in synchronizations),
        f"{api_path}: synchronization preceded target launch",
    )
    return {
        "target_kernel": gpu["name"],
        "correlation_id": corrid,
        "api_function": api["name"],
        "api_start_ns": api_start,
        "api_duration_ns": api_duration,
        "kernel_start_ns": kernel_start,
        "kernel_duration_ns": kernel_duration,
        "total_duration_ns": nsys_number(kernel_exec_path, execution, "total dur"),
        "queue_start_ns": nsys_number(kernel_exec_path, execution, "queue start", optional=True),
        "queue_duration_ns": nsys_number(kernel_exec_path, execution, "queue dur", optional=True),
        "device": gpu["device"],
        "stream": nsys_number(gpu_path, gpu, "strm"),
        "grid": [
            nsys_number(gpu_path, gpu, "grdx"),
            nsys_number(gpu_path, gpu, "grdy"),
            nsys_number(gpu_path, gpu, "grdz"),
        ],
        "block": [
            nsys_number(gpu_path, gpu, "blkx"),
            nsys_number(gpu_path, gpu, "blky"),
            nsys_number(gpu_path, gpu, "blkz"),
        ],
        "registers_per_thread": nsys_number(gpu_path, gpu, "reg/trd"),
        "static_shared_memory_mb": nsys_number(gpu_path, gpu, "stcsmem"),
        "dynamic_shared_memory_mb": nsys_number(gpu_path, gpu, "dymsmem"),
        "device_synchronize_calls": len(synchronizations),
        "device_synchronize_total_ns": sum(
            nsys_number(api_path, row, "duration") for row in synchronizations
        ),
    }


def config_signature(log_path: Path) -> str:
    signatures = set()
    for line in log_path.read_text(errors="replace").splitlines():
        marker = "): GemmConfig("
        if "GemmDesc(gemm_type=2" in line and marker in line:
            signatures.add("GemmConfig(" + line.split(marker, 1)[1].strip())
    expect(len(signatures) == 1, f"{log_path}: expected one unique masked-GEMM config; got {len(signatures)}")
    return next(iter(signatures))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--alignment", type=int, choices=ALIGNMENTS, required=True)
    parser.add_argument("--attempt", required=True, help="generated profile attempt basename")
    args = parser.parse_args()

    manifest_path = args.manifest.resolve(strict=True)
    expect(manifest_path.is_file() and not args.manifest.is_symlink(), f"unsafe manifest: {args.manifest}")
    manifest = campaign._init_or_load_manifest(
        manifest_path, create=False, require_measurement_head=False
    )
    campaign_id = manifest["campaign_id"]
    attempt_pattern = re.compile(
        rf"profile_{re.escape(campaign_id)}_\d{{8}}T\d{{6}}Z_\d+_\d+"
    )
    expect(Path(args.attempt).name == args.attempt, "--attempt must be a basename, not a path")
    expect(attempt_pattern.fullmatch(args.attempt) is not None, "attempt name/campaign mismatch")

    run_name = "moe-w2-packed-baseline" if args.alignment == 128 else f"moe-w2-alignment{args.alignment}"
    profile_root = (ROOT / "profile").resolve()
    run_dir = require_directory(profile_root / run_name, profile_root)
    report_root = require_directory(run_dir / "reports", run_dir)
    profile_analysis_root = require_directory(run_dir / "analysis" / "profiles", run_dir / "analysis")
    report_dir = require_directory(report_root / args.attempt, report_root)
    attempt_dir = require_directory(profile_analysis_root / args.attempt, profile_analysis_root)
    derived_dir = require_directory(attempt_dir / "analysis", attempt_dir)
    nsys_trace_dir = require_directory(derived_dir / "nsys", derived_dir)
    cache_dir = require_directory(run_dir / "cache", run_dir)

    expected_reports = {
        f"{stem}_{bucket}{suffix}"
        for bucket in BUCKETS
        for stem, suffix, _, _ in KINDS.values()
    }
    expected_logs = {
        f"{stem}_{bucket}.log"
        for bucket in BUCKETS
        for stem, _, _, _ in KINDS.values()
    }
    expected_metadata = {
        f"{metadata_stem}_{bucket}.json"
        for bucket in BUCKETS
        for _, _, metadata_stem, _ in KINDS.values()
    }
    expected_extracts = {
        f"{extract_stem}_{bucket}.csv"
        for bucket in BUCKETS
        for _, _, _, extract_stem in KINDS.values()
    }
    observed_reports = {
        path.name
        for path in report_dir.iterdir()
        if path.name.endswith((".ncu-rep", ".nsys-rep"))
    }
    observed_logs = {path.name for path in report_dir.glob("*.log")}
    observed_metadata = {path.name for path in attempt_dir.glob("*_metadata_*.json")}
    observed_extracts = {path.name for path in attempt_dir.glob("*.csv")}
    expected_nsys_traces = {
        f"{stem}_{bucket}.csv"
        for bucket in BUCKETS
        for stem in NSYS_TRACES.values()
    }
    observed_nsys_traces = {path.name for path in nsys_trace_dir.glob("*.csv")}
    expect(observed_reports == expected_reports, "profile report set is incomplete or contaminated")
    expect(observed_logs == expected_logs, "profile log set is incomplete or contaminated")
    expect(observed_metadata == expected_metadata, "config metadata set is incomplete or contaminated")
    expect(observed_extracts == expected_extracts, "profile extract set is incomplete or contaminated")
    expect(
        observed_nsys_traces == expected_nsys_traces,
        "offline Nsys trace export set is incomplete or contaminated",
    )

    ncu_report = load_ncu_report()
    artifact_hashes: dict[str, str] = {str(manifest_path): sha256(manifest_path)}
    report_audits: dict[str, Any] = {}
    gpu_uuids: set[str] = set()
    gpu_names: set[str] = set()
    stock_sms: set[int] = set()
    config_signatures: set[str] = set()
    required_count = 0
    trace_count = 0

    for bucket, workload in BUCKETS.items():
        report_audits[workload] = {}
        for kind, (stem, suffix, metadata_stem, extract_stem) in KINDS.items():
            report = require_artifact(report_dir / f"{stem}_{bucket}{suffix}", report_dir)
            log = require_artifact(report_dir / f"{stem}_{bucket}.log", report_dir)
            metadata = require_artifact(attempt_dir / f"{metadata_stem}_{bucket}.json", attempt_dir)
            extract = require_artifact(attempt_dir / f"{extract_stem}_{bucket}.csv", attempt_dir)
            data = campaign.validate_config_metadata(
                metadata,
                manifest,
                workload=workload,
                alignment=args.alignment,
                num_sms=None,
            )
            expect(Path(data["config_log_path"]).resolve() == log, f"{metadata}: wrong exact config log")
            gpu_uuids.add(data["active_gpu"]["uuid"])
            gpu_names.add(data["active_gpu"]["name"])
            stock_sms.add(int(data["stock_num_sms"]))
            config_signatures.add(config_signature(log))

            if kind == "nsys":
                audit = audit_nsys_extract(extract)
                trace_paths = {
                    trace_kind: require_artifact(
                        nsys_trace_dir / f"{trace_stem}_{bucket}.csv", nsys_trace_dir
                    )
                    for trace_kind, trace_stem in NSYS_TRACES.items()
                }
                audit["trace_exports"] = audit_nsys_trace_exports(
                    trace_paths["api"], trace_paths["gpu"], trace_paths["kernel_exec"]
                )
                for trace_path in trace_paths.values():
                    artifact_hashes[str(trace_path.relative_to(ROOT))] = sha256(trace_path)
                    trace_count += 1
            elif kind == "ncu_full":
                audit = audit_full_report(ncu_report, report)
                expect(TARGET_KERNEL in extract.read_text(errors="replace"), f"{extract}: target absent")
            else:
                audit = audit_source_report(ncu_report, report)
                expect(TARGET_KERNEL in extract.read_text(errors="replace"), f"{extract}: target absent")
            report_audits[workload][kind] = audit
            for artifact in (report, log, metadata, extract):
                artifact_hashes[str(artifact.relative_to(ROOT))] = sha256(artifact)
            required_count += 1

    expect(required_count == 12, f"internal artifact matrix error: {required_count}")
    expect(trace_count == 12, f"internal Nsys trace matrix error: {trace_count}")
    expect(len(gpu_uuids) == len(gpu_names) == len(stock_sms) == 1, "mixed GPU/SM provenance")
    expect(all("B200" in name for name in gpu_names), f"profile was not collected on B200: {gpu_names}")
    expect(stock_sms == {148}, f"unexpected stock SM count: {stock_sms}")
    expect(len(config_signatures) == 1, "selected DeepGEMM config differs within attempt")

    cache_hashes = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in sorted(cache_dir.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }
    expect(cache_hashes, f"empty JIT cache: {cache_dir}")
    payload = {
        "schema_version": 2,
        "validation": "PASS",
        "campaign_id": campaign_id,
        "selection": {
            "alignment": args.alignment,
            "role": "stock" if args.alignment == 128 else "candidate",
            "attempt": args.attempt,
            "run": str(run_dir.relative_to(ROOT)),
        },
        "provenance": {
            "kernel_harness_head": manifest["git_heads"]["kernel_harness"],
            "sglang_head": manifest["git_heads"]["sglang"],
            "gpu_uuid": next(iter(gpu_uuids)),
            "gpu_name": next(iter(gpu_names)),
            "stock_num_sms": next(iter(stock_sms)),
            "jit_cache": str(cache_dir.relative_to(ROOT)),
            "selected_config_sha256": hashlib.sha256(next(iter(config_signatures)).encode()).hexdigest(),
            "ncu_python_module": str(Path(ncu_report.__file__).resolve()),
        },
        "artifact_counts": {
            "reports": 12,
            "config_metadata": 12,
            "logs": 12,
            "extracts": 12,
            "nsys_trace_exports": 12,
            "jit_cache_files": len(cache_hashes),
        },
        "reports": report_audits,
        "artifact_sha256": dict(sorted(artifact_hashes.items())),
        "jit_cache_sha256": dict(sorted(cache_hashes.items())),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    summary_path = attempt_dir / "validated_summary.json"
    atomic_write_new_or_check(summary_path, rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
