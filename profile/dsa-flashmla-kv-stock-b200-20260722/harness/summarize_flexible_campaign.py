#!/usr/bin/env python3
"""Validate and summarize one Goal 22 flexible-GPU measurement campaign.

The campaign is intentionally indivisible: all stock baselines and all paired
eager/CUDA-Graph sessions must carry the same scheduler identity.  This script
recomputes producer statistics from the raw samples and fails closed when the
campaign moved to another physical GPU, UUID, or logical CUDA ordinal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


GATE_SPEEDUP = 1.03
BUCKETS = ("m16", "m32")
SESSIONS = (1, 2, 3)
MODES = ("eager", "cuda_graph")
VARIANTS = ("control", "candidate")
SNAPSHOTS = (
    ("device_start.json", "start"),
    ("device_after_paired.json", "after_paired"),
    ("device_after_nsys.json", "after_nsys"),
    ("device_end.json", "end"),
)
FILE_PREFIX = {
    ("eager", "control"): "paired_control",
    ("eager", "candidate"): "paired_combine32",
    ("cuda_graph", "control"): "graph_control",
    ("cuda_graph", "candidate"): "graph_combine32",
}
EXPECTED_LABEL = {
    "control": "stock-pybind-tensor",
    "candidate": "combine32-m16-tensor",
}
SUMMARY_JSON = "paired_measurements_summary.json"
SUMMARY_MARKDOWN = "paired_measurements_summary.md"
WRAPPER_ALLOCATION_RE = re.compile(
    r"^allocated physical GPU ([0-3]) \((GPU-[^)]+)\) "
    r"as logical GPU 0 for one locked command$"
)


@dataclass(frozen=True)
class CampaignIdentity:
    campaign_id: str
    physical_gpu: int
    logical_gpu: int
    gpu_uuid: str


def _load(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise ValueError(f"missing required campaign artifact: {path}")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top-level JSON must be an object")
    return value, hashlib.sha256(raw).hexdigest()


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label}: expected an object")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}: expected a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label}: expected a finite number")
    return result


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label}: expected an integer")
    return value


def _close(actual: Any, expected: float, label: str) -> None:
    actual_number = _number(actual, label)
    if not math.isclose(actual_number, expected, rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError(f"{label}: stored {actual_number!r} != recomputed {expected!r}")


def _campaign_identity(value: Any, label: str) -> CampaignIdentity:
    item = _mapping(value, label)
    campaign_id = item.get("campaign_id")
    gpu_uuid = item.get("gpu_uuid")
    if not isinstance(campaign_id, str) or not campaign_id:
        raise ValueError(f"{label}.campaign_id: expected a non-empty string")
    if not isinstance(gpu_uuid, str) or not gpu_uuid:
        raise ValueError(f"{label}.gpu_uuid: expected a non-empty string")
    identity = CampaignIdentity(
        campaign_id=campaign_id,
        physical_gpu=_integer(item.get("physical_gpu"), f"{label}.physical_gpu"),
        logical_gpu=_integer(item.get("logical_gpu"), f"{label}.logical_gpu"),
        gpu_uuid=gpu_uuid,
    )
    _validate_identity_range(identity, label)
    return identity


def _snapshot_identity(value: dict[str, Any], label: str) -> CampaignIdentity:
    campaign_id = value.get("campaign_id")
    gpu_uuid = value.get("gpu_uuid")
    if not isinstance(campaign_id, str) or not campaign_id:
        raise ValueError(f"{label}.campaign_id: expected a non-empty string")
    if not isinstance(gpu_uuid, str) or not gpu_uuid:
        raise ValueError(f"{label}.gpu_uuid: expected a non-empty string")
    identity = CampaignIdentity(
        campaign_id=campaign_id,
        physical_gpu=_integer(
            value.get("physical_gpu_index"), f"{label}.physical_gpu_index"
        ),
        logical_gpu=_integer(
            value.get("logical_gpu_index"), f"{label}.logical_gpu_index"
        ),
        gpu_uuid=gpu_uuid,
    )
    _validate_identity_range(identity, label)
    return identity


def _validate_identity_range(identity: CampaignIdentity, label: str) -> None:
    if identity.physical_gpu not in range(4):
        raise ValueError(f"{label}: physical GPU must be one of 0,1,2,3")
    if identity.logical_gpu != 0:
        raise ValueError(f"{label}: flexible allocation must appear as logical GPU 0")


def _same_identity(
    actual: CampaignIdentity, expected: CampaignIdentity, label: str
) -> None:
    if actual != expected:
        raise ValueError(
            f"{label}: campaign identity mismatch: {asdict(actual)!r} "
            f"!= {asdict(expected)!r}"
        )


def _relative(path: Path, campaign_root: Path) -> str:
    return str(path.relative_to(campaign_root))


def _wrapper_allocation(
    campaign_root: Path, identity: CampaignIdentity
) -> dict[str, Any]:
    path = campaign_root / "wrapper.log"
    if not path.is_file():
        raise ValueError(f"missing required campaign artifact: {path}")
    raw = path.read_bytes()
    try:
        text = raw.decode()
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path}: wrapper log is not UTF-8") from exc
    matches = [
        match
        for line in text.splitlines()
        if (match := WRAPPER_ALLOCATION_RE.fullmatch(line))
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{path}: expected exactly one flexible-wrapper allocation line, "
            f"found {len(matches)}"
        )
    physical_gpu = int(matches[0].group(1))
    gpu_uuid = matches[0].group(2)
    if physical_gpu != identity.physical_gpu or gpu_uuid != identity.gpu_uuid:
        raise ValueError(
            f"{path}: wrapper allocation ({physical_gpu}, {gpu_uuid}) disagrees "
            f"with campaign identity {asdict(identity)!r}"
        )
    return {
        "raw_file": _relative(path, campaign_root),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "allocation_line": matches[0].group(0),
        "physical_gpu": physical_gpu,
        "logical_gpu": 0,
        "gpu_uuid": gpu_uuid,
    }


def _validate_environment(
    data: dict[str, Any], identity: CampaignIdentity, label: str
) -> None:
    environment = _mapping(data.get("environment_flags"), f"{label}.environment_flags")
    visible = environment.get("CUDA_VISIBLE_DEVICES")
    if visible != str(identity.physical_gpu):
        raise ValueError(
            f"{label}.environment_flags.CUDA_VISIBLE_DEVICES: {visible!r} "
            f"!= physical GPU {identity.physical_gpu}"
        )
    if environment.get("SGLANG_GLM52_OPT") != "0":
        raise ValueError(f"{label}: SGLANG_GLM52_OPT must be '0'")


def _validate_summary_samples(
    summary: Any, expected_count: int, label: str
) -> list[float]:
    value = _mapping(summary, label)
    samples = value.get("samples_ms")
    if not isinstance(samples, list) or len(samples) != expected_count:
        actual_count = len(samples) if isinstance(samples, list) else "not-a-list"
        raise ValueError(
            f"{label}.samples_ms: {actual_count} samples != repeat={expected_count}"
        )
    numbers = [_number(sample, f"{label}.samples_ms[{index}]") for index, sample in enumerate(samples)]
    if any(sample <= 0.0 for sample in numbers):
        raise ValueError(f"{label}.samples_ms: all timings must be positive")
    _close(value.get("median_ms"), statistics.median(numbers), f"{label}.median_ms")
    return numbers


def _expected_task(bucket: str) -> str:
    return f"dsa_flashmla_kv_decode_{bucket}"


def _validate_paired_samples(
    data: dict[str, Any], repeat: int, label: str
) -> tuple[list[float], list[float], list[float]]:
    reference = _validate_summary_samples(data.get("reference"), repeat, f"{label}.reference")
    candidate = _validate_summary_samples(data.get("candidate"), repeat, f"{label}.candidate")
    pairs = data.get("paired_samples")
    if not isinstance(pairs, list) or len(pairs) != repeat:
        actual_count = len(pairs) if isinstance(pairs, list) else "not-a-list"
        raise ValueError(f"{label}.paired_samples: {actual_count} pairs != repeat={repeat}")

    speedups: list[float] = []
    for index, raw_pair in enumerate(pairs):
        pair = _mapping(raw_pair, f"{label}.paired_samples[{index}]")
        if pair.get("pair") != index:
            raise ValueError(f"{label}.paired_samples[{index}]: unexpected pair index")
        expected_order = (
            ["reference", "candidate"]
            if index % 2 == 0
            else ["candidate", "reference"]
        )
        if pair.get("order") != expected_order:
            raise ValueError(
                f"{label}.paired_samples[{index}].order: "
                f"{pair.get('order')!r} != {expected_order!r}"
            )
        _close(pair.get("reference_ms"), reference[index], f"{label}.pair[{index}].reference_ms")
        _close(pair.get("candidate_ms"), candidate[index], f"{label}.pair[{index}].candidate_ms")
        speedup = reference[index] / candidate[index]
        _close(pair.get("speedup"), speedup, f"{label}.pair[{index}].speedup")
        speedups.append(speedup)
    return reference, candidate, speedups


def _measurement(
    path: Path,
    campaign_root: Path,
    mode: str,
    bucket: str,
    variant: str,
    session: int,
    identity: CampaignIdentity,
) -> dict[str, Any]:
    data, digest = _load(path)
    label = _relative(path, campaign_root)
    _same_identity(_campaign_identity(data.get("campaign"), f"{label}.campaign"), identity, label)
    _validate_environment(data, identity, label)

    if "task" in data:
        task = data["task"]
    else:
        task = _mapping(data.get("workload"), f"{label}.workload").get("name")
    if task != _expected_task(bucket):
        raise ValueError(f"{label}: task {task!r} does not match filename bucket {bucket}")
    warmup = _integer(data.get("warmup"), f"{label}.warmup")
    repeat = _integer(data.get("repeat"), f"{label}.repeat")
    if warmup < 0 or repeat < 1:
        raise ValueError(f"{label}: invalid warmup/repeat ({warmup}, {repeat})")
    reference, candidate, speedups = _validate_paired_samples(data, repeat, label)
    paired_median = statistics.median(speedups)
    expected_gate = paired_median >= GATE_SPEEDUP

    evidence = _mapping(data.get("candidate_evidence"), f"{label}.candidate_evidence")
    if evidence.get("label") != EXPECTED_LABEL[variant]:
        raise ValueError(
            f"{label}: candidate label {evidence.get('label')!r} "
            f"does not match filename variant {variant!r}"
        )
    extension_sha = evidence.get("extension_sha256")
    if not isinstance(extension_sha, str) or len(extension_sha) != 64:
        raise ValueError(f"{label}: invalid candidate extension SHA-256")

    if mode == "eager":
        if data.get("execution_mode") != "eager_cuda_event":
            raise ValueError(f"{label}: expected eager_cuda_event")
        candidate_summary = _mapping(data.get("candidate"), f"{label}.candidate")
        _close(candidate_summary.get("speedup"), paired_median, f"{label}.candidate.speedup")
        gate = candidate_summary.get("passes_3pct_median_gate")
        correctness_basis = "producer pre-timing reference/candidate comparison completed"
    else:
        if data.get("mode") != "real_cuda_graph_replay":
            raise ValueError(f"{label}: expected real_cuda_graph_replay")
        _close(data.get("paired_median_speedup"), paired_median, f"{label}.paired_median_speedup")
        gate = data.get("passes_3pct_gate")
        correctness = _mapping(data.get("correctness"), f"{label}.correctness")
        required_true = (
            "initial_exact_dtype_and_tolerance",
            "mutated_inputs_match",
        )
        for field in required_true:
            if correctness.get(field) is not True:
                raise ValueError(f"{label}.correctness.{field}: expected true")
        if correctness.get("outputs_alias") is not False:
            raise ValueError(f"{label}.correctness.outputs_alias: expected false")
        reference_ptr = _integer(
            correctness.get("reference_output_data_ptr"),
            f"{label}.correctness.reference_output_data_ptr",
        )
        candidate_ptr = _integer(
            correctness.get("candidate_output_data_ptr"),
            f"{label}.correctness.candidate_output_data_ptr",
        )
        if reference_ptr == candidate_ptr:
            raise ValueError(f"{label}: graph reference and candidate outputs alias")
        for field in (
            "mutated_reference_change_max_abs",
            "mutated_candidate_change_max_abs",
        ):
            if _number(correctness.get(field), f"{label}.correctness.{field}") <= 1e-3:
                raise ValueError(f"{label}.correctness.{field}: stale graph replay")
        correctness_basis = "explicit graph match, mutation, and anti-alias checks"

    if not isinstance(gate, bool) or gate != expected_gate:
        raise ValueError(
            f"{label}: stored 3% gate {gate!r} disagrees with paired median "
            f"{paired_median!r}"
        )

    return {
        "mode": mode,
        "bucket": bucket,
        "variant": variant,
        "session": session,
        "raw_file": label,
        "raw_sha256": digest,
        "task": task,
        "warmup": warmup,
        "repeat": repeat,
        "reference_median_ms": statistics.median(reference),
        "candidate_median_ms": statistics.median(candidate),
        "paired_median_speedup": paired_median,
        "passes_3pct_gate": gate,
        "correctness_pass": True,
        "correctness_basis": correctness_basis,
        "candidate_label": evidence["label"],
        "candidate_extension_sha256": extension_sha,
        "candidate_source_commit": evidence.get("source_commit"),
    }


def _snapshot(
    path: Path,
    campaign_root: Path,
    expected_stage: str,
    identity: CampaignIdentity | None,
) -> tuple[dict[str, Any], CampaignIdentity]:
    data, digest = _load(path)
    label = _relative(path, campaign_root)
    actual = _snapshot_identity(data, label)
    if identity is not None:
        _same_identity(actual, identity, label)
    if data.get("stage") != expected_stage:
        raise ValueError(f"{label}: stage {data.get('stage')!r} != {expected_stage!r}")

    smi = _mapping(data.get("nvidia_smi"), f"{label}.nvidia_smi")
    if smi.get("uuid") != actual.gpu_uuid:
        raise ValueError(f"{label}: nvidia-smi UUID disagrees with campaign UUID")
    try:
        smi_index = int(smi.get("index"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}.nvidia_smi.index: expected an integer") from exc
    if smi_index != actual.physical_gpu:
        raise ValueError(f"{label}: nvidia-smi index disagrees with physical GPU")

    environment = _mapping(data.get("environment"), f"{label}.environment")
    expected_environment = {
        "CUDA_VISIBLE_DEVICES": str(actual.physical_gpu),
        "GOAL22_CAMPAIGN_ID": actual.campaign_id,
        "GOAL22_PHYSICAL_GPU": str(actual.physical_gpu),
        "GOAL22_GPU_UUID": actual.gpu_uuid,
    }
    for key, expected in expected_environment.items():
        if environment.get(key) != expected:
            raise ValueError(
                f"{label}.environment.{key}: {environment.get(key)!r} != {expected!r}"
            )

    return (
        {
            "stage": expected_stage,
            "raw_file": label,
            "raw_sha256": digest,
            "recorded_at_utc": data.get("recorded_at_utc"),
            "pstate": smi.get("pstate"),
            "graphics_clock_mhz": smi.get("clocks.current.graphics"),
            "sm_clock_mhz": smi.get("clocks.current.sm"),
            "memory_clock_mhz": smi.get("clocks.current.memory"),
            "power_draw_w": smi.get("power.draw"),
            "temperature_c": smi.get("temperature.gpu"),
        },
        actual,
    )


def _baseline(
    path: Path,
    campaign_root: Path,
    bucket: str,
    session: int,
    identity: CampaignIdentity,
) -> dict[str, Any]:
    data, digest = _load(path)
    label = _relative(path, campaign_root)
    _same_identity(_campaign_identity(data.get("campaign"), f"{label}.campaign"), identity, label)
    _validate_environment(data, identity, label)
    if data.get("execution_mode") != "eager_cuda_event":
        raise ValueError(f"{label}: expected eager_cuda_event")
    if data.get("candidate") is not None:
        raise ValueError(f"{label}: stock baseline must not contain a candidate")
    workload = _mapping(data.get("workload"), f"{label}.workload")
    if workload.get("name") != _expected_task(bucket):
        raise ValueError(f"{label}: workload does not match filename bucket {bucket}")
    warmup = _integer(data.get("warmup"), f"{label}.warmup")
    repeat = _integer(data.get("repeat"), f"{label}.repeat")
    if warmup < 0 or repeat < 1:
        raise ValueError(f"{label}: invalid warmup/repeat ({warmup}, {repeat})")
    samples = _validate_summary_samples(data.get("reference"), repeat, f"{label}.reference")
    return {
        "bucket": bucket,
        "session": session,
        "raw_file": label,
        "raw_sha256": digest,
        "warmup": warmup,
        "repeat": repeat,
        "reference_median_ms": statistics.median(samples),
    }


def _groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
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
                if len(selected) != len(SESSIONS):
                    raise AssertionError("internal error: incomplete measurement group")
                evidence_identities = {
                    (
                        row["candidate_label"],
                        row["candidate_extension_sha256"],
                        row["candidate_source_commit"],
                    )
                    for row in selected
                }
                if len(evidence_identities) != 1:
                    raise ValueError(
                        f"{mode}/{bucket}/{variant}: candidate build identity changed "
                        "within the campaign"
                    )
                speedups = [row["paired_median_speedup"] for row in selected]
                passes = sum(row["passes_3pct_gate"] for row in selected)
                result.append(
                    {
                        "mode": mode,
                        "bucket": bucket,
                        "variant": variant,
                        "session_paired_median_speedups": speedups,
                        "median_session_paired_speedup": statistics.median(speedups),
                        "minimum_session_paired_speedup": min(speedups),
                        "maximum_session_paired_speedup": max(speedups),
                        "sessions_passing_3pct_gate": passes,
                        "multiple_sessions_pass_3pct_gate": passes > 1,
                    }
                )
    return result


def _validate_variant_builds(rows: list[dict[str, Any]]) -> None:
    for variant in VARIANTS:
        identities = {
            (
                row["candidate_label"],
                row["candidate_extension_sha256"],
                row["candidate_source_commit"],
            )
            for row in rows
            if row["variant"] == variant
        }
        if len(identities) != 1:
            raise ValueError(
                f"{variant}: candidate build identity changed across the campaign"
            )


def _build_summary(campaign_root: Path) -> dict[str, Any]:
    analysis = campaign_root / "analysis"
    if not analysis.is_dir():
        raise ValueError(f"missing campaign analysis directory: {analysis}")

    snapshots: list[dict[str, Any]] = []
    identity: CampaignIdentity | None = None
    for filename, stage in SNAPSHOTS:
        row, actual = _snapshot(analysis / filename, campaign_root, stage, identity)
        if identity is None:
            identity = actual
        snapshots.append(row)
    assert identity is not None
    wrapper_allocation = _wrapper_allocation(campaign_root, identity)

    baselines = [
        _baseline(
            analysis / f"baseline_stock_{bucket}_r{session}.json",
            campaign_root,
            bucket,
            session,
            identity,
        )
        for bucket in BUCKETS
        for session in SESSIONS
    ]
    rows = [
        _measurement(
            analysis
            / f"{FILE_PREFIX[(mode, variant)]}_{bucket}_r{session}.json",
            campaign_root,
            mode,
            bucket,
            variant,
            session,
            identity,
        )
        for mode in MODES
        for bucket in BUCKETS
        for variant in VARIANTS
        for session in SESSIONS
    ]
    if len(rows) != 24:
        raise AssertionError("internal error: expected exactly 24 paired artifacts")

    _validate_variant_builds(rows)
    groups = _groups(rows)
    candidate_groups = [group for group in groups if group["variant"] == "candidate"]
    drift = {
        bucket: (
            next(
                row["reference_median_ms"]
                for row in baselines
                if row["bucket"] == bucket and row["session"] == 3
            )
            / next(
                row["reference_median_ms"]
                for row in baselines
                if row["bucket"] == bucket and row["session"] == 1
            )
            - 1.0
        )
        for bucket in BUCKETS
    }
    return {
        "schema_version": 1,
        "generated_by": "harness/summarize_flexible_campaign.py",
        "campaign": asdict(identity),
        "wrapper_allocation": wrapper_allocation,
        "method": {
            "gate_speedup": GATE_SPEEDUP,
            "paired_speedup_definition": (
                "median of per-pair reference_ms/candidate_ms speedups"
            ),
            "scheduler_contract": (
                "all artifacts share one flexible-wrapper physical GPU/UUID and "
                "observe it as logical GPU 0"
            ),
        },
        "device_snapshots": snapshots,
        "stock_baselines": {
            "acceptance_use": "context only; excluded from paired gates",
            "measurements": baselines,
            "r1_to_r3_relative_change": drift,
        },
        "measurements": rows,
        "groups": groups,
        "outcome": {
            "artifact_count": len(rows),
            "all_correctness_checks_pass": all(row["correctness_pass"] for row in rows),
            "candidate_groups_with_multiple_3pct_sessions": sum(
                group["multiple_sessions_pass_3pct_gate"]
                for group in candidate_groups
            ),
            "candidate_sessions_passing_3pct_gate": sum(
                group["sessions_passing_3pct_gate"] for group in candidate_groups
            ),
        },
    }


def _fmt(value: Any) -> str:
    return "" if value is None else str(value)


def _render_markdown(summary: dict[str, Any]) -> str:
    identity = summary["campaign"]
    lines = [
        "# Flexible-GPU paired campaign summary",
        "",
        f"- Campaign: `{identity['campaign_id']}`",
        f"- Scheduler allocation: physical GPU {identity['physical_gpu']} "
        f"(`{identity['gpu_uuid']}`) exposed as logical GPU {identity['logical_gpu']}",
        f"- Wrapper evidence: `{summary['wrapper_allocation']['raw_file']}` "
        f"(SHA-256 `{summary['wrapper_allocation']['raw_sha256']}`)",
        f"- Validated raw paired artifacts: {summary['outcome']['artifact_count']}/24",
        f"- All producer correctness checks passed: "
        f"`{str(summary['outcome']['all_correctness_checks_pass']).lower()}`",
        "",
        "The paired p50 is the median of the raw per-pair "
        "`reference_ms / candidate_ms` ratios; the 3% gate is `>= 1.03`.",
        "",
        "## Device snapshots",
        "",
        "| Stage | Raw file | P-state | Graphics MHz | SM MHz | Memory MHz | Power W | Temp C |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["device_snapshots"]:
        lines.append(
            "| {stage} | `{raw}` | {pstate} | {graphics} | {sm} | {memory} | {power} | {temperature} |".format(
                stage=row["stage"],
                raw=row["raw_file"],
                pstate=_fmt(row["pstate"]),
                graphics=_fmt(row["graphics_clock_mhz"]),
                sm=_fmt(row["sm_clock_mhz"]),
                memory=_fmt(row["memory_clock_mhz"]),
                power=_fmt(row["power_draw_w"]),
                temperature=_fmt(row["temperature_c"]),
            )
        )

    lines.extend(
        [
            "",
            "## Stock baseline context",
            "",
            "| Bucket | Session | Reference p50 (ms) | Raw file |",
            "|---|---:|---:|---|",
        ]
    )
    for row in summary["stock_baselines"]["measurements"]:
        lines.append(
            f"| {row['bucket'].upper()} | {row['session']} | "
            f"{row['reference_median_ms']!r} | `{row['raw_file']}` |"
        )

    lines.extend(
        [
            "",
            "## Paired measurements",
            "",
            "| Mode | Bucket | Variant | Session | Reference p50 (ms) | Candidate p50 (ms) | Paired p50 speedup | Gate | Correctness |",
            "|---|---|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in summary["measurements"]:
        lines.append(
            "| {mode} | {bucket} | {variant} | {session} | {reference!r} | "
            "{candidate!r} | {speedup!r} | {gate} | PASS |".format(
                mode=row["mode"],
                bucket=row["bucket"].upper(),
                variant=row["variant"],
                session=row["session"],
                reference=row["reference_median_ms"],
                candidate=row["candidate_median_ms"],
                speedup=row["paired_median_speedup"],
                gate="PASS" if row["passes_3pct_gate"] else "FAIL",
            )
        )

    lines.extend(
        [
            "",
            "## Three-session groups",
            "",
            "| Mode | Bucket | Variant | Session speedups | Median session speedup | 3% passes | Repeated gate |",
            "|---|---|---|---|---:|---:|---|",
        ]
    )
    for group in summary["groups"]:
        speedups = ", ".join(repr(value) for value in group["session_paired_median_speedups"])
        lines.append(
            "| {mode} | {bucket} | {variant} | {speedups} | {median!r} | "
            "{passes}/3 | {repeated} |".format(
                mode=group["mode"],
                bucket=group["bucket"].upper(),
                variant=group["variant"],
                speedups=speedups,
                median=group["median_session_paired_speedup"],
                passes=group["sessions_passing_3pct_gate"],
                repeated="PASS" if group["multiple_sessions_pass_3pct_gate"] else "FAIL",
            )
        )

    lines.extend(
        [
            "",
            "## Validation outcome",
            "",
            f"- Candidate groups with repeated 3% sessions: "
            f"{summary['outcome']['candidate_groups_with_multiple_3pct_sessions']}/4.",
            f"- Candidate sessions passing the 3% gate: "
            f"{summary['outcome']['candidate_sessions_passing_3pct_gate']}/12.",
            f"- Baseline r1-to-r3 change: M16 "
            f"{summary['stock_baselines']['r1_to_r3_relative_change']['m16']!r}; "
            f"M32 {summary['stock_baselines']['r1_to_r3_relative_change']['m32']!r}.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_or_check(path: Path, content: str, check: bool) -> None:
    if check:
        if not path.is_file():
            raise SystemExit(f"missing generated campaign summary: {path}")
        if path.read_text() != content:
            raise SystemExit(f"stale generated campaign summary: {path}")
        return
    try:
        with path.open("x") as handle:
            handle.write(content)
    except FileExistsError as exc:
        raise SystemExit(
            f"refusing to overwrite campaign summary: {path}; use --check to verify"
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument(
        "--check", action="store_true", help="verify existing summaries without writing"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    campaign_root = Path(args.campaign_root).expanduser().resolve()
    if not campaign_root.is_dir():
        raise ValueError(f"campaign root is not a directory: {campaign_root}")
    analysis = campaign_root / "analysis"
    summary = _build_summary(campaign_root)
    json_text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    markdown_text = _render_markdown(summary)

    outputs = (analysis / SUMMARY_JSON, analysis / SUMMARY_MARKDOWN)
    if not args.check and any(path.exists() for path in outputs):
        existing = ", ".join(str(path) for path in outputs if path.exists())
        raise SystemExit(
            f"refusing to overwrite campaign summary: {existing}; use --check to verify"
        )
    _write_or_check(outputs[0], json_text, args.check)
    _write_or_check(outputs[1], markdown_text, args.check)
    action = "verified" if args.check else "wrote"
    print(f"{action}: {outputs[0]}")
    print(f"{action}: {outputs[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
