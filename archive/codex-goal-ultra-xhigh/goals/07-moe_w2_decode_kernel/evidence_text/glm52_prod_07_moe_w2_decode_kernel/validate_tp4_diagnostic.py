#!/usr/bin/env python3
"""Fail-closed validator for one goal-07 TP4 diagnostic attempt.

This validates four-rank diagnostic evidence only.  It deliberately cannot
produce or imply TP8/DP8/EP8 acceptance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import statistics
import subprocess
from pathlib import Path
from typing import Any


EVIDENCE = Path(__file__).resolve().parent
ATTEMPT_ROOT = (EVIDENCE / "tp4_diagnostic").resolve()
COLLECTOR = EVIDENCE / "run_tp4_diagnostics.sh"
WORKLOADS = (
    "ep4_deepep_ll_dispatch_decode_m16",
    "ep4_deepep_ll_combine_decode_m16",
    "ep4_deepep_ll_moe_region_decode_m16",
    "ep4_deepep_ll_dispatch_decode_m32",
    "ep4_deepep_ll_combine_decode_m32",
    "ep4_deepep_ll_moe_region_decode_m32",
)


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    expect(path.is_file() and path.stat().st_size > 0, f"missing JSON artifact: {path}")
    data = json.loads(path.read_text())
    expect(isinstance(data, dict), f"{path}: expected JSON object")
    return data


def atomic_write_new_or_check(path: Path, text: str) -> None:
    if path.exists():
        expect(path.read_text() == text, f"existing summary drifted: {path}")
        return
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def expected_labels() -> list[str]:
    labels = ["runtime_probe", "topology", "nvlink", "gpu_state_before"]
    labels.extend(
        f"{workload}_trial{trial}"
        for trial in (1, 2, 3)
        for workload in WORKLOADS
    )
    labels.extend(f"nsys_{workload}" for workload in WORKLOADS)
    labels.append("gpu_state_after")
    return labels


def validate_manifest(attempt: Path) -> tuple[dict[str, str], list[dict[str, Any]]]:
    path = attempt / "manifest.tsv"
    expect(path.is_file(), f"missing TP4 manifest: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    labels = expected_labels()
    expect(
        [row["label"] for row in rows[: len(labels)]] == labels,
        "TP4 manifest prefix is incomplete, duplicated, or reordered",
    )
    tail = rows[len(labels) :]
    expect(
        [row["label"] for row in tail] in ([], ["validate_tp4"], ["validate_tp4", "failures"]),
        f"unexpected TP4 manifest tail: {tail}",
    )
    by_label: dict[str, str] = {}
    for row in rows[: len(labels)]:
        expect(row["exit_code"] == "0", f"TP4 step failed: {row}")
        log = Path(row["log"]).resolve()
        expect(log == (attempt / "logs" / f"{row['label']}.log"), f"wrong log path: {row}")
        expect(log.is_file(), f"missing TP4 log: {log}")
        by_label[row["label"]] = str(log)
    validation_history: list[dict[str, Any]] = []
    for row in tail:
        try:
            exit_code = int(row["exit_code"])
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"invalid TP4 validation-tail exit code: {row}") from exc
        validation_history.append({"label": row["label"], "exit_code": exit_code})

    # A post-collection validator rerun is authoritative over an earlier
    # validator-only failure: every collection command above must still be
    # present and zero, and this invocation independently rechecks every raw
    # result/report before it can emit PASS. Preserve the earlier tail in the
    # summary instead of rewriting the immutable collection manifest.
    return by_label, validation_history


def validate_runtime_probe(attempt: Path) -> dict[str, Any]:
    data = load_json(attempt / "runtime_probe.json")
    expect(data.get("schema_version") == 2, "wrong TP4 runtime-probe schema")
    expect(
        data.get("evidence_scope") == "diagnostic_tp4_not_production_tp8",
        "TP4 runtime probe was relabeled",
    )
    expect(data.get("cuda_visible_devices") == "0,1,2,3", "wrong visible GPUs")
    expect(data.get("visible_device_count") == 4, "TP4 probe did not see four GPUs")
    devices = data.get("devices", [])
    expect(len(devices) == 4, "TP4 probe device inventory is incomplete")
    expect(
        all(item.get("name") and "B200" in item["name"] and item.get("capability") == [10, 0] for item in devices),
        f"TP4 probe did not run on four SM100 B200s: {devices}",
    )
    expect(Path(data["paths"]["attempt"]).resolve() == attempt, "probe attempt path drifted")
    expect(
        Path(data["paths"]["deep_gemm_cache"]).resolve() == attempt / "deep_gemm_cache",
        "probe cache path is not attempt-local",
    )
    env = data["contract_environment"]
    zero_keys = (
        "SGLANG_GLM52_OPT",
        "SGLANG_JIT_DEEPGEMM_PRECOMPILE",
        "SGLANG_JIT_DEEPGEMM_FAST_WARMUP",
        "SGL_DG_USE_NVRTC",
        "DG_JIT_USE_NVRTC",
        "SGLANG_DEEPGEMM_SANITY_CHECK",
    )
    expect(all(env.get(key) == "0" for key in zero_keys), "probe zero-policy drift")
    expect(env.get("SGLANG_DEEPGEMM_PDL") == "true", "probe PDL policy drifted")
    expect(
        data["nvrtc_derivation"]["declared_dg_jit_use_nvrtc"] == "0"
        and data["nvrtc_derivation"]["derived_dg_jit_use_nvrtc"] == "0",
        "SGLang NVRTC derivation was not frozen",
    )
    expect(
        all(value and value != "None" for value in data["configs"].values()),
        "DeepEP config capture is empty",
    )
    return data


def validate_summary(values: dict[str, Any], where: str) -> None:
    required = ("min_ms", "median_ms", "p95_ms")
    for key in required:
        value = values.get(key)
        expect(isinstance(value, (int, float)) and math.isfinite(value) and value > 0, f"{where}: invalid {key}")
    expect(values["min_ms"] <= values["median_ms"], f"{where}: min exceeds median")
    expect(values["median_ms"] <= values["p95_ms"], f"{where}: median exceeds p95")


def validate_topology(attempt: Path, logs: dict[str, str]) -> dict[str, Any]:
    inventories: dict[str, list[dict[str, str]]] = {}
    for label in ("gpu_state_before", "gpu_state_after"):
        path = Path(logs[label])
        rows = list(csv.reader(path.read_text().splitlines(), skipinitialspace=True))
        expect(len(rows) == 4 and all(len(row) == 10 for row in rows), f"{path}: GPU inventory drifted")
        expect([row[0] for row in rows] == ["0", "1", "2", "3"], f"{path}: GPU indices drifted")
        expect(all(row[2] == "NVIDIA B200" for row in rows), f"{path}: non-B200 device")
        uuids = [row[1] for row in rows]
        expect(len(set(uuids)) == 4, f"{path}: GPU UUIDs are not unique")
        inventories[label] = [
            {"index": row[0], "uuid": row[1], "name": row[2], "pci_bus_id": row[3]}
            for row in rows
        ]
    expect(
        [row["uuid"] for row in inventories["gpu_state_before"]]
        == [row["uuid"] for row in inventories["gpu_state_after"]],
        "GPU inventory changed during TP4 collection",
    )

    topology_path = Path(logs["topology"])
    topology = topology_path.read_text(errors="strict")
    expect(topology.count("NV18") == 12, f"{topology_path}: expected all 12 directed GPU pairs to be NV18")
    nvlink_path = Path(logs["nvlink"])
    nvlink = nvlink_path.read_text(errors="strict")
    active_links = re.findall(r"\bLink\s+\d+:\s+53\.125 GB/s", nvlink)
    expect(len(active_links) == 72, f"{nvlink_path}: expected 18 active 53.125 GB/s links per GPU")
    return {
        "gpu_inventory": inventories["gpu_state_before"],
        "directed_gpu_pairs_nv18": 12,
        "active_nvlink_status_records": 72,
        "nvlink_rate_gb_per_s": 53.125,
    }


def validate_workload_log(path: Path) -> None:
    text = path.read_text(errors="replace")
    markers = {
        "deepep_communication_sms": "Only use 20 SMs for DeepEP communication",
        "ibgda_transport": "init failed for transport: IBGDA",
        "nccl_device_mapping": "Guessing device ID based on global rank",
    }
    missing = [name for name, marker in markers.items() if marker not in text]
    expect(not missing, f"{path}: missing diagnostic-environment markers: {missing}")


def validate_scale(contract: dict[str, Any], *, groups: int, mn: int, k: int, where: str) -> None:
    packed_k = k // 512
    aligned_mn = ((mn + 3) // 4) * 4
    expect(contract.get("shape") == [groups, mn, packed_k], f"{where}: scale shape drifted")
    expect(
        contract.get("stride") == [aligned_mn * packed_k, 1, aligned_mn],
        f"{where}: scale stride drifted",
    )
    expect(contract.get("dtype") == "torch.int32", f"{where}: scale dtype drifted")
    expect(
        contract.get("logical_granularity_k") == 128
        and contract.get("packed_scales_per_int32") == 4
        and contract.get("aligned_mn") == aligned_mn,
        f"{where}: packed UE8M0 metadata drifted",
    )


def validate_region_correctness(data: dict[str, Any], *, local_m: int, where: str) -> None:
    correctness = data["correctness"]
    expect(correctness.get("fresh_post_timing_inputs_checked") is True, f"{where}: no fresh check")
    fresh = correctness.get("fresh_external_input_storage")
    expect(fresh and fresh.get("all_storage_distinct") is True, f"{where}: fresh storage reused")
    expected_keys = [
        "x_bf16", "topk_idx", "topk_weights", "w13_weight_fp8",
        "w13_weight_scale", "w2_weight_fp8", "w2_weight_scale",
    ]
    expect(fresh.get("checked_keys") == expected_keys, f"{where}: fresh keys drifted")

    validation = correctness.get("ep4_region_validation")
    expected_tokens = local_m * 4 * 8
    expect(validation.get("status") == "pass", f"{where}: region validation failed")
    expect(
        validation.get("scope") == "lower_level_eager_no_overlap_ep4_not_production_graph"
        and validation.get("validation_kind") == "structural_contract_and_repeatability"
        and validation.get("independent_math_oracle") is False,
        f"{where}: region scope was relabeled",
    )
    expect(validation.get("actual_group_size") == 4, f"{where}: group size drifted")
    expect(
        validation.get("global_sent_tokens")
        == validation.get("global_received_tokens")
        == validation.get("expected_global_tokens")
        == expected_tokens,
        f"{where}: token conservation failed",
    )
    expect(validation.get("recv_count_bounds") == [0, 512], f"{where}: recv bounds drifted")
    expect(validation.get("handle_scalar_contract") == [128, 6144, 256], f"{where}: handle drifted")
    expect(
        validation.get("valid_outputs_finite") is True
        and validation.get("fresh_repeat_allclose_passed") is True,
        f"{where}: output repeatability/finite gate failed",
    )
    validate_scale(validation["dispatch_scale_contract"], groups=64, mn=512, k=6144, where=f"{where}: dispatch")
    validate_scale(validation["w13_weight_scale_contract"], groups=64, mn=4096, k=6144, where=f"{where}: W13 weight")
    validate_scale(validation["w2_activation_scale_contract"], groups=64, mn=512, k=2048, where=f"{where}: W2 activation")
    validate_scale(validation["w2_weight_scale_contract"], groups=64, mn=6144, k=2048, where=f"{where}: W2 weight")

    handoff = correctness.get("ep4_handoff_contract")
    expect(handoff.get("probe_excluded_from_timing") is True, f"{where}: handoff probe was timed")
    expect(
        handoff["data"] == {
            "shape": [64, 512, 2048],
            "stride": [1048576, 2048, 1],
            "dtype": "torch.float8_e4m3fn",
        },
        f"{where}: W2 data handoff drifted",
    )
    validate_scale(handoff["scale"], groups=64, mn=512, k=2048, where=f"{where}: W2 scale handoff")


def validate_result(path: Path, workload: str) -> dict[str, Any]:
    data = load_json(path)
    where = str(path)
    local_m = 16 if workload.endswith("m16") else 32
    if "moe_region" in workload:
        family = "deepep_ll_moe_region"
        scope = "diagnostic_ep4_lower_level_eager_no_overlap"
    elif "dispatch" in workload:
        family = "deepep_ll_dispatch"
        scope = "diagnostic_ep4_not_production"
    else:
        family = "deepep_ll_combine"
        scope = "diagnostic_ep4_not_production"
    item = data.get("workload", {})
    expect(data.get("schema_version") == 2, f"{where}: wrong schema")
    expect(item.get("name") == workload, f"{where}: wrong workload")
    expect(item.get("family") == family and item.get("world_size") == 4, f"{where}: wrong family/world")
    expect(item.get("phase") == "decode", f"{where}: wrong phase")
    expect(data.get("evidence_scope") == scope, f"{where}: evidence scope was relabeled")
    expect(data.get("candidate") is None, f"{where}: TP4 baseline unexpectedly has a candidate")
    expect(data.get("reference_policy") == "SGLANG_GLM52_OPT=0 production path", f"{where}: replacement enabled")
    expect(data.get("execution_mode") == "eager_cuda_event", f"{where}: wrong execution mode")
    expect(data.get("timing_contract") == "maximum CUDA-event latency across ranks", f"{where}: not rank-max timing")
    validate_summary(data["reference"], where)
    expect(data["correctness"].get("reference_completed") is True, f"{where}: reference incomplete")

    params = item.get("params", {})
    common = {"local_tokens": local_m, "max_dispatch_tokens": 128, "hidden": 6144, "experts": 256, "topk": 8}
    expect({key: params.get(key) for key in common} == common, f"{where}: common EP4 params drifted")
    if family == "deepep_ll_moe_region":
        region = {
            "experts_per_rank": 64, "expert_slab": 512,
            "expected_m": 3 if local_m == 16 else 5, "group_size": 128,
            "w13_k": 6144, "w13_n": 4096, "w2_k": 2048, "w2_n": 6144,
        }
        expect({key: params.get(key) for key in region} == region, f"{where}: region params drifted")
        pdl = data.get("deep_gemm_pdl_policy", {})
        expect(
            pdl.get("applicable") is True
            and pdl.get("requested") is True
            and pdl.get("active_during_setup_and_measurement") is True
            and pdl.get("active_before_restore") is True
            and pdl.get("restored") is True,
            f"{where}: PDL lifecycle failed",
        )
        validate_region_correctness(data, local_m=local_m, where=where)
    else:
        expect(data.get("deep_gemm_pdl_policy", {}).get("applicable") is False, f"{where}: unexpected PDL scope")
    return data


def read_nsys_kernel_summary(path: Path) -> tuple[list[dict[str, str]], str]:
    """Read an Nsys CSV after its optional human-readable status preamble."""

    lines = path.read_text(encoding="utf-8", errors="strict").splitlines(keepends=True)
    header_index = None
    for index, line in enumerate(lines):
        try:
            fields = next(csv.reader([line]))
        except csv.Error:
            continue
        canonical = [field.strip().casefold() for field in fields]
        if "instances" in canonical and any(name in canonical for name in ("name", "kernel name")):
            header_index = index
            break
    expect(header_index is not None, f"Nsys kernel-summary header missing: {path}")
    reader = csv.DictReader(io.StringIO("".join(lines[header_index:])))
    fields = reader.fieldnames or []
    name_field = next(
        (field for field in fields if field.strip().casefold() in {"name", "kernel name"}),
        None,
    )
    expect(name_field is not None, f"Nsys kernel-summary name field missing: {path}")
    rows = [row for row in reader if any((value or "").strip() for value in row.values())]
    expect(rows, f"empty Nsys kernel summary: {path}")
    return rows, name_field


def nsys_summary(attempt: Path, workload: str) -> dict[str, Any]:
    report = attempt / "profiles" / f"{workload}.nsys-rep"
    expect(report.is_file() and report.stat().st_size > 0, f"missing Nsys report: {report}")
    analysis = attempt / "analysis"
    analysis.mkdir(exist_ok=True)
    extract = analysis / f"nsys_cuda_gpu_kern_sum_{workload}.csv"
    sqlite = analysis / f"nsys_{workload}.sqlite"
    if not extract.exists():
        completed = subprocess.run(
            [
                "nsys", "stats", "--quiet", "--force-export=true", "--sqlite", str(sqlite),
                "--report", "cuda_gpu_kern_sum", "--format", "csv", "--output", "-",
                str(report),
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        extract.write_text(completed.stdout)
    rows, name_field = read_nsys_kernel_summary(extract)
    if "moe_region" in workload:
        expect(
            any("sm100_fp8_fp4_gemm_1d1d" in row.get(name_field, "") for row in rows),
            f"W2 kernel absent from region Nsys report: {report}",
        )
    return {
        "report": str(report.relative_to(attempt)),
        "report_bytes": report.stat().st_size,
        "report_sha256": sha256(report),
        "kernel_summary": str(extract.relative_to(attempt)),
        "kernel_summary_sha256": sha256(extract),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt", type=Path)
    args = parser.parse_args()
    attempt = args.attempt.resolve()
    expect(attempt.parent == ATTEMPT_ROOT, f"attempt escaped TP4 root: {attempt}")
    expect(attempt.name.startswith("tp4_"), f"invalid TP4 attempt name: {attempt.name}")
    logs, validation_history = validate_manifest(attempt)
    probe = validate_runtime_probe(attempt)
    topology = validate_topology(attempt, logs)

    artifact_hashes: dict[str, str] = {}
    trial_medians: dict[str, list[float]] = {workload: [] for workload in WORKLOADS}
    for trial in (1, 2, 3):
        for workload in WORKLOADS:
            path = attempt / "results" / f"{workload}_trial{trial}.json"
            data = validate_result(path, workload)
            trial_medians[workload].append(float(data["reference"]["median_ms"]))
            artifact_hashes[str(path.relative_to(attempt))] = sha256(path)

    nsys: dict[str, Any] = {}
    for workload in WORKLOADS:
        result_path = attempt / "results" / f"nsys_{workload}.json"
        validate_result(result_path, workload)
        artifact_hashes[str(result_path.relative_to(attempt))] = sha256(result_path)
        nsys[workload] = nsys_summary(attempt, workload)

    for label, raw_path in logs.items():
        path = Path(raw_path)
        expect(path.stat().st_size > 0, f"empty TP4 command log: {path}")
        if label.startswith("ep4_deepep_ll_") or label.startswith("nsys_ep4_deepep_ll_"):
            validate_workload_log(path)
        artifact_hashes[str(path.relative_to(attempt))] = sha256(path)
    probe_path = attempt / "runtime_probe.json"
    artifact_hashes[str(probe_path.relative_to(attempt))] = sha256(probe_path)
    manifest_path = attempt / "manifest.tsv"
    artifact_hashes[str(manifest_path.relative_to(attempt))] = sha256(manifest_path)

    timing = {
        workload: {
            "trial_median_ms": values,
            "median_of_trial_medians_ms": statistics.median(values),
            "min_trial_median_ms": min(values),
            "max_trial_median_ms": max(values),
        }
        for workload, values in trial_medians.items()
    }
    payload = {
        "schema_version": 1,
        "validation": "PASS",
        "evidence_scope": "TP4/DP4/EP4 diagnostic only; never TP8/DP8/EP8 acceptance",
        "attempt": attempt.name,
        "runtime_probe": {
            "kernel_harness_head": probe["git"]["kernel_harness"]["head"],
            "sglang_head": probe["git"]["sglang"]["head"],
            "deep_ep_extension_sha256": probe["deep_ep"]["extension_sha256"],
            "deep_gemm_extension_sha256": probe["deep_gemm"]["extension_sha256"],
            "devices": probe["devices"],
            "deep_ep_config_capture": "opaque_object_presence_only_not_semantic_config",
        },
        "topology": topology,
        "artifact_counts": {
            "baseline_results": 18,
            "nsys_results": 6,
            "nsys_reports": 6,
            "collection_command_logs": 29,
        },
        "collection_contract": {
            "collector": str(COLLECTOR.relative_to(EVIDENCE.parent.parent)),
            "collector_sha256": sha256(COLLECTOR),
            "baseline_warmup": 3,
            "baseline_repeat": 10,
            "nsys_warmup": 1,
            "nsys_repeat": 3,
        },
        "diagnostic_environment": {
            "deep_ep_communication_sms": 20,
            "ibgda_transport_initialization": "failed",
            "process_group_nccl_device_mapping": "guessed_from_global_rank",
            "timing_interpretation": "fallback_environment_diagnostic_not_tuned_production_communication",
        },
        "prior_validation_manifest_tail": validation_history,
        "timing": timing,
        "nsys": nsys,
        "artifact_sha256": dict(sorted(artifact_hashes.items())),
        "production_gates": {
            "tp8_full_region": "BLOCKED_NOT_RUN_ON_FOUR_GPU_HOST",
            "eight_rank_sglang_e2e": "BLOCKED_NOT_RUN_ON_FOUR_GPU_HOST",
        },
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    atomic_write_new_or_check(attempt / "summary.json", rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
