#!/usr/bin/env python3
"""Fail-closed audit for the immutable M4096 indexer validation bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any


REGION = "indexer_fused_prepare_store_prefill_m4096_eager_dual_stream"
ISOLATED = "indexer_wk_weights_prefill_m4096"
FRESH_CONTRACT = (
    "pre-timing shared-input dual-poison full-cache comparison plus post-timing "
    "fresh-seed dual-poison full-cache comparison; "
    "exact dtype/shape, floating rtol=2e-2/atol=2e-2, integer byte equality"
)
SHARED_CONTRACT = (
    "pre-timing shared-input comparison plus post-timing shared-input comparison; "
    "exact dtype/shape, floating rtol=2e-2/atol=2e-2, integer byte equality"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _strict_equal(observed: Any, expected: Any) -> bool:
    """JSON equality that does not conflate bool with int/float."""
    if type(observed) is not type(expected):
        return False
    if isinstance(expected, dict):
        return observed.keys() == expected.keys() and all(
            _strict_equal(observed[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(observed) == len(expected) and all(
            _strict_equal(left, right) for left, right in zip(observed, expected)
        )
    return observed == expected


def _integer(value: Any, *, label: str, expected: int | None = None) -> int:
    _require(type(value) is int, f"{label}: expected JSON integer, not boolean")
    if expected is not None:
        _require(value == expected, f"{label}: expected {expected}, got {value}")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    _require(path.is_file() and path.stat().st_size > 0, f"missing JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _close(observed: float, expected: float) -> bool:
    return math.isclose(observed, expected, rel_tol=1e-10, abs_tol=1e-12)


def _number(value: Any, *, label: str, positive: bool = False) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label}: expected JSON number, not boolean or string",
    )
    converted = float(value)
    _require(math.isfinite(converted), f"{label}: expected finite JSON number")
    if positive:
        _require(converted > 0, f"{label}: expected positive JSON number")
    return converted


def _positive_finite(values: Any, *, label: str, count: int) -> list[float]:
    _require(isinstance(values, list), f"{label}: expected list")
    _require(len(values) == count, f"{label}: expected {count}, got {len(values)}")
    _require(
        all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values),
        f"{label}: samples must be JSON numbers, not booleans or strings",
    )
    converted = [float(value) for value in values]
    _require(
        all(math.isfinite(value) and value > 0 for value in converted),
        f"{label}: samples must be finite and positive",
    )
    return converted


def _validate_workload(
    result: dict[str, Any], expected_name: str, repo: Path
) -> None:
    workload = result.get("workload")
    _require(isinstance(workload, dict), "missing workload object")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from serving_native.workloads import as_dict, get_workload

    expected = as_dict(get_workload(expected_name))
    _require(
        _strict_equal(workload, expected),
        f"wrong complete workload contract for {expected_name}",
    )


def _validate_source_provenance(result: dict[str, Any], repo: Path) -> None:
    provenance = result.get("source_provenance")
    _require(isinstance(provenance, dict), "missing source_provenance")
    for stem, path in (
        ("runner", repo / "serving_native/runner.py"),
        ("workloads", repo / "serving_native/workloads.py"),
    ):
        _require(Path(provenance.get(f"{stem}_path", "")) == path, f"wrong {stem} path")
        _require(
            provenance.get(f"{stem}_sha256") == _sha256(path),
            f"wrong {stem} SHA-256",
        )


def _validate_summary(summary: dict[str, Any], samples: list[float], label: str) -> None:
    median = _number(summary.get("median_ms"), label=f"{label}: median", positive=True)
    minimum = _number(summary.get("min_ms"), label=f"{label}: min", positive=True)
    p95 = _number(summary.get("p95_ms"), label=f"{label}: p95", positive=True)
    _require(_close(median, statistics.median(samples)), f"{label}: median")
    _require(_close(minimum, min(samples)), f"{label}: min")
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, max(0, int(0.95 * len(ordered)) - 1))
    _require(_close(p95, ordered[p95_index]), f"{label}: p95")


def _validate_result(
    path: Path,
    *,
    repo: Path,
    workload: str,
    repeat: int,
    candidate: Path | None,
    backend: str | None,
) -> dict[str, Any]:
    result = _read_json(path)
    _integer(result.get("schema_version"), label=f"{path}: schema", expected=1)
    _validate_workload(result, workload, repo)
    _require(
        result.get("reference_policy") == "SGLANG_GLM52_OPT=0 production path",
        f"{path}: wrong reference policy",
    )
    expected_execution = (
        "eager_dual_stream" if workload == REGION else "eager_cuda_event"
    )
    _require(result.get("execution_mode") == expected_execution, f"{path}: execution mode")
    _validate_source_provenance(result, repo)
    reference_samples = _positive_finite(
        result.get("reference_samples_ms"), label=f"{path}: reference", count=repeat
    )
    reference = result.get("reference")
    _require(isinstance(reference, dict), f"{path}: missing reference summary")
    _validate_summary(reference, reference_samples, f"{path}: reference")

    observed_candidate = result.get("candidate")
    if candidate is None:
        _require(observed_candidate is None, f"{path}: unexpected candidate")
        expected_timing = "maximum CUDA-event latency across ranks"
        _require(result.get("timing_contract") == expected_timing, f"{path}: timing")
        if workload == REGION:
            _require(
                result.get("correctness_contract")
                == "pre-timing and post-timing fresh-seed reference dual-poison full-cache comparisons",
                f"{path}: wrong fused reference correctness contract",
            )
            details = result.get("correctness_details")
            _require(isinstance(details, dict), f"{path}: correctness details")
            for key in (
                "pre_timing_reference_dual_poison",
                "post_timing_reference_dual_poison",
            ):
                _require(details.get(key) == "pass", f"{path}: {key}")
            _integer(
                details.get("post_timing_input_generation"),
                label=f"{path}: fresh seed generation",
                expected=1,
            )
        else:
            _require(
                result.get("correctness_contract") == "reference execution only",
                f"{path}: wrong reference correctness contract",
            )
        return result

    _require(isinstance(observed_candidate, dict), f"{path}: missing candidate")
    _require(
        result.get("timing_contract")
        == "interleaved paired A/B; maximum CUDA-event latency across ranks",
        f"{path}: paired timing contract",
    )
    _require(Path(observed_candidate.get("path", "")) == candidate, f"{path}: candidate path")
    _require(observed_candidate.get("sha256") == _sha256(candidate), f"{path}: candidate SHA")
    expected_contract = FRESH_CONTRACT if workload == REGION else SHARED_CONTRACT
    _require(
        result.get("correctness_contract") == expected_contract,
        f"{path}: wrong hardened correctness contract",
    )
    details = result.get("correctness_details")
    _require(isinstance(details, dict), f"{path}: missing correctness details")
    if workload == REGION:
        for key in (
            "pre_timing_reference_dual_poison",
            "pre_timing_candidate_dual_poison",
            "post_timing_reference_dual_poison",
            "post_timing_candidate_dual_poison",
        ):
            _require(details.get(key) == "pass", f"{path}: {key}")
        _integer(
            details.get("post_timing_input_generation"),
            label=f"{path}: fresh seed generation",
            expected=1,
        )
    else:
        _require(details.get("pre_timing_candidate") == "pass", f"{path}: pre-check")
        _require(details.get("post_timing_candidate") == "pass", f"{path}: post-check")
    candidate_samples = _positive_finite(
        observed_candidate.get("samples_ms"),
        label=f"{path}: candidate",
        count=repeat,
    )
    _validate_summary(observed_candidate, candidate_samples, f"{path}: candidate")
    ratios = [ref / cand for ref, cand in zip(reference_samples, candidate_samples)]
    recorded_ratios = _positive_finite(
        observed_candidate.get("paired_speedups"),
        label=f"{path}: paired speedups",
        count=repeat,
    )
    _require(
        all(_close(observed, expected) for observed, expected in zip(recorded_ratios, ratios)),
        f"{path}: paired ratios do not recompute",
    )
    speedup = statistics.median(ratios)
    recorded_speedup = _number(
        observed_candidate.get("speedup"), label=f"{path}: speedup", positive=True
    )
    _require(_close(recorded_speedup, speedup), f"{path}: speedup")
    expected_gate = speedup >= 1.03
    _require(
        observed_candidate.get("passes_3pct_median_gate") is expected_gate,
        f"{path}: stale 3% gate",
    )
    ordered_ratios = sorted(ratios)
    p10 = ordered_ratios[min(len(ordered_ratios) - 1, int(0.1 * len(ordered_ratios)))]
    p90 = ordered_ratios[
        min(
            len(ordered_ratios) - 1,
            max(0, int(0.9 * len(ordered_ratios)) - 1),
        )
    ]
    recorded_p10 = _number(
        observed_candidate.get("paired_p10_speedup"),
        label=f"{path}: p10",
        positive=True,
    )
    recorded_p90 = _number(
        observed_candidate.get("paired_p90_speedup"),
        label=f"{path}: p90",
        positive=True,
    )
    _require(_close(recorded_p10, p10), f"{path}: p10")
    _require(_close(recorded_p90, p90), f"{path}: p90")
    metadata = observed_candidate.get("metadata")
    if backend is None:
        _require(metadata is None, f"{path}: identity candidate must have no metadata")
    else:
        _require(isinstance(metadata, dict), f"{path}: missing candidate metadata")
        _require(metadata.get("backend") == backend, f"{path}: wrong backend metadata")
        _require(
            _strict_equal(metadata.get("shape_guard_intended"), [4096, 160, 6144]),
            f"{path}: wrong candidate shape guard",
        )
    return result


def _profile_console_result(path: Path) -> dict[str, Any]:
    _require(path.is_file() and path.stat().st_size > 0, f"missing profile log: {path}")
    matches: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "event_total_ms" in value:
            matches.append(value)
    _require(len(matches) == 1, f"{path}: expected one profiler result JSON")
    result = matches[0]
    _number(
        result.get("event_total_ms"),
        label=f"{path}: CUDA-event duration",
        positive=True,
    )
    _integer(result.get("iterations"), label=f"{path}: iterations", expected=1)
    _require(
        result.get("cache_write_coverage_gate") == "dual-poison byte-exact replay",
        f"{path}: missing cache write-coverage gate",
    )
    poison_value_count = result.get(
        "cache_bytes_equal_default_poison_value_after_write"
    )
    _require(
        isinstance(poison_value_count, int)
        and not isinstance(poison_value_count, bool)
        and poison_value_count >= 0,
        f"{path}: invalid diagnostic poison-value count",
    )
    cache_metadata = result.get("outputs", {}).get("index_k_cache", {})
    cache_bytes = _integer(
        cache_metadata.get("bytes"), label=f"{path}: cache byte count"
    )
    _require(cache_bytes > 0, f"{path}: non-positive cache byte count")
    _require(
        poison_value_count <= cache_bytes,
        f"{path}: poison-value count exceeds cache size",
    )
    return result


def _validate_profile_trace_csv(path: Path, *, tag: str) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    _require(rows, f"{path}: header-only CUDA trace")
    range_tag = "stock" if tag == "stock" else "candidate"
    prefix = f"indexer-fused-prefill-m4096-eager-dual-{range_tag}/"
    scoped = [row for row in rows if row.get("Name", "").startswith(prefix)]
    _require(len(scoped) == 4, f"{path}: expected exactly four scoped GPU ops")

    def matches(row: dict[str, str], *, grid: str, aliases: tuple[str, ...]) -> bool:
        name = row.get("Name", "")
        return row.get("GrdX") == grid and any(alias in name for alias in aliases)

    operations = {
        "wq": [
            row
            for row in scoped
            if matches(row, grid="512", aliases=("nvjet_sm100_tst",))
        ],
        "wk": [
            row
            for row in scoped
            if matches(row, grid="128", aliases=("nvjet_sm100_tst",))
        ],
        "q": [
            row
            for row in scoped
            if matches(
                row,
                grid="32768",
                aliases=(
                    "fused_q_indexer_rope_hadamard_quant",
                    "main_q_indexer_rope_first_quant",
                ),
            )
        ],
        "k": [
            row
            for row in scoped
            if matches(
                row,
                grid="1024",
                aliases=(
                    "fused_k_indexer_norm_rope_store",
                    "dpsk_v32_k_indexer_norm_rope_store_p64",
                ),
            )
        ],
    }
    for operation, operation_rows in operations.items():
        _require(
            len(operation_rows) == 1,
            f"{path}: expected one scoped {operation}, got {len(operation_rows)}",
        )
    selected = {name: values[0] for name, values in operations.items()}

    def interval(row: dict[str, str]) -> tuple[int, int]:
        try:
            start = int(row["Start (ns)"])
            duration = int(row["Duration (ns)"])
        except (KeyError, ValueError) as error:
            raise AssertionError(f"{path}: invalid kernel interval") from error
        _require(start >= 0 and duration > 0, f"{path}: non-positive kernel interval")
        return start, start + duration

    intervals = {name: interval(row) for name, row in selected.items()}
    device_contexts = {
        (row.get("Device", ""), row.get("Ctx", "")) for row in selected.values()
    }
    _require(
        all(device.strip() and context.strip() for device, context in device_contexts),
        f"{path}: missing device/context identifier",
    )
    _require(len(device_contexts) == 1, f"{path}: operations span device/contexts")
    streams = {name: row.get("Strm", "") for name, row in selected.items()}
    _require(all(streams.values()), f"{path}: missing stream identifiers")
    if tag == "single-stream":
        _require(len(set(streams.values())) == 1, f"{path}: single-stream mismatch")
        temporal_order = [
            name for name, _ in sorted(intervals.items(), key=lambda item: item[1][0])
        ]
        _require(
            temporal_order == ["wk", "k", "wq", "q"],
            f"{path}: wrong supported single-stream order {temporal_order}",
        )
    else:
        _require(streams["wq"] == streams["k"], f"{path}: wq/K stream mismatch")
        _require(streams["wk"] == streams["q"], f"{path}: wk/Q stream mismatch")
        _require(streams["wq"] != streams["wk"], f"{path}: dual streams collapsed")
        stage_one_end = max(intervals["wq"][1], intervals["wk"][1])
        _require(
            intervals["q"][0] >= stage_one_end
            and intervals["k"][0] >= stage_one_end,
            f"{path}: stage-2 launched before both GEMMs completed",
        )
    return {
        "device_context": list(next(iter(device_contexts))),
        "intervals_ns": {name: list(value) for name, value in intervals.items()},
        "streams": streams,
    }


def _validate_nvtx_summary(path: Path, *, expected_range: str) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    matching = [row for row in rows if row.get("Range") == f":{expected_range}"]
    _require(len(matching) == 1, f"{path}: expected one exact NVTX summary row")
    row = matching[0]
    try:
        instances = int(row["Range Instances"])
        gpu_ops = int(row["Total GPU Ops"])
        projected_ns = int(row["Total Proj Time (ns)"])
        host_ns = int(row["Total Range Time (ns)"])
    except (KeyError, ValueError) as error:
        raise AssertionError(f"{path}: invalid NVTX summary") from error
    _require(instances == 1, f"{path}: expected one NVTX range instance")
    _require(gpu_ops == 4, f"{path}: expected four projected GPU ops")
    _require(projected_ns > 0 and host_ns > 0, f"{path}: non-positive NVTX spans")
    return {
        "host_range_ns": host_ns,
        "projected_gpu_span_ns": projected_ns,
        "range_instances": instances,
        "total_gpu_ops": gpu_ops,
    }


def _validate_profile(
    run_dir: Path,
    *,
    tag: str,
    repo: Path,
    candidate: Path | None,
) -> dict[str, Any]:
    profile_dir = run_dir / "profiles"
    report = profile_dir / f"nsys-{tag}.nsys-rep"
    _require(report.is_file() and report.stat().st_size > 0, f"missing Nsys report: {report}")
    for pattern in (
        f"nsys-{tag}_cuda_gpu_trace_nvtx-name_base.csv",
        f"nsys-{tag}_nvtx_gpu_proj_sum.csv",
    ):
        exported = profile_dir / pattern
        _require(exported.is_file() and exported.stat().st_size > 0, f"missing {exported}")
    trace_csv = profile_dir / f"nsys-{tag}_cuda_gpu_trace_nvtx-name_base.csv"
    kernel_mapping = _validate_profile_trace_csv(trace_csv, tag=tag)
    trace = _read_json(profile_dir / f"abi-{tag}.json")
    _require(trace.get("trace_kind") == "production_shaped_unbound_symbol_reconstruction", "trace kind")
    _validate_workload(trace, REGION, repo)
    mapping = trace.get("production_mapping")
    _require(isinstance(mapping, dict), "missing production mapping")
    _require(mapping.get("model") == "GlmMoeDsaForCausalLM", "profile model")
    _require(
        mapping.get("symbol") == "Indexer._fused_q_prepare_and_store",
        "profile source symbol",
    )
    _require(mapping.get("mode") == "eager_dual_stream", "profile production mode")
    _require(
        mapping.get("model_config_revision")
        == "nvidia/GLM-5.2-NVFP4@aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa",
        "profile model revision",
    )
    _require(
        mapping.get("executed_topology") == "world_size=1 rank-local reconstruction",
        "profile execution topology",
    )
    _require("TP8/DP8/EP8" in mapping.get("shape_mapping", ""), "profile shape mapping")
    mapped_source = Path(mapping.get("source", ""))
    _require(mapped_source.is_file(), "profile mapped source path")
    _require(mapping.get("source_sha256") == _sha256(mapped_source), "profile mapped source SHA")
    _require(trace.get("streams", {}).get("distinct") is True, "profile streams not distinct")
    expected_inputs = {
        "index_k_cache": ([64, 8448], "torch.uint8"),
        "out_cache_loc": ([4096], "torch.int64"),
        "positions": ([4096], "torch.int64"),
        "q_lora": ([4096, 2048], "torch.bfloat16"),
        "wk_weight": ([160, 6144], "torch.bfloat16"),
        "wq_weight": ([4096, 2048], "torch.bfloat16"),
        "x": ([4096, 6144], "torch.bfloat16"),
    }
    for name, (shape, dtype) in expected_inputs.items():
        metadata = trace.get("inputs", {}).get(name, {})
        _require(
            _strict_equal(metadata.get("shape"), shape),
            f"profile input {name} shape",
        )
        _require(metadata.get("dtype") == dtype, f"profile input {name} dtype")
    provenance = trace.get("source_provenance")
    _require(isinstance(provenance, dict), "profile source provenance")
    for stem, source in (
        ("runner", repo / "serving_native/runner.py"),
        ("workloads", repo / "serving_native/workloads.py"),
    ):
        _require(provenance.get(f"{stem}_sha256") == _sha256(source), f"profile {stem} SHA")
    helper_path = Path(provenance.get("profile_helper_path", ""))
    _require(helper_path.is_file(), "profile helper path")
    _require(
        provenance.get("profile_helper_sha256") == _sha256(helper_path),
        "profile helper SHA",
    )
    expected_range = (
        "indexer-fused-prefill-m4096-eager-dual-stock"
        if tag == "stock"
        else "indexer-fused-prefill-m4096-eager-dual-candidate"
    )
    _require(
        _strict_equal(
            trace.get("profile"),
            {"iterations": 1, "nvtx_range": expected_range, "warmup": 10},
        ),
        "profile capture contract",
    )
    nvtx_summary = _validate_nvtx_summary(
        profile_dir / f"nsys-{tag}_nvtx_gpu_proj_sum.csv",
        expected_range=expected_range,
    )
    observed_candidate = trace.get("candidate")
    if candidate is None:
        _require(observed_candidate is None, "stock profile unexpectedly has candidate")
        expected_correctness = {
            "pre_timing_reference_dual_poison": "pass",
            "post_timing_fresh_seed_generation": "pass",
            "post_timing_reference_dual_poison": "pass",
        }
    else:
        _require(isinstance(observed_candidate, dict), "candidate profile metadata")
        _require(Path(observed_candidate.get("path", "")) == candidate, "profile candidate path")
        _require(observed_candidate.get("sha256") == _sha256(candidate), "profile candidate SHA")
        expected_correctness = {
            "pre_timing_reference_dual_poison": "pass",
            "pre_timing_candidate_dual_poison": "pass",
            "post_timing_fresh_seed_generation": "pass",
            "post_timing_reference_dual_poison": "pass",
            "post_timing_candidate_dual_poison": "pass",
        }
    _require(
        trace.get("correctness") == expected_correctness,
        "profile hardened correctness",
    )
    console_result = _profile_console_result(run_dir / "logs" / f"profile-{tag}.log")
    expected_outputs = {
        "index_k_cache": ([64, 8448], "torch.uint8"),
        "q_fp8": ([4096, 32, 128], "torch.float8_e4m3fn"),
        "weights": ([4096, 32, 1], "torch.float32"),
    }
    for name, (shape, dtype) in expected_outputs.items():
        metadata = console_result.get("outputs", {}).get(name, {})
        _require(
            _strict_equal(metadata.get("shape"), shape),
            f"profile output {name} shape",
        )
        _require(metadata.get("dtype") == dtype, f"profile output {name} dtype")
    return {
        "event_total_ms": console_result["event_total_ms"],
        "kernel_mapping": kernel_mapping,
        "nvtx_summary": nvtx_summary,
        "report": str(report),
        "trace": str(profile_dir / f"abi-{tag}.json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    repo = args.repo.resolve()
    results_dir = run_dir / "results"
    source_dir = run_dir / "source"

    for required in (
        run_dir / "environment.txt",
        run_dir / "environment_after.txt",
        run_dir / "check_env.txt",
        run_dir / "pip_freeze.txt",
        run_dir / "module_origins.json",
        run_dir / "source_manifest.sha256",
        run_dir / "source_manifest_check.txt",
        run_dir / "jit_artifact_manifest.sha256",
        run_dir / "jit_artifact_manifest_check.txt",
        run_dir / "status.txt",
    ):
        _require(required.is_file() and required.stat().st_size > 0, f"missing {required}")
    environment = (run_dir / "environment.txt").read_text(encoding="utf-8")
    for marker in (
        "fixed_model_revision=aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa",
        "selected_physical_gpu=",
        "wrapper_lock_canonical_path=",
        "wrapper_lock_fd=",
        "python_prefix=",
        "sgl_kernel_root=",
        "sgl_kernel_init=",
        "sgl_kernel_elementwise=",
        "sgl_kernel_sm100_common_ops=",
        "sgl_kernel_elementwise_matches_source=true",
        "TVM_FFI_CACHE_DIR=",
        "PYTHONNOUSERSITE=1",
        "PYTHONSAFEPATH=1",
        "CUDA_DEVICE_ORDER=PCI_BUS_ID",
        "CUDA_LAUNCH_BLOCKING=unset",
        "CUDA_DEVICE_MAX_CONNECTIONS=unset",
        "CUDA_CACHE_DISABLE=unset",
        "CUDA_FORCE_PTX_JIT=unset",
        "CUBLAS_WORKSPACE_CONFIG=unset",
        "NVIDIA_TF32_OVERRIDE=unset",
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=unset",
        "SGLANG_GLM52_OPT=0",
        "SGLANG_GLM52_NSYS_GATE=0",
        "RANK=0 LOCAL_RANK=0 WORLD_SIZE=1",
        "GPU-",
        "NVIDIA B200",
    ):
        _require(marker in environment, f"environment missing {marker}")
    environment_values = {}
    for line in environment.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            environment_values.setdefault(key, value)
    python_prefix = Path(environment_values["python_prefix"]).resolve()
    expected_sglang_python = Path(environment_values["SGLANG_ROOT"]).resolve() / "python"
    module_origins = _read_json(run_dir / "module_origins.json")
    _require(
        set(module_origins) == {"cutlass", "sgl_kernel", "sglang", "torch", "tvm_ffi"},
        "wrong module-origin key set",
    )
    for module_name, record in module_origins.items():
        _require(isinstance(record, dict), f"module origin record: {module_name}")
        origin = Path(record.get("origin", ""))
        _require(origin.is_file(), f"module origin missing: {module_name}: {origin}")
        expected_root = expected_sglang_python if module_name == "sglang" else python_prefix
        _require(
            origin.resolve().is_relative_to(expected_root),
            f"module origin outside expected root: {module_name}: {origin}",
        )
        _require(
            record.get("sha256") == _sha256(origin),
            f"module origin SHA mismatch: {module_name}",
        )
    check_env = (run_dir / "check_env.txt").read_text(encoding="utf-8")
    _require("Environment check passed." in check_env, "check_env did not pass")
    _require("gpu:         NVIDIA B200 sm_100" in check_env, "wrong check_env GPU")
    manifest_check = (run_dir / "source_manifest_check.txt").read_text(
        encoding="utf-8"
    )
    manifest_lines = [line for line in manifest_check.splitlines() if line.strip()]
    _require(len(manifest_lines) >= 20, "source manifest covers too few files")
    _require(
        all(line.endswith(": OK") for line in manifest_lines),
        "source manifest validation did not pass",
    )
    for executed_path_marker in (
        "/jit_kernel/utils.py: OK",
        "/jit_kernel/dsv4/elementwise.py: OK",
        "/jit_kernel/csrc/deepseek_v4/main_norm_rope.cuh: OK",
        "/jit_kernel/dsv32/elementwise.py: OK",
        "/jit_kernel/csrc/deepseek_v32/indexer_k.cuh: OK",
        "/jit_kernel/include/sgl_kernel/deepseek_v4/fp8_utils.cuh: OK",
    ):
        _require(
            any(executed_path_marker in line for line in manifest_lines),
            f"source manifest missing executed file {executed_path_marker}",
        )
    jit_check = (run_dir / "jit_artifact_manifest_check.txt").read_text(
        encoding="utf-8"
    )
    jit_lines = [line for line in jit_check.splitlines() if line.strip()]
    _require(jit_lines, "empty JIT artifact manifest check")
    _require(
        all(line.endswith(": OK") for line in jit_lines),
        "JIT artifact manifest validation did not pass",
    )
    for module_marker in (
        "dpsk_v4_main_q_indexer_rope_first_quant",
        "dpsk_v32_k_indexer_norm_rope_store_p64",
    ):
        _require(
            any(module_marker in line for line in jit_lines),
            f"JIT artifact manifest missing {module_marker}",
        )

    candidates = {
        "identity": (source_dir / "reference.py", None),
        "torch_mm": (source_dir / "indexer_wk_torch_mm.py", "torch_mm_direct"),
        "tgv": (source_dir / "indexer_wk_cutedsl_tgv.py", "sglang_cutedsl_tgv_direct"),
        "single_stream": (
            source_dir / "indexer_single_stream.py",
            "stock_bf16_single_stream_schedule",
        ),
    }
    for candidate, _ in candidates.values():
        _require(candidate.is_file(), f"missing candidate snapshot: {candidate}")

    baselines: dict[str, list[float]] = {"isolated": [], "region": []}
    for scope, workload in (("isolated", ISOLATED), ("region", REGION)):
        for run in range(1, 4):
            result = _validate_result(
                results_dir / f"{scope}_stock_{run:02d}.json",
                repo=repo,
                workload=workload,
                repeat=30,
                candidate=None,
                backend=None,
            )
            baselines[scope].append(
                _number(
                    result["reference"].get("median_ms"),
                    label=f"{scope} stock {run:02d} median",
                    positive=True,
                )
            )

    series_specs = [
        ("isolated_identity", ISOLATED, "identity", 1),
        ("region_identity", REGION, "identity", 1),
        ("isolated_torch_mm", ISOLATED, "torch_mm", 3),
        ("region_torch_mm", REGION, "torch_mm", 3),
        ("isolated_tgv", ISOLATED, "tgv", 3),
        ("region_tgv", REGION, "tgv", 3),
        ("region_single_stream", REGION, "single_stream", 3),
    ]
    series: dict[str, dict[str, Any]] = {}
    for series_name, workload, candidate_name, runs in series_specs:
        candidate, backend = candidates[candidate_name]
        speedups: list[float] = []
        reference_medians: list[float] = []
        candidate_medians: list[float] = []
        for run in range(1, runs + 1):
            suffix = "" if runs == 1 else f"_{run:02d}"
            result = _validate_result(
                results_dir / f"{series_name}{suffix}.json",
                repo=repo,
                workload=workload,
                repeat=60,
                candidate=candidate,
                backend=backend,
            )
            speedups.append(
                _number(
                    result["candidate"].get("speedup"),
                    label=f"{series_name}{suffix} speedup",
                    positive=True,
                )
            )
            reference_medians.append(
                _number(
                    result["reference"].get("median_ms"),
                    label=f"{series_name}{suffix} reference median",
                    positive=True,
                )
            )
            candidate_medians.append(
                _number(
                    result["candidate"].get("median_ms"),
                    label=f"{series_name}{suffix} candidate median",
                    positive=True,
                )
            )
        series[series_name] = {
            "candidate": candidate_name,
            "candidate_medians_ms": candidate_medians,
            "paired_median_speedups": speedups,
            "reference_medians_ms": reference_medians,
            "repeat_stable_3pct_win": runs >= 3 and all(value >= 1.03 for value in speedups),
        }

    identity_controls_pass = all(
        series[name]["paired_median_speedups"][0] < 1.03
        for name in ("isolated_identity", "region_identity")
    )
    _require(identity_controls_pass, "identity control produced a false >=1.03x win")

    profiles = {
        "stock": _validate_profile(run_dir, tag="stock", repo=repo, candidate=None),
        "torch_mm": _validate_profile(
            run_dir,
            tag="torch-mm",
            repo=repo,
            candidate=candidates["torch_mm"][0],
        ),
        "single_stream": _validate_profile(
            run_dir,
            tag="single-stream",
            repo=repo,
            candidate=candidates["single_stream"][0],
        ),
    }
    stable_candidates = [
        name
        for name in ("region_torch_mm", "region_tgv", "region_single_stream")
        if series[name]["repeat_stable_3pct_win"]
    ]
    output = {
        "schema_version": 1,
        "status": "PASS",
        "run_dir": str(run_dir),
        "correctness": "all candidate runs passed pre-timing and post-timing checks",
        "source_provenance": "all result and profile hashes recomputed",
        "baselines_ms": baselines,
        "series": series,
        "profiles": profiles,
        "identity_controls_pass": identity_controls_pass,
        "stable_region_candidates_at_1_03x": stable_candidates,
        "no_replacement_inner_gate": not stable_candidates,
    }
    args.output_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Hardened same-GPU validation summary",
        "",
        "Every candidate row passed the runner's pre-timing comparison and "
        "post-timing replay; fused prepare/store rows use a fresh deterministic seed.",
        "",
        "| Series | Reference medians (ms) | Candidate medians (ms) | Paired median speedups | Stable >=1.03x |",
        "|---|---|---|---|---|",
    ]
    for name, item in series.items():
        fmt = lambda values: ", ".join(f"{value:.6f}" for value in values)
        lines.append(
            f"| {name} | {fmt(item['reference_medians_ms'])} | "
            f"{fmt(item['candidate_medians_ms'])} | "
            f"{fmt(item['paired_median_speedups'])} | "
            f"{item['repeat_stable_3pct_win']} |"
        )
    lines.extend(
        [
            "",
            f"Stable region candidates: {stable_candidates or 'none'}.",
            "Nsys stock, torch-mm, and single-stream reports plus ABI traces were "
            "validated as non-empty, source-hashed artifacts.",
            "",
        ]
    )
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
