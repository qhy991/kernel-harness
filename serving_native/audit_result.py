#!/usr/bin/env python3
"""Fail-closed auditor for serving-native schema-v2 result artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from serving_native.contract_v2 import (
    MIN_REQUIRED_SERIES,
    PERFORMANCE_THRESHOLD,
    SCHEMA_VERSION,
    canonical_sha256,
    graph_forbidden_nodes,
    graph_kernel_identities,
    graph_node_type_counts,
    latency_summary,
    sha256_file,
)
from serving_native.workloads import W2_EDGE_MASK_CASES, WORKLOADS, as_dict

CALLABLE_CANDIDATE_API = "callable_v1"
TRUSTED_CONFIG_CANDIDATE_API = "reference_with_config_v1"
W2_BM16_BASE_COMMIT = "edcf77b276965de8f03cdc47c23f01b08bf7c7ab"
W2_BM16_CUTLASS_COMMIT = "f3fde58372d33e9a5650ba7b80fc48b3b49d40c8"
W2_BM16_FMT_COMMIT = "553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28"
W2_BM16_STOCK_EXTENSION_SHA256 = (
    "f3b377630a5016fbd1ea0771cab2a450ff859b815d436b9b50624b5cb438141d"
)
W2_BM16_CANDIDATE_EXTENSION_SHA256 = (
    "e4ba14ea9c97674eacd55d1a59ace8b98e1fa005f69ec0d7b1d675b24f2cffc3"
)
W2_BM16_SOURCE_PATCH_SHA256 = (
    "26bb69cd59f2df5a8a8a12292447d9badf3786e8631b028314f1a811c87d8401"
)
W2_BM16_CACHE_DIR = (
    "/home/qinhaiyan/glm52-v2-goal-runs/cache/"
    "26-moe_w2_decode_scoped_bm16/deepgemm"
)
W2_BM16_IMPL_NAME = "sm100_fp8_fp4_gemm_1d1d_impl"


def _w2_template_mapping(kernels: Any, block_m: int) -> bool:
    if not _list(kernels):
        return False
    mangled_segment = (
        f"ELj0ELj6144ELj2048ELj{block_m}ELj128ELj128"
    )
    demangled_segment = f"0,6144,2048,{block_m},128,128"
    for kernel in kernels:
        if not isinstance(kernel, str) or W2_BM16_IMPL_NAME not in kernel:
            continue
        if mangled_segment in kernel:
            return True
        normalized = re.sub(r"\s+", "", kernel)
        normalized = re.sub(r"(?<=\d)[uUlL]+(?=[,>])", "", normalized)
        if demangled_segment in normalized:
            return True
    return False


def _is_w2_leaf_workload(workload: dict[str, Any]) -> bool:
    params = workload.get("params")
    return (
        workload.get("family") == "moe_grouped_masked"
        and _mapping(params)
        and "candidate_jit_identity" in params
    )


class Findings:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def require(self, condition: Any, message: str) -> bool:
        if not condition:
            self.errors.append(message)
            return False
        return True


def _mapping(value: Any) -> bool:
    return isinstance(value, dict)


def _list(value: Any) -> bool:
    return isinstance(value, list)


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _strict_nonnegative_int(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
    )


def _strict_node_type_counts(value: Any) -> bool:
    return _mapping(value) and all(
        isinstance(node_type, str)
        and bool(node_type)
        and _strict_nonnegative_int(count)
        for node_type, count in value.items()
    )


def _close(left: Any, right: Any, *, tolerance: float = 1e-9) -> bool:
    return _number(left) and _number(right) and math.isclose(
        float(left), float(right), rel_tol=tolerance, abs_tol=tolerance
    )


def _graph_signature(capture: dict[str, Any]) -> list[tuple[Any, ...]]:
    signature: list[tuple[Any, ...]] = []
    for node in capture.get("nodes", []):
        signature.append(
            (
                node.get("type"),
                node.get("kernel"),
                tuple(node.get("grid", [])),
                tuple(node.get("block", [])),
                node.get("shared_memory_bytes"),
            )
        )
    return signature


def _audit_latency_summary(
    findings: Findings,
    recorded: Any,
    values: list[float],
    *,
    prefix: str,
    allow_extra_fields: bool = False,
) -> None:
    if not findings.require(_mapping(recorded), f"{prefix} latency summary missing"):
        return
    expected = latency_summary(values)
    findings.require(
        (
            set(expected).issubset(recorded)
            if allow_extra_fields
            else set(recorded) == set(expected)
        ),
        f"{prefix} latency-summary fields do not close",
    )
    for field, expected_value in expected.items():
        findings.require(
            _close(recorded.get(field), expected_value),
            f"{prefix}.{field} does not match raw samples",
        )


def _expected_phase_counts(
    *,
    run_id: str,
    mode: str,
    requested_series: int,
    warmup: int,
    repeat: int,
    workload: dict[str, Any],
) -> dict[str, dict[str, int]]:
    phases: dict[str, dict[str, int]] = {}

    def add(phase: str, reference: int, candidate: int) -> None:
        item = phases.setdefault(
            phase,
            {"reference_calls": 0, "candidate_hits": 0},
        )
        item["reference_calls"] += reference
        item["candidate_hits"] += candidate

    add("pre_timing_correctness", 1, 1)
    add("jit_warmup", max(3, warmup) + 1, max(3, warmup) + 1)
    for series_index in range(requested_series):
        series_id = f"{run_id}:series-{series_index + 1:02d}"
        if mode == "cuda_graph":
            for implementation, suffix in (
                ("reference", "R-first"),
                ("candidate", "C-after-R"),
                ("candidate", "C-first"),
                ("reference", "R-after-C"),
            ):
                is_reference = implementation == "reference"
                add(
                    f"{series_id}:{suffix}:warmup",
                    3 if is_reference else 0,
                    0 if is_reference else 3,
                )
                add(
                    f"{series_id}:{suffix}:capture",
                    1 if is_reference else 0,
                    0 if is_reference else 1,
                )
            # Per series: two eager reference checks plus three replays of
            # each of two reference captures, and three replays of each of two
            # candidate captures.
            add("graph_validation", 8, 6)
        add(f"{series_id}:warmup", warmup, warmup)
        add(f"{series_id}:timing", repeat, repeat)
    add("post_timing_correctness", 1, 1)
    add("fresh_inputs_correctness", 1, 1)
    if mode == "eager":
        add("profiler_reference", 1, 0)
        add("profiler_candidate", 0, 1)
    if _is_w2_leaf_workload(workload):
        for case_index, (name, _counts) in enumerate(W2_EDGE_MASK_CASES):
            add(f"edge:{name}:eager", 1, 1)
            if mode == "cuda_graph":
                for implementation in ("reference", "candidate"):
                    is_reference = implementation == "reference"
                    capture_id = (
                        f"edge:{case_index:02d}:{name}:{implementation}"
                    )
                    add(
                        f"{capture_id}:warmup",
                        3 if is_reference else 0,
                        0 if is_reference else 3,
                    )
                    add(
                        f"{capture_id}:capture",
                        1 if is_reference else 0,
                        0 if is_reference else 1,
                    )
                add(
                    f"edge:{case_index:02d}:{name}:graph_validation",
                    2,
                    2,
                )
    return phases


def _audit_accounting(
    findings: Findings,
    result: dict[str, Any],
    *,
    mode: str,
    workload: dict[str, Any],
    run: dict[str, Any],
) -> tuple[bool, int, int, str | None]:
    implementations = result.get("implementations")
    if not findings.require(_mapping(implementations), "missing implementation accounting"):
        return False, 0, 0, None
    reference = implementations.get("reference")
    candidate = implementations.get("candidate")
    if not findings.require(_mapping(reference), "missing reference implementation accounting"):
        reference = {}
    if not findings.require(_mapping(candidate), "missing candidate implementation accounting"):
        return False, 0, 0, None

    findings.require(
        isinstance(reference.get("identity"), str) and bool(reference["identity"]),
        "reference implementation identity missing",
    )
    findings.require(
        isinstance(candidate.get("identity"), str) and bool(candidate["identity"]),
        "candidate implementation identity missing",
    )
    findings.require(
        isinstance(candidate.get("identity_control"), bool),
        "candidate identity-control flag missing",
    )
    findings.require(
        isinstance(candidate.get("declared_fallback"), bool),
        "candidate declared-fallback flag missing",
    )
    findings.require(
        "REFERENCE_DELEGATION_IS_CANDIDATE" not in candidate,
        "candidate-controlled reference-delegation exemption is forbidden",
    )
    identity_control = candidate.get("identity_control") is True
    candidate_api = candidate.get("api")
    findings.require(
        candidate_api in (CALLABLE_CANDIDATE_API, TRUSTED_CONFIG_CANDIDATE_API),
        "candidate API is missing or untrusted",
    )
    if candidate_api == TRUSTED_CONFIG_CANDIDATE_API:
        findings.require(
            workload.get("family")
            in {"deepep_normal_dispatch", "deepep_normal_combine"},
            "trusted candidate-with-config API used outside DeepEP normal mode",
        )
    if identity_control:
        findings.require(
            candidate_api == CALLABLE_CANDIDATE_API,
            "identity control must use the callable candidate API",
        )

    run_id = run.get("run_id")
    requested_series = run.get("requested_series")
    warmup = run.get("warmup")
    repeat = run.get("repeat")
    if not (
        isinstance(run_id, str)
        and isinstance(requested_series, int)
        and isinstance(warmup, int)
        and isinstance(repeat, int)
    ):
        findings.errors.append("cannot close implementation counts from invalid run metadata")
        return identity_control, 0, 0, candidate_api
    expected = _expected_phase_counts(
        run_id=run_id,
        mode=mode,
        requested_series=requested_series,
        warmup=warmup,
        repeat=repeat,
        workload=workload,
    )
    by_phase = candidate.get("by_phase")
    if not findings.require(_mapping(by_phase), "candidate by_phase accounting missing"):
        return identity_control, 0, 0, candidate_api
    findings.require(
        set(by_phase) == set(expected),
        "candidate by_phase phase set does not close exactly",
    )

    counter_fields = (
        "reference_calls",
        "candidate_hits",
        "candidate_fallbacks",
        "candidate_reference_delegations",
        "candidate_trusted_config_calls",
    )
    totals = {field: 0 for field in counter_fields}
    for phase in sorted(set(expected) | set(by_phase)):
        item = by_phase.get(phase)
        prefix = f"implementations.candidate.by_phase[{phase!r}]"
        if not findings.require(_mapping(item), f"{prefix} missing"):
            continue
        findings.require(
            set(item) == set(counter_fields),
            f"{prefix} counter fields do not close exactly",
        )
        for field in counter_fields:
            value = item.get(field)
            if findings.require(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                f"{prefix}.{field} invalid",
            ):
                totals[field] += value
        expected_item = expected.get(phase)
        if expected_item is None:
            continue
        findings.require(
            item.get("reference_calls") == expected_item["reference_calls"],
            f"{prefix}.reference_calls does not match runner path count",
        )
        findings.require(
            item.get("candidate_hits") == expected_item["candidate_hits"],
            f"{prefix}.candidate_hits does not match runner path count",
        )
        hits = item.get("candidate_hits")
        fallbacks = item.get("candidate_fallbacks")
        delegations = item.get("candidate_reference_delegations")
        trusted = item.get("candidate_trusted_config_calls")
        if all(isinstance(value, int) for value in (hits, fallbacks, delegations, trusted)):
            findings.require(
                fallbacks <= hits and delegations <= hits and trusted <= hits,
                f"{prefix} candidate sub-count exceeds hit count",
            )
            if identity_control:
                findings.require(
                    delegations == hits and fallbacks == 0 and trusted == 0,
                    f"{prefix} identity delegation counts do not close",
                )
            elif candidate_api == TRUSTED_CONFIG_CANDIDATE_API:
                findings.require(
                    delegations == hits and trusted == hits and fallbacks == 0,
                    f"{prefix} trusted config counts do not close",
                )
            else:
                findings.require(
                    trusted == 0 and fallbacks == delegations,
                    f"{prefix} untrusted delegation/fallback counts do not close",
                )

    findings.require(
        reference.get("call_count") == totals["reference_calls"],
        "reference call count does not close against by_phase",
    )
    findings.require(
        candidate.get("hit_count") == totals["candidate_hits"],
        "candidate hit count does not close against by_phase",
    )
    findings.require(
        candidate.get("fallback_count") == totals["candidate_fallbacks"],
        "candidate fallback count does not close against by_phase",
    )
    findings.require(
        candidate.get("reference_delegations")
        == totals["candidate_reference_delegations"],
        "candidate reference-delegation count does not close against by_phase",
    )
    findings.require(
        candidate.get("trusted_config_call_count")
        == totals["candidate_trusted_config_calls"],
        "candidate trusted-config count does not close against by_phase",
    )
    hit_count = totals["candidate_hits"]
    fallback_count = totals["candidate_fallbacks"]
    reference_delegations = totals["candidate_reference_delegations"]
    findings.require(hit_count > 0, "candidate hit count is zero or missing")
    declared_fallback = candidate.get("declared_fallback") is True
    if fallback_count > 0:
        findings.require(declared_fallback, "silent candidate fallback detected")
    if (
        not identity_control
        and candidate_api != TRUSTED_CONFIG_CANDIDATE_API
        and reference_delegations > 0
    ):
        findings.require(
            fallback_count == reference_delegations,
            "non-identity reference delegation bypassed fallback accounting",
        )
    return identity_control, fallback_count, reference_delegations, candidate_api


def _audit_artifacts(
    findings: Findings,
    provenance: dict[str, Any],
    *,
    verify_files: bool,
) -> dict[str, dict[str, Any]]:
    artifacts = provenance.get("artifacts")
    if not findings.require(_list(artifacts) and bool(artifacts), "missing provenance.artifacts"):
        return {}
    by_role: dict[str, dict[str, Any]] = {}
    for index, artifact in enumerate(artifacts):
        prefix = f"provenance.artifacts[{index}]"
        if not findings.require(_mapping(artifact), f"{prefix} must be an object"):
            continue
        role = artifact.get("role")
        path_raw = artifact.get("path")
        digest = artifact.get("sha256")
        findings.require(isinstance(role, str) and bool(role), f"{prefix}.role missing")
        findings.require(
            isinstance(path_raw, str) and bool(path_raw),
            f"{prefix}.path missing",
        )
        findings.require(
            isinstance(digest, str) and len(digest) == 64,
            f"{prefix}.sha256 invalid",
        )
        if isinstance(role, str):
            if role in by_role:
                findings.errors.append(f"duplicate artifact role: {role}")
            else:
                by_role[role] = artifact
        if not verify_files or not isinstance(path_raw, str):
            continue
        path = Path(path_raw)
        if not findings.require(path.is_absolute(), f"{prefix}.path must be absolute"):
            continue
        if not findings.require(path.is_file(), f"{prefix}.path not found: {path}"):
            continue
        actual = sha256_file(path)
        findings.require(
            actual == digest,
            f"{prefix} hash mismatch: recorded={digest} actual={actual}",
        )
        findings.require(
            artifact.get("size_bytes") == path.stat().st_size,
            f"{prefix} size mismatch",
        )
    for role in ("runner", "workloads", "candidate"):
        findings.require(role in by_role, f"missing required artifact role: {role}")
    for role, expected_path in (
        ("runner", HERE / "runner.py"),
        ("workloads", HERE / "workloads.py"),
    ):
        artifact = by_role.get(role)
        if artifact is not None and isinstance(artifact.get("path"), str):
            findings.require(
                Path(artifact["path"]).resolve() == expected_path.resolve(),
                f"{role} artifact is not the canonical serving-native source",
            )
    return by_role


def _audit_imports(
    findings: Findings,
    provenance: dict[str, Any],
    by_role: dict[str, dict[str, Any]],
    *,
    verify_files: bool,
) -> None:
    imports = provenance.get("imports")
    if not findings.require(_mapping(imports), "missing provenance.imports"):
        return
    executable = imports.get("python_executable")
    findings.require(
        isinstance(executable, str) and Path(executable).is_absolute(),
        "provenance.imports.python_executable must be absolute",
    )
    if verify_files and isinstance(executable, str):
        findings.require(
            Path(executable).exists(),
            f"python executable does not exist: {executable}",
        )
    modules = imports.get("modules")
    if not findings.require(_list(modules) and bool(modules), "missing actual Python import paths"):
        return
    module_names: set[str] = set()
    candidate_paths: set[str] = set()
    shared_module_paths: set[str] = set()
    for index, item in enumerate(modules):
        prefix = f"provenance.imports.modules[{index}]"
        if not findings.require(_mapping(item), f"{prefix} must be an object"):
            continue
        name = item.get("module")
        path_raw = item.get("path")
        findings.require(isinstance(name, str) and bool(name), f"{prefix}.module missing")
        findings.require(
            isinstance(path_raw, str) and Path(path_raw).is_absolute(),
            f"{prefix}.path must be absolute",
        )
        if isinstance(name, str):
            module_names.add(name)
        if name == "serving_native_candidate" and isinstance(path_raw, str):
            candidate_paths.add(str(Path(path_raw).resolve()))
        if item.get("kind") == "shared_object" and isinstance(path_raw, str):
            shared_module_paths.add(str(Path(path_raw).resolve()))
        if verify_files and isinstance(path_raw, str):
            findings.require(Path(path_raw).is_file(), f"{prefix}.path not found: {path_raw}")
    for root in ("torch", "sglang", "deep_gemm", "serving_native_candidate"):
        findings.require(
            root in module_names or any(name.startswith(f"{root}.") for name in module_names),
            f"actual import provenance omits {root}",
        )
    candidate_artifact = by_role.get("candidate")
    if candidate_artifact is not None:
        expected = str(Path(candidate_artifact["path"]).resolve())
        findings.require(
            candidate_paths == {expected},
            "candidate import path does not match the hashed candidate artifact",
        )
    shared_objects = imports.get("shared_objects")
    findings.require(
        _list(shared_objects) and bool(shared_objects),
        "missing actual shared-object import paths",
    )
    if _list(shared_objects):
        for index, path_raw in enumerate(shared_objects):
            findings.require(
                isinstance(path_raw, str) and Path(path_raw).is_absolute(),
                f"provenance.imports.shared_objects[{index}] must be absolute",
            )
            if verify_files and isinstance(path_raw, str):
                findings.require(
                    Path(path_raw).is_file(),
                    f"shared object path not found: {path_raw}",
                )
        findings.require(
            not shared_module_paths
            or shared_module_paths.issubset(
                {str(Path(path).resolve()) for path in shared_objects}
            ),
            "module/shared-object provenance is inconsistent",
        )


def _audit_repositories(findings: Findings, provenance: dict[str, Any]) -> None:
    repositories = provenance.get("repositories")
    if not findings.require(_mapping(repositories), "missing provenance.repositories"):
        return
    for name in ("kernel_harness", "sglang"):
        item = repositories.get(name)
        if not findings.require(_mapping(item), f"missing repository provenance: {name}"):
            continue
        findings.require(
            isinstance(item.get("path"), str) and Path(item["path"]).is_absolute(),
            f"repository {name} path must be absolute",
        )
        head = item.get("head")
        findings.require(
            isinstance(head, str) and len(head) == 40,
            f"repository {name} head is invalid",
        )
        findings.require(
            isinstance(item.get("dirty"), bool),
            f"repository {name} dirty flag missing",
        )
        findings.require(
            _list(item.get("status")),
            f"repository {name} status missing",
        )


def _audit_hardware(findings: Findings, provenance: dict[str, Any]) -> None:
    hardware = provenance.get("hardware")
    if not findings.require(_mapping(hardware), "missing provenance.hardware"):
        return
    findings.require(
        isinstance(hardware.get("uuid"), str)
        and hardware["uuid"].startswith("GPU-"),
        "GPU UUID missing or invalid",
    )
    findings.require(
        isinstance(hardware.get("driver_version"), str)
        and bool(hardware["driver_version"]),
        "CUDA driver version missing",
    )
    findings.require(
        isinstance(hardware.get("cuda_runtime_version"), str)
        and bool(hardware["cuda_runtime_version"]),
        "CUDA runtime version missing",
    )
    clocks = hardware.get("clock_samples")
    if not findings.require(_list(clocks) and len(clocks) >= 2, "GPU clock samples incomplete"):
        return
    for index, sample in enumerate(clocks):
        prefix = f"provenance.hardware.clock_samples[{index}]"
        findings.require(_mapping(sample), f"{prefix} must be an object")
        if not _mapping(sample):
            continue
        findings.require(
            sample.get("uuid") == hardware.get("uuid"),
            f"{prefix} GPU UUID mismatch",
        )
        findings.require(
            isinstance(sample.get("sm_clock_mhz"), int)
            and sample["sm_clock_mhz"] > 0,
            f"{prefix} SM clock invalid",
        )
        findings.require(
            isinstance(sample.get("memory_clock_mhz"), int)
            and sample["memory_clock_mhz"] > 0,
            f"{prefix} memory clock invalid",
        )


def _audit_jit(
    findings: Findings,
    provenance: dict[str, Any],
    *,
    expected_phases: list[str],
) -> None:
    jit = provenance.get("jit")
    if not findings.require(_mapping(jit), "missing provenance.jit"):
        return
    findings.require(jit.get("warmup_completed") is True, "JIT/warmup did not complete")
    warmup_activity = jit.get("warmup_activity")
    findings.require(
        _mapping(warmup_activity)
        and warmup_activity.get("phase") == "jit_warmup",
        "JIT warmup activity record missing",
    )
    observations = jit.get("observations")
    if not findings.require(
        _list(observations) and bool(observations),
        "missing JIT phase observations",
    ):
        return
    findings.require(
        [item.get("phase") if _mapping(item) else None for item in observations]
        == expected_phases,
        "JIT observation phases do not close against capture/timing phases",
    )
    detected = False
    for index, observation in enumerate(observations):
        prefix = f"provenance.jit.observations[{index}]"
        if not findings.require(_mapping(observation), f"{prefix} must be an object"):
            detected = True
            continue
        findings.require(observation.get("clean") is True, f"{prefix} is not clean")
        if observation.get("clean") is not True:
            detected = True
        for field in (
            "new_imports",
            "new_shared_objects",
            "cache_changes",
            "candidate_artifact_changes",
        ):
            findings.require(
                observation.get(field) == [],
                f"{prefix}.{field} must be empty",
            )
            if observation.get(field) != []:
                detected = True
    findings.require(
        jit.get("capture_or_timing_detected") is detected,
        "JIT capture/timing summary does not match observations",
    )
    findings.require(
        not detected,
        "JIT or import/artifact activity detected during capture or timing",
    )


def _audit_correctness(
    findings: Findings,
    result: dict[str, Any],
    mode: str | None,
) -> None:
    correctness = result.get("correctness")
    if not findings.require(_mapping(correctness), "missing correctness record"):
        return
    findings.require(correctness.get("status") == "pass", "correctness status is not pass")
    for field in (
        "pre_timing_reference",
        "pre_timing_candidate",
        "post_timing_reference",
        "post_timing_candidate",
        "fresh_inputs_post_timing",
    ):
        findings.require(correctness.get(field) is True, f"correctness.{field} did not pass")
    tolerance = correctness.get("tolerance")
    findings.require(
        _mapping(tolerance)
        and _number(tolerance.get("rtol"))
        and _number(tolerance.get("atol")),
        "approved numeric tolerance is missing",
    )
    if mode == "cuda_graph":
        findings.require(
            correctness.get("graph_validation") is True,
            "graph correctness validation did not pass",
        )


def _expected_order(start_order: str, pair_index: int) -> str:
    if pair_index % 2 == 0:
        return start_order
    return "BA" if start_order == "AB" else "AB"


def _audit_one_series(
    findings: Findings,
    series: dict[str, Any],
    *,
    index: int,
    mode: str,
    run_id: str,
    expected_repeat: int,
    expected_warmup: int,
    capture_pools: dict[str, list[str]] | None,
) -> tuple[float | None, bool | None, list[float], list[float]]:
    prefix = f"series[{index}]"
    findings.require(series.get("series_index") == index, f"{prefix}.series_index mismatch")
    expected_series_id = f"{run_id}:series-{index + 1:02d}"
    findings.require(
        series.get("series_id") == expected_series_id,
        f"{prefix}.series_id is not bound to run identity",
    )
    findings.require(series.get("independent") is True, f"{prefix} is not independent")
    findings.require(series.get("execution_mode") == mode, f"{prefix} execution-mode mismatch")
    expected_start = "AB" if index % 2 == 0 else "BA"
    start_order = series.get("start_order")
    findings.require(start_order == expected_start, f"{prefix} start order mismatch")
    repeat = series.get("repeat")
    if not findings.require(
        repeat == expected_repeat and isinstance(repeat, int) and repeat >= 2,
        f"{prefix}.repeat does not close against run.repeat",
    ):
        return None, None, [], []
    findings.require(
        series.get("warmup_pairs") == expected_warmup,
        f"{prefix}.warmup_pairs does not close against run.warmup",
    )
    samples = series.get("raw_ordered_samples")
    if not findings.require(
        _list(samples) and len(samples) == repeat * 2,
        f"{prefix} raw ordered samples incomplete",
    ):
        return None, None, [], []
    by_pair: dict[int, dict[str, float]] = {}
    capture_ordinals = {"reference": 0, "candidate": 0}
    for sequence, sample in enumerate(samples):
        sample_prefix = f"{prefix}.raw_ordered_samples[{sequence}]"
        if not findings.require(_mapping(sample), f"{sample_prefix} must be an object"):
            continue
        pair_index = sample.get("pair_index")
        position = sample.get("position")
        findings.require(sample.get("sequence") == sequence, f"{sample_prefix}.sequence mismatch")
        findings.require(
            isinstance(pair_index, int) and 0 <= pair_index < repeat,
            f"{sample_prefix}.pair_index invalid",
        )
        findings.require(position in (0, 1), f"{sample_prefix}.position invalid")
        if not isinstance(pair_index, int) or position not in (0, 1):
            continue
        order = _expected_order(start_order, pair_index)
        expected_impl = (
            ("reference", "candidate")
            if order == "AB"
            else ("candidate", "reference")
        )[position]
        implementation = sample.get("implementation")
        findings.require(sample.get("order") == order, f"{sample_prefix}.order mismatch")
        findings.require(
            implementation == expected_impl,
            f"{sample_prefix}.implementation breaks AB/BA ordering",
        )
        findings.require(
            sample.get("label") == ("A" if implementation == "reference" else "B"),
            f"{sample_prefix}.label mismatch",
        )
        if mode == "cuda_graph":
            pools = capture_pools or {}
            pool = pools.get(str(implementation), [])
            capture_id = sample.get("graph_capture_id")
            if findings.require(
                bool(pool),
                f"{sample_prefix} has no independently captured graph pool",
            ) and implementation in capture_ordinals:
                ordinal = capture_ordinals[implementation]
                expected_capture_id = pool[ordinal % len(pool)]
                findings.require(
                    capture_id == expected_capture_id,
                    f"{sample_prefix}.graph_capture_id is not bound to "
                    "the round-robin independent capture",
                )
                capture_ordinals[implementation] += 1
        else:
            findings.require(
                "graph_capture_id" not in sample,
                f"{sample_prefix} eager sample carries a graph_capture_id",
            )
        latency = sample.get("latency_ms")
        findings.require(
            _number(latency) and float(latency) > 0.0,
            f"{sample_prefix}.latency_ms invalid",
        )
        if _number(latency) and implementation in ("reference", "candidate"):
            pair = by_pair.setdefault(pair_index, {})
            findings.require(
                implementation not in pair,
                f"{sample_prefix} duplicates {implementation} in pair {pair_index}",
            )
            pair[implementation] = float(latency)
    findings.require(
        len(by_pair) == repeat
        and all(set(pair) == {"reference", "candidate"} for pair in by_pair.values()),
        f"{prefix} does not contain complete A/B pairs",
    )
    if len(by_pair) != repeat or not all(
        set(pair) == {"reference", "candidate"} for pair in by_pair.values()
    ):
        return None, None, [], []
    ratios = [
        by_pair[pair]["reference"] / by_pair[pair]["candidate"]
        for pair in range(repeat)
    ]
    reference_values = [by_pair[pair]["reference"] for pair in range(repeat)]
    candidate_values = [by_pair[pair]["candidate"] for pair in range(repeat)]
    _audit_latency_summary(
        findings,
        series.get("reference"),
        reference_values,
        prefix=f"{prefix}.reference",
    )
    _audit_latency_summary(
        findings,
        series.get("candidate"),
        candidate_values,
        prefix=f"{prefix}.candidate",
    )
    recorded_ratios = series.get("paired_speedups")
    findings.require(
        _list(recorded_ratios)
        and len(recorded_ratios) == len(ratios)
        and all(_close(left, right) for left, right in zip(recorded_ratios, ratios)),
        f"{prefix}.paired_speedups do not match raw samples",
    )
    median_speedup = statistics.median(ratios)
    findings.require(
        _close(series.get("median_speedup"), median_speedup),
        f"{prefix}.median_speedup mismatch",
    )
    passed = median_speedup >= PERFORMANCE_THRESHOLD
    findings.require(
        series.get("passes_3pct_gate") is passed,
        f"{prefix}.passes_3pct_gate mismatch",
    )
    return median_speedup, passed, reference_values, candidate_values


def _audit_graph_series(
    findings: Findings,
    series: dict[str, Any],
    *,
    index: int,
    identity_control: bool,
    candidate_api: str | None,
) -> dict[str, list[str]] | None:
    prefix = f"series[{index}].graph"
    graph = series.get("graph")
    if not findings.require(_mapping(graph), f"{prefix} missing"):
        return None
    findings.require(
        graph.get("capture_policy") == "bidirectional_R-C_then_C-R_round_robin",
        f"{prefix}.capture_policy mismatch",
    )
    findings.require(
        graph.get("reference_candidate_captured_independently") is True,
        f"{prefix} did not independently capture reference and candidate",
    )
    captures = graph.get("captures")
    if not findings.require(_list(captures) and len(captures) == 4, f"{prefix} capture set incomplete"):
        return None
    expected_implementations = ("reference", "candidate", "candidate", "reference")
    expected_suffixes = ("R-first", "C-after-R", "C-first", "R-after-C")
    series_id = series.get("series_id")
    capture_ids: set[str] = set()
    raw_graph_handles: set[int] = set()
    stream_ids: set[int] = set()
    reference_signatures: list[list[tuple[Any, ...]]] = []
    candidate_signatures: list[list[tuple[Any, ...]]] = []
    pools: dict[str, list[str]] = {"reference": [], "candidate": []}
    for capture_index, capture in enumerate(captures):
        capture_prefix = f"{prefix}.captures[{capture_index}]"
        if not findings.require(_mapping(capture), f"{capture_prefix} must be an object"):
            continue
        capture_id = capture.get("capture_id")
        expected_capture_id = f"{series_id}:{expected_suffixes[capture_index]}"
        findings.require(
            capture_id == expected_capture_id,
            f"{capture_prefix}.capture_id is not bound to its series/capture order",
        )
        if isinstance(capture_id, str):
            findings.require(capture_id not in capture_ids, f"duplicate graph capture id: {capture_id}")
            capture_ids.add(capture_id)
        implementation = capture.get("implementation")
        findings.require(
            implementation == expected_implementations[capture_index],
            f"{capture_prefix}.implementation/capture order mismatch",
        )
        if (
            isinstance(capture_id, str)
            and implementation in ("reference", "candidate")
        ):
            pools[implementation].append(capture_id)

        raw_graph_handle = capture.get("raw_graph_handle")
        if findings.require(
            isinstance(raw_graph_handle, int)
            and not isinstance(raw_graph_handle, bool)
            and raw_graph_handle > 0,
            f"{capture_prefix}.raw_graph_handle invalid",
        ):
            findings.require(
                raw_graph_handle not in raw_graph_handles,
                f"{capture_prefix} reuses a CUDA graph handle",
            )
            raw_graph_handles.add(raw_graph_handle)
        stream_id = capture.get("stream_id")
        default_stream_id = capture.get("default_stream_id")
        valid_stream_ids = (
            isinstance(stream_id, int)
            and not isinstance(stream_id, bool)
            and stream_id >= 0
            and isinstance(default_stream_id, int)
            and not isinstance(default_stream_id, bool)
            and default_stream_id >= 0
        )
        findings.require(valid_stream_ids, f"{capture_prefix} stream IDs invalid")
        computed_non_default = valid_stream_ids and stream_id != default_stream_id
        findings.require(
            capture.get("non_default_stream") is computed_non_default,
            f"{capture_prefix}.non_default_stream does not match stream IDs",
        )
        findings.require(computed_non_default, f"{capture_prefix} did not use a non-default stream")
        if isinstance(stream_id, int):
            findings.require(
                stream_id not in stream_ids,
                f"{capture_prefix} reuses a capture stream",
            )
            stream_ids.add(stream_id)
        for field in (
            "stable_input_pointers",
            "stable_output_pointers",
            "input_mutation_replayed",
            "output_poison_replayed",
            "deterministic_replay",
            "approved_tolerance_passed",
        ):
            findings.require(
                capture.get(field) is True,
                f"{capture_prefix}.{field} did not pass",
            )
        nodes = capture.get("nodes")
        findings.require(_list(nodes) and bool(nodes), f"{capture_prefix}.nodes missing")
        if _list(nodes):
            for node_index, node in enumerate(nodes):
                node_prefix = f"{capture_prefix}.nodes[{node_index}]"
                if not findings.require(_mapping(node), f"{node_prefix} must be an object"):
                    continue
                recorded_index = node.get("index")
                findings.require(
                    _strict_nonnegative_int(recorded_index)
                    and recorded_index == node_index,
                    f"{node_prefix}.index is not a strict integer or mismatches",
                )
                node_type = node.get("type")
                findings.require(
                    isinstance(node_type, str) and bool(node_type),
                    f"{node_prefix}.type missing",
                )
                if isinstance(node_type, str) and "KERNEL" in node_type:
                    findings.require(
                        isinstance(node.get("kernel"), str)
                        and bool(node["kernel"]),
                        f"{node_prefix}.kernel identity missing",
                    )
                    for dimension in ("grid", "block"):
                        value = node.get(dimension)
                        findings.require(
                            _list(value)
                            and len(value) == 3
                            and all(
                                isinstance(item, int)
                                and not isinstance(item, bool)
                                and item > 0
                                for item in value
                            ),
                            f"{node_prefix}.{dimension} invalid",
                        )
                    findings.require(
                        isinstance(node.get("shared_memory_bytes"), int)
                        and not isinstance(node["shared_memory_bytes"], bool)
                        and node["shared_memory_bytes"] >= 0,
                        f"{node_prefix}.shared_memory_bytes invalid",
                    )
            node_count = len(nodes)
            valid_nodes = [node for node in nodes if _mapping(node)]
            node_type_counts = graph_node_type_counts(valid_nodes)
            kernels = graph_kernel_identities(valid_nodes)
            forbidden = graph_forbidden_nodes(valid_nodes)
            recorded_node_count = capture.get("node_count")
            findings.require(
                _strict_nonnegative_int(recorded_node_count),
                f"{capture_prefix}.node_count is not a strict integer",
            )
            findings.require(
                recorded_node_count == node_count,
                f"{capture_prefix}.node_count does not match nodes",
            )
            recorded_type_counts = capture.get("node_type_counts")
            findings.require(
                _strict_node_type_counts(recorded_type_counts),
                f"{capture_prefix}.node_type_counts values are not strict integers",
            )
            findings.require(
                recorded_type_counts == node_type_counts,
                f"{capture_prefix}.node_type_counts do not match nodes",
            )
            findings.require(
                capture.get("kernel_identities") == kernels,
                f"{capture_prefix}.kernel_identities do not match nodes",
            )
            findings.require(
                capture.get("forbidden_nodes") == forbidden,
                f"{capture_prefix}.forbidden_nodes do not match nodes",
            )
            findings.require(not forbidden, f"{capture_prefix} has forbidden graph nodes")
            findings.require(bool(kernels), f"{capture_prefix} has no kernel identity")
        fallback = capture.get("fallback")
        reference_delegated = capture.get("reference_delegated")
        trusted_config = capture.get("trusted_config")
        for field, value in (
            ("fallback", fallback),
            ("reference_delegated", reference_delegated),
            ("trusted_config", trusted_config),
        ):
            findings.require(
                isinstance(value, bool),
                f"{capture_prefix}.{field} missing",
            )
        if implementation == "reference":
            findings.require(
                fallback is False
                and reference_delegated is False
                and trusted_config is False,
                f"{capture_prefix} reference capture has candidate delegation state",
            )
        elif identity_control:
            findings.require(
                fallback is False
                and reference_delegated is True
                and trusted_config is False,
                f"{capture_prefix} identity capture delegation state is invalid",
            )
        elif candidate_api == TRUSTED_CONFIG_CANDIDATE_API:
            findings.require(
                fallback is False
                and reference_delegated is True
                and trusted_config is True,
                f"{capture_prefix} trusted config capture state is invalid",
            )
        elif implementation == "candidate":
            findings.require(
                trusted_config is False and fallback is reference_delegated,
                f"{capture_prefix} untrusted delegation bypassed fallback state",
            )
        if implementation == "reference":
            reference_signatures.append(_graph_signature(capture))
        elif implementation == "candidate":
            candidate_signatures.append(_graph_signature(capture))
    independent = (
        len(capture_ids) == 4
        and len(raw_graph_handles) == 4
        and len(stream_ids) == 4
    )
    findings.require(
        graph.get("reference_candidate_captured_independently") is independent,
        f"{prefix} independent-capture flag does not match capture IDs",
    )
    findings.require(independent, f"{prefix} captures are not independent")
    if identity_control and reference_signatures and candidate_signatures:
        canonical = reference_signatures[0]
        findings.require(
            all(signature == canonical for signature in reference_signatures + candidate_signatures),
            f"{prefix} identity A/B graph signatures differ",
        )
    return {
        "reference": [pools["reference"][0], pools["reference"][1]]
        if len(pools["reference"]) == 2
        else pools["reference"],
        "candidate": [pools["candidate"][0], pools["candidate"][1]]
        if len(pools["candidate"]) == 2
        else pools["candidate"],
    }


def _audit_kernel_profiles(
    findings: Findings,
    execution: dict[str, Any],
    *,
    identity_control: bool,
) -> None:
    profiles = execution.get("kernel_profiles")
    if not findings.require(_mapping(profiles), "eager execution lacks kernel profiler capture"):
        return
    identities: dict[str, list[str]] = {}
    for implementation in ("reference", "candidate"):
        profile = profiles.get(implementation)
        prefix = f"execution.kernel_profiles.{implementation}"
        if not findings.require(_mapping(profile), f"{prefix} missing"):
            continue
        findings.require(profile.get("captured") is True, f"{prefix} was not captured")
        kernels = profile.get("kernel_identities")
        findings.require(_list(kernels) and bool(kernels), f"{prefix} kernel identities missing")
        events = profile.get("events")
        findings.require(_list(events) and bool(events), f"{prefix} events missing")
        recomputed: list[str] = []
        if _list(events):
            names: set[str] = set()
            for event_index, event in enumerate(events):
                event_prefix = f"{prefix}.events[{event_index}]"
                if not findings.require(_mapping(event), f"{event_prefix} must be an object"):
                    continue
                name = event.get("name")
                findings.require(
                    isinstance(name, str) and bool(name),
                    f"{event_prefix}.name missing",
                )
                findings.require(
                    isinstance(event.get("device_type"), str)
                    and "cuda" in event["device_type"].lower(),
                    f"{event_prefix}.device_type is not CUDA",
                )
                findings.require(
                    _number(event.get("duration_us"))
                    and float(event["duration_us"]) >= 0.0,
                    f"{event_prefix}.duration_us invalid",
                )
                if isinstance(name, str) and name:
                    names.add(name)
            recomputed = sorted(names)
        findings.require(
            kernels == recomputed,
            f"{prefix}.kernel_identities do not match profiler events",
        )
        if recomputed:
            identities[implementation] = recomputed
    if identity_control and set(identities) == {"reference", "candidate"}:
        findings.require(
            identities["reference"] == identities["candidate"],
            "identity A/B eager kernel identities differ",
        )


def _audit_exact_single_kernel_graph(
    findings: Findings,
    capture: dict[str, Any],
    *,
    prefix: str,
) -> list[str]:
    """Recompute one-kernel graph identity from raw nodes, never summaries."""
    nodes = capture.get("nodes")
    if not findings.require(
        _list(nodes) and bool(nodes),
        f"{prefix}.nodes missing",
    ):
        return []
    for node_index, node in enumerate(nodes):
        node_prefix = f"{prefix}.nodes[{node_index}]"
        if not findings.require(
            _mapping(node),
            f"{node_prefix} must be an object",
        ):
            continue
        recorded_index = node.get("index")
        findings.require(
            _strict_nonnegative_int(recorded_index)
            and recorded_index == node_index,
            f"{node_prefix}.index is not a strict integer or mismatches",
        )
        node_type = node.get("type")
        findings.require(
            isinstance(node_type, str) and bool(node_type),
            f"{node_prefix}.type missing",
        )
        if isinstance(node_type, str) and "KERNEL" in node_type:
            findings.require(
                isinstance(node.get("kernel"), str)
                and bool(node["kernel"]),
                f"{node_prefix}.kernel identity missing",
            )
            for dimension in ("grid", "block"):
                value = node.get(dimension)
                findings.require(
                    _list(value)
                    and len(value) == 3
                    and all(
                        isinstance(item, int)
                        and not isinstance(item, bool)
                        and item > 0
                        for item in value
                    ),
                    f"{node_prefix}.{dimension} invalid",
                )
            findings.require(
                isinstance(node.get("shared_memory_bytes"), int)
                and not isinstance(node["shared_memory_bytes"], bool)
                and node["shared_memory_bytes"] >= 0,
                f"{node_prefix}.shared_memory_bytes invalid",
            )

    valid_nodes = [node for node in nodes if _mapping(node)]
    node_count = len(nodes)
    type_counts = graph_node_type_counts(valid_nodes)
    kernels = graph_kernel_identities(valid_nodes)
    forbidden = graph_forbidden_nodes(valid_nodes)
    recorded_node_count = capture.get("node_count")
    findings.require(
        _strict_nonnegative_int(recorded_node_count),
        f"{prefix}.node_count is not a strict integer",
    )
    findings.require(
        recorded_node_count == node_count,
        f"{prefix}.node_count does not match raw nodes",
    )
    recorded_type_counts = capture.get("node_type_counts")
    findings.require(
        _strict_node_type_counts(recorded_type_counts),
        f"{prefix}.node_type_counts values are not strict integers",
    )
    findings.require(
        recorded_type_counts == type_counts,
        f"{prefix}.node_type_counts do not match raw nodes",
    )
    findings.require(
        capture.get("kernel_identities") == kernels,
        f"{prefix}.kernel_identities do not match raw nodes",
    )
    findings.require(
        capture.get("forbidden_nodes") == forbidden,
        f"{prefix}.forbidden_nodes do not match raw nodes",
    )
    findings.require(
        not forbidden,
        f"{prefix} has forbidden graph nodes",
    )
    findings.require(
        node_count == 1
        and len(valid_nodes) == 1
        and type_counts == {"CU_GRAPH_NODE_TYPE_KERNEL": 1}
        and len(kernels) == 1,
        f"{prefix} must contain exactly one CUDA KERNEL node",
    )
    return kernels


def _profile_event_names(profile: Any) -> list[str]:
    if not _mapping(profile) or not _list(profile.get("events")):
        return []
    return [
        event["name"]
        for event in profile["events"]
        if _mapping(event)
        and isinstance(event.get("name"), str)
        and bool(event["name"])
    ]


def _audit_w2_edge_masks(
    findings: Findings,
    result: dict[str, Any],
    workload: dict[str, Any],
    *,
    mode: str,
) -> None:
    correctness = result.get("correctness")
    edge = (
        correctness.get("edge_masks")
        if _mapping(correctness)
        else None
    )
    if workload.get("family") == "moe_compute_region":
        findings.require(
            edge is None,
            "W2 containing-region result must not relabel leaf edge evidence",
        )
        return
    if not findings.require(
        _mapping(edge),
        "W2/BM16 explicit edge-mask correctness evidence is missing",
    ):
        return
    findings.require(
        edge.get("status") == "pass"
        and edge.get("scope") == "single_B200_leaf_correctness_only"
        and edge.get("execution_mode") == mode,
        "W2/BM16 edge-mask scope or status mismatch",
    )
    cases = edge.get("cases")
    if not findings.require(
        _list(cases) and len(cases) == len(W2_EDGE_MASK_CASES),
        "W2/BM16 edge-mask case count mismatch",
    ):
        return
    expected = dict(W2_EDGE_MASK_CASES)
    findings.require(
        [case.get("name") if _mapping(case) else None for case in cases]
        == [name for name, _counts in W2_EDGE_MASK_CASES],
        "W2/BM16 edge-mask case order or identity mismatch",
    )
    for index, case in enumerate(cases):
        prefix = f"correctness.edge_masks.cases[{index}]"
        if not findings.require(_mapping(case), f"{prefix} must be an object"):
            continue
        name = case.get("name")
        counts = expected.get(name)
        findings.require(
            counts is not None and case.get("masked_m") == list(counts),
            f"{prefix}.masked_m is not the canonical explicit edge mask",
        )
        if counts is None:
            continue
        findings.require(
            case.get("active_rows") == sum(counts)
            and case.get("max_count") == max(counts)
            and case.get("empty_experts")
            == [
                expert
                for expert, count in enumerate(counts)
                if count == 0
            ],
            f"{prefix} mask summary does not close",
        )
        eager = case.get("eager")
        if not findings.require(
            _mapping(eager), f"{prefix}.eager evidence is missing"
        ):
            continue
        findings.require(
            eager.get("stock_candidate_match") is True,
            f"{prefix}.eager stock/candidate correctness failed",
        )
        findings.require(
            eager.get("output_poisoned_before_launch")
            == {"reference": True, "candidate": True},
            f"{prefix}.eager output poison proof failed",
        )
        zero_mask = sum(counts) == 0
        expected_sentinel_elements = (
            32 * 1024 * 6144
            if zero_mask
            else sum(counts) * 6144
        )
        findings.require(
            eager.get("sentinel_scope")
            == (
                "entire_output_buffer"
                if zero_mask
                else "active_rows"
            )
            and eager.get("sentinel_elements_checked")
            == expected_sentinel_elements
            and eager.get("non_vacuous_sentinel_coverage") is True,
            f"{prefix}.eager sentinel coverage is vacuous or incomplete",
        )
        findings.require(
            eager.get("zero_full_output_poison_preserved")
            == (
                {"reference": True, "candidate": True}
                if zero_mask
                else None
            ),
            f"{prefix}.zero-mask full output did not remain poisoned",
        )
        findings.require(
            eager.get("masked_m_unmodified")
            == {"reference": True, "candidate": True},
            f"{prefix}.masked_m was modified",
        )
        findings.require(
            eager.get("return_contract")
            == {
                "reference": "None",
                "candidate": "None",
                "enforced_before_TaskResult": True,
            },
            f"{prefix}.return contract mismatch",
        )
        stream = eager.get("stream")
        stream_fields = (
            "default_stream_id",
            "before",
            "after_reference",
            "after_candidate",
        )
        valid_stream_fields = _mapping(stream) and all(
            _strict_nonnegative_int(stream.get(field))
            for field in stream_fields
        )
        findings.require(
            valid_stream_fields
            and stream.get("unchanged_default_stream") is True
            and stream.get("before")
            == stream.get("after_reference")
            == stream.get("after_candidate")
            == stream.get("default_stream_id"),
            f"{prefix}.eager stream contract mismatch",
        )

        graph = case.get("graph")
        if mode == "eager":
            findings.require(
                graph is None,
                f"{prefix} eager result carries graph edge evidence",
            )
            continue
        if not findings.require(
            _mapping(graph),
            f"{prefix}.graph edge evidence is missing",
        ):
            continue
        reference_capture = graph.get("reference")
        candidate_capture = graph.get("candidate")
        findings.require(
            graph.get("stock_candidate_match") is True
            and graph.get("reference_candidate_captured_independently")
            is True,
            f"{prefix}.graph stock/candidate correctness failed",
        )
        capture_ids: set[str] = set()
        raw_graph_handles: set[int] = set()
        stream_ids: set[int] = set()
        for implementation, capture, block_m in (
            ("reference", reference_capture, 128),
            ("candidate", candidate_capture, 16),
        ):
            capture_prefix = f"{prefix}.graph.{implementation}"
            if not findings.require(
                _mapping(capture),
                f"{capture_prefix} capture is missing",
            ):
                continue
            expected_capture_id = (
                f"edge:{index:02d}:{name}:{implementation}"
            )
            capture_id = capture.get("capture_id")
            findings.require(
                capture_id == expected_capture_id,
                f"{capture_prefix}.capture_id mismatch",
            )
            if isinstance(capture_id, str):
                findings.require(
                    capture_id not in capture_ids,
                    f"{capture_prefix}.capture_id is reused",
                )
                capture_ids.add(capture_id)

            raw_graph_handle = capture.get("raw_graph_handle")
            if findings.require(
                isinstance(raw_graph_handle, int)
                and not isinstance(raw_graph_handle, bool)
                and raw_graph_handle > 0,
                f"{capture_prefix}.raw_graph_handle invalid",
            ):
                findings.require(
                    raw_graph_handle not in raw_graph_handles,
                    f"{capture_prefix}.raw_graph_handle is reused",
                )
                raw_graph_handles.add(raw_graph_handle)
            stream_id = capture.get("stream_id")
            default_stream_id = capture.get("default_stream_id")
            valid_streams = (
                isinstance(stream_id, int)
                and not isinstance(stream_id, bool)
                and stream_id > 0
                and isinstance(default_stream_id, int)
                and not isinstance(default_stream_id, bool)
                and default_stream_id >= 0
            )
            findings.require(
                valid_streams,
                f"{capture_prefix} stream IDs invalid",
            )
            computed_non_default = (
                valid_streams and stream_id != default_stream_id
            )
            findings.require(
                capture.get("non_default_stream") is computed_non_default,
                f"{capture_prefix}.non_default_stream does not match IDs",
            )
            findings.require(
                computed_non_default,
                f"{capture_prefix} did not use its capture stream",
            )
            if isinstance(stream_id, int):
                findings.require(
                    stream_id not in stream_ids,
                    f"{capture_prefix}.stream_id is reused",
                )
                stream_ids.add(stream_id)

            kernels = _audit_exact_single_kernel_graph(
                findings,
                capture,
                prefix=capture_prefix,
            )
            mapping_ok = _w2_template_mapping(kernels, block_m)
            if implementation == "candidate":
                mapping_ok = mapping_ok and not _w2_template_mapping(
                    kernels, 128
                )
            else:
                mapping_ok = mapping_ok and not _w2_template_mapping(
                    kernels, 16
                )
            findings.require(
                capture.get("implementation") == implementation
                and mapping_ok,
                f"{capture_prefix} W2 template mapping mismatch",
            )
            for field in (
                "stable_input_pointers",
                "stable_output_pointers",
                "fixed_edge_mask_replayed",
                "output_poison_replayed",
                "deterministic_replay",
                "approved_tolerance_passed",
            ):
                findings.require(
                    capture.get(field) is True,
                    f"{capture_prefix}.{field} did not pass",
                )
            findings.require(
                capture.get("input_mutation_replayed") is False,
                f"{capture_prefix} falsely claims post-capture input "
                "mutation coverage",
            )
            findings.require(
                capture.get("sentinel_elements_checked")
                == expected_sentinel_elements
                and capture.get("non_vacuous_sentinel_coverage") is True
                and capture.get("zero_full_output_poison_preserved")
                == (True if zero_mask else None),
                f"{capture_prefix} zero/active sentinel coverage failed",
            )
            findings.require(
                capture.get("fallback") is False
                and capture.get("reference_delegated") is False
                and capture.get("trusted_config") is False,
                f"{capture_prefix} used a fallback or reference delegation",
            )
        independently_bound = (
            len(capture_ids) == 2
            and len(raw_graph_handles) == 2
            and len(stream_ids) == 2
        )
        findings.require(
            graph.get("reference_candidate_captured_independently")
            is independently_bound,
            f"{prefix} independent-capture flag does not match raw IDs",
        )
        findings.require(
            independently_bound,
            f"{prefix} edge graphs or streams are not independent",
        )
        observations = graph.get("capture_observations")
        for implementation in ("reference", "candidate"):
            observation = (
                observations.get(implementation)
                if _mapping(observations)
                else None
            )
            findings.require(
                _mapping(observation)
                and observation.get("clean") is True
                and observation.get("new_imports") == []
                and observation.get("new_shared_objects") == []
                and observation.get("cache_changes") == []
                and observation.get("candidate_artifact_changes") == [],
                f"{prefix}.graph.{implementation} capture had runtime/JIT "
                "activity",
            )


def _audit_w2_bm16_candidate(
    findings: Findings,
    result: dict[str, Any],
    workload: dict[str, Any],
    *,
    mode: str,
    identity_control: bool,
    verify_files: bool,
) -> None:
    params = workload.get("params")
    if (
        identity_control
        or not _mapping(params)
        or "candidate_jit_identity" not in params
    ):
        return

    expected_m = params.get("expected_m")
    decode_m = params.get("decode_m")
    expected_token = params.get("candidate_jit_identity")
    findings.require(
        (decode_m, expected_m) in {(16, 4), (16, 5), (32, 8), (32, 9)},
        "W2/BM16 workload identity is outside the four accepted buckets",
    )
    findings.require(
        workload.get("family")
        in {"moe_grouped_masked", "moe_compute_region"},
        "W2/BM16 candidate is attached to an unsupported workload family",
    )
    findings.require(
        isinstance(expected_token, str)
        and expected_token.endswith(f"glm52_w2_bm16_v2_em{expected_m}"),
        "W2/BM16 canonical JIT identity is invalid",
    )
    _audit_w2_edge_masks(
        findings,
        result,
        workload,
        mode=mode,
    )

    implementations = result.get("implementations")
    candidate = (
        implementations.get("candidate")
        if _mapping(implementations)
        else None
    )
    runtime = (
        candidate.get("runtime_contract") if _mapping(candidate) else None
    )
    if not findings.require(
        _mapping(runtime),
        "W2/BM16 candidate runtime contract evidence is missing",
    ):
        return
    expected_build_id = (
        "glm52-w2-bm16-v2:sgl-deep-gemm-0.1.4.post1@"
        f"{W2_BM16_BASE_COMMIT}:sm100:e32:m1024:k2048:n6144:"
        "bm16:pdl1:sms148:no-recipe:no-overlap"
    )
    exact_fields = {
        "base_commit": W2_BM16_BASE_COMMIT,
        "base_version": "0.1.4.post1",
        "cutlass_commit": W2_BM16_CUTLASS_COMMIT,
        "fmt_commit": W2_BM16_FMT_COMMIT,
        "build_id": expected_build_id,
        "physical_num_sms": 148,
        "stock_num_sms": 148,
        "candidate_num_sms": 148,
        "stock_pdl": True,
        "candidate_pdl": True,
        "runtime_modules_distinct": True,
        "runtime_extension_modules_distinct": True,
        "stock_extension_sha256": W2_BM16_STOCK_EXTENSION_SHA256,
        "candidate_extension_sha256": W2_BM16_CANDIDATE_EXTENSION_SHA256,
        "dg_jit_cache_dir": W2_BM16_CACHE_DIR,
        "sglang_dg_cache_dir": W2_BM16_CACHE_DIR,
        "decode_m": decode_m,
        "expected_m": expected_m,
        "candidate_jit_identity": expected_token,
        "forward_mode": "DECODE",
        "op_tag": "moe_down_proj",
        "reference_opt_level_after_prepare": "0",
        "workload": workload.get("name"),
        "setup_environment_restored": {
            "SGLANG_GLM52_OPT": True,
            "SGLANG_GLM52_OPT_PROFILE": True,
            "SGLANG_GLM52_OPT_OPS": True,
            "SGLANG_DEEPGEMM_PDL": True,
        },
    }
    for field, expected in exact_fields.items():
        findings.require(
            runtime.get(field) == expected,
            f"W2/BM16 runtime contract {field} mismatch",
        )
    findings.require(
        runtime.get("stock_tc_util") == runtime.get("candidate_tc_util")
        and isinstance(runtime.get("stock_tc_util"), int),
        "W2/BM16 stock/candidate tc_util state differs",
    )
    findings.require(
        runtime.get("compute_capability") in ((10, 0), [10, 0]),
        "W2/BM16 runtime is not SM100",
    )
    findings.require(
        runtime.get("independence_probe_num_sms") == 146
        and isinstance(runtime.get("independence_probe_tc_util"), int),
        "W2/BM16 independent DeviceRuntime probe is missing",
    )

    artifacts = result.get("provenance", {}).get("artifacts", [])
    artifact_hashes = {
        item.get("sha256") for item in artifacts if _mapping(item)
    }
    for digest, label in (
        (W2_BM16_SOURCE_PATCH_SHA256, "source patch"),
        (W2_BM16_STOCK_EXTENSION_SHA256, "stock _C"),
        (W2_BM16_CANDIDATE_EXTENSION_SHA256, "candidate _C"),
        (runtime.get("manifest_sha256"), "overlay manifest"),
    ):
        findings.require(
            isinstance(digest, str) and digest in artifact_hashes,
            f"W2/BM16 {label} is not bound to a hashed artifact",
        )

    kernel_root = (Path(W2_BM16_CACHE_DIR) / "cache").resolve()
    kernel_dir_raw = runtime.get("jit_cache_kernel_dir")
    source_path_raw = runtime.get("jit_cache_source_path")
    cubin_path_raw = runtime.get("jit_cache_cubin_path")
    cache_key = runtime.get("jit_cache_key")
    kernel_dir = (
        Path(kernel_dir_raw).resolve()
        if isinstance(kernel_dir_raw, str)
        else None
    )
    expected_dir_name = (
        f"kernel.{expected_token}.{cache_key}"
        if isinstance(cache_key, str)
        else None
    )
    findings.require(
        runtime.get("jit_cache_kernel_name") == expected_token,
        "W2/BM16 exact emX JIT cache kernel name mismatch",
    )
    findings.require(
        isinstance(cache_key, str)
        and re.fullmatch(r"[0-9a-f]{32}", cache_key) is not None,
        "W2/BM16 JIT cache key is not a 32-character lowercase hex digest",
    )
    findings.require(
        kernel_dir is not None
        and kernel_dir.parent == kernel_root
        and kernel_dir.name == expected_dir_name,
        "W2/BM16 exact emX JIT cache directory is not task-local or canonical",
    )
    expected_source = kernel_dir / "kernel.cu" if kernel_dir else None
    expected_cubin = kernel_dir / "kernel.cubin" if kernel_dir else None
    findings.require(
        isinstance(source_path_raw, str)
        and expected_source is not None
        and Path(source_path_raw).resolve() == expected_source,
        "W2/BM16 generated kernel source path is not bound to its emX cache",
    )
    findings.require(
        isinstance(cubin_path_raw, str)
        and expected_cubin is not None
        and Path(cubin_path_raw).resolve() == expected_cubin,
        "W2/BM16 cubin path is not bound to its emX cache",
    )
    findings.require(
        runtime.get("jit_cache_generated_impl") == W2_BM16_IMPL_NAME
        and runtime.get("jit_cache_generated_block_m") == 16
        and runtime.get("jit_cache_template_segment")
        == [0, 6144, 2048, 16, 128, 128],
        "W2/BM16 generated source does not prove the BM16 template mapping",
    )

    artifacts_by_path = {
        str(Path(item["path"]).resolve()): item
        for item in artifacts
        if _mapping(item) and isinstance(item.get("path"), str)
    }
    for path_raw, digest_field, label in (
        (
            source_path_raw,
            "jit_cache_source_sha256",
            "generated kernel source",
        ),
        (cubin_path_raw, "jit_cache_cubin_sha256", "emX cubin"),
    ):
        digest = runtime.get(digest_field)
        artifact = (
            artifacts_by_path.get(str(Path(path_raw).resolve()))
            if isinstance(path_raw, str)
            else None
        )
        findings.require(
            isinstance(digest, str)
            and len(digest) == 64
            and _mapping(artifact)
            and artifact.get("sha256") == digest,
            f"W2/BM16 {label} is not bound to its exact hashed artifact",
        )
    if (
        verify_files
        and expected_source is not None
        and expected_source.is_file()
    ):
        generated_source = expected_source.read_text(errors="replace")
        normalized_source = re.sub(r"\s+", "", generated_source)
        findings.require(
            W2_BM16_IMPL_NAME in generated_source
            and "0,6144,2048,16,128,128" in normalized_source,
            "W2/BM16 hashed generated source lacks the positional BM16 "
            "template segment",
        )

    if mode == "eager":
        profiles = result.get("execution", {}).get("kernel_profiles", {})
        reference_profile = (
            profiles.get("reference") if _mapping(profiles) else None
        )
        candidate_profile = (
            profiles.get("candidate") if _mapping(profiles) else None
        )
        reference_events = _profile_event_names(reference_profile)
        candidate_events = _profile_event_names(candidate_profile)
        reference_w2 = [
            index
            for index, kernel in enumerate(reference_events)
            if _w2_template_mapping([kernel], 128)
        ]
        candidate_w2 = [
            index
            for index, kernel in enumerate(candidate_events)
            if _w2_template_mapping([kernel], 16)
        ]
        if _is_w2_leaf_workload(workload):
            findings.require(
                len(reference_events) == 1
                and reference_w2 == [0]
                and not _w2_template_mapping(reference_events, 16),
                "W2/BM16 leaf stock eager profile must contain exactly one "
                "CUDA event: the W2 BM128 kernel",
            )
            findings.require(
                len(candidate_events) == 1
                and candidate_w2 == [0]
                and not _w2_template_mapping(candidate_events, 128),
                "W2/BM16 leaf candidate eager profile must contain exactly "
                "one CUDA event: W2 BM16 with no BM128 or adapter kernel",
            )
        else:
            findings.require(
                len(reference_events) == len(candidate_events)
                and len(reference_events) > 1,
                "W2/BM16 containing-region eager profiles must have equal "
                "non-leaf CUDA event counts",
            )
            findings.require(
                len(reference_w2) == 1
                and not _w2_template_mapping(reference_events, 16),
                "W2/BM16 containing-region stock profile must contain "
                "exactly one W2 BM128 event and no BM16 event",
            )
            findings.require(
                len(candidate_w2) == 1
                and not _w2_template_mapping(candidate_events, 128),
                "W2/BM16 containing-region candidate profile must contain "
                "exactly one W2 BM16 event and no BM128 event",
            )
            reference_non_w2 = [
                kernel
                for index, kernel in enumerate(reference_events)
                if index not in reference_w2
            ]
            candidate_non_w2 = [
                kernel
                for index, kernel in enumerate(candidate_events)
                if index not in candidate_w2
            ]
            findings.require(
                bool(reference_non_w2)
                and reference_non_w2 == candidate_non_w2,
                "W2/BM16 containing-region non-W2 CUDA event order differs",
            )
            findings.require(
                Counter(reference_non_w2) == Counter(candidate_non_w2),
                "W2/BM16 containing-region non-W2 CUDA event multiset differs",
            )
    elif mode == "cuda_graph":
        for series_index, series in enumerate(result.get("series", [])):
            captures = (
                series.get("graph", {}).get("captures", [])
                if _mapping(series)
                else []
            )
            for capture_index, capture in enumerate(captures):
                if not _mapping(capture):
                    continue
                capture_prefix = (
                    "W2/BM16 leaf graph "
                    f"series[{series_index}].captures[{capture_index}]"
                )
                kernels = _audit_exact_single_kernel_graph(
                    findings,
                    capture,
                    prefix=capture_prefix,
                )
                if capture.get("implementation") == "reference":
                    findings.require(
                        _w2_template_mapping(
                            kernels, 128
                        )
                        and not _w2_template_mapping(
                            kernels, 16
                        ),
                        "W2/BM16 stock graph lacks its BM128 template mapping "
                        f"at series {series_index} capture {capture_index}",
                    )
                elif capture.get("implementation") == "candidate":
                    findings.require(
                        _w2_template_mapping(
                            kernels, 16
                        )
                        and not _w2_template_mapping(
                            kernels, 128
                        ),
                        "W2/BM16 candidate graph must contain BM16 and no W2 "
                        "BM128 template mapping at series "
                        f"{series_index} capture {capture_index}",
                    )


def audit_document(
    result: Any,
    *,
    verify_files: bool = True,
) -> dict[str, Any]:
    findings = Findings()
    if not findings.require(_mapping(result), "result root must be an object"):
        return {"valid": False, "errors": findings.errors, "warnings": findings.warnings}
    findings.require(result.get("schema_version") == SCHEMA_VERSION, "schema_version must be 2")
    findings.require(
        result.get("result_kind") == "serving_native_v2",
        "result_kind must be serving_native_v2",
    )
    run_value = result.get("run")
    run: dict[str, Any] = run_value if _mapping(run_value) else {}
    if findings.require(bool(run), "missing run metadata"):
        for field in ("run_id", "started_utc", "finished_utc"):
            findings.require(
                isinstance(run.get(field), str) and bool(run[field]),
                f"run.{field} missing",
            )
        findings.require(_list(run.get("command")) and bool(run["command"]), "run.command missing")
        findings.require(
            isinstance(run.get("requested_series"), int)
            and not isinstance(run["requested_series"], bool)
            and run["requested_series"] >= MIN_REQUIRED_SERIES,
            "run.requested_series is incomplete",
        )
        findings.require(
            isinstance(run.get("warmup"), int)
            and not isinstance(run["warmup"], bool)
            and run["warmup"] >= 1,
            "run.warmup invalid",
        )
        findings.require(
            isinstance(run.get("repeat"), int)
            and not isinstance(run["repeat"], bool)
            and run["repeat"] >= 2,
            "run.repeat invalid",
        )

    workload_value = result.get("workload")
    workload: dict[str, Any] = workload_value if _mapping(workload_value) else {}
    canonical_workload: dict[str, Any] | None = None
    workload_hash: str | None = None
    if findings.require(bool(workload), "missing workload"):
        for field in ("name", "family", "phase", "source_symbol"):
            findings.require(
                isinstance(workload.get(field), str) and bool(workload[field]),
                f"workload.{field} missing",
            )
        findings.require(_mapping(workload.get("params")), "workload.params missing")
        findings.require(
            _list(workload.get("execution_modes")) and bool(workload["execution_modes"]),
            "workload.execution_modes missing",
        )
        workload_name = workload.get("name")
        registered = (
            WORKLOADS.get(workload_name)
            if isinstance(workload_name, str)
            else None
        )
        if findings.require(
            registered is not None,
            "reported workload is absent from the canonical WORKLOADS registry",
        ):
            canonical_workload = as_dict(registered)
            findings.require(
                workload == canonical_workload,
                "reported workload does not match the canonical WORKLOADS registry",
            )
            workload_hash = canonical_sha256(canonical_workload)

    execution_value = result.get("execution")
    execution: dict[str, Any] = execution_value if _mapping(execution_value) else {}
    mode: str | None = None
    if findings.require(bool(execution), "missing execution contract"):
        mode = execution.get("mode")
        findings.require(mode in ("eager", "cuda_graph"), "execution.mode invalid")
        if workload:
            findings.require(
                mode in workload.get("execution_modes", []),
                "execution-mode mismatch with workload contract",
            )
        findings.require(
            execution.get("reference_candidate_captured_separately")
            is (mode == "cuda_graph"),
            "execution graph-capture contract mismatch",
        )
        findings.require(
            isinstance(execution.get("timer"), str)
            and bool(execution["timer"]),
            "execution timer identity missing",
        )
        if mode == "cuda_graph":
            findings.require(
                execution.get("capture_stream")
                == "independent non-default streams",
                "execution capture-stream contract mismatch",
            )
            findings.require(
                execution.get("graph_capture_policy")
                == "bidirectional_R-C_then_C-R_round_robin",
                "execution graph capture policy mismatch",
            )
            findings.require(
                execution.get("kernel_profiles") is None,
                "CUDA Graph execution must use graph nodes, not eager profiles",
            )
        elif mode == "eager":
            findings.require(
                execution.get("capture_stream") is None
                and execution.get("graph_capture_policy") is None,
                "eager execution carries a CUDA Graph contract",
            )

    requested_series = run.get("requested_series")
    expected_jit_phases: list[str] = []
    if (
        isinstance(requested_series, int)
        and not isinstance(requested_series, bool)
        and isinstance(run.get("run_id"), str)
        and mode in ("eager", "cuda_graph")
    ):
        for series_index in range(requested_series):
            series_id = f"{run['run_id']}:series-{series_index + 1:02d}"
            if mode == "cuda_graph":
                expected_jit_phases.extend(
                    f"{series_id}:{suffix}:capture"
                    for suffix in (
                        "R-first",
                        "C-after-R",
                        "C-first",
                        "R-after-C",
                    )
                )
            expected_jit_phases.append(f"{series_id}:timing")

    provenance_value = result.get("provenance")
    provenance: dict[str, Any] = (
        provenance_value if _mapping(provenance_value) else {}
    )
    by_role: dict[str, dict[str, Any]] = {}
    if findings.require(bool(provenance), "missing provenance"):
        findings.require(
            workload_hash is not None
            and provenance.get("workload_sha256") == workload_hash,
            "workload hash mismatch",
        )
        by_role = _audit_artifacts(findings, provenance, verify_files=verify_files)
        _audit_imports(findings, provenance, by_role, verify_files=verify_files)
        _audit_repositories(findings, provenance)
        _audit_hardware(findings, provenance)
        _audit_jit(
            findings,
            provenance,
            expected_phases=expected_jit_phases,
        )

    identity_control = False
    fallback_count = 0
    reference_delegations = 0
    candidate_api: str | None = None
    if (
        mode in ("eager", "cuda_graph")
        and canonical_workload is not None
        and run
    ):
        (
            identity_control,
            fallback_count,
            reference_delegations,
            candidate_api,
        ) = _audit_accounting(
            findings,
            result,
            mode=mode,
            workload=canonical_workload,
            run=run,
        )
    else:
        findings.require(
            _mapping(result.get("implementations")),
            "missing implementation accounting",
        )

    candidate_value = result.get("candidate")
    candidate_record: dict[str, Any] = (
        candidate_value if _mapping(candidate_value) else {}
    )
    if findings.require(bool(candidate_record), "missing candidate result record"):
        findings.require(
            candidate_record.get("identity_control") is identity_control,
            "candidate identity-control fields disagree",
        )
        findings.require(
            candidate_record.get("api") == candidate_api,
            "candidate API fields disagree",
        )
        candidate_artifact = by_role.get("candidate")
        if candidate_artifact is not None:
            findings.require(
                isinstance(candidate_record.get("path"), str)
                and Path(candidate_record["path"]).resolve()
                == Path(candidate_artifact["path"]).resolve(),
                "candidate result path does not match hashed artifact",
            )

    _audit_correctness(findings, result, mode)

    series_value = result.get("series")
    series: list[Any] = series_value if _list(series_value) else []
    series_passes: list[bool] = []
    series_medians: list[float] = []
    all_reference: list[float] = []
    all_candidate: list[float] = []
    series_ids: set[str] = set()
    exact_series_count = (
        isinstance(requested_series, int)
        and not isinstance(requested_series, bool)
        and len(series) == requested_series
        and len(series) >= MIN_REQUIRED_SERIES
    )
    if findings.require(exact_series_count, "requested/raw series counts do not close exactly") and isinstance(mode, str):
        for index, item in enumerate(series):
            if not findings.require(_mapping(item), f"series[{index}] must be an object"):
                continue
            series_id = item.get("series_id")
            if isinstance(series_id, str):
                findings.require(series_id not in series_ids, f"duplicate series identity: {series_id}")
                series_ids.add(series_id)
            capture_pools = None
            if mode == "cuda_graph":
                capture_pools = _audit_graph_series(
                    findings,
                    item,
                    index=index,
                    identity_control=identity_control,
                    candidate_api=candidate_api,
                )
            else:
                findings.require(
                    "graph" not in item,
                    f"series[{index}] eager series carries graph captures",
                )
            median, passed, reference_values, candidate_values = _audit_one_series(
                findings,
                item,
                index=index,
                mode=mode,
                run_id=str(run.get("run_id", "")),
                expected_repeat=int(run.get("repeat", 0)),
                expected_warmup=int(run.get("warmup", 0)),
                capture_pools=capture_pools,
            )
            if passed is not None:
                series_passes.append(passed)
            if median is not None:
                series_medians.append(median)
            all_reference.extend(reference_values)
            all_candidate.extend(candidate_values)
        if mode == "eager" and _mapping(execution):
            _audit_kernel_profiles(
                findings,
                execution,
                identity_control=identity_control,
            )
        if canonical_workload is not None:
            _audit_w2_bm16_candidate(
                findings,
                result,
                canonical_workload,
                mode=mode,
                identity_control=identity_control,
                verify_files=verify_files,
            )

    if all_reference:
        _audit_latency_summary(
            findings,
            result.get("reference"),
            all_reference,
            prefix="reference",
        )
    if all_candidate and candidate_record:
        _audit_latency_summary(
            findings,
            candidate_record,
            all_candidate,
            prefix="candidate",
            allow_extra_fields=True,
        )
        recorded_series_medians = candidate_record.get(
            "series_median_speedups"
        )
        findings.require(
            _list(recorded_series_medians)
            and len(recorded_series_medians) == len(series_medians)
            and all(
                _close(recorded, expected)
                for recorded, expected in zip(
                    recorded_series_medians,
                    series_medians,
                )
            ),
            "candidate series medians do not close against raw samples",
        )

    expected_gate = False
    aggregate_value = result.get("aggregate")
    aggregate: dict[str, Any] = (
        aggregate_value if _mapping(aggregate_value) else {}
    )
    if findings.require(bool(aggregate), "missing aggregate gate"):
        required = aggregate.get("required_series")
        completed = aggregate.get("completed_series")
        findings.require(
            required == MIN_REQUIRED_SERIES,
            "aggregate.required_series must equal the contract minimum",
        )
        findings.require(
            isinstance(completed, int)
            and not isinstance(completed, bool)
            and completed == len(series)
            and completed == requested_series
            and completed >= MIN_REQUIRED_SERIES,
            "requested/completed/raw series counts do not close exactly",
        )
        findings.require(
            _close(aggregate.get("threshold"), PERFORMANCE_THRESHOLD),
            "aggregate threshold must be 1.03",
        )
        every_passes = (
            bool(series_passes)
            and len(series_passes) == len(series)
            and all(series_passes)
        )
        findings.require(
            aggregate.get("every_series_passes_3pct") is every_passes,
            "aggregate all-series gate mismatch",
        )
        expected_gate = (
            every_passes
            and not identity_control
            and fallback_count == 0
            and (
                reference_delegations == 0
                or candidate_api == TRUSTED_CONFIG_CANDIDATE_API
            )
        )
        findings.require(
            aggregate.get("performance_gate_passed") is expected_gate,
            "aggregate performance-gate decision mismatch",
        )
        if identity_control:
            findings.require(
                aggregate.get("performance_gate_passed") is False,
                "identity A/B must not pass the performance gate",
            )
        findings.require(
            aggregate.get("identity_control_forced_non_win")
            is identity_control,
            "aggregate identity-control disposition mismatch",
        )
        if (
            not identity_control
            and candidate_api != TRUSTED_CONFIG_CANDIDATE_API
            and reference_delegations > 0
        ):
            findings.require(
                aggregate.get("performance_gate_passed") is False,
                "non-identity reference delegation cannot claim a win",
            )

    valid = not findings.errors
    return {
        "valid": valid,
        "schema_version": result.get("schema_version"),
        "workload": workload.get("name") if _mapping(workload) else None,
        "execution_mode": mode,
        # This is an auditor-owned disposition, never an echo of the
        # untrusted document.  Invalid artifacts are non-promotable even when
        # their raw aggregate field claims a win.
        "performance_gate_passed": bool(valid and expected_gate),
        "errors": findings.errors,
        "warnings": findings.warnings,
    }


def audit_path(path: Path, *, verify_files: bool = True) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "valid": False,
            "errors": [f"cannot read result JSON: {exc}"],
            "warnings": [],
        }
    return audit_document(result, verify_files=verify_files)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument(
        "--no-verify-files",
        action="store_true",
        help="validate recorded hashes and schema without reading artifact paths",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = audit_path(
        args.result.expanduser().resolve(),
        verify_files=not args.no_verify_files,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    elif report["valid"]:
        disposition = (
            "PASS" if report.get("performance_gate_passed") else "VALID_NON_WIN"
        )
        print(
            f"VALID {disposition}: {report.get('workload')} "
            f"mode={report.get('execution_mode')}"
        )
    else:
        print("INVALID serving-native result", file=sys.stderr)
        for error in report["errors"]:
            print(f"- {error}", file=sys.stderr)
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
