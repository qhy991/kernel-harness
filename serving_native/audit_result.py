#!/usr/bin/env python3
"""Fail-closed auditor for serving-native schema-v2 result artifacts."""

from __future__ import annotations

import argparse
import json
import math
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
from serving_native.workloads import WORKLOADS, as_dict

CALLABLE_CANDIDATE_API = "callable_v1"
TRUSTED_CONFIG_CANDIDATE_API = "reference_with_config_v1"
W13_BASE_COMMIT = "731e7c7a97d269e4b9f482ea18d0e709a948f293"
W13_CANDIDATE_COMMIT = "87e0359edbb461181d3bba218442132007b9a738"
W13_CUTLASS_COMMIT = "f3fde58372d33e9a5650ba7b80fc48b3b49d40c8"
W13_FMT_COMMIT = "553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28"
W13_DIFF_SHA256 = "465c8373c0a37970225a0e93267b6c399431b23e22cf35b4511db2308df98092"
W13_STOCK_TREE_SHA256 = (
    "917592ab68ea0608c9be33208c2c609bc7f20bd9b1603f32743dd0d1ae03d0ed"
)
W13_CANDIDATE_TREE_SHA256 = (
    "d682daa65b8ba0ac3846d766910b8c751e0568fe62087084271bb354e46c49e4"
)
W13_BASE_BLOB_SHA256 = {
    "csrc/apis/gemm.hpp": "0840d64249e2a5a4a994d495e8320a0fff26bad9ca107426a1a1226e7d621186",
    "csrc/jit_kernels/heuristics/sm100.hpp": (
        "487cac2ff19027c781b08e9a0391836e77c03cdffcb7ceb3346d8633c8eb0884"
    ),
    "csrc/jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp": (
        "cca1ddb5b5787942c31b39a9d5618929ee609c6c3b57b877fe636df39540366b"
    ),
    "csrc/tvm_ffi_api.cpp": (
        "d1e5dbd833f257d2c4be516772404c02f1747247eef5075315ff2d1220a64c1f"
    ),
    "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh": (
        "9c1e70677ede6ba09ab98e629482da7874182f8227907382efe0a81658da5a37"
    ),
    "sgl_deep_gemm/__init__.py": (
        "243eeaa71fa65cecaddd7298245438cb371ca765d7bf914a9427e132be8d5f26"
    ),
}


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


def _close(left: Any, right: Any, *, tolerance: float = 1e-9) -> bool:
    return (
        _number(left)
        and _number(right)
        and math.isclose(
            float(left), float(right), rel_tol=tolerance, abs_tol=tolerance
        )
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
    if not findings.require(
        _mapping(implementations), "missing implementation accounting"
    ):
        return False, 0, 0, None
    reference = implementations.get("reference")
    candidate = implementations.get("candidate")
    if not findings.require(
        _mapping(reference), "missing reference implementation accounting"
    ):
        reference = {}
    if not findings.require(
        _mapping(candidate), "missing candidate implementation accounting"
    ):
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
        findings.errors.append(
            "cannot close implementation counts from invalid run metadata"
        )
        return identity_control, 0, 0, candidate_api
    expected = _expected_phase_counts(
        run_id=run_id,
        mode=mode,
        requested_series=requested_series,
        warmup=warmup,
        repeat=repeat,
    )
    by_phase = candidate.get("by_phase")
    if not findings.require(
        _mapping(by_phase), "candidate by_phase accounting missing"
    ):
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
        if all(
            isinstance(value, int) for value in (hits, fallbacks, delegations, trusted)
        ):
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
    if not findings.require(
        _list(artifacts) and bool(artifacts), "missing provenance.artifacts"
    ):
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
    if not findings.require(
        _list(modules) and bool(modules), "missing actual Python import paths"
    ):
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
        findings.require(
            isinstance(name, str) and bool(name), f"{prefix}.module missing"
        )
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
            findings.require(
                Path(path_raw).is_file(), f"{prefix}.path not found: {path_raw}"
            )
    for root in ("torch", "sglang", "deep_gemm", "serving_native_candidate"):
        findings.require(
            root in module_names
            or any(name.startswith(f"{root}.") for name in module_names),
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
        if not findings.require(
            _mapping(item), f"missing repository provenance: {name}"
        ):
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
        isinstance(hardware.get("uuid"), str) and hardware["uuid"].startswith("GPU-"),
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
    if not findings.require(
        _list(clocks) and len(clocks) >= 2, "GPU clock samples incomplete"
    ):
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
            isinstance(sample.get("sm_clock_mhz"), int) and sample["sm_clock_mhz"] > 0,
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
        _mapping(warmup_activity) and warmup_activity.get("phase") == "jit_warmup",
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


def _audit_w13_runtime(
    findings: Findings,
    provenance: dict[str, Any],
    *,
    verify_files: bool,
) -> None:
    prefix = "provenance.w13_runtime"
    runtime = provenance.get("w13_runtime")
    if not findings.require(_mapping(runtime), f"{prefix} missing"):
        return

    manifest_text = runtime.get("manifest")
    manifest_path = (
        Path(manifest_text).resolve()
        if isinstance(manifest_text, str) and manifest_text
        else None
    )
    findings.require(manifest_path is not None, f"{prefix}.manifest missing")
    manifest_sha = runtime.get("manifest_sha256")
    findings.require(
        isinstance(manifest_sha, str) and len(manifest_sha) == 64,
        f"{prefix}.manifest_sha256 invalid",
    )
    manifest = None
    if (
        verify_files
        and manifest_path is not None
        and findings.require(
            manifest_path.is_file(),
            f"{prefix}.manifest not found",
        )
    ):
        findings.require(
            sha256_file(manifest_path) == manifest_sha,
            f"{prefix}.manifest hash mismatch",
        )
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            findings.errors.append(f"{prefix}.manifest is not valid JSON")
        else:
            source = manifest.get("source", {})
            findings.require(
                manifest.get("schema_version") == 3,
                f"{prefix}.manifest schema mismatch",
            )
            expected_source = {
                "base_commit": W13_BASE_COMMIT,
                "candidate_commit": W13_CANDIDATE_COMMIT,
                "cutlass_commit": W13_CUTLASS_COMMIT,
                "fmt_commit": W13_FMT_COMMIT,
                "candidate_diff_sha256": W13_DIFF_SHA256,
                "candidate_diff_file_sha256": W13_DIFF_SHA256,
                "base_blob_sha256": W13_BASE_BLOB_SHA256,
                "stock_source_tree_sha256": W13_STOCK_TREE_SHA256,
                "candidate_source_tree_sha256": W13_CANDIDATE_TREE_SHA256,
            }
            findings.require(
                {key: source.get(key) for key in expected_source} == expected_source,
                f"{prefix}.manifest source identity mismatch",
            )
            build = manifest.get("build", {})
            required_build = {
                "cuda_arch": "10.0a",
                "stock_candidate_command_identical": True,
                "compile_api": "tvm_ffi.cpp.build",
                "force_clean_build_directories": True,
                "jit_compiler": "nvcc",
                "max_jobs": "4",
                "elf_symbol_binding": "Bsymbolic",
                "elf_symbol_visibility": "hidden",
            }
            findings.require(
                {key: build.get(key) for key in required_build} == required_build,
                f"{prefix}.manifest build contract mismatch",
            )
            build_plan_sha = build.get("normalized_build_plan_sha256")
            findings.require(
                isinstance(build_plan_sha, str) and len(build_plan_sha) == 64,
                f"{prefix}.manifest normalized build-plan hash missing",
            )
            for compiler in ("cxx", "nvcc"):
                compiler_path = Path(str(build.get(f"{compiler}_path", ""))).resolve()
                compiler_sha = build.get(f"{compiler}_sha256")
                findings.require(
                    compiler_path.is_file()
                    and isinstance(compiler_sha, str)
                    and sha256_file(compiler_path) == compiler_sha,
                    f"{prefix}.manifest {compiler} identity mismatch",
                )

    variant = runtime.get("variant")
    configs = {
        "bm16_2sm": [16, 128, 128, 12, 2],
        "bm16_1sm": [16, 128, 128, 11, 1],
    }
    findings.require(
        runtime.get("manifest_schema") == 3,
        f"{prefix}.manifest_schema mismatch",
    )
    findings.require(variant in configs, f"{prefix}.variant invalid")
    if variant in configs:
        findings.require(
            runtime.get("config") == configs[variant],
            f"{prefix}.config does not match variant",
        )
    findings.require(
        runtime.get("broad_precompile_enabled") is False,
        f"{prefix} broad precompile was not disabled",
    )
    findings.require(
        runtime.get("jit_use_nvrtc") is False,
        f"{prefix} did not freeze the NVCC JIT backend",
    )
    findings.require(
        runtime.get("candidate_call_path")
        == (
            "sglang.glm52_opt.hotspot_provider.run_moe_masked"
            " -> API-v1 provider moe_w13 -> exact DeepGEMM symbol"
        ),
        f"{prefix}.candidate_call_path mismatch",
    )

    provider_identity: Any = None
    provider = runtime.get("provider")
    if findings.require(_mapping(provider), f"{prefix}.provider missing"):
        provider_path_text = provider.get("path")
        provider_path = (
            Path(provider_path_text).resolve()
            if isinstance(provider_path_text, str) and provider_path_text
            else None
        )
        expected_provider_name = {
            "bm16_2sm": "provider_bm16_2sm.py",
            "bm16_1sm": "provider_bm16_1sm.py",
        }.get(variant)
        findings.require(
            provider_path is not None
            and provider_path.name == expected_provider_name,
            f"{prefix}.provider path does not match variant",
        )
        findings.require(
            isinstance(provider.get("sha256"), str)
            and len(provider["sha256"]) == 64,
            f"{prefix}.provider.sha256 invalid",
        )
        if verify_files and provider_path is not None:
            findings.require(
                provider_path.is_file()
                and sha256_file(provider_path) == provider.get("sha256"),
                f"{prefix}.provider file identity mismatch",
            )
        state = provider.get("state")
        if findings.require(_mapping(state), f"{prefix}.provider.state missing"):
            provider_info = state.get("provider_info")
            expected_provider_identity = {
                "bm16_2sm": (
                    "infini_kernel_glm52_moe_w13_decode_bm16_2sm",
                    "bm16-2sm-stage12-api-v1",
                ),
                "bm16_1sm": (
                    "infini_kernel_glm52_moe_w13_decode_bm16_1sm",
                    "bm16-1sm-stage11-api-v1",
                ),
            }.get(variant)
            findings.require(
                state.get("ready") is True
                and state.get("reason") == "ready"
                and state.get("selected_ops") == ["moe_gate_proj"],
                f"{prefix}.provider state is not the selected ready W13 provider",
            )
            if findings.require(
                _mapping(provider_info),
                f"{prefix}.provider.state.provider_info missing",
            ) and expected_provider_identity is not None:
                findings.require(
                    provider_info.get("name") == expected_provider_identity[0]
                    and provider_info.get("build_id") == expected_provider_identity[1]
                    and provider_info.get("git_commit") == W13_CANDIDATE_COMMIT,
                    f"{prefix}.provider API-v1 identity mismatch",
                )
            if provider_path is not None:
                findings.require(
                    isinstance(state.get("module_ref"), str)
                    and Path(state["module_ref"]).resolve() == provider_path,
                    f"{prefix}.provider module_ref mismatch",
                )
        provider_identity = provider.get("identity")
        if findings.require(
            _mapping(provider_identity),
            f"{prefix}.provider.identity missing",
        ):
            findings.require(
                variant in configs
                and provider_identity.get("name") == variant
                and provider_identity.get("config") == configs.get(variant),
                f"{prefix}.provider runtime identity mismatch",
            )
            findings.require(
                provider_identity.get("manifest_sha256") == manifest_sha,
                f"{prefix}.provider manifest identity mismatch",
            )
            findings.require(
                manifest_path is not None
                and isinstance(provider_identity.get("manifest"), str)
                and Path(provider_identity["manifest"]).resolve() == manifest_path,
                f"{prefix}.provider manifest path mismatch",
            )

    required_state = {"pdl": True, "num_sms": 148, "tc_util": 100}
    states = runtime.get("runtime_state")
    if findings.require(_mapping(states), f"{prefix}.runtime_state missing"):
        for name in ("installed_downstream", "stock", "candidate"):
            findings.require(
                states.get(name) == required_state,
                f"{prefix}.runtime_state.{name} mismatch",
            )

    independence = runtime.get("state_independence")
    mutations = {"pdl": False, "num_sms": 147, "tc_util": 99}
    if findings.require(_mapping(independence), f"{prefix}.state_independence missing"):
        for field, mutation in mutations.items():
            proof = independence.get(field)
            if not findings.require(
                _mapping(proof), f"{prefix}.state_independence.{field} missing"
            ):
                continue
            for mutated, other in (
                ("stock", "candidate"),
                ("candidate", "stock"),
            ):
                direction = proof.get(f"mutate_{mutated}")
                if not findings.require(
                    _mapping(direction),
                    f"{prefix}.state_independence.{field}.mutate_{mutated} missing",
                ):
                    continue
                findings.require(
                    direction.get("mutated_value") == mutation,
                    f"{prefix}.{field} {mutated} mutation mismatch",
                )
                findings.require(
                    direction.get(f"{other}_unchanged") == required_state[field],
                    f"{prefix}.{field} changed independent {other} runtime",
                )
                findings.require(
                    direction.get("restored") == required_state[field],
                    f"{prefix}.{field} {mutated} restore mismatch",
                )

    modules = runtime.get("modules")
    resolved: dict[str, dict[str, Path]] = {}
    if findings.require(_mapping(modules), f"{prefix}.modules missing"):
        for name in ("stock", "candidate"):
            item = modules.get(name)
            if not findings.require(_mapping(item), f"{prefix}.modules.{name} missing"):
                continue
            paths: dict[str, Path] = {}
            for field in ("package", "shared_object", "jit_cache"):
                value = item.get(field)
                if findings.require(
                    isinstance(value, str) and bool(value),
                    f"{prefix}.modules.{name}.{field} missing",
                ):
                    paths[field] = Path(value).resolve()
            resolved[name] = paths
            for field in ("package_init_sha256", "shared_object_sha256"):
                findings.require(
                    isinstance(item.get(field), str) and len(item[field]) == 64,
                    f"{prefix}.modules.{name}.{field} invalid",
                )
            artifacts = item.get("jit_artifacts")
            findings.require(
                _mapping(artifacts) and bool(artifacts),
                f"{prefix}.modules.{name}.jit_artifacts empty",
            )
            if (
                verify_files
                and {"package", "shared_object", "jit_cache"} <= paths.keys()
            ):
                init_py = paths["package"] / "__init__.py"
                shared_object = paths["shared_object"]
                findings.require(
                    init_py.is_file()
                    and sha256_file(init_py) == item.get("package_init_sha256"),
                    f"{prefix}.modules.{name} package init hash mismatch",
                )
                findings.require(
                    shared_object.is_file()
                    and sha256_file(shared_object) == item.get("shared_object_sha256"),
                    f"{prefix}.modules.{name} DSO hash mismatch",
                )
                for relative, digest in (
                    artifacts.items() if _mapping(artifacts) else ()
                ):
                    relative_path = Path(relative)
                    safe_relative = (
                        not relative_path.is_absolute()
                        and ".." not in relative_path.parts
                    )
                    findings.require(
                        safe_relative and isinstance(digest, str) and len(digest) == 64,
                        f"{prefix}.modules.{name} JIT artifact identity invalid",
                    )
                    if safe_relative:
                        artifact = paths["jit_cache"] / relative_path
                        findings.require(
                            artifact.is_file() and sha256_file(artifact) == digest,
                            f"{prefix}.modules.{name} JIT artifact hash mismatch",
                        )
            if manifest is not None:
                manifest_record = manifest.get("variants", {}).get(name, {})
                expected_tree = (
                    W13_STOCK_TREE_SHA256
                    if name == "stock"
                    else W13_CANDIDATE_TREE_SHA256
                )
                required_record = {
                    "package": item.get("package"),
                    "package_init_sha256": item.get("package_init_sha256"),
                    "shared_object": item.get("shared_object"),
                    "shared_object_sha256": item.get("shared_object_sha256"),
                    "jit_cache": item.get("jit_cache"),
                    "commit": (
                        W13_BASE_COMMIT
                        if name == "stock"
                        else W13_CANDIDATE_COMMIT
                    ),
                    "source_tree_sha256": expected_tree,
                    "normalized_build_plan_sha256": manifest.get("build", {}).get(
                        "normalized_build_plan_sha256"
                    ),
                }
                findings.require(
                    {key: manifest_record.get(key) for key in required_record}
                    == required_record,
                    f"{prefix}.modules.{name} does not match manifest variant",
                )
                build_ninja_text = manifest_record.get("build_ninja")
                build_ninja = (
                    Path(build_ninja_text).resolve()
                    if isinstance(build_ninja_text, str)
                    else None
                )
                findings.require(
                    build_ninja is not None
                    and build_ninja.is_file()
                    and sha256_file(build_ninja)
                    == manifest_record.get("build_ninja_sha256"),
                    f"{prefix}.modules.{name} build-plan file mismatch",
                )
    if all(name in resolved for name in ("stock", "candidate")):
        for field in ("package", "shared_object", "jit_cache"):
            if all(field in resolved[name] for name in ("stock", "candidate")):
                findings.require(
                    resolved["stock"][field] != resolved["candidate"][field],
                    f"{prefix} stock/candidate {field} paths alias",
                )
    if _mapping(provider_identity) and _mapping(modules):
        candidate_module = modules.get("candidate")
        if findings.require(
            _mapping(candidate_module),
            f"{prefix}.provider has no candidate module binding",
        ):
            for field in (
                "shared_object",
                "shared_object_sha256",
                "jit_cache",
                "jit_artifacts",
            ):
                findings.require(
                    provider_identity.get(field) == candidate_module.get(field),
                    f"{prefix}.provider {field} does not match candidate module",
                )


def _audit_correctness(
    findings: Findings,
    result: dict[str, Any],
    mode: str | None,
) -> None:
    correctness = result.get("correctness")
    if not findings.require(_mapping(correctness), "missing correctness record"):
        return
    findings.require(
        correctness.get("status") == "pass", "correctness status is not pass"
    )
    for field in (
        "pre_timing_reference",
        "pre_timing_candidate",
        "post_timing_reference",
        "post_timing_candidate",
        "fresh_inputs_post_timing",
    ):
        findings.require(
            correctness.get(field) is True, f"correctness.{field} did not pass"
        )
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


def _recompute_performance_estimates(
    raw_samples: list[Any],
) -> dict[str, Any]:
    if not raw_samples or len(raw_samples) % 2:
        raise ValueError("incomplete ordered sample pairs")
    reference_values: list[float] = []
    candidate_values: list[float] = []
    by_order: dict[str, list[float]] = {"AB": [], "BA": []}
    for offset in range(0, len(raw_samples), 2):
        pair = raw_samples[offset : offset + 2]
        if not all(_mapping(sample) for sample in pair):
            raise ValueError("sample pair is not object-valued")
        order = pair[0].get("order")
        if order not in by_order or pair[1].get("order") != order:
            raise ValueError("sample pair order mismatch")
        values: dict[str, float] = {}
        for sample in pair:
            implementation = sample.get("implementation")
            latency = sample.get("latency_ms")
            if (
                implementation not in ("reference", "candidate")
                or implementation in values
                or not _number(latency)
                or float(latency) <= 0.0
            ):
                raise ValueError("sample pair latency/implementation invalid")
            values[implementation] = float(latency)
        if set(values) != {"reference", "candidate"}:
            raise ValueError("sample pair is incomplete")
        ratio = values["reference"] / values["candidate"]
        if not math.isfinite(ratio) or ratio <= 0.0:
            raise ValueError("sample pair ratio is non-finite")
        reference_values.append(values["reference"])
        candidate_values.append(values["candidate"])
        by_order[order].append(ratio)
    if not by_order["AB"] or not by_order["BA"]:
        raise ValueError("both AB and BA observations are required")
    pooled = statistics.median(reference_values) / statistics.median(candidate_values)
    ab_estimate = statistics.median(by_order["AB"])
    ba_estimate = statistics.median(by_order["BA"])
    order_balanced = math.sqrt(ab_estimate * ba_estimate)
    if not all(
        math.isfinite(value) and value > 0.0
        for value in (pooled, order_balanced, ab_estimate, ba_estimate)
    ):
        raise ValueError("performance estimate is non-finite")
    return {
        "contract": "finite_pooled_order_balanced_ab_ba_v1",
        "pair_count": len(raw_samples) // 2,
        "pooled_speedup": pooled,
        "order_balanced_speedup": order_balanced,
        "ab_median_speedup": ab_estimate,
        "ba_median_speedup": ba_estimate,
        "all_finite": True,
    }


def _four_estimator_gate_passes(estimates: dict[str, Any]) -> bool:
    return all(
        _number(estimates.get(field))
        and float(estimates[field]) >= PERFORMANCE_THRESHOLD
        for field in (
            "pooled_speedup",
            "order_balanced_speedup",
            "ab_median_speedup",
            "ba_median_speedup",
        )
    )


def _audit_performance_estimates(
    findings: Findings,
    recorded: Any,
    raw_samples: list[Any],
    *,
    prefix: str,
) -> bool:
    if not findings.require(
        _mapping(recorded),
        f"{prefix} performance estimates are missing",
    ):
        return False
    try:
        expected = _recompute_performance_estimates(raw_samples)
    except ValueError as exc:
        findings.require(False, f"{prefix} estimates cannot be recomputed: {exc}")
        return False
    findings.require(
        set(recorded) == set(expected),
        f"{prefix} performance-estimate fields do not close",
    )
    valid = True
    for field, expected_value in expected.items():
        if isinstance(expected_value, float):
            matches = _close(recorded.get(field), expected_value)
        else:
            matches = recorded.get(field) == expected_value
            if field == "pair_count":
                matches = bool(
                    matches
                    and isinstance(recorded.get(field), int)
                    and not isinstance(recorded.get(field), bool)
                )
            if field == "all_finite":
                matches = recorded.get(field) is True
        findings.require(
            matches,
            f"{prefix}.performance_estimates.{field} mismatch",
        )
        valid = bool(valid and matches)
    return valid


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
    findings.require(
        series.get("series_index") == index, f"{prefix}.series_index mismatch"
    )
    expected_series_id = f"{run_id}:series-{index + 1:02d}"
    findings.require(
        series.get("series_id") == expected_series_id,
        f"{prefix}.series_id is not bound to run identity",
    )
    findings.require(series.get("independent") is True, f"{prefix} is not independent")
    findings.require(
        series.get("execution_mode") == mode, f"{prefix} execution-mode mismatch"
    )
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
        findings.require(
            sample.get("sequence") == sequence, f"{sample_prefix}.sequence mismatch"
        )
        findings.require(
            isinstance(pair_index, int) and 0 <= pair_index < repeat,
            f"{sample_prefix}.pair_index invalid",
        )
        findings.require(position in (0, 1), f"{sample_prefix}.position invalid")
        if not isinstance(pair_index, int) or position not in (0, 1):
            continue
        order = _expected_order(start_order, pair_index)
        expected_impl = (
            ("reference", "candidate") if order == "AB" else ("candidate", "reference")
        )[position]
        implementation = sample.get("implementation")
        findings.require(
            sample.get("order") == order, f"{sample_prefix}.order mismatch"
        )
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
            if (
                findings.require(
                    bool(pool),
                    f"{sample_prefix} has no independently captured graph pool",
                )
                and implementation in capture_ordinals
            ):
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
    estimates_valid = _audit_performance_estimates(
        findings,
        series.get("performance_estimates"),
        samples,
        prefix=prefix,
    )
    try:
        recomputed_estimates = _recompute_performance_estimates(samples)
    except ValueError:
        passed = False
    else:
        passed = bool(
            estimates_valid and _four_estimator_gate_passes(recomputed_estimates)
        )
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
    workload_family: str,
    candidate_store_block_m: int,
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
    if not findings.require(
        _list(captures) and len(captures) == 4, f"{prefix} capture set incomplete"
    ):
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
        if not findings.require(
            _mapping(capture), f"{capture_prefix} must be an object"
        ):
            continue
        capture_id = capture.get("capture_id")
        expected_capture_id = f"{series_id}:{expected_suffixes[capture_index]}"
        findings.require(
            capture_id == expected_capture_id,
            f"{capture_prefix}.capture_id is not bound to its series/capture order",
        )
        if isinstance(capture_id, str):
            findings.require(
                capture_id not in capture_ids,
                f"duplicate graph capture id: {capture_id}",
            )
            capture_ids.add(capture_id)
        implementation = capture.get("implementation")
        findings.require(
            implementation == expected_implementations[capture_index],
            f"{capture_prefix}.implementation/capture order mismatch",
        )
        if isinstance(capture_id, str) and implementation in ("reference", "candidate"):
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
        findings.require(
            computed_non_default, f"{capture_prefix} did not use a non-default stream"
        )
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
        if workload_family in ("moe_grouped_masked", "moe_w13_region"):
            for field in (
                "masked_m_mutation_replayed",
                "untouched_masked_regions_preserved",
            ):
                findings.require(
                    capture.get(field) is True,
                    f"{capture_prefix}.{field} did not pass",
                )
            findings.require(
                capture.get("masked_store_contract")
                == "poison preserved outside scheduled full store_block_m tiles",
                f"{capture_prefix}.masked_store_contract invalid",
            )
            store_observation = capture.get("masked_store_observation")
            if findings.require(
                _mapping(store_observation),
                f"{capture_prefix}.masked_store_observation missing",
            ):
                expected_outputs = {"out"}
                if workload_family == "moe_w13_region":
                    expected_outputs.add("down_out")
                findings.require(
                    set(store_observation) == expected_outputs,
                    f"{capture_prefix}.masked_store_observation outputs mismatch",
                )
                for output_name in expected_outputs:
                    output = store_observation.get(output_name)
                    output_prefix = (
                        f"{capture_prefix}.masked_store_observation.{output_name}"
                    )
                    if not findings.require(
                        _mapping(output),
                        f"{output_prefix} missing",
                    ):
                        continue
                    expected_block_m = 128
                    if (
                        output_name == "out"
                        and capture.get("implementation") == "candidate"
                        and capture.get("reference_delegated") is False
                    ):
                        expected_block_m = candidate_store_block_m
                    findings.require(
                        output.get("store_block_m") == expected_block_m,
                        f"{output_prefix}.store_block_m mismatch",
                    )
                    for field in (
                        "padding_rows_written",
                        "untouched_rows_checked",
                    ):
                        findings.require(
                            isinstance(output.get(field), int)
                            and not isinstance(output.get(field), bool)
                            and output[field] >= 0,
                            f"{output_prefix}.{field} invalid",
                        )
                    findings.require(
                        isinstance(output.get("untouched_rows_checked"), int)
                        and output["untouched_rows_checked"] > 0,
                        f"{output_prefix} did not check any untouched rows",
                    )
        nodes = capture.get("nodes")
        findings.require(
            _list(nodes) and bool(nodes), f"{capture_prefix}.nodes missing"
        )
        if _list(nodes):
            for node_index, node in enumerate(nodes):
                node_prefix = f"{capture_prefix}.nodes[{node_index}]"
                if not findings.require(
                    _mapping(node), f"{node_prefix} must be an object"
                ):
                    continue
                findings.require(
                    node.get("index") == node_index,
                    f"{node_prefix}.index mismatch",
                )
                node_type = node.get("type")
                findings.require(
                    isinstance(node_type, str) and bool(node_type),
                    f"{node_prefix}.type missing",
                )
                if isinstance(node_type, str) and "KERNEL" in node_type:
                    findings.require(
                        isinstance(node.get("kernel"), str) and bool(node["kernel"]),
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
            findings.require(
                capture.get("node_count") == node_count,
                f"{capture_prefix}.node_count does not match nodes",
            )
            findings.require(
                capture.get("node_type_counts") == node_type_counts,
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
            findings.require(
                not forbidden, f"{capture_prefix} has forbidden graph nodes"
            )
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
        len(capture_ids) == 4 and len(raw_graph_handles) == 4 and len(stream_ids) == 4
    )
    findings.require(
        graph.get("reference_candidate_captured_independently") is independent,
        f"{prefix} independent-capture flag does not match capture IDs",
    )
    findings.require(independent, f"{prefix} captures are not independent")
    if identity_control and reference_signatures and candidate_signatures:
        canonical = reference_signatures[0]
        findings.require(
            all(
                signature == canonical
                for signature in reference_signatures + candidate_signatures
            ),
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
    if not findings.require(
        _mapping(profiles), "eager execution lacks kernel profiler capture"
    ):
        return
    identities: dict[str, list[str]] = {}
    for implementation in ("reference", "candidate"):
        profile = profiles.get(implementation)
        prefix = f"execution.kernel_profiles.{implementation}"
        if not findings.require(_mapping(profile), f"{prefix} missing"):
            continue
        findings.require(profile.get("captured") is True, f"{prefix} was not captured")
        kernels = profile.get("kernel_identities")
        findings.require(
            _list(kernels) and bool(kernels), f"{prefix} kernel identities missing"
        )
        events = profile.get("events")
        findings.require(_list(events) and bool(events), f"{prefix} events missing")
        recomputed: list[str] = []
        if _list(events):
            names: set[str] = set()
            for event_index, event in enumerate(events):
                event_prefix = f"{prefix}.events[{event_index}]"
                if not findings.require(
                    _mapping(event), f"{event_prefix} must be an object"
                ):
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


def audit_document(
    result: Any,
    *,
    verify_files: bool = True,
) -> dict[str, Any]:
    findings = Findings()
    if not findings.require(_mapping(result), "result root must be an object"):
        return {
            "valid": False,
            "errors": findings.errors,
            "warnings": findings.warnings,
        }
    findings.require(
        result.get("schema_version") == SCHEMA_VERSION, "schema_version must be 2"
    )
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
        findings.require(
            _list(run.get("command")) and bool(run["command"]), "run.command missing"
        )
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
            _list(workload.get("execution_modes"))
            and bool(workload["execution_modes"]),
            "workload.execution_modes missing",
        )
        workload_name = workload.get("name")
        registered = (
            WORKLOADS.get(workload_name) if isinstance(workload_name, str) else None
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
            isinstance(execution.get("timer"), str) and bool(execution["timer"]),
            "execution timer identity missing",
        )
        if mode == "cuda_graph":
            findings.require(
                execution.get("capture_stream") == "independent non-default streams",
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
    provenance: dict[str, Any] = provenance_value if _mapping(provenance_value) else {}
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
        if canonical_workload is not None and str(
            canonical_workload.get("name", "")
        ).startswith("moe_w13_"):
            _audit_w13_runtime(
                findings,
                provenance,
                verify_files=verify_files,
            )

    identity_control = False
    fallback_count = 0
    reference_delegations = 0
    candidate_api: str | None = None
    if mode in ("eager", "cuda_graph") and canonical_workload is not None and run:
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
    all_raw_samples: list[Any] = []
    all_reference: list[float] = []
    all_candidate: list[float] = []
    series_ids: set[str] = set()
    candidate_store_block_m = 32
    if canonical_workload is not None and str(
        canonical_workload.get("name", "")
    ).startswith("moe_w13_"):
        w13_runtime = provenance.get("w13_runtime")
        w13_config = (
            w13_runtime.get("config") if _mapping(w13_runtime) else None
        )
        if (
            _list(w13_config)
            and len(w13_config) == 5
            and isinstance(w13_config[0], int)
            and not isinstance(w13_config[0], bool)
        ):
            candidate_store_block_m = w13_config[0]
    exact_series_count = (
        isinstance(requested_series, int)
        and not isinstance(requested_series, bool)
        and len(series) == requested_series
        and len(series) >= MIN_REQUIRED_SERIES
    )
    if findings.require(
        exact_series_count, "requested/raw series counts do not close exactly"
    ) and isinstance(mode, str):
        for index, item in enumerate(series):
            if not findings.require(
                _mapping(item), f"series[{index}] must be an object"
            ):
                continue
            series_id = item.get("series_id")
            if isinstance(series_id, str):
                findings.require(
                    series_id not in series_ids,
                    f"duplicate series identity: {series_id}",
                )
                series_ids.add(series_id)
            raw_samples = item.get("raw_ordered_samples")
            if _list(raw_samples):
                all_raw_samples.extend(raw_samples)
            capture_pools = None
            if mode == "cuda_graph":
                capture_pools = _audit_graph_series(
                    findings,
                    item,
                    index=index,
                    identity_control=identity_control,
                    candidate_api=candidate_api,
                    workload_family=str(canonical_workload.get("family", "")),
                    candidate_store_block_m=candidate_store_block_m,
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
        recorded_series_medians = candidate_record.get("series_median_speedups")
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
    aggregate: dict[str, Any] = aggregate_value if _mapping(aggregate_value) else {}
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
        findings.require(
            aggregate.get("series_gate_contract")
            == "all_four_estimates_each_series_gte_1p03_v1",
            "aggregate series-gate contract mismatch",
        )
        aggregate_estimates_valid = _audit_performance_estimates(
            findings,
            aggregate.get("performance_estimates"),
            all_raw_samples,
            prefix="aggregate",
        )
        required_estimates_finite = bool(
            aggregate_estimates_valid
            and all_raw_samples
            and all(
                _mapping(item)
                and _mapping(item.get("performance_estimates"))
                and item["performance_estimates"].get("all_finite") is True
                for item in series
            )
        )
        findings.require(
            aggregate.get("required_estimates_finite") is required_estimates_finite,
            "aggregate required-estimates-finite mismatch",
        )
        expected_gate = (
            every_passes
            and required_estimates_finite
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
            aggregate.get("identity_control_forced_non_win") is identity_control,
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
