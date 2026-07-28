"""GPU-free adversarial regression tests for the serving-native V2 auditor."""

from __future__ import annotations

import copy
import inspect
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from serving_native.audit_result import (
    W2_BM16_BASE_COMMIT,
    W2_BM16_CACHE_DIR,
    W2_BM16_CANDIDATE_EXTENSION_SHA256,
    W2_BM16_CUTLASS_COMMIT,
    W2_BM16_FMT_COMMIT,
    W2_BM16_SOURCE_PATCH_SHA256,
    W2_BM16_STOCK_EXTENSION_SHA256,
    audit_document,
)
from serving_native.contract_v2 import (
    canonical_sha256,
    file_artifact,
    graph_forbidden_nodes,
    graph_kernel_identities,
    graph_node_type_counts,
    latency_summary,
    module_path_snapshot,
)
from serving_native.runner import (
    Runtime,
    TaskResult,
    _graph_mutation_target,
    _load_candidate,
)
from serving_native.workloads import (
    W2_EDGE_MASK_CASES,
    WORKLOADS,
    as_dict,
    get_workload,
)

HERE = Path(__file__).resolve().parent
RUN_ID = "fixture-run"
SERIES = 3
WARMUP = 3
REPEAT = 2
CALLABLE_API = "callable_v1"
TRUSTED_CONFIG_API = "reference_with_config_v1"
COUNTER_FIELDS = (
    "reference_calls",
    "candidate_hits",
    "candidate_fallbacks",
    "candidate_reference_delegations",
    "candidate_trusted_config_calls",
)


class ContractV2AuditTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.runner = HERE / "runner.py"
        self.workloads = HERE / "workloads.py"
        self.candidate = self._file("candidate.py", "def run(inputs, runtime):\n    pass\n")
        self.torch = self._file("torch.py", "torch\n")
        self.sglang = self._file("sglang.py", "sglang\n")
        self.deep_gemm = self._file("deep_gemm.py", "deep_gemm\n")
        self.shared_object = self._file("candidate_kernel.so", "shared\n")
        self.valid_eager = self._valid_fixture("eager")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _file(self, name: str, contents: str) -> Path:
        path = self.root / name
        path.write_text(contents)
        return path

    @staticmethod
    def _capture_ids(series_index: int) -> dict[str, list[str]]:
        series_id = f"{RUN_ID}:series-{series_index + 1:02d}"
        return {
            "reference": [
                f"{series_id}:R-first",
                f"{series_id}:R-after-C",
            ],
            "candidate": [
                f"{series_id}:C-after-R",
                f"{series_id}:C-first",
            ],
        }

    @classmethod
    def _samples(cls, series_index: int, mode: str) -> list[dict]:
        start = "AB" if series_index % 2 == 0 else "BA"
        samples: list[dict] = []
        capture_ids = cls._capture_ids(series_index)
        capture_ordinals = {"reference": 0, "candidate": 0}
        for pair_index in range(REPEAT):
            order = (
                start
                if pair_index % 2 == 0
                else ("BA" if start == "AB" else "AB")
            )
            implementations = (
                ("reference", "candidate")
                if order == "AB"
                else ("candidate", "reference")
            )
            for position, implementation in enumerate(implementations):
                sample = {
                    "sequence": len(samples),
                    "pair_index": pair_index,
                    "position": position,
                    "order": order,
                    "implementation": implementation,
                    "label": "A" if implementation == "reference" else "B",
                    "latency_ms": 1.0,
                }
                if mode == "cuda_graph":
                    ordinal = capture_ordinals[implementation]
                    sample["graph_capture_id"] = capture_ids[implementation][
                        ordinal % 2
                    ]
                    capture_ordinals[implementation] += 1
                samples.append(sample)
        return samples

    @staticmethod
    def _capture(
        series_index: int,
        capture_index: int,
        implementation: str,
        suffix: str,
        *,
        identity_control: bool,
        candidate_api: str,
    ) -> dict:
        series_id = f"{RUN_ID}:series-{series_index + 1:02d}"
        is_candidate = implementation == "candidate"
        reference_delegated = is_candidate and (
            identity_control or candidate_api == TRUSTED_CONFIG_API
        )
        trusted_config = is_candidate and candidate_api == TRUSTED_CONFIG_API
        node = {
            "index": 0,
            "type": "CU_GRAPH_NODE_TYPE_KERNEL",
            "kernel": "fixture_kernel",
            "grid": [1, 1, 1],
            "block": [32, 1, 1],
            "shared_memory_bytes": 0,
        }
        return {
            "capture_id": f"{series_id}:{suffix}",
            "implementation": implementation,
            "raw_graph_handle": 10_000 + series_index * 10 + capture_index,
            "stream_id": 20_000 + series_index * 10 + capture_index,
            "default_stream_id": 0,
            "non_default_stream": True,
            "node_count": 1,
            "nodes": [node],
            "node_type_counts": {"CU_GRAPH_NODE_TYPE_KERNEL": 1},
            "kernel_identities": ["fixture_kernel"],
            "forbidden_nodes": [],
            "stable_input_pointers": True,
            "stable_output_pointers": True,
            "input_mutation_replayed": True,
            "output_poison_replayed": True,
            "deterministic_replay": True,
            "approved_tolerance_passed": True,
            "fallback": False,
            "reference_delegated": reference_delegated,
            "trusted_config": trusted_config,
        }

    @staticmethod
    def _refresh_graph_metadata(capture: dict) -> None:
        nodes = capture["nodes"]
        capture["node_count"] = len(nodes)
        capture["node_type_counts"] = graph_node_type_counts(nodes)
        capture["kernel_identities"] = graph_kernel_identities(nodes)
        capture["forbidden_nodes"] = graph_forbidden_nodes(nodes)

    @classmethod
    def _append_graph_kernel(cls, capture: dict, kernel: str) -> None:
        node = copy.deepcopy(capture["nodes"][0])
        node["index"] = len(capture["nodes"])
        node["kernel"] = kernel
        capture["nodes"].append(node)
        cls._refresh_graph_metadata(capture)

    @classmethod
    def _series(
        cls,
        index: int,
        mode: str,
        *,
        identity_control: bool,
        candidate_api: str,
    ) -> dict:
        samples = cls._samples(index, mode)
        item = {
            "series_index": index,
            "series_id": f"{RUN_ID}:series-{index + 1:02d}",
            "independent": True,
            "execution_mode": mode,
            "start_order": "AB" if index % 2 == 0 else "BA",
            "warmup_pairs": WARMUP,
            "repeat": REPEAT,
            "raw_ordered_samples": samples,
            "reference": latency_summary([1.0, 1.0]),
            "candidate": latency_summary([1.0, 1.0]),
            "paired_speedups": [1.0, 1.0],
            "median_speedup": 1.0,
            "passes_3pct_gate": False,
        }
        if mode == "cuda_graph":
            plan = (
                ("reference", "R-first"),
                ("candidate", "C-after-R"),
                ("candidate", "C-first"),
                ("reference", "R-after-C"),
            )
            item["graph"] = {
                "capture_policy": "bidirectional_R-C_then_C-R_round_robin",
                "reference_candidate_captured_independently": True,
                "captures": [
                    cls._capture(
                        index,
                        capture_index,
                        implementation,
                        suffix,
                        identity_control=identity_control,
                        candidate_api=candidate_api,
                    )
                    for capture_index, (implementation, suffix) in enumerate(plan)
                ],
            }
        return item

    @staticmethod
    def _phase_counts(
        mode: str,
        *,
        workload: dict,
        identity_control: bool,
        candidate_api: str,
    ) -> dict[str, dict[str, int]]:
        phases: dict[str, dict[str, int]] = {}

        def add(phase: str, reference: int, candidate: int) -> None:
            item = phases.setdefault(
                phase,
                {
                    "reference_calls": 0,
                    "candidate_hits": 0,
                    "candidate_fallbacks": 0,
                    "candidate_reference_delegations": 0,
                    "candidate_trusted_config_calls": 0,
                },
            )
            item["reference_calls"] += reference
            item["candidate_hits"] += candidate
            if identity_control or candidate_api == TRUSTED_CONFIG_API:
                item["candidate_reference_delegations"] += candidate
            if candidate_api == TRUSTED_CONFIG_API:
                item["candidate_trusted_config_calls"] += candidate

        add("pre_timing_correctness", 1, 1)
        add("jit_warmup", max(3, WARMUP) + 1, max(3, WARMUP) + 1)
        for series_index in range(SERIES):
            series_id = f"{RUN_ID}:series-{series_index + 1:02d}"
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
                add("graph_validation", 8, 6)
            add(f"{series_id}:warmup", WARMUP, WARMUP)
            add(f"{series_id}:timing", REPEAT, REPEAT)
        add("post_timing_correctness", 1, 1)
        add("fresh_inputs_correctness", 1, 1)
        if mode == "eager":
            add("profiler_reference", 1, 0)
            add("profiler_candidate", 0, 1)
        params = workload.get("params")
        is_w2_leaf = (
            workload.get("family") == "moe_grouped_masked"
            and isinstance(params, dict)
            and "candidate_jit_identity" in params
        )
        if is_w2_leaf:
            for case_index, (name, _counts) in enumerate(
                W2_EDGE_MASK_CASES
            ):
                add(f"edge:{name}:eager", 1, 1)
                if mode == "cuda_graph":
                    for implementation in ("reference", "candidate"):
                        is_reference = implementation == "reference"
                        capture_id = (
                            f"edge:{case_index:02d}:{name}:"
                            f"{implementation}"
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

    @staticmethod
    def _jit_phases(mode: str) -> list[str]:
        phases: list[str] = []
        for series_index in range(SERIES):
            series_id = f"{RUN_ID}:series-{series_index + 1:02d}"
            if mode == "cuda_graph":
                phases.extend(
                    f"{series_id}:{suffix}:capture"
                    for suffix in (
                        "R-first",
                        "C-after-R",
                        "C-first",
                        "R-after-C",
                    )
                )
            phases.append(f"{series_id}:timing")
        return phases

    @staticmethod
    def _clean_observation(phase: str) -> dict:
        return {
            "phase": phase,
            "clean": True,
            "new_imports": [],
            "new_shared_objects": [],
            "cache_changes": [],
            "candidate_artifact_changes": [],
        }

    def _valid_fixture(
        self,
        mode: str,
        *,
        task: str = "linear_attn_o_decode_m16",
        identity_control: bool = True,
        candidate_api: str = CALLABLE_API,
    ) -> dict:
        workload = as_dict(get_workload(task))
        series = [
            self._series(
                index,
                mode,
                identity_control=identity_control,
                candidate_api=candidate_api,
            )
            for index in range(SERIES)
        ]
        kernel_profile = {
            "captured": True,
            "events": [
                {
                    "name": "fixture_kernel",
                    "duration_us": 1.0,
                    "device_type": "DeviceType.CUDA",
                }
            ],
            "kernel_identities": ["fixture_kernel"],
        }
        by_phase = self._phase_counts(
            mode,
            workload=workload,
            identity_control=identity_control,
            candidate_api=candidate_api,
        )
        totals = {
            field: sum(item[field] for item in by_phase.values())
            for field in COUNTER_FIELDS
        }
        return {
            "schema_version": 2,
            "result_kind": "serving_native_v2",
            "run": {
                "run_id": RUN_ID,
                "started_utc": "2026-07-28T00:00:00Z",
                "finished_utc": "2026-07-28T00:01:00Z",
                "command": ["python", str(self.runner)],
                "requested_series": SERIES,
                "warmup": WARMUP,
                "repeat": REPEAT,
            },
            "workload": workload,
            "execution": {
                "mode": mode,
                "timer": "CUDA events",
                "reference_candidate_captured_separately": mode == "cuda_graph",
                "capture_stream": (
                    "independent non-default streams"
                    if mode == "cuda_graph"
                    else None
                ),
                "graph_capture_policy": (
                    "bidirectional_R-C_then_C-R_round_robin"
                    if mode == "cuda_graph"
                    else None
                ),
                "kernel_profiles": (
                    None
                    if mode == "cuda_graph"
                    else {
                        "reference": copy.deepcopy(kernel_profile),
                        "candidate": copy.deepcopy(kernel_profile),
                    }
                ),
            },
            "correctness": {
                "status": "pass",
                "pre_timing_reference": True,
                "pre_timing_candidate": True,
                "post_timing_reference": True,
                "post_timing_candidate": True,
                "fresh_inputs_post_timing": True,
                "graph_validation": mode == "cuda_graph",
                "tolerance": {
                    "dtype_and_shape_exact": True,
                    "rtol": 0.02,
                    "atol": 0.02,
                    "integer": "exact",
                    "deterministic_graph_replay": "exact",
                },
            },
            "provenance": {
                "workload_sha256": canonical_sha256(workload),
                "artifacts": [
                    file_artifact("runner", self.runner),
                    file_artifact("workloads", self.workloads),
                    file_artifact("candidate", self.candidate),
                ],
                "imports": {
                    "python_executable": sys.executable,
                    "python_version": "3.12.0",
                    "modules": [
                        {
                            "module": "torch",
                            "path": str(self.torch),
                            "kind": "python",
                        },
                        {
                            "module": "sglang",
                            "path": str(self.sglang),
                            "kind": "python",
                        },
                        {
                            "module": "deep_gemm",
                            "path": str(self.deep_gemm),
                            "kind": "python",
                        },
                        {
                            "module": "serving_native_candidate",
                            "path": str(self.candidate),
                            "kind": "python",
                        },
                    ],
                    "shared_objects": [str(self.shared_object)],
                },
                "repositories": {
                    "kernel_harness": {
                        "path": str(self.root),
                        "head": "a" * 40,
                        "branch": "fixture",
                        "dirty": False,
                        "status": [],
                    },
                    "sglang": {
                        "path": str(self.root),
                        "head": "b" * 40,
                        "branch": "fixture",
                        "dirty": False,
                        "status": [],
                    },
                },
                "hardware": {
                    "uuid": "GPU-fixture",
                    "driver_version": "999.0",
                    "cuda_runtime_version": "13.0",
                    "clock_samples": [
                        {
                            "uuid": "GPU-fixture",
                            "sm_clock_mhz": 1000,
                            "memory_clock_mhz": 2000,
                        },
                        {
                            "uuid": "GPU-fixture",
                            "sm_clock_mhz": 1000,
                            "memory_clock_mhz": 2000,
                        },
                    ],
                },
                "jit": {
                    "warmup_completed": True,
                    "warmup_activity": self._clean_observation("jit_warmup"),
                    "capture_or_timing_detected": False,
                    "observations": [
                        self._clean_observation(phase)
                        for phase in self._jit_phases(mode)
                    ],
                },
            },
            "implementations": {
                "reference": {
                    "identity": "stock",
                    "call_count": totals["reference_calls"],
                },
                "candidate": {
                    "identity": "identity control" if identity_control else "candidate",
                    "api": candidate_api,
                    "identity_control": identity_control,
                    "declared_fallback": False,
                    "hit_count": totals["candidate_hits"],
                    "fallback_count": totals["candidate_fallbacks"],
                    "reference_delegations": totals[
                        "candidate_reference_delegations"
                    ],
                    "trusted_config_call_count": totals[
                        "candidate_trusted_config_calls"
                    ],
                    "by_phase": by_phase,
                },
            },
            "series": series,
            "reference": latency_summary([1.0] * (SERIES * REPEAT)),
            "candidate": {
                "path": str(self.candidate),
                "api": candidate_api,
                "identity_control": identity_control,
                "series_median_speedups": [1.0] * SERIES,
                **latency_summary([1.0] * (SERIES * REPEAT)),
            },
            "aggregate": {
                "required_series": 3,
                "completed_series": SERIES,
                "threshold": 1.03,
                "every_series_passes_3pct": False,
                "performance_gate_passed": False,
                "identity_control_forced_non_win": identity_control,
            },
        }

    def _valid_w2_candidate_fixture(self, mode: str, task: str) -> dict:
        result = self._valid_fixture(
            mode,
            task=task,
            identity_control=False,
            candidate_api=CALLABLE_API,
        )
        params = result["workload"]["params"]
        token = params["candidate_jit_identity"]
        manifest_sha = "d" * 64
        cache_key = "a" * 32
        kernel_dir = (
            Path(W2_BM16_CACHE_DIR)
            / "cache"
            / f"kernel.{token}.{cache_key}"
        )
        source_path = kernel_dir / "kernel.cu"
        cubin_path = kernel_dir / "kernel.cubin"
        source_sha = "e" * 64
        cubin_sha = "f" * 64
        stock_symbol = (
            "_ZN9deep_gemm28sm100_fp8_fp4_gemm_1d1d_impl"
            "ELj0ELj6144ELj2048ELj128ELj128ELj128E"
        )
        candidate_symbol = (
            "_ZN9deep_gemm28sm100_fp8_fp4_gemm_1d1d_impl"
            "ELj0ELj6144ELj2048ELj16ELj128ELj128E"
        )
        runtime_contract = {
            "base_commit": W2_BM16_BASE_COMMIT,
            "base_version": "0.1.4.post1",
            "cutlass_commit": W2_BM16_CUTLASS_COMMIT,
            "fmt_commit": W2_BM16_FMT_COMMIT,
            "build_id": (
                "glm52-w2-bm16-v2:sgl-deep-gemm-0.1.4.post1@"
                f"{W2_BM16_BASE_COMMIT}:sm100:e32:m1024:k2048:n6144:"
                "bm16:pdl1:sms148:no-recipe:no-overlap"
            ),
            "physical_num_sms": 148,
            "stock_initial_num_sms": 148,
            "candidate_initial_num_sms": 148,
            "stock_num_sms": 148,
            "candidate_num_sms": 148,
            "stock_initial_tc_util": 87,
            "candidate_initial_tc_util": 87,
            "stock_tc_util": 87,
            "candidate_tc_util": 87,
            "stock_initial_pdl": False,
            "candidate_initial_pdl": False,
            "stock_pdl": True,
            "candidate_pdl": True,
            "runtime_modules_distinct": True,
            "runtime_extension_modules_distinct": True,
            "independence_probe_num_sms": 146,
            "independence_probe_tc_util": 88,
            "compute_capability": [10, 0],
            "stock_extension_sha256": W2_BM16_STOCK_EXTENSION_SHA256,
            "candidate_extension_sha256": (
                W2_BM16_CANDIDATE_EXTENSION_SHA256
            ),
            "manifest_sha256": manifest_sha,
            "dg_jit_cache_dir": W2_BM16_CACHE_DIR,
            "sglang_dg_cache_dir": W2_BM16_CACHE_DIR,
            "decode_m": params["decode_m"],
            "expected_m": params["expected_m"],
            "candidate_jit_identity": token,
            "forward_mode": "DECODE",
            "op_tag": "moe_down_proj",
            "reference_opt_level_after_prepare": "0",
            "workload": result["workload"]["name"],
            "setup_environment_restored": {
                "SGLANG_GLM52_OPT": True,
                "SGLANG_GLM52_OPT_PROFILE": True,
                "SGLANG_GLM52_OPT_OPS": True,
                "SGLANG_DEEPGEMM_PDL": True,
            },
            "jit_cache_kernel_name": token,
            "jit_cache_kernel_dir": str(kernel_dir),
            "jit_cache_key": cache_key,
            "jit_cache_source_path": str(source_path),
            "jit_cache_source_sha256": source_sha,
            "jit_cache_cubin_path": str(cubin_path),
            "jit_cache_cubin_sha256": cubin_sha,
            "jit_cache_generated_impl": (
                "sm100_fp8_fp4_gemm_1d1d_impl"
            ),
            "jit_cache_generated_block_m": 16,
            "jit_cache_template_segment": [
                0,
                6144,
                2048,
                16,
                128,
                128,
            ],
        }
        result["implementations"]["candidate"][
            "runtime_contract"
        ] = runtime_contract
        for index, digest in enumerate(
            (
                W2_BM16_SOURCE_PATCH_SHA256,
                W2_BM16_STOCK_EXTENSION_SHA256,
                W2_BM16_CANDIDATE_EXTENSION_SHA256,
                manifest_sha,
                source_sha,
                cubin_sha,
            )
        ):
            artifact_path = (
                source_path
                if digest == source_sha
                else cubin_path
                if digest == cubin_sha
                else self.root / f"artifact-{index}"
            )
            result["provenance"]["artifacts"].append(
                {
                    "role": f"candidate_artifact_{index:02d}",
                    "path": str(artifact_path),
                    "sha256": digest,
                    "size_bytes": 1,
                }
            )

        if mode == "eager":
            reference_profile = result["execution"]["kernel_profiles"][
                "reference"
            ]
            candidate_profile = result["execution"]["kernel_profiles"][
                "candidate"
            ]
            if result["workload"]["family"] == "moe_compute_region":
                common_region_kernels = [
                    "fixture_moe_w13_grouped_kernel",
                    "fixture_moe_swiglu_quant_kernel",
                ]
                reference_names = [
                    common_region_kernels[0],
                    common_region_kernels[1],
                    stock_symbol,
                ]
                candidate_names = [
                    common_region_kernels[0],
                    common_region_kernels[1],
                    candidate_symbol,
                ]
            else:
                reference_names = [stock_symbol]
                candidate_names = [candidate_symbol]
            for profile, names in (
                (reference_profile, reference_names),
                (candidate_profile, candidate_names),
            ):
                profile["events"] = [
                    {
                        "name": name,
                        "duration_us": 1.0,
                        "device_type": "DeviceType.CUDA",
                    }
                    for name in names
                ]
                profile["kernel_identities"] = sorted(set(names))
        else:
            for series in result["series"]:
                for capture in series["graph"]["captures"]:
                    symbol = (
                        candidate_symbol
                        if capture["implementation"] == "candidate"
                        else stock_symbol
                    )
                    capture["nodes"][0]["kernel"] = symbol
                    capture["kernel_identities"] = [symbol]
        edge_cases = []
        for case_index, (name, counts) in enumerate(W2_EDGE_MASK_CASES):
            zero_mask = sum(counts) == 0
            sentinel_elements = (
                32 * 1024 * 6144
                if zero_mask
                else sum(counts) * 6144
            )
            graph_record = None
            if mode == "cuda_graph":
                captures = result["series"][0]["graph"]["captures"]
                reference_capture = copy.deepcopy(captures[0])
                candidate_capture = copy.deepcopy(captures[1])
                for capture_index, capture in enumerate(
                    (reference_capture, candidate_capture)
                ):
                    capture.update(
                        {
                            "capture_id": (
                                f"edge:{case_index:02d}:{name}:"
                                f"{capture['implementation']}"
                            ),
                            "raw_graph_handle": (
                                50_000 + case_index * 10 + capture_index
                            ),
                            "stream_id": (
                                60_000 + case_index * 10 + capture_index
                            ),
                            "input_mutation_replayed": False,
                            "fixed_edge_mask_replayed": True,
                            "zero_full_output_poison_preserved": (
                                True if zero_mask else None
                            ),
                            "sentinel_elements_checked": sentinel_elements,
                            "non_vacuous_sentinel_coverage": True,
                        }
                    )
                clean_observation = {
                    "clean": True,
                    "new_imports": [],
                    "new_shared_objects": [],
                    "cache_changes": [],
                    "candidate_artifact_changes": [],
                }
                graph_record = {
                    "stock_candidate_match": True,
                    "reference_candidate_captured_independently": True,
                    "reference": reference_capture,
                    "candidate": candidate_capture,
                    "capture_observations": {
                        "reference": dict(clean_observation),
                        "candidate": dict(clean_observation),
                    },
                }
            edge_cases.append(
                {
                    "name": name,
                    "masked_m": list(counts),
                    "active_rows": sum(counts),
                    "empty_experts": [
                        index
                        for index, count in enumerate(counts)
                        if count == 0
                    ],
                    "max_count": max(counts),
                    "eager": {
                        "stock_candidate_match": True,
                        "output_poisoned_before_launch": {
                            "reference": True,
                            "candidate": True,
                        },
                        "sentinel_scope": (
                            "entire_output_buffer"
                            if zero_mask
                            else "active_rows"
                        ),
                        "sentinel_elements_checked": sentinel_elements,
                        "non_vacuous_sentinel_coverage": True,
                        "zero_full_output_poison_preserved": (
                            {
                                "reference": True,
                                "candidate": True,
                            }
                            if zero_mask
                            else None
                        ),
                        "masked_m_unmodified": {
                            "reference": True,
                            "candidate": True,
                        },
                        "return_contract": {
                            "reference": "None",
                            "candidate": "None",
                            "enforced_before_TaskResult": True,
                        },
                        "stream": {
                            "default_stream_id": 0,
                            "before": 0,
                            "after_reference": 0,
                            "after_candidate": 0,
                            "unchanged_default_stream": True,
                        },
                    },
                    "graph": graph_record,
                }
            )
        result["correctness"]["edge_masks"] = {
            "status": "pass",
            "scope": "single_B200_leaf_correctness_only",
            "execution_mode": mode,
            "cases": edge_cases,
        }
        if result["workload"]["family"] == "moe_compute_region":
            result["correctness"]["edge_masks"] = None
        return result

    @staticmethod
    def _apply_speedup(result: dict, candidate_ms: float = 0.9) -> None:
        all_reference: list[float] = []
        all_candidate: list[float] = []
        medians: list[float] = []
        for series in result["series"]:
            reference_values: list[float] = []
            candidate_values: list[float] = []
            for sample in series["raw_ordered_samples"]:
                if sample["implementation"] == "reference":
                    sample["latency_ms"] = 1.0
                    reference_values.append(1.0)
                else:
                    sample["latency_ms"] = candidate_ms
                    candidate_values.append(candidate_ms)
            ratios = [
                reference / candidate
                for reference, candidate in zip(
                    reference_values,
                    candidate_values,
                )
            ]
            median = sorted(ratios)[len(ratios) // 2]
            series["reference"] = latency_summary(reference_values)
            series["candidate"] = latency_summary(candidate_values)
            series["paired_speedups"] = ratios
            series["median_speedup"] = median
            series["passes_3pct_gate"] = median >= 1.03
            medians.append(median)
            all_reference.extend(reference_values)
            all_candidate.extend(candidate_values)
        result["reference"] = latency_summary(all_reference)
        result["candidate"].update(latency_summary(all_candidate))
        result["candidate"]["series_median_speedups"] = medians
        every = all(item["passes_3pct_gate"] for item in result["series"])
        result["aggregate"]["every_series_passes_3pct"] = every
        candidate = result["implementations"]["candidate"]
        trusted = candidate["api"] == TRUSTED_CONFIG_API
        result["aggregate"]["performance_gate_passed"] = (
            every
            and not candidate["identity_control"]
            and candidate["fallback_count"] == 0
            and (candidate["reference_delegations"] == 0 or trusted)
        )

    def assert_invalid(self, result: dict, needle: str) -> None:
        report = audit_document(result, verify_files=True)
        self.assertFalse(report["valid"], report)
        self.assertTrue(
            any(needle in error for error in report["errors"]),
            report,
        )

    def test_complete_eager_identity_artifact_is_valid_non_win(self) -> None:
        report = audit_document(self.valid_eager, verify_files=True)
        self.assertTrue(report["valid"], report)
        self.assertFalse(report["performance_gate_passed"])

    def test_complete_graph_identity_artifact_is_valid_non_win(self) -> None:
        report = audit_document(
            self._valid_fixture("cuda_graph"),
            verify_files=True,
        )
        self.assertTrue(report["valid"], report)

    def test_four_w2_identities_support_eager_and_cuda_graph(self) -> None:
        expected = {
            "moe_w2_grouped_decode_m16": (16, 4),
            "moe_w2_grouped_decode_m16_current_source_m5": (16, 5),
            "moe_w2_grouped_decode_m32": (32, 8),
            "moe_w2_grouped_decode_m32_current_source_m9": (32, 9),
        }
        for name, pair in expected.items():
            with self.subTest(task=name):
                workload = WORKLOADS[name]
                self.assertEqual(
                    (
                        workload.params["decode_m"],
                        workload.params["expected_m"],
                    ),
                    pair,
                )
                self.assertEqual(
                    workload.execution_modes,
                    ("eager", "cuda_graph"),
                )
                self.assertTrue(
                    workload.params["candidate_jit_identity"].endswith(
                        f"glm52_w2_bm16_v2_em{pair[1]}"
                    )
                )
                for mode in workload.execution_modes:
                    report = audit_document(
                        self._valid_fixture(mode, task=name),
                        verify_files=True,
                    )
                    self.assertTrue(report["valid"], report)

    def test_w2_leaf_and_graph_identity_evidence_is_audited(self) -> None:
        tasks = (
            "moe_w2_grouped_decode_m16",
            "moe_w2_grouped_decode_m16_current_source_m5",
            "moe_w2_grouped_decode_m32",
            "moe_w2_grouped_decode_m32_current_source_m9",
        )
        for task in tasks:
            for mode in ("eager", "cuda_graph"):
                with self.subTest(task=task, mode=mode):
                    result = self._valid_w2_candidate_fixture(mode, task)
                    report = audit_document(result, verify_files=False)
                    self.assertTrue(report["valid"], report)
        for task in (
            "moe_w13_swiglu_w2_region_decode_m16_current_source_m5",
            "moe_w13_swiglu_w2_region_decode_m32_current_source_m9",
        ):
            with self.subTest(task=task, mode="eager"):
                result = self._valid_w2_candidate_fixture("eager", task)
                report = audit_document(result, verify_files=False)
                self.assertTrue(report["valid"], report)

    def test_w2_runtime_and_observed_identity_tampering_is_rejected(
        self,
    ) -> None:
        cases = (
            (
                "pdl",
                "stock_pdl mismatch",
                lambda value: value["implementations"]["candidate"][
                    "runtime_contract"
                ].update(stock_pdl=False),
            ),
            (
                "tc_util",
                "tc_util state differs",
                lambda value: value["implementations"]["candidate"][
                    "runtime_contract"
                ].update(candidate_tc_util=86),
            ),
            (
                "extension",
                "candidate_extension_sha256 mismatch",
                lambda value: value["implementations"]["candidate"][
                    "runtime_contract"
                ].update(candidate_extension_sha256="0" * 64),
            ),
            (
                "graph_identity",
                "candidate graph must contain BM16 and no W2 BM128",
                lambda value: (
                    value["series"][0]["graph"]["captures"][1]["nodes"][0].update(
                        kernel="stock_kernel"
                    ),
                    value["series"][0]["graph"]["captures"][1].update(
                        kernel_identities=["stock_kernel"]
                    ),
                ),
            ),
        )
        for name, needle, mutate in cases:
            with self.subTest(name=name):
                result = self._valid_w2_candidate_fixture(
                    "cuda_graph",
                    "moe_w2_grouped_decode_m16",
                )
                mutate(result)
                report = audit_document(result, verify_files=False)
                self.assertFalse(report["valid"], report)
                self.assertTrue(
                    any(needle in error for error in report["errors"]),
                    report,
                )

    def test_w2_candidate_double_launch_symbols_are_rejected(self) -> None:
        eager = self._valid_w2_candidate_fixture(
            "eager",
            "moe_w2_grouped_decode_m16",
        )
        eager_profiles = eager["execution"]["kernel_profiles"]
        stock_symbol = eager_profiles["reference"]["kernel_identities"][0]
        candidate_profile = eager_profiles["candidate"]
        candidate_profile["events"].append(
            {
                "name": stock_symbol,
                "duration_us": 1.0,
                "device_type": "DeviceType.CUDA",
            }
        )
        candidate_profile["kernel_identities"] = sorted(
            {
                event["name"]
                for event in candidate_profile["events"]
            }
        )
        eager_report = audit_document(eager, verify_files=False)
        self.assertFalse(eager_report["valid"], eager_report)
        self.assertTrue(
            any(
                "leaf candidate eager profile must contain exactly one CUDA event"
                in error
                for error in eager_report["errors"]
            ),
            eager_report,
        )

        graph = self._valid_w2_candidate_fixture(
            "cuda_graph",
            "moe_w2_grouped_decode_m16",
        )
        series_capture = graph["series"][0]["graph"]["captures"][1]
        stock_symbol = graph["series"][0]["graph"]["captures"][0][
            "kernel_identities"
        ][0]
        self._append_graph_kernel(series_capture, stock_symbol)
        graph_report = audit_document(graph, verify_files=False)
        self.assertFalse(graph_report["valid"], graph_report)
        self.assertTrue(
            any(
                "must contain exactly one CUDA KERNEL node"
                in error
                for error in graph_report["errors"]
            ),
            graph_report,
        )

        edge_graph = self._valid_w2_candidate_fixture(
            "cuda_graph",
            "moe_w2_grouped_decode_m16",
        )
        edge_case = edge_graph["correctness"]["edge_masks"]["cases"][0]
        stock_symbol = edge_case["graph"]["reference"][
            "kernel_identities"
        ][0]
        self._append_graph_kernel(
            edge_case["graph"]["candidate"],
            stock_symbol,
        )
        edge_report = audit_document(edge_graph, verify_files=False)
        self.assertFalse(edge_report["valid"], edge_report)
        self.assertTrue(
            any(
                "must contain exactly one CUDA KERNEL node"
                in error
                for error in edge_report["errors"]
            ),
            edge_report,
        )

    def test_w2_leaf_unrelated_adapter_kernels_are_rejected(self) -> None:
        eager = self._valid_w2_candidate_fixture(
            "eager",
            "moe_w2_grouped_decode_m16",
        )
        candidate_profile = eager["execution"]["kernel_profiles"][
            "candidate"
        ]
        candidate_profile["events"].append(
            {
                "name": "fixture_unrelated_adapter_kernel",
                "duration_us": 1.0,
                "device_type": "DeviceType.CUDA",
            }
        )
        candidate_profile["kernel_identities"] = sorted(
            {event["name"] for event in candidate_profile["events"]}
        )
        self.assert_invalid(
            eager,
            "leaf candidate eager profile must contain exactly one CUDA event",
        )

        main_graph = self._valid_w2_candidate_fixture(
            "cuda_graph",
            "moe_w2_grouped_decode_m16",
        )
        main_capture = main_graph["series"][0]["graph"]["captures"][1]
        self._append_graph_kernel(
            main_capture,
            "fixture_unrelated_adapter_kernel",
        )
        self.assert_invalid(
            main_graph,
            "must contain exactly one CUDA KERNEL node",
        )

        edge_graph = self._valid_w2_candidate_fixture(
            "cuda_graph",
            "moe_w2_grouped_decode_m16",
        )
        edge_capture = edge_graph["correctness"]["edge_masks"]["cases"][1][
            "graph"
        ]["candidate"]
        self._append_graph_kernel(
            edge_capture,
            "fixture_unrelated_adapter_kernel",
        )
        self.assert_invalid(
            edge_graph,
            "must contain exactly one CUDA KERNEL node",
        )

    def test_w2_edge_graph_raw_nodes_and_binding_are_fail_closed(
        self,
    ) -> None:
        def candidate_capture(result: dict) -> dict:
            return result["correctness"]["edge_masks"]["cases"][0][
                "graph"
            ]["candidate"]

        raw_bm128 = self._valid_w2_candidate_fixture(
            "cuda_graph",
            "moe_w2_grouped_decode_m16",
        )
        stock_symbol = raw_bm128["correctness"]["edge_masks"]["cases"][0][
            "graph"
        ]["reference"]["nodes"][0]["kernel"]
        # Keep the forged BM16 summary while the raw node is actually BM128.
        candidate_capture(raw_bm128)["nodes"][0]["kernel"] = stock_symbol
        self.assert_invalid(
            raw_bm128,
            "kernel_identities do not match raw nodes",
        )

        duplicate_bm16 = self._valid_w2_candidate_fixture(
            "cuda_graph",
            "moe_w2_grouped_decode_m16",
        )
        duplicate_capture = candidate_capture(duplicate_bm16)
        duplicate_node = copy.deepcopy(duplicate_capture["nodes"][0])
        duplicate_node["index"] = 1
        duplicate_capture["nodes"].append(duplicate_node)
        # Deliberately retain the forged one-node summaries.
        self.assert_invalid(
            duplicate_bm16,
            "must contain exactly one CUDA KERNEL node",
        )

        forged_metadata = self._valid_w2_candidate_fixture(
            "cuda_graph",
            "moe_w2_grouped_decode_m16",
        )
        candidate_capture(forged_metadata)["node_type_counts"] = {}
        self.assert_invalid(
            forged_metadata,
            "node_type_counts do not match raw nodes",
        )

        for field, value, needle in (
            ("capture_id", "edge:forged:candidate", "capture_id mismatch"),
            ("raw_graph_handle", 0, "raw_graph_handle invalid"),
            ("stream_id", 0, "stream IDs invalid"),
        ):
            with self.subTest(field=field):
                tampered = self._valid_w2_candidate_fixture(
                    "cuda_graph",
                    "moe_w2_grouped_decode_m16",
                )
                candidate_capture(tampered)[field] = value
                self.assert_invalid(tampered, needle)

    def test_w2_main_and_edge_graph_integer_metadata_rejects_bools(
        self,
    ) -> None:
        def main_capture(result: dict) -> dict:
            return result["series"][0]["graph"]["captures"][1]

        def edge_capture(result: dict) -> dict:
            return result["correctness"]["edge_masks"]["cases"][0][
                "graph"
            ]["candidate"]

        for scope, get_capture in (
            ("main", main_capture),
            ("edge", edge_capture),
        ):
            for field, mutate, needle in (
                (
                    "node_count",
                    lambda capture: capture.update(node_count=True),
                    "node_count is not a strict integer",
                ),
                (
                    "node_index",
                    lambda capture: capture["nodes"][0].update(index=False),
                    "index is not a strict integer",
                ),
                (
                    "node_type_count",
                    lambda capture: capture.update(
                        node_type_counts={
                            "CU_GRAPH_NODE_TYPE_KERNEL": True
                        }
                    ),
                    "node_type_counts values are not strict integers",
                ),
            ):
                with self.subTest(scope=scope, field=field):
                    result = self._valid_w2_candidate_fixture(
                        "cuda_graph",
                        "moe_w2_grouped_decode_m16",
                    )
                    mutate(get_capture(result))
                    self.assert_invalid(result, needle)

    def test_w2_edge_eager_stream_ids_reject_bools(self) -> None:
        result = self._valid_w2_candidate_fixture(
            "eager",
            "moe_w2_grouped_decode_m16",
        )
        stream = result["correctness"]["edge_masks"]["cases"][0]["eager"][
            "stream"
        ]
        for field in (
            "default_stream_id",
            "before",
            "after_reference",
            "after_candidate",
        ):
            stream[field] = False
        stream["unchanged_default_stream"] = True
        self.assert_invalid(result, "eager stream contract mismatch")

    def test_w2_edge_phase_counts_are_leaf_scoped_and_exact(self) -> None:
        leaf = self._valid_w2_candidate_fixture(
            "cuda_graph",
            "moe_w2_grouped_decode_m16",
        )
        phases = leaf["implementations"]["candidate"]["by_phase"]
        self.assertEqual(
            phases["edge:zero_all_experts:eager"]["reference_calls"],
            1,
        )
        self.assertEqual(
            phases[
                "edge:00:zero_all_experts:reference:warmup"
            ]["reference_calls"],
            3,
        )
        self.assertEqual(
            phases[
                "edge:00:zero_all_experts:candidate:capture"
            ]["candidate_hits"],
            1,
        )
        self.assertEqual(
            phases[
                "edge:00:zero_all_experts:graph_validation"
            ],
            {
                "reference_calls": 2,
                "candidate_hits": 2,
                "candidate_fallbacks": 0,
                "candidate_reference_delegations": 0,
                "candidate_trusted_config_calls": 0,
            },
        )
        self.assertTrue(
            audit_document(leaf, verify_files=False)["valid"],
        )

        wrong_count = copy.deepcopy(leaf)
        wrong_count["implementations"]["candidate"]["by_phase"][
            "edge:00:zero_all_experts:graph_validation"
        ]["candidate_hits"] = 3
        self.assert_invalid(wrong_count, "runner path count")

        missing_phase = copy.deepcopy(leaf)
        missing_phase["implementations"]["candidate"]["by_phase"].pop(
            "edge:00:zero_all_experts:reference:warmup"
        )
        self.assert_invalid(missing_phase, "phase set does not close")

        region = self._valid_w2_candidate_fixture(
            "eager",
            "moe_w13_swiglu_w2_region_decode_m16_current_source_m5",
        )
        region_phases = region["implementations"]["candidate"]["by_phase"]
        self.assertFalse(
            any(phase.startswith("edge:") for phase in region_phases)
        )
        self.assertTrue(
            audit_document(region, verify_files=False)["valid"],
        )
        region_phases["edge:zero_all_experts:eager"] = {
            field: 0 for field in COUNTER_FIELDS
        }
        self.assert_invalid(region, "phase set does not close")

    def test_w2_containing_region_kernel_sequence_is_exact(self) -> None:
        task = (
            "moe_w13_swiglu_w2_region_decode_m16_current_source_m5"
        )
        baseline = self._valid_w2_candidate_fixture("eager", task)
        self.assertTrue(
            audit_document(baseline, verify_files=False)["valid"],
        )

        extra = copy.deepcopy(baseline)
        extra_profile = extra["execution"]["kernel_profiles"]["candidate"]
        extra_profile["events"].append(
            {
                "name": "fixture_unrelated_adapter_kernel",
                "duration_us": 1.0,
                "device_type": "DeviceType.CUDA",
            }
        )
        extra_profile["kernel_identities"] = sorted(
            {event["name"] for event in extra_profile["events"]}
        )
        self.assert_invalid(extra, "equal non-leaf CUDA event counts")

        reordered = copy.deepcopy(baseline)
        reordered_events = reordered["execution"]["kernel_profiles"][
            "candidate"
        ]["events"]
        reordered_events[0], reordered_events[1] = (
            reordered_events[1],
            reordered_events[0],
        )
        self.assert_invalid(reordered, "non-W2 CUDA event order differs")

        substituted = copy.deepcopy(baseline)
        substituted_profile = substituted["execution"]["kernel_profiles"][
            "candidate"
        ]
        substituted_profile["events"][0]["name"] = (
            "fixture_different_non_w2_kernel"
        )
        substituted_profile["kernel_identities"] = sorted(
            {event["name"] for event in substituted_profile["events"]}
        )
        report = audit_document(substituted, verify_files=False)
        self.assertFalse(report["valid"], report)
        self.assertTrue(
            any(
                "non-W2 CUDA event order differs" in error
                or "non-W2 CUDA event multiset differs" in error
                for error in report["errors"]
            ),
            report,
        )

    def test_w2_exact_cache_and_zero_mask_sentinel_are_fail_closed(
        self,
    ) -> None:
        cache_tamper = self._valid_w2_candidate_fixture(
            "eager",
            "moe_w2_grouped_decode_m16",
        )
        cache_tamper["implementations"]["candidate"][
            "runtime_contract"
        ]["jit_cache_kernel_dir"] = "/tmp/forged"
        report = audit_document(cache_tamper, verify_files=False)
        self.assertFalse(report["valid"], report)
        self.assertTrue(
            any("emX JIT cache directory" in error for error in report["errors"]),
            report,
        )

        block_tamper = self._valid_w2_candidate_fixture(
            "eager",
            "moe_w2_grouped_decode_m16",
        )
        block_tamper["implementations"]["candidate"][
            "runtime_contract"
        ]["jit_cache_generated_block_m"] = 128
        report = audit_document(block_tamper, verify_files=False)
        self.assertFalse(report["valid"], report)
        self.assertTrue(
            any("BM16 template mapping" in error for error in report["errors"]),
            report,
        )

        sentinel_tamper = self._valid_w2_candidate_fixture(
            "cuda_graph",
            "moe_w2_grouped_decode_m16",
        )
        zero_case = sentinel_tamper["correctness"]["edge_masks"]["cases"][0]
        zero_case["eager"]["zero_full_output_poison_preserved"][
            "candidate"
        ] = False
        zero_case["graph"]["candidate"][
            "zero_full_output_poison_preserved"
        ] = False
        report = audit_document(sentinel_tamper, verify_files=False)
        self.assertFalse(report["valid"], report)
        self.assertTrue(
            any(
                "zero-mask full output did not remain poisoned" in error
                or "sentinel coverage failed" in error
                for error in report["errors"]
            ),
            report,
        )

    def test_w2_candidate_routes_containing_region_through_bound_w2(
        self,
    ) -> None:
        candidate_path = (
            HERE / "candidates" / "moe_w2_bm16.py"
        )
        module = _load_candidate(str(candidate_path))
        bound_w2 = object()
        inputs = {"region": object()}
        expected = TaskResult(object())
        runtime = SimpleNamespace(
            workload=SimpleNamespace(family="moe_compute_region"),
            run_moe_compute_region=Mock(return_value=expected),
        )
        with patch.object(
            module,
            "_candidate_callable",
            return_value=bound_w2,
        ):
            observed = module.run(inputs, runtime)
        self.assertIs(observed, expected)
        runtime.run_moe_compute_region.assert_called_once_with(
            inputs,
            w2_gemm=bound_w2,
        )

    def test_w2_candidate_binds_exact_em_cache_source_and_cubin(
        self,
    ) -> None:
        candidate_path = HERE / "candidates" / "moe_w2_bm16.py"
        module = _load_candidate(str(candidate_path))
        token = (
            "sm100_m_grouped_fp8_fp4_gemm_masked_1d1d_"
            "glm52_w2_bm16_v2_em4"
        )
        cache_root = self.root / "deepgemm"
        kernel_dir = (
            cache_root
            / "cache"
            / f"kernel.{token}.{'a' * 32}"
        )
        kernel_dir.mkdir(parents=True)
        source = kernel_dir / "kernel.cu"
        source.write_text(
            "sm100_fp8_fp4_gemm_1d1d_impl<"
            "cute::UMMA::Major::K,cute::UMMA::Major::K,128,128,128,"
            "0,6144,2048,16,128,128,32>;"
        )
        cubin = kernel_dir / "kernel.cubin"
        cubin.write_bytes(b"fixture-cubin")
        module._STATE.clear()
        module._STATE["evidence"] = {
            "candidate_jit_identity": token,
            "dg_jit_cache_dir": str(cache_root),
        }
        try:
            with patch.object(module, "_TASK_DG_CACHE", cache_root):
                runtime_paths = module.runtime_artifact_paths()
                evidence = module.runtime_evidence()
        finally:
            module._STATE.clear()
        self.assertEqual(runtime_paths, (str(source), str(cubin)))
        self.assertEqual(evidence["jit_cache_kernel_name"], token)
        self.assertEqual(evidence["jit_cache_generated_block_m"], 16)
        self.assertEqual(
            evidence["jit_cache_template_segment"],
            [0, 6144, 2048, 16, 128, 128],
        )
        self.assertEqual(len(evidence["jit_cache_cubin_sha256"]), 64)

    def test_w2_candidate_restores_every_setup_environment_variable(
        self,
    ) -> None:
        candidate_path = HERE / "candidates" / "moe_w2_bm16.py"
        module = _load_candidate(str(candidate_path))
        names = (
            "SGLANG_GLM52_OPT",
            "SGLANG_GLM52_OPT_PROFILE",
            "SGLANG_GLM52_OPT_OPS",
            "SGLANG_DEEPGEMM_PDL",
        )
        setup = {
            "SGLANG_GLM52_OPT": "1",
            "SGLANG_GLM52_OPT_PROFILE": "moe_w2_bm16",
            "SGLANG_GLM52_OPT_OPS": "moe_down_proj",
            "SGLANG_DEEPGEMM_PDL": "1",
        }
        saved_process_environment = dict(os.environ)
        with patch.dict(os.environ, saved_process_environment, clear=True):
            os.environ["SGLANG_GLM52_OPT"] = "legacy-opt"
            for name in names[1:]:
                os.environ.pop(name, None)
            before = {name: os.environ.get(name) for name in names}
            with module._temporary_environment(setup) as observed:
                self.assertEqual(observed, before)
                self.assertEqual(
                    {name: os.environ.get(name) for name in names},
                    setup,
                )
            self.assertEqual(
                {name: os.environ.get(name) for name in names},
                before,
            )

    def test_packed_weight_scale_marker_matches_production_loader(self) -> None:
        int32 = object()
        runtime = object.__new__(Runtime)
        runtime.torch = SimpleNamespace(int32=int32)
        scale = SimpleNamespace(dtype=int32)
        observed = runtime._mark_packed_ue8m0_weight_scale(scale)
        self.assertIs(observed, scale)
        self.assertIs(scale.format_ue8m0, True)

        with self.assertRaisesRegex(RuntimeError, "not packed int32 UE8M0"):
            runtime._mark_packed_ue8m0_weight_scale(
                SimpleNamespace(dtype=object())
            )

    def test_grouped_reference_uses_precomputed_views_without_mask_host_read(
        self,
    ) -> None:
        source = inspect.getsource(Runtime.reference)
        self.assertNotIn("masked_m\"].tolist", source)
        self.assertNotIn(
            ".tolist",
            inspect.getsource(Runtime.run_moe_compute_region),
        )

        called = Mock(return_value=None)
        modules: dict[str, types.ModuleType] = {}
        names = (
            "sglang",
            "sglang.srt",
            "sglang.srt.layers",
            "sglang.srt.layers.deep_gemm_wrapper",
            "sglang.srt.layers.deep_gemm_wrapper.entrypoint",
        )
        for name in names:
            module = types.ModuleType(name)
            module.__path__ = []
            modules[name] = module
        modules[names[-1]].grouped_gemm_nt_f8f8bf16_masked = called

        masked_m = SimpleNamespace(
            tolist=Mock(side_effect=AssertionError("host mask read"))
        )
        observed = (object(), object())
        inputs = {
            "activation_fp8": object(),
            "activation_scale": object(),
            "weight_fp8": object(),
            "weight_scale": object(),
            "out": object(),
            "masked_m": masked_m,
            "expected_m": 4,
            "observed_out": observed,
        }
        runtime = object.__new__(Runtime)
        runtime._inside_candidate = False
        runtime.accounting = SimpleNamespace(reference=Mock())
        runtime.workload = get_workload("moe_w2_grouped_decode_m16")
        with patch.dict(sys.modules, modules):
            result = Runtime.reference(runtime, inputs)
        self.assertIsInstance(result, TaskResult)
        self.assertIs(result.observed, observed)
        masked_m.tolist.assert_not_called()
        called.assert_called_once()

    def test_graph_mutation_target_is_scoped_to_two_families(self) -> None:
        packed = object()
        grouped = object()
        runtime = SimpleNamespace(
            workload=SimpleNamespace(family="packed_fp8_gemm")
        )
        self.assertIs(
            _graph_mutation_target(runtime, {"x_fp8": packed}),
            packed,
        )
        runtime.workload.family = "moe_grouped_masked"
        self.assertIs(
            _graph_mutation_target(
                runtime, {"activation_fp8": grouped}
            ),
            grouped,
        )
        for family in ("bf16_linear", "moe_swiglu_quant", "allgather"):
            with self.subTest(family=family):
                runtime.workload.family = family
                with self.assertRaisesRegex(
                    RuntimeError,
                    "supports only packed_fp8_gemm and moe_grouped_masked",
                ):
                    _graph_mutation_target(
                        runtime,
                        {
                            "x_fp8": packed,
                            "activation_fp8": grouped,
                        },
                    )

    def test_runner_owned_config_api_can_carry_a_valid_win(self) -> None:
        result = self._valid_fixture(
            "eager",
            task="deepep_normal_dispatch_prefill",
            identity_control=False,
            candidate_api=TRUSTED_CONFIG_API,
        )
        self._apply_speedup(result)
        report = audit_document(result, verify_files=True)
        self.assertTrue(report["valid"], report)
        self.assertTrue(report["performance_gate_passed"])

    def test_non_filesystem_pseudo_module_is_not_recorded(self) -> None:
        name = "serving_native_test_pseudo_module"
        module = types.ModuleType(name)
        module.__file__ = str(self.root / "does-not-exist.py")
        sys.modules[name] = module
        try:
            self.assertNotIn(name, module_path_snapshot())
        finally:
            sys.modules.pop(name, None)

    def test_candidate_loader_preserves_sys_modules_provenance(self) -> None:
        module = _load_candidate(str(self.candidate))
        try:
            self.assertIs(sys.modules.get("serving_native_candidate"), module)
            self.assertEqual(module.__candidate_api__, CALLABLE_API)
        finally:
            sys.modules.pop("serving_native_candidate", None)

    def test_candidate_cannot_self_authorize_reference_delegation(self) -> None:
        path = self._file(
            "old_escape.py",
            "REFERENCE_DELEGATION_IS_CANDIDATE = True\n"
            "def run(inputs, runtime):\n"
            "    return runtime.reference(inputs)\n",
        )
        module = _load_candidate(str(path))
        try:
            self.assertEqual(module.__candidate_api__, CALLABLE_API)
            self.assertFalse(
                hasattr(module, "__trusted_reference_delegation__")
            )
        finally:
            sys.modules.pop("serving_native_candidate", None)

    def test_runner_owned_config_candidate_is_declarative(self) -> None:
        path = HERE / "candidates" / "deepep_config.py"
        module = _load_candidate(str(path))
        try:
            self.assertEqual(module.__candidate_api__, TRUSTED_CONFIG_API)
            self.assertFalse(callable(getattr(module, "run", None)))
            self.assertIsInstance(module.CANDIDATE_CONFIG, dict)
        finally:
            sys.modules.pop("serving_native_candidate", None)

    def test_wrong_artifact_hash_is_rejected(self) -> None:
        result = copy.deepcopy(self.valid_eager)
        result["provenance"]["artifacts"][2]["sha256"] = "0" * 64
        self.assert_invalid(result, "hash mismatch")

    def test_noncanonical_artifact_path_is_rejected(self) -> None:
        result = copy.deepcopy(self.valid_eager)
        fake_runner = self._file("forged_runner.py", self.runner.read_text())
        result["provenance"]["artifacts"][0] = file_artifact(
            "runner",
            fake_runner,
        )
        self.assert_invalid(result, "not the canonical serving-native source")

    def test_self_consistent_but_noncanonical_workload_is_rejected(self) -> None:
        result = copy.deepcopy(self.valid_eager)
        result["workload"]["params"]["m"] = 17
        result["provenance"]["workload_sha256"] = canonical_sha256(
            result["workload"]
        )
        self.assert_invalid(result, "canonical WORKLOADS registry")

    def test_unknown_workload_is_rejected(self) -> None:
        result = copy.deepcopy(self.valid_eager)
        result["workload"]["name"] = "forged_workload"
        result["provenance"]["workload_sha256"] = canonical_sha256(
            result["workload"]
        )
        self.assert_invalid(result, "absent from the canonical WORKLOADS")

    def test_candidate_hit_zero_is_rejected(self) -> None:
        result = copy.deepcopy(self.valid_eager)
        result["implementations"]["candidate"]["hit_count"] = 0
        self.assert_invalid(result, "hit count")

    def test_silent_fallback_is_rejected_with_closed_totals(self) -> None:
        result = self._valid_fixture(
            "eager",
            identity_control=False,
            candidate_api=CALLABLE_API,
        )
        phase = result["implementations"]["candidate"]["by_phase"][
            "pre_timing_correctness"
        ]
        phase["candidate_fallbacks"] = 1
        phase["candidate_reference_delegations"] = 1
        candidate = result["implementations"]["candidate"]
        candidate["fallback_count"] = 1
        candidate["reference_delegations"] = 1
        self.assert_invalid(result, "silent candidate fallback")

    def test_old_candidate_controlled_delegation_escape_cannot_claim_win(self) -> None:
        result = self._valid_fixture(
            "eager",
            identity_control=False,
            candidate_api=CALLABLE_API,
        )
        self._apply_speedup(result)
        phase = result["implementations"]["candidate"]["by_phase"][
            "pre_timing_correctness"
        ]
        phase["candidate_reference_delegations"] = 1
        candidate = result["implementations"]["candidate"]
        candidate["reference_delegations"] = 1
        candidate["REFERENCE_DELEGATION_IS_CANDIDATE"] = True
        report = audit_document(result, verify_files=True)
        self.assertFalse(report["valid"], report)
        self.assertFalse(report["performance_gate_passed"], report)
        self.assertTrue(
            any(
                "delegation/fallback counts do not close" in error
                for error in report["errors"]
            ),
            report,
        )

    def test_trusted_config_api_is_rejected_outside_owned_workloads(self) -> None:
        result = self._valid_fixture(
            "eager",
            identity_control=False,
            candidate_api=TRUSTED_CONFIG_API,
        )
        self.assert_invalid(result, "outside DeepEP normal mode")

    def test_execution_mode_mismatch_is_rejected(self) -> None:
        result = copy.deepcopy(self.valid_eager)
        result["series"][1]["execution_mode"] = "cuda_graph"
        self.assert_invalid(result, "execution-mode mismatch")

    def test_series_and_repeat_counts_close_exactly(self) -> None:
        cases = (
            ("requested", "requested/raw series counts", lambda value: value["run"].update(requested_series=4)),
            ("completed", "requested/completed/raw", lambda value: value["aggregate"].update(completed_series=2)),
            ("repeat", "repeat does not close", lambda value: value["series"][0].update(repeat=3)),
            ("warmup", "warmup_pairs does not close", lambda value: value["series"][0].update(warmup_pairs=4)),
            ("raw", "raw ordered samples incomplete", lambda value: value["series"][0]["raw_ordered_samples"].pop()),
        )
        for name, needle, mutate in cases:
            with self.subTest(name=name):
                result = copy.deepcopy(self.valid_eager)
                mutate(result)
                self.assert_invalid(result, needle)

    def test_call_totals_and_by_phase_counts_close_exactly(self) -> None:
        cases = (
            (
                "reference_total",
                "reference call count does not close",
                lambda value: value["implementations"]["reference"].update(
                    call_count=value["implementations"]["reference"]["call_count"] + 1
                ),
            ),
            (
                "candidate_total",
                "candidate hit count does not close",
                lambda value: value["implementations"]["candidate"].update(
                    hit_count=value["implementations"]["candidate"]["hit_count"] + 1
                ),
            ),
            (
                "phase_count",
                "runner path count",
                lambda value: value["implementations"]["candidate"]["by_phase"][
                    "pre_timing_correctness"
                ].update(reference_calls=2),
            ),
            (
                "missing_phase",
                "phase set does not close",
                lambda value: value["implementations"]["candidate"]["by_phase"].pop(
                    "profiler_candidate"
                ),
            ),
            (
                "extra_counter",
                "counter fields do not close",
                lambda value: value["implementations"]["candidate"]["by_phase"][
                    "profiler_candidate"
                ].update(forged=1),
            ),
        )
        for name, needle, mutate in cases:
            with self.subTest(name=name):
                result = copy.deepcopy(self.valid_eager)
                mutate(result)
                self.assert_invalid(result, needle)

    def test_jit_during_timing_is_rejected(self) -> None:
        result = copy.deepcopy(self.valid_eager)
        jit = result["provenance"]["jit"]
        jit["capture_or_timing_detected"] = True
        jit["observations"][0]["clean"] = False
        jit["observations"][0]["cache_changes"] = [{"path": "late.cubin"}]
        self.assert_invalid(result, "JIT or import/artifact activity")

    def test_missing_correctness_and_provenance_are_rejected(self) -> None:
        for field, needle in (
            ("correctness", "missing correctness"),
            ("provenance", "missing provenance"),
        ):
            with self.subTest(field=field):
                result = copy.deepcopy(self.valid_eager)
                del result[field]
                self.assert_invalid(result, needle)

    def test_identity_cannot_claim_a_performance_pass(self) -> None:
        result = copy.deepcopy(self.valid_eager)
        result["aggregate"]["performance_gate_passed"] = True
        self.assert_invalid(result, "identity A/B must not pass")

    def test_raw_ordering_and_summary_are_fail_closed(self) -> None:
        cases = (
            (
                "ordering",
                "breaks AB/BA ordering",
                lambda value: value["series"][0]["raw_ordered_samples"][0].update(
                    implementation="candidate"
                ),
            ),
            (
                "series_summary",
                "does not match raw samples",
                lambda value: value["series"][0]["reference"].update(median_ms=2.0),
            ),
            (
                "top_summary",
                "does not match raw samples",
                lambda value: value["reference"].update(p95_ms=2.0),
            ),
        )
        for name, needle, mutate in cases:
            with self.subTest(name=name):
                result = copy.deepcopy(self.valid_eager)
                mutate(result)
                self.assert_invalid(result, needle)

    def test_workload_hash_is_fail_closed(self) -> None:
        result = copy.deepcopy(self.valid_eager)
        result["provenance"]["workload_sha256"] = "f" * 64
        self.assert_invalid(result, "workload hash mismatch")

    def test_graph_semantic_failures_are_rejected(self) -> None:
        cases = {
            "stable_input_pointers": "stable_input_pointers",
            "stable_output_pointers": "stable_output_pointers",
            "input_mutation_replayed": "input_mutation_replayed",
            "output_poison_replayed": "output_poison_replayed",
            "deterministic_replay": "deterministic_replay",
            "approved_tolerance_passed": "approved_tolerance_passed",
        }
        for field, needle in cases.items():
            with self.subTest(field=field):
                result = self._valid_fixture("cuda_graph")
                result["series"][0]["graph"]["captures"][0][field] = False
                self.assert_invalid(result, needle)

    def test_graph_capture_ids_are_bound_to_independent_round_robin_captures(self) -> None:
        mutations = (
            (
                "missing_sample_id",
                "graph_capture_id",
                lambda value: value["series"][0]["raw_ordered_samples"][0].pop(
                    "graph_capture_id"
                ),
            ),
            (
                "unknown_sample_id",
                "graph_capture_id",
                lambda value: value["series"][0]["raw_ordered_samples"][0].update(
                    graph_capture_id="forged"
                ),
            ),
            (
                "wrong_pool",
                "graph_capture_id",
                lambda value: value["series"][0]["raw_ordered_samples"][0].update(
                    graph_capture_id=f"{RUN_ID}:series-01:C-after-R"
                ),
            ),
            (
                "duplicate_handle",
                "reuses a CUDA graph handle",
                lambda value: value["series"][0]["graph"]["captures"][1].update(
                    raw_graph_handle=value["series"][0]["graph"]["captures"][0][
                        "raw_graph_handle"
                    ]
                ),
            ),
            (
                "duplicate_stream",
                "reuses a capture stream",
                lambda value: value["series"][0]["graph"]["captures"][1].update(
                    stream_id=value["series"][0]["graph"]["captures"][0][
                        "stream_id"
                    ]
                ),
            ),
            (
                "capture_id",
                "capture_id is not bound",
                lambda value: value["series"][0]["graph"]["captures"][1].update(
                    capture_id="forged"
                ),
            ),
        )
        for name, needle, mutate in mutations:
            with self.subTest(name=name):
                result = self._valid_fixture("cuda_graph")
                mutate(result)
                self.assert_invalid(result, needle)

    def test_graph_metadata_is_recomputed_from_nodes_and_ids(self) -> None:
        mutations = (
            (
                "node_count",
                "node_count does not match",
                lambda capture: capture.update(node_count=2),
            ),
            (
                "type_counts",
                "node_type_counts do not match",
                lambda capture: capture.update(node_type_counts={"forged": 1}),
            ),
            (
                "kernel_ids",
                "kernel_identities do not match",
                lambda capture: capture.update(kernel_identities=["forged"]),
            ),
            (
                "forbidden",
                "forbidden_nodes do not match",
                lambda capture: capture.update(
                    forbidden_nodes=[{"type": "CU_GRAPH_NODE_TYPE_MEMCPY"}]
                ),
            ),
            (
                "non_default_truth",
                "non_default_stream does not match",
                lambda capture: capture.update(
                    stream_id=capture["default_stream_id"],
                    non_default_stream=True,
                ),
            ),
        )
        for name, needle, mutate in mutations:
            with self.subTest(name=name):
                result = self._valid_fixture("cuda_graph")
                mutate(result["series"][0]["graph"]["captures"][0])
                self.assert_invalid(result, needle)

    def test_generic_graph_integer_metadata_rejects_bools(self) -> None:
        for field, mutate, needle in (
            (
                "node_count",
                lambda capture: capture.update(node_count=True),
                "node_count is not a strict integer",
            ),
            (
                "node_index",
                lambda capture: capture["nodes"][0].update(index=False),
                "index is not a strict integer",
            ),
            (
                "node_type_count",
                lambda capture: capture.update(
                    node_type_counts={"CU_GRAPH_NODE_TYPE_KERNEL": True}
                ),
                "node_type_counts values are not strict integers",
            ),
        ):
            with self.subTest(field=field):
                result = self._valid_fixture("cuda_graph")
                capture = result["series"][0]["graph"]["captures"][0]
                mutate(capture)
                self.assert_invalid(result, needle)

    def test_graph_copy_and_adapter_nodes_are_rejected_after_recomputation(self) -> None:
        for name, node in (
            (
                "copy_node",
                {"index": 1, "type": "CU_GRAPH_NODE_TYPE_MEMCPY"},
            ),
            (
                "adapter_kernel",
                {
                    "index": 1,
                    "type": "CU_GRAPH_NODE_TYPE_KERNEL",
                    "kernel": "hidden_adapter_kernel",
                    "grid": [1, 1, 1],
                    "block": [32, 1, 1],
                    "shared_memory_bytes": 0,
                },
            ),
        ):
            with self.subTest(name=name):
                result = self._valid_fixture("cuda_graph")
                capture = result["series"][0]["graph"]["captures"][1]
                capture["nodes"].append(node)
                capture["node_count"] = 2
                capture["node_type_counts"] = (
                    {
                        "CU_GRAPH_NODE_TYPE_KERNEL": 1,
                        "CU_GRAPH_NODE_TYPE_MEMCPY": 1,
                    }
                    if name == "copy_node"
                    else {"CU_GRAPH_NODE_TYPE_KERNEL": 2}
                )
                capture["kernel_identities"] = [
                    item["kernel"]
                    for item in capture["nodes"]
                    if "kernel" in item
                ]
                capture["forbidden_nodes"] = []
                self.assert_invalid(result, "has forbidden graph nodes")

    def test_eager_kernel_identities_are_recomputed_from_events(self) -> None:
        result = copy.deepcopy(self.valid_eager)
        result["execution"]["kernel_profiles"]["candidate"][
            "kernel_identities"
        ] = ["forged"]
        self.assert_invalid(result, "do not match profiler events")

    def test_candidate_result_path_must_match_hashed_artifact(self) -> None:
        result = copy.deepcopy(self.valid_eager)
        result["candidate"]["path"] = str(self.root / "forged.py")
        self.assert_invalid(result, "path does not match hashed artifact")


if __name__ == "__main__":
    unittest.main()
