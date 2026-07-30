"""GPU-free adversarial regression tests for the serving-native V2 auditor."""

from __future__ import annotations

import copy
import json
import statistics
import sys
import tempfile
import types
import unittest
from pathlib import Path

from serving_native.audit_result import audit_document
from serving_native.contract_v2 import (
    canonical_sha256,
    file_artifact,
    latency_summary,
    module_path_snapshot,
    sha256_file,
)
from serving_native.runner import _load_candidate, _performance_estimates
from serving_native.workloads import as_dict, get_workload

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
        self.candidate = self._file(
            "candidate.py", "def run(inputs, runtime):\n    pass\n"
        )
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
            order = start if pair_index % 2 == 0 else ("BA" if start == "AB" else "AB")
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
        store_block_m = 32 if is_candidate and not reference_delegated else 128
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
            "masked_m_mutation_replayed": True,
            "output_poison_replayed": True,
            "untouched_masked_regions_preserved": True,
            "masked_store_contract": (
                "poison preserved outside scheduled full store_block_m tiles"
            ),
            "masked_store_observation": {
                "out": {
                    "store_block_m": store_block_m,
                    "padding_rows_written": 1,
                    "untouched_rows_checked": 1,
                }
            },
            "deterministic_replay": True,
            "approved_tolerance_passed": True,
            "fallback": False,
            "reference_delegated": reference_delegated,
            "trusted_config": trusted_config,
        }

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
            "performance_estimates": _performance_estimates(samples),
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

    def _w13_runtime_identity(self) -> dict:
        source = {
            "base_commit": "731e7c7a97d269e4b9f482ea18d0e709a948f293",
            "candidate_commit": "87e0359edbb461181d3bba218442132007b9a738",
            "cutlass_commit": "f3fde58372d33e9a5650ba7b80fc48b3b49d40c8",
            "fmt_commit": "553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28",
            "candidate_diff_sha256": (
                "465c8373c0a37970225a0e93267b6c399431b23e22cf35b4511db2308df98092"
            ),
            "candidate_diff_file_sha256": (
                "465c8373c0a37970225a0e93267b6c399431b23e22cf35b4511db2308df98092"
            ),
            "base_blob_sha256": {
                "csrc/apis/gemm.hpp": (
                    "0840d64249e2a5a4a994d495e8320a0fff26bad9ca107426a1a1226e7d621186"
                ),
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
            },
            "stock_source_tree_sha256": (
                "917592ab68ea0608c9be33208c2c609bc7f20bd9b1603f32743dd0d1ae03d0ed"
            ),
            "candidate_source_tree_sha256": (
                "d682daa65b8ba0ac3846d766910b8c751e0568fe62087084271bb354e46c49e4"
            ),
        }
        manifest = self.root / "w13-manifest.json"
        modules = {}
        build_plans = {}
        for name in ("stock", "candidate"):
            package = self.root / f"w13-{name}" / "package"
            cache = self.root / f"w13-{name}" / "jit"
            build_directory = self.root / f"w13-{name}" / "build"
            package.mkdir(parents=True, exist_ok=True)
            cache.mkdir(parents=True, exist_ok=True)
            build_directory.mkdir(parents=True, exist_ok=True)
            init_py = package / "__init__.py"
            shared_object = package / "_C.so"
            jit_artifact = cache / "kernel.cubin"
            build_ninja = build_directory / "build.ninja"
            init_py.write_text(f"{name} package\n")
            shared_object.write_text(f"{name} dso\n")
            jit_artifact.write_text(f"{name} jit\n")
            build_ninja.write_text("command = c++ <SOURCE> -o <BUILD>\n")
            build_plans[name] = build_ninja
            modules[name] = {
                "package": str(package),
                "package_init_sha256": sha256_file(init_py),
                "shared_object": str(shared_object),
                "shared_object_sha256": sha256_file(shared_object),
                "jit_cache": str(cache),
                "jit_artifacts": {"kernel.cubin": sha256_file(jit_artifact)},
            }
        normalized_build_plan_sha256 = sha256_file(build_plans["stock"])
        variants = {
            name: {key: value for key, value in item.items() if key != "jit_artifacts"}
            | {
                "commit": (
                    source["base_commit"]
                    if name == "stock"
                    else source["candidate_commit"]
                ),
                "source_tree_sha256": source[f"{name}_source_tree_sha256"],
                "build_ninja": str(build_plans[name]),
                "build_ninja_sha256": sha256_file(build_plans[name]),
                "normalized_build_plan_sha256": (normalized_build_plan_sha256),
            }
            for name, item in modules.items()
        }
        cxx = self.root / "fixture-cxx"
        nvcc = self.root / "fixture-nvcc"
        cxx.write_text("fixture cxx\n")
        nvcc.write_text("fixture nvcc\n")
        manifest.write_text(
            __import__("json").dumps(
                {
                    "schema_version": 3,
                    "source": source,
                    "build": {
                        "cuda_arch": "10.0a",
                        "stock_candidate_command_identical": True,
                        "compile_api": "tvm_ffi.cpp.build",
                        "force_clean_build_directories": True,
                        "jit_compiler": "nvcc",
                        "max_jobs": "4",
                        "elf_symbol_binding": "Bsymbolic",
                        "elf_symbol_visibility": "hidden",
                        "normalized_build_plan_sha256": (normalized_build_plan_sha256),
                        "cxx_path": str(cxx),
                        "cxx_sha256": sha256_file(cxx),
                        "nvcc_path": str(nvcc),
                        "nvcc_sha256": sha256_file(nvcc),
                    },
                    "variants": variants,
                },
                sort_keys=True,
            )
        )
        required = {"pdl": True, "num_sms": 148, "tc_util": 100}
        mutations = {"pdl": False, "num_sms": 147, "tc_util": 99}
        independence = {
            field: {
                "mutate_stock": {
                    "mutated_value": mutation,
                    "candidate_unchanged": required[field],
                    "restored": required[field],
                },
                "mutate_candidate": {
                    "mutated_value": mutation,
                    "stock_unchanged": required[field],
                    "restored": required[field],
                },
            }
            for field, mutation in mutations.items()
        }
        variant = "bm16_1sm"
        # Round-2 widened `w13_config`; the sixth element must be 0 for every
        # validated identity.
        config = [16, 128, 128, 11, 1, 0]
        provider_path = self.root / "provider_bm16_1sm.py"
        provider_path.write_text("fixture API-v1 provider\n")
        provider_name = "infini_kernel_glm52_moe_w13_decode_bm16_1sm"
        manifest_sha256 = sha256_file(manifest)
        return {
            "manifest": str(manifest),
            "manifest_sha256": manifest_sha256,
            "manifest_schema": 3,
            "variant": variant,
            "config": config,
            "provider": {
                "path": str(provider_path),
                "sha256": sha256_file(provider_path),
                "state": {
                    "gpu_id": 0,
                    "module_name": "fixture_w13_provider",
                    "module_ref": str(provider_path),
                    "provider_info": {
                        "name": provider_name,
                        "build_id": "bm16-1sm-stage11-api-v1",
                        "git_commit": source["candidate_commit"],
                    },
                    "ready": True,
                    "reason": "ready",
                    "selected_ops": ["moe_gate_proj"],
                },
                "identity": {
                    "name": variant,
                    "config": config,
                    "manifest": str(manifest),
                    "manifest_sha256": manifest_sha256,
                    "shared_object": modules["candidate"]["shared_object"],
                    "shared_object_sha256": modules["candidate"][
                        "shared_object_sha256"
                    ],
                    "jit_cache": modules["candidate"]["jit_cache"],
                    "jit_artifacts": modules["candidate"]["jit_artifacts"],
                },
            },
            "runtime_state": {
                "installed_downstream": required,
                "stock": required,
                "candidate": required,
            },
            "state_independence": independence,
            "modules": modules,
            "broad_precompile_enabled": False,
            "jit_use_nvrtc": False,
            "candidate_call_path": (
                "sglang.glm52_opt.hotspot_provider.run_moe_masked"
                " -> API-v1 provider moe_w13 -> exact DeepGEMM symbol"
            ),
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
        all_raw_samples = [
            sample for item in series for sample in item["raw_ordered_samples"]
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
            identity_control=identity_control,
            candidate_api=candidate_api,
        )
        totals = {
            field: sum(item[field] for item in by_phase.values())
            for field in COUNTER_FIELDS
        }
        result = {
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
                    "independent non-default streams" if mode == "cuda_graph" else None
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
                    "reference_delegations": totals["candidate_reference_delegations"],
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
                "series_gate_contract": ("all_four_estimates_each_series_gte_1p03_v1"),
                "required_estimates_finite": True,
                "performance_estimates": _performance_estimates(all_raw_samples),
                "performance_gate_passed": False,
                "identity_control_forced_non_win": identity_control,
            },
        }
        if task.startswith("moe_w13_"):
            result["provenance"]["w13_runtime"] = self._w13_runtime_identity()
            if mode == "cuda_graph":
                for item in result["series"]:
                    for capture in item["graph"]["captures"]:
                        if (
                            capture["implementation"] == "candidate"
                            and capture["reference_delegated"] is False
                        ):
                            capture["masked_store_observation"]["out"][
                                "store_block_m"
                            ] = 16
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
            estimates = _performance_estimates(series["raw_ordered_samples"])
            series["performance_estimates"] = estimates
            series["passes_3pct_gate"] = all(
                estimates[field] >= 1.03
                for field in (
                    "pooled_speedup",
                    "order_balanced_speedup",
                    "ab_median_speedup",
                    "ba_median_speedup",
                )
            )
            medians.append(median)
            all_reference.extend(reference_values)
            all_candidate.extend(candidate_values)
        result["reference"] = latency_summary(all_reference)
        result["candidate"].update(latency_summary(all_candidate))
        result["candidate"]["series_median_speedups"] = medians
        every = all(item["passes_3pct_gate"] for item in result["series"])
        all_raw_samples = [
            sample
            for series in result["series"]
            for sample in series["raw_ordered_samples"]
        ]
        result["aggregate"]["every_series_passes_3pct"] = every
        result["aggregate"]["required_estimates_finite"] = True
        result["aggregate"]["performance_estimates"] = _performance_estimates(
            all_raw_samples
        )
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
            self.assertFalse(hasattr(module, "__trusted_reference_delegation__"))
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
        result["provenance"]["workload_sha256"] = canonical_sha256(result["workload"])
        self.assert_invalid(result, "canonical WORKLOADS registry")

    def test_unknown_workload_is_rejected(self) -> None:
        result = copy.deepcopy(self.valid_eager)
        result["workload"]["name"] = "forged_workload"
        result["provenance"]["workload_sha256"] = canonical_sha256(result["workload"])
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
            (
                "requested",
                "requested/raw series counts",
                lambda value: value["run"].update(requested_series=4),
            ),
            (
                "completed",
                "requested/completed/raw",
                lambda value: value["aggregate"].update(completed_series=2),
            ),
            (
                "repeat",
                "repeat does not close",
                lambda value: value["series"][0].update(repeat=3),
            ),
            (
                "warmup",
                "warmup_pairs does not close",
                lambda value: value["series"][0].update(warmup_pairs=4),
            ),
            (
                "raw",
                "raw ordered samples incomplete",
                lambda value: value["series"][0]["raw_ordered_samples"].pop(),
            ),
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

    def test_four_estimator_records_are_recomputed_fail_closed(self) -> None:
        for scope, field, value in (
            ("series", "ba_median_speedup", 9.0),
            ("aggregate", "pooled_speedup", float("nan")),
        ):
            with self.subTest(scope=scope, field=field):
                result = copy.deepcopy(self.valid_eager)
                estimates = (
                    result["series"][0]["performance_estimates"]
                    if scope == "series"
                    else result["aggregate"]["performance_estimates"]
                )
                estimates[field] = value
                self.assert_invalid(result, field)

    def test_paired_median_cannot_hide_failing_ba_stratum(self) -> None:
        result = self._valid_fixture("eager", identity_control=False)
        all_reference: list[float] = []
        all_candidate: list[float] = []
        all_samples: list[dict] = []
        medians: list[float] = []
        for series in result["series"]:
            reference_values: list[float] = []
            candidate_values: list[float] = []
            ratios: list[float] = []
            for sample in series["raw_ordered_samples"]:
                if sample["implementation"] == "reference":
                    sample["latency_ms"] = 1.0
                    reference_values.append(1.0)
                else:
                    ratio = 1.10 if sample["order"] == "AB" else 1.00
                    sample["latency_ms"] = 1.0 / ratio
                    candidate_values.append(sample["latency_ms"])
            for offset in range(0, len(series["raw_ordered_samples"]), 2):
                pair = series["raw_ordered_samples"][offset : offset + 2]
                values = {
                    sample["implementation"]: sample["latency_ms"] for sample in pair
                }
                ratios.append(values["reference"] / values["candidate"])
            estimates = _performance_estimates(series["raw_ordered_samples"])
            self.assertGreaterEqual(statistics.median(ratios), 1.03)
            self.assertGreaterEqual(estimates["pooled_speedup"], 1.03)
            self.assertGreaterEqual(
                estimates["order_balanced_speedup"],
                1.03,
            )
            self.assertGreaterEqual(estimates["ab_median_speedup"], 1.03)
            self.assertLess(estimates["ba_median_speedup"], 1.03)
            series["reference"] = latency_summary(reference_values)
            series["candidate"] = latency_summary(candidate_values)
            series["paired_speedups"] = ratios
            series["median_speedup"] = statistics.median(ratios)
            series["performance_estimates"] = estimates
            series["passes_3pct_gate"] = False
            medians.append(series["median_speedup"])
            all_reference.extend(reference_values)
            all_candidate.extend(candidate_values)
            all_samples.extend(series["raw_ordered_samples"])
        result["reference"] = latency_summary(all_reference)
        result["candidate"].update(latency_summary(all_candidate))
        result["candidate"]["series_median_speedups"] = medians
        result["aggregate"]["every_series_passes_3pct"] = False
        result["aggregate"]["required_estimates_finite"] = True
        result["aggregate"]["performance_estimates"] = _performance_estimates(
            all_samples
        )
        result["aggregate"]["performance_gate_passed"] = False
        report = audit_document(result, verify_files=True)
        self.assertTrue(report["valid"], report)
        self.assertFalse(report["performance_gate_passed"], report)

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

    def test_graph_capture_ids_are_bound_to_independent_round_robin_captures(
        self,
    ) -> None:
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
                    stream_id=value["series"][0]["graph"]["captures"][0]["stream_id"]
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

    def test_graph_copy_and_adapter_nodes_are_rejected_after_recomputation(
        self,
    ) -> None:
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
                    item["kernel"] for item in capture["nodes"] if "kernel" in item
                ]
                capture["forbidden_nodes"] = []
                self.assert_invalid(result, "has forbidden graph nodes")

    def test_w13_graph_requires_mask_mutation_and_untouched_region_proofs(self) -> None:
        task = "moe_w13_grouped_decode_m16_em4"
        result = self._valid_fixture(
            "cuda_graph",
            task=task,
            identity_control=False,
        )
        self.assertTrue(audit_document(result, verify_files=True)["valid"])
        for field in (
            "masked_m_mutation_replayed",
            "untouched_masked_regions_preserved",
        ):
            with self.subTest(field=field):
                broken = self._valid_fixture(
                    "cuda_graph",
                    task=task,
                    identity_control=False,
                )
                broken["series"][0]["graph"]["captures"][0][field] = False
                self.assert_invalid(broken, f"{field} did not pass")
        broken = self._valid_fixture(
            "cuda_graph",
            task=task,
            identity_control=False,
        )
        broken["series"][0]["graph"]["captures"][0]["masked_store_contract"] = (
            "valid rows only"
        )
        self.assert_invalid(broken, "masked_store_contract invalid")
        broken = self._valid_fixture(
            "cuda_graph",
            task=task,
            identity_control=False,
        )
        broken["series"][0]["graph"]["captures"][1]["masked_store_observation"]["out"][
            "store_block_m"
        ] = 128
        self.assert_invalid(broken, "store_block_m mismatch")
        broken = self._valid_fixture(
            "cuda_graph",
            task=task,
            identity_control=False,
        )
        broken["series"][0]["graph"]["captures"][0]["masked_store_observation"]["out"][
            "untouched_rows_checked"
        ] = 0
        self.assert_invalid(broken, "did not check any untouched rows")

    def test_w13_manifest_build_source_and_runtime_binding_fail_closed(self) -> None:
        task = "moe_w13_grouped_decode_m16_em4"

        def mutate_and_audit(mutator, expected: str) -> None:
            result = self._valid_fixture("eager", task=task)
            runtime = result["provenance"]["w13_runtime"]
            manifest_path = Path(runtime["manifest"])
            manifest = json.loads(manifest_path.read_text())
            mutator(manifest)
            manifest_path.write_text(json.dumps(manifest, sort_keys=True))
            runtime["manifest_sha256"] = sha256_file(manifest_path)
            self.assert_invalid(result, expected)

        with self.subTest("base blob"):
            mutate_and_audit(
                lambda manifest: manifest["source"]["base_blob_sha256"].__setitem__(
                    "csrc/apis/gemm.hpp", "0" * 64
                ),
                "manifest source identity mismatch",
            )
        with self.subTest("build contract"):
            mutate_and_audit(
                lambda manifest: manifest["build"].__setitem__(
                    "elf_symbol_visibility", "default"
                ),
                "manifest build contract mismatch",
            )
        with self.subTest("runtime binding"):
            mutate_and_audit(
                lambda manifest: manifest["variants"]["stock"].__setitem__(
                    "jit_cache", "/wrong/cache"
                ),
                "does not match manifest variant",
            )

    def test_eager_kernel_identities_are_recomputed_from_events(self) -> None:
        result = copy.deepcopy(self.valid_eager)
        result["execution"]["kernel_profiles"]["candidate"]["kernel_identities"] = [
            "forged"
        ]
        self.assert_invalid(result, "do not match profiler events")

    def test_candidate_result_path_must_match_hashed_artifact(self) -> None:
        result = copy.deepcopy(self.valid_eager)
        result["candidate"]["path"] = str(self.root / "forged.py")
        self.assert_invalid(result, "path does not match hashed artifact")


if __name__ == "__main__":
    unittest.main()
