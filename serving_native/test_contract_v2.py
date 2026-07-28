"""GPU-free adversarial regression tests for the serving-native V2 auditor."""

from __future__ import annotations

import copy
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
)
from serving_native.runner import _load_candidate
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
