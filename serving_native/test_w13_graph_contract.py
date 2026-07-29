"""GPU-free W13 workload and graph-observation contract tests."""

from __future__ import annotations

import inspect
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch
except ImportError:  # Structural CI intentionally has no Torch dependency.
    torch = None

from serving_native.runner import (
    MOE_OUTPUT_POISON,
    Runtime,
    TaskResult,
    _assert_untouched_masked_regions,
    _compare_masked,
)
from serving_native.w13_runtime import W13Runtime
from serving_native.workloads import WORKLOADS, get_workload


class _NoDeviceRead:
    def tolist(self):
        raise AssertionError("device masked_m must not be read on the host")


class _FakeDeepGemm:
    def __init__(self, name: str):
        self.name = name
        self.pdl = False
        self.num_sms = 0
        self.tc_util = 0

    def set_pdl(self, value):
        self.pdl = bool(value)

    def get_pdl(self):
        return self.pdl

    def set_num_sms(self, value):
        self.num_sms = int(value)

    def get_num_sms(self):
        return self.num_sms

    def set_tc_util(self, value):
        self.tc_util = int(value)

    def get_tc_util(self):
        return self.tc_util


@unittest.skipIf(torch is None, "repo-local Torch environment is not configured")
class W13GraphContractTest(unittest.TestCase):
    @staticmethod
    def _runtime(task: str = "moe_w13_grouped_decode_m16_em4") -> Runtime:
        runtime = object.__new__(Runtime)
        runtime.torch = torch
        runtime.device = torch.device("cpu")
        runtime.workload = get_workload(task)
        return runtime

    def test_all_four_expected_m_points_are_independently_named(self) -> None:
        leaf = {
            (
                workload.params["decode_m"],
                workload.params["expected_m"],
            )
            for workload in WORKLOADS.values()
            if workload.name.startswith("moe_w13_grouped_decode_")
        }
        region = {
            (
                workload.params["decode_m"],
                workload.params["expected_m"],
            )
            for workload in WORKLOADS.values()
            if workload.name.startswith("moe_w13_region_decode_")
        }
        required = {(16, 4), (16, 5), (32, 8), (32, 9)}
        self.assertEqual(leaf, required)
        self.assertEqual(region, required)
        self.assertTrue(
            all(
                workload.execution_modes == ("eager", "cuda_graph")
                for workload in WORKLOADS.values()
                if workload.name.startswith(
                    ("moe_w13_grouped_decode_", "moe_w13_region_decode_")
                )
            )
        )

    def test_mask_metadata_is_cpu_known_and_has_a_distinct_replay_pattern(self) -> None:
        runtime = self._runtime()
        metadata = runtime._decode_mask_metadata(runtime.workload.params)
        self.assertEqual(sum(metadata["masked_m_initial_cpu"]), 128)
        self.assertEqual(sum(metadata["masked_m_replay_cpu"]), 128)
        self.assertNotEqual(
            metadata["masked_m_initial_cpu"],
            metadata["masked_m_replay_cpu"],
        )
        self.assertNotIn("mask_observe_rows_cpu", metadata)

    def test_observation_uses_static_cpu_metadata_not_device_mask(self) -> None:
        runtime = self._runtime()
        output = torch.arange(2 * 5 * 3).reshape(2, 5, 3)
        inputs = {
            "out": output,
            "masked_m": _NoDeviceRead(),
        }
        observed = runtime._observe_masked_output(inputs, "out")
        self.assertIs(observed, output)
        self.assertEqual(tuple(observed.shape), (2, 5, 3))
        self.assertNotIn(".tolist()", inspect.getsource(Runtime.reference))

    def test_full_output_poison_and_unscheduled_tiles_are_enforced(self) -> None:
        runtime = self._runtime()
        counts = tuple([1, 129] + [0] * 30)
        output = torch.empty((32, 256, 3), dtype=torch.bfloat16)
        inputs = {"out": output}
        Runtime.prepare_inputs(runtime, inputs)
        self.assertTrue(bool(output.eq(MOE_OUTPUT_POISON).all().item()))
        output[0, :128].zero_()
        output[1, :256].zero_()
        observation = _assert_untouched_masked_regions(runtime, inputs, counts)
        self.assertEqual(observation["out"]["store_block_m"], 128)
        self.assertEqual(observation["out"]["untouched_rows_checked"], 7808)
        output[0, 200, 0] = 0
        with self.assertRaisesRegex(AssertionError, "outside scheduled"):
            _assert_untouched_masked_regions(runtime, inputs, counts)

    def test_masked_comparison_uses_only_cpu_known_valid_rows(self) -> None:
        reference = torch.zeros((2, 4, 3), dtype=torch.bfloat16)
        candidate = reference.clone()
        candidate[0, 1:].fill_(17)
        candidate[1, 2:].fill_(-19)
        _compare_masked(
            TaskResult(reference),
            TaskResult(candidate),
            (1, 2),
        )
        candidate[1, 1, 0] = 3
        with self.assertRaisesRegex(AssertionError, "max abs diff"):
            _compare_masked(
                TaskResult(reference),
                TaskResult(candidate),
                (1, 2),
            )

    def test_containing_region_poison_covers_w13_and_w2_outputs(self) -> None:
        runtime = self._runtime("moe_w13_region_decode_m32_em9")
        inputs = {
            "out": torch.empty((32, 2, 2), dtype=torch.bfloat16),
            "down_out": torch.empty((32, 2, 2), dtype=torch.bfloat16),
        }
        Runtime.prepare_inputs(runtime, inputs)
        self.assertTrue(bool(inputs["out"].eq(MOE_OUTPUT_POISON).all().item()))
        self.assertTrue(bool(inputs["down_out"].eq(MOE_OUTPUT_POISON).all().item()))

    def test_harness_runtime_binds_stock_then_candidate_and_restores_env(self) -> None:
        from sglang.srt.layers.glm52_opt import w13_decode

        stock = _FakeDeepGemm("stock")
        candidate = _FakeDeepGemm("candidate")
        installed = _FakeDeepGemm("installed")
        events = []
        compile_utils = SimpleNamespace(_ENABLE_JIT_DEEPGEMM_PRECOMPILE=True)
        fake_cuda = SimpleNamespace(
            current_device=lambda: 0,
            get_device_capability=lambda _gpu: (10, 0),
            synchronize=lambda _device: events.append(("synchronize",)),
            empty_cache=lambda: events.append(("empty_cache",)),
        )
        fake_torch = SimpleNamespace(cuda=fake_cuda)
        device = SimpleNamespace(index=0)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            manifest.write_text("{}")
            records = {}
            snapshots = {}
            for name in ("stock", "candidate"):
                package = root / name / "package"
                cache = root / name / "jit"
                package.mkdir(parents=True)
                cache.mkdir(parents=True)
                records[name] = {
                    "package": str(package),
                    "package_init_sha256": f"{name}-init",
                    "shared_object": str(package / "_C.so"),
                    "shared_object_sha256": f"{name}-dso",
                    "jit_cache": str(cache),
                }
                snapshots[str(cache.resolve())] = {f"{name}.cubin": f"{name}-jit"}

            def load_variant(_manifest, name, **_kwargs):
                events.append(("load", name, os.environ["DG_JIT_CACHE_DIR"]))
                return (
                    stock if name == "stock" else candidate,
                    records[name],
                    {},
                )

            def set_state(module, label):
                events.append(("set_state", module.name, label))
                module.set_pdl(True)
                module.set_num_sms(148)
                module.set_tc_util(100)
                return {"pdl": True, "num_sms": 148, "tc_util": 100}

            def launch(module, _tensors, expected_m, config):
                events.append(
                    (
                        "launch",
                        module.name,
                        expected_m,
                        config,
                        os.environ["DG_JIT_CACHE_DIR"],
                    )
                )

            saved_dg = os.environ.get("DG_JIT_CACHE_DIR")
            saved_sglang = os.environ.get("SGLANG_DG_CACHE_DIR")
            with (
                patch.dict(
                    os.environ,
                    {
                        "DG_JIT_CACHE_DIR": "before-dg",
                        "SGLANG_DG_CACHE_DIR": "before-sglang",
                    },
                    clear=False,
                ),
                patch.dict(sys.modules, {"deep_gemm": installed}),
                patch.object(
                    w13_decode,
                    "_variant_record",
                    side_effect=lambda _path, name: (records[name], {}),
                ),
                patch.object(w13_decode, "load_variant", side_effect=load_variant),
                patch.object(
                    w13_decode,
                    "_set_required_runtime_state",
                    side_effect=set_state,
                ),
                patch.object(w13_decode, "_allocate_warm_inputs", return_value={}),
                patch.object(
                    w13_decode,
                    "_launch_named_config",
                    side_effect=launch,
                ),
                patch.object(
                    w13_decode,
                    "_cache_snapshot",
                    side_effect=lambda path: snapshots[str(path.resolve())],
                ),
                patch.object(
                    w13_decode,
                    "_prove_runtime_state_independence",
                    return_value={"proof": True},
                ),
                patch.object(
                    w13_decode,
                    "_sha256",
                    return_value="manifest-hash",
                ),
                patch(
                    "serving_native.w13_runtime.importlib.import_module",
                    side_effect=lambda _name: (
                        events.append(
                            ("compile_utils_import", os.environ["DG_JIT_CACHE_DIR"])
                        )
                        or compile_utils
                    ),
                ),
            ):
                runtime = W13Runtime(
                    fake_torch,
                    device,
                    manifest_path=manifest,
                    variant="bm32_1sm",
                )
                self.assertEqual(os.environ["DG_JIT_CACHE_DIR"], "before-dg")
                self.assertEqual(
                    os.environ["SGLANG_DG_CACHE_DIR"],
                    "before-sglang",
                )

            stock_load = next(
                index
                for index, event in enumerate(events)
                if event[:2] == ("load", "stock")
            )
            compile_import = next(
                index
                for index, event in enumerate(events)
                if event[0] == "compile_utils_import"
            )
            candidate_load = next(
                index
                for index, event in enumerate(events)
                if event[:2] == ("load", "candidate")
            )
            self.assertLess(stock_load, compile_import)
            self.assertLess(compile_import, candidate_load)
            self.assertEqual(
                events[stock_load][2],
                records["stock"]["jit_cache"],
            )
            self.assertEqual(
                events[candidate_load][2],
                records["candidate"]["jit_cache"],
            )
            self.assertFalse(compile_utils._ENABLE_JIT_DEEPGEMM_PRECOMPILE)
            self.assertFalse(runtime.identity["jit_use_nvrtc"])
            self.assertEqual(
                runtime.identity["runtime_state"],
                {
                    "installed_downstream": {
                        "pdl": True,
                        "num_sms": 148,
                        "tc_util": 100,
                    },
                    "stock": {"pdl": True, "num_sms": 148, "tc_util": 100},
                    "candidate": {"pdl": True, "num_sms": 148, "tc_util": 100},
                },
            )

            if saved_dg is None:
                os.environ.pop("DG_JIT_CACHE_DIR", None)
            else:
                os.environ["DG_JIT_CACHE_DIR"] = saved_dg
            if saved_sglang is None:
                os.environ.pop("SGLANG_DG_CACHE_DIR", None)
            else:
                os.environ["SGLANG_DG_CACHE_DIR"] = saved_sglang


if __name__ == "__main__":
    unittest.main()
