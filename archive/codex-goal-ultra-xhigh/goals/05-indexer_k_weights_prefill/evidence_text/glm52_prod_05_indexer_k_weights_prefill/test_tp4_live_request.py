#!/usr/bin/env python3
"""CPU-only tests for fail-closed TP4 Indexer trace attribution."""

from __future__ import annotations

import copy
import importlib.util
import unittest
from pathlib import Path


HELPER = Path(__file__).with_name("tp4_live_request.py")
SPEC = importlib.util.spec_from_file_location("tp4_live_request", HELPER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _row(
    *,
    device: str,
    context: str,
    stream: str,
    start: int,
    duration: int,
    grid: int,
    scope: str,
    kernel: str,
) -> dict[str, str]:
    return {
        "Start (ns)": str(start),
        "Duration (ns)": str(duration),
        "CorrId": str(start),
        "GrdX": str(grid),
        "Device": device,
        "Ctx": context,
        "Strm": stream,
        "Name": f"{{'Module': '{scope}'}}/{kernel}",
    }


def _valid_rows(*, same_device: bool = False) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index in range(4):
        device = "NVIDIA B200 (0)" if same_device else f"NVIDIA B200 ({index})"
        context = str(index + 1)
        base = index * 10_000_000
        parent = f"model.layers.{index}.self_attn.indexer"
        rows.extend(
            [
                _row(
                    device=device,
                    context=context,
                    stream="13",
                    start=base + 100,
                    duration=40,
                    grid=512,
                    scope=f"{parent}.wq_b",
                    kernel="nvjet_sm100_tst_wq",
                ),
                _row(
                    device=device,
                    context=context,
                    stream="7",
                    start=base + 200,
                    duration=20,
                    grid=128,
                    scope=f"{parent}.wk_weights_proj",
                    kernel="nvjet_sm100_tst_wk",
                ),
                _row(
                    device=device,
                    context=context,
                    stream="7",
                    start=base + 300,
                    duration=30,
                    grid=32768,
                    scope=parent,
                    kernel="fused_q_indexer_rope_hadamard_quant",
                ),
                _row(
                    device=device,
                    context=context,
                    stream="13",
                    start=base + 350,
                    duration=4,
                    grid=1024,
                    scope=parent,
                    kernel="fused_k_indexer_norm_rope_store",
                ),
            ]
        )
    return rows


class TraceAttributionTest(unittest.TestCase):
    def test_server_info_requires_exact_model_and_path_discriminators(self) -> None:
        expected_model = "nvidia/GLM-5.2-NVFP4"
        expected_revision = "fixed-revision"
        server_info = {
            "attention_backend": "dsa",
            "chunked_prefill_size": 4096,
            "context_length": 8192,
            "disable_flashinfer_autotune": True,
            "disable_radix_cache": True,
            "dp_size": 4,
            "dsa_prefill_backend": "trtllm",
            "dsa_topk_backend": "sgl-kernel",
            "enable_deepseek_v4_fp4_indexer": False,
            "enable_dp_attention": True,
            "enable_layerwise_nvtx_marker": True,
            "ep_size": 4,
            "kv_cache_dtype": "fp8_e4m3",
            "load_format": "dummy",
            "max_prefill_tokens": 4096,
            "max_running_requests": 4,
            "max_total_tokens": 8192,
            "mem_fraction_static": 0.8,
            "model_path": expected_model,
            "moe_a2a_backend": "deepep",
            "page_size": 64,
            "prefill_max_requests": 1,
            "quantization": "modelopt_fp4",
            "revision": expected_revision,
            "skip_tokenizer_init": True,
            "tp_size": 4,
            "trust_remote_code": True,
            "cuda_graph_config": {
                "prefill": {"backend": "disabled"},
                "decode": {"backend": "disabled"},
            },
        }
        self.assertEqual(
            MODULE._server_info_mismatches(
                server_info,
                expected_model=expected_model,
                expected_revision=expected_revision,
            ),
            [],
        )

        wrong = copy.deepcopy(server_info)
        wrong["model_path"] = f"/cache/{expected_model.split('/')[-1]}"
        wrong["disable_flashinfer_autotune"] = False
        wrong["enable_deepseek_v4_fp4_indexer"] = True
        mismatched_fields = {
            mismatch["field"]
            for mismatch in MODULE._server_info_mismatches(
                wrong,
                expected_model=expected_model,
                expected_revision=expected_revision,
            )
        }
        self.assertEqual(
            mismatched_fields,
            {
                "disable_flashinfer_autotune",
                "enable_deepseek_v4_fp4_indexer",
                "model_path",
            },
        )

    def test_accepts_four_distinct_exact_devices(self) -> None:
        analysis = MODULE._analyze_trace_rows(_valid_rows(), 4)
        self.assertTrue(analysis["trace_reachability_ok"])
        self.assertEqual(len(analysis["exact_schedule_devices"]), 4)

    def test_rejects_generic_same_grid_gemms(self) -> None:
        rows = copy.deepcopy(_valid_rows())
        for row in rows:
            if "nvjet_sm100_tst" in row["Name"]:
                row["Name"] = row["Name"].replace(
                    "self_attn.indexer", "mlp.unrelated"
                )
        analysis = MODULE._analyze_trace_rows(rows, 4)
        self.assertFalse(analysis["trace_reachability_ok"])

    def test_rejects_four_contexts_on_one_device(self) -> None:
        analysis = MODULE._analyze_trace_rows(_valid_rows(same_device=True), 4)
        self.assertFalse(analysis["trace_reachability_ok"])
        self.assertEqual(len(analysis["exact_schedule_devices"]), 1)

    def test_rejects_intervening_same_stream_kernel(self) -> None:
        rows = _valid_rows()
        rows.append(
            _row(
                device="NVIDIA B200 (0)",
                context="1",
                stream="13",
                start=200,
                duration=10,
                grid=1,
                scope="model.layers.0.self_attn.indexer",
                kernel="intervening_kernel",
            )
        )
        analysis = MODULE._analyze_trace_rows(rows, 4)
        self.assertFalse(analysis["trace_reachability_ok"])

    def test_rejects_adjacent_gemms_from_other_layers(self) -> None:
        rows = copy.deepcopy(_valid_rows())
        for row in rows:
            if "wq_b" in row["Name"]:
                row["Name"] = row["Name"].replace(
                    "model.layers.0", "model.layers.77"
                ).replace("model.layers.1", "model.layers.77").replace(
                    "model.layers.2", "model.layers.77"
                ).replace("model.layers.3", "model.layers.77")
            elif "wk_weights_proj" in row["Name"]:
                row["Name"] = row["Name"].replace(
                    "model.layers.0", "model.layers.76"
                ).replace("model.layers.1", "model.layers.76").replace(
                    "model.layers.2", "model.layers.76"
                ).replace("model.layers.3", "model.layers.76")
        analysis = MODULE._analyze_trace_rows(rows, 4)
        self.assertFalse(analysis["trace_reachability_ok"])


if __name__ == "__main__":
    unittest.main()
