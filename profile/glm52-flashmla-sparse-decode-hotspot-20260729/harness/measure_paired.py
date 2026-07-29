#!/usr/bin/env python3
"""Fair four-lane paired measurement for the exact FlashMLA hotspot ABI."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[3]
SGLANG_ROOT = Path(
    "/home/qinhaiyan/glm52-hotspot-goal-runs/worktrees/"
    "flashmla-sparse-decode/sglang"
).resolve()
PROVIDER = (
    REPO_ROOT
    / "serving_native/candidates/flashmla_sparse_decode_provider.py"
).resolve()
if Path(os.environ.get("SGLANG_ROOT", "")).resolve() != SGLANG_ROOT:
    raise RuntimeError(f"SGLANG_ROOT must be {SGLANG_ROOT}")
if "GLM52_PHYSICAL_GPU" not in os.environ:
    raise RuntimeError("CUDA work must run through with_hotspot_gpu.sh")

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SGLANG_ROOT / "python"))

from serving_native.runner import Runtime  # noqa: E402
from serving_native.workloads import get_workload  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, choices=(16, 32), required=True)
    parser.add_argument(
        "--comparison",
        choices=("stock_stock", "stock_provider"),
        required=True,
    )
    parser.add_argument("--series", type=int, default=3)
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.series < 3:
        parser.error("--series must be at least 3")
    if args.pairs < 100 or args.pairs % 2:
        parser.error("--pairs must be even and at least 100 (50 AB and 50 BA)")
    if args.warmup < 3:
        parser.error("--warmup must be at least 3")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ratio_of_medians(rows: list[dict[str, object]]) -> float:
    return statistics.median(float(row["a_us"]) for row in rows) / statistics.median(
        float(row["b_us"]) for row in rows
    )


def summarize_series(rows: list[dict[str, object]]) -> dict[str, object]:
    ab = [row for row in rows if row["order"] == "AB"]
    ba = [row for row in rows if row["order"] == "BA"]
    if not ab or not ba:
        raise AssertionError("both AB and BA observations are required")
    pooled = ratio_of_medians(rows)
    ab_median = ratio_of_medians(ab)
    ba_median = ratio_of_medians(ba)
    balanced = math.sqrt(ab_median * ba_median)
    estimators = {
        "pooled_ratio_of_medians": pooled,
        "order_balanced_sqrt_ab_ba": balanced,
        "ab_ratio_of_medians": ab_median,
        "ba_ratio_of_medians": ba_median,
    }
    return {
        "pairs": len(rows),
        "ab_pairs": len(ab),
        "ba_pairs": len(ba),
        "a_median_us": statistics.median(float(row["a_us"]) for row in rows),
        "b_median_us": statistics.median(float(row["b_us"]) for row in rows),
        "estimators": estimators,
        "all_estimators_finite": all(math.isfinite(value) for value in estimators.values()),
        "all_estimators_ge_1_03": all(value >= 1.03 for value in estimators.values()),
    }


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite evidence: {output}")

    task = f"dsa_flashmla_kv_decode_m{args.m}"
    runtime = Runtime(get_workload(task))
    torch = runtime.torch
    try:
        inputs = runtime.build_inputs()
        expected_pages = 2049 if args.m == 16 else 4097
        if tuple(inputs["kv_cache"].shape) != (expected_pages, 64, 1, 656):
            raise AssertionError("page fixture does not match the promotional bucket")

        from sgl_kernel.flash_mla import flash_mla_with_kvcache
        from sglang.srt.layers.attention.dsa_backend import (
            DeepseekSparseAttnBackend,
        )

        provider_module = None
        if args.comparison == "stock_provider":
            os.environ.update(
                {
                    "SGLANG_GLM52_OPT": "1",
                    "SGLANG_GLM52_OPT_PROFILE": "hotspot_candidates",
                    "SGLANG_GLM52_OPT_OPS": "flashmla_sparse_decode",
                    "SGLANG_GLM52_OPT_M_BUCKETS": "dsa_decode_attn:16|32",
                    "SGLANG_GLM52_HOTSPOT_MODULE": str(PROVIDER),
                }
            )
            from sglang.srt.layers.glm52_opt import config, hotspot_provider
            from sglang.srt.layers.glm52_opt.context import set_forward_mode
            from sglang.srt.model_executor.forward_batch_info import ForwardMode

            config.load_manifest.cache_clear()
            hotspot_provider._reset_hotspot_provider_for_tests()
            set_forward_mode(ForwardMode.DECODE, args.m)
            hotspot_provider.initialize_hotspot_provider(gpu_id=0)
            state = hotspot_provider.provider_state()
            provider_module = sys.modules[state["module_name"]]

        def stock_leaf():
            return flash_mla_with_kvcache(
                q=inputs["q"],
                k_cache=inputs["kv_cache"],
                block_table=inputs["block_table"],
                cache_seqlens=inputs["cache_seqlens"],
                head_dim_v=inputs["head_dim_v"],
                tile_scheduler_metadata=inputs["tile_scheduler_metadata"],
                num_splits=inputs["num_splits"],
                softmax_scale=inputs["softmax_scale"],
                causal=False,
                is_fp8_kvcache=True,
                indices=inputs["indices"],
            )

        if args.comparison == "stock_provider":
            from sglang.srt.layers.glm52_opt import hotspot_provider

            callback_kwargs = {
                "q": inputs["q"],
                "k_cache": inputs["kv_cache"],
                "cache_seqlens": inputs["cache_seqlens"],
                "head_dim_v": inputs["head_dim_v"],
                "tile_scheduler_metadata": inputs["tile_scheduler_metadata"],
                "num_splits": inputs["num_splits"],
                "softmax_scale": inputs["softmax_scale"],
                "indices": inputs["indices"],
                "block_table": inputs["block_table"],
                "is_fp8_kvcache": True,
            }

            def b_leaf():
                return hotspot_provider.run_flashmla_sparse_decode(**callback_kwargs)

        else:
            b_leaf = stock_leaf

        backend = object.__new__(DeepseekSparseAttnBackend)
        backend.real_page_size = 64
        backend.kv_cache_dim = 656
        backend.dsa_kv_cache_store_fp8 = True
        backend.flashmla_kv_num_q_heads = 64
        backend.dsa_index_topk = 2048
        layer = SimpleNamespace(tp_q_head_num=64, head_dim=576)
        metadata = SimpleNamespace(
            dsa_cache_seqlens_int32=inputs["cache_seqlens"],
            flashmla_metadata=SimpleNamespace(
                flashmla_metadata=inputs["tile_scheduler_metadata"],
                num_splits=inputs["num_splits"],
            ),
        )

        def stock_containing():
            return backend._forward_flashmla_kv(
                q_all=inputs["q"].view(args.m, 64, 576),
                kv_cache=inputs["kv_cache"],
                v_head_dim=512,
                sm_scale=0.0625,
                layer=layer,
                metadata=metadata,
                page_table_1=inputs["indices"].squeeze(1),
                use_glm52_hotspot=False,
            )

        if args.comparison == "stock_provider":

            def b_containing():
                return backend._forward_flashmla_kv(
                    q_all=inputs["q"].view(args.m, 64, 576),
                    kv_cache=inputs["kv_cache"],
                    v_head_dim=512,
                    sm_scale=0.0625,
                    layer=layer,
                    metadata=metadata,
                    page_table_1=inputs["indices"].squeeze(1),
                    use_glm52_hotspot=True,
                )

        else:
            b_containing = stock_containing

        # Exact-shape warmup before any timing or capture.
        for _ in range(args.warmup):
            stock_leaf()
            b_leaf()
            stock_containing()
            b_containing()
        torch.cuda.synchronize(runtime.device)

        def capture(fn):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = fn()
            return graph, captured

        lanes = {
            "leaf_eager": (stock_leaf, b_leaf, False),
            "leaf_graph": (stock_leaf, b_leaf, True),
            "containing_eager": (stock_containing, b_containing, False),
            "containing_graph": (stock_containing, b_containing, True),
        }
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        def one(fn) -> float:
            start_event.record()
            fn()
            end_event.record()
            end_event.synchronize()
            return float(start_event.elapsed_time(end_event) * 1000.0)

        lane_results: dict[str, object] = {}
        for lane_name, (a_source, b_source, is_graph) in lanes.items():
            series_results = []
            for series_index in range(args.series):
                capture_order = None
                captured_outputs = None
                if is_graph:
                    # Graph capture position itself showed a stable ~2% bias in
                    # the first stock-vs-stock control. Recreate both graphs for
                    # each independent series and alternate capture order so a
                    # candidate cannot inherit a fixed first/second advantage.
                    if series_index % 2 == 0:
                        a_graph, a_captured = capture(a_source)
                        b_graph, b_captured = capture(b_source)
                        capture_order = "AB"
                    else:
                        b_graph, b_captured = capture(b_source)
                        a_graph, a_captured = capture(a_source)
                        capture_order = "BA"
                    captured_outputs = (a_captured, b_captured)
                    a_fn = a_graph.replay
                    b_fn = b_graph.replay
                    torch.cuda.synchronize(runtime.device)
                else:
                    a_fn = a_source
                    b_fn = b_source
                for _ in range(args.warmup):
                    a_fn()
                    b_fn()
                torch.cuda.synchronize(runtime.device)
                rows = []
                start_with_ab = series_index % 2 == 0
                for pair_index in range(args.pairs):
                    is_ab = (pair_index % 2 == 0) == start_with_ab
                    if is_ab:
                        a_us = one(a_fn)
                        b_us = one(b_fn)
                        order = "AB"
                    else:
                        b_us = one(b_fn)
                        a_us = one(a_fn)
                        order = "BA"
                    rows.append(
                        {
                            "pair": pair_index,
                            "order": order,
                            "a_us": a_us,
                            "b_us": b_us,
                            "paired_speedup": a_us / b_us,
                        }
                    )
                series_results.append(
                    {
                        "series": series_index + 1,
                        "starts_with": "AB" if start_with_ab else "BA",
                        "graph_capture_order": capture_order,
                        "summary": summarize_series(rows),
                        "raw_pairs": rows,
                    }
                )
                del captured_outputs
            lane_results[lane_name] = {
                "series": series_results,
                "passes_every_series_every_estimator_1_03": all(
                    series["summary"]["all_estimators_ge_1_03"]
                    for series in series_results
                ),
            }

        extension = (
            None
            if provider_module is None
            else Path(provider_module._EXTENSION.__file__).resolve()
        )
        physical_gpu = os.environ["GLM52_PHYSICAL_GPU"]
        clock_snapshot = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                physical_gpu,
                "--query-gpu=timestamp,uuid,pstate,clocks.sm,clocks.mem,"
                "temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()
        evidence = {
            "schema_version": 1,
            "stage": (
                "A0_fairness_control"
                if args.comparison == "stock_stock"
                else "provider_fair_measurement"
            ),
            "task": task,
            "m": args.m,
            "comparison": args.comparison,
            "a": "installed_sgl_kernel_stock",
            "b": (
                "installed_sgl_kernel_stock"
                if args.comparison == "stock_stock"
                else provider_module.PROVIDER_INFO["variant"]
            ),
            "timing": {
                "unit": "microseconds",
                "method": "CUDA events; one synchronization per observation",
                "series": args.series,
                "pairs_per_series": args.pairs,
                "ab_pairs_per_series": args.pairs // 2,
                "ba_pairs_per_series": args.pairs // 2,
                "warmup_per_boundary": args.warmup,
                "series_start_order": [
                    "AB" if index % 2 == 0 else "BA"
                    for index in range(args.series)
                ],
            },
            "lanes": lane_results,
            "runtime": runtime.runtime_evidence(inputs),
            "gpu": {
                "physical_index": int(physical_gpu),
                "uuid": os.environ["GLM52_PHYSICAL_GPU_UUID"],
                "logical_index": 0,
                "name": torch.cuda.get_device_properties(0).name,
                "compute_capability": list(torch.cuda.get_device_capability(0)),
                "clock_snapshot": clock_snapshot,
            },
            "provider": (
                None
                if provider_module is None
                else provider_module.candidate_evidence()
            ),
            "candidate_extension": (
                None
                if extension is None
                else {
                    "path": str(extension),
                    "sha256": sha256(extension),
                    "size_bytes": extension.stat().st_size,
                }
            ),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
        concise = {
            lane: [
                item["summary"]["estimators"]
                for item in details["series"]
            ]
            for lane, details in lane_results.items()
        }
        print(json.dumps(concise, indent=2, sort_keys=True))
        return 0
    finally:
        if args.comparison == "stock_provider":
            from sglang.srt.layers.glm52_opt.context import set_forward_mode

            set_forward_mode(None)
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
