#!/usr/bin/env python3
"""Directly paired provider-versus-provider CUDA Graph measurement.

Plan hypothesis 1 does not accept a combine win measured only against ancient
installed stock: it must improve on the *current* P1-plus-stock-combine chain.
This script measures that denominator directly instead of deriving it, by
putting the combine identity control in the A arm and the combine candidate in
the B arm of the same alternating AB/BA series, in one process, on one GPU.

Both arms run through the identical mechanism -- same provider file, same
``api_combine.cpp``, same pinned P1 main, same fixed workspaces, same SGLang
API-v1 dispatch -- so the only difference between them is the combine kernel's
machine code. ``audit_combine_binaries.py`` proves the A arm's combine SASS is
the stock combine and that both arms' main SASS is the reference P1 main.

Mechanism note: SGLang derives its provider module name from the provider *path*,
so re-initializing for the second variant replaces ``sys.modules`` entry. This
script keeps a strong reference to each loaded provider module (and therefore to
each shared object) for the whole run, so a captured graph's device code can
never be unloaded underneath a replay. Selection happens at capture time; replay
does not re-enter host code, which is verified by a launch-count probe.

Only CUDA Graph lanes are offered. Under the production graph-only default the
eager containing region is required to fall back to stock, so an eager
provider-versus-provider comparison would not be measuring two providers.
"""

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
    REPO_ROOT / "serving_native/candidates/flashmla_combine_decode_provider.py"
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
    parser.add_argument("--a-variant", required=True)
    parser.add_argument("--b-variant", required=True)
    parser.add_argument("--series", type=int, default=3)
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--replays-per-observation", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.series < 3:
        parser.error("--series must be at least 3")
    if args.pairs < 100 or args.pairs % 2:
        parser.error("--pairs must be even and at least 100 (50 AB and 50 BA)")
    if args.warmup < 3:
        parser.error("--warmup must be at least 3")
    if args.replays_per_observation < 1:
        parser.error("--replays-per-observation must be at least 1")
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
    estimators = {
        "pooled_ratio_of_medians": pooled,
        "order_balanced_sqrt_ab_ba": math.sqrt(ab_median * ba_median),
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
        "all_estimators_finite": all(
            math.isfinite(value) for value in estimators.values()
        ),
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
    loaded_modules: list[object] = []  # keeps every provider .so alive
    try:
        inputs = runtime.build_inputs()
        expected_pages = 2049 if args.m == 16 else 4097
        if tuple(inputs["kv_cache"].shape) != (expected_pages, 64, 1, 656):
            raise AssertionError("page fixture does not match the promotional bucket")
        expected_splits = list(range(0, 129, 8 if args.m == 16 else 4))
        actual_splits = inputs["num_splits"].detach().cpu().tolist()
        if actual_splits != expected_splits:
            raise AssertionError(
                f"unexpected num_splits: {actual_splits} != {expected_splits}"
            )

        os.environ.update(
            {
                "SGLANG_GLM52_OPT": "1",
                "SGLANG_GLM52_OPT_PROFILE": "hotspot_candidates",
                "SGLANG_GLM52_OPT_OPS": "flashmla_sparse_decode",
                "SGLANG_GLM52_OPT_M_BUCKETS": "dsa_decode_attn:16|32",
                "SGLANG_GLM52_HOTSPOT_MODULE": str(PROVIDER),
            }
        )
        from sglang.srt.layers.attention.dsa_backend import (
            DeepseekSparseAttnBackend,
        )
        from sglang.srt.layers.glm52_opt import config, hotspot_provider
        from sglang.srt.layers.glm52_opt.context import set_forward_mode
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        set_forward_mode(ForwardMode.DECODE, args.m)

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

        def leaf():
            return hotspot_provider.run_flashmla_sparse_decode(**callback_kwargs)

        def containing():
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

        def activate(variant: str):
            """Point SGLang's API-v1 dispatch at one combine variant."""
            os.environ["GLM52_FLASHMLA_COMBINE_VARIANT"] = variant
            config.load_manifest.cache_clear()
            hotspot_provider._reset_hotspot_provider_for_tests()
            hotspot_provider.initialize_hotspot_provider(gpu_id=0)
            state = hotspot_provider.provider_state()
            module = sys.modules[state["module_name"]]
            if module.PROVIDER_INFO["variant"] != variant:
                raise AssertionError(
                    f"provider reports {module.PROVIDER_INFO['variant']!r}, "
                    f"expected {variant!r}"
                )
            loaded_modules.append(module)
            return module

        # Build both extensions and warm both arms before any capture or timing.
        variant_modules: dict[str, object] = {}
        for variant in (args.a_variant, args.b_variant):
            module = activate(variant)
            variant_modules[variant] = module
            for _ in range(args.warmup):
                leaf()
                containing()
            torch.cuda.synchronize(runtime.device)

        # Prove selection happens at capture and never during replay.
        module_b = variant_modules[args.b_variant]
        before = int(module_b._EXTENSION.launch_count())
        probe_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(probe_graph):
            probe_captured = containing()
        torch.cuda.synchronize(runtime.device)
        capture_delta = int(module_b._EXTENSION.launch_count()) - before
        before = int(module_b._EXTENSION.launch_count())
        probe_graph.replay()
        torch.cuda.synchronize(runtime.device)
        replay_delta = int(module_b._EXTENSION.launch_count()) - before
        del probe_graph, probe_captured
        selection_probe = {
            "capture_provider_launches": capture_delta,
            "replay_host_launches": replay_delta,
            "graph_selects_candidate": capture_delta > 0,
            "replay_does_not_reenter_host_launch": replay_delta == 0,
        }

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        replays = args.replays_per_observation

        def one(fn) -> float:
            start_event.record()
            for _ in range(replays):
                fn()
            end_event.record()
            end_event.synchronize()
            return float(start_event.elapsed_time(end_event) * 1000.0 / replays)

        def capture(source):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = source()
            return graph, captured

        lane_results: dict[str, object] = {}
        for lane_name, source in (("containing_graph", containing), ("leaf_graph", leaf)):
            series_results = []
            for series_index in range(args.series):
                # Alternate which arm is captured first: graph capture position
                # itself carried a measurable bias in the round-2 control.
                if series_index % 2 == 0:
                    activate(args.a_variant)
                    a_graph, a_captured = capture(source)
                    activate(args.b_variant)
                    b_graph, b_captured = capture(source)
                    capture_order = "AB"
                else:
                    activate(args.b_variant)
                    b_graph, b_captured = capture(source)
                    activate(args.a_variant)
                    a_graph, a_captured = capture(source)
                    capture_order = "BA"
                torch.cuda.synchronize(runtime.device)
                a_fn, b_fn = a_graph.replay, b_graph.replay
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
                del a_graph, b_graph, a_captured, b_captured
            lane_results[lane_name] = {
                "series": series_results,
                "passes_every_series_every_estimator_1_03": all(
                    series["summary"]["all_estimators_ge_1_03"]
                    for series in series_results
                ),
            }

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
            "stage": "provider_pair_fair_measurement",
            "task": task,
            "m": args.m,
            "comparison": "provider_provider",
            "a": args.a_variant,
            "b": args.b_variant,
            "is_null_control": args.a_variant == args.b_variant,
            "lane_order": list(lane_results),
            "timing": {
                "unit": "microseconds",
                "method": "CUDA events; one synchronization per observation",
                "series": args.series,
                "pairs_per_series": args.pairs,
                "ab_pairs_per_series": args.pairs // 2,
                "ba_pairs_per_series": args.pairs // 2,
                "warmup_per_boundary": args.warmup,
                "replays_per_observation": replays,
                "observation_unit": "microseconds per single call",
                "series_start_order": [
                    "AB" if index % 2 == 0 else "BA" for index in range(args.series)
                ],
            },
            "lanes": lane_results,
            "graph_only_selection_probe": selection_probe,
            "runtime": runtime.runtime_evidence(inputs),
            "gpu": {
                "physical_index": int(physical_gpu),
                "uuid": os.environ["GLM52_PHYSICAL_GPU_UUID"],
                "logical_index": 0,
                "name": torch.cuda.get_device_properties(0).name,
                "compute_capability": list(torch.cuda.get_device_capability(0)),
                "clock_snapshot": clock_snapshot,
            },
            "arms": {
                variant: {
                    **module.candidate_evidence(),
                    "extension_sha256": sha256(
                        Path(module._EXTENSION.__file__).resolve()
                    ),
                }
                for variant, module in variant_modules.items()
            },
            "num_splits": actual_splits,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
        print(
            json.dumps(
                {
                    lane: [
                        item["summary"]["estimators"] for item in details["series"]
                    ]
                    for lane, details in lane_results.items()
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    finally:
        from sglang.srt.layers.glm52_opt.context import set_forward_mode

        set_forward_mode(None)
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
