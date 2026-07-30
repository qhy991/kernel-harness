#!/usr/bin/env python3
"""Time each host stage of the API-v1 FlashMLA dispatch guard, without cProfile.

cProfile's per-call overhead is comparable to the ~17 us being attributed, so
this uses perf_counter_ns around the real stages with real CUDA tensors. Only
host time is measured; the device launch is included only where noted.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path


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


def bench(fn, iters: int) -> float:
    """Median host microseconds per call over `iters` calls."""
    samples = []
    for _ in range(7):
        start = time.perf_counter_ns()
        for _ in range(iters):
            fn()
        samples.append((time.perf_counter_ns() - start) / iters / 1000.0)
    return statistics.median(samples)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, choices=(16, 32), default=16)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="before")
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite evidence: {args.output}")

    runtime = Runtime(get_workload(f"dsa_flashmla_kv_decode_m{args.m}"))
    torch = runtime.torch
    try:
        inputs = runtime.build_inputs()
        # Runtime() sets SGLANG_GLM52_OPT=0 to freeze the reference path, so the
        # hotspot environment has to be applied after it is constructed.
        os.environ.update(
        {
                "SGLANG_GLM52_OPT": "1",
                "SGLANG_GLM52_OPT_PROFILE": "hotspot_candidates",
                "SGLANG_GLM52_OPT_OPS": "flashmla_sparse_decode",
                "SGLANG_GLM52_OPT_M_BUCKETS": "dsa_decode_attn:16|32",
                "SGLANG_GLM52_HOTSPOT_MODULE": str(PROVIDER),
            }
        )
        from sglang.srt.layers.glm52_opt import config, dispatch, hotspot_provider
        from sglang.srt.layers.glm52_opt.context import set_forward_mode
        from sglang.srt.layers.glm52_opt.registry import lookup
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        config.load_manifest.cache_clear()
        hotspot_provider._reset_hotspot_provider_for_tests()
        set_forward_mode(ForwardMode.DECODE, args.m)
        hotspot_provider.initialize_hotspot_provider(gpu_id=0)

        kwargs = {
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
        m = args.m
        spec = lookup("dsa_decode_attn", "decode", m=m)
        if spec is None:
            raise AssertionError("no spec resolved; dispatch would not select")

        # Confirm the candidate really is selected before attributing anything.
        first = dispatch.try_dispatch_flashmla_sparse_decode(**kwargs)
        if first is None:
            raise AssertionError("dispatch returned None; provider did not hit")
        torch.cuda.synchronize(runtime.device)

        stages = {
            "full_try_dispatch_including_launch": lambda: (
                dispatch.try_dispatch_flashmla_sparse_decode(**kwargs)
            ),
            "provider_callback_including_launch": lambda: (
                hotspot_provider.run_flashmla_sparse_decode(**kwargs)
            ),
            "config_is_enabled": config.is_enabled,
            "current_phase": lambda: dispatch._current_phase(m),
            "registry_lookup": lambda: lookup("dsa_decode_attn", "decode", m=m),
            "abi_guard": lambda: dispatch._flashmla_hotspot_abi_matches(
                spec, **kwargs
            ),
            "profiler_range_name": lambda: dispatch._profiler_range_name(spec, m),
            "nvtx_range_ctx_enter_exit": lambda: (
                dispatch._nvtx_range("x").__enter__()
            ),
            "record_hit": lambda: dispatch._record_hit(
                "timing_probe", "dsa_decode_attn", "decode", m=m
            ),
            "emit_nvtx_config_read": config.emit_infini_kernel_nvtx,
        }
        results = {name: bench(fn, args.iters) for name, fn in stages.items()}
        torch.cuda.synchronize(runtime.device)

        guard_only_sum = sum(
            results[name]
            for name in (
                "config_is_enabled",
                "current_phase",
                "registry_lookup",
                "abi_guard",
                "profiler_range_name",
                "nvtx_range_ctx_enter_exit",
                "record_hit",
            )
        )
        evidence = {
            "schema_version": 1,
            "label": args.label,
            "m": args.m,
            "iters_per_sample": args.iters,
            "unit": "median host microseconds per call",
            "stages": results,
            "derived": {
                "dispatch_overhead_over_provider_callback": (
                    results["full_try_dispatch_including_launch"]
                    - results["provider_callback_including_launch"]
                ),
                "sum_of_named_guard_stages": guard_only_sum,
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
        print(json.dumps(evidence["stages"], indent=2, sort_keys=True))
        print(json.dumps(evidence["derived"], indent=2, sort_keys=True))
        return 0
    finally:
        from sglang.srt.layers.glm52_opt.context import set_forward_mode

        set_forward_mode(None)
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
