#!/usr/bin/env python3
"""Minimal NCU target: repeated exact-ABI leaf calls, nothing else timed.

Used to test the plan's stated critical-path mechanism (long-scoreboard and
barrier stalls in the main kernel). The identity variant is source-identical to
stock but is built with -lineinfo, so NCU can attribute stalls to source lines.
"""

from __future__ import annotations

import argparse
import os
import sys
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, choices=(16, 32), required=True)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument(
        "--arm",
        choices=("stock", "provider"),
        default="provider",
        help="stock uses installed sgl_kernel; provider uses GLM52_FLASHMLA_COMBINE_VARIANT",
    )
    args = parser.parse_args()

    runtime = Runtime(get_workload(f"dsa_flashmla_kv_decode_m{args.m}"))
    torch = runtime.torch
    try:
        inputs = runtime.build_inputs()
        if args.arm == "stock":
            from sgl_kernel.flash_mla import flash_mla_with_kvcache

            def call():
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

        else:
            import importlib.util

            spec = importlib.util.spec_from_file_location("_glm52_ncu", PROVIDER)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.initialize(gpu_id=0)
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

            def call():
                return module.flashmla_sparse_decode(**kwargs)

        for _ in range(args.iters):
            call()
        torch.cuda.synchronize(runtime.device)
        print(f"completed {args.iters} {args.arm} leaf calls at M{args.m}")
        return 0
    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
