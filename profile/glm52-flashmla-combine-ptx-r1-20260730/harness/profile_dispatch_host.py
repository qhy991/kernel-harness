#!/usr/bin/env python3
"""Attribute the API-v1 containing-eager host overhead with cProfile.

The identity control shows a constant +16.5..17.5 us per call on the containing
eager lane with a kernel that is source-identical to stock, so the cost is in
the SGLang dispatch path rather than in the device kernel. This script names the
responsible functions instead of guessing.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import os
import pstats
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, choices=(16, 32), default=16)
    parser.add_argument("--calls", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
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
        from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
        from sglang.srt.layers.glm52_opt import config, hotspot_provider
        from sglang.srt.layers.glm52_opt.context import set_forward_mode
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        config.load_manifest.cache_clear()
        hotspot_provider._reset_hotspot_provider_for_tests()
        set_forward_mode(ForwardMode.DECODE, args.m)
        hotspot_provider.initialize_hotspot_provider(gpu_id=0)

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
        q_all = inputs["q"].view(args.m, 64, 576)
        page_table_1 = inputs["indices"].squeeze(1)

        def containing(use_candidate: bool):
            return backend._forward_flashmla_kv(
                q_all=q_all,
                kv_cache=inputs["kv_cache"],
                v_head_dim=512,
                sm_scale=0.0625,
                layer=layer,
                metadata=metadata,
                page_table_1=page_table_1,
                use_glm52_hotspot=use_candidate,
            )

        for _ in range(50):
            containing(True)
            containing(False)
        torch.cuda.synchronize(runtime.device)

        reports = {}
        for label, use_candidate in (("candidate", True), ("stock", False)):
            profiler = cProfile.Profile()
            profiler.enable()
            for _ in range(args.calls):
                containing(use_candidate)
            profiler.disable()
            torch.cuda.synchronize(runtime.device)
            stream = io.StringIO()
            stats = pstats.Stats(profiler, stream=stream)
            stats.sort_stats("tottime").print_stats(28)
            reports[label] = stream.getvalue()

        evidence = {
            "schema_version": 1,
            "purpose": (
                "attribute the containing-eager host overhead of the API-v1 "
                "hotspot dispatch path"
            ),
            "m": args.m,
            "calls_per_arm": args.calls,
            "cprofile_tottime_top": reports,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
        print(reports["candidate"][:5000])
        return 0
    finally:
        from sglang.srt.layers.glm52_opt.context import set_forward_mode

        set_forward_mode(None)
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
