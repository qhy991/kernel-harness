#!/usr/bin/env python3
"""Emit a minimal Nsys-visible stock/candidate main-plus-combine chain."""

from __future__ import annotations

import argparse
import json
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, choices=(16, 32), required=True)
    parser.add_argument("--graph-replays", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runtime = Runtime(get_workload(f"dsa_flashmla_kv_decode_m{args.m}"))
    torch = runtime.torch
    try:
        inputs = runtime.build_inputs()
        from sgl_kernel.flash_mla import flash_mla_with_kvcache

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

        def stock():
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

        def candidate():
            return hotspot_provider.run_flashmla_sparse_decode(**callback_kwargs)

        for _ in range(5):
            stock()
            candidate()
        torch.cuda.synchronize(runtime.device)

        stock_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(stock_graph):
            stock_graph_result = stock()
        candidate_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(candidate_graph):
            candidate_graph_result = candidate()
        torch.cuda.synchronize(runtime.device)

        with torch.cuda.nvtx.range(f"glm52_m{args.m}_stock_eager"):
            stock_eager = stock()
            torch.cuda.synchronize(runtime.device)
        with torch.cuda.nvtx.range(f"glm52_m{args.m}_candidate_eager"):
            candidate_eager = candidate()
            torch.cuda.synchronize(runtime.device)
        torch.testing.assert_close(
            candidate_eager[0], stock_eager[0], rtol=2e-2, atol=2e-2
        )
        torch.testing.assert_close(
            candidate_eager[1], stock_eager[1], rtol=2e-2, atol=2e-2
        )

        with torch.cuda.nvtx.range(f"glm52_m{args.m}_stock_graph"):
            for _ in range(args.graph_replays):
                stock_graph.replay()
            torch.cuda.synchronize(runtime.device)
        with torch.cuda.nvtx.range(f"glm52_m{args.m}_candidate_graph"):
            for _ in range(args.graph_replays):
                candidate_graph.replay()
            torch.cuda.synchronize(runtime.device)
        torch.testing.assert_close(
            candidate_graph_result[0],
            stock_graph_result[0],
            rtol=2e-2,
            atol=2e-2,
        )
        torch.testing.assert_close(
            candidate_graph_result[1],
            stock_graph_result[1],
            rtol=2e-2,
            atol=2e-2,
        )

        print(
            json.dumps(
                {
                    "m": args.m,
                    "graph_replays": args.graph_replays,
                    "gpu": {
                        "physical_index": int(os.environ["GLM52_PHYSICAL_GPU"]),
                        "uuid": os.environ["GLM52_PHYSICAL_GPU_UUID"],
                        "name": torch.cuda.get_device_properties(0).name,
                    },
                    "provider": provider_module.candidate_evidence(),
                    "expected_chain": ["prefixed V32 main", "prefixed BF16 combine variant"],
                    "eager_correct": True,
                    "graph_correct": True,
                },
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
