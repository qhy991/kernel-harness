#!/usr/bin/env python3
"""Prove the round-3 graph-only selection contract for one variant and bucket.

Round-3 promotion drops the eager-containing speedup gate and replaces it with a
stock-fallback requirement, so that requirement needs direct evidence rather
than an inference from timings:

* under the production default the eager containing region must issue **zero**
  provider launches and must return stock's result;
* under CUDA graph capture the same region must select the candidate;
* setting ``SGLANG_GLM52_FLASHMLA_GRAPH_ONLY=0`` must restore eager selection,
  which is what makes the correctness runs meaningful.

The provider's launch counter increments in host code, which runs at capture and
not at replay, so selection under graph is established at capture time. The
captured device node identity is established separately by ``trace_chain.py``.
"""

from __future__ import annotations

import argparse
import json
import os
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
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite evidence: {output}")

    runtime = Runtime(get_workload(f"dsa_flashmla_kv_decode_m{args.m}"))
    torch = runtime.torch
    try:
        inputs = runtime.build_inputs()
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

        config.load_manifest.cache_clear()
        hotspot_provider._reset_hotspot_provider_for_tests()
        set_forward_mode(ForwardMode.DECODE, args.m)
        hotspot_provider.initialize_hotspot_provider(gpu_id=0)
        state = hotspot_provider.provider_state()
        provider_module = sys.modules[state["module_name"]]

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

        def containing(use_candidate: bool):
            return backend._forward_flashmla_kv(
                q_all=inputs["q"].view(args.m, 64, 576),
                kv_cache=inputs["kv_cache"],
                v_head_dim=512,
                sm_scale=0.0625,
                layer=layer,
                metadata=metadata,
                page_table_1=inputs["indices"].squeeze(1),
                use_glm52_hotspot=use_candidate,
            )

        def launches() -> int:
            return int(provider_module._EXTENSION.launch_count())

        def delta(fn):
            torch.cuda.synchronize(runtime.device)
            before = launches()
            result = fn()
            torch.cuda.synchronize(runtime.device)
            return launches() - before, result

        for _ in range(5):
            containing(False)
            containing(True)
        torch.cuda.synchronize(runtime.device)

        # 1. Production default: eager must not select the provider at all.
        os.environ.pop("SGLANG_GLM52_FLASHMLA_GRAPH_ONLY", None)
        config.load_manifest.cache_clear()
        eager_default_delta, eager_default_out = delta(lambda: containing(True))
        stock_out = containing(False)
        torch.cuda.synchronize(runtime.device)
        eager_matches_stock = bool(torch.equal(eager_default_out, stock_out))

        # 2. Under capture the same region must select the candidate.
        def capture_candidate():
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = containing(True)
            return graph, captured

        capture_delta, (graph, captured) = delta(capture_candidate)
        replay_delta, _ = delta(graph.replay)

        # 3. The diagnostic escape hatch must restore eager selection, which is
        #    what makes the eager containing correctness comparison real.
        os.environ["SGLANG_GLM52_FLASHMLA_GRAPH_ONLY"] = "0"
        config.load_manifest.cache_clear()
        eager_forced_delta, _ = delta(lambda: containing(True))
        os.environ.pop("SGLANG_GLM52_FLASHMLA_GRAPH_ONLY", None)

        del graph, captured

        gates = {
            "eager_default_is_stock_fallback": eager_default_delta == 0,
            "eager_default_returns_stock_values": eager_matches_stock,
            "graph_capture_selects_candidate": capture_delta > 0,
            "replay_does_not_reenter_host_launch": replay_delta == 0,
            "diagnostic_env_restores_eager_selection": eager_forced_delta > 0,
        }
        evidence = {
            "schema_version": 1,
            "stage": "graph_only_selection_contract",
            "m": args.m,
            "variant": provider_module.PROVIDER_INFO["variant"],
            "build_id": provider_module.PROVIDER_INFO["build_id"],
            "measured": {
                "eager_default_provider_launches": eager_default_delta,
                "graph_capture_provider_launches": capture_delta,
                "graph_replay_host_launches": replay_delta,
                "eager_forced_provider_launches": eager_forced_delta,
                "eager_default_output_bitwise_equals_stock": eager_matches_stock,
            },
            "gates": gates,
            "all_gates_pass": all(gates.values()),
            "gpu": {
                "physical_index": int(os.environ["GLM52_PHYSICAL_GPU"]),
                "uuid": os.environ["GLM52_PHYSICAL_GPU_UUID"],
            },
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"m": args.m, **gates,
                          "all_gates_pass": evidence["all_gates_pass"]},
                         indent=2, sort_keys=True))
        return 0 if evidence["all_gates_pass"] else 1
    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
