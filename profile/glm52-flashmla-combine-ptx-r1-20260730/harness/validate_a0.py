#!/usr/bin/env python3
"""Validate one exact FlashMLA provider at leaf and containing DSA boundaries."""

from __future__ import annotations

import argparse
import hashlib
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
    REPO_ROOT
    / "serving_native/candidates/flashmla_combine_decode_provider.py"
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_contract(tensor) -> dict[str, object]:
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "storage_offset": tensor.storage_offset(),
        "data_ptr": tensor.data_ptr(),
        "is_contiguous": tensor.is_contiguous(),
    }


def compare(torch, reference, candidate, name: str) -> dict[str, object]:
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        raise AssertionError(
            f"{name} contract mismatch: "
            f"{reference.shape}/{reference.dtype} != "
            f"{candidate.shape}/{candidate.dtype}"
        )
    ref_f = reference.float()
    cand_f = candidate.float()
    ref_bad = ~torch.isfinite(ref_f)
    cand_bad = ~torch.isfinite(cand_f)
    if not torch.equal(ref_bad, cand_bad):
        raise AssertionError(f"{name} anomaly positions differ")
    torch.testing.assert_close(
        candidate,
        reference,
        rtol=2e-2,
        atol=2e-2,
        equal_nan=False,
    )
    finite = ~ref_bad
    max_abs = (
        float((ref_f[finite] - cand_f[finite]).abs().max().item())
        if finite.any()
        else 0.0
    )
    return {
        "rtol": 2e-2,
        "atol": 2e-2,
        "max_abs": max_abs,
        "anomaly_positions_match": True,
        "exact": bool(torch.equal(reference, candidate)),
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
            raise AssertionError(f"wrong page guard fixture: {inputs['kv_cache'].shape}")
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
        from sgl_kernel.flash_mla import flash_mla_with_kvcache
        from sglang.srt.layers.attention.dsa_backend import (
            DeepseekSparseAttnBackend,
        )
        from sglang.srt.layers.glm52_opt import config, dispatch, hotspot_provider
        from sglang.srt.layers.glm52_opt.context import set_forward_mode
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        config.load_manifest.cache_clear()
        hotspot_provider._reset_hotspot_provider_for_tests()
        dispatch._HIT_COUNTS.clear()
        dispatch._MISS_COUNTS.clear()
        set_forward_mode(ForwardMode.DECODE, args.m)
        hotspot_provider.initialize_hotspot_provider(gpu_id=0)
        provider_state = hotspot_provider.provider_state()
        provider_module = sys.modules[provider_state["module_name"]]

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

        def candidate_leaf():
            return hotspot_provider.run_flashmla_sparse_decode(**callback_kwargs)

        # Warm exact shapes before any graph capture.
        for _ in range(3):
            stock_leaf()
            candidate_leaf()
        torch.cuda.synchronize(runtime.device)

        workspace = provider_module._WORKSPACES[args.m]
        workspace[0].fill_(float("nan"))
        workspace[1].fill_(float("nan"))
        q_before = inputs["q"].clone()
        indices_before = inputs["indices"].clone()
        kv_before = inputs["kv_cache"].clone()
        stock_out, stock_lse = stock_leaf()
        candidate_out, candidate_lse = candidate_leaf()
        torch.cuda.synchronize(runtime.device)
        eager_out = compare(torch, stock_out, candidate_out, "eager.output")
        eager_lse = compare(torch, stock_lse, candidate_lse, "eager.lse")
        if torch.isnan(candidate_out).any() or torch.isnan(candidate_lse).any():
            raise AssertionError("poison remained in candidate output or LSE")
        if not torch.equal(inputs["q"], q_before):
            raise AssertionError("candidate mutated q")
        if not torch.equal(inputs["indices"], indices_before):
            raise AssertionError("candidate mutated indices")
        if not torch.equal(inputs["kv_cache"], kv_before):
            raise AssertionError("candidate mutated kv")

        # Both paths must honor the same non-default current stream.
        side_stream = torch.cuda.Stream(device=runtime.device)
        side_stream.wait_stream(torch.cuda.current_stream(runtime.device))
        with torch.cuda.stream(side_stream):
            stream_stock_out, stream_stock_lse = stock_leaf()
            stream_candidate_out, stream_candidate_lse = candidate_leaf()
            stream_done = torch.cuda.Event()
            stream_done.record()
        torch.cuda.current_stream(runtime.device).wait_event(stream_done)
        torch.cuda.synchronize(runtime.device)
        stream_out = compare(
            torch, stream_stock_out, stream_candidate_out, "stream.output"
        )
        stream_lse = compare(
            torch, stream_stock_lse, stream_candidate_lse, "stream.lse"
        )

        # Capture stock and candidate independently, poison before replay, then
        # mutate Q and sparse indices in place to prove fresh graph execution.
        capture_stream = torch.cuda.Stream(device=runtime.device)
        capture_stream.wait_stream(torch.cuda.current_stream(runtime.device))
        with torch.cuda.stream(capture_stream):
            stock_leaf()
            candidate_leaf()
        torch.cuda.current_stream(runtime.device).wait_stream(capture_stream)
        torch.cuda.synchronize(runtime.device)

        stock_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(stock_graph):
            graph_stock_out, graph_stock_lse = stock_leaf()
        candidate_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(candidate_graph):
            graph_candidate_out, graph_candidate_lse = candidate_leaf()

        graph_stock_out.fill_(float("nan"))
        graph_stock_lse.fill_(float("nan"))
        graph_candidate_out.fill_(float("nan"))
        graph_candidate_lse.fill_(float("nan"))
        stock_graph.replay()
        candidate_graph.replay()
        torch.cuda.synchronize(runtime.device)
        graph_out = compare(
            torch, graph_stock_out, graph_candidate_out, "graph.output"
        )
        graph_lse = compare(
            torch, graph_stock_lse, graph_candidate_lse, "graph.lse"
        )
        graph_first_out = graph_candidate_out.clone()
        graph_first_lse = graph_candidate_lse.clone()

        bases = 64 + torch.arange(
            args.m, dtype=torch.int32, device=runtime.device
        ) * 8192
        local_indices = inputs["indices"] - bases[:, None, None]
        inputs["q"].copy_(q_before.mul(-0.75))
        inputs["indices"].copy_(
            bases[:, None, None] + (local_indices + 7919) % 8192
        )
        graph_stock_out.fill_(float("nan"))
        graph_stock_lse.fill_(float("nan"))
        graph_candidate_out.fill_(float("nan"))
        graph_candidate_lse.fill_(float("nan"))
        stock_graph.replay()
        candidate_graph.replay()
        torch.cuda.synchronize(runtime.device)
        graph_mutated_out = compare(
            torch,
            graph_stock_out,
            graph_candidate_out,
            "graph_mutated.output",
        )
        graph_mutated_lse = compare(
            torch,
            graph_stock_lse,
            graph_candidate_lse,
            "graph_mutated.lse",
        )
        mutation_change = float(
            (graph_candidate_out.float() - graph_first_out.float())
            .abs()
            .max()
            .item()
        )
        if mutation_change <= 1e-3:
            raise AssertionError("graph replay remained stale after input mutation")
        deterministic_out = graph_candidate_out.clone()
        deterministic_lse = graph_candidate_lse.clone()
        candidate_graph.replay()
        torch.cuda.synchronize(runtime.device)
        if not torch.equal(graph_candidate_out, deterministic_out):
            raise AssertionError("candidate graph output is nondeterministic")
        if not torch.equal(graph_candidate_lse, deterministic_lse):
            raise AssertionError("candidate graph LSE is nondeterministic")

        inputs["q"].copy_(q_before)
        inputs["indices"].copy_(indices_before)

        # Validate the real SGLang API-v1 dispatch once, then prove an arbitrary
        # page count misses before the selected provider can launch.
        before_dispatch = provider_module._EXTENSION.launch_count()
        dispatched = dispatch.try_dispatch_flashmla_sparse_decode(**callback_kwargs)
        torch.cuda.synchronize(runtime.device)
        after_dispatch = provider_module._EXTENSION.launch_count()
        if dispatched is None or after_dispatch - before_dispatch != 1:
            raise AssertionError("selected provider did not launch exactly once")
        compare(torch, stock_leaf()[0], dispatched, "dispatch.output")

        bad_kwargs = dict(callback_kwargs)
        bad_kwargs["k_cache"] = inputs["kv_cache"][:-1]
        before_bad_page = provider_module._EXTENSION.launch_count()
        if dispatch.try_dispatch_flashmla_sparse_decode(**bad_kwargs) is not None:
            raise AssertionError("wrong-page ABI unexpectedly dispatched")
        after_bad_page = provider_module._EXTENSION.launch_count()
        if after_bad_page != before_bad_page:
            raise AssertionError("wrong-page ABI reached the provider")

        # Call the exact containing SGLang method with a minimal instance whose
        # fields match the production flashmla_kv path.
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

        containing_stock = containing(False)
        before_containing = provider_module._EXTENSION.launch_count()
        containing_candidate = containing(True)
        torch.cuda.synchronize(runtime.device)
        after_containing = provider_module._EXTENSION.launch_count()
        if after_containing - before_containing != 1:
            raise AssertionError("containing DSA provider launch count was not one")
        containing_eager = compare(
            torch,
            containing_stock,
            containing_candidate,
            "containing_eager.output",
        )

        # Independently captured complete containing regions.
        containing_stock_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(containing_stock_graph):
            containing_graph_stock = containing(False)
        containing_candidate_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(containing_candidate_graph):
            containing_graph_candidate = containing(True)
        containing_graph_stock.fill_(float("nan"))
        containing_graph_candidate.fill_(float("nan"))
        containing_stock_graph.replay()
        containing_candidate_graph.replay()
        torch.cuda.synchronize(runtime.device)
        containing_graph = compare(
            torch,
            containing_graph_stock,
            containing_graph_candidate,
            "containing_graph.output",
        )

        extension_path = Path(provider_module._EXTENSION.__file__).resolve()
        evidence = {
            "schema_version": 1,
            "stage": f"correctness_{provider_module.PROVIDER_INFO['variant']}",
            "task": task,
            "m": args.m,
            "gpu": {
                "physical_index": int(os.environ["GLM52_PHYSICAL_GPU"]),
                "uuid": os.environ["GLM52_PHYSICAL_GPU_UUID"],
                "logical_index": 0,
                "name": torch.cuda.get_device_properties(0).name,
                "compute_capability": list(torch.cuda.get_device_capability(0)),
            },
            "runtime": runtime.runtime_evidence(inputs),
            "contracts": {
                "stock_output": tensor_contract(stock_out),
                "candidate_output": tensor_contract(candidate_out),
                "stock_lse": tensor_contract(stock_lse),
                "candidate_lse": tensor_contract(candidate_lse),
                "output_lse_alias": candidate_out.data_ptr()
                == candidate_lse.data_ptr(),
                "inputs_immutable": True,
                "poison_overwritten": True,
                "exact_pages": expected_pages,
                "num_splits": actual_splits,
            },
            "correctness": {
                "eager_output": eager_out,
                "eager_lse": eager_lse,
                "nondefault_stream_output": stream_out,
                "nondefault_stream_lse": stream_lse,
                "graph_output": graph_out,
                "graph_lse": graph_lse,
                "graph_mutated_output": graph_mutated_out,
                "graph_mutated_lse": graph_mutated_lse,
                "graph_mutation_change_max_abs": mutation_change,
                "graph_replay_deterministic_exact": True,
                "containing_eager": containing_eager,
                "containing_graph": containing_graph,
            },
            "dispatch": {
                "selected_launch_delta": after_dispatch - before_dispatch,
                "wrong_page_launch_delta": after_bad_page - before_bad_page,
                "hits": dict(dispatch._HIT_COUNTS),
                "misses": dict(dispatch._MISS_COUNTS),
                "no_fallback_after_selection": True,
            },
            "provider": provider_module.candidate_evidence(),
            "extension": {
                "path": str(extension_path),
                "sha256": sha256(extension_path),
                "size_bytes": extension_path.stat().st_size,
            },
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
        print(json.dumps(evidence, indent=2, sort_keys=True))
        return 0
    finally:
        from sglang.srt.layers.glm52_opt.context import set_forward_mode

        set_forward_mode(None)
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
