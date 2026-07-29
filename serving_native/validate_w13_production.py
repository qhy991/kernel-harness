#!/usr/bin/env python3
"""Trace the opt-in W13 selector through the real SGLang wrapper on a lease."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
SGLANG_ROOT = REPO_ROOT.parent / "sglang"
for path in (REPO_ROOT, SGLANG_ROOT / "python"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


class _DecodeMode:
    @staticmethod
    def is_decode() -> bool:
        return True


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _production_counts(torch, assignments: int, seed: int, device):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    expert_ids = torch.randint(
        32,
        (assignments,),
        dtype=torch.int64,
        generator=generator,
    )
    return torch.bincount(expert_ids, minlength=32).to(
        dtype=torch.int32,
        device=device,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--variant",
        choices=("bm32_2sm", "bm32_1sm"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = args.manifest.expanduser().resolve()
    os.environ["SGLANG_GLM52_OPT"] = "0"
    os.environ["SGLANG_GLM52_W13_DECODE_VARIANT"] = args.variant
    os.environ["SGLANG_GLM52_W13_DECODE_MANIFEST"] = str(manifest)
    os.environ["SGLANG_JIT_DEEPGEMM_PRECOMPILE"] = "0"
    os.environ["SGLANG_DEEPGEMM_PDL"] = "1"
    os.environ["DG_JIT_USE_NVRTC"] = "0"
    os.environ["SGL_DG_USE_NVRTC"] = "0"

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("production W13 trace requires a leased CUDA device")
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    from sglang.srt.layers.deep_gemm_wrapper import entrypoint
    from sglang.srt.layers.glm52_opt import w13_decode
    from sglang.srt.layers.glm52_opt.w13_context import (
        get_w13_decode_forward_marker,
        w13_decode_forward_scope,
    )

    entrypoint.update_deep_gemm_config(
        0,
        SimpleNamespace(chunked_prefill_size=8192, base_gpu_id=0),
    )
    state = w13_decode.dispatch_state()
    if not state["enabled"] or state["variant"] != args.variant:
        raise RuntimeError(f"production W13 selector failed to initialize: {state}")

    tensors = w13_decode._allocate_warm_inputs(device)
    candidate_module = w13_decode._STATE.candidate_module
    if candidate_module is None:
        raise RuntimeError("production W13 candidate module is absent")
    import deep_gemm

    candidate_original = candidate_module.fp8_m_grouped_gemm_nt_masked
    stock_original = deep_gemm.fp8_m_grouped_gemm_nt_masked
    candidate_calls = []
    stock_calls = []

    def candidate_trace(*call_args, **call_kwargs):
        candidate_calls.append(
            {
                "stream": int(torch.cuda.current_stream(device).cuda_stream),
                "expected_m": int(call_args[4]),
                "compiled_dims": call_kwargs.get("compiled_dims"),
                "disable_ue8m0_cast": call_kwargs.get("disable_ue8m0_cast"),
                "w13_config": list(call_kwargs.get("w13_config", ())),
            }
        )
        return candidate_original(*call_args, **call_kwargs)

    def stock_trace(*call_args, **call_kwargs):
        stock_calls.append(
            {
                "stream": int(torch.cuda.current_stream(device).cuda_stream),
                "expected_m": int(call_args[4]),
                "kwargs": sorted(call_kwargs),
            }
        )
        return stock_original(*call_args, **call_kwargs)

    candidate_module.fp8_m_grouped_gemm_nt_masked = candidate_trace
    deep_gemm.fp8_m_grouped_gemm_nt_masked = stock_trace
    stream = torch.cuda.Stream(device=device)
    forward_batch = SimpleNamespace(forward_mode=_DecodeMode())
    selected = []
    try:
        for expected_m, token_bucket, graph_capture, assignments in (
            (4, 16, False, 128),
            (5, 16, True, 128),
            (8, 32, False, 256),
            (9, 32, True, 256),
        ):
            tensors["masked_m"].copy_(
                _production_counts(
                    torch,
                    assignments,
                    2026072400 + expected_m,
                    device,
                )
            )
            tensors["out"].fill_(-57344.0)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                with w13_decode_forward_scope(
                    forward_batch,
                    token_bucket,
                    graph_capture=graph_capture,
                ):
                    returned = entrypoint.grouped_gemm_nt_f8f8bf16_masked(
                        (tensors["a"], tensors["a_scale"]),
                        (tensors["b"], tensors["b_scale"]),
                        tensors["out"],
                        tensors["masked_m"],
                        expected_m,
                    )
                    marker = get_w13_decode_forward_marker()
                    if marker is None:
                        raise AssertionError("production W13 marker was not visible")
                if get_w13_decode_forward_marker() is not None:
                    raise AssertionError("production W13 marker leaked after forward")
            stream.synchronize()
            if returned is not None:
                raise AssertionError("selected production W13 return was not None")
            selected.append(
                {
                    "expected_m": expected_m,
                    "token_bucket": token_bucket,
                    "graph_capture_marker": graph_capture,
                    "return": None,
                    "candidate_calls_after": len(candidate_calls),
                    "stock_calls_after": len(stock_calls),
                }
            )

        stock_before_error = len(stock_calls)

        def candidate_error(*_args, **_kwargs):
            raise RuntimeError("injected selected-candidate failure")

        candidate_module.fp8_m_grouped_gemm_nt_masked = candidate_error
        error_propagated = False
        with w13_decode_forward_scope(
            forward_batch,
            16,
            graph_capture=False,
        ):
            try:
                entrypoint.grouped_gemm_nt_f8f8bf16_masked(
                    (tensors["a"], tensors["a_scale"]),
                    (tensors["b"], tensors["b_scale"]),
                    tensors["out"],
                    tensors["masked_m"],
                    4,
                )
            except RuntimeError as exc:
                error_propagated = str(exc) == "injected selected-candidate failure"
        if not error_propagated or len(stock_calls) != stock_before_error:
            raise AssertionError("selected candidate failure retried stock")
        candidate_module.fp8_m_grouped_gemm_nt_masked = candidate_trace

        # With no private production marker, the exact tensor shape must take
        # the ordinary stock path before any candidate launch.
        candidates_before_fallback = len(candidate_calls)
        returned = entrypoint.grouped_gemm_nt_f8f8bf16_masked(
            (tensors["a"], tensors["a_scale"]),
            (tensors["b"], tensors["b_scale"]),
            tensors["out"],
            tensors["masked_m"],
            4,
        )
        torch.cuda.synchronize(device)
        if (
            returned is not None
            or len(candidate_calls) != candidates_before_fallback
            or len(stock_calls) != stock_before_error + 1
        ):
            raise AssertionError("marker-free exact W13 call did not select stock")
    finally:
        candidate_module.fp8_m_grouped_gemm_nt_masked = candidate_original
        deep_gemm.fp8_m_grouped_gemm_nt_masked = stock_original

    result = {
        "schema_version": 1,
        "kind": "glm52_w13_production_wrapper_trace",
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "variant": args.variant,
        "dispatch_state": state,
        "nondefault_stream": int(stream.cuda_stream),
        "selected_calls": selected,
        "candidate_low_level_calls": candidate_calls,
        "stock_low_level_calls": stock_calls,
        "selected_call_count": len(candidate_calls),
        "selected_error_propagated_without_stock_retry": True,
        "marker_free_exact_call_selected_stock": True,
        "marker_reset_after_every_forward": True,
        "all_selected_returns_exact_none": True,
    }
    if len(candidate_calls) != 4:
        raise AssertionError(
            f"expected four selected calls, got {len(candidate_calls)}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
