#!/usr/bin/env python3
"""Trace current API-v1 W13 selection, graphs, and fail-closed fallback."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
SGLANG_ROOT = REPO_ROOT.parent / "sglang"
for path in (REPO_ROOT, SGLANG_ROOT / "python"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from serving_native.contract_v2 import inspect_cuda_graph


VARIANT_PROVIDERS = {
    "bm16_2sm": "provider_bm16_2sm.py",
    "bm16_1sm": "provider_bm16_1sm.py",
}
VARIANT_SUFFIX = {
    "bm16_2sm": "bm16_2sm",
    "bm16_1sm": "bm16_1sm",
}
OUTPUT_POISON = -57344.0


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
    counts_cpu = tuple(
        int(value)
        for value in torch.bincount(expert_ids, minlength=32).tolist()
    )
    return counts_cpu, torch.tensor(
        counts_cpu,
        dtype=torch.int32,
        device=device,
    )


def _allocate_exact(torch, device) -> dict[str, Any]:
    def empty_strided(shape, stride, dtype, fill):
        tensor = torch.empty_strided(shape, stride, device=device, dtype=dtype)
        tensor.fill_(fill)
        return tensor

    return {
        "a": empty_strided(
            (32, 1024, 6144),
            (6291456, 6144, 1),
            torch.float8_e4m3fn,
            1.0,
        ),
        "a_scale": empty_strided(
            (32, 1024, 12),
            (12288, 1, 1024),
            torch.int32,
            0x7F7F7F7F,
        ),
        "b": empty_strided(
            (32, 4096, 6144),
            (25165824, 6144, 1),
            torch.float8_e4m3fn,
            1.0,
        ),
        "b_scale": empty_strided(
            (32, 4096, 12),
            (49152, 1, 4096),
            torch.int32,
            0x7F7F7F7F,
        ),
        "out": empty_strided(
            (32, 1024, 4096),
            (4194304, 4096, 1),
            torch.bfloat16,
            OUTPUT_POISON,
        ),
        "masked_m": torch.zeros((32,), device=device, dtype=torch.int32),
    }


def _valid_snapshot(torch, out, counts) -> list[Any]:
    return [
        out[expert, :count].clone()
        for expert, count in enumerate(counts)
        if int(count)
    ]


def _assert_valid_written(out, counts) -> None:
    for expert, count in enumerate(counts):
        if count and bool(out[expert, :count].eq(OUTPUT_POISON).any().item()):
            raise AssertionError(f"selected graph left poison in expert {expert}")


def _call_entrypoint(
    entrypoint,
    op_context,
    set_forward_mode,
    forward_mode,
    token_bucket: int,
    tensors: dict[str, Any],
    expected_m: int,
    **kwargs,
):
    set_forward_mode(forward_mode, token_bucket)
    with op_context("moe_gate_proj"):
        return entrypoint.grouped_gemm_nt_f8f8bf16_masked(
            (tensors["a"], tensors["a_scale"]),
            (tensors["b"], tensors["b_scale"]),
            tensors["out"],
            tensors["masked_m"],
            expected_m,
            **kwargs,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--variant",
        choices=tuple(VARIANT_PROVIDERS),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--provider-root",
        type=Path,
        default=SGLANG_ROOT / "third_party" / "deepgemm_w13",
    )
    args = parser.parse_args()

    manifest = args.manifest.expanduser().resolve()
    provider = (
        args.provider_root.expanduser().resolve()
        / VARIANT_PROVIDERS[args.variant]
    )
    os.environ.update(
        {
            "SGLANG_GLM52_OPT": "1",
            "SGLANG_GLM52_OPT_PROFILE": "hotspot_candidates",
            "SGLANG_GLM52_OPT_OPS": "moe_w13",
            "SGLANG_GLM52_OPT_M_BUCKETS": "moe_gate_proj:16|32",
            "SGLANG_GLM52_HOTSPOT_MODULE": str(provider),
            "SGLANG_GLM52_OPT_HIT_FILE": (
                "/home/qinhaiyan/glm52-hotspot-goal-runs/cache/"
                f"moe_w13_decode/hits/production_trace_{os.getpid()}.json"
            ),
            "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "0",
            "SGLANG_DEEPGEMM_PDL": "1",
            "DG_JIT_USE_NVRTC": "0",
            "SGL_DG_USE_NVRTC": "0",
        }
    )

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("production W13 trace requires a leased CUDA device")
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)

    import deep_gemm
    from sglang.srt.layers.deep_gemm_wrapper import entrypoint
    from sglang.srt.layers.glm52_opt import hotspot_provider
    from sglang.srt.layers.glm52_opt.context import op_context, set_forward_mode
    from sglang.srt.layers.glm52_opt.dispatch import _HIT_COUNTS, _MISS_COUNTS
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    deep_gemm.set_pdl(True)
    deep_gemm.set_num_sms(148)
    deep_gemm.set_tc_util(100)
    hotspot_provider._reset_hotspot_provider_for_tests()
    hotspot_provider.initialize_hotspot_provider(gpu_id=0)
    provider_state = hotspot_provider.provider_state()
    provider_module = sys.modules[provider_state["module_name"]]
    provider_object = provider_module._PROVIDER
    candidate_original = provider_object._launcher
    if candidate_original is None:
        raise RuntimeError("API-v1 provider has no bound candidate launcher")
    stock_original = deep_gemm.fp8_m_grouped_gemm_nt_masked

    candidate_calls: list[dict[str, Any]] = []
    stock_calls: list[dict[str, Any]] = []
    execute_stock = False

    def candidate_trace(*call_args, **call_kwargs):
        record = {
            "stream": int(torch.cuda.current_stream(device).cuda_stream),
            "expected_m": int(call_args[4]),
            "compiled_dims": call_kwargs.get("compiled_dims"),
            "disable_ue8m0_cast": call_kwargs.get("disable_ue8m0_cast"),
            "w13_config": list(call_kwargs.get("w13_config", ())),
        }
        candidate_calls.append(record)
        returned = candidate_original(*call_args, **call_kwargs)
        if returned is not None:
            raise AssertionError("API-v1 provider low-level return is not None")
        return returned

    def stock_trace(*call_args, **call_kwargs):
        stock_calls.append(
            {
                "stream": int(torch.cuda.current_stream(device).cuda_stream),
                "expected_m": int(call_args[4]),
                "kwargs": sorted(call_kwargs),
                "executed": execute_stock,
            }
        )
        if execute_stock:
            return stock_original(*call_args, **call_kwargs)
        return None

    provider_object._launcher = candidate_trace
    deep_gemm.fp8_m_grouped_gemm_nt_masked = stock_trace
    tensors = _allocate_exact(torch, device)
    stream = torch.cuda.Stream(device=device)
    selected: list[dict[str, Any]] = []
    graphs = []
    try:
        for expected_m, token_bucket, assignments in (
            (4, 16, 128),
            (5, 16, 128),
            (8, 32, 256),
            (9, 32, 256),
        ):
            _, initial_counts_device = _production_counts(
                torch,
                assignments,
                2026072900 + expected_m,
                device,
            )
            tensors["masked_m"].copy_(initial_counts_device)
            tensors["out"].fill_(OUTPUT_POISON)
            stream.wait_stream(torch.cuda.current_stream(device))
            before_candidate = len(candidate_calls)
            before_stock = len(stock_calls)
            with torch.cuda.stream(stream):
                returned = _call_entrypoint(
                    entrypoint,
                    op_context,
                    set_forward_mode,
                    ForwardMode.DECODE,
                    token_bucket,
                    tensors,
                    expected_m,
                )
            stream.synchronize()
            if returned is not tensors["out"]:
                raise AssertionError("selected public wrapper did not return caller out")
            if (
                len(candidate_calls) != before_candidate + 1
                or len(stock_calls) != before_stock
            ):
                raise AssertionError("selected eager call was not exactly one candidate")

            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph, stream=stream):
                captured_return = _call_entrypoint(
                    entrypoint,
                    op_context,
                    set_forward_mode,
                    ForwardMode.DECODE,
                    token_bucket,
                    tensors,
                    expected_m,
                )
            if captured_return is not tensors["out"]:
                raise AssertionError("captured wrapper changed caller-output ownership")
            stream.synchronize()
            if (
                len(candidate_calls) != before_candidate + 2
                or len(stock_calls) != before_stock
            ):
                raise AssertionError("graph capture was not exactly one candidate call")

            replay_counts, replay_counts_device = _production_counts(
                torch,
                assignments,
                2026073000 + expected_m,
                device,
            )
            tensors["masked_m"].copy_(replay_counts_device)
            tensors["a"].fill_(-1.0)
            tensors["out"].fill_(OUTPUT_POISON)
            torch.cuda.synchronize(device)
            graph.replay()
            torch.cuda.synchronize(device)
            _assert_valid_written(tensors["out"], replay_counts)
            first = _valid_snapshot(torch, tensors["out"], replay_counts)
            tensors["out"].fill_(OUTPUT_POISON)
            graph.replay()
            torch.cuda.synchronize(device)
            _assert_valid_written(tensors["out"], replay_counts)
            second = _valid_snapshot(torch, tensors["out"], replay_counts)
            if len(first) != len(second) or any(
                not torch.equal(left, right)
                for left, right in zip(first, second)
            ):
                raise AssertionError("selected graph replay is not bitwise repeatable")

            inspected = inspect_cuda_graph(int(graph.raw_cuda_graph()))
            symbol = (
                "infini_kernel_glm52_moe_w13_decode_"
                f"em{expected_m}_{VARIANT_SUFFIX[args.variant]}"
            )
            if (
                inspected["node_count"] != 1
                or inspected["forbidden_nodes"]
                or inspected["kernel_identities"] != [symbol]
            ):
                raise AssertionError(
                    f"selected graph identity mismatch for {symbol}: {inspected}"
                )
            node = inspected["nodes"][0]
            expected_shared = 230188 if args.variant == "bm16_2sm" else 223020
            if (
                node.get("grid") != [148, 1, 1]
                or node.get("block") != [256, 1, 1]
                or node.get("shared_memory_bytes") != expected_shared
            ):
                raise AssertionError(
                    f"selected graph launch topology mismatch: {node}"
                )
            selected.append(
                {
                    "expected_m": expected_m,
                    "token_bucket": token_bucket,
                    "eager_public_return_is_out": True,
                    "candidate_calls_for_eager_and_capture": 2,
                    "graph_post_capture_mask_and_input_mutation": True,
                    "graph_output_poison_overwritten": True,
                    "graph_replay_bitwise_exact": True,
                    "graph": inspected,
                }
            )
            graphs.append(graph)
            tensors["a"].fill_(1.0)

        stock_before_error = len(stock_calls)
        calls_before_error = len(candidate_calls)

        def candidate_error(*_args, **_kwargs):
            raise RuntimeError("injected selected-candidate failure")

        provider_object._launcher = candidate_error
        _, failure_counts_device = _production_counts(
            torch, 128, 2026073199, device
        )
        tensors["masked_m"].copy_(failure_counts_device)
        error_propagated = False
        try:
            _call_entrypoint(
                entrypoint,
                op_context,
                set_forward_mode,
                ForwardMode.DECODE,
                16,
                tensors,
                4,
            )
        except RuntimeError as exc:
            error_propagated = str(exc) == "injected selected-candidate failure"
        if (
            not error_propagated
            or len(stock_calls) != stock_before_error
            or len(candidate_calls) != calls_before_error
        ):
            raise AssertionError("selected failure was hidden or retried as stock")
        provider_object._launcher = candidate_trace

        fallback_cases: list[dict[str, Any]] = []

        def expect_stock(
            name: str,
            *,
            token_bucket: int,
            forward_mode,
            op_name: str | None,
            expected_m: int,
            lhs=None,
            rhs=None,
            out=None,
            masked_m=None,
            kwargs=None,
        ) -> None:
            candidate_before = len(candidate_calls)
            stock_before = len(stock_calls)
            set_forward_mode(forward_mode, token_bucket)
            call_lhs = lhs or (tensors["a"], tensors["a_scale"])
            call_rhs = rhs or (tensors["b"], tensors["b_scale"])
            call_out = out if out is not None else tensors["out"]
            call_mask = masked_m if masked_m is not None else tensors["masked_m"]
            with op_context(op_name):
                entrypoint.grouped_gemm_nt_f8f8bf16_masked(
                    call_lhs,
                    call_rhs,
                    call_out,
                    call_mask,
                    expected_m,
                    **(kwargs or {}),
                )
            if (
                len(candidate_calls) != candidate_before
                or len(stock_calls) != stock_before + 1
            ):
                raise AssertionError(f"{name}: fallback occurred after candidate")
            fallback_cases.append(
                {
                    "name": name,
                    "candidate_calls": 0,
                    "stock_calls": 1,
                    "fallback_before_candidate_invocation": True,
                }
            )

        expect_stock(
            "missing_op_context",
            token_bucket=16,
            forward_mode=ForwardMode.DECODE,
            op_name=None,
            expected_m=4,
        )
        expect_stock(
            "unsupported_expected_m_for_m16",
            token_bucket=16,
            forward_mode=ForwardMode.DECODE,
            op_name="moe_gate_proj",
            expected_m=8,
        )
        expect_stock(
            "unsupported_forward_m",
            token_bucket=64,
            forward_mode=ForwardMode.DECODE,
            op_name="moe_gate_proj",
            expected_m=4,
        )
        expect_stock(
            "prefill_mode",
            token_bucket=16,
            forward_mode=ForwardMode.EXTEND,
            op_name="moe_gate_proj",
            expected_m=4,
        )
        expect_stock(
            "w2_context",
            token_bucket=16,
            forward_mode=ForwardMode.DECODE,
            op_name="moe_down_proj",
            expected_m=4,
        )
        expect_stock(
            "unsupported_group_topology",
            token_bucket=16,
            forward_mode=ForwardMode.DECODE,
            op_name="moe_gate_proj",
            expected_m=4,
            lhs=(tensors["a"][:31], tensors["a_scale"][:31]),
            rhs=(tensors["b"][:31], tensors["b_scale"][:31]),
            out=tensors["out"][:31],
            masked_m=tensors["masked_m"][:31],
        )
        expect_stock(
            "unsupported_scale_stride",
            token_bucket=16,
            forward_mode=ForwardMode.DECODE,
            op_name="moe_gate_proj",
            expected_m=4,
            lhs=(tensors["a"], tensors["a_scale"].contiguous()),
        )
        expect_stock(
            "recipe_a_present",
            token_bucket=16,
            forward_mode=ForwardMode.DECODE,
            op_name="moe_gate_proj",
            expected_m=4,
            kwargs={"recipe_a": (1, 128)},
        )
        expect_stock(
            "overlap_present",
            token_bucket=16,
            forward_mode=ForwardMode.DECODE,
            op_name="moe_gate_proj",
            expected_m=4,
            kwargs={"overlap_args": SimpleNamespace(num_sms=148, signal=None)},
        )
    finally:
        provider_object._launcher = candidate_original
        deep_gemm.fp8_m_grouped_gemm_nt_masked = stock_original
        set_forward_mode(None)

    expected_candidate_calls = 8
    if len(candidate_calls) != expected_candidate_calls:
        raise AssertionError(
            f"expected {expected_candidate_calls} selected calls, "
            f"got {len(candidate_calls)}"
        )
    result = {
        "schema_version": 2,
        "kind": "glm52_w13_current_api_v1_production_trace",
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "variant": args.variant,
        "provider": str(provider),
        "provider_sha256": _sha256(provider),
        "provider_state": provider_state,
        "provider_identity": dict(provider_object.identity),
        "physical_gpu": os.environ.get("GLM52_PHYSICAL_GPU"),
        "physical_gpu_uuid": os.environ.get("GLM52_PHYSICAL_GPU_UUID"),
        "nondefault_stream": int(stream.cuda_stream),
        "selected_calls": selected,
        "candidate_low_level_calls": candidate_calls,
        "stock_low_level_calls": stock_calls,
        "selected_call_count": len(candidate_calls),
        "selected_eager_and_graph_capture_call_count": 8,
        "selected_error_propagated_without_stock_retry": True,
        "all_api_v1_low_level_returns_exact_none": True,
        "all_public_returns_preserved_caller_out": True,
        "fallback_cases": fallback_cases,
        "unsupported_cases_fell_back_before_candidate": True,
        "dispatch_hits": dict(_HIT_COUNTS),
        "dispatch_misses": dict(_MISS_COUNTS),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
