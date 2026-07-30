#!/usr/bin/env python3
"""Validate adversarial value, sparse-index, and scheduler cases."""

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
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def compare(torch, reference, candidate, name: str) -> dict[str, object]:
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        raise AssertionError(f"{name}: shape or dtype mismatch")
    ref_f = reference.float()
    cand_f = candidate.float()
    ref_bad = ~torch.isfinite(ref_f)
    cand_bad = ~torch.isfinite(cand_f)
    if not torch.equal(ref_bad, cand_bad):
        raise AssertionError(f"{name}: anomaly positions differ")
    finite = ~ref_bad
    if finite.any():
        torch.testing.assert_close(
            cand_f[finite],
            ref_f[finite],
            rtol=2e-2,
            atol=2e-2,
            equal_nan=False,
        )
        max_abs = float((cand_f[finite] - ref_f[finite]).abs().max().item())
    else:
        max_abs = 0.0
    return {
        "exact": bool(torch.equal(reference, candidate)),
        "finite_elements": int(finite.sum().item()),
        "anomaly_elements": int(ref_bad.sum().item()),
        "anomaly_positions_match": True,
        "max_abs": max_abs,
        "rtol": 2e-2,
        "atol": 2e-2,
    }


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite evidence: {output}")

    runtime = Runtime(get_workload(f"dsa_flashmla_kv_decode_m{args.m}"))
    torch = runtime.torch
    try:
        inputs = runtime.build_inputs()
        from sgl_kernel.flash_mla import (
            flash_mla_with_kvcache,
            get_mla_metadata,
        )
        from sglang.srt.layers.attention.dsa.quant_k_cache import quantize_k_cache

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
        workspace = provider_module._WORKSPACES[args.m]

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
            return hotspot_provider.run_flashmla_sparse_decode(
                q=inputs["q"],
                k_cache=inputs["kv_cache"],
                cache_seqlens=inputs["cache_seqlens"],
                head_dim_v=inputs["head_dim_v"],
                tile_scheduler_metadata=inputs["tile_scheduler_metadata"],
                num_splits=inputs["num_splits"],
                softmax_scale=inputs["softmax_scale"],
                indices=inputs["indices"],
                block_table=inputs["block_table"],
                is_fp8_kvcache=True,
            )

        for _ in range(3):
            stock()
            candidate()
        torch.cuda.synchronize(runtime.device)

        results: list[dict[str, object]] = []

        def run_case(name: str, category: str) -> None:
            workspace[0].fill_(float("nan"))
            workspace[1].fill_(float("nan"))
            q_before = inputs["q"].clone()
            indices_before = inputs["indices"].clone()
            stock_out, stock_lse = stock()
            candidate_out, candidate_lse = candidate()
            torch.cuda.synchronize(runtime.device)
            out_check = compare(torch, stock_out, candidate_out, f"{name}.output")
            lse_check = compare(torch, stock_lse, candidate_lse, f"{name}.lse")
            if torch.isnan(candidate_out).any() or torch.isnan(candidate_lse).any():
                raise AssertionError(f"{name}: poisoned output remained")
            if not torch.equal(inputs["q"], q_before):
                raise AssertionError(f"{name}: q was mutated")
            if not torch.equal(inputs["indices"], indices_before):
                raise AssertionError(f"{name}: indices were mutated")
            results.append(
                {
                    "name": name,
                    "category": category,
                    "output": out_check,
                    "lse": lse_check,
                    "poison_overwritten": True,
                    "q_and_indices_immutable": True,
                }
            )

        baseline = {
            key: inputs[key].clone()
            for key in (
                "q",
                "kv_cache",
                "cache_seqlens",
                "indices",
                "tile_scheduler_metadata",
                "num_splits",
            )
        }

        def restore_baseline() -> None:
            for key, value in baseline.items():
                inputs[key] = value.clone()

        def copy_vector_pattern(q_vector, kv_vector) -> None:
            inputs["q"].copy_(q_vector.view(1, 1, 1, 576))
            logical = torch.empty(
                (*inputs["kv_cache"].shape[:3], 576),
                dtype=torch.bfloat16,
                device=runtime.device,
            )
            logical.copy_(kv_vector.view(1, 1, 1, 576))
            quantized = quantize_k_cache(logical)
            inputs["kv_cache"].copy_(quantized)
            del logical, quantized

        run_case("random_baseline", "value")

        zero = torch.zeros(576, dtype=torch.bfloat16, device=runtime.device)
        copy_vector_pattern(zero, zero)
        run_case("zero_q_kv", "value")

        q_ramp = torch.linspace(
            -0.25, 0.25, 576, dtype=torch.bfloat16, device=runtime.device
        )
        kv_ramp = torch.linspace(
            0.25, -0.25, 576, dtype=torch.bfloat16, device=runtime.device
        )
        copy_vector_pattern(q_ramp, kv_ramp)
        run_case("signed_ramp_q_kv", "value")

        signs = torch.where(
            torch.arange(576, device=runtime.device) % 2 == 0,
            torch.tensor(1.0, device=runtime.device),
            torch.tensor(-1.0, device=runtime.device),
        ).to(torch.bfloat16)
        copy_vector_pattern(signs * 128.0, signs * 448.0)
        run_case("extreme_finite_q_kv", "value")

        exponents = (
            torch.arange(576, device=runtime.device, dtype=torch.int32) % 16
        ) - 8
        exponent_values = torch.ldexp(
            signs.float(), exponents
        ).to(torch.bfloat16)
        copy_vector_pattern(exponent_values, exponent_values.roll(1))
        run_case("exponent_boundary_q_kv", "value")

        for suffix, seed in (("a", 2026072916), ("b", 2026072932)):
            generator = torch.Generator(device=runtime.device)
            generator.manual_seed(seed + args.m)
            inputs["q"].copy_(
                torch.randn(
                    inputs["q"].shape,
                    dtype=torch.bfloat16,
                    device=runtime.device,
                    generator=generator,
                )
                * 0.05
            )
            logical = (
                torch.randn(
                    (*inputs["kv_cache"].shape[:3], 576),
                    dtype=torch.bfloat16,
                    device=runtime.device,
                    generator=generator,
                )
                * 0.05
            )
            inputs["kv_cache"].copy_(quantize_k_cache(logical))
            del logical
            run_case(f"repeated_changed_q_kv_{suffix}", "value")

        restore_baseline()
        bases = 64 + torch.arange(
            args.m, dtype=torch.int32, device=runtime.device
        ) * 8192
        positions = torch.arange(
            2048, dtype=torch.int32, device=runtime.device
        )

        def set_positions(local_positions) -> None:
            inputs["indices"].copy_(
                bases[:, None, None] + local_positions.view(1, 1, 2048)
            )

        set_positions(torch.zeros_like(positions))
        run_case("duplicate_indices", "indices")

        set_positions(
            torch.cat((positions[::2], positions[1::2]), dim=0)
        )
        run_case("interleaved_indices", "indices")

        set_positions(positions)
        run_case("sorted_indices", "indices")

        set_positions(positions.flip(0))
        run_case("unsorted_reverse_indices", "indices")

        page_boundaries = torch.tensor(
            [0, 63, 64, 8191], dtype=torch.int32, device=runtime.device
        ).repeat(512)
        set_positions(page_boundaries)
        run_case("boundary_page_indices", "indices")

        set_positions(positions)
        inputs["indices"][:, :, 2::3] = -1
        run_case("mixed_minus_one_indices", "indices")

        inputs["indices"].fill_(-1)
        run_case("all_minus_one_indices", "indices")

        def set_scheduler(lengths) -> None:
            inputs["cache_seqlens"] = lengths.contiguous()
            metadata, splits = get_mla_metadata(
                cache_seqlens=inputs["cache_seqlens"],
                num_q_tokens_per_head_k=64,
                num_heads_k=1,
                num_heads_q=64,
                is_fp8_kvcache=True,
                topk=2048,
            )
            inputs["tile_scheduler_metadata"] = metadata
            inputs["num_splits"] = splits
            local = positions.view(1, 1, 2048).expand(args.m, 1, 2048)
            candidate_indices = bases[:, None, None] + local
            valid = local < lengths[:, None, None]
            inputs["indices"] = torch.where(
                valid,
                candidate_indices,
                torch.full_like(candidate_indices, -1),
            ).contiguous()

        minimum_lengths = torch.ones(
            args.m, dtype=torch.int32, device=runtime.device
        )
        set_scheduler(minimum_lengths)
        run_case("minimum_cache_length_and_split", "scheduler")

        edge_values = torch.tensor(
            [1, 63, 64, 65, 127, 128, 129, 2048],
            dtype=torch.int32,
            device=runtime.device,
        )
        set_scheduler(edge_values.repeat((args.m + 7) // 8)[: args.m])
        run_case("mixed_cache_and_split_edges", "scheduler")

        restore_baseline()
        run_case("maximum_cache_length_and_split", "scheduler")

        evidence = {
            "schema_version": 1,
            "stage": "adversarial_correctness_matrix",
            "m": args.m,
            "gpu": {
                "physical_index": int(os.environ["GLM52_PHYSICAL_GPU"]),
                "uuid": os.environ["GLM52_PHYSICAL_GPU_UUID"],
                "name": torch.cuda.get_device_properties(0).name,
            },
            "provider": provider_module.PROVIDER_INFO,
            "production_tolerance": {"rtol": 2e-2, "atol": 2e-2},
            "separate_graph_and_containing_evidence": (
                f"b3_b5_correctness_m{args.m}.json"
            ),
            "cases": results,
            "case_count": len(results),
            "all_cases_passed": True,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
        print(
            json.dumps(
                {
                    "m": args.m,
                    "case_count": len(results),
                    "all_cases_passed": True,
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
