#!/usr/bin/env python3
"""Leased-B200 correctness and ABI matrix for same-source W13 variants."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from serving_native.runner import (
    MOE_OUTPUT_POISON,
    Runtime,
    _load_candidate,
)
from serving_native.workloads import get_workload

VARIANT_CANDIDATES = {
    "bm32_2sm": HERE / "candidates" / "w13_bm32_2sm.py",
    "bm32_1sm": HERE / "candidates" / "w13_bm32_1sm.py",
}
EXPECTED_TENSOR_CONTRACT = {
    "activation_fp8": {
        "shape": (32, 1024, 6144),
        "stride": (6291456, 6144, 1),
        "dtype": "torch.float8_e4m3fn",
    },
    "activation_scale": {
        "shape": (32, 1024, 12),
        "stride": (12288, 1, 1024),
        "dtype": "torch.int32",
    },
    "weight_fp8": {
        "shape": (32, 4096, 6144),
        "stride": (25165824, 6144, 1),
        "dtype": "torch.float8_e4m3fn",
    },
    "weight_scale": {
        "shape": (32, 4096, 12),
        "stride": (49152, 1, 4096),
        "dtype": "torch.int32",
    },
    "out": {
        "shape": (32, 1024, 4096),
        "stride": (4194304, 4096, 1),
        "dtype": "torch.bfloat16",
    },
    "masked_m": {
        "shape": (32,),
        "stride": (1,),
        "dtype": "torch.int32",
    },
}


def _tensor_bytes_sha256(tensor: Any) -> str:
    value = tensor.detach().contiguous().cpu()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _assert_tensor_contract(inputs: dict[str, Any]) -> dict[str, Any]:
    observed = {}
    device = inputs["activation_fp8"].device
    for name, expected in EXPECTED_TENSOR_CONTRACT.items():
        tensor = inputs[name]
        actual = {
            "shape": tuple(tensor.shape),
            "stride": tuple(tensor.stride()),
            "dtype": str(tensor.dtype),
            "storage_offset": int(tensor.storage_offset()),
            "device": str(tensor.device),
        }
        if (
            actual["shape"] != expected["shape"]
            or actual["stride"] != expected["stride"]
            or actual["dtype"] != expected["dtype"]
            or actual["storage_offset"] != 0
            or tensor.device != device
        ):
            raise AssertionError(
                f"W13 tensor contract mismatch for {name}: "
                f"actual={actual}, expected={expected}"
            )
        observed[name] = actual
    return observed


def _production_counts(torch, assignments: int, seed: int) -> tuple[int, ...]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    expert_ids = torch.randint(
        32,
        (assignments,),
        generator=generator,
        dtype=torch.int64,
    )
    return tuple(
        int(value) for value in torch.bincount(expert_ids, minlength=32).tolist()
    )


def _cases(torch) -> list[dict[str, Any]]:
    zero = (0,) * 32
    minimum = (1,) + (0,) * 31
    maximum = (1024,) + (0,) * 31
    skewed = (256,) + (0,) * 31
    boundaries = (31, 32, 33) + (0,) * 29
    return [
        {
            "name": "production_m16_em4_random",
            "expected_m": 4,
            "counts": _production_counts(torch, 128, 2026072404),
            "data": "random",
        },
        {
            "name": "production_m16_em5_ramp",
            "expected_m": 5,
            "counts": _production_counts(torch, 128, 2026072405),
            "data": "ramp",
        },
        {
            "name": "production_m32_em8_extreme_finite",
            "expected_m": 8,
            "counts": _production_counts(torch, 256, 2026072408),
            "data": "extreme_finite",
        },
        {
            "name": "production_m32_em9_changed",
            "expected_m": 9,
            "counts": _production_counts(torch, 256, 2026072409),
            "data": "changed",
        },
        {
            "name": "zero_counts_poisoned_invalid_rows",
            "expected_m": 4,
            "counts": zero,
            "data": "poison_invalid",
        },
        {
            "name": "minimum_count",
            "expected_m": 4,
            "counts": minimum,
            "data": "constant",
        },
        {
            "name": "maximum_count",
            "expected_m": 4,
            "counts": maximum,
            "data": "constant",
        },
        {
            "name": "highly_skewed",
            "expected_m": 9,
            "counts": skewed,
            "data": "constant",
        },
        {
            "name": "bm32_boundaries_31_32_33",
            "expected_m": 5,
            "counts": boundaries,
            "data": "poison_invalid",
        },
    ]


def _set_data_pattern(torch, inputs: dict[str, Any], case: dict[str, Any]) -> None:
    activation = inputs["activation_fp8"]
    weight = inputs["weight_fp8"]
    pattern = case["data"]
    if pattern == "random":
        return
    if pattern in ("ramp", "changed"):
        k = activation.shape[-1]
        ramp_i32 = (
            torch.arange(k, device=activation.device, dtype=torch.int32)
            .remainder(7)
            .sub(3)
        )
        if pattern == "changed":
            ramp_i32.neg_()
        ramp = ramp_i32.to(dtype=activation.dtype)
        activation.copy_(ramp)
        weight.copy_(ramp if pattern == "ramp" else ramp.flip(0))
        return
    if pattern == "extreme_finite":
        activation.fill_(448.0)
        weight.fill_(1.0)
        return
    if pattern == "constant":
        activation.fill_(1.0)
        weight.fill_(1.0)
        return
    if pattern == "poison_invalid":
        activation.fill_(float("nan"))
        weight.fill_(1.0)
        for expert, count in enumerate(case["counts"]):
            if count:
                activation[expert, :count].fill_(1.0)
        return
    raise ValueError(pattern)


def _masked_store_limit(count: int, block_m: int, slab: int) -> int:
    if count <= 0:
        return 0
    return min(((count + block_m - 1) // block_m) * block_m, slab)


def _assert_untouched(
    output: Any,
    counts: tuple[int, ...],
    *,
    block_m: int,
) -> dict[str, int]:
    """Validate rows outside the scheduled MGroupedMasked store tiles.

    DeepGEMM predicates CTA scheduling with ``masked_m`` but stores complete
    ``store_block_m`` tiles for scheduled CTAs. Rows between ``masked_m`` and
    the last scheduled tile boundary are padding, not untouched storage.
    """

    padding_rows_written = 0
    untouched_rows_checked = 0
    slab = int(output.shape[1])
    for expert, count in enumerate(counts):
        store_limit = _masked_store_limit(count, block_m, slab)
        padding = output[expert, count:store_limit]
        if padding.numel():
            padding_rows_written += int(
                padding.ne(MOE_OUTPUT_POISON).any(dim=-1).sum().item()
            )
        untouched = output[expert, store_limit:]
        untouched_rows_checked += int(untouched.shape[0])
        if not bool(untouched.eq(MOE_OUTPUT_POISON).all().item()):
            raise AssertionError(
                "expert "
                f"{expert} wrote outside scheduled masked-store envelope "
                f"[0:{store_limit}] for masked_m={count}, block_m={block_m}"
            )
    return {
        "store_block_m": block_m,
        "padding_rows_written": padding_rows_written,
        "untouched_rows_checked": untouched_rows_checked,
    }


def _compare_valid(
    torch,
    reference: Any,
    candidate: Any,
    counts: tuple[int, ...],
) -> dict[str, Any]:
    max_abs = 0.0
    max_rel = 0.0
    anomaly_mismatches = 0
    failing_elements = 0
    compared_elements = 0
    reference_nonfinite = 0
    candidate_nonfinite = 0
    for expert, count in enumerate(counts):
        if not count:
            continue
        ref = reference[expert, :count].float()
        cand = candidate[expert, :count].float()
        ref_finite = torch.isfinite(ref)
        cand_finite = torch.isfinite(cand)
        reference_nonfinite += int((~ref_finite).sum().item())
        candidate_nonfinite += int((~cand_finite).sum().item())
        anomaly_mismatches += int(ref_finite.ne(cand_finite).sum().item())
        finite = ref_finite & cand_finite
        if not bool(finite.any().item()):
            continue
        diff = (ref[finite] - cand[finite]).abs()
        rel = diff / ref[finite].abs().clamp_min(1e-6)
        max_abs = max(max_abs, float(diff.max().item()))
        max_rel = max(max_rel, float(rel.max().item()))
        failing_elements += int(((diff > 2e-2) & (rel > 2e-2)).sum().item())
        compared_elements += int(diff.numel())
    if anomaly_mismatches or failing_elements:
        raise AssertionError(
            "W13 candidate differs from same-source stock: "
            f"anomaly_mismatches={anomaly_mismatches}, "
            f"reference_nonfinite={reference_nonfinite}, "
            f"candidate_nonfinite={candidate_nonfinite}, "
            f"failing_elements={failing_elements}, "
            f"max_abs={max_abs}, max_rel={max_rel}"
        )
    return {
        "compared_elements": compared_elements,
        "anomaly_mismatches": anomaly_mismatches,
        "reference_nonfinite": reference_nonfinite,
        "candidate_nonfinite": candidate_nonfinite,
        "failing_elements": failing_elements,
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "rtol": 2e-2,
        "atol": 2e-2,
    }


def _validate_variant(manifest: Path, variant: str) -> dict[str, Any]:
    import torch

    os.environ["SGLANG_GLM52_W13_DECODE_MANIFEST"] = str(manifest)
    candidate = _load_candidate(str(VARIANT_CANDIDATES[variant]))
    assert candidate is not None
    runtime = Runtime(
        get_workload("moe_w13_grouped_decode_m16_em4"),
        candidate,
    )
    try:
        inputs = runtime.build_inputs()
        tensor_contract = _assert_tensor_contract(inputs)
        scale_hashes_before = {
            name: _tensor_bytes_sha256(inputs[name])
            for name in ("activation_scale", "weight_scale")
        }
        reference_out = inputs["out"]
        candidate_out = torch.empty_strided(
            reference_out.shape,
            reference_out.stride(),
            device=reference_out.device,
            dtype=reference_out.dtype,
        )
        reference_guard = torch.empty_strided(
            reference_out.shape,
            reference_out.stride(),
            device=reference_out.device,
            dtype=reference_out.dtype,
        )
        output_pointers = {
            int(reference_out.data_ptr()),
            int(candidate_out.data_ptr()),
            int(reference_guard.data_ptr()),
        }
        if len(output_pointers) != 3:
            raise AssertionError("stock, candidate, and guard outputs alias")
        stream = torch.cuda.Stream(device=runtime.device)
        cases = []
        for case in _cases(torch):
            counts = case["counts"]
            inputs["masked_m"].copy_(
                torch.tensor(counts, device=runtime.device, dtype=torch.int32)
            )
            _set_data_pattern(torch, inputs, case)
            reference_out.fill_(MOE_OUTPUT_POISON)
            candidate_out.fill_(MOE_OUTPUT_POISON)
            stream.wait_stream(torch.cuda.current_stream(runtime.device))
            with torch.cuda.stream(stream):
                stock_return = runtime.w13_runtime.stock_launcher(
                    (inputs["activation_fp8"], inputs["activation_scale"]),
                    (inputs["weight_fp8"], inputs["weight_scale"]),
                    reference_out,
                    inputs["masked_m"],
                    case["expected_m"],
                )
                reference_guard.copy_(reference_out)
                candidate_return = runtime.w13_runtime.candidate_launcher(
                    (inputs["activation_fp8"], inputs["activation_scale"]),
                    (inputs["weight_fp8"], inputs["weight_scale"]),
                    candidate_out,
                    inputs["masked_m"],
                    case["expected_m"],
                )
            stream.synchronize()
            if stock_return is not None or candidate_return is not None:
                raise AssertionError(
                    "no-overlap W13 launch did not preserve exact None return"
                )
            if not torch.equal(
                reference_out.view(torch.int16),
                reference_guard.view(torch.int16),
            ):
                changed_words = int(
                    reference_out.view(torch.int16)
                    .ne(reference_guard.view(torch.int16))
                    .sum()
                    .item()
                )
                raise AssertionError(
                    f"{variant}/{case['name']}: candidate launch changed "
                    f"{changed_words} words in the distinct stock output"
                )
            stock_untouched = _assert_untouched(
                reference_guard,
                counts,
                block_m=128,
            )
            candidate_untouched = _assert_untouched(
                candidate_out,
                counts,
                block_m=32,
            )
            try:
                comparison = _compare_valid(
                    torch,
                    reference_guard,
                    candidate_out,
                    counts,
                )
            except AssertionError as exc:
                raise AssertionError(f"{variant}/{case['name']}: {exc}") from exc
            cases.append(
                {
                    "name": case["name"],
                    "expected_m": case["expected_m"],
                    "counts": list(counts),
                    "data_pattern": case["data"],
                    "stock_return": None,
                    "candidate_return": None,
                    "stock_output_preserved_after_candidate": True,
                    "masked_store_contract": (
                        "rows outside ceil(masked_m/store_block_m) store tiles "
                        "remain pre-poisoned"
                    ),
                    "untouched_masked_regions": {
                        "stock": stock_untouched,
                        "candidate": candidate_untouched,
                    },
                    "comparison": comparison,
                }
            )
        scale_hashes_after = {
            name: _tensor_bytes_sha256(inputs[name])
            for name in ("activation_scale", "weight_scale")
        }
        if scale_hashes_after != scale_hashes_before:
            raise AssertionError("packed int32 scale bytes changed during W13 calls")
        return {
            "variant": variant,
            "status": "pass",
            "tensor_contract": tensor_contract,
            "output_ownership_distinct": True,
            "nondefault_stream": int(stream.cuda_stream),
            "packed_scale_sha256_before": scale_hashes_before,
            "packed_scale_sha256_after": scale_hashes_after,
            "runtime_identity": runtime.w13_runtime.identity,
            "cases": cases,
        }
    finally:
        runtime.close()
        del runtime
        torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--variant",
        choices=(*VARIANT_CANDIDATES, "both"),
        default="both",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = args.manifest.expanduser().resolve()
    variants = tuple(VARIANT_CANDIDATES) if args.variant == "both" else (args.variant,)
    result = {
        "schema_version": 1,
        "kind": "glm52_w13_same_source_correctness",
        "manifest": str(manifest),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "variants": [_validate_variant(manifest, variant) for variant in variants],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
