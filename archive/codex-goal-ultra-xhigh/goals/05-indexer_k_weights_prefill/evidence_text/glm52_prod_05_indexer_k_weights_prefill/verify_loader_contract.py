#!/usr/bin/env python3
"""CPU-only checks for the GLM-5.2 fused indexer checkpoint loader contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.srt.models.deepseek_common import deepseek_weight_loader as loader


PREFIX = "model.layers.0.self_attn.indexer"
FUSED_NAME = f"{PREFIX}.wk_weights_proj.weight"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fused = torch.nn.Parameter(
        torch.full((160, 6144), float("nan"), dtype=torch.bfloat16)
    )
    params = {FUSED_NAME: fused}
    pending: dict[str, dict[str, torch.Tensor]] = {}

    wk = torch.full((128, 6144), 1.25, dtype=torch.bfloat16)
    assert loader._load_fused_indexer_wk(
        f"{PREFIX}.wk.weight", wk, params, pending, None
    )
    assert torch.equal(fused[:128], wk)
    assert torch.isnan(fused[128:].float()).all()

    weights = torch.full((32, 6144), -2.5, dtype=torch.bfloat16)
    assert loader._load_fused_indexer_wk(
        f"{PREFIX}.weights_proj.weight", weights, params, pending, None
    )
    assert torch.equal(fused[:128], wk)
    assert torch.equal(fused[128:], weights)
    assert not pending

    # Exercise the block-FP8 rendezvous without requiring a GPU kernel. The
    # monkeypatch checks that weight and scale are held until both arrive and
    # that their dequantized BF16 rows fill only the K portion.
    calls: list[dict[str, object]] = []
    original_dequant = loader.block_quant_dequant

    def fake_dequant(weight, scale, block_size, output_dtype):
        calls.append(
            {
                "weight_shape": list(weight.shape),
                "weight_dtype": str(weight.dtype),
                "scale_shape": list(scale.shape),
                "block_size": list(block_size),
                "output_dtype": str(output_dtype),
            }
        )
        return torch.full(weight.shape, 3.0, dtype=output_dtype)

    loader.block_quant_dequant = fake_dequant
    try:
        fused.data.fill_(float("nan"))
        fp8_weight = torch.zeros((128, 6144), dtype=torch.float8_e4m3fn)
        fp8_scale = torch.ones((1, 48), dtype=torch.float32)
        quant_config = SimpleNamespace(weight_block_size=[128, 128])
        assert loader._load_fused_indexer_wk(
            f"{PREFIX}.wk.weight_scale_inv",
            fp8_scale,
            params,
            pending,
            quant_config,
        )
        assert FUSED_NAME in pending and "weight" not in pending[FUSED_NAME]
        assert loader._load_fused_indexer_wk(
            f"{PREFIX}.wk.weight",
            fp8_weight,
            params,
            pending,
            quant_config,
        )
    finally:
        loader.block_quant_dequant = original_dequant

    assert not pending
    assert len(calls) == 1
    assert torch.all(fused[:128] == 3.0)
    assert torch.isnan(fused[128:].float()).all()

    assert not loader._load_fused_indexer_wk(
        f"{PREFIX}.wk.weight", wk, {}, {}, None
    )
    non_bf16 = {FUSED_NAME: torch.nn.Parameter(torch.empty(160, 6144))}
    assert not loader._load_fused_indexer_wk(
        f"{PREFIX}.wk.weight", wk, non_bf16, {}, None
    )

    result = {
        "status": "pass",
        "bf16_direct": {
            "wk_rows": [0, 128],
            "weights_rows": [128, 160],
            "fused_shape": [160, 6144],
            "fused_dtype": "torch.bfloat16",
        },
        "fp8_block_dequant": calls[0],
        "fallback_when_fused_param_absent": True,
        "fallback_when_fused_param_not_bf16": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
