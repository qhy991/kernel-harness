#!/usr/bin/env python3
"""Verify the pinned GLM-5.2 NVFP4 indexer contract without CUDA."""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import urllib.request
from pathlib import Path
from typing import Any


MODEL = "nvidia/GLM-5.2-NVFP4"
REVISION = "aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa"
BASE = f"https://huggingface.co/{MODEL}/resolve/{REVISION}"
TENSORS = {
    "model.layers.0.self_attn.indexer.wq_b.weight": {
        "dtype": "BF16",
        "shape": [4096, 2048],
    },
    "model.layers.0.self_attn.indexer.wk.weight": {
        "dtype": "BF16",
        "shape": [128, 6144],
    },
    "model.layers.0.self_attn.indexer.weights_proj.weight": {
        "dtype": "BF16",
        "shape": [32, 6144],
    },
}


def _read_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def _read_range(url: str, start: int, end: int) -> bytes:
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read(end + 1)
        if response.status == 206:
            return payload
        # Some HTTP stacks discard Range while following the signed CDN URL.
        # Reading only through the requested end keeps this bounded.
        return payload[start : end + 1]


def _read_safetensors_header(url: str) -> dict[str, Any]:
    prefix = _read_range(url, 0, 7)
    if len(prefix) != 8:
        raise RuntimeError(f"short safetensors prefix: {len(prefix)} bytes")
    header_len = struct.unpack("<Q", prefix)[0]
    if not 2 <= header_len <= 64 * 1024 * 1024:
        raise RuntimeError(f"implausible safetensors header length: {header_len}")
    payload = _read_range(url, 8, 8 + header_len - 1)
    if len(payload) != header_len:
        raise RuntimeError(
            f"short safetensors header: {len(payload)} != {header_len} bytes"
        )
    return json.loads(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if "SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN" in os.environ:
        raise RuntimeError(
            "fixed production recipe leaves SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN unset"
        )

    config_url = f"{BASE}/config.json"
    index_url = f"{BASE}/model.safetensors.index.json"
    config = _read_json(config_url)
    index = _read_json(index_url)
    quant = config["quantization_config"]

    assert config["architectures"] == ["GlmMoeDsaForCausalLM"]
    assert config["max_position_embeddings"] == 1048576
    assert config["rope_parameters"] == {
        "rope_theta": 8000000,
        "rope_type": "default",
    }
    assert config["indexer_rope_interleave"] is True
    assert config.get("index_k_norm_type") is None
    assert quant["quant_algo"] == "NVFP4"
    from sglang.srt.environ import envs
    from sglang.srt.layers.linear import ReplicatedLinear
    from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config
    from sglang.srt.layers.quantization.utils import is_layer_skipped

    assert envs.SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN.get() is False
    # model_loader._get_quantization_config injects the model class's packed
    # mapping before ModelOptFp4Config.from_config. DeepseekV2ForCausalLM starts
    # with an empty mapping; its instance later adds only fused_qkv_a.
    quant_for_dispatch = dict(quant)
    quant_for_dispatch["packed_modules_mapping"] = {}
    quant_config = ModelOptFp4Config.from_config(quant_for_dispatch)
    indexer_layer_ids = sorted(
        {
            int(match.group(1))
            for name in index["weight_map"]
            if (
                match := re.fullmatch(
                    r"model\.layers\.(\d+)\.self_attn\.indexer\.wq_b\.weight",
                    name,
                )
            )
        }
    )
    assert indexer_layer_ids
    full_decoder_layer_ids = [
        layer_id
        for layer_id, indexer_type in enumerate(config["indexer_types"])
        if indexer_type == "full"
    ]
    mtp_layer_ids = list(
        range(
            config["num_hidden_layers"],
            config["num_hidden_layers"] + config["num_nextn_predict_layers"],
        )
    )
    expected_checkpoint_layer_ids = full_decoder_layer_ids + mtp_layer_ids
    assert indexer_layer_ids == expected_checkpoint_layer_ids
    indexer_prefixes = [
        f"model.layers.{layer_id}.self_attn.indexer.wq_b"
        for layer_id in indexer_layer_ids
    ]
    def is_excluded_by_actual_dispatch(prefix: str) -> bool:
        return is_layer_skipped(
            prefix,
            quant_config.exclude_modules,
            quant_config.packed_modules_mapping,
        ) or quant_config.is_layer_excluded(prefix)

    uncovered_prefixes = [
        prefix for prefix in indexer_prefixes if not is_excluded_by_actual_dispatch(prefix)
    ]
    assert not uncovered_prefixes, uncovered_prefixes
    configured_layer_ids = list(
        range(config["num_hidden_layers"] + config["num_nextn_predict_layers"])
    )
    configured_prefixes = [
        f"model.layers.{layer_id}.self_attn.indexer.wq_b"
        for layer_id in configured_layer_ids
    ]
    uncovered_configured_prefixes = [
        prefix
        for prefix in configured_prefixes
        if not is_excluded_by_actual_dispatch(prefix)
    ]
    assert not uncovered_configured_prefixes, uncovered_configured_prefixes
    layer = ReplicatedLinear(
        2048,
        4096,
        bias=False,
        params_dtype=__import__("torch").bfloat16,
        quant_config=quant_config,
        prefix="model.layers.3.self_attn.indexer.wq_b",
    )
    quant_method = type(layer.quant_method).__name__
    assert quant_method == "UnquantizedLinearMethod"

    shard_names = {index["weight_map"][name] for name in TENSORS}
    assert len(shard_names) == 1
    shard_name = shard_names.pop()
    shard_url = f"{BASE}/{shard_name}"
    header = _read_safetensors_header(shard_url)
    observed_tensors = {}
    for name, expected in TENSORS.items():
        observed = {key: header[name][key] for key in ("dtype", "shape")}
        assert observed == expected, f"{name}: {observed} != {expected}"
        observed_tensors[name] = observed

    result = {
        "status": "PASS",
        "model": MODEL,
        "revision": REVISION,
        "sources": {
            "config": config_url,
            "safetensors_index": index_url,
            "safetensors_shard": shard_url,
        },
        "model_contract": {
            "architecture": config["architectures"][0],
            "max_position_embeddings": config["max_position_embeddings"],
            "rope_parameters": config["rope_parameters"],
            "indexer_rope_interleave": config["indexer_rope_interleave"],
            "index_k_norm_type": "layer (field absent)",
            "attention_fp8_override": False,
            "resolved_indexer_wq_b_quant_method": quant_method,
            "indexer_wq_b_ignore_coverage": {
                "dispatch_predicate": (
                    "is_layer_skipped(...) or ModelOptFp4Config.is_layer_excluded(...)"
                ),
                "configured_layer_count_including_mtp": len(configured_layer_ids),
                "configured_layer_ids": configured_layer_ids,
                "checkpoint_owned_full_indexer_layer_count": len(indexer_layer_ids),
                "checkpoint_owned_full_indexer_layer_ids": indexer_layer_ids,
                "mtp_indexer_layer_ids": mtp_layer_ids,
                "shared_indexer_layer_ids": [
                    layer_id
                    for layer_id, indexer_type in enumerate(config["indexer_types"])
                    if indexer_type == "shared"
                ],
                "uncovered_checkpoint_prefixes": uncovered_prefixes,
                "uncovered_configured_prefixes": uncovered_configured_prefixes,
            },
        },
        "checkpoint_tensors": observed_tensors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
