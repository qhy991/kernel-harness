"""DSA experiment: exercise FlashInfer's device-tensor BMM1 scale path.

The first untimed correctness call materializes the scalar on the target device.
The public FlashInfer API still performs its documented log2(e) conversion on
every invocation, so this candidate measures the production-callable path as-is.
"""

from __future__ import annotations

import torch


_SCALES: dict[tuple[torch.device, float], torch.Tensor] = {}


def _device_scale(inputs) -> torch.Tensor:
    value = float(inputs["bmm1_scale"])
    key = (inputs["query"].device, value)
    scale = _SCALES.get(key)
    if scale is None:
        scale = torch.tensor(value, dtype=torch.float32, device=key[0])
        _SCALES[key] = scale
    return scale


def run(inputs, runtime):
    import flashinfer.decode

    out = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
        query=inputs["query"],
        kv_cache=inputs["kv_cache"],
        workspace_buffer=inputs["workspace"],
        qk_nope_head_dim=192,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        block_tables=inputs["block_tables"],
        seq_lens=inputs["seq_lens"],
        max_seq_len=inputs["max_seq_len"],
        sparse_mla_top_k=inputs["sparse_topk"],
        bmm1_scale=_device_scale(inputs),
        backend="trtllm-gen",
    )
    return out.squeeze(1) if out.ndim == 4 and out.shape[1] == 1 else out
