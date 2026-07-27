"""DSA experiment: reuse the output buffer through FlashInfer's public ABI."""

from __future__ import annotations

import torch


_OUTPUTS: dict[tuple[int, tuple[int, ...]], torch.Tensor] = {}


def _output(inputs) -> torch.Tensor:
    query = inputs["query"]
    shape = (*query.shape[:-1], 512)
    key = (query.data_ptr(), shape)
    out = _OUTPUTS.get(key)
    if out is None:
        out = torch.empty(shape, dtype=torch.bfloat16, device=query.device)
        _OUTPUTS[key] = out
    return out


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
        bmm1_scale=inputs["bmm1_scale"],
        backend="trtllm-gen",
        out=_output(inputs),
    )
    return out.squeeze(1) if out.ndim == 4 and out.shape[1] == 1 else out
