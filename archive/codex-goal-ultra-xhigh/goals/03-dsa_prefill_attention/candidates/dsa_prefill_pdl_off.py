"""Exact TRT-LLM DSA prefill call with programmatic dependent launch disabled."""

from __future__ import annotations

def run(inputs, runtime):
    import flashinfer.decode

    p = runtime.workload.params
    q_shape = tuple(inputs["query"].shape)
    kv_shape = tuple(inputs["kv_cache"].shape)
    exact_measured_abi = (
        q_shape == (4096, 1, 64, 576)
        and len(kv_shape) == 4
        and kv_shape[1:] == (1, 64, 576)
        and tuple(inputs["block_tables"].shape) == (4096, 1, 2048)
        and tuple(inputs["seq_lens"].shape) == (4096,)
        and inputs["max_seq_len"] == 32768
        and inputs["query"].dtype == runtime.torch.float8_e4m3fn
        and inputs["kv_cache"].dtype == runtime.torch.float8_e4m3fn
    )
    enable_pdl = False if exact_measured_abi else None
    out = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
        query=inputs["query"],
        kv_cache=inputs["kv_cache"],
        workspace_buffer=inputs["workspace"],
        qk_nope_head_dim=p["qk_nope_head_dim"],
        kv_lora_rank=p["kv_lora_rank"],
        qk_rope_head_dim=p["qk_rope_head_dim"],
        block_tables=inputs["block_tables"],
        seq_lens=inputs["seq_lens"],
        max_seq_len=inputs["max_seq_len"],
        sparse_mla_top_k=inputs["sparse_topk"],
        bmm1_scale=inputs["bmm1_scale"],
        backend="trtllm-gen",
        enable_pdl=enable_pdl,
    )
    if out.ndim == 4 and out.shape[1] == 1:
        out = out.squeeze(1)
    return out
