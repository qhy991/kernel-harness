"""Shared exact K/scale-gather launch experiment for indexer score prefill."""

from __future__ import annotations


def _gather(
    pool,
    buf,
    *,
    page_indices,
    seq_len_tensor,
    seq_len_sum,
    max_seq_len,
    block_size: int,
    num_warps: int,
):
    import torch

    from sglang.srt.layers.attention.dsa.index_buf_accessor import (
        _get_k_and_s_triton_kernel,
    )

    index_head_dim = pool.index_head_dim
    page_size = pool.page_size
    k_out = torch.empty(
        (seq_len_sum, index_head_dim),
        dtype=torch.uint8,
        device=buf.device,
    )
    s_out = torch.empty(
        (seq_len_sum, 4),
        dtype=torch.uint8,
        device=buf.device,
    )
    _, buf_numel_per_page = buf.shape
    _, page_indice_batch_offset = page_indices.shape
    num_token_blocks = (max_seq_len + block_size - 1) // block_size
    seq_num = seq_len_tensor.shape[0]
    seq_num_pow2 = 1 << (seq_num - 1).bit_length()
    grid = (seq_num, num_token_blocks, 1)
    _get_k_and_s_triton_kernel[grid](
        buf_ptr=buf,
        page_indices_ptr=page_indices,
        k_out_ptr=k_out,
        s_out_ptr=s_out,
        seq_len_ptr=seq_len_tensor,
        seq_len_num_pow=seq_num_pow2,
        page_size=page_size,
        buf_numel_per_page=buf_numel_per_page,
        index_head_dim=index_head_dim,
        s_offset_in_page=page_size * index_head_dim,
        page_indice_batch_offset=page_indice_batch_offset,
        BLOCK_SIZE=block_size,
        BLOCK_SIZE_K=128,
        num_warps=num_warps,
    )
    return k_out, s_out


def run_with_config(inputs, runtime, *, block_size: int, num_warps: int):
    fixture = inputs["fixture"]
    pool = fixture.forward_context.attn_backend.token_to_kv_pool

    def override(pool_arg, buf, **kwargs):
        return _gather(
            pool_arg,
            buf,
            block_size=block_size,
            num_warps=num_warps,
            **kwargs,
        )

    original = pool.get_k_and_s_override
    pool.get_k_and_s_override = override
    try:
        if runtime.workload.family == "indexer_score_prefill":
            return {
                "topk_indices": runtime.run_indexer_score_prefill(inputs)
            }
        if runtime.workload.family == "indexer_graph_split_prefill":
            return inputs["region"].run_graph_split()
        return runtime.run_indexer_prefill_region(inputs)
    finally:
        pool.get_k_and_s_override = original
