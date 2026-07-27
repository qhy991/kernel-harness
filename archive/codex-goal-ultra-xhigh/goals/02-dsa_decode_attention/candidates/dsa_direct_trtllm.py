"""DSA experiment: invoke the pinned TRT-LLM-GEN module without wrapper work.

The launch arguments mirror FlashInfer 0.6.12's validated public wrapper for
the fixed serving_native ABI.  This isolates Python enqueue overhead while
retaining the stock selector, cubin archive, PDL policy, and output allocation.
"""

from __future__ import annotations

import torch

from flashinfer.mla._core import get_trtllm_gen_fmha_module
from flashinfer.utils import device_support_pdl, get_device_sm_count


_MODULE = get_trtllm_gen_fmha_module()


def run(inputs, runtime):
    query = inputs["query"]
    kv_cache = inputs["kv_cache"]
    workspace = inputs["workspace"]
    batch_size, max_q_len, _, _ = query.shape
    out = torch.empty(
        (*query.shape[:-1], 512), dtype=torch.bfloat16, device=query.device
    )
    _MODULE.trtllm_paged_attention_decode(
        out,
        None,
        query.flatten(0, 1),
        kv_cache,
        kv_cache,
        workspace,
        inputs["block_tables"],
        inputs["seq_lens"],
        max_q_len,
        inputs["max_seq_len"],
        inputs["bmm1_scale"],
        1.0,
        -1,
        -1,
        0,
        batch_size,
        -1,
        inputs["sparse_topk"],
        get_device_sm_count(query.device),
        device_support_pdl(query.device),
        workspace.numel() * workspace.element_size(),
        None,
        None,
        None,
        None,
        None,
        True,
        None,
        0,
        0,
    )
    return out.squeeze(1)
