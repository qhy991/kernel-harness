"""FlashInfer selector experiment for GLM-5.2 DSA M16.

The vendored header removes the stock factor-of-two cap on KV splits.  With
the production M16 shape this permits 9 splits / 144 main CTAs instead of
8 / 128; M32 remains at 4 / 128.  The extension is compiled at import time
and loads the same pinned FlashInfer 0.6.12 TRT-LLM-GEN cubin archive.
"""

from __future__ import annotations

from pathlib import Path

import torch

from flashinfer.jit import setup_cubin_loader
from flashinfer.jit.attention.modules import gen_trtllm_gen_fmha_module
from flashinfer.jit.core import gen_jit_spec
from flashinfer.utils import device_support_pdl, get_device_sm_count


HERE = Path(__file__).resolve().parent


def _build_module():
    base = gen_trtllm_gen_fmha_module()
    artifact_flags = [
        flag for flag in base.extra_cuda_cflags if flag.startswith("-DTLLM_GEN_")
    ]
    spec = gen_jit_spec(
        "fmha_gen_glm52_dsa_split9_v1",
        base.sources,
        extra_include_paths=[HERE / "include", *base.extra_include_dirs],
        extra_cuda_cflags=artifact_flags,
    )
    module = spec.build_and_load()
    setup_cubin_loader(spec.get_library_path())
    return module


_MODULE = _build_module()


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
