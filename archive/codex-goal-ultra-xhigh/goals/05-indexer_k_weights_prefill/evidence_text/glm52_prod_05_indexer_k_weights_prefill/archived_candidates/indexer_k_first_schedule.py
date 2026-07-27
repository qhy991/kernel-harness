"""Archived rejected schedule trial; requires SGLang commit a75a772a2."""

import torch

from serving_native.runner import TaskResult


CANDIDATE_METADATA = {
    "backend": "stock_bf16_gemm_k_before_q_schedule",
    "shape_guard_intended": [4096, 160, 6144],
    "execution_mode": "eager_dual_stream",
    "required_sglang_trial_commit": "a75a772a2a113e2847a4baaabba89182f78f7ae8",
    "decision": "rejected_and_reverted",
}


def wk_backend(x, weight):
    return torch.nn.functional.linear(x, weight)


def run(inputs, runtime):
    from sglang.srt.layers.attention.dsa.dsa_indexer import Indexer

    indexer = runtime._indexer_region_proxy(wk_backend)
    common = runtime._indexer_region_common
    q_fp8, weights = Indexer._fused_q_prepare_and_store(
        indexer,
        inputs["x"],
        inputs["q_lora"],
        inputs["positions"],
        common["forward_batch"],
        0,
        common["act_quant"],
        enable_dual_stream=True,
        schedule_k_before_q=True,
    )
    return TaskResult(
        {
            "q_fp8": q_fp8,
            "weights": weights,
            "index_k_cache": inputs["index_k_cache"],
        }
    )
