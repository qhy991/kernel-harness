"""Direct ATen MM trial for the fused indexer BF16 projection."""

import torch


CANDIDATE_METADATA = {
    "backend": "torch_mm_direct",
    "shape_guard_intended": [4096, 160, 6144],
}


def wk_backend(x, weight):
    return torch.mm(x, weight.t())


def run(inputs, runtime):
    if runtime.workload.family == "bf16_linear":
        return wk_backend(inputs["x"], inputs["weight"])
    return runtime.indexer_fused_prepare_store(inputs, wk_backend=wk_backend)
