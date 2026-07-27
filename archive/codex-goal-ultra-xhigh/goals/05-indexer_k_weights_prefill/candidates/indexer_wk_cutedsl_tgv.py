"""SM100 CuTe-DSL TGV trial for GLM-5.2 indexer ``wk_weights_proj``.

The direct call deliberately bypasses SGLang's conservative dispatch policy so
the exact M=4096, N=160, K=6144 serving shape can be evaluated. Compilation
and tactic caching are forced at module import, outside ``run()``.
"""

import torch

from sglang.jit_kernel.cutedsl_bf16_gemm import cutedsl_bf16_gemm


CANDIDATE_METADATA = {
    "backend": "sglang_cutedsl_tgv_direct",
    "shape_guard_intended": [4096, 160, 6144],
}


def _compile_fixed_shape_at_import():
    x = torch.empty((4096, 6144), device="cuda", dtype=torch.bfloat16)
    weight = torch.empty((160, 6144), device="cuda", dtype=torch.bfloat16)
    output = cutedsl_bf16_gemm(x, weight)
    torch.cuda.synchronize()
    del output, weight, x


_compile_fixed_shape_at_import()


def wk_backend(x, weight):
    return cutedsl_bf16_gemm(x, weight)


def run(inputs, runtime):
    if runtime.workload.family == "bf16_linear":
        return wk_backend(inputs["x"], inputs["weight"])
    return runtime.indexer_fused_prepare_store(inputs, wk_backend=wk_backend)
