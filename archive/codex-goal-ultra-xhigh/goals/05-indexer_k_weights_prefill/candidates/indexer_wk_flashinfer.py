"""FlashInfer BF16 GEMM backend sweep for indexer ``wk_weights_proj``.

Set ``INDEXER_WK_FLASHINFER_BACKEND`` to one of the documented FlashInfer
backends.  The module-level callable stays stable so the fused-region runner
can retain backend-specific state between paired samples.
"""

import os

import flashinfer
import torch


_ALLOWED = {"auto", "cublaslt", "cudnn", "cutlass", "tgv", "tinygemm"}
_BACKEND = os.environ.get("INDEXER_WK_FLASHINFER_BACKEND", "auto")
if _BACKEND not in _ALLOWED:
    raise ValueError(
        "INDEXER_WK_FLASHINFER_BACKEND must be one of " + ", ".join(sorted(_ALLOWED))
    )

CANDIDATE_METADATA = {
    "backend": f"flashinfer_mm_bf16_{_BACKEND}",
    "shape_guard_intended": [4096, 160, 6144],
}


def _initialize_fixed_shape_at_import():
    """Resolve workspace, module, and tactic state outside the timed ABI."""
    x = torch.empty((4096, 6144), device="cuda", dtype=torch.bfloat16)
    weight = torch.empty((160, 6144), device="cuda", dtype=torch.bfloat16)
    output = flashinfer.mm_bf16(x, weight.t(), backend=_BACKEND)
    torch.cuda.synchronize()
    del output, weight, x


_initialize_fixed_shape_at_import()


def wk_backend(x, weight):
    return flashinfer.mm_bf16(x, weight.t(), backend=_BACKEND)


def run(inputs, runtime):
    if runtime.workload.family == "bf16_linear":
        return wk_backend(inputs["x"], inputs["weight"])
    return runtime.indexer_fused_prepare_store(inputs, wk_backend=wk_backend)
