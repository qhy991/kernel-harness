"""GLM-5.2 DSA Indexer K Norm+RoPE+Paged-Cache Store (fused region) (decode) — the one file to edit for this task.

This file is the DEFAULT candidate, not the only one: `./run.sh --candidate PATH`
tests any .py defining run(inputs), from anywhere on disk, without touching the task.
Editing this file is just the convenient path.

Run `./run.sh --describe` for the full contract. The short version:

`inputs` is the frozen dict from glm52_ops.build_inputs. The very same dict feeds
the reference, so do NOT re-quantize, re-seed, or rebuild any tensor inside
run() — that would measure a different problem than the one the gate checked.

Tensors at M=16:

    k_projection     (16, 160)                torch.bfloat16
    norm_weight      (128,)                   torch.float32
    norm_bias        (128,)                   torch.float32
    cos_sin_cache    (65553, 64)              torch.float32
    positions        (16,)                    torch.int64
    out_cache_loc    (16,)                    torch.int64
    cache            (1025, 8448)             torch.uint8
    _cache0          (1025, 8448)             torch.uint8

Return exactly `(updated_paged_index_k_cache)`; every logical output is gated. Correctness against glm52_ops.reference on these inputs is
FlashMLA's three-layer check: matching inf/nan positions, then every element
abs_err < abs_tol OR rel_err < 0.0157, then DeepGEMM's calc_diff
<= 5e-06. `./run.sh --describe` prints all of it.

The production output buffer(s) are pre-allocated and may be written in place,
but the harness poisons every observable row before calling run(). Returning a
shared buffer unwritten therefore FAILS.

Baseline to beat: the call below, timed CUPTI cold-L2 on these same inputs.

    ./run.sh
"""
from __future__ import annotations

from sglang.jit_kernel.dsv32 import (
    fused_k_indexer_norm_rope_store,
)


def run(inputs: dict):
    # Preserve the production non-contiguous [:, :128] projection view and the
    # exact paged 132-byte (fp8 values + f32 scale) cache ABI.
    cache = inputs["cache"]
    fused_k_indexer_norm_rope_store(
        inputs["k_projection"][:, :128],
        cache,
        inputs["out_cache_loc"],
        inputs["norm_weight"],
        inputs["norm_bias"],
        inputs["eps"],
        inputs["cos_sin_cache"],
        inputs["positions"],
        inputs["page_size"],
    )
    return cache
