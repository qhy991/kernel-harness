"""GLM-5.2 DSA Sparse Attention (decode) — the one file to edit for this task.

Run `./run.sh --describe` for the full contract. The short version:

`inputs` is the frozen dict from glm52_ops.build_inputs. The very same dict feeds
the reference, so do NOT re-quantize, re-seed, or rebuild any tensor inside
run() — that would measure a different problem than the one the gate checked.

Tensors at M=16:

    q                (16, 64, 576)            torch.bfloat16
    kv               (65536, 1, 576)          torch.bfloat16
    indices          (16, 1, 2048)            torch.int32

Return the output. Correctness is cosine >= 0.99 AND
rel_l2 <= 0.141421 against glm52_ops.reference on these inputs;
cosine alone is scale-blind, so both gate.

Baseline to beat: the call below, timed CUPTI cold-L2 on these same inputs.

    ./run.sh --repeat 3
"""
from __future__ import annotations

from sgl_kernel.flash_mla import flash_mla_sparse_fwd


def run(inputs: dict):
    # Starting point: the reference call itself — correct, speedup ~1.0. Replace it.
    return flash_mla_sparse_fwd(inputs["q"], inputs["kv"], inputs["indices"],
                                inputs["sm_scale"], inputs["d_v"])
