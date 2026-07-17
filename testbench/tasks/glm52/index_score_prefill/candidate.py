"""GLM-5.2 Indexer Score (MQA logits) (prefill) — the one file to edit for this task.

Run `./run.sh --describe` for the full contract. The short version:

`inputs` is the frozen dict from glm52_ops.build_inputs. The very same dict feeds
the reference, so do NOT re-quantize, re-seed, or rebuild any tensor inside
run() — that would measure a different problem than the one the gate checked.

Tensors at M=1024:

    q_fp8            (1024, 32, 128)          torch.float8_e4m3fn
    k_fp8            (65536, 128)             torch.float8_e4m3fn
    k_scale          (65536,)                 torch.float32
    weights          (1024, 32)               torch.float32
    ks               (1024,)                  torch.int32
    ke               (1024,)                  torch.int32

Return the output. Correctness is cosine >= 0.999 AND
rel_l2 <= 0.044721 against glm52_ops.reference on these inputs;
cosine alone is scale-blind, so both gate.

Baseline to beat: the call below, timed CUPTI cold-L2 on these same inputs.

    ./run.sh
"""
from __future__ import annotations

import deep_gemm


def run(inputs: dict):
    # Starting point: the reference call itself — correct, speedup ~1.0. Replace it.
    return deep_gemm.fp8_mqa_logits(
        inputs["q_fp8"], (inputs["k_fp8"], inputs["k_scale"]), inputs["weights"],
        inputs["ks"], inputs["ke"], clean_logits=False,
    )
