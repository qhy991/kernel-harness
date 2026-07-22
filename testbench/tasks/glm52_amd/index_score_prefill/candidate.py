"""GLM-5.2 Indexer Score (MQA logits) (prefill) — the one file to edit for this task.

Platform: AMD (this task is on the amd-only tree
`testbench/tasks/glm52_amd/`; the sibling platform lives under
`glm52_cuda/`).

This file is the DEFAULT candidate, not the only one: `./run.sh --candidate PATH`
tests any .py defining run(inputs), from anywhere on disk, without touching the task.
Editing this file is just the convenient path.

Run `./run.sh --describe` for the full contract. The short version:

`inputs` is the frozen dict from glm52_ops.build_inputs. The very same dict feeds
the reference, so do NOT re-quantize, re-seed, or rebuild any tensor inside
run() — that would measure a different problem than the one the gate checked.

Tensors at M=1024:

    q_fp8            (1024, 32, 128)          torch.float8_e4m3fnuz
    q_scale          (1024, 32)               torch.float32
    k_fp8            (65536, 128)             torch.float8_e4m3fnuz
    k_scale          (65536,)                 torch.float32
    weights          (1024, 32, 1)            torch.float32
    ks               (1024,)                  torch.int32
    ke               (1024,)                  torch.int32

Return the output. Correctness against glm52_ops.reference on these inputs is
FlashMLA's three-layer check: matching inf/nan positions, then every element
abs_err < abs_tol OR rel_err < 0.0157, then DeepGEMM's calc_diff
<= 5e-06. `./run.sh --describe` prints all of it.

Baseline to beat: the call below, timed CUPTI cold-L2 on these same inputs.

    ./run.sh
"""
from __future__ import annotations

# indexer score on MI300X uses aiter.ops.triton.fp8_mqa_logits. weights already
# folds q_scale and softmax_scale (see dsa_indexer.py). Starting from the
# provider's reference() dispatch — replace with a direct aiter call for the win.
from testbench.harness import glm52_ops


def run(inputs: dict):
    return glm52_ops.reference("index_score", "prefill", inputs)
