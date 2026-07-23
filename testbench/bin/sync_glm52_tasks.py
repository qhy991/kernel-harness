#!/usr/bin/env python
"""Regenerate the 24 GLM-5.2 task directories from glm52_ops.

glm52_ops is the only place an operator is defined. This tool projects it onto
the task tree, so a task directory holds nothing it could disagree with:

    task.json       operator/phase/gate only — no restated shapes or thresholds
    problem.json    the full problem definition, machine-readable; == `run.sh
                    --describe --json`. Generated, so it is a projection of
                    glm52_ops rather than a second copy that could contradict it.
    workload.jsonl  the sweep, read from glm52_ops
    candidate.py    the agent's file; NEVER overwritten if it already exists
    run.sh          the single entry point
    README.md       generated verbatim from glm52_ops.describe()

Re-run it after changing glm52_ops and the whole tree follows. Everything it
writes except candidate.py is derived, so there is no second copy to drift.

    python testbench/bin/sync_glm52_tasks.py [--check] [--force-candidate]

--check exits 1 if anything is stale instead of writing (for CI).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_HARNESS = _REPO / "testbench" / "harness"

# The shim imports `testbench.harness.backends` as a package, so the repo root
# must be on sys.path BEFORE _load_ops() is called.
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Platform → (task tree, backend env vars, deployment tag, default candidate templates).
# Each row is what the sync tool applies BEFORE loading glm52_ops so the module
# imports the correct concrete impl (_cuda / _amd) via the shim.
_PLATFORM_ENV = {
    "cuda": {
        "KERNEL_HARNESS_PLATFORM": "cuda",
        "KERNEL_HARNESS_PROFILE":  "cuda-b200",
        "KERNEL_HARNESS_PROVIDER": "deep-gemm-sgl-kernel",
        "KERNEL_HARNESS_TIMER":    "auto",
    },
    "amd": {
        "KERNEL_HARNESS_PLATFORM": "rocm",
        "KERNEL_HARNESS_PROFILE":  "amd-mi300x",
        "KERNEL_HARNESS_PROVIDER": "aiter-torch-reference",
        "KERNEL_HARNESS_TIMER":    "event",
    },
}
_PLATFORM_TASK_DIR = {"cuda": "glm52_cuda", "amd": "glm52_amd"}


def _apply_platform_env(platform: str) -> None:
    """Populate KERNEL_HARNESS_* env vars for the target platform BEFORE
    importing glm52_ops, so the shim routes to the right concrete impl."""
    for k, v in _PLATFORM_ENV[platform].items():
        os.environ[k] = v


def _load_ops():
    """Fresh import of glm52_ops with whatever env is currently active."""
    spec = importlib.util.spec_from_file_location(
        "glm52_ops_sync_target", _HARNESS / "glm52_ops.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ops is set inside main() after --platform is parsed and env is applied.
ops = None

# (directory, operator, phase). Directory names are historical — several predate
# the operator names and are referenced by the kda/* branches, so they are kept
# as-is; task.json's `operator` field carries the real mapping and the runner
# never infers it from the path.
TASKS: list[tuple[str, str, str]] = [
    ("fused_qkv_a_prefill",    "fused_qkv_a",    "prefill"),
    ("fused_qkv_a_decode",     "fused_qkv_a",    "decode"),
    ("q_b_prefill",            "q_b",            "prefill"),
    ("q_b_decode",             "q_b",            "decode"),
    ("o_proj_prefill",         "o_proj",         "prefill"),
    ("o_proj_decode",          "o_proj",         "decode"),
    ("index_q_upproj_prefill", "index_q_upproj", "prefill"),
    ("index_q_upproj_decode",  "index_q_upproj", "decode"),
    ("index_k_prefill",        "index_k",        "prefill"),
    ("index_k_proj_decode",    "index_k",        "decode"),
    ("absorbed_W_UK_prefill",  "absorbed_W_UK",  "prefill"),
    ("absorbed_W_UK_decode",   "absorbed_W_UK",  "decode"),
    ("absorbed_W_UV_prefill",  "absorbed_W_UV",  "prefill"),
    ("absorbed_W_UV_decode",   "absorbed_W_UV",  "decode"),
    ("moe_gate_proj_prefill",  "moe_gate",       "prefill"),
    ("moe_gate_proj_decode",   "moe_gate",       "decode"),
    ("moe_up_proj_prefill",    "moe_up",         "prefill"),
    ("moe_up_proj_decode",     "moe_up",         "decode"),
    ("moe_down_proj_prefill",  "moe_down",       "prefill"),
    ("moe_down_proj_decode",   "moe_down",       "decode"),
    ("moe_total_prefill",      "moe_total",      "prefill"),
    ("moe_total_decode",       "moe_total",      "decode"),
    ("dsa_prefill_attn",       "dsa_attn",       "prefill"),
    ("dsa_attn_decode",        "dsa_attn",       "decode"),
    ("index_score_prefill",    "index_score",    "prefill"),
    ("index_score_decode",     "index_score",    "decode"),
    # Communication family — driven by torchrun / evaluate_comm_task.py,
    # every task requires N GPUs (WORLD_SIZE defaults to 8; override
    # KERNEL_HARNESS_COMM_WORLD_SIZE to use fewer for smoke tests).
    ("all_reduce_prefill",     "all_reduce",     "prefill"),
    ("all_reduce_decode",      "all_reduce",     "decode"),
    ("all_gather_prefill",     "all_gather",     "prefill"),
    ("all_gather_decode",      "all_gather",     "decode"),
    ("deepep_dispatch_prefill", "deepep_dispatch", "prefill"),
    ("deepep_dispatch_decode",  "deepep_dispatch", "decode"),
    ("deepep_combine_prefill",  "deepep_combine",  "prefill"),
    ("deepep_combine_decode",   "deepep_combine",  "decode"),
]

# Files from the superseded stacks. Each was a second definition of the same
# task, and each disagreed with the others: impl.py/verify.py were the opbench
# scaffold (whose cosine-only gate accepts a no-op), solution.py + definition.json
# + reference.py were the legacy evaluate.py stack (different quant helpers,
# different backend, and it padded M).
OBSOLETE = ("impl.py", "verify.py", "solution.py", "definition.json", "reference.py")

RUN_SH_TEMPLATE = '''#!/usr/bin/env bash
# The single entry point for this task.
#
#   ./run.sh --describe          # what is this problem? (generated from glm52_ops)
#   ./run.sh --describe --json   # ...the same thing, machine-readable (== problem.json)
#   ./run.sh                 # full sweep; defaults warmup=3, repeat=10
#   ./run.sh --M {m}         # one shape
#   ./run.sh --repeat 1      # fast probe. CANNOT gate a win.
#
# To test a kernel that is NOT this directory's candidate.py — the usual case, since
# nothing should have to edit the task to be measured:
#
#   ./run.sh --candidate ~/my_kernels/o_proj.py    # any .py defining run(inputs)
#   ./run.sh --candidate ~/my_kernels/             # or a dir holding candidate.py
#
# Exit: 0 correct+fast · 1 correct+not-faster · 2 incorrect · 3 infra/contract error
set -euo pipefail
HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
TESTBENCH="$(cd "$HERE/../../.." && pwd)"
REPO="$(cd "$TESTBENCH/.." && pwd)"
{python_selector}
{env_exports}
exec "$PYTHON" "$TESTBENCH/harness/evaluate_task.py" "$HERE" "$@"
'''

# CUDA runner: default python is repo .venv; env not exported (registry defaults
# already resolve to cuda-b200 when unset).
_RUN_SH_PY_CUDA = 'PYTHON="${REPO}/.venv/bin/python"\nif [[ ! -x "$PYTHON" ]]; then\n  PYTHON="$(command -v python3)"\nfi'
_RUN_SH_ENV_CUDA = ""  # empty — registry defaults are already cuda

# AMD runner: default python is ROCm venv (fallback repo .venv → system python3);
# platform env vars force the shim onto glm52_ops_amd + aiter provider.
_RUN_SH_PY_AMD = (
    'PYTHON="${ROCM_TORCH_PYTHON:-}"\n'
    'if [[ -z "$PYTHON" && -n "${ROCM_TORCH_VENV:-}" && -x "${ROCM_TORCH_VENV}/bin/python" ]]; then\n'
    '  PYTHON="${ROCM_TORCH_VENV}/bin/python"\n'
    'fi\n'
    'if [[ -z "$PYTHON" ]]; then\n'
    '  PYTHON="${REPO}/.venv/bin/python"\n'
    'fi\n'
    'if [[ ! -x "$PYTHON" ]]; then\n'
    '  PYTHON="$(command -v python3)"\n'
    'fi'
)
_RUN_SH_ENV_AMD = (
    'export KERNEL_HARNESS_PLATFORM="${KERNEL_HARNESS_PLATFORM:-rocm}"\n'
    'export KERNEL_HARNESS_PROFILE="${KERNEL_HARNESS_PROFILE:-amd-mi300x}"\n'
    'export KERNEL_HARNESS_PROVIDER="${KERNEL_HARNESS_PROVIDER:-aiter-torch-reference}"\n'
    'export KERNEL_HARNESS_TIMER="${KERNEL_HARNESS_TIMER:-event}"\n'
    'export SGLANG_USE_AITER="${SGLANG_USE_AITER:-1}"'
)

_RUN_SH_PARTS = {
    "cuda": (_RUN_SH_PY_CUDA, _RUN_SH_ENV_CUDA),
    "amd":  (_RUN_SH_PY_AMD,  _RUN_SH_ENV_AMD),
}


# Multi-process run.sh for comm / deepep tasks — dispatches through torchrun
# so every rank shares a process group. WORLD_SIZE defaults to 8; override
# KERNEL_HARNESS_COMM_WORLD_SIZE to use fewer (e.g. 2 for a smoke test on a
# node where some GPUs are unavailable).
COMM_RUN_SH_TEMPLATE = '''#!/usr/bin/env bash
# Multi-process entry point for this communication task.
# Every rank runs its own Python; the process group is the collective.
#
#   ./run.sh --describe                   # what is this problem? (single process; no dist)
#   ./run.sh                              # full sweep (WORLD_SIZE default 8)
#   ./run.sh --M {m}                      # one shape
#   ./run.sh --repeat 1                   # fast probe
#   KERNEL_HARNESS_COMM_WORLD_SIZE=4 ./run.sh   # 4-GPU smoke on a partial node
#   ./run.sh --candidate ~/my_kernel.py   # any .py defining run(inputs)
#
# Exit: 0 correct+fast · 1 correct+not-faster · 2 incorrect · 3 infra
set -euo pipefail
HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
TESTBENCH="$(cd "$HERE/../../.." && pwd)"
REPO="$(cd "$TESTBENCH/.." && pwd)"
{python_selector}
{env_exports}
WORLD_SIZE="${{KERNEL_HARNESS_COMM_WORLD_SIZE:-8}}"

# --describe short-circuits the multi-process launch — the problem statement
# is single-process readable.
if [[ "${{1:-}}" == "--describe" ]]; then
  exec "$PYTHON" "$TESTBENCH/harness/evaluate_comm_task.py" "$HERE" "$@"
fi

# All ranks return 0 to torchrun (avoids ChildFailedError banner); rank 0
# writes runs/comm/<task>/last_exit_code with the real 0/1/2/3 verdict.
TASK_NAME="$(basename "$HERE")"
LAST_EXIT="$REPO/runs/comm/$TASK_NAME/last_exit_code"
rm -f "$LAST_EXIT"

"$PYTHON" -m torch.distributed.run \\
  --standalone --nproc-per-node="$WORLD_SIZE" \\
  "$TESTBENCH/harness/evaluate_comm_task.py" "$HERE" "$@"

# Reconstitute the verdict from the file rank 0 wrote.
if [[ -f "$LAST_EXIT" ]]; then
  exit "$(cat "$LAST_EXIT")"
fi
echo "[run.sh] rank 0 did not write $LAST_EXIT — treating as infra error" >&2
exit 3
'''


def _run_sh(platform: str, m: int, is_comm: bool = False) -> str:
    py, env = _RUN_SH_PARTS[platform]
    template = COMM_RUN_SH_TEMPLATE if is_comm else RUN_SH_TEMPLATE
    return template.format(m=m, python_selector=py, env_exports=env)

# Per-platform, per-family default candidate: the real backend call spelled out,
# so the agent starts from the baseline it has to beat rather than from an
# indirection. CUDA uses deep_gemm / sgl_kernel; AMD uses aiter / hipBLASLt.
#
# The AMD templates fall back to `glm52_ops.reference` for anything the local
# aiter build might not expose. That is still a real production dispatch (the
# provider's `.reference()` method), and it means the sync always writes a
# candidate that runs — the agent is free to replace it with a direct aiter
# call once they confirm which entry points their aiter has.
_BODY_CUDA = {
    "gemm": '''import deep_gemm


def run(inputs: dict):
    # Starting point: the reference call itself — correct, speedup ~1.0. Replace it.
    out = inputs["out"]
    deep_gemm.fp8_gemm_nt(
        (inputs["x_fp8"], inputs["x_scale"]),
        (inputs["w_fp8"], inputs["w_scale"]),
        out,
    )
    return out
''',
    "bmm": '''import torch
from sgl_kernel import bmm_fp8


def run(inputs: dict):
    # Starting point: the reference call itself — correct, speedup ~1.0. Replace it.
    return bmm_fp8(inputs["A_fp8"], inputs["B_fp8"],
                   inputs["A_scale"], inputs["B_scale"], torch.bfloat16)
''',
    "moe": '''import deep_gemm


def run(inputs: dict):
    # Starting point: the reference call itself — correct, speedup ~1.0. Replace it.
    out = inputs["out"]
    deep_gemm.fp8_m_grouped_gemm_nt_masked(
        (inputs["x_fp8"], inputs["x_scale"]),
        (inputs["w_fp8"], inputs["w_scale"]),
        out, inputs["masked_m"], inputs["expected_m"],
    )
    return out
''',
    "mla": '''from sgl_kernel.flash_mla import flash_mla_sparse_fwd


def run(inputs: dict):
    # Starting point: the reference call itself — correct, speedup ~1.0. Replace it.
    return flash_mla_sparse_fwd(inputs["q"], inputs["kv"], inputs["indices"],
                                inputs["sm_scale"], inputs["d_v"])
''',
    "moe_fused": '''from testbench.harness import glm52_ops


def run(inputs: dict):
    # Starting point: the SGLang fused MoE reference — correct, speedup ~1.0.
    # Replace with a fused MoE kernel to beat it (e.g. a Triton fused_moe with
    # your own tuning table).
    return glm52_ops.reference("moe_total", "prefill", inputs)
''',
    "score_prefill": '''import deep_gemm


def run(inputs: dict):
    # Starting point: the reference call itself — correct, speedup ~1.0. Replace it.
    return deep_gemm.fp8_mqa_logits(
        inputs["q_fp8"], (inputs["k_fp8"], inputs["k_scale"]), inputs["weights"],
        inputs["ks"], inputs["ke"], clean_logits=False,
    )
''',
    "score_decode": '''import deep_gemm


def run(inputs: dict):
    # Starting point: the reference call itself — correct, speedup ~1.0. Replace it.
    return deep_gemm.fp8_paged_mqa_logits(
        inputs["q_fp8"], inputs["kv_cache_fp8"], inputs["weights"], inputs["seqlens"],
        inputs["block_tables"], inputs["schedule_metadata"], inputs["max_seq_len"],
        clean_logits=False,
    )
''',
    # ── Communication ops (multi-process; run via torchrun) ─────────────
    "comm_all_reduce": '''import torch
import torch.distributed as dist


def run(inputs: dict):
    """B200 TP AllReduce reference — uses NCCL through torch.distributed.
    Replace with sglang.srt.distributed.device_communicators.custom_all_reduce
    (CustomAllreduce) or a one-shot NVLINK kernel to beat NCCL SUM."""
    out = inputs["x"].clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    return out
''',
    "comm_all_gather": '''import torch
import torch.distributed as dist


def run(inputs: dict):
    """B200 TP AllGather reference — NCCL all_gather across ranks."""
    x = inputs["x"]
    ws = inputs["world_size"]
    gather = [torch.empty_like(x) for _ in range(ws)]
    dist.all_gather(gather, x)
    return torch.cat(gather, dim=0)
''',
    # DeepEP dispatch/combine — CUDA production path is nvshmem-based
    # DeepEP kernels. Starting from an emulation via torch.distributed so
    # the task runs without deep_ep installed; replace with real DeepEP
    # dispatch/combine to beat it.
    "deepep_dispatch": '''# Starting from the reference emulation. Replace with a real DeepEP
# dispatch (deep_ep.Buffer(...).dispatch(...)) for the win.
from testbench.harness import glm52_ops


def run(inputs: dict):
    return glm52_ops.reference("deepep_dispatch", "prefill", inputs)
''',
    "deepep_combine": '''from testbench.harness import glm52_ops


def run(inputs: dict):
    return glm52_ops.reference("deepep_combine", "prefill", inputs)
''',
}

# AMD templates: aiter's production path. For families where aiter's precise
# entry point varies across builds, we start from a `glm52_ops.reference`
# indirection — that still dispatches to the aiter provider, so timing is real
# and correctness is by construction; the agent replaces the body with a direct
# aiter call once they see what their aiter build exposes.
_BODY_AMD = {
    "gemm": '''from aiter.ops.triton.gemm_a8w8_blockscale import gemm_a8w8_blockscale
import torch


def run(inputs: dict):
    # Starting point: aiter's Triton blockscale FP8 GEMM (a fallback path — sglang's
    # default gfx942 dispatch is bpreshuffle_asm/ck, but the ASM build is not always
    # available on every node). Replace with a direct MFMA kernel to beat it.
    out = gemm_a8w8_blockscale(inputs["x_fp8"], inputs["w_fp8"],
                               inputs["x_scale"], inputs["w_scale"],
                               dtype=torch.bfloat16)
    inputs["out"].copy_(out)
    return inputs["out"]
''',
    "bmm": '''import torch

# absorbed_W_UK / _UV BMM on MI300X. There is no fused FP8 BMM on gfx942 — sglang's
# production path loops per-head torch._scaled_mm (hipBLASLt). This candidate is
# the reference call itself; replace with a batched kernel (MFMA head-folding) to win.
from testbench.harness import glm52_ops


def run(inputs: dict):
    return glm52_ops.reference("absorbed_W_UK", "prefill", inputs)  # phase inferred by shape
''',
    "moe": '''from aiter.fused_moe import fused_moe
from aiter import ActivationType, QuantType
import torch


def run(inputs: dict):
    # aiter's fused_moe consumes per-tensor scales on gfx942. If your aiter build
    # uses a different signature, fall back to glm52_ops.reference in the outer
    # dispatch and shape-branch here for the wins.
    out = fused_moe(
        inputs["x_fp8"], inputs["w_fp8"], None,
        inputs["masked_m"], inputs["masked_m"],
        activation=ActivationType.Silu, quant_type=QuantType.per_1x128,
        w1_scale=inputs["x_scale"], w2_scale=inputs["w_scale"],
    )
    inputs["out"].copy_(out)
    return inputs["out"]
''',
    "mla": '''# sparse-MLA on MI300X. Production sglang path: aiter.mla.mla_decode_fwd
# (ASM stage1 + reduce). The reference() dispatch below IS that path when the
# provider is aiter-torch-reference; replace with a direct aiter.mla call or a
# tk-split flash-decode + fused combine kernel to beat it.
from testbench.harness import glm52_ops


def run(inputs: dict):
    return glm52_ops.reference("dsa_attn", inputs.get("_phase", "decode"), inputs)
''',
    "moe_fused": '''# Fused MoE total on MI300X — sglang's production path is
# sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe. Starting from the
# provider reference dispatch — replace with a direct fused kernel to win.
from testbench.harness import glm52_ops


def run(inputs: dict):
    return glm52_ops.reference("moe_total", "prefill", inputs)
''',
    "score_prefill": '''# indexer score on MI300X uses aiter.ops.triton.fp8_mqa_logits. weights already
# folds q_scale and softmax_scale (see dsa_indexer.py). Starting from the
# provider's reference() dispatch — replace with a direct aiter call for the win.
from testbench.harness import glm52_ops


def run(inputs: dict):
    return glm52_ops.reference("index_score", "prefill", inputs)
''',
    "score_decode": '''# AMD decode index_score also uses ksrange (NOT paged): same signature as prefill.
# The provider reference here dispatches to aiter.ops.triton.fp8_mqa_logits.
from testbench.harness import glm52_ops


def run(inputs: dict):
    return glm52_ops.reference("index_score", "decode", inputs)
''',
    # ── Communication ops (multi-process; torchrun) ───────────────────
    "comm_all_reduce": '''import torch
import torch.distributed as dist


def run(inputs: dict):
    """MI300X TP AllReduce reference — RCCL via torch.distributed.
    Replace with aiter.dist.device_communicators.custom_all_reduce
    (AiterCustomAllreduce, 2-stage cross_device_reduce) to beat RCCL SUM.
    That is exactly what sglang production dispatches when
    SGLANG_USE_AITER_AR=true (the default on gfx942)."""
    out = inputs["x"].clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    return out
''',
    "comm_all_gather": '''import torch
import torch.distributed as dist


def run(inputs: dict):
    """MI300X TP AllGather reference — RCCL all_gather across ranks."""
    x = inputs["x"]
    ws = inputs["world_size"]
    gather = [torch.empty_like(x) for _ in range(ws)]
    dist.all_gather(gather, x)
    return torch.cat(gather, dim=0)
''',
    "deepep_dispatch": '''# Starting from the reference emulation. Replace with a real DeepEP
# (or aiter EP) dispatch call for the win. sglang gfx942 does NOT use
# DeepEP today (fused_moe absorbs the dispatch), so this task is a
# reserved slot for when EP>1 deployments land.
from testbench.harness import glm52_ops


def run(inputs: dict):
    return glm52_ops.reference("deepep_dispatch", "prefill", inputs)
''',
    "deepep_combine": '''from testbench.harness import glm52_ops


def run(inputs: dict):
    return glm52_ops.reference("deepep_combine", "prefill", inputs)
''',
}

_BODY_BY_PLATFORM = {"cuda": _BODY_CUDA, "amd": _BODY_AMD}


def _candidate_src(op: str, phase: str, device, platform: str) -> str:
    s = ops.spec(op, phase)
    fam = s["family"]
    # Family → template key. Score is phase-split (paged decode vs ksrange prefill);
    # comm is op-split (all_reduce vs all_gather). Everything else is one key per family.
    if fam == "score":
        key = f"score_{phase}"
    elif fam == "comm":
        key = f"comm_{op}"           # comm_all_reduce / comm_all_gather
    elif fam == "deepep":
        key = op                     # deepep_dispatch / deepep_combine
    else:
        key = fam
    tensors = []
    try:
        ins = ops.build_inputs(op, phase, s["sweep"][0], s["S"], device, s["seed"])
        import torch
        for k, v in ins.items():
            if torch.is_tensor(v):
                tensors.append(f"    {k:<16} {str(tuple(v.shape)):<24} {v.dtype}")
    except Exception:
        pass
    table = "\n".join(tensors) or "    (run ./run.sh --describe on a GPU node for the tensor table)"
    doc = f'''"""GLM-5.2 {s['label']} ({phase}) — the one file to edit for this task.

Platform: {platform.upper()} (this task is on the {platform}-only tree
`testbench/tasks/glm52_{platform}/`; the sibling platform lives under
`glm52_{"amd" if platform == "cuda" else "cuda"}/`).

This file is the DEFAULT candidate, not the only one: `./run.sh --candidate PATH`
tests any .py defining run(inputs), from anywhere on disk, without touching the task.
Editing this file is just the convenient path.

Run `./run.sh --describe` for the full contract. The short version:

`inputs` is the frozen dict from glm52_ops.build_inputs. The very same dict feeds
the reference, so do NOT re-quantize, re-seed, or rebuild any tensor inside
run() — that would measure a different problem than the one the gate checked.

Tensors at M={s['sweep'][0]}:

{table}

Return the output. Correctness against glm52_ops.reference on these inputs is
FlashMLA's three-layer check: matching inf/nan positions, then every element
abs_err < abs_tol OR rel_err < {s['rel_tol']:.4f}, then DeepGEMM's calc_diff
<= {s['diff_tol']:.0e}. `./run.sh --describe` prints all of it.
'''
    if s["has_output_buffer"]:
        doc += '''
`inputs["out"]` is pre-allocated and may be written in place, but the harness
NaN-poisons it before calling run(): returning it unwritten FAILS.
'''
    doc += f'''
Baseline to beat: the call below, timed CUPTI cold-L2 on these same inputs.

    ./run.sh
"""
from __future__ import annotations

'''
    return doc + _BODY_BY_PLATFORM[platform][key]


def _task_json(dirname: str, op: str, phase: str, platform: str) -> str:
    s = ops.spec(op, phase)
    return json.dumps({
        "_note": ("Generated by testbench/bin/sync_glm52_tasks.py. Declares only WHICH "
                  "problem this is and how hard the bar is. Shapes, dtypes, inputs, the "
                  "reference kernel, thresholds, masks, the cost model, the peaks and the "
                  "timing protocol all live in testbench/harness/glm52_ops.py and are NOT "
                  "restated here — this file has nothing it could lie about. Run "
                  "`./run.sh --describe` for the real contract."),
        "name": dirname,
        "model": "glm52",
        "platform": platform,
        "operator": op,
        "phase": phase,
        # Generated mirror of glm52_ops.spec(op, phase)["family"]. It exists only
        # so stdlib-only tools (inventory.py, selftest.py) can route without
        # importing torch. evaluate_task re-derives it and exits 3 on any
        # disagreement, so it cannot drift — unlike the fields task.json is
        # forbidden from restating, which nothing would check.
        "family": s["family"],
        "goal": (f"Optimize candidate.py for GLM-5.2 {s['label']} ({phase}). It must match "
                 f"glm52_ops.reference on every shape, and beat its latency on at least "
                 f"one shape without regressing on any. Run `./run.sh --describe` for the "
                 f"contract."),
        "entrypoint": "candidate.py",
        "runner": "testbench/harness/evaluate_task.py",
        "deployment": ops.DEVICE_PROFILE.deployment,
        "performance_gate": {
            "min_speedup": 1.0,
            "basis": "conservative",
            "detail": ("a shape WINS when reference_p10 / candidate_p90 > min_speedup "
                       "and REGRESSES when reference_p90 / candidate_p10 < 1.0. The "
                       "run passes with >=1 win and 0 regressions; shapes inside the "
                       "noise band are neutral and do not veto. run() may fall back to "
                       "glm52_ops.reference on shapes it cannot win — see "
                       "`./run.sh --describe`."),
        },
        "verdict": {
            "exit_0": "correct on every shape, at least one shape wins, and no shape regresses",
            "exit_1": "correct on every shape, performance gate not met",
            "exit_2": "incorrect, incomplete sweep, or correctness did not survive timing",
            "exit_3": "infrastructure error, or task.json disagrees with glm52_ops",
            "terminal_state": {
                "COMPLETE_WIN": "exit 0; correct complete sweep with >=1 win and 0 regressions",
                "NO_WIN_WITH_EVIDENCE": "exit 1; correct complete sweep but every shape is neutral",
                "PARTIAL_OR_REGRESSED_WITH_EVIDENCE": "exit 1; correct complete sweep but at least one shape regressed",
                "INCORRECT_OR_INCOMPLETE": "exit 2; correctness failed, the sweep was incomplete, or post-timing correctness failed",
            },
        },
    }, indent=2) + "\n"


def _workload(dirname: str, op: str, phase: str) -> str:
    return "".join(
        json.dumps({"uuid": f"glm52-{dirname}-M{m}", "axes": {"M": m}},
                   separators=(",", ":")) + "\n"
        for m in ops.spec(op, phase)["sweep"])


def _readme(dirname: str, op: str, phase: str, device) -> str:
    return (f"# glm52 / {dirname}\n\n"
            "Generated by `testbench/bin/sync_glm52_tasks.py` from "
            "`testbench/harness/glm52_ops.py` — do not edit by hand; edit the module and "
            "re-run the sync. `./run.sh --describe` prints this same text live.\n\n"
            "```text\n" + ops.describe(op, phase, device=device) + "\n```\n")


def _problem_json(dirname: str, op: str, phase: str, device, tasks_root: Path) -> str:
    """Project problem(); when no GPU, keep previously captured tensor tables."""
    problem = ops.problem(op, phase, device)
    if device is None:
        prev_path = tasks_root / dirname / "problem.json"
        if prev_path.is_file():
            try:
                prev = json.loads(prev_path.read_text())
                prev_tensors = (prev.get("contract") or {}).get("tensors")
                if prev_tensors and not (problem.get("contract") or {}).get("tensors"):
                    problem.setdefault("contract", {})["tensors"] = prev_tensors
                    problem["contract"]["tensors_error"] = None
            except Exception:
                pass
    return json.dumps(problem, indent=2) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", choices=("cuda", "amd"), default="cuda",
                    help="target task tree: cuda → glm52_cuda/, amd → glm52_amd/. "
                         "Sets KERNEL_HARNESS_* env vars before loading glm52_ops, "
                         "so the shim routes to the concrete impl.")
    ap.add_argument("--check", action="store_true", help="exit 1 if stale; write nothing")
    ap.add_argument("--force-candidate", action="store_true",
                    help="overwrite candidate.py too (DESTROYS agent work)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    platform = args.platform
    _apply_platform_env(platform)
    global ops
    ops = _load_ops()

    tasks_root = _REPO / "testbench" / "tasks" / _PLATFORM_TASK_DIR[platform]

    import torch
    device = args.device if torch.cuda.is_available() else None
    if device is None:
        print("warning: no CUDA — README tensor tables will be omitted", file=sys.stderr)

    stale, wrote, removed = [], 0, 0
    for dirname, op, phase in TASKS:
        d = tasks_root / dirname
        d.mkdir(parents=True, exist_ok=True)
        want = {
            "task.json": _task_json(dirname, op, phase, platform),
            "problem.json": _problem_json(dirname, op, phase, device, tasks_root),
            "workload.jsonl": _workload(dirname, op, phase),
            "run.sh": _run_sh(platform, ops.spec(op, phase)["sweep"][0],
                              is_comm=ops.spec(op, phase)["family"] in ("comm", "deepep")),
            "README.md": _readme(dirname, op, phase, device),
        }
        cand = d / "candidate.py"
        if args.force_candidate or not cand.is_file():
            want["candidate.py"] = _candidate_src(op, phase, device, platform)

        for name, text in want.items():
            p = d / name
            if p.is_file() and p.read_text() == text:
                continue
            if args.check:
                stale.append(str(p.relative_to(_REPO)))
                continue
            p.write_text(text)
            if name == "run.sh":
                p.chmod(0o755)
            wrote += 1

        for name in OBSOLETE:
            p = d / name
            if p.exists():
                if args.check:
                    stale.append(f"{p.relative_to(_REPO)} (obsolete, should be deleted)")
                else:
                    p.unlink()
                    removed += 1
        pyc = d / "__pycache__"
        if pyc.exists() and not args.check:
            shutil.rmtree(pyc)

    if args.check:
        if stale:
            print(f"STALE ({len(stale)}) [{platform}]:")
            for s in stale:
                print(f"  {s}")
            return 1
        print(f"{len(TASKS)} {platform} task dirs are in sync with glm52_ops")
        return 0
    print(f"[{platform}] {len(TASKS)} task dirs synced: {wrote} files written, "
          f"{removed} obsolete removed  →  {tasks_root.relative_to(_REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
