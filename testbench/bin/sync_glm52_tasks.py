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
import shutil
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_HARNESS = _REPO / "testbench" / "harness"
_TASKS = _REPO / "testbench" / "tasks" / "glm52"

_spec = importlib.util.spec_from_file_location("glm52_ops", _HARNESS / "glm52_ops.py")
ops = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ops)

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
    ("dsa_prefill_attn",       "dsa_attn",       "prefill"),
    ("dsa_attn_decode",        "dsa_attn",       "decode"),
    ("index_score_prefill",    "index_score",    "prefill"),
    ("index_score_decode",     "index_score",    "decode"),
]

# Files from the superseded stacks. Each was a second definition of the same
# task, and each disagreed with the others: impl.py/verify.py were the opbench
# scaffold (whose cosine-only gate accepts a no-op), solution.py + definition.json
# + reference.py were the legacy evaluate.py stack (different quant helpers,
# different backend, and it padded M).
OBSOLETE = ("impl.py", "verify.py", "solution.py", "definition.json", "reference.py")

RUN_SH = '''#!/usr/bin/env bash
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
PYTHON="${{REPO}}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3)"
fi
exec "$PYTHON" "$TESTBENCH/harness/evaluate_task.py" "$HERE" "$@"
'''

# Per-family default candidate: the real backend call, spelled out, so the agent
# starts from the baseline it has to beat rather than from an indirection.
_BODY = {
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
}


def _candidate_src(op: str, phase: str, device) -> str:
    s = ops.spec(op, phase)
    fam = s["family"]
    key = f"score_{phase}" if fam == "score" else fam
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

This file is the DEFAULT candidate, not the only one: `./run.sh --candidate PATH`
tests any .py defining run(inputs), from anywhere on disk, without touching the task.
Editing this file is just the convenient path.

Run `./run.sh --describe` for the full contract. The short version:

`inputs` is the frozen dict from glm52_ops.build_inputs. The very same dict feeds
the reference, so do NOT re-quantize, re-seed, or rebuild any tensor inside
run() — that would measure a different problem than the one the gate checked.

Tensors at M={s['sweep'][0]}:

{table}

Return the output. Correctness is cosine >= {s['cosine_threshold']} AND
rel_l2 <= {s['rel_l2_threshold']:.6f} against glm52_ops.reference on these inputs;
cosine alone is scale-blind, so both gate.
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
    return doc + _BODY[key]


def _task_json(dirname: str, op: str, phase: str) -> str:
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
        "operator": op,
        "phase": phase,
        # Generated mirror of glm52_ops.spec(op, phase)["family"]. It exists only
        # so stdlib-only tools (inventory.py, selftest.py) can route without
        # importing torch. evaluate_task re-derives it and exits 3 on any
        # disagreement, so it cannot drift — unlike the fields task.json is
        # forbidden from restating, which nothing would check.
        "family": s["family"],
        "goal": (f"Optimize candidate.py for GLM-5.2 {s['label']} ({phase}). It must match "
                 f"glm52_ops.reference under the cosine AND rel-L2 gates on every shape, "
                 f"AND beat its latency on every shape."),
        "entrypoint": "candidate.py",
        "runner": "testbench/harness/evaluate_task.py",
        "deployment": "B200-DP1-TP1-EP32",
        "performance_gate": {
            "min_speedup": 1.0,
            "basis": "conservative",
            "detail": ("min over shapes of (reference fastest sample / candidate slowest "
                       "sample) > min_speedup, mirroring testbench/bin/evaluate.py"),
        },
        "verdict": {
            "exit_0": "correct on every shape AND performance gate met",
            "exit_1": "correct on every shape, performance gate not met",
            "exit_2": "incorrect, incomplete sweep, or correctness did not survive timing",
            "exit_3": "infrastructure error, or task.json disagrees with glm52_ops",
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="exit 1 if stale; write nothing")
    ap.add_argument("--force-candidate", action="store_true",
                    help="overwrite candidate.py too (DESTROYS agent work)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    import torch
    device = args.device if torch.cuda.is_available() else None
    if device is None:
        print("warning: no CUDA — README tensor tables will be omitted", file=sys.stderr)

    stale, wrote, removed = [], 0, 0
    for dirname, op, phase in TASKS:
        d = _TASKS / dirname
        d.mkdir(parents=True, exist_ok=True)
        want = {
            "task.json": _task_json(dirname, op, phase),
            "problem.json": json.dumps(ops.problem(op, phase, device), indent=2) + "\n",
            "workload.jsonl": _workload(dirname, op, phase),
            "run.sh": RUN_SH.format(m=ops.spec(op, phase)["sweep"][0]),
            "README.md": _readme(dirname, op, phase, device),
        }
        cand = d / "candidate.py"
        if args.force_candidate or not cand.is_file():
            want["candidate.py"] = _candidate_src(op, phase, device)

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
            print(f"STALE ({len(stale)}):")
            for s in stale:
                print(f"  {s}")
            return 1
        print(f"{len(TASKS)} task dirs are in sync with glm52_ops")
        return 0
    print(f"{len(TASKS)} task dirs synced: {wrote} files written, {removed} obsolete removed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
