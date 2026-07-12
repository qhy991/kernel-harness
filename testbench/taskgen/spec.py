"""Declarative task spec + the single canonical writer for every kernel task.

A `TaskSpec` describes ONE (op, phase) kernel task: its axes, inputs, outputs, sweep,
tolerance, and metadata. `write_task()` turns a spec into the 6 on-disk files
(definition.json, reference.py, solution.py, workload.jsonl, task.json, kersor-note.txt).

Every family in this repo is now just a function that yields `TaskSpec`s — the ~30
hand-rolled `emit_*` functions collapsed into data. Adding a kernel = declare a spec,
not copy 40 lines of boilerplate.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# --- shared sweeps + default tolerance (single source of truth) ---------------
PREFILL_SWEEP = [512, 1024, 2048, 4096, 8192, 16384, 32768]
DECODE_SWEEP = [1, 2, 4, 8, 16, 32, 64, 128, 256]
ROUTER_SWEEP = [1, 2, 4, 8, 16]                     # dsv3_router_gemm caps num_tokens <= 16
MASKED_SWEEP = [1, 2, 4, 8, 16, 32, 64, 128]        # tokens/expert on the DeepEP masked path

TOLERANCE = {"max_atol": 0.1, "max_rtol": 0.05, "required_matched_ratio": 0.98}
EXACT_TOL = {"max_atol": 0.0, "max_rtol": 0.0, "required_matched_ratio": 1.0}


def sweep_for(phase: str) -> list[int]:
    return PREFILL_SWEEP if phase == "prefill" else DECODE_SWEEP


# --- axis / tensor helpers: tiny constructors so specs read like a schema -----
def var(desc: str = "") -> dict:
    return {"type": "var", "description": desc} if desc else {"type": "var"}


def const(value: int, desc: str = "") -> dict:
    d = {"type": "const", "value": value}
    if desc:
        d["description"] = desc
    return d


def expr(expression: str, desc: str = "") -> dict:
    d = {"type": "expr", "expression": expression}
    if desc:
        d["description"] = desc
    return d


def tensor(shape: Optional[list], dtype: str) -> dict:
    return {"shape": shape, "dtype": dtype}


@dataclass
class TaskSpec:
    """One (op, phase) kernel task. `recipe` is the family's reference.py source."""
    model: str                       # "kimi_k27" | "minimax_m3" — also the tasks/ subdir
    name: str                        # task dir name, e.g. "q_b_decode"
    op: str                          # human-readable op, e.g. "Q_b"
    family: str                      # e.g. "fp8-linear-gemm"
    hf_id: str
    recipe: str                      # reference.py source text
    axes: dict[str, dict]
    inputs: dict[str, dict]
    outputs: dict[str, dict]
    sweep: list[int]
    description: str
    goal: str
    phase: str = "decode"
    backend: str = ""
    tolerance: dict = field(default_factory=lambda: dict(TOLERANCE))
    meta: dict[str, Any] = field(default_factory=dict)   # extra task.json fields
    baseline_us: Optional[float] = None                  # fp8 legacy csv wall-clock
    sglang_dir: Optional[str] = None                     # per-task sglang build override


# --- kersor note (unchanged behavior, single copy) ----------------------------
def kersor_note(task_dir: Path, name: str, goal: str, solexec: str,
                csv_wallclock_us: Optional[float] = None) -> str:
    t = str(task_dir)
    out = f"/tmp/kersor-tb/{name}"
    run = f"cd {solexec} && python scripts/run_dataset.py {t}"
    lines = [
        "language=cuda",
        f"task_dir={t}",
        f"Correctness Command: {run} --solution-name solution.py --rerun -o {out}-correctness",
        f"Benchmark Command: {run} --solution-name solution.py --rerun -o {out}-benchmark",
        f"Baseline Command: {run} --solution-name reference.py --rerun -o {out}-baseline",
        "Baseline: reference.py",
        "Baseline Status: present (measured live each run; do NOT freeze a number)",
        f"candidate solution file: {t}/solution.py",
        "每轮新候选必须先写入该 solution.py 再跑 Correctness/Benchmark 命令",
        "Tolerance: see workload.jsonl (max_atol/max_rtol/required_matched_ratio)",
        "Target GPU: NVIDIA B200",
        "Timing Method: cupti device-kernel time (median, L2 cleared per iter)",
        "Integration Pattern: standalone",
    ]
    if csv_wallclock_us is not None:
        lines.append(f"CSV wall-clock reference (launch-bound, NOT the denominator): {csv_wallclock_us:.1f}us")
    lines.append(f"Goal: {goal}")
    return "; ".join(lines)


# --- the ONE writer every family goes through ---------------------------------
def write_task(spec: TaskSpec, tasks_root: Path, solexec: str) -> str:
    """Materialize a TaskSpec into its 6 files under tasks_root/<model>/<name>/."""
    d = tasks_root / spec.model / spec.name
    d.mkdir(parents=True, exist_ok=True)

    definition = {
        "name": f"{spec.model}_{spec.name}",
        "hf_id": spec.hf_id,
        "description": spec.description,
        "axes": spec.axes,
        "custom_inputs_entrypoint": "get_inputs",
        "inputs": spec.inputs,
        "outputs": spec.outputs,
        "reference": spec.recipe,
    }
    (d / "definition.json").write_text(json.dumps(definition, indent=4))
    (d / "reference.py").write_text(spec.recipe)
    (d / "solution.py").write_text(spec.recipe)

    inp = {k: {"type": "custom"} for k in spec.inputs}
    lines = [json.dumps({"uuid": f"{spec.model}-{spec.name}-{m:05d}", "axes": {"M": m},
                         "inputs": inp, "tolerance": spec.tolerance}) for m in spec.sweep]
    (d / "workload.jsonl").write_text("\n".join(lines) + "\n")

    (d / "kersor-note.txt").write_text(
        kersor_note(d, spec.name, spec.goal, solexec, spec.baseline_us) + "\n")

    task_json = {
        "name": spec.name, "op": spec.op, "phase": spec.phase, "family": spec.family,
        **spec.meta,
        "sweep": spec.sweep,
        "backend": spec.backend,
        "tolerance": spec.tolerance,
        "baseline": {
            "source": "live",
            "measure": "reference.py timed by sol-execbench CUPTI (median device-kernel us)",
            "denominator": "the live per-shape reference latency (see .baseline_cache.json)",
        },
    }
    if spec.baseline_us is not None:
        task_json["csv_wallclock_us_reference"] = spec.baseline_us
    if spec.sglang_dir is not None:
        task_json["sglang_dir"] = spec.sglang_dir
    (d / "task.json").write_text(json.dumps(task_json, indent=2))
    return spec.name
