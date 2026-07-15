"""Declarative task spec + the single canonical writer for every kernel task.

A `TaskSpec` describes ONE (op, phase) kernel task: its axes, inputs, outputs, sweep,
tolerance, and metadata. `write_task()` turns a spec into the five on-disk task files
(definition.json, reference.py, solution.py, workload.jsonl, task.json).

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

# GLM-5.2 focused sweeps (plan: prefill seq={1024,2048,4096}, decode bs={16,32})
GLM52_PREFILL_SWEEP = [1024, 2048, 4096]
GLM52_DECODE_SWEEP = [16, 32]
GLM52_SPARSE_MLA_CTX = [1024, 2048, 4096, 8192, 32768]

BF16_TOL = {"max_atol": 0.02, "max_rtol": 0.01, "required_matched_ratio": 0.999}
FP8_TOL = {"max_atol": 0.1, "max_rtol": 0.05, "required_matched_ratio": 0.999}
NVFP4_TOL = {"max_atol": 0.2, "max_rtol": 0.1, "required_matched_ratio": 0.99}
ATTENTION_TOL = {"max_atol": 0.03, "max_rtol": 0.02, "required_matched_ratio": 0.999}
# FP8 DSA sparse MLA (SGLang DSA unit-test headroom for FP8 KV)
SPARSE_MLA_FP8_TOL = {"max_atol": 0.2, "max_rtol": 0.2, "required_matched_ratio": 0.99}
EXACT_TOL = {"max_atol": 0.0, "max_rtol": 0.0, "required_matched_ratio": 1.0}

_FP8_FAMILIES = {
    "fp8-linear-gemm", "grouped-moe", "grouped-moe-contiguous", "act-fp8-quant",
    "swiglu-fp8-quant",
}
_NVFP4_FAMILIES = {
    "nvfp4-linear-gemm", "nvfp4-moe", "nvfp4-moe-contiguous",
    "swiglu-nvfp4-quant",
}
_ATTENTION_FAMILIES = {
    "mla-attention", "dsa-decode-attn", "dsa-prefill-attn", "sparse-mla-decode",
}
_EXACT_FAMILIES = {
    "embedding", "moe-gate", "dsa-decode-topk", "dsa-prefill-topk",
    "dsa-store-kv-index",
}


def tolerance_for(family: str) -> dict:
    """Return the conservative default appropriate for one operator family."""
    if family in _EXACT_FAMILIES:
        return dict(EXACT_TOL)
    if family == "sparse-mla-decode":
        return dict(SPARSE_MLA_FP8_TOL)
    if family in _FP8_FAMILIES:
        return dict(FP8_TOL)
    if family in _NVFP4_FAMILIES:
        return dict(NVFP4_TOL)
    if family in _ATTENTION_FAMILIES:
        return dict(ATTENTION_TOL)
    return dict(BF16_TOL)


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
    model: str                       # "kimi_k27" | "minimax_m3" | "glm52" — also the tasks/ subdir
    name: str                        # task dir name, e.g. "q_b_decode"
    op: str                          # human-readable op, e.g. "Q_b"
    family: str                      # e.g. "fp8-linear-gemm"
    hf_id: str
    recipe: str                      # reference.py source text
    axes: dict[str, dict]
    inputs: dict[str, dict]
    outputs: dict[str, dict]
    sweep: list[int]                 # legacy M-only sweep (used when workloads is None)
    description: str
    goal: str
    phase: str = "decode"
    backend: str = ""
    tolerance: Optional[dict] = None
    meta: dict[str, Any] = field(default_factory=dict)   # extra task.json fields
    baseline_us: Optional[float] = None                  # fp8 legacy csv wall-clock
    # Optional diagnostic model (read-only feedback for agents; never gates WIN).
    flops_expr: Optional[str] = None                     # e.g. "2*M*K*N"
    performance_model: Optional[dict[str, Any]] = None   # family-specific metric schema
    workload_metrics: Optional[list[str]] = None         # named metric keys to emit
    # Multi-axis workloads (e.g. [{M, ctx}, ...]). When set, replaces the M-only sweep
    # expansion in workload.jsonl; `sweep` remains the M values for task.json/profile.
    workloads: Optional[list[dict[str, int]]] = None

    def __post_init__(self):
        if self.tolerance is None:
            self.tolerance = tolerance_for(self.family)


# --- the ONE writer every family goes through ---------------------------------
def write_task(spec: TaskSpec, tasks_root: Path) -> str:
    """Materialize a TaskSpec into its five files under tasks_root/<model>/<name>/."""
    d = tasks_root / spec.model / spec.name
    d.mkdir(parents=True, exist_ok=True)

    definition = {
        "name": f"{spec.model}_{spec.name}",
        "hf_id": spec.hf_id,
        "family": spec.family,
        "description": spec.description,
        "axes": spec.axes,
        "custom_inputs_entrypoint": "get_inputs",
        "inputs": spec.inputs,
        "outputs": spec.outputs,
        "reference": spec.recipe,
    }
    if spec.flops_expr is not None:
        definition["flops_expr"] = spec.flops_expr
    if spec.performance_model is not None:
        definition["performance_model"] = spec.performance_model
    if spec.workload_metrics is not None:
        definition["workload_metrics"] = spec.workload_metrics
    (d / "definition.json").write_text(json.dumps(definition, indent=4))
    (d / "reference.py").write_text(spec.recipe)
    (d / "solution.py").write_text(spec.recipe)

    inp = {k: {"type": "custom"} for k in spec.inputs}
    if spec.workloads is not None:
        axis_rows = list(spec.workloads)
    else:
        axis_rows = [{"M": m} for m in spec.sweep]
    lines = []
    for axes in axis_rows:
        # uuid encodes every swept axis so multi-dim rows stay unique
        tag = "-".join(f"{k}{axes[k]}" for k in sorted(axes))
        lines.append(json.dumps({
            "uuid": f"{spec.model}-{spec.name}-{tag}",
            "axes": axes,
            "inputs": inp,
            "tolerance": spec.tolerance,
        }))
    (d / "workload.jsonl").write_text("\n".join(lines) + "\n")

    task_json = {
        "name": spec.name, "model": spec.model, "op": spec.op,
        "phase": spec.phase, "family": spec.family,
        "goal": spec.goal,
        **spec.meta,
        "sweep": spec.sweep,
        "backend": spec.backend,
        "tolerance": spec.tolerance,
        "baseline": {
            "source": "live",
            "measure": "reference.py timed by the harness CUPTI runtime (median device-kernel us)",
            "denominator": "the live per-shape reference latency (see .baseline_cache.json)",
        },
    }
    if spec.workloads is not None:
        task_json["workloads"] = spec.workloads
    if spec.baseline_us is not None:
        task_json["csv_wallclock_us_reference"] = spec.baseline_us
    if spec.performance_model is not None:
        task_json["performance_model"] = spec.performance_model
    if spec.workload_metrics is not None:
        task_json["workload_metrics"] = spec.workload_metrics
    (d / "task.json").write_text(json.dumps(task_json, indent=2))
    return spec.name
