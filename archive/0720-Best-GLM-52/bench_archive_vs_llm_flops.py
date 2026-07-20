#!/usr/bin/env python3
"""Benchmark archive best candidates vs harness reference (llm_flops-aligned).

Uses Kernel-Harness timing: CUPTI cold-L2 device-kernel median
(same primitive as evaluate_task / accept_layer).

Also loads llm_flops CSV (CUDA Graph) for side-by-side context — those
numbers are NOT directly comparable to harness_us; they are labeled separately.

Outputs:
  vs_llm_flops/DECODE_REPORT.md
  vs_llm_flops/PREFILL_REPORT.md
  vs_llm_flops/decode_results.json
  vs_llm_flops/prefill_results.json
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from functools import partial
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_ARCHIVE = Path(__file__).resolve().parent
_BEST = _ARCHIVE / "best"
_OUT = _ARCHIVE / "vs_llm_flops"
_HARNESS = _REPO / "testbench" / "harness"
_TASKS = _REPO / "testbench" / "tasks" / "glm52"
_LLM_FLOPS = Path("/home/qinhaiyan/llm_flops/runs/20260715-ue8m0")

sys.path[:] = [p for p in sys.path if p and Path(p).resolve() != _HARNESS]


def _sibling(name: str):
    spec = importlib.util.spec_from_file_location(f"_tb_{name}", _HARNESS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"_tb_{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


import torch  # noqa: E402

ops = _sibling("glm52_ops")
candidate_loader = _sibling("candidate_loader")
tb_timing = _sibling("timing")

TIMING = ("cupti-cold-l2-device-kernel-median" if tb_timing._HAVE_CUPTI
          else "event-cold-l2-median-NO-CUPTI")
WARMUP = 3
ITERATIONS = 10

# harness op -> llm_flops CSV name
LLM_FLOPS_NAMES = {
    "fused_qkv_a": "fused_qkv_a_proj",
    "q_b": "q_b_proj",
    "absorbed_W_UK": "absorbed_W_UK",
    "absorbed_W_UV": "absorbed_W_UV",
    "o_proj": "o_proj",
    "dsa_attn": {"decode": "dsa_decode_attn", "prefill": "dsa_prefill_attn"},
    "index_k": "index_k_proj",
    "index_q_upproj": "index_q_upproj",
    "index_score": "index_score",
    "moe_gate": "moe_gate_proj",
    "moe_up": "moe_up_proj",
    "moe_down": "moe_down_proj",
}

# archive dir -> harness task name
ARCHIVE_TO_TASK = {
    # decode
    "q_b_decode": "q_b_decode",
    "o_proj_decode_hbm35": "o_proj_decode",
    "o_proj_decode": "o_proj_decode",
    "o_proj_decode_hbm40_extreme": "o_proj_decode",
    "index_q_upproj_decode_hbm15": "index_q_upproj_decode",
    "moe_gate_proj_decode_hbm40": "moe_gate_proj_decode",
    "moe_up_proj_decode_hbm40": "moe_up_proj_decode",
    "moe_down_proj_decode_hbm40": "moe_down_proj_decode",
    "index_score_decode_hbm82": "index_score_decode",
    "dsa_attn_decode_hbm40": "dsa_attn_decode",
    "absorbed_W_UV_decode_hbm86": "absorbed_W_UV_decode",
    # prefill
    "index_k_prefill_bw70": "index_k_prefill",
    "moe_down_proj_prefill_mfu65": "moe_down_proj_prefill",
    "o_proj_prefill": "o_proj_prefill",
    "moe_gate_proj_prefill_mfu": "moe_gate_proj_prefill",
}


def clone_inputs(d: dict) -> dict:
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in d.items()}


def time_fn(fn, inputs, device) -> float:
    setup = lambda: clone_inputs(inputs)  # noqa: E731
    return tb_timing.time_runnable(fn, setup=setup, warmup=WARMUP,
                                   rep=ITERATIONS, device=device)


def load_llm_flops_csv(path: Path) -> dict[tuple[str, int, int], float]:
    """(llm_flops_name, M, S) -> avg_ms"""
    out: dict[tuple[str, int, int], float] = {}
    if not path.is_file():
        return out
    with path.open() as f:
        for row in csv.DictReader(f):
            m = int(row["batch"] if "batch" in row else row["M"])
            s = int(row["S"])
            out[(row["name"], m, s)] = float(row["avg_ms"])
    return out


def llm_flops_lookup(table: dict, op: str, phase: str, M: int, S: int) -> float | None:
    name = LLM_FLOPS_NAMES[op]
    if isinstance(name, dict):
        name = name[phase]
    return table.get((name, M, S))


def task_index() -> dict[tuple[str, str], Path]:
    out: dict[tuple[str, str], Path] = {}
    for d in sorted(_TASKS.iterdir()):
        meta_path = d / "task.json"
        if d.is_dir() and meta_path.is_file():
            meta = json.loads(meta_path.read_text())
            out[(meta["operator"], meta["phase"])] = d
    return out


def bench_op(op: str, phase: str, M: int, device, index: dict,
             candidate_dir: Path | None) -> dict:
    meta = ops.spec(op, phase)
    S = int(meta["S"])
    seed = int(meta["seed"])
    task_dir = index[(op, phase)]
    inputs = ops.build_inputs(op, phase, M, S, device, seed)
    ref_ms = time_fn(partial(ops.reference, op, phase), inputs, device)

    cand_ms = None
    speedup = None
    source = "reference"
    err = None
    if candidate_dir is not None:
        try:
            inputs = ops.build_inputs(op, phase, M, S, device, seed)
            run_fn, source, _ = candidate_loader.resolve(
                task_dir, op, phase, override=str(candidate_dir))
            if source != "reference":
                cand_ms = time_fn(run_fn, inputs, device)
                speedup = ref_ms / cand_ms
        except Exception as e:
            err = f"{type(e).__name__}: {e}"[:300]

    llm_key = None
    return {
        "op": op,
        "M": M,
        "S": S,
        "harness_ref_us": round(ref_ms * 1000, 3),
        "harness_cand_us": round(cand_ms * 1000, 3) if cand_ms else None,
        "speedup_vs_harness_ref": round(speedup, 4) if speedup else None,
        "candidate_source": source,
        "error": err,
        "llm_flops_key": LLM_FLOPS_NAMES[op],
    }


def run_phase(phase: str, shapes: list[int], llm_csv: Path,
              archives: list[tuple[str, Path]], device) -> dict:
    index = task_index()
    llm_table = load_llm_flops_csv(llm_csv)

    # Reference layer: all 12 ops
    ref_layer: dict[int, list[dict]] = {M: [] for M in shapes}
    for M in shapes:
        for op in ops.ALL_OPS:
            row = bench_op(op, phase, M, device, index, None)
            lf = llm_flops_lookup(llm_table, op, phase, M, row["S"])
            row["llm_flops_us"] = round(lf * 1000, 3) if lf is not None else None
            row["harness_vs_llm_flops_ratio"] = (
                round(row["harness_ref_us"] / row["llm_flops_us"], 4)
                if row["llm_flops_us"] else None)
            ref_layer[M].append(row)

    # Archive candidates
    candidates: list[dict] = []
    for arch_name, cand_dir in archives:
        task_name = ARCHIVE_TO_TASK[arch_name]
        task_dir = _TASKS / task_name
        meta = json.loads((task_dir / "task.json").read_text())
        op = meta["operator"]
        kind_path = _BEST / arch_name / "SOURCE.md"
        kind = "unknown"
        manifest = _ARCHIVE / "manifest.json"
        if manifest.is_file():
            for item in json.loads(manifest.read_text()):
                if item["op"] == arch_name:
                    kind = item["kind"]
                    break

        per_shape = []
        for M in shapes:
            row = bench_op(op, phase, M, device, index, cand_dir)
            lf = llm_flops_lookup(llm_table, op, phase, M, row["S"])
            row["llm_flops_us"] = round(lf * 1000, 3) if lf is not None else None
            if row["harness_cand_us"] and row["llm_flops_us"]:
                row["cand_vs_llm_flops_ratio"] = round(
                    row["harness_cand_us"] / row["llm_flops_us"], 4)
            else:
                row["cand_vs_llm_flops_ratio"] = None
            per_shape.append(row)

        candidates.append({
            "archive": arch_name,
            "task": task_name,
            "operator": op,
            "kind": kind,
            "candidate_dir": str(cand_dir),
            "per_shape": per_shape,
        })

    layer_totals = {}
    for M in shapes:
        rows = ref_layer[M]
        ref_sum = sum(r["harness_ref_us"] for r in rows)
        llm_sum = sum(r["llm_flops_us"] for r in rows if r["llm_flops_us"])
        layer_totals[str(M)] = {
            "harness_ref_layer_us": round(ref_sum, 3),
            "llm_flops_layer_us_12ops": round(llm_sum, 3) if llm_sum else None,
            "note": "llm_flops layer excludes index_weights_proj (13th op in CSV)",
        }

    return {
        "phase": phase,
        "timing": TIMING,
        "warmup": WARMUP,
        "iterations": ITERATIONS,
        "shapes": shapes,
        "llm_flops_csv": str(llm_csv),
        "reference_layer": ref_layer,
        "layer_totals": layer_totals,
        "archive_candidates": candidates,
    }


def fmt_speedup(v: float | None) -> str:
    if v is None:
        return "—"
    if v >= 1.001:
        return f"**{v:.2f}×**"
    if v <= 0.999:
        return f"{v:.2f}× (慢)"
    return f"{v:.3f}×"


def write_report(path: Path, data: dict, title: str) -> None:
    lines = [
        f"# {title}",
        "",
        f"Timing: **{data['timing']}** (warmup={data['warmup']}, iterations={data['iterations']})",
        f"llm_flops reference CSV: `{data['llm_flops_csv']}` (CUDA Graph — **不同协议，仅供参考**)",
        "",
        "## Harness reference layer（12 ops，无替换）",
        "",
        "| M | harness ref layer (µs) | llm_flops 12-op sum (µs) | harness/llm_flops |",
        "|---:|---:|---:|---:|",
    ]
    for M in data["shapes"]:
        lt = data["layer_totals"][str(M)]
        llm = lt["llm_flops_layer_us_12ops"]
        ratio = (lt["harness_ref_layer_us"] / llm) if llm else None
        lines.append(
            f"| {M} | {lt['harness_ref_layer_us']:.1f} | "
            f"{llm or '—'} | {f'{ratio:.2f}×' if ratio else '—'} |")

    lines += [
        "",
        "## Per-op harness reference vs llm_flops",
        "",
    ]
    for M in data["shapes"]:
        lines.append(f"### M={M}")
        lines.append("")
        lines.append("| op | harness ref (µs) | llm_flops (µs) | harness/llm_flops |")
        lines.append("|---|---:|---:|---:|")
        for r in data["reference_layer"][M]:
            ratio = r.get("harness_vs_llm_flops_ratio")
            lines.append(
                f"| {r['op']} | {r['harness_ref_us']:.1f} | "
                f"{r['llm_flops_us'] or '—'} | "
                f"{f'{ratio:.2f}×' if ratio else '—'} |")
        lines.append("")

    lines += [
        "## Archive best candidates vs harness reference",
        "",
        "speedup = harness_ref / harness_cand（>1 为加速）",
        "",
    ]
    for cand in data["archive_candidates"]:
        lines.append(f"### `{cand['archive']}` ({cand['kind']})")
        lines.append("")
        lines.append(f"- task: `{cand['task']}` · op: `{cand['operator']}`")
        lines.append("")
        lines.append("| M | harness ref (µs) | candidate (µs) | speedup | llm_flops (µs) | cand/llm_flops |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for r in cand["per_shape"]:
            lines.append(
                f"| {r['M']} | {r['harness_ref_us']:.1f} | "
                f"{r['harness_cand_us'] or '—'} | "
                f"{fmt_speedup(r['speedup_vs_harness_ref'])} | "
                f"{r['llm_flops_us'] or '—'} | "
                f"{r.get('cand_vs_llm_flops_ratio') or '—'} |")
        if cand["per_shape"] and cand["per_shape"][0].get("error"):
            lines.append(f"\nError: {cand['per_shape'][0]['error']}")
        lines.append("")

    lines += [
        "## 说明",
        "",
        "- **可比**：archive candidate vs harness reference（同协议、同形状、同算子契约）",
        "- **不可直接比绝对值**：harness CUPTI vs llm_flops CUDA Graph；`harness/llm_flops` 列仅作量级对照",
        "- llm_flops CSV 含 `index_weights_proj`（第 13 算子），本 harness layer 为 12 op",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> int:
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        return 1
    torch.cuda.set_device(device)
    _OUT.mkdir(parents=True, exist_ok=True)

    decode_archives = [
        (name, _BEST / name / "candidate")
        for name in ARCHIVE_TO_TASK
        if name.endswith("_decode") or "decode_" in name
    ]
    prefill_archives = [
        (name, _BEST / name / "candidate")
        for name in ARCHIVE_TO_TASK
        if name.endswith("_prefill") or "prefill_" in name
    ]

    print("=== DECODE ===", flush=True)
    decode = run_phase(
        "decode", [16, 32],
        _LLM_FLOPS / "glm5_decode_perf.csv",
        [(n, p) for n, p in decode_archives if p.is_dir()],
        device,
    )
    (_OUT / "decode_results.json").write_text(json.dumps(decode, indent=2))
    write_report(_OUT / "DECODE_REPORT.md", decode, "Archive vs baseline — DECODE")

    print("=== PREFILL ===", flush=True)
    prefill = run_phase(
        "prefill", [1024, 2048, 4096],
        _LLM_FLOPS / "glm5_unified_perf.csv",
        [(n, p) for n, p in prefill_archives if p.is_dir()],
        device,
    )
    (_OUT / "prefill_results.json").write_text(json.dumps(prefill, indent=2))
    write_report(_OUT / "PREFILL_REPORT.md", prefill, "Archive vs baseline — PREFILL")

    print(f"Wrote {_OUT}/DECODE_REPORT.md and PREFILL_REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
