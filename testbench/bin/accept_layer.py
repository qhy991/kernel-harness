#!/usr/bin/env python3
"""Acceptance: llm_flops-style layer swap — NOT the primary gate.

Port of PR1 opbench/allLatency.py onto the current glm52_ops + candidate.py
contract. After `run.sh` / evaluate_task reports a per-op result, this script
answers a different question:

  if this candidate is swapped into the 12-op layer budget (the same operator
  set llm_flops / bench_glm5_{prefill,decode} cover), what happens to the
  layer total?

It does not gate a WIN, does not replace run.sh, and does not prove an
in-place sglang drop-in. It is one acceptance step: end-to-end layer delta
under a controlled swap.

Usage (from repo root):

  .venv/bin/python testbench/bin/accept_layer.py --M 32
  .venv/bin/python testbench/bin/accept_layer.py --M 32 --task o_proj_decode
  .venv/bin/python testbench/bin/accept_layer.py --M 4096 --op o_proj \\
      --candidate ~/kernels/o_proj.py
  .venv/bin/python testbench/bin/accept_layer.py --M 32 \\
      --swap o_proj=~/kernels/o_proj.py \\
      --swap moe_down=~/kernels/moe_down.py
  .venv/bin/python testbench/bin/accept_layer.py --M 32 --task o_proj_decode --json

--task / --op : only that operator uses its candidate; the other 11 stay on
                the reference backend (isolates one candidate's layer impact).
--swap OP=PATH : repeatable; swaps several explicit external candidates while
                 every unlisted operator stays on the reference backend.
                 Omit all selectors to activate each task-local candidate.

Timing matches evaluate_task's authoritative path: CUPTI cold-L2 device-kernel
median (falls back to CUDA-event cold-L2 if cupti is missing). Defaults are
lighter than the gate (warmup=3, iterations=10) because this is advisory.

Exit: 0 measured · 2 infrastructure / argument error.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from functools import partial
from pathlib import Path

_BIN = Path(__file__).resolve().parent
_HARNESS = _BIN.parent / "harness"
_TASKS = _BIN.parent / "tasks" / "glm52"
_REPO = _BIN.parents[1]

# Same shadowing guard as evaluate_task.py: keep harness/ off sys.path[0].
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

TIMING_PROTOCOL = ("cupti-cold-l2-device-kernel-median" if tb_timing._HAVE_CUPTI
                   else "event-cold-l2-median-NO-CUPTI")

CAT = {"gemm": "GEMM", "bmm": "BMM", "moe": "MoE", "mla": "MLA", "score": "Score"}


def clone_inputs(d: dict) -> dict:
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in d.items()}


def _index_tasks() -> dict[tuple[str, str], Path]:
    """(operator, phase) -> task_dir, from each task.json (no hardcoded names)."""
    out: dict[tuple[str, str], Path] = {}
    for d in sorted(_TASKS.iterdir()):
        meta_path = d / "task.json"
        if not (d.is_dir() and meta_path.is_file()):
            continue
        meta = json.loads(meta_path.read_text())
        out[(meta["operator"], meta["phase"])] = d
    return out


def _resolve_focus(args, phase: str, index: dict[tuple[str, str], Path]):
    """Return the single operator to swap, or None for 'all candidates'."""
    if args.task is not None and args.op is not None:
        raise SystemExit("pass only one of --task / --op")
    if args.task is not None:
        task = Path(args.task)
        if not task.is_absolute() and not task.exists():
            # bare name: o_proj_decode
            cand = _TASKS / task.name
            if cand.is_dir():
                task = cand
        task = task.resolve()
        meta = json.loads((task / "task.json").read_text())
        if meta["phase"] != phase:
            raise SystemExit(
                f"--task {task.name} is phase={meta['phase']}, but --M={args.M} "
                f"implies phase={phase}")
        return meta["operator"], task
    if args.op is not None:
        if args.op not in ops.ALL_OPS:
            raise SystemExit(f"unknown --op {args.op!r}; known: {', '.join(ops.ALL_OPS)}")
        task = index.get((args.op, phase))
        if task is None:
            raise SystemExit(f"no task directory for {args.op}/{phase}")
        return args.op, task
    return None, None


def _time(fn, inputs, warmup: int, iterations: int, device) -> float:
    setup = lambda: clone_inputs(inputs)  # noqa: E731 — cost is not timed
    return tb_timing.time_runnable(fn, setup=setup, warmup=warmup,
                                   rep=iterations, device=device)


def _snapshot_deep_gemm_state() -> dict[str, object]:
    """Capture caller-visible process-global knobs so candidates stay isolated."""
    import deep_gemm

    state = {}
    for key in ("pdl", "num_sms", "tc_util"):
        getter = getattr(deep_gemm, f"get_{key}", None)
        if getter is not None:
            try:
                state[key] = getter()
            except Exception:
                pass
    return state


def _restore_deep_gemm_state(state: dict[str, object]) -> None:
    import deep_gemm

    for key, value in state.items():
        setter = getattr(deep_gemm, f"set_{key}", None)
        if setter is not None:
            try:
                setter(value)
            except Exception:
                pass


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Acceptance: swap candidates into the 12-op layer budget "
                    "(advisory; not the primary gate).")
    ap.add_argument("--M", type=int, required=True,
                    help="shape; phase inferred (prefill if M>=1024 else decode)")
    ap.add_argument("--S", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--task", default=None,
                    help="task dir or name; ONLY this op uses its candidate")
    ap.add_argument("--op", choices=ops.ALL_OPS, default=None,
                    help="ONLY this op uses its candidate (rest stay on backend)")
    ap.add_argument("--candidate", default=None,
                    help="override candidate for the focused op "
                         "(requires --task or --op)")
    ap.add_argument("--swap", action="append", default=[], metavar="OP=PATH",
                    help="repeatable explicit multi-swap; unlisted ops use reference")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iterations", type=int, default=10,
                    help="cold-L2 reps per op (lighter than the gate; advisory)")
    ap.add_argument("--json", action="store_true",
                    help="emit one RESULT_JSON block instead of the table")
    args = ap.parse_args()

    if args.candidate is not None and args.task is None and args.op is None:
        raise SystemExit("--candidate requires --task or --op")
    if args.swap and (args.task is not None or args.op is not None
                      or args.candidate is not None):
        raise SystemExit("--swap cannot be combined with --task/--op/--candidate")

    swaps: dict[str, str] = {}
    for item in args.swap:
        op_name, sep, path = item.partition("=")
        if not sep or not op_name or not path:
            raise SystemExit(f"invalid --swap {item!r}; expected OP=PATH")
        if op_name not in ops.ALL_OPS:
            raise SystemExit(
                f"unknown --swap op {op_name!r}; known: {', '.join(ops.ALL_OPS)}")
        if op_name in swaps:
            raise SystemExit(f"duplicate --swap for {op_name}")
        swaps[op_name] = str(Path(path).expanduser())

    phase = ops.infer_phase(args.M)
    index = _index_tasks()
    focus_op, focus_task = _resolve_focus(args, phase, index)

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    # Measure the complete reference layer before importing any candidate.
    # Candidate modules may compile extensions or touch process-global backend
    # knobs at import time; interleaving reference/candidate measurements would
    # contaminate later reference baselines in a multi-swap run.
    rows = []
    for op in ops.ALL_OPS:
        meta = ops.spec(op, phase)
        S = int(args.S if args.S is not None else meta["S"])
        seed = int(args.seed if args.seed is not None else meta["seed"])
        task_dir = index.get((op, phase))
        if task_dir is None:
            rows.append(dict(op=op, cat=CAT[meta["family"]], error="no task dir",
                             back=None, cand=None, used="backend", speedup=None,
                             source="missing"))
            continue

        try:
            inputs = ops.build_inputs(op, phase, args.M, S, device, seed)
            back_ms = _time(partial(ops.reference, op, phase), inputs,
                            args.warmup, args.iterations, device)
        except Exception as e:
            rows.append(dict(op=op, cat=CAT[meta["family"]],
                             error=f"{type(e).__name__}: {e}"[:200],
                             back=None, cand=None, used="backend", speedup=None,
                             source="reference"))
            continue

        rows.append(dict(op=op, cat=CAT[meta["family"]], error=None,
                         back=back_ms, cand=None, used="backend", speedup=None,
                         source="reference", used_ms=back_ms))

    # Candidate pass: rebuild one op's inputs at a time so prefill does not keep
    # all 12 large input sets resident simultaneously.
    for row in rows:
        if row.get("back") is None:
            continue
        op = row["op"]
        task_dir = index[(op, phase)]
        override = None
        if swaps:
            use_cand = op in swaps
            if use_cand:
                override = swaps[op]
        elif focus_op is not None:
            use_cand = (op == focus_op)
            if use_cand and args.candidate is not None:
                override = args.candidate
        else:
            use_cand = True  # every op may use its own candidate if present

        if not use_cand:
            continue

        source = "candidate"
        backend_state = _snapshot_deep_gemm_state()
        if use_cand:
            try:
                meta = ops.spec(op, phase)
                S = int(args.S if args.S is not None else meta["S"])
                seed = int(args.seed if args.seed is not None else meta["seed"])
                inputs = ops.build_inputs(op, phase, args.M, S, device, seed)
                run_fn, source, _ = candidate_loader.resolve(
                    focus_task if (focus_op == op and focus_task) else task_dir,
                    op, phase, override=override)
                if source != "reference":
                    cand_ms = _time(run_fn, inputs, args.warmup, args.iterations,
                                    device)
                    row.update(cand=cand_ms, used="cand",
                               speedup=row["back"] / cand_ms,
                               source=source, used_ms=cand_ms)
            except Exception as e:
                row.update(error=f"candidate: {type(e).__name__}: {e}"[:200],
                           source=source)
            finally:
                _restore_deep_gemm_state(backend_state)

    # Sort by backend latency descending (llm_flops / allLatency style).
    timed = [r for r in rows if r.get("back") is not None]
    timed.sort(key=lambda r: r["back"], reverse=True)
    failed = [r for r in rows if r.get("back") is None]

    total_back = sum(r["back"] for r in timed)
    total_used = sum(r.get("used_ms", r["back"]) for r in timed)
    n_cand = sum(1 for r in timed if r["used"] == "cand")
    layer_speedup = (total_back / total_used) if total_used > 0 else None
    layer_delta_pct = (((total_back - total_used) / total_back * 100)
                       if total_back > 0 else None)

    result = {
        "kind": "accept_layer",
        "role": "acceptance",  # not the primary gate
        "M": args.M,
        "phase": phase,
        "timing": TIMING_PROTOCOL,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "focus_op": focus_op,
        "focus_task": focus_task.name if focus_task else None,
        "candidate_override": args.candidate,
        "candidate_swaps": swaps,
        "n_candidates_active": n_cand,
        "layer_total_backend_ms": round(total_back, 6) if timed else None,
        "layer_total_swapped_ms": round(total_used, 6) if timed else None,
        "layer_speedup": round(layer_speedup, 6) if layer_speedup else None,
        "layer_delta_pct": round(layer_delta_pct, 4) if layer_delta_pct is not None else None,
        "ops": [
            {k: (round(v, 6) if isinstance(v, float) else v)
             for k, v in r.items() if k != "used_ms"}
            for r in timed + failed
        ],
        "note": ("12 opbench/llm_flops ops; index_weights_proj (bf16) not modeled. "
                 "Primary gate remains task run.sh / evaluate_task.py."),
    }

    if args.json:
        print("RESULT_JSON:" + json.dumps(result, ensure_ascii=False))
        return 0

    print("=" * 96)
    print(f"ACCEPTANCE (layer swap) — not the primary gate")
    print(f"GLM-5.2 {phase.upper()}  M={args.M}  timing={TIMING_PROTOCOL}  "
          f"warmup={args.warmup} iters={args.iterations}")
    if swaps:
        print("focus: explicit multi-swap — "
              + ", ".join(f"{op}={path}" for op, path in swaps.items()))
    elif focus_op:
        print(f"focus: ONLY {focus_op} uses candidate"
              + (f" ({args.candidate})" if args.candidate else
                 f" ({focus_task.name})" if focus_task else ""))
    else:
        print("focus: every op with a non-reference candidate")
    print("=" * 96)
    print(f"  {'op':<18s} {'cat':<6s} {'backend(ms)':>12s} {'candidate(ms)':>14s} "
          f"{'used':>9s} {'speedup':>9s}")
    print("  " + "-" * 92)
    for r in timed:
        cand_str = f"{r['cand']:.4f}" if r["cand"] is not None else "-"
        spd_str = f"{r['speedup']:.2f}x" if r["speedup"] else "-"
        print(f"  {r['op']:<18s} {r['cat']:<6s} {r['back']:>12.4f} {cand_str:>14s} "
              f"{r['used']:>9s} {spd_str:>9s}")
    for r in failed:
        print(f"  {r['op']:<18s} {r['cat']:<6s}   FAILED: {r.get('error')}")
    print("  " + "-" * 92)
    print(f"  layer TOTAL (all backend):        {total_back:>10.4f} ms")
    print(f"  layer TOTAL (candidates swapped): {total_used:>10.4f} ms   "
          f"({n_cand} candidate(s) active)")
    if layer_speedup is not None and layer_delta_pct is not None:
        print(f"  end-to-end layer speedup:         {layer_speedup:.4f}x  "
              f"({layer_delta_pct:+.2f}%)")
    print("=" * 96)
    print("note: advisory acceptance only. Verify with run.sh before trusting a "
          "per-op speedup; this script does not check correctness.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
