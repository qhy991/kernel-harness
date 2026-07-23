#!/usr/bin/env python
"""Multi-process evaluator for the comm / deepep task families.

Launched by `torchrun --nproc-per-node=$WS`, one process per GPU:

    torchrun --standalone --nproc-per-node=8 \\
        testbench/harness/evaluate_comm_task.py <task_dir> [--M N] [--repeat R]

Each rank:
  1. joins the process group (env:// backend)
  2. sets torch.cuda.set_device(local_rank)
  3. reads task.json → (op, phase); glm52_ops.spec / build_inputs give the frozen
     per-rank input dict (rank/world_size are read from the process group)
  4. calls glm52_ops.reference(op, phase, inputs) — the torch.distributed
     ground-truth collective — and clones the output
  5. calls candidate.run(inputs) — the candidate kernel, which is expected to
     issue its own collective and return the same result
  6. compares candidate vs reference locally via torch.allclose; a rank-0
     dist.reduce collects the "all ranks correct" verdict
  7. times both with HIP/CUDA events wrapped in dist.barrier
  8. rank 0 prints the row + writes runs/comm/<task>/<run_id>/result.json

Exit codes match evaluate_task.py:
  0 = correct + candidate faster than reference on >= 1 shape, none regressing
  1 = correct + performance gate not met
  2 = correctness failed (on any rank, on any shape)
  3 = infrastructure / contract error
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
HARNESS_DIR = REPO / "testbench" / "harness"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _sibling(name: str):
    spec = importlib.util.spec_from_file_location(f"_tb_{name}", HARNESS_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"_tb_{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


ops = _sibling("glm52_ops")


import torch  # noqa: E402


def _init_dist() -> tuple[int, int, int]:
    """Initialise the process group from torchrun env vars. Returns
    (rank, local_rank, world_size). Uses nccl backend (works on both
    CUDA/NCCL and ROCm/RCCL)."""
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if not dist.is_initialized():
        # nccl for both CUDA (NCCL) and ROCm (RCCL). CPU fallback: gloo.
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def _load_candidate(task_dir: Path, override: str | None):
    """Import candidate.py (or a --candidate override) and return its
    `run(inputs)` function."""
    if override:
        p = Path(override)
        if p.is_dir():
            p = p / "candidate.py"
    else:
        p = task_dir / "candidate.py"
    if not p.is_file():
        raise FileNotFoundError(f"candidate.py not found at {p}")
    spec = importlib.util.spec_from_file_location(f"_candidate_{p.stem}", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "run") or not callable(mod.run):
        raise AttributeError(f"{p}: expected a callable run(inputs) -> output")
    return mod.run, str(p)


def _load_workloads(task_dir: Path, only_M: int | None) -> list[dict]:
    """Read workload.jsonl. If only_M given, filter to that shape."""
    wl_path = task_dir / "workload.jsonl"
    if not wl_path.is_file():
        raise FileNotFoundError(wl_path)
    rows = []
    for line in wl_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        w = json.loads(line)
        if only_M is not None and int(w["axes"]["M"]) != only_M:
            continue
        rows.append(w)
    return rows


def _all_close(ref: torch.Tensor, cand: torch.Tensor, atol: float, rtol: float) -> tuple[bool, float, float]:
    """Return (pass, calc_diff, max_abs_err) for a rank-local comparison."""
    if ref.shape != cand.shape:
        return False, float("inf"), float("inf")
    r = ref.float()
    c = cand.float()
    denom = (r * r + c * c).sum().double()
    calc_diff = (1 - 2 * (r * c).sum().double() / denom).item() if denom > 0 else 0.0
    max_abs = (c - r).abs().max().item()
    ok = bool(torch.allclose(c, r, atol=atol, rtol=rtol))
    return ok, calc_diff, max_abs


def _time_one(fn, warmup: int, iters: int, device) -> float:
    """HIP/CUDA-event median time over `iters` iterations, with dist.barrier
    around the timing window to avoid launch-skew across ranks."""
    import torch.distributed as dist

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    dist.barrier()

    per_iter = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        per_iter.append(start.elapsed_time(end) * 1e-3)   # → seconds
    return statistics.median(per_iter)


def _pct(xs, q: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    n = len(xs)
    if n == 1:
        return xs[0]
    idx = q * (n - 1)
    lo, hi = int(idx), min(int(idx) + 1, n - 1)
    frac = idx - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def evaluate(task_dir: Path, args, rank: int, local_rank: int, world_size: int) -> tuple[dict, int]:
    import torch.distributed as dist

    meta = json.loads((task_dir / "task.json").read_text())
    op = meta["operator"]
    phase = meta["phase"]
    spec = ops.spec(op, phase)
    fam = spec["family"]
    if fam not in ("comm", "deepep"):
        raise SystemExit(
            f"[evaluate_comm_task] task {op}/{phase} is family={fam}, not comm/deepep. "
            "Use evaluate_task.py for compute-family tasks.")

    S = int(meta.get("S", spec["S"]))
    seed = int(meta.get("seed", spec["seed"]))
    min_speedup_gate = float(meta.get("performance_gate", {}).get("min_speedup", 1.0))
    diff_tol = float(spec["diff_tol"])
    rel_tol = float(spec.get("rel_tol", 1e-2))
    atol = 5e-2                     # bf16 collective outputs are noisy at large H
    rtol = 5e-2

    cand_fn, cand_path = _load_candidate(task_dir, args.candidate)

    device = torch.device(f"cuda:{local_rank}")

    run_id = datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S")
    started_at = datetime.now(timezone.utc).isoformat()

    if rank == 0:
        print(f"COMM TASK  {op}/{phase}  world_size={world_size}  run={run_id}")
        print(f"     candidate = {cand_path}")
        print(f"     spec: family={fam}  H={spec['H']}  diff_tol={diff_tol}")
        print(f"     timing: HIP/CUDA-event median, dist.barrier() bounded")
        hdr = (f"{'shape':>7} {'ok':>5} {'calc_diff':>10} {'max_abs':>9} "
               f"{'cand_us':>9} {'ref_us':>9} {'speedup':>8} "
               f"{'GB/s':>9} {'BW_util':>8} {'verdict':>8}")
        print(hdr)

    workloads = _load_workloads(task_dir, args.M)
    if not workloads:
        raise SystemExit(f"no workloads in {task_dir}/workload.jsonl (filter --M {args.M})")

    per_shape = []
    any_win = False
    any_regress = False
    all_correct = True

    for wl in workloads:
        M = int(wl["axes"]["M"])
        row = {"uuid": wl.get("uuid", f"{task_dir.name}-M{M}"), "axes": {"M": M, "S": S}}

        # ── build inputs on every rank ──
        try:
            inputs = ops.build_inputs(op, phase, M, S, device, seed)
        except Exception as exc:
            row.update(correct=False, error=f"build_inputs: {type(exc).__name__}: {exc}"[:300])
            all_correct = False
            per_shape.append(row)
            if rank == 0:
                print(f"M={M:>5} {'ERR':>5}  {row['error'][:70]}")
            continue

        # ── correctness on each rank, dist-reduced to all-ranks-ok ──
        try:
            ref_out = ops.reference(op, phase, inputs)
            ref_out_clone = ref_out.detach().clone()
            cand_out = cand_fn(inputs)
            local_ok, cd, max_abs = _all_close(ref_out_clone, cand_out, atol, rtol)
        except Exception as exc:
            local_ok, cd, max_abs = False, float("inf"), float("inf")
            row["error"] = f"correctness: {type(exc).__name__}: {exc}"[:300]

        # Reduce ok across ranks: all must pass.
        ok_tensor = torch.tensor(int(local_ok), device=device, dtype=torch.int32)
        dist.all_reduce(ok_tensor, op=dist.ReduceOp.MIN)
        ok = bool(ok_tensor.item())
        row.update(correct=ok, calc_diff=cd, max_abs_err=max_abs)
        if not ok:
            all_correct = False
            per_shape.append(row)
            if rank == 0:
                reason = row.get("error", f"allclose failed (calc_diff={cd:.2e})")
                print(f"M={M:>5} {'FAIL':>5} {cd:>10.2e} {max_abs:>9.2e}  {reason[:70]}")
            continue

        # ── timing: reference + candidate, iters × repeat, median-of-medians ──
        cand_lat, ref_lat = [], []
        for _ in range(args.repeat):
            cand_lat.append(_time_one(lambda: cand_fn(inputs),
                                      args.warmup, args.iterations, device))
            ref_lat.append(_time_one(lambda: ops.reference(op, phase, inputs),
                                     args.warmup, args.iterations, device))
        cand_med = statistics.median(cand_lat)
        ref_med = statistics.median(ref_lat)
        cand_p90 = _pct(cand_lat, 0.9)
        cand_p10 = _pct(cand_lat, 0.1)
        ref_p90 = _pct(ref_lat, 0.9)
        ref_p10 = _pct(ref_lat, 0.1)

        speedup_med = ref_med / cand_med if cand_med > 0 else 0.0
        speedup_cons = ref_p10 / cand_p90 if cand_p90 > 0 else 0.0
        regress_cons = ref_p90 / cand_p10 if cand_p10 > 0 else 0.0

        # ── reward using ops.cost + ops.reward (interconnect bw util) ──
        flops, bytes_moved, dtype = ops.cost(op, phase, M, S)
        r = ops.reward(latency_ms=cand_med * 1000.0, flops=flops,
                       bytes_hbm=bytes_moved, compute_dtype=dtype)

        if speedup_cons > min_speedup_gate:
            verdict = "win"
            any_win = True
        elif regress_cons < 1.0:
            verdict = "regress"
            any_regress = True
        else:
            verdict = "neutral"

        row.update(
            candidate_lat_s=cand_med, reference_lat_s=ref_med,
            speedup_median=speedup_med, speedup_conservative=speedup_cons,
            regress_conservative=regress_cons, verdict=verdict,
            gbps=r["gbps"], bw_util=r["bw_util"], reward=r["reward"],
            bound=r["bound"], compute_dtype=dtype,
            bytes_moved=bytes_moved,
        )
        per_shape.append(row)
        if rank == 0:
            print(f"M={M:>5} {'ok':>5} {cd:>10.2e} {max_abs:>9.2e} "
                  f"{cand_med*1e6:>9.1f} {ref_med*1e6:>9.1f} "
                  f"{speedup_med:>7.3f}x "
                  f"{r['gbps']:>9.1f} {r['bw_util']*100:>7.1f}% {verdict:>8}")

    # ── aggregate ──
    correct_count = sum(1 for r in per_shape if r.get("correct"))
    winning = sum(1 for r in per_shape if r.get("verdict") == "win")
    regressing = sum(1 for r in per_shape if r.get("verdict") == "regress")

    if not all_correct:
        exit_code = 2
    elif any_win and not any_regress:
        exit_code = 0
    else:
        exit_code = 1

    result = {
        "task": task_dir.name,
        "operator": op, "phase": phase, "family": fam,
        "world_size": world_size,
        "run_id": run_id, "started_at": started_at,
        "candidate": cand_path,
        "shapes_total": len(per_shape),
        "shapes_correct": correct_count,
        "shapes_won": winning,
        "shapes_regressed": regressing,
        "gate_passed": exit_code == 0,
        "min_speedup": min_speedup_gate,
        "per_shape": per_shape,
    }

    if rank == 0:
        print()
        print(f"SUMMARY  correct={correct_count}/{len(per_shape)}  "
              f"won={winning}  regressed={regressing}  exit={exit_code}")
        # persist under runs/comm/<task>/<run>/
        out_root = REPO / "runs" / "comm" / task_dir.name / run_id
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        # Also write a well-known "last exit code" file that run.sh reads:
        # torchrun treats any non-zero exit as a ChildFailedError and prints a
        # noisy banner. We report the semantic exit contract (0/1/2/3) through
        # this file, and all ranks return 0 to Python so torchrun stays quiet.
        latest = REPO / "runs" / "comm" / task_dir.name / "last_exit_code"
        latest.write_text(f"{exit_code}\n")
        print(f"     wrote {out_root.relative_to(REPO)}/result.json")

    return result, exit_code


def _describe(task_dir: Path, json_out: bool = False) -> int:
    """Print the problem statement without needing torch.distributed init.
    Uses a solo world_size so build_inputs shapes are still legible."""
    os.environ.setdefault("KERNEL_HARNESS_COMM_WORLD_SIZE", "8")
    meta = json.loads((task_dir / "task.json").read_text())
    op, phase = meta["operator"], meta["phase"]
    problem = ops.problem(op, phase, device=None) if hasattr(ops, "problem") else {
        "operator": op, "phase": phase, "spec": ops.spec(op, phase),
    }
    if json_out:
        print(json.dumps(problem, indent=2, default=str))
    else:
        s = ops.spec(op, phase)
        f, b, d = ops.cost(op, phase, s["sweep"][0], s["S"])
        print(f"COMM TASK  {op}/{phase} — {s['label']}")
        print(f"  family={s['family']}  H={s['H']}  dtype={s['dtype']}")
        print(f"  requires_multi_gpu={s['requires_multi_gpu']}  diff_tol={s['diff_tol']:.0e}")
        print(f"  sweep: M in {s['sweep']}")
        print(f"  cost @ M={s['sweep'][0]}: bytes_moved={b:.2e} dtype={d!r}")
        print()
        print("  Run with torchrun (multi-process). ./run.sh handles the launch.")
        print("  Contract: run(inputs) -> tensor.  inputs is the frozen per-rank dict.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("task_dir", type=Path)
    ap.add_argument("--describe", action="store_true")
    ap.add_argument("--json", action="store_true", help="pair with --describe")
    ap.add_argument("--M", type=int, default=None)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--candidate", type=str, default=None,
                    help="path to candidate .py or dir")
    args = ap.parse_args()

    if args.describe:
        return _describe(args.task_dir, json_out=args.json)

    if "RANK" not in os.environ:
        sys.stderr.write(
            "[evaluate_comm_task] no RANK env — this evaluator must be launched with "
            "torchrun. Use ./run.sh in the task directory, which does that for you.\n"
        )
        return 3

    rank, local_rank, world_size = _init_dist()
    try:
        _, code = evaluate(args.task_dir, args, rank, local_rank, world_size)
        # Every rank returns 0 to torchrun to keep the launcher quiet; the real
        # 0/1/2/3 exit contract is in <task>/last_exit_code (rank 0 writes it,
        # run.sh reads it). Elastic launchers otherwise print a giant
        # ChildFailedError banner for anything but 0.
        return 0
    except SystemExit as se:
        return int(getattr(se, "code", 3) or 0)
    except Exception:
        if rank == 0:
            traceback.print_exc()
        return 3
    finally:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
