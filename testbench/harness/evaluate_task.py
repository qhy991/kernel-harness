#!/usr/bin/env python
"""Unified per-task evaluation: one contract, one command, one source of truth.

Every operator definition — inputs, reference, thresholds, masks, cost model,
peaks — lives in glm52_ops. This file only orchestrates. A task directory
therefore declares nothing but WHICH problem it is, and cannot drift from what
actually runs. `--describe` renders the problem statement from that same module.

For every shape in the task's workload:

  1. Build ONE frozen input dict (glm52_ops.build_inputs) — the same tensors
     feed the reference and the candidate.
  2. Run the reference, clone its output, then POISON the shared output buffer.
  3. Run the candidate on those inputs; gate on anomaly positions, then
     elementwise (abs OR rel), then DeepGEMM's calc_diff.
  4. Only if correct, time candidate and reference on the same inputs and ABI.
  5. Re-check correctness on freshly built inputs after timing.
  6. Turn the candidate latency into a bound-aware roofline reward, and judge the
     shape as win / regress / neutral.

Why step 2 exists
-----------------
build_inputs pre-allocates a shared `out` buffer for the gemm and moe families,
and reference() writes its result into it. Cloning ref_out only stops the
candidate from clobbering ref_out; it leaves the correct
answer sitting in inputs["out"]. A candidate whose entire body is
`return inputs["out"]` therefore scores a perfect match — and, being a no-op,
would then time near zero and take the reward to its ceiling. Poisoning the
buffer between the two calls is what makes "correct" and "fast" describe the
same kernel: the no-op now arrives as all-NaN and the anomaly check names it.

Timing: CUPTI cold-L2 **device-kernel** median (testbench harness.timing), the
same primitive testbench/bin/evaluate.py gates on. Inputs are cloned per
iteration and L2 is flushed before each, both outside the measured window; the
number is the device-side kernel span, median over `--iterations` reps.

Device-kernel time, not wall clock, is the only thing that can back a roofline
reward. The reward is a hardware-utilisation ratio (achieved FLOP/s over a peak),
so pairing it with a host-inclusive wall time yields something that is not a
utilisation at all. That is not a hypothetical: on B200 this op's deep_gemm
Python wrapper costs ~65us of host enqueue per call while the kernel itself runs
~47us, so eager dispatch is the binding cost. Timing one call between CUDA events
with a sync per iteration reports ~99us for that 47us kernel — a 109% inflation
that is pure host stall. rewardbench's warm-L2 numbers dodge this with CUDA
graphs; a per-call event timer walks straight into it. CUPTI sidesteps both by
correlating launches to kernels and measuring only the device span.

The real cold-vs-warm penalty, once dispatch is excluded, is ~12% for this op
(53us cold vs 47us warm), not the ~2.4x a per-call event timer suggests.

`--repeat K` (default 10) takes K samples per shape and gates on the conservative
margin: the candidate's p90 against the reference's p10. Not the median, because
noise here is +-5% and at K=1 the margin collapses to median-vs-median, where a
candidate that *is* the reference scores 0.947x-1.022x and passes a >1.0 gate
roughly half the time — so --repeat 1 is a probe, never a verdict.

Not max/min either, though that is what evaluate.py does and what this did first.
Dividing two extremes lets ONE bad sample decide the verdict, and at K=10 that is
likely rather than rare: observed sp_cons 0.347x against a 0.999x median because
one sample of ten came back 2.9x high. Medians are stable to ~0.2% across runs, so
those are measurement artifacts, not kernel behaviour — CUPTI session churn, GPU
clock ramp and per-process allocator warmup were each tested and refuted as the
cause. At a quantile, more samples make the gate better instead of more fragile,
which is the only reason to raise K at all. The true min/max are still recorded and
still drive the instability warning. Default warmup is 3.

Unlike evaluate.py the samples are in-process, so they capture run-level but not
process-level noise; result.json records this as repeat_scope="in-process".

    ./run.sh
    python testbench/harness/evaluate_task.py <task_dir>
"""
from __future__ import annotations

import argparse
import io
import json
import math
import statistics
import sys
import traceback
from functools import partial
from pathlib import Path

import importlib.util  # noqa: E402

_HARNESS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _HARNESS_DIR.parents[1]

# Python puts a script's OWN directory on sys.path[0]. This directory holds
# profile.py, which then shadows the stdlib `profile` that cProfile imports as
# `import profile as _pyprofile` — so the moment a candidate imports sglang
# (-> torchvision -> torch._dynamo -> cProfile) the run dies inside OUR file
# with "No module named 'harness.inputs'". Real GLM-5.2 candidates import
# sglang, so this must go before any third-party import happens.
sys.path[:] = [p for p in sys.path
               if p and Path(p).resolve() != _HARNESS_DIR]


def _sibling(name: str):
    """Import a testbench/harness module by explicit path, keeping that directory
    OFF sys.path (see the shadowing note above)."""
    spec = importlib.util.spec_from_file_location(f"_tb_{name}", _HARNESS_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"_tb_{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


import torch  # noqa: E402

ops = _sibling("glm52_ops")          # the single source of truth for all 12 ops
candidate_loader = _sibling("candidate_loader")
result_store = _sibling("result_store")
RH = _sibling("reward_hack")
tb_timing = _sibling("timing")       # testbench CUPTI timer

TIMING_PROTOCOL = ("cupti-cold-l2-device-kernel-median" if tb_timing._HAVE_CUPTI
                   else "event-cold-l2-median-NO-CUPTI")


def clone_inputs(d: dict) -> dict:
    """Fresh tensor copies per timed iteration; non-tensors pass through."""
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in d.items()}


class _Tee(io.TextIOBase):
    """Mirror stdout into a buffer so stdout.log is the literal terminal output."""

    def __init__(self, stream):
        self._stream = stream
        self.buffer_text = io.StringIO()

    def write(self, s):
        self._stream.write(s)
        self.buffer_text.write(s)
        return len(s)

    def flush(self):
        self._stream.flush()


# ── correctness ──────────────────────────────────────────────────────────────
def _clone_out(ref_out):
    if torch.is_tensor(ref_out):
        return ref_out.clone()
    return tuple(t.clone() for t in ref_out)


def _check_outputs(cand_out) -> None:
    tensors = list(cand_out) if isinstance(cand_out, (tuple, list)) else [cand_out]
    RH.check_lazy_outputs(tensors)


def _correctness(op, phase, M, S, seed, device, cand_fn) -> dict:
    inputs = ops.build_inputs(op, phase, M, S, device, seed)
    ref_out = _clone_out(ops.reference(op, phase, inputs))
    poisoned = ops.poison(inputs)
    cand_out = cand_fn(inputs)
    _check_outputs(cand_out)
    r = ops.compare(ref_out, cand_out, op, phase, inputs)
    r["poisoned"] = poisoned
    r["inputs"] = inputs
    return r


# ── main ─────────────────────────────────────────────────────────────────────
def _load_workloads(task_dir: Path, only_M, max_workloads):
    if only_M is not None:
        return [{"uuid": f"{task_dir.name}-M{only_M}", "axes": {"M": only_M}}]
    rows = []
    for line in (task_dir / "workload.jsonl").read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows[:max_workloads] if max_workloads else rows


def _geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


# Quantile the conservative margin is taken at. 0.90 keeps the gate a statement about
# the tail ("the candidate's slow end still beats the reference's fast end") while
# staying out of reach of a single artifact sample. At --repeat 1 or 2 it degenerates
# to the min/max it replaces, so a probe behaves exactly as before.
CONS_Q = 0.90


def _pct(xs, q: float) -> float:
    """Linear-interpolated percentile (numpy's default method), stdlib only."""
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    i = q * (len(s) - 1)
    lo, hi = math.floor(i), math.ceil(i)
    return s[lo] if lo == hi else s[lo] + (s[hi] - s[lo]) * (i - lo)


class ContractError(RuntimeError):
    """task.json disagrees with glm52_ops. Exit 3, never a silent measurement."""


def _validate(task_dir: Path, meta: dict, op: str, phase: str) -> None:
    """The task directory declares almost nothing, so almost nothing can drift —
    but what little it does declare still has to be true."""
    if op not in ops.ALL_OPS:
        raise ContractError(f"task.json operator={op!r} is not a GLM-5.2 op. "
                            f"Known: {', '.join(ops.ALL_OPS)}")
    if phase not in ("prefill", "decode"):
        raise ContractError(f"task.json phase={phase!r} must be prefill or decode")
    declared = sorted(int(w["axes"]["M"]) for w in _load_workloads(task_dir, None, None))
    expected = sorted(ops.spec(op, phase)["sweep"])
    if declared != expected:
        raise ContractError(
            f"workload.jsonl sweep {declared} != glm52_ops sweep {expected} for "
            f"{op}/{phase}. The workload is not the task's to redefine; fix the "
            f"file or change DEFAULT_SWEEP in glm52_ops.py.")
    want_fam = ops.spec(op, phase)["family"]
    if meta.get("family") != want_fam:
        raise ContractError(
            f"task.json family={meta.get('family')!r} != glm52_ops {want_fam!r} for "
            f"{op}/{phase}. It is a generated mirror — re-run "
            f"testbench/bin/sync_glm52_tasks.py.")
    for stale in ("diff_tol", "rel_tol", "abs_tol_factor", "correctness",
                  "performance", "contract", "K", "N", "sweep"):
        if stale in meta:
            raise ContractError(
                f"task.json restates {stale!r}, which glm52_ops owns. A second "
                f"copy is a copy that can lie — delete it. (`--describe` prints "
                f"the real contract.)")


def evaluate(task_dir: Path, args) -> tuple[dict, int]:
    meta = json.loads((task_dir / "task.json").read_text())
    # task.json declares only WHICH problem this is and how hard the bar is.
    # Everything else — shapes, inputs, reference, thresholds, cost model, peaks —
    # comes from glm52_ops, so a task directory has nothing it could lie about.
    op = meta["operator"]
    phase = meta["phase"]
    _validate(task_dir, meta, op, phase)
    op_meta = ops.spec(op, phase)
    S = int(meta.get("S", op_meta["S"]))
    seed = int(meta.get("seed", op_meta["seed"]))
    min_speedup_gate = float(meta.get("performance_gate", {}).get("min_speedup", 1.0))

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    cand_fn, cand_label, cand_path = candidate_loader.resolve(
        task_dir, op, phase, override=args.candidate)
    RH.check_monkey_patch()
    cand_sha = result_store.sha256_file(cand_path) if cand_path else None
    # The run record must name the exact bytes that ran, not where we hoped they were.

    run_id = result_store.new_run_id()
    started = result_store.utc_now()

    print(f"TASK {op}/{phase}  run={run_id}  "
          f"candidate={cand_label}{f' sha={cand_sha[:12]}' if cand_sha else ''}")
    print(f"     timing={TIMING_PROTOCOL} iters={args.iterations} warmup={args.warmup} "
          f"repeat={args.repeat}  S={S} seed={seed}  device={args.device}")
    print()
    peak = ops.PEAK_FLOPS[op_meta["peak_dtype"]]
    ridge = peak / ops.HBM_BYTES_PER_S
    print(f"     roofline: {op_meta['peak_dtype']} peak {peak/1e15:.2f} PFLOP/s, HBM "
          f"{ops.HBM_BYTES_PER_S/1e12:.1f} TB/s, ridge {ridge:.1f} FLOP/byte "
          f"-> reward = utilisation of whichever resource binds")
    print()
    # reward IS the utilisation of the binding resource — bw_util when memory-bound,
    # compute_util when compute-bound. Printing both alongside makes that identity
    # visible rather than something the reader has to take on trust. The reference
    # sub-row is the ceiling: without it a low reward reads as candidate headroom
    # when it may simply be the op's roof.
    hdr = (f"{'shape':>7} {'ok':>5} {'calc_diff':>10} {'cand_us':>9} {'ref_us':>9} "
           f"{'speedup':>8} {'sp_cons':>8} {'verdict':>8} {'AI':>7} {'bound':>7} "
           f"{'TFLOP/s':>9} {'MFU':>7} {'GB/s':>9} {'BW':>7} {'reward':>8}")
    print(hdr)

    per_shape, sp_med_all, sp_cons_all, shape_verdicts = [], [], [], []
    all_correct = True
    workloads = _load_workloads(task_dir, args.M, args.max_workloads)

    for wl in workloads:
        M = int(wl["axes"]["M"])
        shape = f"M={M}"
        row = {"uuid": wl.get("uuid", f"{task_dir.name}-M{M}"), "axes": {"M": M, "S": S}}

        # ── correctness (gates everything below it) ──
        try:
            c = _correctness(op, phase, M, S, seed, device, cand_fn)
        except Exception as exc:
            all_correct = False
            row.update(correct=False, error=f"{type(exc).__name__}: {exc}"[:300])
            per_shape.append(row)
            print(f"{shape:>8} {'ERROR':>8}  {row['error'][:60]}")
            continue

        ok = c["pass"]
        row.update(correct=ok, output_kind=op_meta["output_kind"],
                   output_buffer_poisoned=c["poisoned"],
                   **{k: c[k] for k in ("calc_diff", "max_abs_err", "max_rel_err",
                                        "abs_tol", "rel_tol", "diff_tol",
                                        "elementwise_failed", "anomaly_ok", "elements",
                                        "cosine", "best_fit_scale")
                      if k in c})
        if not ok:
            all_correct = False
            row["error"] = c["reason"]
            per_shape.append(row)
            dstr = "-" if c.get("calc_diff") is None else f"{c['calc_diff']:.2e}"
            print(f"{shape:>7} {'FAIL':>5} {dstr:>10}  {c['reason']}")
            continue

        # ── performance (same inputs, same ABI) ──
        inputs = c["inputs"]
        ops.poison(inputs)
        ref_fn = partial(ops.reference, op, phase)
        setup = lambda: clone_inputs(inputs)  # noqa: E731 — cost is not timed
        cand_s, ref_s = [], []
        for _ in range(args.repeat):
            cand_s.append(tb_timing.time_runnable(cand_fn, setup=setup,
                                                  warmup=args.warmup,
                                                  rep=args.iterations, device=device))
            ref_s.append(tb_timing.time_runnable(ref_fn, setup=setup,
                                                 warmup=args.warmup,
                                                 rep=args.iterations, device=device))
        RH.check_monkey_patch()

        c_lo, c_med, c_hi = min(cand_s), statistics.median(cand_s), max(cand_s)
        b_lo, b_med, b_hi = min(ref_s), statistics.median(ref_s), max(ref_s)
        s_med = b_med / c_med
        # The conservative margin is the candidate's slow tail against the reference's
        # fast tail, at CONS_Q. It used to be max/min, which mirrored evaluate.py — but
        # max/min divides two extremes, so ONE bad sample decides the verdict, and at
        # --repeat 10 that is likely rather than rare: observed sp_cons 0.347x against a
        # 0.999x median, purely because one of ten samples came back 2.9x high. Medians
        # here are stable to ~0.2% across runs, so those samples are measurement
        # artifacts, not kernel behaviour (CUPTI session churn, GPU clock ramp and
        # per-process allocator warmup were each tested and refuted as causes). Taking
        # a quantile instead means more samples make the gate *better* rather than more
        # fragile, which is the whole point of raising repeat. The true min/max are
        # still recorded and still drive the instability warning.
        c_tail, b_tail = _pct(cand_s, CONS_Q), _pct(ref_s, 1.0 - CONS_Q)
        s_cons = b_tail / c_tail
        # The mirror image: the candidate's fast tail against the reference's slow one.
        # A shape only counts as a regression if the candidate loses even under this,
        # the reading most favourable to it — anything else is inside the noise.
        s_opt = _pct(ref_s, CONS_Q) / _pct(cand_s, 1.0 - CONS_Q)
        # Three outcomes, not two. min(sp_cons) > 1 required EVERY shape to win, which
        # is unreachable the moment one shape merely matches: an identical-to-reference
        # candidate measures sp_cons 0.855-0.989, never above 1.0. That made per-shape
        # fallback — what SGLang itself does, see
        # deepgemm_w8a8_block_fp8_linear_with_fallback — impossible to express: winning
        # 1.5x on M=16 and falling back on M=32 scored "not faster". A shape now wins,
        # regresses, or is neutral, and neutral does not veto.
        shape_verdict = ("win" if s_cons > min_speedup_gate
                         else "regress" if s_opt < 1.0 else "neutral")
        sp_med_all.append(s_med)
        sp_cons_all.append(s_cons)
        shape_verdicts.append(shape_verdict)

        # The conservative margin divides extremes, so ONE bad sample decides the
        # verdict. Medians here are stable to ~0.2% across runs, yet a whole
        # sample block occasionally comes back ~3x high (observed: 178us against
        # a 53us median, dragging sp_cons to 0.296x for a candidate identical to
        # the reference). Root cause is not CUPTI session churn, GPU clock ramp,
        # or per-process allocator warmup — all three were tested and refuted. It
        # is infrequent and only ever costs a win, never grants one. Rather than
        # silently gate on a number we know is junk, say so.
        spread = max(c_hi / c_lo, b_hi / b_lo)
        unstable = spread > 1.25
        row["timing_spread"] = round(spread, 3)
        row["timing_unstable"] = unstable
        if unstable:
            print(f"{'':>7} {'WARN':>5}   timing samples spread {spread:.2f}x "
                  f"(cand {c_lo*1e3:.1f}-{c_hi*1e3:.1f}us, ref {b_lo*1e3:.1f}-"
                  f"{b_hi*1e3:.1f}us) — sp_cons unreliable, re-run before trusting it")

        # ── post-timing correctness on FRESH inputs ──
        # Catches a candidate that mutates its inputs or drifts across the timed
        # iterations; the pre-check alone would not see it.
        try:
            post = _correctness(op, phase, M, S, seed, device, cand_fn)
            post_ok = post["pass"]
            row["post_timing_calc_diff"] = post.get("calc_diff")
        except Exception as exc:
            post_ok = False
            row["post_timing_error"] = f"{type(exc).__name__}: {exc}"[:200]
        row["post_timing_correct"] = post_ok
        if not post_ok:
            all_correct = False
            row["error"] = "correctness did not survive timing (state drift)"

        # ── reward ──
        flops, byts, dtype = ops.cost(op, phase, M, S)
        cand_r = ops.reward(c_med, flops, byts, dtype)
        ref_r = ops.reward(b_med, flops, byts, dtype)
        row.update(
            flops=flops, bytes_hbm=byts, compute_dtype=dtype,
            candidate_us=round(c_med * 1e3, 3), candidate_us_lo=round(c_lo * 1e3, 3),
            candidate_us_hi=round(c_hi * 1e3, 3),
            reference_us=round(b_med * 1e3, 3), reference_us_lo=round(b_lo * 1e3, 3),
            reference_us_hi=round(b_hi * 1e3, 3),
            samples=args.repeat,
            candidate_us_p90=round(c_tail * 1e3, 3),
            reference_us_p10=round(b_tail * 1e3, 3),
            conservative_quantile=CONS_Q,
            speedup=round(s_med, 4), speedup_conservative=round(s_cons, 4),
            speedup_optimistic=round(s_opt, 4), shape_verdict=shape_verdict,
            bound=cand_r["bound"], arithmetic_intensity=cand_r["arithmetic_intensity"],
            ridge=cand_r["ridge"],
            reward=cand_r["reward"], reference_reward=ref_r["reward"],
            achieved_tflops=cand_r["tflops"], achieved_gbps=cand_r["gbps"],
            compute_util=cand_r["compute_util"], bw_util=cand_r["bw_util"],
        )
        per_shape.append(row)

        mark = "PASS" if (ok and post_ok) else "DRIFT"
        print(f"{shape:>7} {mark:>5} {c['calc_diff']:>10.2e} "
              f"{c_med*1e3:>9.2f} {b_med*1e3:>9.2f} {s_med:>7.3f}x {s_cons:>7.3f}x "
              f"{shape_verdict.upper() if shape_verdict != 'neutral' else 'neutral':>8} "
              f"{cand_r['arithmetic_intensity']:>7.1f} {cand_r['bound']:>7} "
              f"{cand_r['tflops']:>9.1f} {cand_r['compute_util']*100:>6.2f}% "
              f"{cand_r['gbps']:>9.1f} {cand_r['bw_util']*100:>6.2f}% "
              f"{cand_r['reward']:>8.4f}")
        print(f"{'':>7} {'└ ref':>5} {'baseline':>10} "
              f"{'':>9} {'':>9} {'':>8} {'':>8} {'':>8} {'':>7} {'':>7} "
              f"{ref_r['tflops']:>9.1f} {ref_r['compute_util']*100:>6.2f}% "
              f"{ref_r['gbps']:>9.1f} {ref_r['bw_util']*100:>6.2f}% "
              f"{ref_r['reward']:>8.4f}")

    # ── aggregate ──
    rewards = [r["reward"] for r in per_shape if "reward" in r]
    diffs = [r["calc_diff"] for r in per_shape if r.get("calc_diff") is not None]
    complete = len(per_shape) == len(workloads)
    wins = shape_verdicts.count("win")
    regressions = shape_verdicts.count("regress")
    aggregate = {
        "min_speedup": round(min(sp_med_all), 4) if sp_med_all else None,
        "geomean_speedup": round(_geomean(sp_med_all), 4) if sp_med_all else None,
        "min_speedup_conservative": round(min(sp_cons_all), 4) if sp_cons_all else None,
        "best_reward": round(max(rewards), 4) if rewards else None,
        "worst_reward": round(min(rewards), 4) if rewards else None,
        "worst_calc_diff": max(diffs) if diffs else None,
        "shapes_evaluated": len(per_shape),
        "complete_sweep": complete,
        "timing_unstable_shapes": [r["uuid"] for r in per_shape
                                   if r.get("timing_unstable")],
        "shapes_won": wins,
        "shapes_regressed": regressions,
        "shapes_neutral": shape_verdicts.count("neutral"),
        "regressed_shapes": [r["uuid"] for r in per_shape
                             if r.get("shape_verdict") == "regress"],
    }

    correct = bool(all_correct and per_shape and complete)
    # A win is a real gain somewhere with no regression anywhere. Requiring a gain
    # EVERYWHERE punished the correct engineering answer; requiring one nowhere would
    # pass a candidate that only ever falls back.
    perf_ok = bool(correct and wins >= 1 and regressions == 0)
    status = "CORRECT" if correct else "INCORRECT"
    exit_code = 0 if perf_ok else (1 if correct else 2)

    print()
    print(f"VERDICT: {status}")
    if correct:
        print(f"{wins}/{len(shape_verdicts)} shapes WIN, {regressions} regressed, "
              f"{shape_verdicts.count('neutral')} neutral   "
              f"geomean_speedup={aggregate['geomean_speedup']}x  "
              f"best_reward={aggregate['best_reward']}")
        print(f"performance_gate: >=1 win AND 0 regressions -> "
              f"{'MET' if perf_ok else 'NOT MET'}")
        if wins == 0 and regressions == 0:
            print("  (every shape is inside the noise band — a candidate that only "
                  "matches the baseline is not a win)")
        if regressions:
            print(f"  regressed: {', '.join(aggregate['regressed_shapes'])} — the "
                  f"candidate loses there even at its fastest sample vs the "
                  f"reference's slowest. Fall back to the reference on those shapes.")
        if aggregate["timing_unstable_shapes"]:
            print(f"WARNING: unstable timing on "
                  f"{', '.join(aggregate['timing_unstable_shapes'])} — the "
                  f"conservative margin above is not trustworthy; re-run.")

    result = {
        "schema_version": result_store.SCHEMA_VERSION,
        "task": {
            "name": task_dir.name, "model": meta.get("model", "glm52"),
            "operator": op, "phase": phase, "S": S, "seed": seed,
            "family": op_meta["family"], "output_kind": op_meta["output_kind"],
            "backend": op_meta["backend"],
            "diff_tol": op_meta["diff_tol"], "rel_tol": op_meta["rel_tol"],
            "abs_tol_factor": op_meta["abs_tol_factor"],
            "performance_gate": {"min_speedup": min_speedup_gate,
                                 "basis": f"conservative (q={CONS_Q})"},
        },
        "run": {
            "run_id": run_id, "started_utc": started,
            "finished_utc": result_store.utc_now(),
            "repeat": args.repeat, "repeat_scope": "in-process",
            "iterations": args.iterations, "warmup": args.warmup,
            "timing_protocol": TIMING_PROTOCOL, "device": args.device,
            "correctness_standard": ("FlashMLA check_is_allclose structure: anomaly "
                                     "positions + elementwise (abs OR rel) + DeepGEMM "
                                     "calc_diff, on a poisoned output buffer"),
            "reward_standard": "rewardbench bound-aware roofline (PR2)",
        },
        "candidate": {"path": cand_label, "sha256": cand_sha,
                      "is_reference_fallback": cand_label == "reference",
                      "external": bool(args.candidate),
                      "_abspath": cand_path},
        "environment": result_store.capture_environment(),
        "cost_model": ops.PEAKS,
        "per_shape": per_shape,
        "aggregate": aggregate,
        "verdict": {"correct": correct, "performance_ok": perf_ok,
                    "status": status, "exit_code": exit_code},
    }
    return result, exit_code


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("task_dir", nargs="?", default=None)
    # Default 10, not 1. At --repeat 1 the conservative margin collapses to the
    # median one, and measured noise on this hardware is +-5%: a candidate that
    # IS the reference then scores 0.947x-1.022x and passes the >1.0 gate about
    # half the time. Ten samples make the gate demand a real margin.
    ap.add_argument("--repeat", type=int, default=10,
                    help="samples per shape; 1 is a probe and cannot gate a win")
    ap.add_argument("--iterations", type=int, default=30, help="cold-L2 reps per sample")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--max-workloads", type=int, default=None)
    ap.add_argument("--candidate", default=None, metavar="PATH",
                    help="a .py defining run(inputs), or a directory holding "
                         "candidate.py/solution.py/impl.py. May live anywhere — the "
                         "kernel under test need not be in this repo, and testing it "
                         "does not require editing the task. "
                         "Default: <task_dir>/candidate.py")
    ap.add_argument("--M", type=int, default=None, help="single shape instead of the sweep")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-persist", action="store_true")
    ap.add_argument("--describe", action="store_true",
                    help="print the problem statement (generated from glm52_ops) and exit")
    ap.add_argument("--json", action="store_true",
                    help="with --describe: emit the problem definition as JSON")
    args = ap.parse_args()

    task_dir = Path(args.task_dir).resolve() if args.task_dir else Path.cwd()
    args.repeat = max(1, args.repeat)

    if not (task_dir / "task.json").is_file():
        print(f"ERROR: no task.json in {task_dir}", file=sys.stderr)
        return 3

    if args.describe:
        meta = json.loads((task_dir / "task.json").read_text())
        op = meta["operator"]
        # Pass the device only when there is one: describe() reads the tensor table
        # off a real build_inputs call, which needs a GPU. Without it the contract
        # still prints, minus the shape table.
        dev = args.device if torch.cuda.is_available() else None
        if args.json:
            print(json.dumps(ops.problem(op, meta["phase"], device=dev), indent=2))
        else:
            print(ops.describe(op, meta["phase"], device=dev))
        return 0

    if not torch.cuda.is_available():
        print("ERROR: CUDA required", file=sys.stderr)
        return 3

    real_stdout = sys.stdout
    tee = _Tee(real_stdout)
    sys.stdout = tee
    try:
        result, code = evaluate(task_dir, args)
    except ContractError as exc:
        sys.stdout = real_stdout
        print(f"CONTRACT ERROR: {exc}", file=sys.stderr)
        return 3
    except Exception:
        sys.stdout = real_stdout
        traceback.print_exc()
        return 3
    finally:
        sys.stdout = real_stdout

    # A Path is not JSON-serialisable and this key is internal, so it has to leave
    # `result` before persist() serialises it — but persist still needs the location
    # to copy the exact bytes that ran into the run directory.
    cand_abspath = result["candidate"].pop("_abspath", None)

    if not args.no_persist:
        try:
            d = result_store.persist(
                result, model=result["task"]["model"], task=task_dir.name,
                run_id=result["run"]["run_id"], stdout_text=tee.buffer_text.getvalue(),
                candidate_path=cand_abspath)
            rel = d.relative_to(_REPO_ROOT)
            print(f"result={rel}/result.json")
            result["run"]["result_dir"] = str(rel)
        except Exception as exc:
            print(f"warning: persistence failed: {exc}", file=sys.stderr)

    print()
    print("RESULT_JSON_BEGIN")
    print(json.dumps(result, indent=2))
    print("RESULT_JSON_END")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
