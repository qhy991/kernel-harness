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
  5. Re-check correctness on freshly built inputs with a DIFFERENT seed after
     timing, so a candidate cannot memoize the one frozen oracle answer.
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

`--repeat K` (default and minimum gate-eligible value: 10) alternates adjacent
reference/candidate pairs as R/C, C/R, ... and gates on the p10 and p90 of the
per-pair `reference / candidate` ratios.  Pairing cancels the 2.1--4.7x
op-specific in-process drift measured during the campaign; independent candidate
and reference quantiles do not.  Raw ordered pairs and the legacy unpaired values
are both persisted.  A timing spread above 1.25x triggers one automatic 3x retry;
if it remains unstable the shape is `UNSTABLE_NO_VERDICT`, never a win.  Restricted
sweeps and repeat counts below 10 are probes and cannot return exit 0.

The inner timer uses warmup=8 and discards its first recorded iteration, which was
measured at 3--5x steady state even after the old warmup=3.

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

ops = _sibling("glm52_ops")          # source of truth for all leaf/region contracts
candidate_loader = _sibling("candidate_loader")
result_store = _sibling("result_store")
RH = _sibling("reward_hack")
tb_timing = _sibling("timing")       # testbench CUPTI timer
gpu_lease = _sibling("gpu_lease")    # free-GPU pick + per-GPU timing flock
paired_stats = _sibling("paired_stats")

TIMING_PROTOCOL = (
    "cupti-cold-l2-device-kernel-median-paired-ratio-first-sample-discarded"
    if tb_timing._HAVE_CUPTI else
    "event-cold-l2-median-paired-ratio-first-sample-discarded-NO-CUPTI"
)
MIN_GATE_REPEAT = 10
MIN_GATE_ITERATIONS = 2
DEFAULT_WARMUP = 8
RETRY_MULTIPLIER = 3
POST_TIMING_SEED_XOR = 0x5EED
PHYSICAL_REWARD_LIMIT = 1.0


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


CONS_Q = paired_stats.CONSERVATIVE_QUANTILE


def _measure_pairs(cand_fn, ref_fn, setup, *, warmup, iterations, repeat, device):
    """Measure adjacent, order-balanced pairs and retain their literal order."""
    candidate_samples: list[float] = []
    reference_samples: list[float] = []
    orders = paired_stats.balanced_orders(repeat)

    def one(fn):
        return tb_timing.time_runnable(
            fn, setup=setup, warmup=warmup, rep=iterations, device=device)

    for order in orders:
        if order == "reference,candidate":
            reference_samples.append(one(ref_fn))
            candidate_samples.append(one(cand_fn))
        else:
            candidate_samples.append(one(cand_fn))
            reference_samples.append(one(ref_fn))
    return candidate_samples, reference_samples, orders


def _timing_pair_rows(candidate_samples, reference_samples, orders):
    return [
        {
            "pair": index,
            "order": order,
            "candidate_us": round(candidate_ms * 1e3, 3),
            "reference_us": round(reference_ms * 1e3, 3),
            "speedup": round(reference_ms / candidate_ms, 6),
        }
        for index, (candidate_ms, reference_ms, order) in enumerate(
            zip(candidate_samples, reference_samples, orders)
        )
    ]


def _compact_gpu_telemetry(snapshot: dict) -> dict:
    """Keep the timed GPU row per attempt; the environment block retains all GPUs."""
    return {
        key: snapshot.get(key)
        for key in ("captured_utc", "physical_index", "selected")
    }


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

    if getattr(args, "auto_gpu", False) and not getattr(args, "_gpu_autoselected", False):
        args.device = f"cuda:{gpu_lease.pick_idle_gpu(gpu_lease.device_index(args.device))}"
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
    measurement_invalid = False
    all_workloads = _load_workloads(task_dir, None, None)
    workloads = _load_workloads(task_dir, args.M, args.max_workloads)
    expected_ms = sorted(int(wl["axes"]["M"]) for wl in all_workloads)
    requested_ms = sorted(int(wl["axes"]["M"]) for wl in workloads)
    full_sweep_requested = requested_ms == expected_ms
    probe_reasons = []
    if not full_sweep_requested:
        probe_reasons.append(
            f"restricted workload sweep {requested_ms}; canonical sweep is {expected_ms}")
    if args.repeat < MIN_GATE_REPEAT:
        probe_reasons.append(
            f"repeat={args.repeat} is below the gate minimum {MIN_GATE_REPEAT}")
    if args.warmup < DEFAULT_WARMUP:
        probe_reasons.append(
            f"warmup={args.warmup} is below the gate minimum {DEFAULT_WARMUP}")
    if args.iterations < MIN_GATE_ITERATIONS:
        probe_reasons.append(
            f"iterations={args.iterations} cannot discard an internal first sample")
    if probe_reasons:
        print("     PROBE ONLY: " + "; ".join(probe_reasons))
        print()

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

        # ── performance (same inputs, same ABI, adjacent balanced pairs) ──
        inputs = c["inputs"]
        ops.poison(inputs)
        ref_fn = partial(ops.reference, op, phase)
        setup = lambda: clone_inputs(inputs)  # noqa: E731 — cost is not timed
        physical_index = getattr(args, "_gpu_lease_physical_index", None)

        # The CLI holds the per-GPU lock around the whole GPU gate. Keep this
        # fallback for direct evaluate() callers that did not take the outer lock.
        inner_lock = not (getattr(args, "no_gpu_lock", False) or
                          getattr(args, "_gpu_lock_held", False))
        attempts = []
        attempt_repeat = args.repeat
        with gpu_lease.gpu_timing_lock(device, enabled=inner_lock):
            while True:
                telemetry_before = result_store.capture_gpu_telemetry(physical_index)
                cand_s, ref_s, orders = _measure_pairs(
                    cand_fn, ref_fn, setup, warmup=args.warmup,
                    iterations=args.iterations, repeat=attempt_repeat, device=device)
                telemetry_after = result_store.capture_gpu_telemetry(physical_index)
                summary = paired_stats.summarize_pairs(cand_s, ref_s)
                attempts.append({
                    "attempt": len(attempts) + 1,
                    "repeat": attempt_repeat,
                    "pairs": _timing_pair_rows(cand_s, ref_s, orders),
                    "timing_spread": round(summary["timing_spread"], 6),
                    "timing_unstable": summary["timing_unstable"],
                    "gpu_telemetry_before": _compact_gpu_telemetry(telemetry_before),
                    "gpu_telemetry_after": _compact_gpu_telemetry(telemetry_after),
                })
                if summary["timing_unstable"] and len(attempts) == 1:
                    attempt_repeat = args.repeat * RETRY_MULTIPLIER
                    print(f"{'':>7} {'RETRY':>5}   timing spread "
                          f"{summary['timing_spread']:.2f}x > "
                          f"{paired_stats.TIMING_SPREAD_LIMIT:.2f}x; retrying with "
                          f"{attempt_repeat} adjacent pairs")
                    continue
                break
        RH.check_monkey_patch()

        c_lo = summary["candidate_min_ms"]
        c_med = summary["candidate_median_ms"]
        c_hi = summary["candidate_max_ms"]
        b_lo = summary["reference_min_ms"]
        b_med = summary["reference_median_ms"]
        b_hi = summary["reference_max_ms"]
        s_med = summary["speedup_median"]
        s_cons = summary["speedup_conservative"]
        s_opt = summary["speedup_optimistic"]
        unstable = summary["timing_unstable"]

        if unstable:
            shape_verdict = "unstable"
            print(f"{'':>7} {'WARN':>5}   retry still spread "
                  f"{summary['timing_spread']:.2f}x (cand {c_lo*1e3:.1f}-"
                  f"{c_hi*1e3:.1f}us, ref {b_lo*1e3:.1f}-{b_hi*1e3:.1f}us); "
                  "this shape has no performance verdict")
        else:
            shape_verdict = ("win" if s_cons > min_speedup_gate
                             else "regress" if s_opt < 1.0 else "neutral")

        # ── post-timing correctness on a DIFFERENT seed ──
        # A fresh allocation with the same values does not catch memoization. The
        # distinct seed makes the post-check an adversarial unseen input as well as
        # a state-drift check.
        post_seed = seed ^ POST_TIMING_SEED_XOR
        try:
            post = _correctness(op, phase, M, S, post_seed, device, cand_fn)
            post_ok = post["pass"]
            row["post_timing_calc_diff"] = post.get("calc_diff")
        except Exception as exc:
            post_ok = False
            row["post_timing_error"] = f"{type(exc).__name__}: {exc}"[:200]
        row["post_timing_seed"] = post_seed
        row["post_timing_correct"] = post_ok
        row["correct"] = bool(ok and post_ok)
        if not post_ok:
            all_correct = False
            shape_verdict = "invalid"
            row["error"] = "correctness failed on the unseen post-timing seed"

        # ── reward and physical plausibility ──
        flops, byts, dtype = ops.cost(op, phase, M, S)
        cand_r = ops.reward(c_med, flops, byts, dtype)
        ref_r = ops.reward(b_med, flops, byts, dtype)
        physical_violation = max(cand_r["reward"], ref_r["reward"]) > PHYSICAL_REWARD_LIMIT
        if physical_violation:
            measurement_invalid = True
            shape_verdict = "invalid"
            row["physical_reward_violation"] = {
                "limit": PHYSICAL_REWARD_LIMIT,
                "candidate_reward": cand_r["reward"],
                "reference_reward": ref_r["reward"],
            }
            print(f"{'':>7} {'ERROR':>5}   physical reward exceeds 1.0 "
                  f"(candidate={cand_r['reward']:.4f}, reference={ref_r['reward']:.4f}); "
                  "byte/cost model or timing is invalid")

        sp_med_all.append(s_med)
        sp_cons_all.append(s_cons)
        shape_verdicts.append(shape_verdict)
        row.update(
            flops=flops, bytes_hbm=byts, compute_dtype=dtype,
            candidate_us=round(c_med * 1e3, 3), candidate_us_lo=round(c_lo * 1e3, 3),
            candidate_us_hi=round(c_hi * 1e3, 3),
            reference_us=round(b_med * 1e3, 3), reference_us_lo=round(b_lo * 1e3, 3),
            reference_us_hi=round(b_hi * 1e3, 3),
            samples=len(cand_s), timing_attempts=attempts,
            timing_retry_performed=len(attempts) > 1,
            timing_spread=round(summary["timing_spread"], 6),
            timing_unstable=unstable,
            timing_pairs=_timing_pair_rows(cand_s, ref_s, orders),
            conservative_quantile=CONS_Q,
            speedup=round(s_med, 4), speedup_conservative=round(s_cons, 4),
            speedup_optimistic=round(s_opt, 4),
            speedup_unpaired_median=round(summary["speedup_unpaired_median"], 4),
            speedup_unpaired_conservative=round(
                summary["speedup_unpaired_conservative"], 4),
            speedup_unpaired_optimistic=round(
                summary["speedup_unpaired_optimistic"], 4),
            shape_verdict=shape_verdict,
            bound=cand_r["bound"], arithmetic_intensity=cand_r["arithmetic_intensity"],
            ridge=cand_r["ridge"],
            reward=cand_r["reward"], reference_reward=ref_r["reward"],
            achieved_tflops=cand_r["tflops"], achieved_gbps=cand_r["gbps"],
            compute_util=cand_r["compute_util"], bw_util=cand_r["bw_util"],
        )
        per_shape.append(row)

        mark = "PASS" if (ok and post_ok and not physical_violation) else "INVALID"
        shown_verdict = shape_verdict.upper() if shape_verdict != "neutral" else "neutral"
        print(f"{shape:>7} {mark:>5} {c['calc_diff']:>10.2e} "
              f"{c_med*1e3:>9.2f} {b_med*1e3:>9.2f} {s_med:>7.3f}x {s_cons:>7.3f}x "
              f"{shown_verdict:>8} "
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
    requested_complete = len(per_shape) == len(workloads)
    complete = bool(requested_complete and full_sweep_requested)
    wins = shape_verdicts.count("win")
    regressions = shape_verdicts.count("regress")
    unstable_shapes = [r["uuid"] for r in per_shape if r.get("timing_unstable")]
    invalid_shapes = [r["uuid"] for r in per_shape
                      if r.get("shape_verdict") == "invalid"]
    physical_invalid_shapes = [r["uuid"] for r in per_shape
                               if r.get("physical_reward_violation")]
    aggregate = {
        "min_speedup": round(min(sp_med_all), 4) if sp_med_all else None,
        "geomean_speedup": round(_geomean(sp_med_all), 4) if sp_med_all else None,
        "min_speedup_conservative": round(min(sp_cons_all), 4) if sp_cons_all else None,
        "best_reward": round(max(rewards), 4) if rewards else None,
        "worst_reward": round(min(rewards), 4) if rewards else None,
        "worst_calc_diff": max(diffs) if diffs else None,
        "shapes_evaluated": len(per_shape),
        "requested_shapes": len(workloads),
        "canonical_shapes": len(all_workloads),
        "requested_complete": requested_complete,
        "complete_sweep": complete,
        "timing_unstable_shapes": unstable_shapes,
        "invalid_shapes": invalid_shapes,
        "measurement_invalid_shapes": physical_invalid_shapes,
        "shapes_won": wins,
        "shapes_regressed": regressions,
        "shapes_neutral": shape_verdicts.count("neutral"),
        "regressed_shapes": [r["uuid"] for r in per_shape
                             if r.get("shape_verdict") == "regress"],
    }

    correct = bool(all_correct and per_shape and requested_complete)
    gate_eligible = bool(
        complete and args.repeat >= MIN_GATE_REPEAT and args.warmup >= DEFAULT_WARMUP and
        args.iterations >= MIN_GATE_ITERATIONS and
        not unstable_shapes and not measurement_invalid
    )
    # A win is a real gain somewhere with no regression anywhere. Requiring a gain
    # EVERYWHERE punished the correct engineering answer; requiring one nowhere would
    # pass a candidate that only ever falls back.
    perf_ok = bool(correct and gate_eligible and wins >= 1 and regressions == 0)
    status = "INVALID" if measurement_invalid else ("CORRECT" if correct else "INCORRECT")
    exit_code = 2 if measurement_invalid else (0 if perf_ok else (1 if correct else 2))
    if measurement_invalid:
        terminal_state = "INVALID_MEASUREMENT"
        terminal_reason = "roofline reward exceeded the physical limit of 1.0"
    elif not correct:
        terminal_state = "INCORRECT_OR_INCOMPLETE"
        terminal_reason = "correctness failed, the requested sweep was incomplete, or the unseen-seed post-check failed"
    elif probe_reasons:
        terminal_state = "PROBE_ONLY_NO_VERDICT"
        terminal_reason = "; ".join(probe_reasons)
    elif unstable_shapes:
        terminal_state = "UNSTABLE_NO_VERDICT"
        terminal_reason = "timing remained unstable after the automatic 3x retry"
    elif perf_ok:
        terminal_state = "COMPLETE_WIN"
        terminal_reason = "correct on every shape, at least one shape won, and no shape regressed"
    elif correct and wins == 0 and regressions == 0:
        terminal_state = "NO_WIN_WITH_EVIDENCE"
        terminal_reason = "correct complete sweep, but every shape was inside the noise band"
    elif correct:
        terminal_state = "PARTIAL_OR_REGRESSED_WITH_EVIDENCE"
        terminal_reason = "correct complete sweep, but the candidate regressed on at least one shape"
    else:  # Defensive: all semantic states above should be exhaustive.
        terminal_state = "INCORRECT_OR_INCOMPLETE"
        terminal_reason = "the run could not produce a gate-eligible verdict"

    print()
    print(f"VERDICT: {status}  terminal={terminal_state}")
    if correct:
        print(f"{wins}/{len(shape_verdicts)} shapes WIN, {regressions} regressed, "
              f"{shape_verdicts.count('neutral')} neutral   "
              f"geomean_speedup={aggregate['geomean_speedup']}x  "
              f"best_reward={aggregate['best_reward']}")
        print(f"performance_gate: eligible full sweep AND >=1 win AND 0 regressions -> "
              f"{'MET' if perf_ok else 'NOT MET'}")
        if wins == 0 and regressions == 0:
            print("  (every shape is inside the noise band — a candidate that only "
                  "matches the baseline is not a win)")
        if regressions:
            print(f"  regressed: {', '.join(aggregate['regressed_shapes'])} — the "
                  f"candidate loses there even at its fastest sample vs the "
                  f"reference's slowest. Fall back to the reference on those shapes.")
        if unstable_shapes:
            print(f"NO VERDICT: timing remained unstable on {', '.join(unstable_shapes)} "
                  "after the automatic retry.")
        if probe_reasons:
            print("NO VERDICT: probe-only run — " + "; ".join(probe_reasons))
    if measurement_invalid:
        print("HARD FAIL: reward > 1.0 is physically impossible under the recorded "
              "cost model; inspect timing and modeled bytes before using this result.")

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
                                 "basis": f"paired ratio p{int(CONS_Q * 100)}"},
        },
        "run": {
            "run_id": run_id, "started_utc": started,
            "finished_utc": result_store.utc_now(),
            "repeat": args.repeat, "minimum_gate_repeat": MIN_GATE_REPEAT,
            "repeat_scope": "in-process",
            "iterations": args.iterations,
            "minimum_gate_iterations": MIN_GATE_ITERATIONS,
            "warmup": args.warmup,
            "timing_protocol": TIMING_PROTOCOL, "device": args.device,
            "pairing": "adjacent balanced R/C,C/R",
            "unstable_retry_multiplier": RETRY_MULTIPLIER,
            "timing_spread_limit": paired_stats.TIMING_SPREAD_LIMIT,
            "first_recorded_iteration_discarded": True,
            "post_timing_seed": seed ^ POST_TIMING_SEED_XOR,
            "post_timing_seed_differs": True,
            "full_sweep_requested": full_sweep_requested,
            "gate_eligible": gate_eligible,
            "probe_reasons": probe_reasons,
            "physical_reward_limit": PHYSICAL_REWARD_LIMIT,
            "auto_gpu": bool(getattr(args, "auto_gpu", False)),
            "gpu_lock_enabled": not bool(getattr(args, "no_gpu_lock", False)),
            "gpu_lock_held": bool(getattr(args, "_gpu_lock_held", False)),
            "gpu_physical_index": getattr(args, "_gpu_lease_physical_index", None),
            "gpu_lock_path": getattr(args, "_gpu_lock_path", None),
            "correctness_standard": ("FlashMLA check_is_allclose structure: anomaly "
                                     "positions + elementwise (abs OR rel) + DeepGEMM "
                                     "calc_diff, on a poisoned output buffer"),
            "reward_standard": "rewardbench bound-aware roofline (PR2)",
        },
        "candidate": {"path": cand_label, "sha256": cand_sha,
                      "is_reference_fallback": cand_label == "reference",
                      "external": bool(args.candidate),
                      "git": result_store.candidate_git_state(cand_path),
                      "_abspath": cand_path},
        "environment": result_store.capture_environment(),
        "cost_model": ops.PEAKS,
        "per_shape": per_shape,
        "aggregate": aggregate,
        "verdict": {"correct": correct, "performance_ok": perf_ok,
                    "measurement_valid": not measurement_invalid,
                    "status": status, "exit_code": exit_code,
                    "terminal_state": terminal_state,
                    "terminal_reason": terminal_reason},
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
    ap.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    ap.add_argument("--max-workloads", type=int, default=None)
    ap.add_argument("--candidate", default=None, metavar="PATH",
                    help="a .py defining run(inputs), or a directory holding "
                         "candidate.py/solution.py/impl.py. May live anywhere — the "
                         "kernel under test need not be in this repo, and testing it "
                         "does not require editing the task. "
                         "Default: <task_dir>/candidate.py")
    ap.add_argument("--M", type=int, default=None, help="single shape instead of the sweep")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--auto-gpu", action="store_true",
                    help="pick the least-busy GPU via nvidia-smi (overrides --device index)")
    ap.add_argument("--no-gpu-lock", action="store_true",
                    help="do not hold the per-GPU flock around the GPU gate")
    ap.add_argument("--no-persist", action="store_true")
    ap.add_argument("--describe", action="store_true",
                    help="print the problem statement (generated from glm52_ops) and exit")
    ap.add_argument("--json", action="store_true",
                    help="with --describe: emit the problem definition as JSON")
    args = ap.parse_args()

    task_dir = Path(args.task_dir).resolve() if args.task_dir else Path.cwd()
    args.repeat = max(1, args.repeat)
    if args.warmup < 0 or args.iterations < 1:
        ap.error("--warmup must be >= 0 and --iterations must be >= 1")

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
        lock_enabled = not getattr(args, "no_gpu_lock", False)
        default_idx = gpu_lease.device_index(args.device)
        if getattr(args, "auto_gpu", False):
            lock_cm = gpu_lease.locked_idle_gpu(default=default_idx, enabled=lock_enabled)
        else:
            lock_cm = gpu_lease.gpu_timing_lock(args.device, enabled=lock_enabled)
        with lock_cm as lease:
            if getattr(args, "auto_gpu", False) and lease is not None:
                args.device = f"cuda:{lease.index}"
                args._gpu_autoselected = True
            args._gpu_lock_held = bool(lock_enabled and lease is not None and lease.file is not None)
            args._gpu_lease_physical_index = getattr(lease, "physical_index", None) if lease is not None else None
            args._gpu_lock_path = getattr(getattr(lease, "file", None), "name", None) if lease is not None else None
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
            print(f"result={result['run']['result_dir']}/result.json")
        except Exception as exc:
            print(f"warning: persistence failed: {exc}", file=sys.stderr)

    print()
    print("RESULT_JSON_BEGIN")
    print(json.dumps(result, indent=2))
    print("RESULT_JSON_END")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
