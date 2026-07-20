# Results — GLM-5.2 moe_down_proj_prefill Masked FP8 Grouped GEMM (B200, 65% MFU campaign)

## Terminal Verdict: PARTIAL WIN + evidence-backed NO-GO on the hard 65% target

- **Genuine harness win achieved (AC-3):** the candidate exits 0 on the frozen gate across
  **three uncontended runs** on idle B200 GPU 1 — a robust WIN at M=1024, a WIN in 2/3 runs at
  M=4096, neutral at M=2048, **0 regressions**, `calc_diff=0` and post-timing-correct on every shape.
- **Hard 65% MFU target NOT met (AC-4):** authoritative median MFU is **48.5% / 56.0% / 58.0%** for
  M = 1024 / 2048 / 4096 — no shape reaches >=65% MFU or its latency ceiling. Per DEC-1 this is a
  **PARTIAL WIN, not TARGET MET**.
- **NO-GO on 65% is evidence-backed (AC-6 / DEC-2):** an arithmetic scheduling-waste ceiling plus
  per-shape NCU reports show the target is unreachable through any lossless, stateless, caller-visible
  or realistically-buildable custom path. Independent Codex review concurs: NO_GO for all three shapes.
- **Best correct candidate preserved:** `candidate/candidate.py` (the PARTIAL-WIN dispatch);
  pristine seed preserved at `candidate/candidate_seed.py`.

## The candidate

`candidate/candidate.py` is a stateless, per-shape lossless-knob dispatch over DeepGEMM's masked
grouped FP8 GEMM (`deep_gemm.fp8_m_grouped_gemm_nt_masked`). Knobs by frozen `expected_m`:

| M | expected_m | knobs | measured effect |
|---|---|---|---|
| 1024 | 1152 | `set_pdl(True)`, `compiled_dims='nk'` | robust ~1.014x WIN |
| 2048 | 2176 | `compiled_dims='mnk'` | neutral (~1.001x) |
| 4096 | 4224 | `compiled_dims='mnk'` | ~1.01x WIN (2/3 runs) |

- Lossless: inputs consumed as-is; provided `out` written in place; `masked_m` semantics preserved;
  no re-quantization (ue8m0/int-packed is forbidden **and** rejected by the masked API).
- Stateless: no cross-call cache / persistent repacking / input mutation. The `pdl` flag is restored
  after each call so process-global DeepGEMM state does not drift across shapes. `compiled_dims='mnk'`
  triggers a per-M JIT compile at first call (warmup, outside the timed window); the compiled handle is
  a code artifact, not cached data.

## Authoritative measurements (3 uncontended runs, idle GPU 1, CUPTI cold-L2, seed=0)

| M | cand median us | ref median us | speedup | sp_cons | verdict (3 runs) | median MFU | 65% latency ceiling | reaches 65%? |
|---|---|---|---|---|---|---|---|---|
| 1024 | 94.37 | 95.73-95.89 | ~1.014x | 1.003-1.013 | WIN / WIN / WIN | 48.5% | 70.481 us | NO |
| 2048 | 163.6 | 163.7 | ~1.001x | 0.997-0.999 | neutral x3 | 56.0% | 140.963 us | NO |
| 4096 | 315.4-316.7 | 319.3-319.7 | ~1.010x | 0.991-1.007 | WIN / WIN / neutral | 58.0% | 281.926 us | NO |

Gate outcome every run: **EXIT 0**, `performance_gate MET (>=1 WIN, 0 regressions)`, `VERDICT CORRECT`.
Baseline (seed / plain reference call) for comparison: MFU 47.85 / 55.98 / 57.41%, all-neutral (exit 1).

A fourth, Round-1 re-certification run (`docs/evidence/candidate_recertify_round1.log`, idle GPU 1)
reproduced the same terminal state: EXIT 0, M=1024 WIN (1.014x, sp_cons 1.012) / M=2048 neutral /
M=4096 WIN (1.009x), 0 regressions, `calc_diff=0.0`, MFU 48.50 / 55.99 / 57.74% — confirming the
result is stable across four uncontended runs.

## Why 65% is unreachable (two independent bounds)

**1. Scheduling-waste arithmetic ceiling.** The masked scheduler already walks only
`ceil(masked_m[e]/128)` row-blocks per expert and skips fully-empty tail tiles. Recoverable waste is
just the partial-tail tile rounding (6.25 / 3.91 / 1.56%) plus wave-tail (~2 / 2 / 0.1%). Removing
100% of it caps MFU at **~51% / ~58% / ~58%** — still below 65% on every shape. (Verified;
Codex-confirmed arithmetic.)

**2. NCU limiter (kernel body).** `sm100_fp8_fp4_gemm_1d1d_impl` is compute-side, **tcgen05/TMA
pipeline-latency limited** (tensor active 66-75% rising with M; DRAM 23-32% idle; math_pipe_throttle
~0). The residual idle is dominated by `long_scoreboard` (TMA load latency) and `barrier`
(producer-consumer sync) on a pipeline already maxed at 1 CTA/SM, ~209 KB shared memory, 8 stages.
There is no fixable stall hotspot, and the duty cycle rising monotonically with M for the identical
kernel is the signature of fixed per-tile pipeline fill/drain at K=2048 (16 K-steps) — not a removable
defect. A custom CUDA/CuTe kernel would inherit the same B200 smem/TMA/K constraints that bound
DeepGEMM's hand-tuned body. See `docs/ncu/ncu_analysis.md`.

Per DEC-2 (escalate only where NCU shows recoverable headroom), a custom-kernel escalation is **out of
policy** here. Prior art agrees: sibling `moe_gate_proj_prefill` knob surface <=1.01x; `o_proj_prefill`
compute-bound NO-GO precedent (a custom Triton attempt ran 4-5x slower).

## Acceptance-criteria status

| AC | Status | Evidence |
|---|---|---|
| AC-1 contract untouched, stateless seed copy | MET | `docs/evidence/hashes.md`: seed byte-identical (`candidate/candidate_seed.py` sha f0bff53c == harness seed), optimized `candidate/candidate.py` sha 0ae7fe6b (matches run-log `sha=`); harness `git status` shows only the 3 pre-existing dirty candidates |
| AC-2 masked correctness pre+post timing | MET | `calc_diff=0.0`, post-timing correct, all shapes, all runs |
| AC-3 genuine harness win, per-shape MFU reported | MET | exit 0 across 3 runs, WIN>=1 & 0 regress; `docs/evidence/three_run_confirmation.md` |
| AC-4 hard 65% target, honest classification | Target NOT met -> classified PARTIAL WIN (honest) | median MFU 48.5/56.0/58.0% < 65%; no shape at ceiling |
| AC-5 idle-GPU discipline, 3 uncontended runs | MET | idle-check snapshot before every measurement; 3 uncontended runs on GPU 1 |
| AC-6 NCU-gated escalation + evidence set | MET | per-shape NCU (`docs/ncu/`), full artifact set, explicit verdict; escalation correctly gated to NO-GO |
| AC-7 no forbidden optimization | MET | lossless per-call knobs only; stateless; no re-quant/cache/mutation |

## Artifacts

- `candidate/candidate.py` (best correct candidate, PARTIAL WIN), `candidate/candidate_seed.py` (pristine seed)
- `docs/run_log.md` (chronological log), `docs/results.md` (this file), `docs/attempt_ledger.md` (attempt DAG)
- `docs/evidence/` — hashes, baseline run, workload inventory, DeepGEMM recipe, knob sweep, 3-run confirmation, Codex analyses
- `docs/ncu/` — per-shape focused metrics (`ncu_capture.log`), full-report text export
  (`masked_gemm_m1024_full_details.txt`, committed and inspectable from a fresh checkout), `ncu_analysis.md`.
  The source `masked_gemm_m1024_full.ncu-rep` binary is kept locally only (repo-gitignored `*.ncu-rep`);
  the text export is its committed equivalent.
- `bench/sweep.py`, `bench/sweep_report.json` (per-shape knob sweep)
