# Results — glm52 / o_proj_decode (Kernel-Harness, B200)

**Verdict: WIN (gate exit 0, drop-in verified).** GLM-5.2-FP8 Attention O-Projection decode
FP8 blockwise GEMM, DeepGEMM `w8a8_block_fp8`, K=16384, N=6144, M∈{16,32}, on B200 GPU 0.

## Outcome

A single-line change to `solution.py`'s `run()` — dispatching `deep_gemm.fp8_gemm_nt`
directly with **`compiled_dims="nk"`** (baking N=6144 and K=16384 as compile-time template
constants; the production sglang wrapper leaves them dynamic, `compiled_dims=""`) — beats the
SGLang production DeepGEMM baseline on **every** frozen shape while matching output exactly.

Authoritative gate: `evaluate.py testbench/tasks/glm52/o_proj_decode --repeat 3` → **exit 0 (WIN)**.

**Committed executable artifact (reproducibility).** The optimized `solution.py` lives in the sibling
Kernel-Harness checkout behind the untracked repo-root `Kernel-Harness/` symlink, so it is not carried by
this repo's commits on its own. The exact winning file is therefore vendored at
[`../candidate/solution.py`](../candidate/solution.py) (byte-identical; sha256 `f5e1c69d…`), with
[`../candidate/apply.sh`](../candidate/apply.sh) to place it at the harness gate path and
[`../candidate/README.md`](../candidate/README.md) for provenance + reproduction. Run `candidate/apply.sh`
then the gate commands below to reproduce the WIN from the committed files.

| Shape | baseline us | candidate us | speedup (median) | speedup_conservative (base_best/cand_worst) | achieved GB/s | % HBM peak | correctness |
|-------|-------------|--------------|------------------|---------------------------------------------|---------------|------------|-------------|
| M=16  | 32.83       | 30.53        | 1.077            | **1.069**                                   | 3332          | 41.7%      | rel_err 0, cosine 1.0 |
| M=32  | 33.34       | 31.30        | 1.065            | **1.063**                                   | 3259          | 40.7%      | rel_err 0, cosine 1.0 |

- geomean_speedup = 1.071, min_speedup_conservative = **1.063 (> 1.0 on both shapes)**.
- Correctness: matched_ratio = 1.0000 on both shapes (tol atol=0.1/rtol=0.05/ratio>=0.999), no NaN/Inf.
- Drop-in contract: `integrate.py` → **DROP-IN VERIFIED** (invoked=1, match_ratio=1.0, shape_ok, symbol restored).
- `agent_closeout.py glm52/o_proj_decode --repeat 3 --owner qinhaiyan` → exit 0, `drop_in_verified=true`,
  recommendation "WIN + drop-in verified. Record knowledge with integrate=pass."

## Why it works (bound analysis)

The op is memory-bandwidth bound: it streams a ~96 MiB FP8 weight (`bytes_moved`≈101.7 MB) to
produce only 16–32 output rows. One-pass HBM floor ≈ 101.7 MB / ~8 TB/s ≈ 12.7 us.

- Baseline (production wrapper, `compiled_dims=""`): ~32.9 us at M=16 → **38.7% of HBM peak** (≈2.6× the one-pass floor).
- Candidate (`compiled_dims="nk"`): ~30.5 us at M=16 → **41.7% of HBM peak**.

Baking N,K as compile-time constants lets the DeepGEMM scheduler/K-loop specialize (constant
trip counts, fixed tile grid = 48 N-tiles × 1 M-tile), recovering ~6–8% of the wall-clock without
touching the scale layout or adding device work. This is a genuine device-kernel win (CUPTI-timed),
not a Python-wrapper bypass. `compiled_dims="nk"` keeps M dynamic, so the candidate is a valid
drop-in across serving M values.

## Named bound / headroom

Even after the win, both shapes sit at ~40% of HBM peak — still ~2.4× above the one-pass floor.
The residual gap is memory-level-parallelism / occupancy limited (only 48 CTAs on 148 SMs at these
M), which the allowed device-side knobs (num_sms, pdl, tc_util) do not fundamentally close, and
which split-K / a custom persistent kernel (out of scope per DEC-2) would be required to attack.
So the DeepGEMM path is NOT at the true bandwidth floor, but the reachable-knob headroom beyond
`compiled_dims="nk"` is small.

## Attempts

| Attempt | Mechanism | Result |
|---------|-----------|--------|
| `compiled_dims="nk"` (this win) | direct `deep_gemm.fp8_gemm_nt` with N,K baked as template constants | **WIN**: sp_cons 1.076 (M=16) / 1.062 (M=32), correct, drop-in verified |
| `num_sms` re-tune | `deep_gemm.set_num_sms(n)` around the `nk` call, swept n∈{32,40,48,56,64,80,96,128,148} on both shapes | **No improvement** — reverted to plain `nk`. See sweep table below. |

### num_sms sweep (measured, single-shot candidate us, both shapes correct)

| num_sms | M=16 us | M=32 us | note |
|---------|---------|---------|------|
| default (`nk`, no override) | 30.4 | 31.26 | chosen candidate |
| 32 | 40.96 | 40.56 | slower — fewer SMs than the 48-tile grid, serialization |
| 40 | 37.87 | 37.66 | slower |
| 48 | 30.58 | 31.04 | ties default |
| 56 | 30.66 | 31.07 | ties default |
| 64 | 30.43 | 31.02 | ties default (best-looking) |
| 80 | 30.43 | 31.10 | ties default |
| 96 | 30.58 | 31.34 | ties default |
| 128 | 30.59 | 31.26 | ties default |
| 148 | 30.56 | 31.25 | ties default |

Conservative-gate confirmation of the best candidate (`num_sms=64`, `--repeat 3`): sp_cons M=16 1.075 /
M=32 1.066, min 1.0657 — within run-to-run noise of the plain `nk` candidate (min 1.061-1.062), with
M=16 marginally worse and M=32 marginally better. Not a robust both-shapes improvement, and it adds a
global SM-count mutation, so the plain `compiled_dims="nk"` candidate was kept. The sweep exposed no
bottleneck-sensitive setting (all n>=48 tie), so `pdl`/`tc_util` were not pursued. This confirms with
measured K=16384 evidence what the prior K=2048 sweep found: num_sms is not a stable dual-shape lever here.

## Scope note (task-scoped vs global harness status)

The o_proj_decode task-scoped diff is clean: `git -C Kernel-Harness status -- testbench/tasks/glm52/o_proj_decode/`
shows only `solution.py` changed (+14/-3). The GLOBAL harness working tree also contains an unrelated,
pre-existing modification to `testbench/tasks/glm52/dsa_prefill_attn/solution.py` (a different task) — not
touched by this campaign and left as-is. AC-6 "only solution.py changed" refers to the o_proj_decode task
scope; it is NOT a claim that the entire harness working tree is otherwise pristine.

## Relationship to prior knowledge

Prior recorded entry `glm52--o_proj_decode--b200--20260714a.json` was a **no-win at K=2048** (8× smaller
weight; achieved only ~18% of peak → under-occupied/launch-bound) and never tried `compiled_dims`.
Re-baselining at the **current K=16384** (per DEC-4) exposed a real, previously-untried lever. The
prior mechanism-level dead-ends still hold: wrapper-only bypass does not help CUPTI timing; raw
`fp8_gemm_nt` needs the exact packed UE8M0 scale layout (which this candidate preserves); Triton and
CUTLASS paths need float32 scales (unreachable via the fixed, untimed `get_inputs`).

## Recommended follow-up (not done this round, to keep the harness diff = only solution.py)

Record a knowledge entry with `knowledge.py add` capturing: win at K=16384 via `compiled_dims="nk"`,
sp_cons≈1.06–1.07, integrate=pass, and the note that the prior no-win was K=2048.

## Raw evidence

`CLOSEOUT_JSON` (win=true, drop_in_verified=true) and per-shape `VERDICT_JSON` captured from the
`--repeat 3` gate and closeout runs on GPU 0 (see docs/run_log.md for exact commands). Numbers above
are from the closeout re-run; the standalone `--repeat 3` gate agreed (sp_cons 1.078 / 1.061).
