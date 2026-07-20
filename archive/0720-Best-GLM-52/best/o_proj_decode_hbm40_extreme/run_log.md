# Run Log — GLM-5.2 o_proj_decode, 40% HBM extreme campaign

## Environment (captured 2026-07-19, `artifacts/env.txt`)

| item | value |
|---|---|
| GPU | NVIDIA B200 (SM100, capability 10.0), 4× present, GPU 0 pinned |
| Driver / CUDA | 610.43.02 / 13.0 | 
| torch / deep_gemm / triton | 2.11.0+cu130 / 0.1.4 / 3.6.0 |
| L2 cache / SMs | 132,644,864 B (~126 MB) / 148 |
| HBM peak | 8.0 TB/s |
| harness | `/home/qinhaiyan/Kernel-Harness` (HEAD 7d79e5ecbb52…, branch main) |
| campaign worktree HEAD | 58458fdf8233… (branch kda/glm52_harness_o_proj_decode_hbm40_extreme-…) |
| profilers | ncu 2026.1.1, nsys 2025.6.3 |

GPU-idle policy: GPU 0 verified idle (0% util, 0 MiB, no compute processes) before every measurement
(baseline, floor probes, NCU/nsys, final gates). Only GPU 0 used, per the pinned single-GPU policy.

## Harness dirty-tree honesty note (AC-1, `docs/harness_baseline_dirty.txt`)

The shared harness tree was ALREADY dirty at campaign start (prior/concurrent campaigns): modified
`AGENTS.md`, `README*.md`, `testbench/README.md`, `testbench/tasks/glm52/{index_k_proj_decode,
o_proj_decode,o_proj_prefill}/candidate.py`, and untracked `testbench/bin/accept_layer.py`. This
campaign added **nothing** to that tree — re-verified identical after all runs. All campaign work is
in the campaign worktree; the harness `runs/` output is gitignored.

## Seed provenance (AC-1, `docs/seed_provenance.md`)

Copied byte-for-byte from the completed hbm35 candidate
`…/glm52_harness_o_proj_decode_hbm35/candidate/` into `candidate/`:
- `candidate.py`  sha256 `054cf91a7cf5beaa886f5c6c34ed60a36afd7a0f6db30048aa3d64196c258551` (cmp: identical)
- `scale_pack.cu` sha256 `c8106d4227a351c9341f74b5b9acf9f1e3ebc9c7da340de6ce1dcc1d78bef821` (cmp: identical)

## Command ledger

| # | Purpose | Command (abbrev.) | Evidence |
|---|---|---|---|
| 1 | dirty-tree + provenance + env | `sha256sum`, `cmp -s`, `git status`, env capture | `docs/harness_baseline_dirty.txt`, `docs/seed_provenance.md`, `artifacts/env.txt` |
| 2 | build/warmup | `run.sh --candidate candidate --M 16` | `artifacts/baseline/warmup_build_M16.log` |
| 3 | seed baseline ×3 | `run.sh --candidate candidate` ×3 | `artifacts/baseline/baseline_seed_run{1,2,3}.log` |
| 4 | floor decomposition | `floor_probe.py` | `artifacts/probes/floor_probe_run1.log` |
| 5 | NCU pack + GEMM | `ncu --kernel-name … --section …` | `artifacts/ncu/{pack,gemm}_M{16,32}.txt`, `NCU_REPORT.md` |
| 6 | nsys timeline | `nsys profile … ; nsys stats` | `artifacts/nsys/` |
| 7 | single-kernel floor | `kernel_floor_probe.py` | `artifacts/probes/kernel_floor_probe.log` |
| 8 | optimized pack + byte-check + true floor | `opt_measure.py` | `artifacts/probes/opt_measure_run2.log` |
| 9 | PDL + Triton control | `pdl_triton_probe.py` | `artifacts/probes/pdl_triton_probe.log` |
| 10 | final authoritative gates ×3 | `run.sh --candidate candidate` ×3 | `artifacts/final_gates/final_gate{1,2,3}.log` |
| 11 | layer-swap acceptance | `accept_layer.py --M {16,32} --swap o_proj=…` | `artifacts/final_gates/accept_layer_M{16,32}.log` |
| 12 | compliance audit | `ask-codex.sh` (gpt-5.5:xhigh) | `docs/compliance_review.md` |
| 13 | (Round 1) DeepGEMM isolated knob controls | `deep_gemm_knob_sweep.py` (compiled_dims×PDL at defaults; num_sms and tc_util each at `nk`) | `artifacts/probes/deep_gemm_knob_sweep.log`, `docs/deep_gemm_knob_sweep.md` |
| 14 | (Round 1) best-knob variant ×3 gates | `run.sh --candidate variants/best_knob` ×3 | superseded by Round-2 re-gate (row 16) |
| 15 | (Round 2) DeepGEMM knob CROSS-PRODUCT | `deep_gemm_knob_crossproduct.py` ({nk,k,mk,mnk}×PDL×num_sms×tc_util = 96 rows/shape + non-K dominance) | `artifacts/probes/deep_gemm_knob_crossproduct.log` |
| 16 | (Round 2) best-knob (x-product optima) ×3 gates | `run.sh --candidate variants/best_knob` ×3 | `artifacts/final_gates/best_knob_gate{1,2,3}.log` |
| 17 | (Round 3) promote best-knob → `candidate/`; preserve pristine seed → `variants/seed/` | `cp` + SHA verify + knob save/restore | `docs/seed_provenance.md`, `variants/seed/` |
| 18 | (Round 3) re-gate submitted `candidate/` ×3 + knob-restore + byte-identity verify + re-audit | `run.sh --candidate candidate` ×3; `ask-codex.sh` | `artifacts/final_gates/candidate_bestknob_gate{1,2,3}.log`, `docs/codex_r3_compliance_audit.md` |

## Authoritative gate runs (best candidate A0 = seed)

| Run | M=16 µs / HBM% | M=32 µs / HBM% | correct | is_ref_fallback | exit |
|---|---|---|---|---|---|
| baseline r1 | 33.14 / 38.15% | 34.05 / 37.31% | PASS | false | 0 |
| baseline r2 | 33.09 / 38.21% | 34.10 / 37.25% | PASS | false | 0 |
| baseline r3 | 33.15 / 38.15% | 34.10 / 37.26% | PASS | false | 0 |
| final gate1 | 33.144 / 38.15% | 34.056 / 37.30% | PASS | false | 0 |
| final gate2 | 33.184 / 38.10% | 34.063 / 37.29% | PASS | false | 0 |
| final gate3 | 33.185 / 38.10% | 34.080 / 37.27% | PASS | false | 0 |

Per-shape final median: **M=16 33.184 µs / 38.10%**, **M=32 34.063 µs / 37.29%** — both MISS the 40%
ceilings (31.611 / 31.757 µs). Terminal disposition: evidence-backed NO-GO (`docs/no_go_disposition.md`).

## Rounds 1–2: DeepGEMM knob sweep + best-knob gates

Round 1 measured isolated knob controls; Round 2 measured the actual **cross-product** for the
K-compiled `compiled_dims`{nk,k,mk,mnk} × PDL{F,T} × num_sms{74,96,128,148} × tc_util{50,80,100}
(96 rows/shape) plus non-K dominance (200 rows total). **No config clears 40%**
(`ANY_CONFIG_CLEARS_40%=False`; `docs/deep_gemm_knob_sweep.md`). The best cross-product config
(PDL + per-shape optima: M=16 mnk/num_sms=148/tc=50, M=32 mk/num_sms=74/tc=80) was gated officially
as `variants/best_knob/` (Round 2 re-gate, superseding the Round 1 best-knob gate):

| Run | M=16 µs / HBM% | M=32 µs / HBM% | correct | is_ref_fallback | exit |
|---|---|---|---|---|---|
| best_knob gate1 | 32.600 / 38.79% | 33.383 / 38.05% | PASS | false | 0 |
| best_knob gate2 | 32.569 / 38.82% | 33.329 / 38.11% | PASS | false | 0 |
| best_knob gate3 | 32.584 / 38.81% | 33.327 / 38.12% | PASS | false | 0 |

Best-knob per-shape median: **M=16 32.584 µs / 38.81%**, **M=32 33.329 µs / 38.11%** — still MISS.
`num_sms`/`tc_util` restored to defaults (148/100) and PDL to False after the sweep (Round 2 uses a
`finally` block). Disposition maintained: evidence-backed NO-GO on both shapes.

## Round 3: promote best-knob to the submitted `candidate/`; fix knob leak

Code review (P2×2) required the best correct candidate be the submitted one and that the DeepGEMM
knob mutations not leak. Actions: preserved the pristine seed byte-identical at `variants/seed/`
(SHA `054cf91a…`/`c8106d42…`, `cmp -s` identical); promoted the best-knob logic into
`candidate/candidate.py` (per-shape PDL/compiled_dims/num_sms/tc_util) with **save/restore in a
`finally`** (each knob independent); `candidate/scale_pack.cu` unchanged (byte-identical to seed →
packed bytes preserved). Verified: after `run()` the DeepGEMM globals return to defaults (pdl=False,
num_sms=148, tc_util=100) on both shapes; packed scales byte-identical to the seed; adversarial
compliance re-audit PASS (`docs/compliance_review.md`, `docs/codex_r3_compliance_audit.md`).

Three authoritative gates on the submitted `candidate/` (clean reference, knobs restored):

| Run | M=16 µs / HBM% | M=32 µs / HBM% | correct | is_ref_fallback | ref_us |
|---|---|---|---|---|---|
| candidate gate1 | 32.512 / 38.90% | 33.304 / 38.14% | PASS | false | 52.6 / 53.8 |
| candidate gate2 | 32.872 / 38.47% | 33.312 / 38.13% | PASS | false | 52.6 / 53.8 |
| candidate gate3 | 32.849 / 38.49% | 33.249 / 38.20% | PASS | false | 52.6 / 53.8 |

Submitted-candidate per-shape median: **M=16 32.849 µs / 38.49%**, **M=32 33.304 µs / 38.14%** — beats
the pristine seed (33.184 / 34.063) on both shapes but still MISS the 40% ceilings. Reference timing is
now the clean default float32-scale path (~52.6/53.8 µs), confirming the knob-leak fix. Disposition
unchanged: evidence-backed NO-GO on both shapes.
