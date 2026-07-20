# Results — GLM-5.2 o_proj_decode, 40% HBM extreme campaign

**TARGET NOT MET → evidence-backed NO-GO on both shapes.** See `docs/no_go_disposition.md` for the
binding residual analysis. The GEMM already clears 40% when fed prepacked scales; the mandatory
per-call scale pack is a device kernel with a ~2 µs launch floor that exceeds the 0.57–0.89 µs budget.

Authoritative protocol: CUPTI cold-L2 device-kernel median, warmup=3, repeat=10, iterations=30, on
idle B200 GPU 0 (`REMOTE_GPU_ID=0 CUDA_VISIBLE_DEVICES=0`). HBM peak 8.0 TB/s. bytes fixed by the
~100 MB weight stream: M=16 = 101,154,816 B → 40% ceiling 31.611 µs; M=32 = 101,621,760 B → 31.757 µs.

## Attempt ledger

| # | Candidate | M=16 µs / HBM% | M=32 µs / HBM% | Correct | Verdict vs 40% |
|---|---|---|---|---|---|
| A0 | Pristine hbm35 seed (fused CUDA pack + packed DeepGEMM, `compiled_dims=nk`), preserved at `variants/seed/` | 33.184 / 38.10% | 34.063 / 37.29% | PASS (diff 0) | MISS both |
| A1 | Seed + PDL (deep_gemm.set_pdl) | 32.505 / 38.90% | 33.809 / 37.57% | PASS | MISS both |
| A2 | Optimized broadcast CUDA pack (byte-identical) | 38.088 / 33.20% | 43.617 / 29.12% | PASS (byte-exact) | MISS (worse — rejected) |
| **A3** | **SUBMITTED `candidate/`** — best-knob (same byte-identical pack + per-shape PDL/compiled_dims/num_sms/tc_util, saved+restored): M=16 mnk/148/tc50, M=32 mk/74/tc80; 3-gate median | **32.849 / 38.49%** | **33.304 / 38.14%** | PASS (diff 0) | MISS both (best correct candidate) |
| — | Triton weight-pack control | pack_only 2.66 µs | — | — | same ~2 µs floor |
| — | GEMM floor (prepacked oracle, NOT accepted) | 30.720 / 41.16% | 31.184 / 40.73% | — | clears 40% (proves GEMM ok) |
| — | noop kernel floor | 2.07 µs | — | — | irreducible single-kernel floor |

**Submitted candidate: A3 (best-knob) in `candidate/`** — the best correct configuration measured, and
the campaign's submitted implementation. It keeps the seed's fused pack (packed UE8M0 bytes
byte-identical to the seed) and adds per-shape DeepGEMM public launch knobs (saved/restored so nothing
leaks into the harness reference timing). It is faster than the pristine seed on both shapes
(32.849/33.304 vs 33.184/34.063) but still misses 40%, so the disposition is unchanged. The **pristine
hbm35 seed is preserved byte-identical at `variants/seed/`** (A0; provenance in `docs/seed_provenance.md`).

### DeepGEMM launch-knob sweep (t6, `docs/deep_gemm_knob_sweep.md`)
The DeepGEMM public launch knobs were measured in two passes: Round 1 isolated controls
(`compiled_dims`×PDL at defaults; `num_sms` and `tc_util` swept at `nk`), then Round 2 the actual
**cross-product** for the K-compiled `compiled_dims`{`nk`,`k`,`mk`,`mnk`} × PDL{F,T} ×
`num_sms`{74,96,128,148} × `tc_util`{50,80,100} = 96 rows/shape, plus the four non-K `compiled_dims`
measured at their most-favorable knob (measured dominance: +2.36/+2.90 µs above ceiling). All 8
`compiled_dims` correct (calc_diff 0). Over the full **200-row cross-product, no configuration clears
40%** (`ANY_CONFIG_CLEARS_40%=False`). `compiled_dims` K-compile is the main lever (seed's `nk`
near-optimal); PDL a consistent ~0.6 µs; `num_sms` small but interacts with PDL (M=32 best at
num_sms=74, a ~0.47 µs effect the isolated Round-1 sweep missed); `tc_util` negligible (memory-bound).
Best = A3 above. Defaults restored after the sweep.

## Authoritative three-gate confirmation (submitted `candidate/` = best-knob A3)

| Shape | gate1 | gate2 | gate3 | median | HBM% @ median | 40% ceiling | margin |
|---|--:|--:|--:|--:|--:|--:|--:|
| M=16 | 32.512 | 32.872 | 32.849 | **32.849 µs** | **38.49%** | 31.611 | −1.238 µs (miss) |
| M=32 | 33.304 | 33.312 | 33.249 | **33.304 µs** | **38.14%** | 31.757 | −1.547 µs (miss) |

Both shapes: CORRECT, calc_diff 0.0 (pre- and post-timing), `is_reference_fallback=false`, DeepGEMM
globals restored to defaults after each call (reference timing clean at ~52.6/53.8 µs), harness
verdict 2/2 WIN vs the slow float32-scale reference (a reference-relative win, NOT the 40% target).
Raw: `artifacts/final_gates/candidate_bestknob_gate{1,2,3}.log`. Packed scales byte-identical to the
seed (verified). The submitted candidate beats the pristine seed on both shapes but still misses 40%.

Pristine seed (A0, preserved at `variants/seed/`) three-gate median for reference: M=16 33.184 µs /
38.10%, M=32 34.063 µs / 37.29% (`artifacts/final_gates/final_gate{1,2,3}.log`).

## Floor decomposition (the crux)

| M | end2end | GEMM floor | pack_only | residual (pack+gap) | required residual for 40% |
|---|--:|--:|--:|--:|--:|
| 16 | 33.168 | 30.720 (41.16%) | 2.720 | 2.448 | ≤ 0.891 |
| 32 | 34.488 | 31.184 (40.73%) | 2.712 | 3.304 | ≤ 0.573 |

The required residual is below the ~2.07 µs single-kernel floor on both shapes → impossible under the
fixed contract. Full analysis: `docs/floor_table.md`, `docs/no_go_disposition.md`.

## Profiler evidence (AC-2)

- Pack kernel `fused_pack_kernel`: DRAM 0.10–0.12%, Memory 2.4–2.6%, **Waves/SM 0.66**, occupancy
  ~50% → launch/latency-bound, not bandwidth-bound. `artifacts/ncu/pack_M{16,32}.txt`.
- GEMM kernel `sm100_fp8_fp4_gemm_1d1d_impl`: DRAM 49–52%, compute 15–28% → memory-bound near
  achievable; no rewrite justified. `artifacts/ncu/gemm_M{16,32}.txt`, `artifacts/ncu/NCU_REPORT.md`.
- nsys timelines + kernel summaries: `artifacts/nsys/`.

## Layer-swap acceptance (AC-6)

`accept_layer.py` with `o_proj` ← campaign candidate: per-op **1.62×** over the reference backend;
end-to-end layer **+4.89% (M=16)** / **+4.44% (M=32)**. `artifacts/final_gates/accept_layer_M{16,32}.log`.
(Production-class, i.e. the reference-relative win — not the 40% HBM extreme target.)

## Compliance (AC-5)

Adversarial audit PASS: stateless, lossless (byte-identical packing), single weight stream, no
re-quantization, no timer manipulation, compiled path timed. `docs/compliance_review.md`.

## Disposition

Realistic pack-side CUDA / CuTe / Triton headroom measured and exhausted; PDL and reused-scratch
levers evaluated. 40% is unreachable on both shapes under the fixed contract. **NO-GO.**
