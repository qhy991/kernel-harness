# No-Go Disposition — GLM-5.2 o_proj_decode, 40% HBM extreme

**Verdict: evidence-backed NO-GO on both shapes (M=16 and M=32).** The 40% HBM target is physically
unreachable under the fixed contract. This is a *physical-limit* no-go, not a correctness or
compliance failure: the preserved candidate is correct (calc_diff 0), stateless, lossless, and
production-class; it simply cannot reach the extreme latency ceiling because the mandatory per-call
scale pack is a device kernel and any device kernel has a ~2 µs floor in the gated metric.

All numbers are the authoritative CUPTI cold-L2 device-kernel median, repeat=10 × rep=30 × warmup=3,
on idle B200 GPU 0. HBM peak 8.0 TB/s. Raw evidence under `artifacts/`.

## The binding bound (DEC-3 residual analysis)

For 40% HBM on shape M: `end2end ≤ ceiling`, and `end2end = GEMM_floor + residual`, where
`residual = pack + (pack→GEMM gap)` and the pack is at least one legal device kernel. So the target
requires `residual ≤ ceiling − GEMM_floor`:

| M | GEMM floor¹ | 40% ceiling | required residual | measured single-kernel floor² | verdict |
|---|---|---|---|---|---|
| 16 | 30.720 µs | 31.611 µs | **≤ 0.891 µs** | ~2.07 µs | impossible: floor > budget |
| 32 | 31.184 µs | 31.757 µs | **≤ 0.573 µs** | ~2.07 µs | impossible: floor > budget |

¹ GEMM floor = `deep_gemm.fp8_gemm_nt` with scales pre-packed OUTSIDE the timed window (non-accepted
oracle; the hard lower bound). It already clears 40% (41.16% / 40.73%) — the GEMM is not the blocker.
² Hand-written trivial `noop` kernel (writes one int32), same timer. A *lower bound* on any legal
separate pack kernel's contribution to the device-kernel span.

**The required residual (0.573–0.891 µs) is below the irreducible single-kernel floor (~2.07 µs) on
both shapes.** `GEMM_floor + irreducible_pack_overhead` = 30.72 + ~2.0 ≈ 32.7 µs (M=16) and
31.18 + ~2.0 ≈ 33.2 µs (M=32), both above their 40% ceilings. QED.

## Why the pack cannot be removed or hidden (the three exits are all closed)

1. **Cannot cache / precompute the pack** — forbidden (stateless, exact packing every timed call).
2. **Cannot use a compact weight-scale layout to shrink the write** — the ~0.77 MB per-row expansion
   is API-forced by DeepGEMM's SM100 TMA scale descriptor (`shape_mn = N`); and anyway the pack cost
   is launch-floor, not the 0.1 µs write, so shrinking the write does not help.
3. **Cannot fuse the pack into the GEMM** — that is a GEMM rewrite, out of scope
   (`BL-20260718-dont-handroll-fp8-gemm`), and unjustified since the GEMM is already memory-bound
   near its achievable bandwidth (NCU DRAM 49–52%, compute 15–28%).
4. **Cannot overlap the pack away** — pack and GEMM are data-dependent (the GEMM reads the packed
   scales), so overlap is bounded by the GEMM prologue (~0.27 µs measured), far short of the ~1.5 µs
   needed.

## What was measured (exhaustive pack-side sweep, per the campaign mandate)

| approach | pack_only | end2end M=16 | end2end M=32 | note |
|---|---|---|---|---|
| Seed (fused CUDA, 1568×128) | 2.72 µs | 33.17 µs / 38.1% | 34.42–34.49 µs / 36.9% | best correct candidate |
| Seed + PDL (DEC-2) | — | 32.51 µs / 38.9% | 33.81 µs / 37.6% | PDL lever alone |
| **Best-knob** (PDL + per-shape cross-product optima), run.sh 3-gate median | — | **32.584 µs / 38.8%** | **33.329 µs / 38.1%** | best achievable; still misses |
| Optimized broadcast CUDA pack (byte-identical) | 7.2–12.1 µs | 38.1 µs | 43.6 µs | worse: less latency hiding |
| Triton weight-pack control | 2.66 µs | — | — | same ~2 µs floor as CUDA |
| Hand-written noop kernel | 2.07 µs | — | — | irreducible single-kernel floor |
| GEMM floor (prepacked oracle, non-accepted) | — | 30.72 µs / 41.2% | 31.18 µs / 40.7% | clears 40% — proves GEMM ok |

- **Full DeepGEMM launch-knob cross-product** (`docs/deep_gemm_knob_sweep.md`,
  `artifacts/probes/deep_gemm_knob_crossproduct.log`): the K-compiled `compiled_dims`{nk,k,mk,mnk} ×
  PDL{F,T} × num_sms{74,96,128,148} × tc_util{50,80,100} = 96 rows/shape (all correct, calc_diff 0),
  plus the four non-K `compiled_dims` measured at their most-favorable knob (measured dominance:
  +2.36 µs M=16 / +2.90 µs M=32 above ceiling). Over all **200 rows, no combination clears 40%**
  (`ANY_CONFIG_CLEARS_40%=False`). `compiled_dims` K-compile is the main lever (seed's `nk`
  near-optimal); PDL a consistent ~0.6 µs; num_sms interacts with PDL at M=32 (best num_sms=74);
  tc_util negligible. The knobs tune the GEMM/launch, not the pack, so the ~2 µs pack floor is
  untouched.
- **PDL** (the only lever that touches the span) recovers ~0.6 µs by removing the pack→GEMM gap, but
  the best gated config still leaves M=16 short by ~0.97 µs and M=32 short by ~1.57 µs.
- Reused scratch (DEC-1) is a **no-op** in this metric: host-side allocation is outside the
  device-kernel span (confirmed by the timer's definition and NCU showing no memset).
- The 42% stretch is even further out (needs e2e ≤ 30.106 µs < the 30.720 µs GEMM floor).

## Best correct candidate (submitted)

`candidate/candidate.py` + `candidate/scale_pack.cu` — the best-knob configuration (seed's fused pack,
packed UE8M0 bytes byte-identical to the seed per `docs/seed_provenance.md`, plus per-shape DeepGEMM
public launch knobs saved/restored around the GEMM). Authoritative three-gate medians: **M=16
32.849 µs / 38.49%**, **M=32 33.304 µs / 38.14%**; correct on both shapes (calc_diff 0, pre- and
post-timing); `is_reference_fallback=false`; DeepGEMM globals restored to defaults after each call
(clean reference ~52.6/53.8 µs); adversarial compliance audit PASS (`docs/compliance_review.md`);
layer-swap acceptance 1.62× per-op, +4.9%/+4.4% end-to-end (`artifacts/final_gates/accept_layer_*`).
It beats the pristine seed (33.184/34.063, preserved at `variants/seed/`) on both shapes but still
misses the 40% ceilings — consistent with the no-go.

## Conclusion

The GEMM already sits at ~41% / ~40.7% HBM when fed prepacked scales; the entire shortfall to the 40%
*gate* is the mandatory per-call scale pack, whose cost is a ~2 µs kernel-launch floor rather than its
~0.1 µs of bandwidth. Under the fixed rules (stateless per-call packing, no GEMM rewrite, no caching),
no legal candidate can bring both shapes to 40%. **No-go, both shapes** — with M=32 the harder miss.
The realistic pack-side CUDA/CuTe/Triton headroom has been measured and exhausted.
