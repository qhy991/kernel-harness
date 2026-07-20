# Results — glm52 absorbed_W_UV_decode >86% HBM (RLCR round 0)

## Verdict: NO-GO-CONFIRMED (first-class outcome per AC-6)

Strictly >86% HBM on both shapes is **unreachable** for this frozen row-major
layout, cold-L2 CUPTI-span protocol, idle boosted B200, exact byte accounting,
and the single-op FP8-BMM implementation space tested (cuBLAS nvjet, cuDNN,
SM100 CUTLASS TMA warp-specialized swap-AB). Independently reviewed by Codex
(gpt-5.5:xhigh) twice: initial NO-GO (task3) and NO-GO-CONFIRMED after the
falsification test (task8).

## Targets vs measured (three authoritative idle-GPU gates, GPU 3)

| M | bytes_hbm | 86% ceiling | measured median (3 gates) | HBM util | gap to target |
|---:|---:|---:|---:|---:|---:|
| 16 | 9,437,184 | < 1.371684 µs | **5.28 µs** | **22.34%** | 3.85× |
| 32 | 10,485,760 | < 1.524093 µs | **5.47 µs** | **23.95%** | 3.59× |

Three-gate per-shape spread ≤ 0.4%; all shapes calc_diff 0.00e+00, correctness
PASS pre- and post-timing; verdict neutral (best correct candidate == reference
call), Harness performance_gate NOT MET (no win — expected; the target is HBM%,
not a Harness win, and neither is met).

## Floor evidence (AC-3, BL-20260719-single-kernel-span-floor)

| measurement (idle B200 @ boost, gate protocol) | M16 | M32 |
|---|---:|---:|
| 86% span ceiling | < 1.372 µs | < 1.524 µs |
| empty single-kernel span floor (noop) | 1.28–1.98 µs (global min 1.28) | same |
| pure streaming read of bytes_hbm (no compute) | 4.64 µs / 25.4% | 5.06 µs / 25.9% |
| simple-kernel 256 MB asymptotic BW | 63.2% | 63.2% |
| reference bmm_fp8 (NCU DRAM %) | 15.87% | 16.45% |

## Best correct candidate (preserved, AC-6)

`candidate/candidate.py` = frozen seed (`sgl_kernel.bmm_fp8`), SHA
`d4fccc4a…`, correct on both shapes, ~5.28/5.47 µs (~22–24% HBM). A backend
sweep found `flashinfer.bmm_fp8(backend="cublas")` marginally faster (5.12/5.24
µs) but it is the same cuBLAS nvjet kernel within measurement noise and does not
change the verdict; the canonical seed is retained as the runnable best correct
candidate.

## Source attempt (AC-4) — swap-AB CUTLASS falsification

The strongest recommended mechanism (SM100 CUTLASS TMA warp-specialized swap-AB
FP8 GEMM) was run and NCU-profiled: it raised occupancy 0.86→1.73 waves/SM but
achieved DRAM *fell* to ~14% and it was not faster than cuBLAS — occupancy
headroom does not convert to bandwidth for this ~10 MB cold working set. See
`docs/attempt_dag.md`.

## Why NO-GO (three walls, all above the ceiling)

1. **W1 (decisive):** single-kernel CUPTI span floor ~1.3–2 µs ≈/above the
   1.372/1.524 µs ceilings; a legal candidate does strictly more than a noop.
2. **W2:** a ~10 MB cold-L2 working set streams at only ~25% (measured) to ≤63%
   (asymptotic) of 8 TB/s; 86% needs 6880 GB/s, above even the large-transfer
   asymptote.
3. **W3:** the reference is within ~13% of a no-compute streaming read; the
   swap-AB CUTLASS kernel confirmed occupancy headroom does not raise BW.

## Artifacts
- `docs/run_log.md`, `docs/floor_decomposition.md`, `docs/attempt_dag.md`
- Floor scripts `floor/*.py`; logs `docs/floor/*.log`, clock traces `docs/floor/*clock*.csv`
- NCU: `docs/ncu/ncu_ref_M{16,32}_nvtx.log`, `docs/ncu/ncu_cutlass_M{16,32}.log`, backend sweep `docs/ncu/bench_backends.log`
- Three-gate logs: `docs/gates/gate_{1,2,3}.log`
- Codex reviews: `.humanize/skill/2026-07-19_16-52-51-*/output.md` (task3), `.humanize/skill/2026-07-19_17-20-28-*/output.md` (task8)
