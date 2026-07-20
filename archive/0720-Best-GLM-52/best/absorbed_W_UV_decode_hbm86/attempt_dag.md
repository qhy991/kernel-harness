# Attempt DAG / Ledger — glm52 absorbed_W_UV_decode >86% HBM

One mechanism per attempt. All spans are CUPTI cold-L2 median-of-medians
(repeat=10, iters=30) at boost on idle B200 GPU 3, unless noted NCU (isolated,
L2-flushed replay — absolute NCU µs run higher than the gate; use for %/ratios).
Ceilings: M16 < 1.371684 µs, M32 < 1.524093 µs.

```
A0 baseline (seed = sgl_kernel.bmm_fp8, cuBLAS nvjet)          [ROOT, correct]
   gate: M16 5.28 µs / 22.3% · M32 5.47 µs / 24.0% · calc_diff 0
   NCU:  nvjet_sm100_qqtst_128xM · DRAM 16% · 128 blk · 0.86 waves/SM
   → occupancy/latency-bound, not BW-bound
   │
   ├─ A1 backend sweep (flashinfer.bmm_fp8 backend=cublas)     [correct]
   │     gate: M16 5.12 µs / 23.0% · M32 5.24 µs / 25.0% · calc_diff 0
   │     → fastest CORRECT candidate found; ~3% over seed (within noise, same
   │       nvjet kernel). Still 3.7× above ceiling. PRESERVED as best candidate.
   │
   ├─ A2 backend=cudnn                                         [layout: calc_diff 1.0]
   │     gate span ~5.2/5.3 µs (numerically wrong on frozen row-major B; span
   │     representative). No speed advantage.
   │
   └─ A3 swap-AB CUTLASS SM100 (backend=cutlass)  ← Codex's decisive test
         kernel: device_kernel<GemmUniversal<MainloopSm100TmaUmmaWarpSpecialized,
                 SM100_MMA_F8F6F4_SS, tile 64x64x128, TMA load/store, warp-spec>>
         NCU: M16 8.03–9.09 µs, DRAM 13.76%, 256 blk, 1.73 waves/SM
              M32 8.67–9.09 µs, DRAM 13.95%, 256 blk, 1.73 waves/SM
         needs column-major B (transpose) on the frozen layout →
         gate span WITH transpose in loop = 40 µs (extra copy kernel).
         → RESULT: raised occupancy 0.86→1.73 waves but DRAM FELL to ~14% and
           latency did NOT improve. Occupancy headroom does NOT convert to
           bandwidth for this tiny cold working set. Falsification FAILED to
           disprove the no-go — it strengthened it.
```

## Falsification verdict (Codex criterion: ">2 µs at gate ⇒ no-go confirmed")

Best correct FP8 BMM across cuBLAS/cuDNN/CUTLASS backends = **5.12 µs (M16)**,
3.7× above the 1.372 µs ceiling. The strongest recommended mechanism (swap-AB
SM100 CUTLASS TMA warp-specialized) is not faster and sits at ~14% DRAM. Result
lands far above Codex's 2 µs falsification threshold → **NO-GO confirmed**.

## Why no further one-mechanism iterations (task7 justified stop, not a skip)

The A3 CUTLASS kernel already combines the strongest mechanisms at once — TMA
bulk loads, warp specialization, swap-AB operand ordering, shape-specialized
64×64×128 tiles, and 2× the reference occupancy. It still only reaches ~14%
DRAM. The bottleneck is therefore NOT scheduling/occupancy/tiling (persistent
64-head scheduling, alternative tiles, epilogue tweaks all target those) but the
two physical walls: the ~1.3–2 µs single-kernel CUPTI span floor (W1) and the
~25–63% achievable BW for a ~10 MB cold-L2 working set (W2). No one-mechanism
source edit can move a 14–16% DRAM, latency-bound tiny-batch GEMM to the ~86%
effective HBM the target needs (a 3.85× leap into the launch-floor band). Further
iteration would burn budget without changing the verdict.
