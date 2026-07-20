# Results — index_q_upproj_decode >15% HBM

## Contract

`out[M,4096] = x_fp8[M,2048] @ w_fp8[4096,2048].T`, per-token f32 block scales
(`x_scale[M,16]`), per-block weight scales (`w_scale[32,16]`), bf16 output.
Reward is memory-bound `bw_util = bytes_hbm / (latency · 8e12)`, so the strict
>15% target is exactly `latency < bytes_hbm / 1.2e12`:

| M | bytes_hbm | strict ceiling | 18% stretch |
|---|-----------|----------------|-------------|
| 16 | 8,555,520 | `<7.129600 us` | `<5.941333 us` |
| 32 | 8,720,384 | `<7.266987 us` | `<6.055822 us` |

## Headline (Round 1 — TMA kernel, GPU0, repeat=10 ×3 authoritative)

| M | baseline us / % | **candidate us / %** | speedup | target | status |
|---|-----------------|----------------------|---------|--------|--------|
| 16 | 25.4 / 4.14% | **7.33 / 14.59%** | 3.10x | <7.13us | WIN, target NOT met (1.03x) |
| 32 | 26.5 / 4.11% | **8.44 / 12.90%** | 3.08x | <7.27us | WIN, target NOT met (1.16x) |

Medians of three idle-GPU gates (spread ≤1.015). Correctness `calc_diff`
5.3e-10 / 2.6e-10 (≤ 5e-6), survives post-timing poisoned-buffer recheck.
**Verdict: strong stable WIN, strict >15% is a reviewed no-go** (esp. M32).

Round-0 acc-based split-K was 8.12/13.17% and 9.74/11.19%; the TMA kernel lifts
the binding M32 from 11.19% → 12.90% and M16 from 13.17% → 14.59%.

## The kernel (`candidate/candidate.py`)

Single-launch **TMA-streamed split-K** FP8 GEMM (N-major transposed output):
- W[N,K] is row-major, so a TMA tile `[BLOCK_N, BLOCK_K]` keeps K as the
  contiguous inner dim (a TMA requirement). Each CTA computes
  `acc[BN,BM] = dot(w_tile, trans(x_tile)) = out[m,n]^T`, scales per K-group,
  stores transposed. x is tiny (M≤32) → plain masked load.
- Weight tiles load through `TensorDescriptor`/`load_tensor_descriptor` (Triton
  3.6 TMA): bulk async copies via copy engine + mbarrier, so 2 warps keep many
  W loads in flight and drive a deep software pipeline.
- `SPLIT_K=2`: each CTA gets an 8-group K-loop (deep enough to fill the TMA
  pipeline) while doubling CTA count for occupancy; the last CTA per N-tile
  reduces the two L2-resident FP32 partials in place and re-arms the reused
  per-N-tile semaphore — one launch, one reduction, no per-call memset.
- Committed config `BLOCK_N=32, SPLIT_K=2, num_warps=2`, per-shape pipeline depth
  (M16→6, M32→4). All env-overridable. `candidate_acc/` keeps the Round-0
  acc-based kernel for rollback.

## Why we plateau at ~13–14.6%, not 15% — the bottleneck (NCU-grounded)

The op only streams the 8.39 MB weight once (SOL ≈ 1.05us at 8 TB/s), so 15%
(7.13us) is 7× SOL and *looks* easy. Two measured walls stop it:

**1. It was latency-bound; TMA fixed that.** NCU on the acc-based kernel: W read
exactly once, DRAM ~10%, warps stalling on `long_scoreboard` (global-load
latency) at low occupancy. TMA async loads cut that stall 6.6 → 1.2.

**2. Now it is occupancy-bound, and occupancy cannot rise.** NCU on the TMA
kernel: **0.20 eligible warps/cycle**, achieved occupancy 5% (theoretical max 25%,
capped by the deep-pipeline shared memory), 0.22 waves — the machine is ~78% idle.
Raising occupancy needs more CTAs, but the only knobs make it worse:

| lever to add CTAs | effect | measured |
|-------------------|--------|----------|
| SPLIT_K 2→4 (512 CTAs) | K-loop 8→4 (shallower pipe) + 2× partial/reduction | ~12% (worse) |
| BLOCK_N 32→16 (512 CTAs) | 2 KB TMA tiles + tiny [16,·] MMA underutilize | ~12% (worse) |
| BLOCK_K 128→256 (bigger TMA) | needs mid-tile tensor slice | blocked: Triton API |

**3. The physical read floor is below the target for tiled access.** Measured
memory ceilings for this 8.39 MB cold-L2 read:

| Experiment | time | %peak | takeaway |
|-----------|------|-------|----------|
| flat 1D 128-bit read (`flat_sol.py`) | 5.10us | 20.5% | absolute read ceiling — needs ~592 CTAs; cannot feed an MMA |
| 2D fp8 tiled read, no MMA (`stream_sol.py`) | 7.42us | 14.1% | **already < both 15% targets**; the floor for any tiled-access GEMM |
| **TMA split-K GEMM M16** | **7.33us** | **14.59%** | beats the naive 2D floor (TMA > `tl.load`); +MMA/scale/output over flat read |
| **TMA split-K GEMM M32** | **8.44us** | **12.90%** | + BM=32 doubles acc regs → shallower pipe → lower occ |

A tiled-access GEMM must read W in 2D tiles (to feed the MMA), and that read alone
floors at 7.42us = 14.1% — **under the 15% target** — on this GPU/measurement at
these sizes. Only the flat 1D streaming pattern (5.10us) clears it, and it cannot
do a GEMM. The 2.2–3.3us the GEMM adds over the flat read (MMA, block scales,
transposed store, split-K reduction, and sub-25% occupancy) is irreducible enough
that both shapes land below 15%.

## Why not a raw CUDA/CuTe SM100 rewrite or a DeepGEMM fork

- A hand-written **CuTe SM100** kernel would use the *same* TMA + tcgen05 MMA
  hardware the Triton kernel already drives. The wall is structural (tiny M forces
  split-K for CTAs; splitting K shortens the pipeline), not a Triton codegen gap,
  and the 2D-tiled read floor (7.42us > target) is language-independent. Expected
  headroom does not cross 15% on M32; ROI judged low against the measured floor.
- **DeepGEMM fork** [D]: stock `deep_gemm` is the same launch-bound family
  (~4% stock, ~1.6x-over-ref packed ⇒ ~6.8%); its kernels target large M and
  offer no tiny-M memory-bound path. Dominated by the TMA candidate. Parked.

## Floors / comparisons

- **Reference** (`deep_gemm.fp8_gemm_nt`, f32 scales): 25–26us, ~4.1% — launch/
  occupancy bound at skinny M.
- **acc-based split-K** (Round 0, `candidate_acc/`): M16 13.2%, M32 11.2–11.7%.
- **dot_scaled MXFP8**: ~2x slower (M16 7.6%, M32 5.9%). Rejected.
- **TMA split-K** (committed): M16 14.59%, M32 12.90%. Fastest correct.

## Reviewed no-go (AC-7)

Strict >15% on both shapes is **not reachable** for this op on B200 GPU0 under the
frozen contract/measurement. Evidence: (a) three stable authoritative gates at
14.59% / 12.90%; (b) NCU showing the kernel is occupancy-bound at 0.20 eligible
warps/cycle after TMA removed the latency stall, with every CTA-count lever
measured net-negative; (c) the pure 2D-tiled W read floor (7.42us) is already
below the 15% latency ceiling, so no tiled-access GEMM can clear it. M16 is within
2.8% of the wall; M32 is 16% short due to double the accumulator register pressure.
Fastest correct candidate preserved. Attempt DAG in `docs/attempt_dag.md`.
