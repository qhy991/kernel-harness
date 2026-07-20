# Run Log — index_q_upproj_decode >15% HBM

All measurements on **NVIDIA B200 GPU 0** (`CUDA_VISIBLE_DEVICES=0`), idle
(0 MiB / 0% at run time), venv `/home/qinhaiyan/Kernel-Harness/.venv`
(torch 2.11.0+cu130, triton 3.6.0, deep_gemm present). Timing = harness CUPTI
cold-L2 device-kernel median. Seed SHA (frozen):
`11edbebb6641d8abfc776785ba52dba68aef0f9d67b81f8db7dc5cf774fd2ff6`.

Strict ceilings (equality misses): M16 `<7.129600 us`, M32 `<7.266987 us`.

## Round 0

### R0.1 — Baseline (AC-1)
Reference seed (`deep_gemm.fp8_gemm_nt`, f32 block scales) via
`evaluate_task.py --repeat 10`:

| M | ref us | HBM % | timing_spread |
|---|--------|-------|---------------|
| 16 | 25.38 | 4.21% | 1.010 |
| 32 | 26.20 | 4.16% | 1.004 |

Stable (spread << 1.25). The earlier prompt.md probe (27.9/25.7us, 3.1–5.7x
spread) was measurement noise on a contended GPU; GPU0 idle is clean. Baseline
confirmed again on every subsequent candidate run (harness times the reference
alongside the candidate): ref stayed 25–27us / ~4.1%.

### R0.2 — Fused split-K Triton, first correct compile (AC-2, AC-3)
`candidate/candidate.py`: transferred the `index_k_proj_decode` mechanism
(single launch, FP32 partials, last-CTA reduction, reused semaphore) generalized
to N=4096 with **N-tiling** and a **per-N-tile** semaphore vector. First bug: a
module global `_GROUP` referenced inside the `@triton.jit` body — Triton forbids
non-constexpr globals; replaced with a literal. Then correct:
repeat=1 probe → M16 10.03us/10.66% (3.4x), M32 11.38us/9.58% (5.3x),
calc_diff ~1e-10.

### R0.3 — Tuning sweep (AC-3, task4)
`scripts/sweep.py` (same CUPTI primitive as the gate). Findings:
- **num_stages barely matters** (2/3/4 ~equal): K-loop is short.
- **SPLIT_K is the main lever**, but it trades occupancy against FP32-partial
  traffic. SPLIT_K=1 is *worst* (~5%/20us): the serial K-loop's dependent
  accumulation starves memory-level parallelism. Split-K helps by breaking that
  dependency chain, not only by adding CTAs.
- **num_warps=4** beats 8 for M=32; **8** slightly better for M=16.
- Best single config **BLOCK_N=64, SPLIT_K=8** (worst-shape 10.99%); with
  per-shape num_warps (M16→8, M32→4) M16 reaches 13.06%.

### R0.4 — Bottleneck isolation
- B200 has **148 SMs, 132.6 MB L2** → the split-K FP32 partial (≤16 MB) is
  L2-resident, so the reduction is *not* the HBM cost. The cost is the W read.
- `scripts/stream_sol.py` (2D fp8 tiled read, no MMA): floor ~7.4us.
- `scripts/flat_sol.py` (flat 128-bit/int32 read): floor **5.08us = 20.6% peak**.
- ⇒ 15% (7.13us) is reachable; the gap is the **fp8 tiled-access + scale-in-loop
  pattern**, addressable with source-level TMA/async-pipelined 128-bit loads.

### R0.5 — Packed DeepGEMM (AC-4)
`get_mn_major_tma_aligned_packed_ue8m0_tensor` + `fp8_gemm_nt(..,
disable_ue8m0_cast=True)` routes to `fp8_fp4_gemm_nt` and asserts on scale layout
in this build. Not pursued further: the packed path is the same launch/occupancy
-bound kernel family (frozen problem statement: production ≈1.6x over ref ⇒
~6.8% HBM), **already dominated** by the split-K candidate (13.2%/11.2%).

### R0.6 — Authoritative gate on committed candidate
`evaluate_task.py --repeat 10` (persisted):

| M | cand us | HBM % | speedup | sp_cons | calc_diff | verdict |
|---|---------|-------|---------|---------|-----------|---------|
| 16 | 8.12 | 13.17% | 3.18x | 3.09x | 7.6e-10 | WIN |
| 32 | 9.74 | 11.19% | 2.72x | 2.64x | 3.8e-10 | WIN |

Harness gate MET (2 wins, 0 regress). Strict >15% target **not yet met**
(need <7.13 / <7.27us).

## Round 1 — source mechanism (task6/task7)

### R1.1 — NCU the committed acc-based split-K candidate (task6)
`ncu -k regex:fused_splitk --launch-skip 3 --launch-count 1`, GPU0, cold-L2:

| shape | dram_read | dram % | mem SOL | SM SOL | occ | waves | long_scoreboard |
|-------|-----------|--------|---------|--------|-----|-------|-----------------|
| M32   | 8.47 MB (once) | 9.57% | 18.9% | 9.7% | 19.8% | 0.86 | **6.60** |
| M16   | 8.43 MB (once) | 11.6% | 20.1% | 12.9% | 39.4% | 0.69 | **11.08** |

Overturned the "W-read bandwidth" hypothesis: W is read **exactly once** (no
amplification), DRAM only ~10%, and warps stall on **long_scoreboard**
(global-load latency) at low occupancy. It is **latency bound**, not bandwidth
bound. At SPLIT_K=8 each CTA's K-loop is only 16/8=2 iterations, so `num_stages`
cannot pipeline — matching R0.3's "stages don't matter."

### R1.2 — dot_scaled native MXFP8 MMA (task7a) — REJECT
`scripts/proto_scaled.py` (single-tile), `proto_scaled2.py` (split-K). Scales are
genuine UE8M0 (`_dg_*_cast(use_ue8m0=True)`), so byte-extract + `repeat_interleave`
to E8M0/group32 is bit-exact (diff 0–3.7e-11). But `tl.dot_scaled` is **~2x
slower**: best M16 14us/7.6%, M32 18.4us/5.9%. The tcgen05 MXFP8 path does not
suit skinny M and the 4x-replicated scale operands add traffic. Triton's fused-MMA
lever is exhausted.

### R1.3 — occupancy sweep on acc kernel (NCU-motivated)
`scripts/sweep.py`. M32 is grid-starved (512 CTAs, 20% occ) vs M16 (1024, 39%).
`num_warps=2` (smaller CTAs → more resident/SM) lifts M32: **BN=32 SK=8 w2 s3 →
M32 9.33us/11.69%** (was 9.69us/11.25%), M16 unchanged. Preserved in `candidate_acc/`.

### R1.4 — TMA-streamed split-K (task7b, source mechanism [C]) — WIN
`scripts/proto_tma.py`: Triton 3.6 `TensorDescriptor`/`load_tensor_descriptor`.
N-major transposed output keeps W's contiguous K as the TMA inner dim; x (tiny)
uses a plain load + `tl.trans`. TMA issues bulk async W loads via the copy engine
+ mbarrier, so 2 warps keep many loads in flight and drive a deep pipeline.
(First cut had a stride bug — the x load reused the *scale* strides; fixed after
`proto_tma_debug.py` proved the TMA+transpose math correct at diff 7.6e-10.)
Sweep winner **BN=32 SPLIT_K=2 num_warps=2** (8-group K-loop = deep pipeline, 256
CTAs), per-shape stage depth (M16→6, M32→4): M16 ~14.6%, M32 ~13.0%.

### R1.5 — NCU the TMA kernel — bottleneck moved to occupancy
Same probe on `_tma_sk`:

| shape | dram % | occ | waves | long_scoreboard | eligible warps/cyc | issue-active |
|-------|--------|-----|-------|-----------------|--------------------|--------------|
| M32   | 10.2%  | 5.4% | 0.22 | **1.15** (was 6.60) | **0.20** | 9.9% |
| M16   | 12.1%  | 5.3% | 0.16 | 1.47 | — | — |

TMA **eliminated the long_scoreboard stall** (6.6→1.2). The kernel is now
**occupancy/issue-bound**: 0.20 eligible warps/cycle, theoretical-max occupancy
only 25% (deep-pipeline smem), achieved 5%. More warps would help but adding CTAs
needs split-K (shortens pipeline + adds reduction) or tiny BN (MMA-inefficient) —
both measured net-negative (SK=4 → ~12%, BN=16 → ~12%). Structural wall.

### R1.6 — SOL floors refreshed (physical ceiling)
- `flat_sol.py` (1D vectorized 128-bit read of W, 8.39 MB): best **5.10us / 20.5%**
  — but needs ~592 CTAs; at 148 CTAs only 9–15%. The absolute read ceiling.
- `stream_sol.py` (2D fp8 tiled read, no MMA): best **7.42us / 14.1%** (BN=16,
  256 CTAs). **Already below both 15% targets** (7.13/7.27us). Any tiled-access
  GEMM's W read is floored here before adding MMA/scale/output. Only the flat 1D
  pattern beats it, and that cannot feed an MMA. (TMA reads more efficiently than
  the naive 2D `tl.load`, so the GEMM's M16 7.33us actually beats this naive floor.)

### R1.7 — Three authoritative gates on promoted TMA candidate
`candidate/` = TMA kernel. `evaluate_task.py --repeat 10`, GPU0 idle, ×3:

| gate | M16 us / % | M32 us / % |
|------|------------|------------|
| 1 | 7.392 / 14.47% | 8.512 / 12.81% |
| 2 | 7.320 / 14.61% | 8.432 / 12.93% |
| 3 | 7.328 / 14.59% | 8.416 / 12.95% |

Stable (spread ≤1.015), correct (calc_diff 5.3e-10 / 2.6e-10, post-timing recheck
OK), WIN 3.1x both shapes. **Strict >15% NOT met**: M16 median 7.33us (1.028x
short), M32 median 8.45us (1.16x short). Reviewed no-go — see results.md.

