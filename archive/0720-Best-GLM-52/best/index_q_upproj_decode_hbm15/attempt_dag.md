# Attempt DAG — index_q_upproj_decode >15% HBM

Each node: mechanism → measured worst-shape HBM% (M16/M32) → verdict. One
mechanism change per node. Times are CUPTI cold-L2 device-kernel median, GPU0.

```
[baseline] deep_gemm.fp8_gemm_nt (f32 block scales)
    M16 25.4us/4.14%  M32 26.5us/4.11%
    verdict: launch/occupancy bound at skinny M. Reference/oracle — frozen.
    │
    ├─► [A] Packed DeepGEMM (int32 ue8m0, disable_ue8m0_cast)
    │       fp8_gemm_nt routes to fp8_fp4_gemm_nt → scale-layout assert in this build.
    │       Ceiling reasoning: same kernel family, ~1.6x over ref ⇒ ~6.8% HBM.
    │       verdict: DOMINATED by [B*]; not a path to 15%. Parked (AC-4 floor).
    │
    └─► [B] Fused split-K Triton (N-tiled, per-N-tile reused semaphore, FP32 partials)
            single launch, single reduction, no per-call memset.
            │
            ├─ BLOCK_N=128 SPLIT_K=1  → 4.6%/4.2% (23us)  serial K-loop starves MLP. REJECT.
            ├─ BLOCK_N=128 SPLIT_K=2  → 7.6%/7.1%         too few CTAs (64). REJECT.
            ├─ BLOCK_N=128 SPLIT_K=4  → 11.0%/9.7%        128 CTAs. baseline of the family.
            ├─ BLOCK_N=128 SPLIT_K=8  → 12.3%/8.5%        M16 up, M32 down (2× partial). 
            ├─ BLOCK_N=128 SPLIT_K=16 → 10.7%/7.1%        unstable (spread 1.23x). REJECT.
            ├─ BLOCK_N=64  SPLIT_K=4  → 11.1%/10.7% (w4)  best M32 of the SK=4 row.
            ├─ BLOCK_N=32  SPLIT_K=4  → 11.6%/10.4%       
            ├─ num_stages 2/3/4       → ~no effect (short K-loop).
            └─ [B*] BLOCK_N=64 SPLIT_K=8, num_warps per-shape (M16→8, M32→4), s3
                    **M16 8.12us/13.17%  M32 9.74us/11.19%**  (repeat=10)
                    verdict: Round-0 best. WIN 3.18x/2.72x. calc_diff 7.6e-10/3.8e-10.
                    │
                    ├─ [B**] BLOCK_N=32 SPLIT_K=8 num_warps=2 (NCU-motivated occ)
                    │        M32 9.33us/11.69% (small-CTA occupancy). Preserved candidate_acc/.
                    │
                    └─► [E] dot_scaled native MXFP8 MMA (proto_scaled/2.py)
                            UE8M0 byte-extract exact (diff 0-3.7e-11) but ~2x SLOWER
                            (M16 7.6%, M32 5.9%): tcgen05 MXFP8 unsuited to skinny M.
                            verdict: REJECT.

[NCU on B*] (task6): W read ONCE, DRAM ~10%, long_scoreboard 6.6/11.1 at 20-40% occ.
    ⇒ LATENCY bound (not bandwidth); SPLIT_K=8 K-loop=2 iters can't pipeline.
    │
    └─► [C] TMA-streamed split-K (Triton 3.6 TensorDescriptor)  ★ COMMITTED ★
            N-major transposed output; W tile [BN,BK] via load_tensor_descriptor;
            bulk async loads + deep pipeline. (stride bug fixed via proto_tma_debug.)
            ├─ SPLIT_K=1 (128 CTAs)        → ~11-12%  too few CTAs.
            ├─ SPLIT_K=4/8 (512+ CTAs)     → ~12%     K-loop too short, reduction 2-4x.
            ├─ BLOCK_N=16 (512 CTAs)       → ~12%     2KB TMA tiles + tiny MMA.
            ├─ BLOCK_K=256 (bigger TMA)    → blocked: Triton mid-tile slice unsupported.
            └─ [C*] BLOCK_N=32 SPLIT_K=2 num_warps=2, per-shape stages (M16→6,M32→4)
                    **M16 7.33us/14.59%  M32 8.44us/12.90%**  (repeat=10 ×3, spread≤1.015)
                    verdict: WIN 3.10x/3.08x. FASTEST CORRECT. Target >15% NOT met.

[NCU on C*] (task7): long_scoreboard 6.6→1.2 (fixed!). Now OCCUPANCY bound:
    0.20 eligible warps/cyc, occ 5% (max 25%), 0.22 waves. Every CTA-count lever
    net-negative ⇒ structural wall at these sizes.

[SOL probes] (memory ceiling for this pattern; refreshed R1.6)
    flat 1D 128-bit read of W (8.39MB)  → 5.10us / 20.5% peak (needs ~592 CTAs)
    2D fp8 tiled read (no MMA)          → 7.42us / 14.1% peak  ⇒ ALREADY < 15% target
    ⇒ any tiled-access GEMM's W read floors below the target; only the flat 1D
      pattern (can't feed an MMA) clears it. 15% is a reviewed no-go, esp. M32.
```

## Branch status (Round 1 resolved)

- **[C] task-local TMA W-streaming GEMM** — DONE, committed (`[C*]` above). The
  TMA + async multi-stage pipeline mechanism (the core of the AC-5 CUDA/CuTe idea)
  was realized in Triton 3.6 `TensorDescriptor`. It reads W more efficiently than
  the naive 2D `tl.load` (M16 7.33us beats the 7.42us naive-tiled floor) but is
  occupancy-bound at 5%; it does not reach the 5us flat-read regime because a GEMM
  cannot use the flat 1D pattern. A raw CuTe SM100 rewrite would drive the *same*
  TMA/tcgen05 hardware into the *same* structural wall (tiny M ⇒ split-K ⇒ short
  pipeline; 2D-tiled read floor 7.42us > target) — not pursued: ROI below the
  measured floor.
- **[D] isolated DeepGEMM-GLM52 fork** — PARKED. Same launch-bound kernel family
  (stock ~4%, packed ~6.8%), no tiny-M memory-bound path; dominated by `[C*]`.

**Conclusion:** reviewed no-go for strict >15% on both shapes (M16 14.59% is 1.03x
short, M32 12.90% is 1.16x short). Fastest correct candidate = `[C*]` TMA kernel,
preserved in `candidate/`; Round-0 acc kernel in `candidate_acc/`. See results.md.
