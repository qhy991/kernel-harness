# Attempt DAG — glm52 / dsa_attn_decode >40% HBM

Each node = one source mechanism / attempt. Preserve the fastest correct candidate.

```
A0  baseline (stock flash_mla_sparse_fwd)            [CORRECT] 10.92% / 21.27%   ROOT — KEPT
 └─ evidence: NCU grid==M, waves 0.11/0.22, DRAM 15% → occupancy wall
     │
     ├─ A1  split-KV / flash-decoding — raise resident CTAs M → M×splits
     │   ├─ P1  multi-stream Python split (stock kernel, S concurrent launches +
     │   │      base-2 lse merge).  [TRIED — REJECTED, Round 0]
     │   │        correctness: calc_diff 3.77e-6 (< 5e-6 aggregate PASS) BUT
     │   │        elementwise layer FAILS — 35,105 elems, max_abs_err 1.95e-3
     │   │        (abs_tol 2.85e-5), max_rel_err 105x (rel_tol 0.0157). Cause:
     │   │        stock kernel emits BF16 partials; merging them perturbs small
     │   │        outputs ~1 bf16-ulp, which single-pass ref avoids. Inherent to
     │   │        any Python-level split → unfixable without in-kernel fp32 reduce.
     │   │        timing: eager multi-stream span host-gap-dominated (probe ~1.9ms).
     │   │        artifact: attempts/A1_P1_multistream_split.py
     │   ├─ P2  isolated FlashMLA fork: single-launch split-KV, grid=M*splits,
     │   │      in-kernel FP32 reduction before ONE bf16 write. [PRUNED — below floor]
     │   │        Only design that avoids partial DRAM traffic; could plausibly clear
     │   │        M32 but CANNOT beat M16's pure-read gather floor (12.42us ≈ 12.53us
     │   │        ceiling). Not built — no physical slack to justify the fork.
     │   └─ P3  task-local CuTe-DSL single-launch split-KV from scratch. [PRUNED — below floor]
     │            same lower-bound wall as P2; not built.
     └─ A2  split-heads reshape (fold 64-head axis into query axis, free occupancy).
              [TRIED — BLOCKED, Round 0]  artifact: attempts/A2_splitheads.py
                stock head64 kernel dispatches only h_q∈{64,128} (tcgen05 MMA atom
                SM100_MMA<B_H=64>); cannot form smaller head-groups → no free occupancy.
```

## Status — FINAL: NO-GO-CONFIRMED (reviewed)
- A0 ROOT established and profiled (Round 0). CORRECT, calc_diff 0. **Active candidate, kept.**
- A1/P1 REJECTED on correctness (elementwise bf16-partial) — decisive negative result.
- A2 BLOCKED by fixed B_H=64 MMA atom — no free-occupancy path.
- A1/P2 & P3 PRUNED as below-floor: M16 pure-read gather floor (12.42us) ≈ strict-40%
  ceiling (12.53us) leaves no slack for compute+reduce; forks not built. See
  `docs/floor_evidence.md`. Target requires BOTH shapes → M16 is the blocking no-go.

## Ledger
| Node | Mechanism | M16 us | M32 us | Correct | Kept? | Notes |
|------|-----------|-------:|-------:|:------:|:-----:|-------|
| A0 | stock sparse_prefill_fwd | 45.927 | 47.151 | yes | yes (fastest correct) | baseline/oracle; active candidate |
| A1/P1 | multi-stream Python split-KV | n/a | n/a | **no** | no (artifact only) | elementwise fail (bf16 partials); host-gap span |
| A2 | split-heads reshape | n/a | n/a | **no** | no (artifact only) | BLOCKED: fixed B_H=64 MMA atom, no smaller head-groups |
| A1/P2 | FlashMLA fork, in-kernel fp32 reduce | — | — | not built | no | PRUNED below-floor (M16 gather floor ≈ ceiling) |
| A1/P3 | CuTe-DSL split-KV from scratch | — | — | not built | no | PRUNED below-floor (same wall as P2) |
