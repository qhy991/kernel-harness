# Re-optimization Assessment (op-level, S=32k)

Assessment refreshed 2026-07-29 (moe_total_prefill updated to the round-2 winner).
**Scope: op-level GATE-1 only.** See `repro/REPRODUCE.md` (tracked, self-contained) for
the pinned environment, baseline numbers, landed-winner numbers, and the one-command A/B.

## Win/loss vs the production baseline

| Op | Verdict | Notes |
|---|---|---|
| index_score_prefill | **WIN 2.90×** (3/3, bit-exact) | launch-config override of the aiter `_fp8_mqa_logits` Triton kernel (BLOCK_KV tiling only — never the reduction). |
| moe_total_prefill | **WIN ~1.12×** (3/3; round-2 upgrade, +6.7% over round-1) | bit-exact `BLOCK_SIZE_M` 128→256 token-tile resize + per-M `GROUP_SIZE_M`; M=4096 flipped parity→win. |
| moe_total_decode | **WIN 1.0541×** (2/2, bit-exact) | `BLOCK_SIZE_M` shrink on the gated dense-degenerate path. |
| dsa_prefill_attn | **WIN 1.3266×** (3/3, 0 regress, calc_diff 1.87e-6) | purpose-built native-64 Triton sparse-MLA kernel (half the padded-128 ASM FLOPs; `matrix_instr_nonkdim=16`, bf16 PV). **Supersedes** the earlier PyTorch candidate that regressed vs the ASM baseline. |

## Node environment gap (why the baseline looks the way it does)

This node's aiter source-build dispatches two critical kernels to slow fallbacks:
- sparse-MLA fwd: ~662 ms Triton dev placeholder (production ~1.7 ms) → the baseline
  instead uses the compiled ASM `mla_decode_fwd` (~1.7 ms); see `BASELINE.md`.
- `fp8_mqa_logits`: ~15 ms (production ~4.3 ms) — the strongest available on this node.

Root cause: CK/ASM components for gfx942 were not compiled into the source-build. The
harness **code** is correct (aligned with `origin/amd-reopt-0723`); the gap is purely
runtime/binary. This is why absolute µs are node-specific — the win (conservative
speedup, zero regression) is what reproduces.

## Recommendations (next round)

1. **index_score_prefill** — biggest MFU headroom (still only ~11% MFU despite the 2.90×
   win); push the bit-exact launch-config levers further.
2. **moe_total_prefill** — round-2 landed the M=4096 flip (bit-exact `BLOCK_SIZE_M`
   128→256); remaining bit-exact headroom is thin — next lever is `num_stages` /
   `waves_per_eu` scheduling.
3. **moe_total_decode** — memory-bound; small headroom, dense-degenerate-only (honest).
4. **dsa_prefill_attn** — near a structural cap (~20% MFU vs ASM's ~30% per-useful-FLOP);
   accept a further win only if a genuinely new lever appears, never by reward-hacking.
