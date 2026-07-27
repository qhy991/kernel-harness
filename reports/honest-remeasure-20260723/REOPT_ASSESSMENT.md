# Re-optimization Assessment (op-level, S=32k)

Assessment refreshed 2026-07-27. **Scope: op-level GATE-1 only.** See `BASELINE.md` for
the baseline numbers and the landed-winner run_ids.

## Win/loss vs the production baseline

| Op | Verdict | Notes |
|---|---|---|
| index_score_prefill | **WIN 2.90×** (3/3, bit-exact) | launch-config override of the aiter `_fp8_mqa_logits` Triton kernel (BLOCK_KV tiling only — never the reduction). |
| moe_total_prefill | **WIN ~1.04–1.05×** (2/3; M=4096 borderline) | bit-exact `GROUP_SIZE_M` tiling; M=4096 sits at parity — the next-round headroom target. |
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
2. **moe_total_prefill** — move M=4096 off parity with bit-exact scheduling knobs.
3. **moe_total_decode** — memory-bound; small headroom, dense-degenerate-only (honest).
4. **dsa_prefill_attn** — near a structural cap (~20% MFU vs ASM's ~30% per-useful-FLOP);
   accept a further win only if a genuinely new lever appears, never by reward-hacking.
