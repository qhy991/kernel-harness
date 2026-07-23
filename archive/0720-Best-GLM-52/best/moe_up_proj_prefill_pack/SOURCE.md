# moe_up_proj_prefill_pack

- Origin: port of `best/moe_up_proj_decode_hbm40` (fused CUDA UE8M0 pack + PDL). The fused
  pack kernel is shape-agnostic, so the port is the seed itself (only the loaded extension
  name changes `_decode`→`_prefill`); `scale_pack.cu` is byte-identical to the seed.
- CUPTI cold-L2 (2026-07-20, B200, harness gate): geomean ~1.115× (M1024/2048/4096:
  **1.184 / 1.105 / 1.058×**). 3/3 shapes WIN, 0 regress, correct (calc_diff 0.0), exit 0.
- CUDA Graph drop-in (interleaved 30-trial median): M1024 **1.165×**, M2048 **1.085×**,
  M4096 **1.049×** (97% of trials ≥1.00×) → **Graph-safe on all shapes**, UNLIKE the
  `moe_gate_proj_prefill_pack` analog which regressed to ~0.95× at M4096.
- AC2 note: M=4096 CUPTI (1.058×) is below the campaign's 1.10× per-shape bar. This is a
  **physical wall under the allowed levers**, not a missing optimization: the gate-faithful
  packed-GEMM-only CUPTI floor (candidate latency if the pack were free) is **1.084× at
  M=4096** (< 1.10×), because at large M the reference's internal f32→UE8M0 scale cast — the
  only thing the pack removes — is a shrinking fraction (~7.7%) of the compute-bound (66% MFU)
  GEMM. Reaching 1.10× would need a faster grouped GEMM = Forbidden. PDL on/off/auto has no
  material gate effect at M=4096 (1.058/1.055/1.055×).
- **Not default-swapped** in layer benches: `PREFILL_SWAPS['moe_up_proj']` remains the flat
  PDL-only `moe_up_proj_prefill.py`. This candidate is Graph-safe and eligible, but promotion
  is a deliberate layer-owner decision (no Graph-blind / CUPTI-only promotion policy).
- Full evidence + reproduce: KDA-Pilot-Exp worktree
  `llm/glm52_harness_moe_up_proj_prefill_pack/docs/results.md` and
  `.humanize/rlcr/2026-07-20_06-13-14/evidence/` (floor_cupti_v2.py, round2_*.txt).
