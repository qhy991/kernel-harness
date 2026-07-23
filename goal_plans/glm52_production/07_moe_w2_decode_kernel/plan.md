# Goal: tune the production packed W2 grouped GEMM

Read `../COMMON_RULES.md` first and treat it as part of this goal.

## Outcome

Optimize the device kernel used by production W2 decode for the real EP8 local
expert geometry, while preserving the full DeepEP overlap API and using stock
SGLang for any bucket that does not win.

## Production target

- `moe_w2_grouped_decode_m16`: E=32, slab=1024, expected M=4, K=2048, N=6144.
- `moe_w2_grouped_decode_m32`: E=32, slab=1024, expected M=8, K=2048, N=6144.
- Entry: `grouped_gemm_nt_f8f8bf16_masked` with production packed scales.
- Full validation: eight-rank low-latency DeepEP dispatch and combine plus W13 and
  SwiGLU+quant.

## Work

- [ ] Record real `masked_m`, recipe, SM allocation, overlap arguments, selected
  DeepGEMM config, and kernel code/cache identity for both buckets.
- [ ] Establish paired compute baselines and full-region baselines with replacement
  dispatch disabled.
- [ ] NCU the grouped GEMM for tensor activity, weight traffic, CTA waves, mask/tail
  waste, TMA stalls, eligible warps, register pressure, spills, and epilogue stores.
- [ ] Form an evidence-backed variant portfolio.  Candidate source changes may
  alter small-M tile shape, expert scheduling, N tiling, CLC/persistent scheduling,
  TMA pipeline, epilogue, or SM reservation.  Keep each DeepGEMM or CuTe source
  build isolated and reproducible.
- [ ] Inspect ptxas and SASS when a change targets registers, scheduling, vector
  width, or tcgen05/TMEM instructions.  Reject variants whose apparent gain comes
  from omitted masked rows, recipes, or overlap work.
- [ ] Implement a static M16/M32 oracle.  It must preserve the overlap-enabled
  return contract; unsupported recipe or topology combinations fail closed.
- [ ] Re-run correctness, graph replay, eight-rank full-region, and SGLang e2e
  validation before promotion.

## Completion

Complete under the shared production-win or no-replacement definition.  A win
against the frozen f32-scale E=8 baseline is not production evidence.

## Deliverables

Provide input/config capture, source and build SHAs, NCU plus ptxas/SASS evidence,
attempt ledger, paired per-bucket results, overlap-contract tests, full MoE-region
result, end-to-end result, and enable/fallback table.

