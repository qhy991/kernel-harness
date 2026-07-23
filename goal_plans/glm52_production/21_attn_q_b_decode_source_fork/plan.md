# Goal: source-optimize DeepGEMM for native-packed attention Q-B decode

Read `../COMMON_RULES.md` first and treat it as part of this goal.

## Outcome

Develop a reproducible isolated DeepGEMM variant for attention Q-B decode that
accepts production packed UE8M0 inputs and beats stock SGLang at M16 and/or M32.
The reference library must remain untouched and importable in the same comparison.

## Production target

- `linear_attn_q_b_decode_m16`: M16, N16384, K2048.
- `linear_attn_q_b_decode_m32`: M32, N16384, K2048.
- BF16 output, packed int32 UE8M0 scales, SM100, CUDA Graph decode.
- Integration tag: attention `q_b_proj`, never indexer `wq_b`.

## Isolation contract

Use a pinned source checkout, unique build artifact and JIT cache per commit, and an
explicit loader/manifest.  Record the actual module and shared objects resolved by
reference and candidate.  Never overwrite the stock environment or silently let
both paths import the same build.

## Work

- [ ] Establish three alternating paired stock baselines and capture selected
  configs for both buckets.
- [ ] Make the fork consume packed scales natively.  Remove f32-only entry
  assumptions, online unpacking, and temporary layout conversions before judging
  kernel tuning.
- [ ] NCU the stock and fork kernels.  Quantify launch floor, N waves, weight/L2/HBM
  traffic, TMA scale loads, tensor issue, SM count, registers, spills, and epilogue.
- [ ] Create one source variant per measured hypothesis: tile/cluster, N split,
  persistent scheduling, SM clamp, pipeline stage, scale path, or epilogue.  Record
  expected PTX/SASS change and inspect the emitted code.
- [ ] Maintain a per-bucket oracle table and static dispatch.  No runtime autotune,
  host synchronization, or device-to-host shape read is allowed.
- [ ] Integrate only winning variants into SGLang with fail-closed fallback.  Verify
  hit counters, M-specific policy, graph replay, attention layer, and end-to-end
  decode.

## Completion

Complete with a production win for at least one bucket or a no-replacement result
that names the fork's binding limit and retains stock SGLang.  Beating the f32-scale
synthetic reference or reaching a historical HBM percentage alone is insufficient.

## Deliverables

Provide fork and build SHAs, loader/manifest, source diff, variant and rejection
ledger, NCU/ptxas/SASS evidence, paired M16/M32 table, correctness, graph/layer/e2e
results, enable/fallback policy, and rollback steps.

