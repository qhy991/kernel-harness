# Goal: optimize native-packed attention Q-B decode

Read `../COMMON_RULES.md` first and treat it as part of this goal.

## Outcome

Make the attention `q_b_proj` candidate consume production packed int32 UE8M0
scales directly and improve stock SGLang at local M16 and/or M32.  Remove the
current need to skip production inputs or unpack them for an old f32-scale fork.

## Production target

| Bucket | Workload | Shape |
|---|---|---|
| M16 | `linear_attn_q_b_decode_m16` | M16, N16384, K2048 |
| M32 | `linear_attn_q_b_decode_m32` | M32, N16384, K2048 |

The stock callable is production packed `w8a8_block_fp8_matmul_deepgemm`.
Attention Q-B uses all 64 heads on each DP-attention rank and is distinct from
indexer `wq_b`.

## Work

- [ ] Prove both graph buckets hit attention `q_b_proj` and record stock DeepGEMM
  config, packed scale shapes, output allocation, and stream behavior.
- [ ] Baseline stock production and quantify the gap between it and the related
  synthetic f32-scale task.
- [ ] Refactor or replace the experimental kernel so its ABI is packed-native.
  No per-call unpack, temporary f32 scales, or extra conversion kernel is allowed
  in the deployable path.
- [ ] Profile stock and candidate for weight bandwidth, launch floor, TMA/scale
  handling, CTA waves, SM count, registers, and epilogue.  Use an isolated
  DeepGEMM source overlay when the best lever is inside the library.
- [ ] Maintain a static M16/M32 oracle.  It is acceptable and required to enable
  only one bucket when the other is neutral or slower.
- [ ] Verify hit/miss counters, CUDA Graph replay, attention layer timing, and full
  SGLang decode.  Ensure disabled buckets immediately call stock production.

## Completion

Complete with at least one production bucket meeting the shared win gate or an
evidence-backed no-replacement result.  Enabling the f32 adapter merely to obtain a
hit is a failure even if the synthetic task is faster.

## Deliverables

Provide ABI comparison, source/build diff, paired M16/M32 table, profiler evidence,
hit/fallback proof, graph/layer/e2e results, and exact environment policy such as
an M16-only enable when appropriate.

