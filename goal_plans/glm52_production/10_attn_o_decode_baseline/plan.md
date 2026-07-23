# Goal: establish and beat the real attention O-projection decode baseline

Read `../COMMON_RULES.md` first and treat it as part of this goal.

## Outcome

Determine whether attention O projection has a deployable replacement at M16 or
M32 when compared with stock production packed DeepGEMM.  This goal is designed to
expose negative optimization instead of comparing against the slower f32-scale
synthetic reference.

## Production target

| Bucket | Workload | Shape |
|---|---|---|
| M16 | `linear_attn_o_decode_m16` | `[16,16384] x [6144,16384]^T` |
| M32 | `linear_attn_o_decode_m32` | `[32,16384] x [6144,16384]^T` |

The callable is production `w8a8_block_fp8_matmul_deepgemm` with packed int32
UE8M0 scales.  The reference process must have `SGLANG_GLM52_OPT=0`.

## Work

- [ ] Trace the SGLang `o_proj` call for both graph buckets and record the exact
  DeepGEMM kernel/config selected by stock production.
- [ ] Run three paired serving-native baselines.  Separately reproduce the related
  synthetic task only to quantify the baseline mismatch.
- [ ] Measure every candidate component: dispatch, allocation, scale/layout work,
  GEMM, and output conversion.  Reject any candidate whose apparent synthetic win
  is consumed by production adapter or launch cost.
- [ ] Profile stock and strongest candidate with nsys; use NCU if the candidate
  changes the device kernel.  Explain whether weight streaming, launch latency,
  waves, or library dispatch is binding.
- [ ] Implement the smallest production-native candidate that can plausibly win.
  It must accept packed scales directly and return the same output tensor contract.
- [ ] Build an M16/M32 oracle and keep stock SGLang for any bucket below the paired
  3% gate.  Confirm CUDA Graph replay, the attention layer, and end-to-end decode.

## Completion

Complete with a safely enabled winning bucket or an evidence-backed decision to
replace neither bucket.  It is valid and preferable to leave both stock when the
candidate is merely the same DeepGEMM path plus extra dispatch overhead.

## Deliverables

Provide production-versus-synthetic baseline table, path trace, component timing,
profiler evidence, source diff, correctness, paired bucket results, graph/layer/e2e
results, and explicit fallback policy.

