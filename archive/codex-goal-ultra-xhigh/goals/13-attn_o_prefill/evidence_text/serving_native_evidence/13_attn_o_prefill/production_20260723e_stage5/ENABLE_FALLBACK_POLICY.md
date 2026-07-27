# Enable and fallback policy

## Final state

No new attention O-projection prefill kernel is enabled.

- SGLang source is at the rollback commit `8f450dbdf`.
- The DeepGEMM heuristic source is byte-identical to the pre-experiment
  `f93f8867b` state.
- The five-stage overlay is not imported or selected by the production
  registry. It is retained only for replaying the evidence bundle.
- The installed `deep_gemm` package was never overwritten.

## Fail-closed behavior

The normal launch has `SGLANG_GLM52_OPT=0`; it uses
`deepgemm_w8a8_block_fp8_linear_with_fallback` and the stock six-stage
`deep_gemm.fp8_gemm_nt` kernel.

Even if `SGLANG_GLM52_OPT=1` is set alone, the default profile is
`serving_safe`. With no `SGLANG_GLM52_OPT_OPS` allowlist and no explicit
`full` profile, `lookup("o_proj", "prefill", m=4096)` returns `None`, so the
same stock path runs. `validation/fallback_policy_tests.txt` proves both
default-disabled and OPT=1/default-profile cases.

The pre-existing compiled-NK dispatcher can still be requested deliberately
with all of the following:

```text
SGLANG_GLM52_OPT=1
SGLANG_GLM52_OPT_PROFILE=full
SGLANG_GLM52_OPT_OPS=o_proj
SGLANG_GLM52_OPT_M_BUCKETS=o_proj:4096
```

That is an explicit ablation interface, not a promotion from this goal. Its
three leaf and three containing-region series did not pass the 3% gate. It is
not recommended for the default serving configuration.

Unsupported shapes, dtypes, scale layouts, unknown M buckets, missing overlay
capabilities, or registry misses continue through the existing stock SGLang
fallback. No device-to-host read, synchronization-based dispatch, scale
adapter, or hard-coded physical GPU was added.

## Promotion policy

Reconsider an alternative only after it independently passes:

1. exact packed-ABI correctness;
2. at least three same-GPU alternating series with paired p50 speedup of 1.03
   or greater;
3. a non-regressing `Fp8LinearMethod.apply` containing region;
4. CUDA graph/replay semantics where production uses them;
5. the unmodified eight-rank TP8/DP8 GLM-5.2 FP8 attention-layer and complete
   SGLang prefill gates.

Until then, stock is the only enabled path for the named production bucket.
