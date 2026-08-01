# router_jit_gemm

Bounded decode candidate for `router_gemm_topk_decode`.

- M <= 16: SGLang JIT `dsv3_router_gemm` with BF16 inputs and FP32 output.
- M > 16: exact production FP32 `F.linear` fallback.
- Consumer: unchanged production `moe_fused_gate` sigmoid/correction/top-k.

The Harness gates both normalized top-k weights and integer expert IDs. A
projection approximation that changes routing is invalid even if it is faster.

On B200, the complete default-protocol gate
`20260801T053943Z-4007d5` passed all correctness checks: M=16 was 2.721x median /
2.712x conservative, while the exact M=32 fallback was neutral. This result is
provisional until repeated from a clean worktree and confirmed in serving.
