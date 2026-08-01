# GLM-5.2 fusion task integration

Date: 2026-08-01

## Inventory decision

The four original campaign tasks under
`/home/qinhaiyan/glm52-fusion-goal-runs/worktrees/` were not present in the main
`/home/qinhaiyan/Kernel-Harness` task registry. They are now additive tasks; the
existing 24 leaf tasks remain intact. A later B300 nsys audit added eight more
phase-specific fusion tasks, so the current suite contains 36 task directories.

| task | phase | production seam | required outputs |
|---|---|---|---|
| `norm_quant_qkv_decode` | decode | input RMSNorm/quant/QKV-A + DSA indexer | `out`, `residual`, `normed_bf16` |
| `norm_quant_qkv_prefill` | prefill | input RMSNorm/quant/QKV-A + DSA indexer | `out`, `residual`, `normed_bf16` |
| `norm_quant_gate_decode` | decode | post-attention RMSNorm/quant/MoE gate | `out`, `residual` |
| `norm_quant_gate_prefill` | prefill | post-attention RMSNorm/quant/MoE gate | `out`, `residual` |

All four are region tasks: the denominator is the production-order sequence
`fused_add_rmsnorm -> packed-UE8M0 group quant -> fp8_gemm_nt`, with the weight
scale transformed once at input construction just as production does at model
load time. This avoids the earlier artificial per-call scale repack.

## Why QKV differs from gate

The original campaign gated only `(out, residual)` for both seams. That contract
is valid for the post-attention gate path, where no consumer needs the normalized
BF16 activation. It is incomplete for GLM-5.2 QKV: the DSA indexer also consumes
the normalized BF16 activation. The integrated QKV tasks therefore require a
third output and count its mandatory BF16 write in the byte model.

Consequently, the historical two-output QKV prefill speedups are optimistic and
must be remeasured. They are not imported as wins. A candidate that merely returns
the original hidden tensor also fails numerical correctness.

## Preserved campaign evidence

- `norm_quant_gate_prefill`: `gate_v1_fusednq` is saved under the task's
  `variants/` directory. The campaign reproduced it on two B200s across all three
  prefill shapes, but checkpoint-backed TP8/DP8/EP8 and end-to-end serving remain
  outstanding, so it stays non-default.
- `norm_quant_qkv_prefill`: prior positive measurements used the incomplete
  two-output contract; status is **remeasure required** under the corrected ABI.
- `norm_quant_qkv_decode`: corrected packed-scale campaign evidence did not reach
  its 1.15 target (1.1048x / 1.1052x); no replacement is installed.
- `norm_quant_gate_decode`: corrected packed-scale campaign evidence did not reach
  its 1.15 target (1.0257x / 1.1049x); no replacement is installed.

The original reconciliation remains at
`/home/qinhaiyan/glm52-fusion-goal-runs/RECONCILIATION.md`; the production seam
analysis remains at
`/home/qinhaiyan/glm52-fusion-goal-runs/SGLANG_INTEGRATION_FINDINGS.md`.

## Initial four-task verification

Verification on `verda-b200x4` after integration:

- `selftest`: 28 tasks, 0 problems.
- generated mirror check: all 28 task directories in sync.
- inventory: 28 tasks across 6 families, including 4 fusion tasks.
- `accept_layer.py` remains an explicit 12-leaf budget and rejects fusion
  regions, preventing their consuming GEMM from being double-counted.
- full harness verifier: passed, including compile, knowledge, generated-task,
  serving-native, diff-hygiene, run-audit, and pointer-audit lanes.
- all four new default candidates passed initial and unseen-seed post-timing
  correctness probes with `calc_diff=0`.
- `gate_v1_fusednq` passed the complete schema-1.4 Gate-prefill gate: 3/3 WIN,
  0 regressed, conservative paired speedups 1.157x / 1.148x / 1.180x for
  M=1024/2048/4096. Result:
  `runs/glm52/norm_quant_gate_prefill/20260801T032655Z-bdb223/result.json`.
- the same legacy two-output kernel was deliberately tested against corrected
  QKV-prefill and rejected with exit 2: `fusion run(inputs) must return exactly
  (out, residual, normed_bf16)`. Result:
  `runs/glm52/norm_quant_qkv_prefill/20260801T032151Z-3f96fb/result.json`.

The positive run is `PROVISIONAL`, not `OFFICIAL`, because the repository and
candidate were uncommitted during integration. Its numerical and performance
fields are internally consistent, but production promotion still requires a
clean rerun plus checkpoint-backed distributed/end-to-end acceptance.

## B300-derived additive regions

The 2026-08-01 audit added four more production regions, each split into prefill
and decode without removing or replacing any prior task:

| region | production boundary | context/output contract |
|---|---|---|
| `indexer_q_rope_quant` | Q RoPE + FP8 quant + head-gate scale | existing KV S=65536; returns Q and scaled head gate |
| `indexer_k_norm_rope_store` | K LayerNorm + RoPE + cache quant/store | existing KV S=65536; writes assigned page-size-64 cache rows |
| `moe_swiglu_quant` | masked SwiGLU + W2-input quant | returns masked FP8 activations and packed UE8M0 scales |
| `router_gemm_topk` | router GEMM + sigmoid/correction/top-k | returns weights and exact top-k IDs |

The first three are already fused in production and are included so an optimizer
must beat the fused denominator rather than a decomposed proxy. Router GEMM and
top-k remain two kernels in the B300 trace and are the open fusion boundary. Full
evidence, deferred candidates, and B200 measurements are recorded in
[`GLM52_B300_FUSION_AUDIT_20260801.md`](GLM52_B300_FUSION_AUDIT_20260801.md).
