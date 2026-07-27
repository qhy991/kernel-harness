# Serving bake-off: archive-0720 vs Codex

Generated: `2026-07-24T08:38:27Z`

> **解读与 e2e/上线结论**：见上级汇总 [`../TEST_RESULTS_0724.md`](../TEST_RESULTS_0724.md)。  
> 本文件只保留本轮 `serving_native` eager A/B 原始表；**勿把 q_b ~1.5× 当成可上线收益**。

Protocol: `serving_native` interleaved paired A/B (eager CUDA events); warmup=5, repeat=40.

Speedup = stock_ref_p50 / candidate_p50 (>1 is faster). Gate: paired p50 ≥ 1.03×.

## Runnable results

| id | op | M | variant | paired p50 | gate | correct | error |
|---|---|---:|---|---:|---|---|---|
| `indexer_wq_b_m16_0720` | indexer_wq_b | 16 | 0720 | 1.184852 | True | pass |  |
| `indexer_wq_b_m16_codex_packed` | indexer_wq_b | 16 | codex | 1.102777 | True | pass |  |
| `indexer_wq_b_m32_0720` | indexer_wq_b | 32 | 0720 | 1.095693 | True | pass |  |
| `indexer_wq_b_m32_codex_packed` | indexer_wq_b | 32 | codex | 1.159271 | True | pass |  |
| `fused_qkv_a_m16_0720` | fused_qkv_a | 16 | 0720 | 1.065979 | True | pass |  |
| `fused_qkv_a_m32_0720` | fused_qkv_a | 32 | 0720 | 1.049809 | True | pass |  |
| `q_b_m16_codex` | q_b | 16 | codex | 1.507153 | True | pass |  |
| `q_b_m32_codex` | q_b | 32 | codex | 1.348818 | True | pass |  |
| `moe_w13_m16_codex` | moe_w13 | 16 | codex | 1.053089 | True | pass |  |
| `moe_w13_m32_codex` | moe_w13 | 32 | codex | 1.067575 | True | pass |  |
| `moe_w2_m16_codex` | moe_w2 | 16 | codex | 1.069584 | True | pass |  |
| `moe_w2_m32_codex` | moe_w2 | 32 | codex | 1.035428 | True | pass |  |

## Non-run annotations (0720 under production ABI)

| id | op | status | notes |
|---|---|---|---|
| `indexer_wq_b_m16_codex_sms` | indexer_wq_b | `requires_goal_runtime` | Needs runtime.indexer_fused_prepare_store (goal-15 runner only) |
| `indexer_wq_b_m32_codex_sms` | indexer_wq_b | `requires_goal_runtime` | Needs runtime.indexer_fused_prepare_store (goal-15 runner only) |
| `q_b_m16_0720` | q_b | `absorbed_in_stock` | 0720 win was fp8_gemm_nt_fused f32→UE8M0 pack; production already packed |
| `q_b_m32_0720` | q_b | `absorbed_in_stock` | 0720 win was fp8_gemm_nt_fused f32→UE8M0 pack; production already packed |
| `moe_w13_m16_0720` | moe_w13 | `absorbed_in_stock` | 0720 had separate moe_gate/up + pack; production is fused W13 packed |
| `moe_w13_m32_0720` | moe_w13 | `absorbed_in_stock` | 0720 had separate moe_gate/up + pack; production is fused W13 packed |
| `moe_w2_m16_0720` | moe_w2 | `absorbed_in_stock` | 0720 moe_down pack already in production packed path |
| `moe_w2_m32_0720` | moe_w2 | `absorbed_in_stock` | 0720 moe_down pack already in production packed path |
| `dsa_m16_0720` | dsa | `already_stock` | hechenxi dsa_decode_attn is flashinfer trtllm-gen; same as serving stock |
| `dsa_m32_0720` | dsa | `already_stock` | hechenxi dsa_decode_attn is flashinfer trtllm-gen; same as serving stock |
| `o_proj_m16_0720` | o_proj | `absorbed_in_stock` | 0720 win was scale_pack.cu + packed gemm; production already packed |

Machine-readable: [`results/bakeoff_summary.csv`](results/bakeoff_summary.csv).
