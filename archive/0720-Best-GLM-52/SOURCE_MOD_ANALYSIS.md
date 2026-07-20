# 可改源码获益的 Task 分析（相对 llm_flops stock）

基于 2026-07-20 DROP-IN 对比与既有 NO-GO 证据。

## 已验证有效（继续/已落地）

| Task | 机制 | 层贡献 |
|------|------|--------|
| q_b_decode | **DeepGEMM fork fused UE8M0 pack** (`41c6235`) | Decode 大 |
| o_proj / moe_*_decode | task-local fused CUDA scale-pack + packed `fp8_gemm_nt` | Decode 大 |
| index_q_upproj_decode | task-local Triton TMA split-K | Decode 中 |
| index_k_prefill | Triton pack + `disable_ue8m0_cast` | Prefill 中 |
| fused_qkv_a / q_b / index_q_upproj **prefill**（PR#3） | pack / Triton / knobs | Prefill 中小算子 |

## 值得再试源码/ fork（本轮尝试）

| Task | 为何值得 | 路径 |
|------|----------|------|
| **fused_qkv_a_decode** | Decode 仍 stock ~0.023 ms；与 o_proj/q_b 同为 f32-scale DeepGEMM | UE8M0 pack；可选 DeepGEMM `fp8_gemm_nt_fused` |
| **index_k_decode** | Decode stock；prefill 已有 ~1.18× pack 路径 | 移植 `index_k_prefill_bw70` |
| **moe_gate_prefill** | Prefill 仍 stock；decode moe pack 已 ~1.5× | 移植 moe_gate_decode pack+PDL |
| prefill o_proj / moe_up/down | 层上几乎无收益；可选 fork fused 再探 | 低优先 |

## 不建议再冲原硬目标（可改源码也难）

| Task | 原因 |
|------|------|
| index_score ≥82% / dsa ≥40% / absorbed_W_UV >86% | 物理 floor / NCU 已贴顶 |
| index_q >15% / o_proj >40% | 已证据型 NO-GO |
| Prefill DSA / index_score 层主导 | 需 FlashMLA / MQA **内核级**大改，非本次 GEMM fork 范围 |

## DeepGEMM fork 适用面

- **高适配**：skinny-M decode GEMM（q_b 已证；fused_qkv_a / o_proj 同类）
- **中适配**：prefill GEMM（transform 占比更低，收益变薄）
- **低适配**：bmm_fp8、FlashMLA、paged MQA（不同代码路径）
