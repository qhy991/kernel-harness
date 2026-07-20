# 可改源码获益的 Task 分析（相对 llm_flops stock）

基于 2026-07-20 DROP-IN 对比与既有 NO-GO 证据；本轮已实测四条实验。

## 已验证有效（继续/已落地）

| Task | 机制 | 层贡献 |
|------|------|--------|
| q_b_decode | **DeepGEMM fork fused UE8M0 pack** (`41c6235`) | Decode 大 |
| **fused_qkv_a_decode** | **同 fork `fp8_gemm_nt_fused`** → CUPTI **~1.69×** | Decode 中 |
| o_proj / moe_*_decode | task-local fused CUDA scale-pack + packed `fp8_gemm_nt` | Decode 大 |
| index_q_upproj_decode | task-local Triton TMA split-K | Decode 中 |
| **index_k_proj_decode** | Triton pack（移植 prefill）→ CUPTI **~2.67×** | Decode 中 |
| index_k_prefill | Triton pack + `disable_ue8m0_cast` | Prefill 中 |
| **moe_gate_prefill** | decode pack+PDL 移植 → CUPTI **~1.12×**；CUDA Graph 大 M **回退** → **不默认 swap** | Prefill 小（CUPTI only） |
| fused_qkv_a / q_b / index_q_upproj **prefill**（PR#3） | pack / Triton / knobs | Prefill 中小算子 |

## 本轮实验结果（CUPTI cold-L2，vs stock f32-scale）

| Experiment | Status | geomean speedup | 备注 |
|------------|--------|-----------------|------|
| `experiments/fused_qkv_a_decode_deepgemm_fused` | **WIN** | **1.69×** | 已晋升 `best/fused_qkv_a_decode` |
| `experiments/fused_qkv_a_decode_pack` | **FAIL** | — | N=2624、`N%128=64`，packed layout assert |
| `experiments/index_k_decode_pack` | **WIN** | **2.67×** | 已晋升 `best/index_k_proj_decode` |
| `experiments/moe_gate_prefill_pack` | **WIN (CUPTI)** | **1.12×** | Graph M4096 ~0.95× → 不默认 drop-in swap |

## 不建议再冲原硬目标（可改源码也难）

| Task | 原因 |
|------|------|
| index_score ≥82% / dsa ≥40% / absorbed_W_UV >86% | 物理 floor / NCU 已贴顶 |
| index_q >15% / o_proj >40% | 已证据型 NO-GO |
| Prefill DSA / index_score 层主导 | 需 FlashMLA / MQA **内核级**大改，非本次 GEMM fork 范围 |
| fused_qkv_a task-local pack | N 非 128 对齐；必须用 fork fused 或定制 pack |

## DeepGEMM fork 适用面

- **高适配**：skinny-M decode GEMM（q_b / fused_qkv_a 已证；尤其 `N%128≠0` 时 fused 优于外置 pack）
- **中适配**：prefill GEMM / MoE（transform 占比更低，收益变薄；moe_gate ~1.06–1.19×）
- **低适配**：bmm_fp8、FlashMLA、paged MQA（不同代码路径）
