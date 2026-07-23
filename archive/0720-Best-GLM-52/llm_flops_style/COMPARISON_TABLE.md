# GLM-5.2 单算子延时对比（llm_flops DROP-IN）

协议：**CUDA Graph**（warmup=5, runs=20），stock 与 candidate **同一冻结 tensor**。  
单位：ms。speedup = stock / ours（>1 为加速）。

Decode swaps（PR#5 bake-off 后）：
- **PR5** `best-hechenxi-0720/`：`fused_qkv_a`、`index_q_upproj`、`dsa_decode_attn`
- **ours** `best/`：`q_b`（DeepGEMM fork）、`o_proj`、`index_k`、`moe_*`
- Prefill：**与 decode 共用** `o_proj`、`index_k`、`moe_up`、`moe_down`（pack+PDL）；其余仍 PR#3 prefill 专用

复现：`bench_{decode,prefill}.py` · 对打：[`PR5_VS_OURS.md`](PR5_VS_OURS.md)

---

## 层总览

| Phase | M | stock 层 (ms) | swapped 层 (ms) | 层加速 |
|-------|--:|-------------:|----------------:|-------:|
| Decode（PR5+ours） | 16 | 0.3279 | 0.1883 | **1.74×** |
| Decode（PR5+ours） | 32 | 0.3487 | 0.2256 | **1.55×** |
| Prefill (+PR#3+shared decode) | 1024 | 1.1416 | 1.1071 | **1.031×** |
| Prefill (+PR#3+shared decode) | 2048 | 3.7152 | 3.6186 | **1.027×** |
| Prefill (+PR#3+shared decode) | 4096 | 4.7069 | 4.4098 | **1.067×** |

复测时间：`20260720T085336Z` · 协议 CUDA Graph drop-in（warmup=5, runs=20）

† Prefill M=2048 stock 曾受 DSA 抖动影响；Decode 相对上一版（1.51×/1.44×）再抬，主要来自 PR5 `dsa` + 更快的 `fused_qkv_a`/`index_q`。

---

## Decode — 关键算子（★ = PR5）

| op | M | stock (ms) | ours (ms) | speedup | source |
|----|--:|-----------:|----------:|--------:|--------|
| ★ fused_qkv_a_proj | 16 | 0.0231 | 0.0068 | **3.38×** | `best-hechenxi-0720/fused_qkv_a_decode` |
| ★ fused_qkv_a_proj | 32 | 0.0228 | 0.0092 | **2.48×** | `best-hechenxi-0720/fused_qkv_a_decode` |
| q_b_proj | 16 | ~0.024 | ~0.007 | **~3.4×** | `best/q_b_decode` |
| q_b_proj | 32 | ~0.024 | ~0.007 | **~3.3×** | `best/q_b_decode` |
| ★ dsa_decode_attn | 16 | 0.0393 | 0.0184 | **2.14×** | `best-hechenxi-0720/dsa_decode_attn` |
| ★ dsa_decode_attn | 32 | 0.0397 | 0.0249 | **1.60×** | `best-hechenxi-0720/dsa_decode_attn` |
| ★ index_q_upproj | 16 | 0.0165 | 0.0048 | **3.43×** | `best-hechenxi-0720/index_q_upproj_decode` |
| ★ index_q_upproj | 32 | 0.0167 | 0.0063 | **2.63×** | `best-hechenxi-0720/index_q_upproj_decode` |
| index_k_proj | 16/32 | — | — | **~2.4–2.5×** | `best/index_k_proj_decode` |
| o_proj | 16/32 | — | — | **~1.6–2.0×** | `best/o_proj_decode_hbm35` |
| moe_gate/up/down | 16/32 | — | — | **~1.5×** | `best/moe_*_decode_hbm40` |

---

## Prefill

**decode/prefill 共用 swap**（`best/` pack+PDL）：

| op | M1024 | M2048 | M4096 | archive |
|----|------:|------:|------:|--------|
| o_proj | **1.13×** (0.077/0.087 ms) | **1.07×** | **1.27×** | `best/o_proj_decode_hbm35` |
| index_k_proj | **1.15×** | **1.14×** | **1.16×** | `best/index_k_proj_decode` |
| moe_up_proj | **1.09×** | **1.03×** | 0.95× | `best/moe_up_proj_decode_hbm40` |
| moe_down_proj | **1.07×** | **1.02×** | 0.92× | `best/moe_down_proj_decode_hbm40` |

层加速 **1.03–1.07×**（M1024/4096）；M2048 stock 受 DSA/index_score 抖动影响偏大。

原始 CSV（`20260720T085336Z` 复测）：
- `results/comparison_all.csv` — decode+prefill 全量
- `results/comparison_shared_decode_prefill.csv` — decode/prefill 共用 4 算子
- `results/glm5_{decode,prefill}_swapped_perf.csv` — 原始 bench 输出
