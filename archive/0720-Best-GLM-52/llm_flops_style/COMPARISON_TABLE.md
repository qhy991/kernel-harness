# GLM-5.2 单算子延时对比（llm_flops DROP-IN）

协议：**CUDA Graph**（warmup=5, runs=20），stock 与 candidate **同一冻结 tensor**。  
单位：ms。speedup = stock / ours（>1 为加速）。

Decode swaps（PR#5 bake-off 后）：
- **PR5** `best-hechenxi-0720/`：`fused_qkv_a`、`index_q_upproj`、`dsa_decode_attn`
- **ours** `best/`：`q_b`（DeepGEMM fork）、`o_proj`、`index_k`、`moe_*`
- Prefill：PR#3 + `best/index_k` / `best/o_proj_prefill`（未改）

复现：`bench_{decode,prefill}.py` · 对打：[`PR5_VS_OURS.md`](PR5_VS_OURS.md)

---

## 层总览

| Phase | M | stock 层 (ms) | swapped 层 (ms) | 层加速 |
|-------|--:|-------------:|----------------:|-------:|
| Decode（PR5+ours） | 16 | 0.3227 | 0.1843 | **1.75×** |
| Decode（PR5+ours） | 32 | 0.3431 | 0.2208 | **1.55×** |
| Prefill (+PR#3) | 1024 | 1.1643 | 1.1452 | **1.017×** |
| Prefill (+PR#3) | 2048 | 2.9817† | 2.8954 | **1.030×** |
| Prefill (+PR#3) | 4096 | 4.2332 | 4.1988 | **1.008×** |

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

仍约 **1.0×** 层加速；DSA+index_score 占主导。详见上一版与 Prefill 战役文档。

原始 CSV：`results/comparison_{decode,prefill,all}.csv` · `results/glm5_*_swapped_perf.csv`
