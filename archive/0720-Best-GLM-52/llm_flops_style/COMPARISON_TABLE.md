# GLM-5.2 单算子延时对比（llm_flops DROP-IN）

协议：**CUDA Graph**（warmup=5, runs=20），stock 与 candidate **同一冻结 tensor**。  
单位：ms。speedup = stock / ours（>1 为加速）。

- Decode：含本轮新晋 `best/fused_qkv_a_decode`（DeepGEMM fork fused）+ `best/index_k_proj_decode`
- Prefill：含 **PR#3 胡延 10 算子** + 既有 `best/index_k` / `best/o_proj_prefill`  
  （`moe_gate` pack 仅 CUPTI 有收益，Graph 大 M 回退，**不默认 swap**）

复现：`llm_flops_style/bench_{decode,prefill}.py` · 源码尝试：[`../SOURCE_MOD_ANALYSIS.md`](../SOURCE_MOD_ANALYSIS.md)

---

## 层总览

| Phase | M | stock 层 (ms) | swapped 层 (ms) | 层加速 |
|-------|--:|-------------:|----------------:|-------:|
| Decode | 16 | 0.3255 | 0.2159 | **1.51×** |
| Decode | 32 | 0.3475 | 0.2419 | **1.44×** |
| Prefill (+PR#3) | 1024 | 1.1643 | 1.1452 | **1.017×** |
| Prefill (+PR#3) | 2048 | 2.9817† | 2.8954 | **1.030×** |
| Prefill (+PR#3) | 4096 | 4.2332 | 4.1988 | **1.008×** |

† M=2048 本次 stock 受 `dsa_prefill_attn` 抖动抬高（~1.39 ms vs 常态 ~0.60 ms）；层加速方向仍约 1×，勿与上一版绝对 ms 直接比。

Decode 层加速相对上一版（1.37× / 1.30×）再抬 **~0.14×**，主要来自 `fused_qkv_a` + `index_k`。

---

## Decode — 已优化算子（本轮新增标 ★）

| op | M | stock (ms) | ours (ms) | speedup | source |
|----|--:|-----------:|----------:|--------:|--------|
| ★ fused_qkv_a_proj | 16 | 0.0233 | 0.0124 | **1.88×** | `best/fused_qkv_a_decode` |
| ★ fused_qkv_a_proj | 32 | 0.0235 | 0.0125 | **1.88×** | `best/fused_qkv_a_decode` |
| q_b_proj | 16 | 0.0239 | 0.0071 | **3.38×** | `best/q_b_decode` |
| q_b_proj | 32 | 0.0247 | 0.0074 | **3.36×** | `best/q_b_decode` |
| o_proj | 16 | 0.0443 | 0.0228 | **1.95×** | `best/o_proj_decode_hbm35` |
| o_proj | 32 | 0.0449 | 0.0279 | **1.61×** | `best/o_proj_decode_hbm35` |
| ★ index_k_proj | 16 | 0.0215 | 0.0089 | **2.42×** | `best/index_k_proj_decode` |
| ★ index_k_proj | 32 | 0.0212 | 0.0087 | **2.44×** | `best/index_k_proj_decode` |
| index_q_upproj | 16 | 0.0173 | 0.0054 | **3.19×** | `best/index_q_upproj_decode_hbm15` |
| index_q_upproj | 32 | 0.0174 | 0.0072 | **2.43×** | `best/index_q_upproj_decode_hbm15` |
| moe_gate/up/down | 16 | ~0.037 | ~0.025 | **~1.47×** | `best/moe_*_decode_hbm40` |
| moe_gate/up/down | 32 | ~0.037 | ~0.024 | **~1.52×** | `best/moe_*_decode_hbm40` |

---

## Prefill — PR#3 + 既有 winners（drop-in）

| op | M=1024 spd | M=2048 spd | M=4096 spd | source |
|----|----------:|----------:|----------:|--------|
| **fused_qkv_a_proj** | **~1.6×** | **~1.35×** | **~1.15×** | PR#3 |
| **q_b_proj** | **~1.6×** | **~1.3×** | **~1.15×** | PR#3 |
| **index_q_upproj** | **~1.8×** | **~1.55×** | **~1.3×** | PR#3 |
| **index_weights_proj** | **~1.9×** | **~1.3×** | **~1.6×** | PR#3 † |
| **index_k_proj** | **~1.19×** | **~1.15×** | **~1.17×** | `best/index_k_prefill_bw70` |
| moe_gate_proj | stock | stock | stock | CUPTI pack 见 `best/moe_gate_proj_prefill_pack`（不默认 swap） |

其余算子 ≈ stock；DSA + index_score 仍占层时间 ~50%+。

† `index_weights_proj` 外层 Graph 捕获失败 → **cuda_event**。

---

## 本轮源码/移植实验结论

| 尝试 | CUPTI | DROP-IN Graph | 处置 |
|------|-------|---------------|------|
| fused_qkv_a DeepGEMM fused | **1.69×** | **1.88×** | 晋升 + decode swap |
| fused_qkv_a task-local pack | FAIL（N%128≠0） | — | 保留失败实验 |
| index_k decode Triton pack | **2.67×** | **2.4×** | 晋升 + decode swap |
| moe_gate prefill pack+PDL | **1.12×** | 1.10/1.04/**0.95×** | 晋升候选；**不**默认 swap |

详见 [`SOURCE_MOD_ANALYSIS.md`](../SOURCE_MOD_ANALYSIS.md)。

原始 CSV：`results/comparison_{decode,prefill,all}.csv` · `results/glm5_*_swapped_perf.csv`
