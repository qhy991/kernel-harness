# GLM-5.2 关键算子延时对比（llm_flops DROP-IN）

协议：**CUDA Graph**（warmup=5, runs=20），stock 与 candidate **同一冻结 tensor**。  
单位：ms。speedup = stock / ours（>1 为加速）。

- Decode winners：`best/*` archive  
- Prefill：含 **PR#3 胡延 10 算子** + 既有 `best/index_k` / `best/o_proj_prefill`

复现：`llm_flops_style/bench_{decode,prefill}.py` · 协作：[`COLLAB.md`](COLLAB.md)

---

## 层总览

| Phase | M | stock 层 (ms) | swapped 层 (ms) | 层加速 |
|-------|--:|-------------:|----------------:|-------:|
| Decode | 16 | 0.3282 | 0.2400 | **1.37×** |
| Decode | 32 | 0.3513 | 0.2697 | **1.30×** |
| Prefill (+PR#3) | 1024 | 1.1343 | 1.0913 | **1.040×** |
| Prefill (+PR#3) | 2048 | 2.1705 | 2.1240 | **1.022×** |
| Prefill (+PR#3) | 4096 | 4.1832 | 4.1745 | **1.002×** |

---

## Decode — 已优化算子

| op | M | stock (ms) | ours (ms) | speedup | source |
|----|--:|-----------:|----------:|--------:|--------|
| q_b_proj | 16 | 0.0242 | 0.0075 | **3.23×** | `best/q_b_decode` |
| q_b_proj | 32 | 0.0244 | 0.0072 | **3.39×** | `best/q_b_decode` |
| o_proj | 16 | 0.0440 | 0.0219 | **2.01×** | `best/o_proj_decode_hbm35` |
| o_proj | 32 | 0.0451 | 0.0277 | **1.63×** | `best/o_proj_decode_hbm35` |
| index_q_upproj | 16 | 0.0172 | 0.0056 | **3.08×** | `best/index_q_upproj_decode_hbm15` |
| index_q_upproj | 32 | 0.0173 | 0.0073 | **2.37×** | `best/index_q_upproj_decode_hbm15` |
| moe_gate/up/down | 16 | ~0.038 | ~0.025 | **~1.49×** | `best/moe_*_decode_hbm40` |
| moe_gate/up/down | 32 | ~0.038 | ~0.025 | **~1.48×** | `best/moe_*_decode_hbm40` |

---

## Prefill — PR#3 + 既有 winners（drop-in）

| op | M=1024 spd | M=2048 spd | M=4096 spd | source |
|----|----------:|----------:|----------:|--------|
| **fused_qkv_a_proj** | **1.67×** | **1.37×** | **1.17×** | PR#3 `fused_qkv_a_prefill.py` |
| **q_b_proj** | **1.63×** | **1.34×** | **1.18×** | PR#3 `q_b_prefill.py` |
| **index_q_upproj** | **1.88×** | **1.55×** | **1.34×** | PR#3 `index_q_upproj_prefill.py` |
| **index_weights_proj** | **1.58×** | **1.33×** | **1.60×** | PR#3 `index_weights_proj.py`† |
| **index_k_proj** | **1.16×** | **1.18×** | **1.18×** | `best/index_k_prefill_bw70` |
| absorbed_W_UK | 1.00× | 1.00× | 1.00× | PR#3（预分配 out） |
| absorbed_W_UV | 1.00× | 0.99× | 1.00× | PR#3（≈stock） |
| dsa_prefill_attn | 0.98× | 1.01× | 1.01× | PR#3（≈stock） |
| index_score | 0.98× | 0.98× | 1.01× | PR#3（PDL） |
| o_proj | 1.00× | 0.99× | 0.94× | `best/o_proj_prefill` |
| moe_up_proj | 1.00× | 0.99× | 0.92× | PR#3 `moe_up_proj_prefill.py` |
| moe_down_proj | 1.00× | 1.00× | 0.94× | PR#3 `moe_down_proj_prefill.py` |
| moe_gate_proj | 1.00× | 1.00× | 1.00× | llm_flops stock（无候选） |

† `index_weights_proj` 内部自建 CUDA Graph，外层 Graph 捕获失败 → 回退 **cuda_event**；数字仍为同输入对比。

### Prefill 绝对延时（ms）— 有实质加速的算子

| op | M | stock | ours | speedup |
|----|--:|------:|-----:|--------:|
| fused_qkv_a_proj | 1024 | 0.0249 | 0.0149 | **1.67×** |
| fused_qkv_a_proj | 2048 | 0.0367 | 0.0268 | **1.37×** |
| fused_qkv_a_proj | 4096 | 0.0540 | 0.0461 | **1.17×** |
| q_b_proj | 1024 | 0.0413 | 0.0253 | **1.63×** |
| q_b_proj | 2048 | 0.0697 | 0.0519 | **1.34×** |
| q_b_proj | 4096 | 0.1148 | 0.0975 | **1.18×** |
| index_q_upproj | 1024 | 0.0212 | 0.0113 | **1.88×** |
| index_q_upproj | 2048 | 0.0267 | 0.0172 | **1.55×** |
| index_q_upproj | 4096 | 0.0361 | 0.0270 | **1.34×** |
| index_weights_proj | 1024 | 0.0168 | 0.0106 | **1.58×** |
| index_weights_proj | 2048 | 0.0167 | 0.0126 | **1.33×** |
| index_weights_proj | 4096 | 0.0170 | 0.0106 | **1.60×** |
| index_k_proj | 1024 | 0.0850 | 0.0735 | **1.16×** |
| index_k_proj | 2048 | 0.0856 | 0.0728 | **1.18×** |
| index_k_proj | 4096 | 0.0854 | 0.0726 | **1.18×** |

---

## 解读

- **Decode**：层加速仍主要来自 6 个 GEMM/MoE winners（1.30–1.37×）。
- **Prefill + PR#3**：真正拉动层时间的是 `fused_qkv_a` / `q_b` / `index_q_upproj` / `index_weights` / `index_k`；DSA+index_score 仍占 ~50%+，几乎无加速，故层总加速仅 **~1–4%**。
- M=4096 上 `moe_up/down`、`o_proj` 偶发 <1.0×，偏噪声/PDL 全局状态，建议复测确认。

原始 CSV：`results/glm5_{decode,prefill}_swapped_perf.csv`
