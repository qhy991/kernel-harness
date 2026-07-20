# GLM-5.2 关键算子延时对比（llm_flops DROP-IN）

协议：**CUDA Graph**（warmup=5, runs=20），stock 与 candidate **同一冻结 tensor**（`same_inputs=Y`）。  
单位：ms。speedup = stock / ours（>1 为加速）。

复现：`archive/0720-Best-GLM-52/llm_flops_style/bench_{decode,prefill}.py`  
协作说明：见同目录 [`COLLAB.md`](COLLAB.md)。

---

## 层总览

| Phase | M | stock 层 (ms) | swapped 层 (ms) | 层加速 | 节省 |
|-------|--:|-------------:|----------------:|-------:|-----:|
| Decode | 16 | 0.3282 | 0.2400 | **1.37×** | −26.9% |
| Decode | 32 | 0.3513 | 0.2697 | **1.30×** | −23.2% |
| Prefill | 1024 | 1.4138 | 1.4018 | **1.009×** | −0.8% |
| Prefill | 2048 | 2.1490 | 2.1358 | **1.006×** | −0.6% |
| Prefill | 4096 | 4.2124 | 4.1818 | **1.007×** | −0.7% |

---

## Decode — 已优化算子（drop-in）

| op | M | stock (ms) | ours (ms) | speedup | archive candidate |
|----|--:|-----------:|----------:|--------:|-------------------|
| q_b_proj | 16 | 0.0242 | 0.0075 | **3.23×** | `q_b_decode` |
| q_b_proj | 32 | 0.0244 | 0.0072 | **3.39×** | `q_b_decode` |
| o_proj | 16 | 0.0440 | 0.0219 | **2.01×** | `o_proj_decode_hbm35` |
| o_proj | 32 | 0.0451 | 0.0277 | **1.63×** | `o_proj_decode_hbm35` |
| index_q_upproj | 16 | 0.0172 | 0.0056 | **3.08×** | `index_q_upproj_decode_hbm15` |
| index_q_upproj | 32 | 0.0173 | 0.0073 | **2.37×** | `index_q_upproj_decode_hbm15` |
| moe_gate_proj | 16 | 0.0377 | 0.0253 | **1.49×** | `moe_gate_proj_decode_hbm40` |
| moe_gate_proj | 32 | 0.0376 | 0.0254 | **1.48×** | `moe_gate_proj_decode_hbm40` |
| moe_up_proj | 16 | 0.0376 | 0.0253 | **1.49×** | `moe_up_proj_decode_hbm40` |
| moe_up_proj | 32 | 0.0378 | 0.0253 | **1.49×** | `moe_up_proj_decode_hbm40` |
| moe_down_proj | 16 | 0.0388 | 0.0258 | **1.50×** | `moe_down_proj_decode_hbm40` |
| moe_down_proj | 32 | 0.0384 | 0.0261 | **1.47×** | `moe_down_proj_decode_hbm40` |

## Decode — 未优化（llm_flops stock，供对照）

| op | M=16 stock (ms) | M=32 stock (ms) |
|----|----------------:|----------------:|
| fused_qkv_a_proj | 0.0233 | 0.0236 |
| absorbed_W_UK | 0.0025 | 0.0027 |
| absorbed_W_UV | 0.0024 | 0.0025 |
| dsa_decode_attn | 0.0395 | 0.0397 |
| index_k_proj | 0.0212 | 0.0217 |
| index_weights_proj | 0.0133 | 0.0135 |
| index_score | 0.0264 | 0.0468 |

---

## Prefill — 已优化算子（drop-in）

| op | M | stock (ms) | ours (ms) | speedup | archive candidate |
|----|--:|-----------:|----------:|--------:|-------------------|
| index_k_proj | 1024 | 0.0839 | 0.0730 | **1.15×** | `index_k_prefill_bw70` |
| index_k_proj | 2048 | 0.0834 | 0.0728 | **1.14×** | `index_k_prefill_bw70` |
| index_k_proj | 4096 | 0.0832 | 0.0723 | **1.15×** | `index_k_prefill_bw70` |
| o_proj | 1024 | 0.0864 | 0.0852 | 1.01× | `o_proj_prefill` |
| o_proj | 2048 | 0.1588 | 0.1564 | 1.02× | `o_proj_prefill` |
| o_proj | 4096 | 0.3413 | 0.3115 | **1.10×** | `o_proj_prefill` |
| moe_down_proj | 1024 | 0.1015 | 0.1016 | 1.00× | `moe_down_proj_prefill_mfu65` |
| moe_down_proj | 2048 | 0.1844 | 0.1841 | 1.00× | `moe_down_proj_prefill_mfu65` |
| moe_down_proj | 4096 | 0.3459 | 0.3559 | 0.97× | `moe_down_proj_prefill_mfu65` |

## Prefill — 未优化（llm_flops stock，供对照）

| op | M=1024 | M=2048 | M=4096 |
|----|-------:|-------:|-------:|
| fused_qkv_a_proj | 0.0253 | 0.0365 | 0.0530 |
| q_b_proj | 0.0421 | 0.0695 | 0.1152 |
| absorbed_W_UK | 0.0153 | 0.0315 | 0.0585 |
| absorbed_W_UV | 0.0106 | 0.0228 | 0.0443 |
| dsa_prefill_attn | 0.5397 | 0.6012 | 1.2680 |
| index_q_upproj | 0.0221 | 0.0269 | 0.0360 |
| index_weights_proj | 0.0168 | 0.0168 | 0.0168 |
| index_score | 0.2650 | 0.5556 | 1.1837 |
| moe_gate_proj | 0.1018 | 0.1803 | 0.3333 |
| moe_up_proj | 0.1031 | 0.1813 | 0.3332 |

---

## 原始 CSV

- `llm_flops_style/results/glm5_decode_swapped_perf.csv`
- `llm_flops_style/results/glm5_prefill_swapped_perf.csv`
