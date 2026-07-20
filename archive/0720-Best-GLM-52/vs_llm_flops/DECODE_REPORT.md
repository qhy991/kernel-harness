# Archive vs baseline — DECODE

Timing: **cupti-cold-l2-device-kernel-median** (warmup=3, iterations=10)
llm_flops reference CSV: `/home/qinhaiyan/llm_flops/runs/20260715-ue8m0/glm5_decode_perf.csv` (CUDA Graph — **不同协议，仅供参考**)

## Harness reference layer（12 ops，无替换）

| M | harness ref layer (µs) | llm_flops 12-op sum (µs) | harness/llm_flops |
|---:|---:|---:|---:|
| 16 | 519.4 | 308.4 | 1.68× |
| 32 | 461.3 | 331.5 | 1.39× |

## Per-op harness reference vs llm_flops

### M=16

| op | harness ref (µs) | llm_flops (µs) | harness/llm_flops |
|---|---:|---:|---:|
| fused_qkv_a | 33.4 | 22.8 | 1.47× |
| q_b | 46.0 | 23.9 | 1.92× |
| o_proj | 83.4 | 43.9 | 1.90× |
| index_q_upproj | 56.4 | 16.6 | 3.40× |
| index_k | 60.1 | 20.4 | 2.95× |
| absorbed_W_UK | 4.9 | 2.5 | 1.98× |
| absorbed_W_UV | 5.3 | 2.5 | 2.12× |
| moe_gate | 55.8 | 36.6 | 1.52× |
| moe_up | 47.6 | 36.7 | 1.30× |
| moe_down | 47.0 | 38.2 | 1.23× |
| dsa_attn | 46.0 | 38.8 | 1.19× |
| index_score | 33.5 | 25.5 | 1.31× |

### M=32

| op | harness ref (µs) | llm_flops (µs) | harness/llm_flops |
|---|---:|---:|---:|
| fused_qkv_a | 33.7 | 22.8 | 1.48× |
| q_b | 35.2 | 24.3 | 1.45× |
| o_proj | 54.4 | 44.7 | 1.22× |
| index_q_upproj | 33.8 | 16.8 | 2.01× |
| index_k | 32.9 | 20.4 | 1.61× |
| absorbed_W_UK | 5.1 | 2.8 | 1.83× |
| absorbed_W_UV | 5.5 | 2.6 | 2.12× |
| moe_gate | 50.6 | 36.7 | 1.38× |
| moe_up | 57.6 | 36.8 | 1.56× |
| moe_down | 46.6 | 37.8 | 1.23× |
| dsa_attn | 46.9 | 39.1 | 1.20× |
| index_score | 58.9 | 46.7 | 1.26× |

## Archive best candidates vs harness reference

speedup = harness_ref / harness_cand（>1 为加速）

### `q_b_decode` (TARGET_MET)

- task: `q_b_decode` · op: `q_b`

| M | harness ref (µs) | candidate (µs) | speedup | llm_flops (µs) | cand/llm_flops |
|---:|---:|---:|---:|---:|---:|
| 16 | 35.7 | 11.664 | **3.06×** | 23.9 | 0.488 |
| 32 | 45.0 | 11.728 | **3.84×** | 24.3 | 0.4826 |

### `o_proj_decode_hbm35` (TARGET_MET)

- task: `o_proj_decode` · op: `o_proj`

| M | harness ref (µs) | candidate (µs) | speedup | llm_flops (µs) | cand/llm_flops |
|---:|---:|---:|---:|---:|---:|
| 16 | 53.0 | 33.698 | **1.57×** | 43.9 | 0.7676 |
| 32 | 54.3 | 34.687 | **1.57×** | 44.7 | 0.776 |

### `o_proj_decode` (WIN)

- task: `o_proj_decode` · op: `o_proj`

| M | harness ref (µs) | candidate (µs) | speedup | llm_flops (µs) | cand/llm_flops |
|---:|---:|---:|---:|---:|---:|
| 16 | 68.6 | 62.288 | **1.10×** | 43.9 | 1.4189 |
| 32 | 54.0 | 60.673 | 0.89× (慢) | 44.7 | 1.3573 |

### `o_proj_decode_hbm40_extreme` (NO_GO)

- task: `o_proj_decode` · op: `o_proj`

| M | harness ref (µs) | candidate (µs) | speedup | llm_flops (µs) | cand/llm_flops |
|---:|---:|---:|---:|---:|---:|
| 16 | 57.9 | 34.495 | **1.68×** | 43.9 | 0.7858 |
| 32 | 65.4 | 33.873 | **1.93×** | 44.7 | 0.7578 |

### `index_q_upproj_decode_hbm15` (WIN_MISS_TARGET)

- task: `index_q_upproj_decode` · op: `index_q_upproj`

| M | harness ref (µs) | candidate (µs) | speedup | llm_flops (µs) | cand/llm_flops |
|---:|---:|---:|---:|---:|---:|
| 16 | 30.1 | 7.392 | **4.07×** | 16.6 | 0.4453 |
| 32 | 25.9 | 8.337 | **3.11×** | 16.8 | 0.4962 |

### `moe_gate_proj_decode_hbm40` (TARGET_MET)

- task: `moe_gate_proj_decode` · op: `moe_gate`

| M | harness ref (µs) | candidate (µs) | speedup | llm_flops (µs) | cand/llm_flops |
|---:|---:|---:|---:|---:|---:|
| 16 | 47.7 | 30.976 | **1.54×** | 36.6 | 0.8463 |
| 32 | 47.8 | 30.832 | **1.55×** | 36.7 | 0.8401 |

### `moe_up_proj_decode_hbm40` (TARGET_MET)

- task: `moe_up_proj_decode` · op: `moe_up`

| M | harness ref (µs) | candidate (µs) | speedup | llm_flops (µs) | cand/llm_flops |
|---:|---:|---:|---:|---:|---:|
| 16 | 47.3 | 30.945 | **1.53×** | 36.7 | 0.8432 |
| 32 | 47.6 | 31.12 | **1.53×** | 36.8 | 0.8457 |

### `moe_down_proj_decode_hbm40` (TARGET_MET)

- task: `moe_down_proj_decode` · op: `moe_down`

| M | harness ref (µs) | candidate (µs) | speedup | llm_flops (µs) | cand/llm_flops |
|---:|---:|---:|---:|---:|---:|
| 16 | 47.6 | 31.088 | **1.53×** | 38.2 | 0.8138 |
| 32 | 47.2 | 31.361 | **1.51×** | 37.8 | 0.8297 |

### `index_score_decode_hbm82` (NO_GO)

- task: `index_score_decode` · op: `index_score`

| M | harness ref (µs) | candidate (µs) | speedup | llm_flops (µs) | cand/llm_flops |
|---:|---:|---:|---:|---:|---:|
| 16 | 32.6 | 32.527 | **1.00×** | 25.5 | 1.2756 |
| 32 | 58.5 | 58.465 | 1.000× | 46.7 | 1.2519 |

### `dsa_attn_decode_hbm40` (NO_GO)

- task: `dsa_attn_decode` · op: `dsa_attn`

| M | harness ref (µs) | candidate (µs) | speedup | llm_flops (µs) | cand/llm_flops |
|---:|---:|---:|---:|---:|---:|
| 16 | 46.0 | 45.888 | **1.00×** | 38.8 | 1.1827 |
| 32 | 47.1 | 47.249 | 1.00× (慢) | 39.1 | 1.2084 |

### `absorbed_W_UV_decode_hbm86` (NO_GO)

- task: `absorbed_W_UV_decode` · op: `absorbed_W_UV`

| M | harness ref (µs) | candidate (µs) | speedup | llm_flops (µs) | cand/llm_flops |
|---:|---:|---:|---:|---:|---:|
| 16 | 5.4 | 5.232 | **1.03×** | 2.5 | 2.0928 |
| 32 | 5.5 | 5.552 | 0.99× (慢) | 2.6 | 2.1354 |

## 说明

- **可比**：archive candidate vs harness reference（同协议、同形状、同算子契约）
- **不可直接比绝对值**：harness CUPTI vs llm_flops CUDA Graph；`harness/llm_flops` 列仅作量级对照
- llm_flops CSV 含 `index_weights_proj`（第 13 算子），本 harness layer 为 12 op
