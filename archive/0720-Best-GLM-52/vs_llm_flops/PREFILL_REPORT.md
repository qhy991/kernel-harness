# Archive vs baseline — PREFILL

Timing: **cupti-cold-l2-device-kernel-median** (warmup=3, iterations=10)
llm_flops reference CSV: `/home/qinhaiyan/llm_flops/runs/20260715-ue8m0/glm5_unified_perf.csv` (CUDA Graph — **不同协议，仅供参考**)

## Harness reference layer（12 ops，无替换）

| M | harness ref layer (µs) | llm_flops 12-op sum (µs) | harness/llm_flops |
|---:|---:|---:|---:|
| 1024 | 1179.9 | 1112.4 | 1.06× |
| 2048 | 2017.5 | 2012.7 | 1.00× |
| 4096 | 3649.9 | 4008.9 | 0.91× |

## Per-op harness reference vs llm_flops

### M=1024

| op | harness ref (µs) | llm_flops (µs) | harness/llm_flops |
|---|---:|---:|---:|
| fused_qkv_a | 42.8 | 25.4 | 1.68× |
| q_b | 52.0 | 41.6 | 1.25× |
| o_proj | 91.2 | 86.2 | 1.06× |
| index_q_upproj | 33.2 | 21.7 | 1.53× |
| index_k | 102.0 | 84.9 | 1.20× |
| absorbed_W_UK | 16.5 | 15.0 | 1.10× |
| absorbed_W_UV | 15.9 | 10.2 | 1.56× |
| moe_gate | 98.2 | 96.1 | 1.02× |
| moe_up | 98.1 | 96.3 | 1.02× |
| moe_down | 96.7 | 99.8 | 0.97× |
| dsa_attn | 284.1 | 278.2 | 1.02× |
| index_score | 249.2 | 257.0 | 0.97× |

### M=2048

| op | harness ref (µs) | llm_flops (µs) | harness/llm_flops |
|---|---:|---:|---:|
| fused_qkv_a | 50.0 | 35.3 | 1.42× |
| q_b | 70.7 | 64.2 | 1.10× |
| o_proj | 142.6 | 152.5 | 0.94× |
| index_q_upproj | 37.4 | 26.4 | 1.42× |
| index_k | 101.6 | 85.1 | 1.19× |
| absorbed_W_UK | 31.6 | 31.9 | 0.99× |
| absorbed_W_UV | 25.9 | 22.5 | 1.15× |
| moe_gate | 158.7 | 169.7 | 0.94× |
| moe_up | 158.7 | 170.4 | 0.93× |
| moe_down | 162.4 | 179.1 | 0.91× |
| dsa_attn | 557.3 | 554.5 | 1.01× |
| index_score | 520.6 | 521.1 | 1.00× |

### M=4096

| op | harness ref (µs) | llm_flops (µs) | harness/llm_flops |
|---|---:|---:|---:|
| fused_qkv_a | 66.1 | 54.7 | 1.21× |
| q_b | 110.0 | 112.0 | 0.98× |
| o_proj | 271.1 | 286.7 | 0.95× |
| index_q_upproj | 45.8 | 35.6 | 1.29× |
| index_k | 101.5 | 84.9 | 1.20× |
| absorbed_W_UK | 58.8 | 58.7 | 1.00× |
| absorbed_W_UV | 45.4 | 44.6 | 1.02× |
| moe_gate | 290.7 | 320.1 | 0.91× |
| moe_up | 290.3 | 316.0 | 0.92× |
| moe_down | 311.7 | 331.9 | 0.94× |
| dsa_attn | 1110.5 | 1247.4 | 0.89× |
| index_score | 948.0 | 1116.3 | 0.85× |

## Archive best candidates vs harness reference

speedup = harness_ref / harness_cand（>1 为加速）

### `index_k_prefill_bw70` (WIN_MISS_TARGET)

- task: `index_k_prefill` · op: `index_k`

| M | harness ref (µs) | candidate (µs) | speedup | llm_flops (µs) | cand/llm_flops |
|---:|---:|---:|---:|---:|---:|
| 1024 | 101.1 | 84.063 | **1.20×** | 84.9 | 0.9901 |
| 2048 | 98.5 | 83.776 | **1.18×** | 85.1 | 0.9844 |
| 4096 | 98.6 | 84.448 | **1.17×** | 84.9 | 0.9947 |

### `moe_down_proj_prefill_mfu65` (PARTIAL_WIN)

- task: `moe_down_proj_prefill` · op: `moe_down`

| M | harness ref (µs) | candidate (µs) | speedup | llm_flops (µs) | cand/llm_flops |
|---:|---:|---:|---:|---:|---:|
| 1024 | 104.6 | 104.593 | 1.000× | 99.8 | 1.048 |
| 2048 | 181.3 | 182.464 | 0.99× (慢) | 179.1 | 1.0188 |
| 4096 | 339.0 | 338.496 | **1.00×** | 331.9 | 1.0199 |

### `o_proj_prefill` (PARTIAL_WIN)

- task: `o_proj_prefill` · op: `o_proj`
- **Archive bug fixed 2026-07-20:** `candidate/` previously pointed at the task8 Triton spike (~0.21–0.28×). Restored to FINAL (`mnk` + PDL). Spike kept under `candidates/task8_triton_spike/`.

| M | harness ref (µs) | candidate (µs) | speedup | llm_flops (µs) | cand/llm_flops |
|---:|---:|---:|---:|---:|---:|
| 1024 | 91.9 | 90.463 | **1.02×** | 86.2 | 1.0495 |
| 2048 | 141.2 | 139.454 | **1.01×** | 152.5 | 0.9145 |
| 4096 | 270.4 | 269.267 | **1.00×** | 286.7 | 0.9392 |

`run.sh --candidate` gate (repeat=3): exit 0, correct, 2 wins / 0 regressions / 1 neutral.

### `moe_gate_proj_prefill_mfu` (NO_GO)

- task: `moe_gate_proj_prefill` · op: `moe_gate`

| M | harness ref (µs) | candidate (µs) | speedup | llm_flops (µs) | cand/llm_flops |
|---:|---:|---:|---:|---:|---:|
| 1024 | 104.6 | 106.431 | 0.98× (慢) | 96.1 | 1.1075 |
| 2048 | 174.9 | 174.621 | **1.00×** | 169.7 | 1.029 |
| 4096 | 319.8 | 319.277 | **1.00×** | 320.1 | 0.9974 |

## 说明

- **可比**：archive candidate vs harness reference（同协议、同形状、同算子契约）
- **不可直接比绝对值**：harness CUPTI vs llm_flops CUDA Graph；`harness/llm_flops` 列仅作量级对照
- llm_flops CSV 含 `index_weights_proj`（第 13 算子），本 harness layer 为 12 op
