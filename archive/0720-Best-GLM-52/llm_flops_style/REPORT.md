# llm_flops DROP-IN layer latency

**严格可比口径**：stock 与 candidate 使用 **同一套 llm_flops 构造的冻结 tensor**，同一 CUDA Graph 协议（warmup=5, runs=20）。

| 算子类型 | 做法 |
|---------|------|
| 未替换 | 直接调用 `llm_flops` 的 `bench_*` |
| 已替换 | `llm_flops` 量化路径建输入 → 同输入上分别测 stock `deep_gemm.*` 与 `candidate.run` |

复现：
```bash
cd archive/0720-Best-GLM-52/llm_flops_style
CUDA_VISIBLE_DEVICES=0 ../../../.venv/bin/python bench_decode.py
CUDA_VISIBLE_DEVICES=0 ../../../.venv/bin/python bench_prefill.py
```

---

## DECODE（13 ops）

| M | stock layer (ms) | swapped layer (ms) | layer speedup |
|---:|---:|---:|---:|
| 16 | 0.3282 | 0.2400 | **1.37×** |
| 32 | 0.3513 | 0.2697 | **1.30×** |

### Drop-in 替换（同输入 speedup）

| op | archive | M=16 | M=32 |
|---|---|---:|---:|
| q_b_proj | q_b_decode | **3.23×** | **3.39×** |
| index_q_upproj | index_q_upproj_decode_hbm15 | **3.08×** | **2.37×** |
| o_proj | o_proj_decode_hbm35 | **2.01×** | **1.63×** |
| moe_gate/up/down | *_decode_hbm40 | **~1.48–1.50×** | **~1.47–1.49×** |

全部 `same_inputs=Y`，`protocol=cuda_graph`。

---

## PREFILL（13 ops）

| M | stock layer (ms) | swapped layer (ms) | layer speedup |
|---:|---:|---:|---:|
| 1024 | 1.4138 | 1.4018 | **1.009×** |
| 2048 | 2.1490 | 2.1358 | **1.006×** |
| 4096 | 4.2124 | 4.1818 | **1.007×** |

### Drop-in 替换

| op | archive | M1024 | M2048 | M4096 |
|---|---|---:|---:|---:|
| index_k_proj | index_k_prefill_bw70 | **1.15×** | **1.14×** | **1.15×** |
| o_proj | o_proj_prefill | 1.01× | 1.02× | 1.10× |
| moe_down | moe_down_proj_prefill_mfu65 | 1.00× | 1.00× | 0.97× |

层收益仍几乎只来自 `index_k`；DSA + index_score 主导层时间。

---

## 可比性说明（修订后）

| 维度 | 状态 |
|------|------|
| 计时协议 vs llm_flops | ✅ 同 CUDA Graph |
| 输入构造 vs llm_flops stock | ✅ **同一 builder / 同一冻结 tensor** |
| 未替换算子 vs llm_flops CSV | ✅ 同代码路径 |
| 替换加速比 | ✅ **同输入 stock÷cand** |

CSV：`results/glm5_{decode,prefill}_swapped_perf.csv`（含 `same_inputs,dropin` 列）
