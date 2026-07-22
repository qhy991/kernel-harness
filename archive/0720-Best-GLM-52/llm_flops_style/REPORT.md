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
| 16 | 0.3227 | 0.1843 | **1.75×** |
| 32 | 0.3431 | 0.2208 | **1.55×** |

PR#5 + ours 组合；详见 [`COMPARISON_TABLE.md`](COMPARISON_TABLE.md) 与 [`PR5_VS_OURS.md`](PR5_VS_OURS.md)。

---

## PREFILL（13 ops）

**2026-07-20 更新**：prefill 与 decode **共用** `best/` 的 `o_proj`、`index_k`、`moe_up`、`moe_down`（pack+PDL）。

| M | stock layer (ms) | swapped layer (ms) | layer speedup |
|---:|---:|---:|---:|
| 1024 | 1.4059 | 1.3495 | **1.042×** |
| 2048 | 2.3158 | 2.0976 | **1.104×** |
| 4096 | 4.1338 | 4.1137 | **1.005×** |

相对上一版（~1.02× 层加速），M1024/2048 再抬 **~2–8%**；M4096 基本持平。

### decode/prefill 共用 swap

| op | archive | M1024 | M2048 | M4096 |
|---|---|---:|---:|---:|
| o_proj | `best/o_proj_decode_hbm35` | **1.13×** | **1.97×**† | **1.05×** |
| index_k_proj | `best/index_k_proj_decode` | **1.16×** | **1.14×** | **1.15×** |
| moe_up_proj | `best/moe_up_proj_decode_hbm40` | **1.10×** | **1.05×** | 0.94× |
| moe_down_proj | `best/moe_down_proj_decode_hbm40` | **1.08×** | **1.03×** | 1.00× |

† M2048 `o_proj` stock 偶发抖动；其余 PR#3 prefill 专用算子不变。

瓶颈仍是 **DSA + index_score**（~50% 层时间）；见 [`PREFILL_CAMPAIGNS.md`](../PREFILL_CAMPAIGNS.md)。

---

## 可比性说明

| 维度 | 状态 |
|------|------|
| 计时协议 vs llm_flops | ✅ 同 CUDA Graph |
| 输入构造 vs llm_flops stock | ✅ **同一 builder / 同一冻结 tensor** |
| 未替换算子 vs llm_flops CSV | ✅ 同代码路径 |
| 替换加速比 | ✅ **同输入 stock÷cand** |

CSV：`results/glm5_{decode,prefill}_swapped_perf.csv` · `results/comparison_{decode,prefill,all}.csv`
