# GLM-5.2 测试结果汇总（0720 archive × Codex × serving bake-off × e2e）

- 整理日期：2026-07-24
- 范围：生产 ABI（packed int32 UE8M0）下的单算子 bake-off，以及 Codex / e2e 战役结案
- 权威结论一句话：**没有确认可上线、且在端到端里站得住的净收益；默认推理仍走 stock。**

---

## 1. 材料索引

| 路径 | 内容 |
|---|---|
| [`0720-Best-GLM-52/`](0720-Best-GLM-52/) | KDA 0720 best（多在 harness float-scale 上胜出） |
| [`codex-goal-ultra-xhigh/`](codex-goal-ultra-xhigh/) | Codex goal 25 任务归档（候选 + 文本证据） |
| [`serving_bakeoff_0720_vs_codex/`](serving_bakeoff_0720_vs_codex/) | 统一 `serving_native` bake-off 脚本与原始 JSON |
| [`serving_bakeoff_0720_vs_codex/results/bakeoff_summary.csv`](serving_bakeoff_0720_vs_codex/results/bakeoff_summary.csv) | 本轮数值表 |
| `~/glm52-goal-runs/EXPERIMENT_RESULTS.md` | Codex 人工结案（01–08、22–25） |
| `sglang/glm52_opt/history/e2e_candidates_20260723/` | e2e 候选启用/否定归档 |
| `sglang/glm52_opt/e2e_gain_ops_repro_SUMMARY.md` | 短上下文 TTFT「收益」复现失败 |

---

## 2. Serving bake-off（2026-07-24）

### 2.1 协议

| 项 | 值 |
|---|---|
| Runner | `serving_native/run.sh` |
| 计时 | 同进程 interleaved paired A/B，**eager CUDA events** |
| Warmup / repeat | 5 / 40 |
| Speedup | `stock_median / cand_median`（>1 更快）；门禁 paired p50 ≥ 1.03× |
| 形状 | decode `M∈{16,32}`，单卡 |
| **不是** | CUDA Graph replay；也不是 TP8/EP8 端到端 |

> 对本协议：数字只能说明「eager leaf A/B」；**不能**直接当成线上可上线收益。

### 2.2 可跑结果（全部 correct=pass）

| Op | M | 0720-port | Codex | 备注 |
|---|---:|---:|---:|---|
| indexer_wq_b | 16 | **1.185×** | **1.103×** | 同源 packed Triton（goal-15 / hechenxi port） |
| indexer_wq_b | 32 | **1.096×** | **1.159×** | 同上；两次 run 方差可见 |
| fused_qkv_a | 16 | **1.066×** | — | 新建 packed Triton（N=2624）；无独立 Codex 战役 |
| fused_qkv_a | 32 | **1.050×** | — | 同上 |
| q_b | 16 | absorbed_in_stock | **1.507×** | Codex packed_warp；见 §3 假象说明 |
| q_b | 32 | absorbed_in_stock | **1.349×** | 同上 |
| moe_w13 | 16 | absorbed_in_stock | **1.053×** | goal-19 overlay（mainline 适配） |
| moe_w13 | 32 | absorbed_in_stock | **1.068×** | 同上 |
| moe_w2 | 16 | absorbed_in_stock | **1.070×** | goal-06 direct launch（诊断向） |
| moe_w2 | 32 | absorbed_in_stock | **1.035×** | 同上 |

原始 ms 见 CSV：`ref_p50_ms` / `cand_p50_ms`（实为 runner 的 `median_ms`）。

### 2.3 未跑 / 标注

| 状态 | 含义 | 涉及 |
|---|---|---|
| `absorbed_in_stock` | 0720 胜出来自 f32→packed pack，生产 ABI 已含 | q_b / o_proj / 分离 moe_gate·up·down |
| `already_stock` | 0720 内核已是 serving stock 后端 | DSA（trtllm-gen） |
| `requires_goal_runtime` | 依赖 goal worktree Runtime API | indexer SMS / region 等 |

复现：

```bash
cd /home/qinhaiyan/Kernel-Harness/archive/serving_bakeoff_0720_vs_codex
./run_bakeoff.sh
```

---

## 3. 如何读 bake-off 数字（重要）

### 3.1 `q_b` Codex ~1.35–1.51×：**不可部署**

Goal-14 权威报告（已收入 `sglang/glm52_opt/history/.../_negative/14_attn_q_b_packed_REPORT.md`）：

- Eager CUDA-event「加速」含 **CPU→GPU submission gap**；实验 TVM-FFI 提交更快，**设备核更慢**
- 生产 CUDA Graph：M16/M32 ≈ **0.96–0.98×**（更慢）
- NCU：stock 核时更短；结案 **NO REPLACEMENT**，默认 **不 enable** 任何 q_b bucket

因此 bake-off 里 q_b 的 1.5× **不得**写入「有上线收益」。

### 3.2 indexer / fused 的 >1.03×

- indexer packed 路径：leaf 上可过门禁，但 goal-15 e2e 归档为 **数值/SM 问题，未晋升**
- fused_qkv_a packed port：仅 leaf eager；**无** Codex 生产战役结案，也未进 `serving_safe`

### 3.3 MoE overlay / direct launch

- 过 1.03× eager 门禁 ≠ 可进默认 profile
- W2 direct launch 本身是诊断 floor；W13 overlay 需 goal worktree manifest，未作默认 swap

---

## 4. Codex 战役结案（leaf / 上线）

来源：`~/glm52-goal-runs/EXPERIMENT_RESULTS.md`（截至 2026-07-23 已分析 01–08、22–25）。

| 结论 | 内容 |
|---|---|
| 可上线优化数 | **0 / 11** 已结案任务 |
| leaf 有加速但未上线 | goal-07 BM16 W2（~1.06–1.09×，alignment 全局不安全）；goal-08 PSUM（~1.06×）；goal-25 EP4 joint Config（诊断-only） |
| 典型处置 | no-replacement / blocked / leaf-win+no-replacement |

---

## 5. 端到端（serving）现状

| 证据 | 结论 |
|---|---|
| 默认 `SGLANG_GLM52_OPT=0` / `serving_safe` | **无**隐式 archive/Codex swap；线上 stock |
| `e2e_candidates` 试跑集合 | o_proj / MoE PSUM 等仅显式 profile；**未**晋升默认 |
| 明确否定进 e2e enable | q_b packed、indexer wq_b、BM16 W2（全局 alignment）等 |
| 短上下文 TTFT「收益」复现（2026-07-24） | fused / index_q / all_gain 在 seq=2048 **未复现**；现为噪声或变慢 |

**汇总：leaf 偶有加速；e2e 无确认可复现净收益；生产默认路径等于都没用上这些生成算子。**

---

## 6. 推理路径上「会不会跑到」

| 算子 | 推理会不会执行该层 | 是否跑到本归档候选 |
|---|---|---|
| q_b_proj | 会（MLA absorb → ColumnParallelLinear） | **否**，stock `fp8_gemm_nt` |
| indexer wq_b | 会 | **否**（默认）；packed 未 enable |
| fused_qkv_a | 会 | **否**（默认） |
| moe_w13 / w2 | 会 | **否**（默认）；PSUM 仅特殊 DeepEP/layout 试验 |

要显式试：`SGLANG_GLM52_OPT=1` + profile/allowlist（且多数候选仍会 fail-closed 回 stock）。

---

## 7. 代码 / 数据快照（整理时）

| 组件 | Rev / 位置 |
|---|---|
| Kernel-Harness | branch `harness-experience-bank` @ `0e9f8e9`（本归档为工作树未提交内容） |
| sglang | branch `glm52-opt` @ `0a723222c`（已含 e2e/nsys 文档） |
| Bake-off 生成时间 | `2026-07-24T08:38:27Z`（见 `serving_bakeoff_0720_vs_codex/BAKEOFF.md`） |

---

## 8. 建议读法

1. 看数字 → [`serving_bakeoff_0720_vs_codex/results/bakeoff_summary.csv`](serving_bakeoff_0720_vs_codex/results/bakeoff_summary.csv)
2. 看协议与标注 → [`serving_bakeoff_0720_vs_codex/README.md`](serving_bakeoff_0720_vs_codex/README.md)
3. 看「能不能上线」→ 本文 §3–§5，勿只看 bake-off speedup 列
4. 看 Codex 逐 goal → `codex-goal-ultra-xhigh/EXPERIMENT_RESULTS.md` 或 `~/glm52-goal-runs/EXPERIMENT_RESULTS.md`
