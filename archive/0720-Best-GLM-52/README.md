# 0720-Best-GLM-52

2026-07-20 整理：**所有 agent 产出的「该战役最佳正确候选」**，不论是否加速、是否达到硬目标。

协议：CUPTI cold-L2 device-kernel median，B200，HBM peak 8.0 TB/s。

## 主目录：`best/`

每个子目录 = 一次战役的最佳 agent 产物，通常含：

- `candidate/` — 可复测的最好正确实现
- `results.md` / `run_log.md` / `attempt_dag.md`（若有）
- `SOURCE.md` — 原始 worktree 与分类

完整清单见 `CATALOG.md`、`manifest.json`。

### 分类说明

| 标签 | 含义 |
|---|---|
| `TARGET_MET` | 硬目标达成 |
| `WIN` / `WIN_MISS_TARGET` | 相对 harness 参考有加速/WIN，但硬 HBM/MFU 目标未达 |
| `PARTIAL_WIN` | 部分 shape 达标或 harness 过门，硬目标整体未达 |
| `NO_GO` | 证据型不可达；仍保留当时最好正确候选（常为 stock/seed） |

---

## 一览

| Op / campaign | 分类 | 一句话 |
|---|---|---|
| **q_b_decode** | TARGET_MET | DeepGEMM fork fused pack → ~36% HBM |
| **moe_up_proj_decode_hbm40** | TARGET_MET | pack + PDL → ~41% |
| **moe_gate_proj_decode_hbm40** | TARGET_MET | pack + PDL → ~41% |
| **moe_down_proj_decode_hbm40** | TARGET_MET | pack → ~41% |
| **o_proj_decode_hbm35** | TARGET_MET | pack → ~38%（≥35%） |
| **o_proj_decode** | WIN | 早期 drop-in WIN |
| **index_q_upproj_decode_hbm15** | WIN_MISS_TARGET | ~3.1×，14.6%/12.9%，>15% NO-GO |
| **index_k_prefill_bw70** | WIN_MISS_TARGET | ~1.19×，~64% HBM，≥70% NO-GO |
| **o_proj_decode_hbm40_extreme** | NO_GO | ~38% 最好，>40% 不可达 |
| **moe_down_proj_prefill_mfu65** | PARTIAL_WIN | 部分 shape WIN，65% NO-GO |
| **moe_gate_proj_prefill_mfu** | NO_GO | 保留最好正确候选 |
| **o_proj_prefill** | PARTIAL_WIN | 薄加速；`candidate/` 来自 task8 |
| **index_score_decode_hbm82** | NO_GO | stock；≥82% 物理不可达 |
| **dsa_attn_decode_hbm40** | NO_GO | stock；M16 gather floor |
| **absorbed_W_UV_decode_hbm86** | NO_GO | stock；span floor |

DeepGEMM fork 变体：`deepgemm-fork/`（推荐 `41c6235` fused pack）。

---

## 补充

- `LEADERBOARD.md`：按 HBM 排序的达标向排行。
- `deepgemm-fork/`：DeepGEMM-GLM52 变体注册说明。

## 复测示例

```bash
H=/home/qinhaiyan/Kernel-Harness
C=archive/0720-Best-GLM-52/best/moe_up_proj_decode_hbm40/candidate
CUDA_VISIBLE_DEVICES=0 $H/testbench/tasks/glm52/moe_up_proj_decode/run.sh --candidate "$C"
```
