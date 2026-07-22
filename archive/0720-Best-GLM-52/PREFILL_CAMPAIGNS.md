# Prefill 优化战役（2026-07-20，API 计费）

Claude 已切到 **API（infini-ai）**；以下三条 **Prefill** RLCR 已启动。

| 战役 | Phase | 层占比 | 硬目标 | tmux | GPU |
|------|-------|--------|--------|------|-----|
| `dsa_prefill_attn` | **Prefill** | ~27–47% | CUPTI ≥**1.15×**（勿套 decode 40% HBM） | `kda-glm52-dsa-prefill-spd` | 0 |
| `index_score_prefill` | **Prefill** | ~20–28% | CUPTI ≥**1.10×**（dense `fp8_mqa_logits`，勿套 82% HBM） | `kda-glm52-index-score-prefill-spd` | 1 |
| `moe_up_proj_prefill` | **Prefill** | ~8% | CUPTI ≥**1.10×** 且 Graph M4096 ≥**1.00×** | `kda-glm52-moe-up-prefill-pack` | 2 |

**Drop-in 已接入**：`best/moe_up_proj_decode_hbm40` 在 M1024/2048 Graph **1.05–1.10×**；M4096 仍 **~0.94×**，战役目标不变。

**decode→prefill 直搬结论**（见 `llm_flops_style/results/decode_on_prefill/`）：`fused_qkv_a`/`index_q` decode OOM；`dsa` decode 回退；共用 pack 见上表。

查看进度：
```bash
tmux attach -t kda-glm52-dsa-prefill-spd
tmux attach -t kda-glm52-index-score-prefill-spd
tmux attach -t kda-glm52-moe-up-prefill-pack
bash /home/qinhaiyan/QuickSetUp/claude-mode.sh status   # 应为 API
```

Worktrees：
- `KDA-Pilot-Exp-worktrees/dsa-prefill-attn-spd-dsa-prefill-spd-20260720`
- `KDA-Pilot-Exp-worktrees/index-score-prefill-spd-index-score-prefill-spd-20260720`
- `KDA-Pilot-Exp-worktrees/moe-up-prefill-pack-moe-up-prefill-pack-20260720`
