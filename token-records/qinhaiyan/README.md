# qinhaiyan experiment token ledger

Kernel-Harness 下按实验者名归档的 Codex `/goal` token 记录：
`token-records/qinhaiyan/`。

## 文件

| 文件 | 含义 |
|---|---|
| `experiments_tokens.csv` | 每次实验一行（会话汇总） |
| `experiments_tokens_requests.csv` | 每次模型请求一行（细粒度） |
| `extract_tokens.py` | 从 rollout + goals DB 重新提取 |

## 字段说明

来源：`~/.codex/sessions/**/rollout-*.jsonl` 中的 `token_count` 事件：

- `input_tokens`：该次请求的输入（**通常已包含** cached）
- `cached_input_tokens`：输入里命中 cache 的部分
- `uncached_input_tokens`：`input_tokens - cached_input_tokens`
- `output_tokens`：输出（含 reasoning）
- `reasoning_output_tokens`：reasoning 部分
- `non_reasoning_output_tokens`：`output_tokens - reasoning_output_tokens`
- `total_tokens`：该次 `input + output`

汇总表里的 `input_* / output_* / total_tokens` 是对去重后请求的求和；
`final_cum_*` 是最后一个 `total_token_usage` 快照（应与求和一致）。

`goal_tokens_used` 来自 `~/.codex/goals_1.sqlite`，是 UI “Goal usage” 数字，
它更接近 **uncached input**，不等于 `total_tokens`。

提取时只保留 `cum_total_tokens` 前进的事件，避免同一请求的重复 snapshot 被加两次。

## 本批

`glm52-codex-20260714`：5 个 Kernel-Harness GLM-5.2 任务，
`codex --yolo -m gpt-5.5` + `model_reasoning_effort=xhigh`。

重新生成：

```bash
python3 /home/qinhaiyan/Kernel-Harness/token-records/qinhaiyan/extract_tokens.py
```
