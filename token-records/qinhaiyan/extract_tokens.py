#!/usr/bin/env python3
"""Rebuild experiments_tokens*.csv from Codex rollout logs + goals DB.

Default batch: the five glm52 Kernel-Harness /goal runs from 2026-07-14.
"""
from __future__ import annotations

import csv
import json
import sqlite3
import textwrap
from pathlib import Path

OUT = Path(__file__).resolve().parent
SESS_ROOT = Path.home() / ".codex" / "sessions"
GOALS_DB = Path.home() / ".codex" / "goals_1.sqlite"

EXPERIMENTS = [
    dict(
        task="glm52/routed_swiglu_decode",
        tmux_session="kh-glm-routed_swiglu_decode",
        session_id="019f6150-b7f2-7791-9dda-83e4fa81b470",
        result="no-win",
        geomean_speedup=0.9959,
        min_speedup_conservative=0.9656,
        integrate_status="not-run",
        drop_in_verified=None,
        integration_contract="drop-in",
    ),
    dict(
        task="glm52/routed_swiglu_prefill",
        tmux_session="kh-glm-routed_swiglu_prefill",
        session_id="019f6150-b81e-7730-9a0f-ffd286bcec37",
        result="win",
        geomean_speedup=1.9974,
        min_speedup_conservative=1.804,
        integrate_status="pass",
        drop_in_verified=True,
        integration_contract="drop-in",
    ),
    dict(
        task="glm52/sparse_mla_decode",
        tmux_session="kh-glm-sparse_mla_decode",
        session_id="019f6150-b892-7903-89af-519154d0b683",
        result="win",
        geomean_speedup=1.0611,
        min_speedup_conservative=1.0186,
        integrate_status="no-recipe",
        drop_in_verified=None,
        integration_contract="fused-only",
    ),
    dict(
        task="glm52/o_proj_decode",
        tmux_session="kh-glm-o_proj_decode",
        session_id="019f6150-b865-7c73-a939-0b3a0fdbbf06",
        result="no-win",
        geomean_speedup=1.0,
        min_speedup_conservative=0.9875,
        integrate_status="not-run",
        drop_in_verified=None,
        integration_contract="drop-in",
    ),
    dict(
        task="glm52/routed_down_decode",
        tmux_session="kh-glm-routed_down_decode",
        session_id="019f6150-b876-7be1-a1a3-5b44dd07ad2d",
        result="win",
        geomean_speedup=1.2379,
        min_speedup_conservative=1.2074,
        integrate_status="pass",
        drop_in_verified=True,
        integration_contract="drop-in",
    ),
]


def load_goals() -> dict[str, dict]:
    if not GOALS_DB.exists():
        return {}
    con = sqlite3.connect(f"file:{GOALS_DB}?mode=ro", uri=True)
    out = {}
    for sid, tokens, secs, status in con.execute(
        "SELECT thread_id, tokens_used, time_used_seconds, status FROM thread_goals"
    ):
        out[sid] = {
            "tokens_used": tokens,
            "time_used_seconds": secs,
            "status": status,
        }
    con.close()
    return out


def find_rollout(session_id: str) -> Path:
    hits = sorted(SESS_ROOT.rglob(f"*{session_id}*.jsonl"))
    if not hits:
        raise FileNotFoundError(f"rollout for {session_id}")
    return hits[0]


def parse_rollout(path: Path) -> dict:
    model = None
    effort = None
    cwd = None
    started = ended = None
    context_window = None
    requests: list[dict] = []

    for line in path.open():
        o = json.loads(line)
        started = started or o.get("timestamp")
        ended = o.get("timestamp") or ended
        t = o.get("type")
        pl = o.get("payload") or {}

        if t == "session_meta":
            cwd = cwd or pl.get("cwd")
        if t == "turn_context":
            model = model or pl.get("model")
            effort = (
                effort
                or pl.get("effort")
                or pl.get("reasoning_effort")
                or pl.get("model_reasoning_effort")
            )
            cwd = cwd or pl.get("cwd")

        if not (t == "event_msg" and pl.get("type") == "token_count"):
            continue
        info = pl.get("info") or {}
        last = info.get("last_token_usage") or {}
        tot = info.get("total_token_usage") or {}
        context_window = info.get("model_context_window") or context_window
        if not tot:
            continue
        cum_total = int(tot.get("total_tokens") or 0)
        # Drop duplicate snapshots that do not advance cumulative usage.
        if requests and cum_total <= requests[-1]["cum_total_tokens"]:
            continue
        inp = int(last.get("input_tokens") or 0)
        cached = int(last.get("cached_input_tokens") or 0)
        out = int(last.get("output_tokens") or 0)
        reason = int(last.get("reasoning_output_tokens") or 0)
        requests.append(
            {
                "timestamp": o.get("timestamp"),
                "input_tokens": inp,
                "cached_input_tokens": cached,
                "uncached_input_tokens": inp - cached,
                "output_tokens": out,
                "reasoning_output_tokens": reason,
                "non_reasoning_output_tokens": out - reason,
                "total_tokens": int(last.get("total_tokens") or (inp + out)),
                "cum_input_tokens": int(tot.get("input_tokens") or 0),
                "cum_cached_input_tokens": int(tot.get("cached_input_tokens") or 0),
                "cum_uncached_input_tokens": int(tot.get("input_tokens") or 0)
                - int(tot.get("cached_input_tokens") or 0),
                "cum_output_tokens": int(tot.get("output_tokens") or 0),
                "cum_reasoning_output_tokens": int(
                    tot.get("reasoning_output_tokens") or 0
                ),
                "cum_total_tokens": cum_total,
            }
        )

    def sumkey(k: str) -> int:
        return sum(r[k] for r in requests)

    final = requests[-1] if requests else {}
    return {
        "model": model or "gpt-5.5",
        "reasoning_effort": effort or "xhigh",
        "cwd": cwd,
        "started_at": started,
        "ended_at": ended,
        "context_window": context_window,
        "n_requests": len(requests),
        "requests": requests,
        "input_tokens": sumkey("input_tokens"),
        "cached_input_tokens": sumkey("cached_input_tokens"),
        "uncached_input_tokens": sumkey("uncached_input_tokens"),
        "output_tokens": sumkey("output_tokens"),
        "reasoning_output_tokens": sumkey("reasoning_output_tokens"),
        "non_reasoning_output_tokens": sumkey("non_reasoning_output_tokens"),
        "total_tokens": sumkey("total_tokens"),
        "final_cum_input_tokens": final.get("cum_input_tokens", 0),
        "final_cum_cached_input_tokens": final.get("cum_cached_input_tokens", 0),
        "final_cum_uncached_input_tokens": final.get("cum_uncached_input_tokens", 0),
        "final_cum_output_tokens": final.get("cum_output_tokens", 0),
        "final_cum_reasoning_output_tokens": final.get(
            "cum_reasoning_output_tokens", 0
        ),
        "final_cum_total_tokens": final.get("cum_total_tokens", 0),
        "rollout_path": str(path),
    }


def main() -> None:
    goals = load_goals()
    summary_rows: list[dict] = []
    request_rows: list[dict] = []

    for i, exp in enumerate(EXPERIMENTS, 1):
        path = find_rollout(exp["session_id"])
        parsed = parse_rollout(path)
        g = goals.get(exp["session_id"], {})
        eid = f"glm52-codex-{i:02d}"
        row = {
            "experiment_id": eid,
            "batch": "glm52-codex-20260714",
            "owner": "qinhaiyan",
            "task": exp["task"],
            "result": exp["result"],
            "geomean_speedup": exp["geomean_speedup"],
            "min_speedup_conservative": exp["min_speedup_conservative"],
            "evaluate_result": exp["result"],
            "integrate_status": exp.get("integrate_status", ""),
            "drop_in_verified": exp.get("drop_in_verified"),
            "integration_contract": exp.get("integration_contract", ""),
            "tmux_session": exp["tmux_session"],
            "session_id": exp["session_id"],
            "model": parsed["model"],
            "reasoning_effort": parsed["reasoning_effort"],
            "cwd": parsed["cwd"] or "/home/qinhaiyan/Kernel-Harness",
            "started_at": parsed["started_at"],
            "ended_at": parsed["ended_at"],
            "goal_status": g.get("status"),
            "goal_tokens_used": g.get("tokens_used"),
            "goal_time_seconds": g.get("time_used_seconds"),
            "goal_time_minutes": round((g.get("time_used_seconds") or 0) / 60.0, 2),
            "n_requests": parsed["n_requests"],
            "context_window": parsed["context_window"],
            "input_tokens": parsed["input_tokens"],
            "cached_input_tokens": parsed["cached_input_tokens"],
            "uncached_input_tokens": parsed["uncached_input_tokens"],
            "output_tokens": parsed["output_tokens"],
            "reasoning_output_tokens": parsed["reasoning_output_tokens"],
            "non_reasoning_output_tokens": parsed["non_reasoning_output_tokens"],
            "total_tokens": parsed["total_tokens"],
            "final_cum_input_tokens": parsed["final_cum_input_tokens"],
            "final_cum_cached_input_tokens": parsed["final_cum_cached_input_tokens"],
            "final_cum_uncached_input_tokens": parsed[
                "final_cum_uncached_input_tokens"
            ],
            "final_cum_output_tokens": parsed["final_cum_output_tokens"],
            "final_cum_reasoning_output_tokens": parsed[
                "final_cum_reasoning_output_tokens"
            ],
            "final_cum_total_tokens": parsed["final_cum_total_tokens"],
            "rollout_path": parsed["rollout_path"],
            "source_goals_db": str(GOALS_DB),
            "notes": (
                "Sums are over token_count events with increasing cum_total_tokens. "
                "input_tokens includes cached_input_tokens. goal_tokens_used is the "
                "Codex /goal overlay counter (often near uncached input, not total)."
            ),
        }
        summary_rows.append(row)
        for idx, req in enumerate(parsed["requests"], 1):
            request_rows.append(
                {
                    "experiment_id": eid,
                    "task": exp["task"],
                    "session_id": exp["session_id"],
                    "model": parsed["model"],
                    "reasoning_effort": parsed["reasoning_effort"],
                    "request_index": idx,
                    "timestamp": req["timestamp"],
                    "input_tokens": req["input_tokens"],
                    "cached_input_tokens": req["cached_input_tokens"],
                    "uncached_input_tokens": req["uncached_input_tokens"],
                    "output_tokens": req["output_tokens"],
                    "reasoning_output_tokens": req["reasoning_output_tokens"],
                    "non_reasoning_output_tokens": req[
                        "non_reasoning_output_tokens"
                    ],
                    "total_tokens": req["total_tokens"],
                    "cum_input_tokens": req["cum_input_tokens"],
                    "cum_cached_input_tokens": req["cum_cached_input_tokens"],
                    "cum_uncached_input_tokens": req["cum_uncached_input_tokens"],
                    "cum_output_tokens": req["cum_output_tokens"],
                    "cum_reasoning_output_tokens": req[
                        "cum_reasoning_output_tokens"
                    ],
                    "cum_total_tokens": req["cum_total_tokens"],
                }
            )

    sum_path = OUT / "experiments_tokens.csv"
    req_path = OUT / "experiments_tokens_requests.csv"
    with sum_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    with req_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(request_rows[0].keys()))
        w.writeheader()
        w.writerows(request_rows)

    (OUT / "README.md").write_text(
        textwrap.dedent(
            """\
            # qinhaiyan experiment token ledger

            Codex `/goal` 实验的 token 记账目录。

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
            python3 /home/qinhaiyan/qinhaiyan/extract_tokens.py
            ```
            """
        ),
        encoding="utf-8",
    )

    print(f"wrote {sum_path}")
    print(f"wrote {req_path}")
    for r in summary_rows:
        print(
            f"{r['task']:36s} total={r['total_tokens']:,} "
            f"in={r['input_tokens']:,} cached={r['cached_input_tokens']:,} "
            f"uncached={r['uncached_input_tokens']:,} out={r['output_tokens']:,} "
            f"goal={r['goal_tokens_used']:,}"
        )


if __name__ == "__main__":
    main()
