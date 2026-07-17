# Provider A/B benchmark — bmm-absorb (same model, two APIs)

Compare **`claude-opus-4-8`** as served by **Infini-AI** vs **official Anthropic**, both
optimizing the same kernel task `q_nope_absorb_bmm_decode` (bf16 `torch.bmm`, real
headroom). I (the assistant) cannot type `/goal`, change `ANTHROPIC_BASE_URL`, or
restart Claude, so **you drive the two runs**; this kit makes them isolated and the
scoring tamper-proof.

## Isolation (so the runs can't contaminate each other)
- `task_infini/` and `task_official/` — separate working copies, **distinct dir names**
  → distinct `/tmp/kernel-harness/<name>-*` outputs and separate `.baseline_cache.json`.
  (Distinct names matter: `evaluate.py` keys its temp dirs on the task dir *name*, so
  same-named copies — even in different worktrees — would collide in `/tmp`.)
- `_pristine/` — the canonical oracle (never edited); scoring copies from it.
- Run each provider in its **own Claude session** (that's where the API/base_url is
  fixed). For extra OS-level isolation you can also run them on different machines/users;
  worktrees are not required because the dir names already separate the artifacts.

## Run it (per provider, in a session pointed at that provider's API)

**Infini-AI session** (`ANTHROPIC_BASE_URL=https://cloud.infini-ai.com/maas`, current default):
1. `/goal` → paste `GOAL.txt` **but replace the task path with** `.../task_infini`.
2. Send `PROMPT_infini.txt` as the first message. Let it run to its verdict.

**Official-Anthropic session** (set `ANTHROPIC_BASE_URL` to the official endpoint +
official key, restart Claude, keep `ANTHROPIC_MODEL=claude-opus-4-8`):
1. `/goal` → paste `GOAL.txt` with the task path `.../task_official`.
2. Send `PROMPT_official.txt`. Let it run.

Both must keep `ANTHROPIC_MODEL=claude-opus-4-8` so it's the *same model, two providers*.

## Score (fair, anti-cheat — run once, after both finish)
```bash
cd /home/qinhaiyan/kernel-harness/testbench/benchmarks/bmm_absorb_provider_ab
/home/qinhaiyan/kernel-harness/.venv/bin/python score.py --repeat 5
```
For each provider it lifts **only its final `solution.py`**, drops it into a fresh copy
of `_pristine/` (discarding any edits to reference/workload/definition), and
re-benchmarks both **back-to-back on this GPU** — so cross-session timing drift is
removed and oracle tampering can't become a score. Output: per-provider
`correct / win / geomean / min_speedup_conservative / tampered`, the winner (correct +
best worst-case speedup), and `results/scoreboard.json`.

## Reading it
- `win=true` needs correct on every shape AND `min_speedup_conservative > 1.0`.
- `tampered=true` → that provider edited the oracle; its numbers are void, investigate.
- bmm-absorb is memory/launch-bound (~6µs); expect modest margins. The comparison is
  *which provider's session produced the better correct kernel*, not the absolute µs.

Files: `GOAL.txt`, `PROMPT_infini.txt`, `PROMPT_official.txt`, `score.py`,
`task_infini/`, `task_official/`, `_pristine/`, `results/`.
