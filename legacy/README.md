# legacy/ — the proxy benchmark stack

Everything here predates the GLM-5.2 task suite and is **not** an oracle for
anything. `AGENTS.md` has said so about `benchmarks/` since before it moved:

> The root `benchmarks/` directory is an older proxy catalogue and is **not** the
> oracle; ignore it.

It lives under `legacy/` now so that reading is the default rather than something
you have to be told.

```
legacy/
  benchmarks/        per-op micro-benchmarks over shapes.py (linear, grouped MoE,
                     bmm, indexer score, attention). Proxy shapes, not a task suite.
  run_all.py         runs all of benchmarks/ and writes JSON + CSV
  shapes.py          MiniMax-M3 / Kimi-K2.7 / DeepSeek-V3.2 DSA shape tables
  kimi_model_config.py   Kimi-K2 / DeepSeek-V3.2 layer profiles
  scripts/           the Kimi/MiniMax bench drivers that use the above
  docs/              MiniMax-M3 operator inventories and the proxy-baseline notes
  sglang_b200_operator_backend_inventory.xlsx
                     a 63-operator GLM-5.2 / DeepSeek-V4-Pro backend inventory against
                     the nvidia/GLM-5.2-NVFP4 checkpoint. Despite naming GLM-5.2, no
                     code reads it and the live suite does not need it: glm52_ops.py
                     defines the 12 operators outright. Reference material, kept here.
```

## What is NOT here

- **GLM-5.2** — the live suite. Its 12 operators are defined once in
  `testbench/harness/glm52_ops.py`, its 24 tasks live under
  `testbench/tasks/glm52/`, and none of them touch anything in this directory.
  Start at [`AGENTS.md`](../AGENTS.md).
- **Kimi-K2.7 and MiniMax-M3 tasks** — still under `testbench/tasks/`, still served
  by `testbench/bin/evaluate.py` and the knowledge base. Those are real task suites
  with real oracles; only the *proxy* stack moved here.
- `scripts/compare_timing_methods.py` — stayed at `scripts/`, because it now reads
  `glm52_ops` and belongs to the live suite.

## Running it

Self-contained apart from the venv and `logs/`, which it borrows from the repo:

```bash
source activate_env.sh
bash legacy/scripts/run_kimi_minimax_perf.sh
```

The scripts resolve `LEGACY_ROOT` (this directory: `run_all.py`, `benchmarks/`,
`shapes.py`) separately from `REPO_ROOT` (the venv, `logs/`). Those were the same
directory until the move, so anything added here must keep them apart.
