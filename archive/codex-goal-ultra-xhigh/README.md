# codex-goal-ultra-xhigh

Curated archive of the **GLM-5.2 Codex production goal** campaign run under
`/home/qinhaiyan/glm52-goal-runs` (2026-07-22 … 2026-07-23).

| Field | Value |
|---|---|
| Runner | `~/glm52-goal-runs` (25 isolated harness + sglang worktrees) |
| Model | `gpt-5.6-sol` with `model_reasoning_effort=ultra` |
| Plans | `Kernel-Harness/goal_plans/glm52_production/` |
| Live source | left in place under `~/glm52-goal-runs/` (~20G with traces) |

## What is archived

Per goal under `goals/<NN>-<slug>/`:

| Path | Contents |
|---|---|
| `SOURCE.md` | Disposition, harness branch tip, pointers back to the live worktree |
| `candidates/` | Custom `.py` / candidate dirs (stock helpers omitted) |
| `reports/` | `REPORT.md` and sibling small markdown/json summaries from `profile/` |
| `evidence_text/` | Text evidence (`FINAL_REPORT.md`, ledgers, paired summaries, validators) |

Intentionally **not** archived: nsys/ncu binaries, sqlite traces, nested `sglang/`
checkouts, or the older nested `archive/0720-Best-GLM-52` copies inside worktrees.

## Index files

- [`CATALOG.md`](CATALOG.md) — human table of all 25 goals
- [`manifest.json`](manifest.json) — machine-readable inventory
- [`EXPERIMENT_RESULTS.md`](EXPERIMENT_RESULTS.md) — human campaign write-up (01–08, 22–25 analyzed at archive time)
- [`tasks.tsv`](tasks.tsv) — goal runner task list

## Highlights

- **leaf-win, not production-swapped:** goal-07 (DeepGEMM BM16 alignment), goal-08 (`moe_w2_contig_psum`)
- **Confirmed production replacements from this campaign:** none (at archive time)
- Earlier KDA bests remain in [`../0720-Best-GLM-52/`](../0720-Best-GLM-52/)
