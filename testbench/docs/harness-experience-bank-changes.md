# Change Summary — `harness-experience-bank` branch

**Scope:** everything on this branch relative to baseline `fad7d34` (the last commit
before the branch): **98 files, +3123/−269** (excluding `archive/`). Two authored
commits plus a concurrent harness-review/verification layer landed in the same tree.

> This doc summarizes *what changed and why*. It does not restate the contract in
> `AGENTS.md`; read that for how to use the harness.

---

## One-line theme

The harness gained a **feedback/verification spine**: it already produced excellent
proof (cold-L2 CUPTI timing, `calc_diff` oracle, anti-cheat); this branch makes that
proof **reusable** (experience bank + warm-start), **honest under contention**
(GPU-lease timing), **legible** (named terminal states + deterministic auto-check),
and **cumulative** (distill recurring lessons → promote to a durable owner).

Framed against the Harness-Engineering theses, it closes the three gaps that review
flagged: *turn feedback into infrastructure*, *prove the outcome legibly*, and
*route context just in time*.

---

## 1. Experience bank — build & reuse *(committed: `d53a727`, `773e929`)*

| Area | Change | Files |
|---|---|---|
| Warm-start | `brief.py <task>` — one call: best prior run (`runs/`), internal recipes, library ledger, KernelWiki prior-art, recurring pitfalls, roofline | `testbench/bin/brief.py` |
| Union query | `knowledge.py brief` (internal + ledger + external) and `index` (deterministic `queries/*.md`, CI `--check`) | `testbench/bin/knowledge.py` |
| External bridge | `kwiki_bridge.py` — best-effort read-only bridge to KernelWiki; degrades to internal-only if absent | `testbench/bin/kwiki_bridge.py` |
| Library-first ledger | `candidates.json` — per-op best library drop-in, call signature, scale/layout contract | `testbench/knowledge/candidates.json` |
| Shared vocabulary | `vocab.md` — bottleneck→KernelWiki symptom map, technique tags | `testbench/knowledge/vocab.md` |
| **Feedback loop** | `knowledge.py distill` (mine recurring failure-classes + proven techniques → `distilled.{json,md}`) and `promote` (append-only `promotions.json` ladder: prompt→doc→reviewer→diagnostic→typed-boundary→lint). `brief` surfaces distilled pitfalls at warm-start. | `knowledge.py`, `testbench/knowledge/distilled.*`, `promotions.json` |
| Shared bank | `$KH_KNOWLEDGE_ROOT` relocates the bank so sibling worktrees share one | `knowledge.py` |
| Entry point | `AGENTS.md` points every session at `brief.py` so retrieval is load-bearing | `AGENTS.md` |

*Effect:* a session now starts from the frontier (best prior run + what was tried & why
+ already-tried dead-ends for its bottleneck) instead of rediscovering it, and ends by
writing a recipe the next session reads. On the current 12-entry bank `distill` already
finds 3 recurring dead-ends (e.g. `explicit-deepgemm-recipe` failing the scale-shape
assertion across 2 entries → suggests a `typed-boundary|diagnostic` owner).

## 2. Honest timing under contention *(committed: `d53a727`)*

| Change | Files |
|---|---|
| `--auto-gpu` picks the least-busy GPU via `nvidia-smi`; a per-GPU `flock` wraps the timed sweep so parallel gate runs can't pollute device-span medians (opt out `--no-gpu-lock`). Timing math unchanged. | `testbench/harness/gpu_lease.py`, `evaluate_task.py` |
| `reward(attainable_bw=…)` — additive achievability fields (`bw_util_attainable`, `attainable_frac_of_peak`); a pure-copy `bw_ceiling.py` measures the physical roof for a constructive-proof NO-GO. Existing `reward` number unchanged. | `glm52_ops.py`, `testbench/bin/bw_ceiling.py` |

## 3. Verification & legibility layer *(concurrent, uncommitted at time of writing)*

This is a separate, complementary body of work that landed in the same tree:

| Change | What it does | Files |
|---|---|---|
| **Named terminal states** | The verdict contract restated as `≥1 win, no regression` plus explicit `COMPLETE_WIN` / `NO_WIN_WITH_EVIDENCE` / `PARTIAL_OR_REGRESSED_WITH_EVIDENCE` / `INCORRECT_OR_INCOMPLETE`, regenerated onto all 24 tasks from the `glm52_ops` source of truth | `glm52_ops.py`, all `tasks/glm52/*/{task,problem}.json` + `README.md`, `sync_glm52_tasks.py` |
| **Deterministic auto-check** | `audit_result.py` — reviewer-side cheap pass (no CUDA): verifies a `result.json` is internally consistent, names the exact candidate bytes (sha), applies current gate semantics, classifies dirty-tree provenance | `testbench/bin/audit_result.py` |
| **Provenance in the record** | `result_store.py` → schema `1.3`: git-status capture (`candidate_git_state`, `dirty`/`git_dirty`), atomic writes | `testbench/harness/result_store.py` |
| **One-command review** | `verify_harness.py` — GPU-free CI: selftest + knowledge lint + `index --check` + **`distill --check`** + task-sync + audit sweep; `VERIFY.md` is the review handoff | `verify_harness.py`, `testbench/VERIFY.md` |
| **Env robustness** | `check_env.py` (+97) and `setup_env.sh` (+80) hardening | those files |
| **Real kernel wins** | Actual optimized candidates landed: `o_proj_decode` (M=16 owned Triton FP8 block-scaled GEMM, M=32 deliberate fallback), `index_k_proj_decode`, `o_proj_prefill` | `tasks/glm52/*/candidate.py` |

> Note: the concurrent verification lane **already consumes the experience-bank work** —
> `VERIFY.md`/`verify_harness.py` run `knowledge.py distill --check` and `index --check`
> as part of the standard review. The two halves are integrated.

---

## How the pieces line up with the review gaps

- **Turn feedback into infrastructure** → §1 `distill` + `promote` ladder + pitfalls-in-brief.
- **Prove the outcome legibly** → §3 named terminal states + `audit_result.py` (auto-check) + provenance in `result.json`.
- **Route context just in time** → §1 `brief.py` warm-start + `$KH_KNOWLEDGE_ROOT` shared projection.
- **Honest measurement** (proof integrity) → §2 GPU-lease flock + attainable-BW ceiling.

## Verification status

- **GPU-free, verified:** `selftest` 24/0, `knowledge lint` 12/0, `index --check` 0 stale,
  `distill --check` up to date, task-sync in agreement, `verify_harness.py` lane green.
- **Needs a GPU node + venv (unverified here):** end-to-end gate under `--auto-gpu` + the
  timing flock, `bw_ceiling.py --sweep`, the `reward(attainable_bw=…)` fields, and the
  landed `candidate.py` kernels. See `testbench/VERIFY.md` for the GPU lane.

## Commit / ownership status

- Committed & pushed to `origin/harness-experience-bank`: `d53a727` (bank + GPU-lease) and
  `773e929` (distill + promotion ladder) — the §1/§2 files only.
- §3 (verification layer, kernels) was uncommitted concurrent work in the tree at the time
  this summary was written; it is authored separately and not part of the two commits above.
