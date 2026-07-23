# Archive replay — 2026-07-23 · lichangye + huyan candidates vs current baseline

Test run replaying the two archived MI300X candidate collections through the
current kernel-harness gate, on the current sglang/aiter production dispatch
baseline. Answers the question **"do any of these archived kernels still win
against what the harness dispatches today?"**

**One-line result: 2 of 17 (candidate, shape) combos are real wins** —
`huyan/o_proj_prefill @ M=1024` (1.315x) and `@ M=2048` (1.196x). Everything
else is neutral, regressed, incorrect, or crashed. See `results.csv`.

---

## What was tested

Twelve archived candidate files, spanning six operators:

| Archive path | Author | Files | Ops covered |
|---|---|---|---|
| `archive/0720-Best-GLM-52/lichangye/` | lichangye | 4 candidate.py (one per op) | `dsa_prefill_attn`, `index_score_prefill`, `moe_total_prefill`, `moe_total_decode` |
| `archive/0723-amd-glm52/` | huyan | 8 wrappers over `_amd_kernels.py` (per-shape) | `dsa_attn_decode` (BS 16/32), `index_k_prefill` (M 1024/2048/4096), `o_proj_prefill` (M 1024/2048/4096) |

Sweep = every shape in each task's `workload.jsonl` — 17 unique (candidate,
shape) rows in `results.csv`.

## Hardware & backend

| Field | Value |
|---|---|
| Node | 8× AMD MI300X (gfx942, ROCm 7.x); this replay used **GPU 0 only** (single-process, no TP) |
| Platform bundle | `rocm / amd-mi300x / aiter-torch-reference / event` |
| Reference kernel | Whatever `glm52_ops.reference(op, phase, inputs)` dispatches to on this bundle — for every op in this set that is the aiter production path SGLang would dispatch at runtime (see `rewardbench/amd/operator_mapping.md` §"harness baseline ↔ sglang 生产 dispatch 对照") |
| Python | `/root/venvs/rocm-torch/bin/python` (repo `.venv` fallback if unset) |
| FP8 dtype | `torch.float8_e4m3fnuz`, `FP8_MAX=224.0`; no UE8M0 |

## Timing protocol

For each shape, the harness `testbench/harness/evaluate_task.py` does:

1. **Correctness gate first** — build inputs, poison the shared `out` buffer,
   run reference + clone, run candidate, apply the FlashMLA three-layer check:
   1. inf / -inf / nan positions match,
   2. every element passes `abs_err < abs_tol` OR `rel_err < rel_tol` (per-op tol, in `problem.json`),
   3. aggregate `calc_diff <= diff_tol` (DeepGEMM's `calc_diff`, verbatim).
   Any layer fails → verdict `INCORRECT` and timing is skipped.
2. **Timing** — same frozen inputs, HIP graph capture+replay median
   (`Timer.id = hipgraph-or-event-median` on rocm/amd-mi300x), cold L2 flush
   between iterations, `repeat=3 iterations=10 warmup=3` for this replay
   (lower than the 10/30 default because we're screening 12 candidates).
3. **Post-timing correctness re-check** on fresh inputs (catches a kernel
   that mutates its inputs or drifts).
4. **Reward** = bound-aware roofline utilisation:
   `achieved_flops / min(peak_flops, ai × HBM_peak)`; `bw_util` = `bytes/s /
   HBM_peak` (interconnect peak instead of HBM for comm-family; not used
   here).

Verdict per shape (from `task.json.performance_gate`):

| verdict | rule |
|---|---|
| **WIN** | `reference_p10 / candidate_p90 > 1.0` — ahead even on the reading least favourable to the candidate |
| **REGRESS** | `reference_p90 / candidate_p10 < 1.0` — behind even on the reading most favourable to the candidate |
| **neutral** | inside the noise band; does not veto the run |

Terminal state per candidate (aggregate across all shapes):

| terminal_state | meaning |
|---|---|
| `COMPLETE_WIN` | correct sweep, ≥1 win, 0 regressions → exit 0 |
| `NO_WIN_WITH_EVIDENCE` | correct sweep, all shapes neutral → exit 1 |
| `PARTIAL_OR_REGRESSED_WITH_EVIDENCE` | correct sweep, at least one shape regressed → exit 1 |
| `INCORRECT_OR_INCOMPLETE` | correctness failed, sweep incomplete, or post-timing re-check failed → exit 2 |

## How the tests were invoked

Every candidate file is a `.py` defining `run(inputs: dict) -> tensor`.
Passed to the harness via `./run.sh --candidate <abspath>` on the matching
task directory:

```bash
# lichangye — 4 candidates, full sweep
testbench/tasks/glm52_amd/dsa_prefill_attn/run.sh \
  --candidate archive/0720-Best-GLM-52/lichangye/dsa_prefill_attn/candidate/candidate.py \
  --repeat 3 --iterations 10 --warmup 3 --no-gpu-lock

testbench/tasks/glm52_amd/index_score_prefill/run.sh \
  --candidate archive/0720-Best-GLM-52/lichangye/index_score_prefill/candidate/candidate.py \
  --repeat 3 --iterations 10 --warmup 3 --no-gpu-lock
# ... (moe_total_prefill, moe_total_decode)

# huyan — 8 wrappers, each pinned to one shape via --M
testbench/tasks/glm52_amd/o_proj_prefill/run.sh \
  --candidate archive/0723-amd-glm52/o_proj_prefill_m1024.py \
  --M 1024 --repeat 3 --iterations 10 --warmup 3 --no-gpu-lock
testbench/tasks/glm52_amd/o_proj_prefill/run.sh \
  --candidate archive/0723-amd-glm52/o_proj_prefill_m2048.py \
  --M 2048 --repeat 3 --iterations 10 --warmup 3 --no-gpu-lock
# ... (m4096, index_k_prefill × 3M, dsa_attn_decode × 2 BS)
```

`--no-gpu-lock` skips the multi-tenant timing flock (single-tenant replay on
GPU 0). Env exports the run.sh handles automatically:

```
KERNEL_HARNESS_PLATFORM=rocm
KERNEL_HARNESS_PROFILE=amd-mi300x
KERNEL_HARNESS_PROVIDER=aiter-torch-reference
KERNEL_HARNESS_TIMER=event
SGLANG_USE_AITER=1
```

Full driver script: was written to `/tmp/run_archive_tests.sh` for this
replay; kept ad-hoc since it's a one-off audit. Per-log outputs went to
`/tmp/archive_test_<label>.log`.

## Result columns in `results.csv`

| column | meaning |
|---|---|
| `candidate_source` | `huyan` or `lichangye` |
| `candidate_file` | repo-relative path to the candidate `.py` |
| `task` | `glm52_amd/<task>` — which task the candidate was tested against |
| `M` | the M value in the task's sweep (empty for TIMEOUT — no shape data) |
| `phase` | `prefill` or `decode` |
| `correct` | `PASS` / `FAIL` / `ERROR` / `TIMEOUT` |
| `calc_diff` | DeepGEMM's aggregate `1 − 2⟨x,y⟩/(‖x‖² + ‖y‖²)`; empty when correctness didn't complete |
| `cand_us`, `ref_us` | median candidate / reference latency in μs |
| `speedup_median` | `ref_us / cand_us` |
| `speedup_conservative` | `ref_p10 / cand_p90` — verdict-driving figure |
| `reward` | candidate's bound-aware roofline utilisation |
| `ref_reward` | reference's roofline utilisation at the same shape (context — a low candidate reward vs a low ref_reward means the op is at its own roof, not that the candidate has headroom) |
| `verdict` | per-shape verdict: WIN / neutral / REGRESS / INCORRECT / ERROR / TIMEOUT |
| `note` | one-line reason (regress magnitude, cosine value on FAIL, error type) |

## Findings

| bucket | count | which |
|---|---:|---|
| **Real wins (>10%)** | **2 shapes** | `huyan o_proj_prefill @ M=1024` (**1.315x**), `M=2048` (**1.196x**) — correctness bit-close (calc_diff 2e-10) |
| Marginal wins (<1%, in noise) | 2 | `lichangye moe_total_decode @ M=16` (1.008x), `M=32` (1.005x) |
| Neutral (correct, no speedup) | 3 | `lichangye index_score_prefill` @ all 3 M |
| **Regressed (slower than baseline)** | 2 | `lichangye dsa_prefill_attn @ M=1024` (**0.263x**), `M=2048` (**0.233x**) |
| Correctness failed | 7 | `huyan index_k_prefill` × 3M (cosine 0.002, wrong direction), `huyan o_proj_prefill @ M=4096` (cosine 0.0003), `lichangye dsa_prefill_attn @ M=4096` (1/134M elem past tol) |
| Crash / hang | 3 | `huyan dsa_attn_decode @ BS=16/32` (IndexError — inputs schema mismatch), `lichangye moe_total_prefill` (>400s hang) |

Interpretation notes:

- **`huyan o_proj_prefill` M=4096 fails correctness** even though M=1024 and
  M=2048 pass. Likely the kernel selects a config that hits a bug at the
  bigger tile size. If author fixes M=4096, this becomes a 3-shape sweep win.
- **`lichangye dsa_prefill_attn` regressed 3.8-4.3x** because the current
  baseline is now `aiter sparse-MLA production` (via `aiter-torch-reference`
  provider); lichangye's candidate was tuned against a slower older reference.
  Not a bug — a baseline change.
- **`huyan index_k_prefill` cosine 0.002** means the output is
  algorithmically orthogonal to the reference — not a scale/quant error, a
  layout or indexing bug. The archive predates the platform split; may
  assume a different inputs schema than the one `glm52_ops_amd.build_inputs`
  now produces.
- **`huyan dsa_attn_decode` IndexError** — same class of issue: the wrapper
  indexes `inputs["indices"]` (or similar) with a dimension the harness no
  longer produces. Post platform-split, MLA indices are `int64 [M, tk]` on
  AMD (2-D), not `int32 [M, 1, tk]` (3-D, CUDA).

## What to do next

- **Ship the win**: port `huyan/o_proj_prefill_m1024.py` + `m2048.py` into
  `testbench/tasks/glm52_amd/o_proj_prefill/candidate.py` as a shape-branched
  default (M=1024/2048 use the tuned kernel; M=4096 falls back to
  `glm52_ops.reference` until huyan fixes the M=4096 correctness). That's a
  real end-user speedup on GLM-5.2 prefill compute-bound o_proj.
- **Retire or refactor**: everything else in these two archives needs the
  author to re-tune against the current baseline (aiter production) or fix
  their inputs schema assumptions. Leaving them archived is fine; they're
  historical evidence of what beat a *previous* baseline.

## Reproduce

```bash
cd /root/repos/kernel-harness
export ROCM_TORCH_VENV=/root/venvs/rocm-torch
export HIP_VISIBLE_DEVICES=0

# One row of results.csv — huyan/o_proj_prefill @ M=2048 in this example
testbench/tasks/glm52_amd/o_proj_prefill/run.sh \
  --candidate archive/0723-amd-glm52/o_proj_prefill_m2048.py \
  --M 2048 --repeat 3 --iterations 10 --warmup 3 --no-gpu-lock
# exit 0 = WIN · 1 = correct but not faster · 2 = incorrect · 3 = infra
```

The full RESULT_JSON is persisted under `runs/glm52/<task>/<run_id>/result.json`
by the harness — that's the source-of-truth per row; `results.csv` is the
compacted summary.
