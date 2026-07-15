# GLM-5.2 moe_gate_proj_prefill — Phase 1 Prompt

Develop a kernel that minimizes latency while preserving numerical correctness.
Target machine: NVIDIA B200. Allowed approaches include CUDA C++, CuTe DSL,
Triton, Python-wrapped CUDA extensions, CUTLASS, DeepGEMM retuning, or any
other path available in the Kernel-Harness `.venv`.

This is **Phase 1**: research and produce the first **correct independent**
B200 implementation. Performance matters, but correctness and a clean design
are the priority. Do not treat Phase 2/3 shape specialization as in-scope yet.

## Task Contract (Kernel-Harness)

- Task: `glm52/moe_gate_proj_prefill`
- Op: MoE Gate Projection
- Phase: prefill
- Family: grouped-moe
- Backend / baseline: deep_gemm fp8_m_grouped_gemm_nt_masked (blackwell)
- Goal: Optimize solution.py against SGLang's DeepGEMM fp8_m_grouped_gemm_nt_masked baseline for GLM-5.2 MoE Gate Projection prefill; beat every workload and match SGLang output.
- Description: GLM-5.2 MoE Gate Projection (prefill, EP8 local E=32), matching llm_flops moe_gate_proj: FP8 masked grouped GEMM out[E,Mp,2048] = a[E,Mp,6144] @ w[E,2048,6144].T. Routing = fixed-seed top-8/256 filtered to this rank; K=6144, N=2048. Quant offline; only the grouped GEMM is timed.

Authoritative task directory in this worktree:
`testbench/tasks/glm52/moe_gate_proj_prefill/`

### Axes / sweep

| Axis | Role | Value |
|------|------|-------|
| `M` | variable (sweep) | prefill token/batch population (sweep) |
| `E` | const | `32` — local experts (EP8 of 256) |
| `K` | const | `6144`  |
| `N` | const | `2048`  |
| `layout` | const | `1` — 1=masked (llm_flops prefill+decode path) |
| `n_global` | const | `256`  |
| `topk` | const | `8`  |

Sweep shapes: `[1024, 2048, 4096]`

### Correctness

- `max_atol = 0.1`
- `max_rtol = 0.05`
- `required_matched_ratio = 0.999`

Oracle = live `reference.py` (SGLang / production path). Candidate must match
on every shape.

### Performance (later phases; measure now, do not block Phase 1)

Authoritative WIN requires correct on every shape AND
`solution_us < baseline_us` on every shape (CUPTI cold-L2). Beating baseline is
**desirable but NOT required** to exit Phase 1.

## Validation Commands

Fast advisory smoke (from this worktree root; use this worktree's `.venv` via the
shared Kernel-Harness env if present, otherwise the main checkout `.venv`):

```bash
cd /home/qinhaiyan/KDA-Exp/worktrees/glm52-moe_gate_proj_prefill
PY=/home/qinhaiyan/Kernel-Harness/.venv/bin/python
PYTHONPATH=testbench $PY -m harness.profile \
  testbench/tasks/glm52/moe_gate_proj_prefill --shape 1024
```

Authoritative gate:

```bash
cd /home/qinhaiyan/KDA-Exp/worktrees/glm52-moe_gate_proj_prefill
PY=/home/qinhaiyan/Kernel-Harness/.venv/bin/python
$PY testbench/bin/evaluate.py testbench/tasks/glm52/moe_gate_proj_prefill --max-workloads 1
$PY testbench/bin/evaluate.py testbench/tasks/glm52/moe_gate_proj_prefill
```

Or: `cd testbench/tasks/glm52/moe_gate_proj_prefill && ./run.sh` (forwards to evaluate.py;
prefer the `$PY` form above so the shared `.venv` is used).

Exit codes: `0` = WIN, `1` = correct but not faster, `2` = incorrect.

## Workflow Requirements

- Consult KernelWiki for Blackwell/B200, DeepGEMM / FP8 GEMM / MoE grouped GEMM,
  CUTLASS/CuTe, Triton, TMA/TMEM/`tcgen05` as relevant to this op.
- Use `ncu-report-skill` when collecting or interpreting Nsight Compute reports.
- Record candidates in `kda/candidates.jsonl` and runs in `kda/benchmark.csv`.
- Keep NCU artifacts under `kda/profile/`.
- Do **not** copy final contest release kernels into this workspace.
- Edit only `testbench/tasks/glm52/moe_gate_proj_prefill/solution.py` for the candidate
  (do not edit `reference.py`).

## Phase 1 Goal

Research prior art and produce the first **correct independent** implementation
that can replace `solution.py` while matching `reference.py` within tolerance on
all sweep shapes `[1024, 2048, 4096]`.

Phase 1 success criterion: authoritative evaluate reports `correct=true` on all
shapes. Prefer a path that can later be optimized (not a thin re-export of the
baseline op). Calling the exact baseline kernel as the only body of `run()` is
**not** an acceptable Phase-1 deliverable.

## Plan Draft Requirement

Before implementing, write an implementation-plan draft to:

```text
kda/docs/draft.md
```

The draft must include baseline behavior + validation, risks/unknowns, ranked
candidate directions, first concrete steps, exact validation commands, and
evidence rules to promote/revise/reject.

Then convert the draft with Humanize:

```text
/humanize:gen-plan --input kda/docs/draft.md --output kda/docs/plan.md --direct
```
