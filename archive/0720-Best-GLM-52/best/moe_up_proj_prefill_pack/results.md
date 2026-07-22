# moe_up_proj_prefill_pack — results

Dual-protocol results for the GLM-5.2 prefill MoE Up Projection packed-UE8M0 + PDL candidate,
ported from `best/moe_up_proj_decode_hbm40`. B200, GPU 2 (verified idle), M ∈ {1024,2048,4096}.

| Shape  | CUPTI cold-L2 (harness gate) | CUDA Graph drop-in (median, %≥1.0/30) | packed-GEMM floor (ref/floor) | correct |
|--------|------------------------------|----------------------------------------|-------------------------------|---------|
| M=1024 | **1.184×**                   | **1.165×** (100%)                      | 1.239×                        | diff 0  |
| M=2048 | **1.105×**                   | **1.085×** (97%)                       | 1.141×                        | diff 0  |
| M=4096 | **1.058×**                   | **1.049×** (97%)                       | **1.084× (< 1.10× — wall)**   | diff 0  |

- Harness CUPTI gate: `3/3 WIN, 0 regressed`, geomean 1.115×, `status=CORRECT`, exit 0.
- **Graph-safe on all shapes** (the key differentiator vs `moe_gate_proj_prefill_pack`, which
  regressed to ~0.95× at M=4096).
- **AC2 M=4096 is a physical wall.** The gate-faithful packed-GEMM-only floor (candidate
  latency if the pack were free, `candidate_loader.resolve` + `clone_inputs`, cand_us matches
  the gate) is 1.084× at M=4096 — even a free pack cannot reach 1.10×. Closing it needs a
  faster grouped GEMM (Forbidden). Kept here (AC5 branch 2: keep-best + document), not
  default-swapped.

Kept under `best/` as the correct, Graph-safe prefill port. Layer-table promotion is a separate
layer-owner decision (not done here). Full evidence, scripts and raw logs live in the
KDA-Pilot-Exp worktree: `llm/glm52_harness_moe_up_proj_prefill_pack/docs/results.md` and
`.humanize/rlcr/2026-07-20_06-13-14/evidence/`.
