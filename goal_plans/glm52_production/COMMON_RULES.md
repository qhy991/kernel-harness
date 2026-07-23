# Shared rules for GLM-5.2 production optimization goals

Read this file before executing any sibling `plan.md`.  It is part of every
goal's objective, constraints, and definition of done.

## Outcome hierarchy

Optimize the implementation reached by the current SGLang serving configuration,
not merely an operator with similar mathematics.  Evidence has this priority:

1. A current SGLang code trace plus a short runtime trace proving the callable,
   dtype, layout, local shape, graph mode, and topology actually reached.
2. The matching `serving_native/` workload, whose reference invokes that
   production symbol with production inputs.
3. A frozen synthetic task used only to develop a candidate or isolate a kernel.

If the old logical operator is fused, renamed, skipped, or handled by a different
backend, pivot to the reachable production operator.  Do not optimize a dead path
to satisfy the old task name.

## Fixed deployment contract

- Target hardware is NVIDIA B200/SM100.
- Decode local `M` buckets are independently fixed at `16` and `32`, including
  DP8.  Never divide these values by data-parallel world size.
- The default balanced prefill test point is local `M=4096` for DP8.  If a live
  trace proves another bucket is important, add a separately named workload;
  do not silently change an existing workload.
- Production distributed validation uses one node with TP8/DP8/EP8.  The four-GPU
  TP4/DP4/EP4 lane is diagnostic and must be reported separately.
- FP8 linear and grouped-GEMM inputs use the production packed `int32` UE8M0 scale
  ABI.  A timed float32-to-packed or packed-to-float32 adapter is not a deployable
  optimization unless the live caller itself produces that representation.
- DSA prefill and decode backends are resolved independently.  On current SM100
  code the no-flag FP8 default may be TRT-LLM, but an explicit
  `--dsa-decode-backend flashmla_kv` or a trace containing
  `flash_fwd_splitkv_mla*` selects `sgl_kernel.flash_mla`.  Freeze the backend
  from the exact production launch and trace; never substitute one DSA backend
  for another under the same benchmark name.
- The CUDA indexer uses `wq_b` plus fused BF16 `wk_weights_proj`; do not substitute
  the old independent K or weights projection without a reachability proof.
- MoE uses fused W13, fused SwiGLU+quant, W2, and DeepEP.  Preserve all recipe,
  signal, stream, overlap, return-value, and graph-capture contracts.

Shapes are intentionally fixed for these tests.  Do not broaden a goal into a
dynamic-shape project unless a runtime trace shows the fixed buckets are wrong.

## Shared GPU scheduling

Codex source analysis and builds must not reserve a GPU.  For a single-GPU CUDA
command, use `/home/qinhaiyan/glm52-goal-runs/with_flexible_gpu.sh -- <command>`;
it selects any idle physical B200 and exposes it as logical GPU 0 only for the
duration of that command.  Put a complete alternating reference/candidate series
and its profiler collection into one script and one wrapper invocation so paired
measurements remain on the same physical GPU.  Record the printed physical GPU,
UUID, clocks, and environment with the result.  Never compare separate unpaired
runs from different GPUs as a performance claim.

Four-rank diagnostics must use
`/home/qinhaiyan/glm52-goal-runs/with_all_gpus_lock.sh <command>`.  This reserves
physical GPUs 0-3 only for that distributed command and temporarily prevents new
single-GPU measurements from starting.  Exit status 75 means resources are busy:
continue CPU-only work and retry later rather than bypassing the scheduler.

## Authorized edit scope

Source-level optimization is explicitly allowed.  The goal may modify:

- `/home/qinhaiyan/sglang`, including dispatch, integration, SGLang kernels, and
  tests relevant to the target operator;
- an isolated, pinned source checkout or overlay of DeepGEMM, FlashInfer, DeepEP,
  sgl-kernel, CUTLASS/CuTe, Triton, or another directly reached open-source kernel
  library;
- an external Kernel-Harness candidate passed with `--candidate`;
- `serving_native/` only to add a new, explicitly named production workload or
  candidate needed by the goal, with its structural tests updated.

Do not overwrite installed packages in place.  Record the upstream repository,
base commit, local commit, build command, artifact path, and import resolution.
Keep the stock implementation available as the reference and fallback.

Do not edit the frozen synthetic oracle, timing code, correctness rules, generated
task metadata, workloads, or historical knowledge entries.  Existing user changes
in either repository must be preserved.  Use a clean dedicated worktree for each
goal and never reset or discard unrelated changes.

## Required execution loop

1. **Lock reachability.** Record the SGLang entry point, concrete source symbol,
   selected backend, input/output ABI, local shape, graph/eager mode, stream, and
   distributed world size.  Confirm it with a short trace or equivalent runtime
   evidence.  Absence of a Python NVTX range during graph replay is not proof of
   absence; use kernel names, hit counters, and code mapping together.
2. **Freeze the reference.** Run with `SGLANG_GLM52_OPT=0`.  Capture repository
   SHAs, package/import paths, environment, clocks, topology, and at least three
   uncontended baselines.  Distributed latency is the maximum across ranks.
3. **Make the production microbenchmark exact.** Prefer an existing
   `serving_native` workload.  If none exists, add a new named workload that calls
   the exact symbol and preserves the live ABI.  Keep the synthetic task unchanged.
4. **Characterize before rewriting.** Use Nsight Systems for launch gaps,
   synchronization, communication, and overlap.  Use Nsight Compute for the
   dominant kernel when the proposed change is device-code level.  Record ptxas
   resources and inspect PTX/SASS when register pressure, spills, instruction mix,
   vectorization, or tcgen05/TMEM scheduling is part of the hypothesis.
5. **Run one evidence-backed experiment at a time.** Each attempt must record:
   hypothesis; baseline evidence; exact code/config delta; expected PTX/SASS or
   runtime effect; correctness result; paired p50 and distribution; profiler delta;
   risk; decision; and rollback point.  Preserve negative results.
6. **Build a per-bucket oracle.** Compare reference and candidate in alternating
   order in the same session.  Treat a paired p50 gain below 3% as noise unless a
   tighter local noise floor is demonstrated.  Enable only winning operator × M ×
   ABI × topology buckets.  A device-to-host read or synchronization for dispatch
   is forbidden.  Losing M16 or M32 buckets must remain on stock SGLang.
7. **Integrate without adapter tax.** The SGLang path must accept the tensors the
   caller already has, avoid extra allocation/copy/pack kernels, preserve output
   and stream semantics, work in CUDA Graph replay where required, and fail closed
   to the stock implementation for unsupported cases.
8. **Validate the containing region.** A microbenchmark win is necessary but not
   sufficient.  Test the layer/region and complete SGLang workload.  For MoE this
   is `DeepEP dispatch -> W13 -> SwiGLU+quant -> W2 -> DeepEP combine`, including
   overlap.  For DSA/indexer this includes cache preparation, score/top-k, and the
   selected attention backend.  For collectives use the real rank count and graph
   mode used by serving.
9. **Promote or decline.** Promote only buckets that pass correctness, paired
   microbenchmark, graph/overlap, and end-to-end gates.  Otherwise leave the stock
   path active and finish with an evidence-backed no-replacement result.

## Standard commands

Run from `/home/qinhaiyan/Kernel-Harness` unless noted:

```bash
.venv/bin/python serving_native/selftest.py
serving_native/run.sh --list
serving_native/run.sh --describe <workload>
serving_native/run.sh <workload> --candidate <candidate.py-or-directory>
python3 testbench/bin/verify_harness.py
```

For a related frozen task, first run `run.sh --describe`, then `run.sh` with an
external candidate.  A frozen-task win is development evidence only; audit its
`result.json` before citing it.

For eight-rank communication:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 serving_native/run.sh <workload> --candidate <candidate>
```

For the independent four-rank diagnostic lane:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 serving_native/run.sh <tp4-or-ep4-workload> --candidate <candidate>
```

## Definition of done

A goal completes with exactly one honest disposition:

- **Production win:** at least one named bucket improves paired p50 by at least 3%,
  no enabled bucket regresses, correctness passes, graph/overlap semantics pass,
  the containing region and SGLang end-to-end metric improve, and stock fallback
  remains available for all other buckets.
- **No replacement:** reachability, baselines, profiler evidence, and at least one
  justified source or configuration attempt show no deployable gain.  The stock
  path remains enabled and the report explains the binding limit and rejected
  attempts.  A faster synthetic result alone cannot change this disposition.

Required deliverables are the source diff, tests, raw benchmark outputs, paired
summary table, profiler artifacts, attempt ledger, exact enable/fallback policy,
and a concise final report.  Do not claim completion from a single favorable run,
an unpaired comparison, a four-GPU result for an eight-GPU deployment, or an
isolated kernel win that makes the full region slower.
