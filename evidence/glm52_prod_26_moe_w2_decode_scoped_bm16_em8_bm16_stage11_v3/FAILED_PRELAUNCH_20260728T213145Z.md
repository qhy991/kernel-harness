# Task 26 stage11-v3 failed pre-launch attempt

## Immutable identity

- UTC claim: `2026-07-28T21:31:45Z`
- run root:
  `/home/qinhaiyan/glm52-v2-goal-runs/worktrees/26-moe-w2-decode-scoped-bm16/kernel-harness/runs/glm52_prod_26_moe_w2_decode_scoped_bm16_em8_bm16_stage11_v3/20260728T213145Z`
- persistent sentinel:
  `/home/qinhaiyan/glm52-v2-goal-runs/cache/26-moe_w2_decode_scoped_bm16/em8_bm16_stage11_v3/ONE_ATTEMPT_CONSUMED`
- Kernel-Harness:
  `f16f401fe10e37da6113b04cfe6307fb3b5dba6a`
- SGLang:
  `fca51ff68ffcf9234ff4fa11e548921eb54fe98c`
- leased physical GPU: index `0`,
  UUID `GPU-30b619de-87f2-1862-0d07-a595da8fe417`

The claim and failure records remain in place. They must not be deleted,
rewritten, or reused.

## Failure

The first declared lane,
`moe_w2_grouped_decode_m32_em8_bm16_stage11__eager`, stopped in
`run_with_exact_post1_stock.sh` while verifying:

`build/deepgemm-w2-em8-bm16-stage11-v3-overlays/edcf77b27696-26fbaca849ee-dc731d5442c0/manifest.json`

The manifest did not exist. `overlay_manifest.py verify` raised
`FileNotFoundError` before the runner process was executed.

## What did not run

- no stock or candidate DeepGEMM overlay was built;
- no DeepGEMM/Triton/Torch-extension JIT cache entry was created;
- no CUDA runtime or candidate module was imported by the runner;
- no stock or candidate GPU kernel launched;
- no correctness check, warmup, AB/BA pair, profiler capture, result JSON, or
  independent result audit ran.

The task-local cache contains only `ONE_ATTEMPT_CONSUMED/{CLAIMED,FAILED}`.
The run root contains the environment/GPU identity, initial snapshot, failure
marker, and the manifest traceback.

## Disposition

This is preserved as a mandatory failed preparation attempt, not a performance
result and not a no-op identity control. Production remains default-off. Any
future work must keep this v3 evidence immutable and must receive an
independent fairness decision before defining a materially corrected,
separately versioned build/launch flow.
