# Task 26 `em8_bm16_stage11` v3 pre-GPU CPU review

Status at `2026-07-28T21:10:58Z`: ready for independent source/CPU review.
No overlay build, DeepGEMM JIT, CUDA command, profiler, run root, or persistent
one-attempt sentinel has been created for this variant.

## Strict-profile config follow-up

CPU-only follow-up at `2026-07-28T21:17:30Z`:

- SGLang strict config boundary:
  `887721602bbc0af3b1a80b4097b1b4ac72e9a094`
  (`Fail closed on stage11 config load errors`)
- Kernel-Harness review base:
  `2ef28964318e959e1267b59e59348e7da1a73865`
- Explicit `moe_w2_em8_bm16_stage11` and canonical
  `moe_w2_em8_bm16_stage11_v3` config-import failures now propagate before
  stock fallback. Predicate-evaluation failures do the same for the canonical
  strict profile. Non-stage11 import/evaluation failures retain best-effort
  stock behavior.
- The combined SGLang W2 contract launcher passed `78` tests with
  `CUDA_VISIBLE_DEVICES=''`. Python compilation and repository diff checks
  passed.

This follow-up did not build the overlay, initialize CUDA, create a v3 run
root, or claim the persistent one-attempt sentinel. A fresh independent
source/CPU release is still required.

## Reviewed source heads

- Kernel-Harness source/driver head:
  `b72cc7327ae4bc3f5c0d1ce4c864692c6996c32a`
- SGLang integration/overlay head:
  `f4945adfa521658799b5e5a478e1e608ceec4495`
- authoritative DeepGEMM base:
  `edcf77b276965de8f03cdc47c23f01b08bf7c7ab` (`v0.1.4.post1`)
- CUTLASS:
  `f3fde58372d33e9a5650ba7b80fc48b3b49d40c8`
- fmt:
  `553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28`
- runtime `source.patch` SHA-256:
  `26fbaca849eedb1788e3a1bd70e72ea7eb3332936920c9686fc939b39715e01f`
- separate `build_tool.patch` SHA-256:
  `dc731d5442c0bdf0758b17380e02e67b580cf3aa579f4832a497d1b68e3a85c7`
- core-source hash-list SHA-256:
  `92d01a9de3c31d3c1273d8435645961d3bddcee8cfed42b0e95cb589837a4164`
- prospective build key:
  `edcf77b27696-26fbaca849ee-dc731d5442c0`

## CPU gates

- `python3 testbench/bin/verify_harness.py --skip-task-projection`: PASS,
  including 85 Task26 contract/driver tests, 46 serving-native workloads,
  task self-test, knowledge checks, compile checks, and diff checks. The
  device-backed generated-task projection was skipped because CUDA remains
  locked; running that projection CPU-only omits tensor tables and reports the
  generated task documents stale.
- SGLang contract launcher with `CUDA_VISIBLE_DEVICES=''`: 73 passed,
  22 warnings.
- Fresh exact-base source proof: PASS for `git apply --check`, actual apply,
  `git diff --check`, binary re-diff equality, core hashes, exact base, and
  exact submodule SHAs. The ignored working source diff and every core file
  also match the tracked patch/hash list byte-for-byte.
- Python compile, shell syntax, focused Ruff lint/format, and repository diff
  checks: PASS.
- Root free space at review: `17,606,172 KiB`, above the 8 GiB stop line.

## Fail-closed experiment contract

Only exact normal decode local M32/`expected_m=8` may launch the candidate,
with per-call `(masked_block_m_override=16,
masked_num_stages_override=11)`, packed-int32 UE8M0, E32/slab1024/K2048/N6144,
SM100/148 SMs, PDL true, equal stock/candidate `tc_util`, and no
recipe/overlap. The zero-override source path retains stock max-stage
selection and does not execute stage11-only bounds or memory assertions.

The runner and independent auditor recompute pooled, order-balanced, AB
median, and BA median speedups from raw ordered samples. Every estimator in
every series must be finite and at least 1.03; the paired median remains
diagnostic. Candidate leaf/graph evidence must prove BM16/stage11 and forbid
BM128/stage12 or any extra kernel, while stock must prove BM128/stage12.

The one-attempt driver is executable and contains exactly three lanes: leaf
eager, leaf CUDA Graph, and containing-region eager. It requires clean exact
heads, the wrapper lease FD, the campaign lock, one B200 identity, fresh v3
caches/output, and an unclaimed persistent sentinel. Stage10 is only a
predeclared, ineligible fallback.

The consumed stage12 attempt `20260728T182705Z` remains failed evidence and is
not reused or reinterpreted. Do not build, JIT, or launch this v3 experiment
until an independent audit explicitly releases the exact clean heads.
