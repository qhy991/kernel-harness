# Task 26 v4 CPU provenance remediation

Date: 2026-07-28

This checkpoint is source- and CPU-only. No CUDA query, JIT compilation,
NVCC invocation, GPU lease, attempt claim, timing sample, or promotion
decision was made.

## Blocking audit finding

The pre-build audit rejected the original v4 READY design because its package
record bound only `__init__.py`, `VERSION`, and `_C.so`. The staged DeepGEMM
packages also contain Python/JIT sources and DeepGEMM/CUTLASS headers that can
affect the generated runtime kernel. Those files were therefore mutable
outside the READY identity.

## Remediation

- Manifest schema v6 and provenance schema v5 bind the complete stock and
  candidate package trees.
- Every directory and regular file is bound by relative POSIX path and
  permission mode; every regular file is additionally bound by byte count and
  SHA-256.
- Symlinks, hardlinks, and special files are forbidden.
- Build staging removes bytecode caches and makes both package trees
  read-only before hashing.
- The manifest generator and independent READY verifier recompute identical
  tree records.
- The SGLang runtime checks both READY tree identities against the verified
  manifest before stock import or any CUDA query.
- Kernel-Harness requires the two runtime tree identities and independently
  closes the package-tree schema, policies, counts, paths, modes, file hashes,
  and canonical tree digest.
- The clean-head release audit raised the timed portfolio from 10 to exactly
  50 paired A/B measurements per series. The canonical workload and auditor
  enforce this value across three series and all four eager/graph,
  leaf/containing-region lanes.

On a historical exact-post1 staged package, both independent tree walkers
produced the same SHA-256:

```text
entries: 952
files: 878
directories: 74
file bytes: 29706691
tree SHA-256: 2c3126a28901b7c2fa66083378239c3fc971f4f3e84c53251202d586c230492c
```

## CPU validation

- Kernel-Harness focused suites: 67 passed, including 93 subtests.
- The campaign-wide CPU verifier now includes the v4 candidate, driver test,
  and evidence path; all 96 included unit tests passed.
- SGLang Task 26 v4 focused suites: 64 passed.
- Earlier combined SGLang Task 26 contract sweep: 147 passed.
- Ruff, Python bytecode compilation, shell syntax, and `git diff --check`
  passed for the modified files.
- Adversarial tests cover JIT-source and header mutation, file addition and
  deletion, file and directory mode changes, symlinks, hardlinks, malformed
  policies/counts, replay drift, provenance drift, source drift, and dirty
  repositories.

## Gate state

The clean-head source release audit passed after closing the 50-pair and
campaign-verifier coverage findings. It verified the exact M32/expected-M8
packed-UE8M0 ABI, independent exact-post1 stock/candidate runtimes, stage
12-versus-11 generated identities, no fallback after candidate admission,
four required eager/graph and leaf/containing-region lanes, strict four-
estimator gates, and READY verification before GPU inspection, run-root
creation, or attempt claim.

The two-phase CPU-only build is released. READY does not yet exist, and GPU
execution remains forbidden until the built bundle and fresh source replay are
inspected, generated provenance is committed, both repositories are clean,
and READY is published and independently verified.
