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

- Kernel-Harness focused suites: 66 passed, including 93 subtests.
- SGLang Task 26 v4 focused suites: 64 passed.
- Earlier combined SGLang Task 26 contract sweep: 147 passed.
- Ruff, Python bytecode compilation, shell syntax, and `git diff --check`
  passed for the modified files.
- Adversarial tests cover JIT-source and header mutation, file addition and
  deletion, file and directory mode changes, symlinks, hardlinks, malformed
  policies/counts, replay drift, provenance drift, source drift, and dirty
  repositories.

## Gate state

The provenance defect is remediated in source, but Task 26 v4 remains
unreleased. A fresh clean-commit audit is required before the two-phase
CPU-only build may begin. READY does not exist, and GPU execution remains
forbidden until the built bundle is inspected, its tracked provenance is
committed, both repositories are clean, and READY is published.
