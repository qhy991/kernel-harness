# Partial production bundle

The scheduler selected physical GPU 2
(`GPU-df8b1d78-b06c-39a2-54f0-66b9fabf3a99`).  The bundle completed:

- the production-PDL runtime trace;
- three paired stock identity series;
- three `Fp8LinearMethod.apply` baseline series;
- the supported packed-DeepGEMM configuration sweep; and
- the Nsys capture.

It stopped before NCU because Nsight Systems 2025.6 created an SQLite sidecar
whose timestamp triggered a subsequent `nsys stats` refusal.  The command
requires `--force-export=true`.  The Nsys CSV exports were recovered
CPU-only with that option, but no NCU report belongs to this bundle.

The completed timing evidence remains valid and is preserved.  It is not
paired with NCU collected in a later GPU allocation.  A fresh `20260723d`
bundle reruns the entire alternating series and profiler sequence together.
