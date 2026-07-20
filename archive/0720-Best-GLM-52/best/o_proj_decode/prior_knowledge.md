# Prior knowledge digest — glm52/o_proj_decode (B200)

Source: `Kernel-Harness/testbench/knowledge/entries/glm52--o_proj_decode--b200--20260714a.json`

## Headline

- Status: **no-win** (geomean ≈ 1.0; conservative min speedup ~0.987 at M=16)
- Bound: **memory-bandwidth** (small-M FP8 GEMM dominated by reading large weight)
- Lesson: production DeepGEMM wrapper is already the safe path; CUPTI device-kernel
  timing does **not** reward Python-wrapper bypasses.

## Dead ends already tried

| Approach | Outcome |
|---|---|
| Call lower-level `deep_gemm_fp8_fp8_bf16_nt` packing | slower |
| Raw `deep_gemm.fp8_gemm_nt` | error (scale layout) |
| Force Triton w8a8 block fp8 | incorrect |
| `num_sms` sweep | abandoned (no dual-shape win) |
| Restore production wrapper | correct, not faster on every shape |

Refresh live with:

```bash
cd /home/qinhaiyan/Kernel-Harness
.venv/bin/python testbench/bin/knowledge.py query --task glm52/o_proj_decode
```

## Important correction (2026-07-16): prior no-win was K=2048, current task is K=16384

The `20260714a` entry records `shapes.K = 2048` ("reading the 6144x2048 FP8 weight matrix"),
but the current `definition.json`/`task.json` specify **K=16384** (local_heads*v_head = 64*256).
At K=2048 the kernel achieved only ~1.4 TB/s (~18% of the ~8 TB/s B200 HBM peak) — under-occupied /
launch-bound, not a saturated bandwidth floor. Re-baselining at K=16384 showed the production
DeepGEMM wrapper at ~32.9 us / ~38.7% of HBM peak on both M∈{16,32}.

## New result (2026-07-16): WIN at K=16384 via compiled_dims="nk"

A previously-untried lever — calling `deep_gemm.fp8_gemm_nt(..., compiled_dims="nk")` directly
(baking N=6144,K=16384 as compile-time constants; the production wrapper uses `compiled_dims=""`)
— beats the baseline on both shapes: **sp_cons 1.069 (M=16) / 1.063 (M=32)**, geomean 1.071,
correct (rel_err 0), **drop-in verified**. Keeps M dynamic (safe drop-in). See docs/results.md.
The five prior mechanism-level dead-ends still hold; `compiled_dims` was simply never tried, and
the K=2048 regime was too small to reveal the benefit.

