# Prior Knowledge — index_score decode >=82% HBM

## Contract

- Backend: stock `deep_gemm.fp8_paged_mqa_logits(clean_logits=False)`.
- M={16,32}, S=65536, heads=32, head_dim=128.
- KV cache pages: `(16384,64,1,132)` uint8.
- AI≈60.21; paged FP8 KV traffic dominates.
- Seed SHA:
  `a1969f1148fb273886026f990366583ea50304db373d4089f04870feb86b6995`.

## Inclusive 82% limits

- M16: 142,673,920 B → `<=21.749073171 us`.
- M32: 285,347,840 B → `<=43.498146341 us`.

User prior is ~73% HBM. Local idle probe was lower: 54.31% / 61.39%.
Reproduce three idle baselines before trusting either.

## Required BitLessons

- `BL-20260719-single-kernel-span-floor`
- `BL-20260718-latency-util-equivalence`
- `BL-20260718-fused-repack-single-launch`

## Source options

1. Isolated DeepGEMM-GLM52 fork: paged MQA scheduler/kernel, TMA/pipeline,
   split-KV/chunking, persistent waves, page/address arithmetic and epilogue.
2. Task-local CUDA/CuTe same-ABI paged MQA.

Reference remains stock DeepGEMM. Do not use GEMM UE8M0 prepack lessons here;
this is a paged MQA path with already-packed FP8 KV rows.
