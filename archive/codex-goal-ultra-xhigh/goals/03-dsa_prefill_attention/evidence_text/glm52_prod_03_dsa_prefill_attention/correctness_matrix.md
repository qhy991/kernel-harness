# Correctness and execution matrix

Date: 2026-07-22

The serving-native runner executes and synchronizes one reference and one
candidate result before timing. It requires matching tensor structure and shape,
then casts floating outputs to FP32 and applies
`torch.allclose(rtol=2e-2, atol=2e-2, equal_nan=False)`; integer outputs require
exact equality. It does not independently gate matching floating dtype or equal
infinities. A comparison failure aborts before result JSON is written. The
backend hit trace separately checks that its stock output is finite BF16.

| Scope / distribution | Candidate | Mode / topology | Result | Evidence |
|---|---|---|---|---|
| exact raw pool: one 32K sequence, final M4096, scattered top-k, 513 pages | Q32 Swaps | eager, one B200 | pass in all three paired runs | `profile/dsa-prefill-trtllm-m4096-rawpool-tactic-oracle-20260722/results/rawpool_swaps_q32_*.json` |
| same exact raw pool | Q16 Swaps | eager, one B200 | pass in all three paired runs | `profile/dsa-prefill-trtllm-m4096-rawpool-tactic-oracle-20260722/results/rawpool_swaps_q16_*.json` |
| compact 512-page one-sequence trailing top-k | Q32 Swaps | eager, one B200 | pass | `profile/dsa-prefill-trtllm-m4096-tactic-oracle-20260722/results/correctness_q32_trailing.json` |
| compact 512-page one-sequence trailing top-k | Q16 Swaps | eager, one B200 | pass | `profile/dsa-prefill-trtllm-m4096-tactic-oracle-20260722/results/correctness_q16_trailing.json` |
| compact eight-request M4096/context4096 mismatch | Q32 overlay | eager, one B200 | pass; exact predicate fails closed to stock | `profile/dsa-prefill-trtllm-m4096-tactic-oracle-20260722/results/correctness_fallback_8seq.json` |
| real backend-class fixture, generated scattered physical top-k | stock | eager, one B200 | exactly one TRTLLM hit; finite BF16 output | `profile/dsa-backend-prefill-m4096-fixture-20260722/results/backend_hit_trace_zeroed_v2.json` |
| backend-class stock controls | identity stock candidate | eager, one B200 | pass in three 20-pair runs | `profile/dsa-backend-prefill-m4096-fixture-20260722/results/backend_zeroed_stock_stock_*.json` |
| corrected raw-pool rank-local leaf on every host GPU | identity stock candidate | eager, DP4 diagnostic, rank-max | pass in three 20-pair runs | `profile/dsa-prefill-trtllm-m4096-dp4-diagnostic-20260722/results/dp4_stock_stock_*.json` |

The backend identity candidate proves deterministic runner wiring but is not an
independent mathematical oracle. The source-tactic rows do compare distinct
stock and custom generated tactics and passed before every persisted timing
series.

The backend trace and all source attempts run eager and record no CUDA graph
capture. No graph-replay result is claimed. The generated scattered/trailing
tables are diagnostic inputs, not observations of the live GLM-5.2 indexer.

Real-request distribution, model-level accuracy, complete-region correctness,
and TP8/DP8/EP8 graph/overlap acceptance remain external and unpassed.
