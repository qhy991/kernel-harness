# Attempt ledger — decode `index_q_upproj` graph-only fixed-N/K, round 1

Bounded to ≤2 identities per plan. One identity was decisive; the second was
not spent.

## Attempt 1 — stock vs `compiled_dims="nk"` at locked ABI + graph-only registration

- **Hypothesis:** decode `index_q_upproj` (`wq_b`, N=4096/K=2048) is a tiny,
  launch/latency-bound FP8 GEMM (prior NCU: SM ~3–7%, DRAM ~10%). Baking N/K as
  compile-time immediates via `deep_gemm.fp8_gemm_nt(compiled_dims="nk")` should
  reduce the launch/scheduling cost under CUDA-graph replay, mirroring the
  accepted decode `o_proj` result — while eager pays a dispatch tax and must
  fall back to stock.
- **Baseline evidence:** stock `fp8_gemm_nt` default compile; captured leaf is
  one GEMM node with dynamic N/K immediates (`…Lj0ELj0E…`).
- **Delta:** `KernelSpec.graph_only=True` on the `index_q_upproj` `_E2E_DECODE`
  entry; `_graph_only_declines` gate in `try_dispatch_fp8_gemm` (declines in
  eager before ABI/lock, selects under capture); `graph_only_enabled` +
  `SGLANG_GLM52_INDEX_Q_UPPROJ_GRAPH_ONLY` env. No numerics change — same
  DeepGEMM math, only compile specialization differs.
- **Expected binary/runtime effect:** distinct kernel mangled name differing
  only in the N/K immediates; identical grid/block/smem; bit-identical output;
  faster graph-replay leaf; eager tax visible as a sub-1.0 identity lane.
- **Correctness:** bit-exact vs stock at M16 and M32 — eager, graph replay, and
  region max-abs-err all `0.0`. (Mandatory: any numeric drift corrupts the
  downstream `fp8_mqa_logits` top-k, the failure mode that rejected the prior
  Triton attempt.)
- **Paired result (3 series, one lease, GPU-30b619de):**
  graph leaf M16 1.218× / M32 1.188×; graph region M16 1.173× / M32 1.159×;
  every series × every estimator ≥ 1.126 (floor 1.03). Independent confirmation
  lease: leaf 1.238×/1.189×, region 1.153×/1.161×.
- **Profiler / topology:** candidate leaf = 1 clean GEMM node, no forbidden
  nodes; identity differs from stock; capture identical with graph-only on/off.
- **Risk:** eager decode regresses ~9% (identity lane 0.91×) — mitigated by the
  graph-only decline (eager uses stock, no provider launch). Production decode
  is CUDA-graph-replayed, so the tax is not paid on the hot path.
- **Decision:** **accept as external-acceptance-candidate, default off.** Clears
  the leaf and containing-region gates with wide margin at both M buckets.
- **Rollback point:** revert the four sglang files; the candidate is
  explicit-only and off by default, so no revert is required for safety.

## Attempt 2 — optional epilogue/schedule identity

- **Not spent.** Plan hypothesis 2 was conditional on attempt 1 being
  region-limited with clear headroom. Attempt 1 cleared both the leaf and the
  containing-region gate at both M buckets with margin, so the region was not
  the binding limit and the second identity was not justified.

## Prior negative evidence honored

`glm52-goal-runs/15-indexer_wq_b_decode` ended no-replacement via (a) a native
packed Triton kernel that was numerically wrong downstream (broke top-k) and
(b) a stock-DeepGEMM SM-budget grid change that regressed (0.976–0.991×). Neither
was the `compiled_dims="nk"` graph-only path; both are treated as negative
evidence and were not re-enabled. The archived `best-*` / packed leaves remain
disabled.
