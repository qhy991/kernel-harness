# Attempt ledger

## A0 — frozen-task-only tuning (rejected before measurement)

- Hypothesis: optimize the old `index_score_prefill` candidate in isolation.
- Rejection: it does not include production page-cache gathering, the current
  PAGED fused top-k handoff, memory-budget chunking, or the fused producer.
- Decision: leave the frozen task untouched and build an exact serving-native
  workload.

## A1 — equalize rows across the minimum safe chunk count

- Hypothesis: the stock `max_rows=floor(budget/bytes_per_row)` schedule leaves
  a large tail chunk (`2921+1175` at c256). Keeping the same two launches but
  using `2048+2048` may reduce the longest-kernel tail and improve top-k
  handoff locality.
- Baseline evidence: production's cached 3,063,848,894-byte static budget and
  the c256 4-GiB logits matrix force two chunks.
- Delta: the external candidate computes
  `ceil(M / ceil(M/stock_max_rows))` and temporarily supplies the corresponding
  smaller budget to the exact production method. Unchunked cases call
  `runtime.reference` directly.
- Expected runtime effect: same kernel code, same number of score and top-k
  launches, no chunk above the stock byte budget, less launch-duration
  imbalance, no new tensor adapter/copy/synchronization.
- Single-rank correctness: PASS for all nine exact
  score/complete/graph/DSA workloads. SGL-Kernel top-k is intentionally
  unordered, so the gate checks row-wise selected-page-set equality; the DSA
  lanes additionally compare the floating attention output. A5 records the
  later rank-specific TP4 correctness failure.
- Paired result: the main and c256 score/region workloads all missed the 3%
  gate. The pooled c256 score result was 0.96917x and the complete, graph,
  and DSA results were 0.99571x, 0.99887x, and 1.00693x. The mixed score-only
  distribution reached 1.04642x with all three series non-regressing.
- Profiler delta: c256 captured kernel time was effectively unchanged
  (452.192 µs stock, 453.151 µs balanced), while mixed score kernel time fell
  from 957.312 to 862.559 µs because the nonuniform `ks/ke` work was divided
  more evenly.
- Risk: applying the schedule to every two-chunk bucket regresses the c256
  score path and would be an unsafe broad dispatch.
- Decision: reject the broad candidate. Follow the mixed-only signal through
  a host-metadata, fail-closed bucket and all containing-region gates in A4.
- Rollback: remove candidate selection; stock SGLang is never overwritten.

## A2 — device-kernel rewrite threshold

A DeepGEMM/CUTLASS rewrite is attempted only if NCU shows a concrete
instruction, occupancy, or memory-transaction limit not already at the library
kernel's roof. PTX/SASS claims require source-level NCU evidence. If A1 misses
the 3% gate and the dominant MQA kernel is already efficient, this avenue is
declined rather than introducing a speculative fork.

- Evidence: MQA uses all 148 SMs, 168 registers/thread, 221,696 bytes shared
  memory/block, one block/SM, and has no local-memory loads or stores. The
  stock c256 head+tail NCU time (84.512+53.248 µs) is the same as two
  balanced launches (2×68.992 µs). Top-k likewise retains the same aggregate
  work.
- Decision: decline a library fork. Resource pressure exists, but neither the
  paired region data nor the per-chunk device totals identify a deployable
  3% region opportunity. The installed cubin also lacks line information, so
  inventing a source edit would not satisfy the source-evidence rule.

## A3 — tune the reached K+scale gather launch

- Hypothesis: alternate Triton block/warp choices might improve the
  production page-64 `GetKAndS` gather.
- Delta: three external variants launch the existing production Triton kernel
  with `(BLOCK_SIZE, num_warps)` of `(64,4)`, `(128,4)`, and `(128,8)`.
- Correctness: an untimed diagnostic compared all 33,554,432 K bytes and
  1,048,576 scale bytes for the c256 fixture; all three variants were exact.
- Profiler evidence: the stock gather is 5.632 µs / 1.9% of the unchunked
  Nsys score range and 8.512 µs in NCU while reading 8.66 MB at 1.018 TB/s.
  It is only 13.6 µs in the c256 score capture and 13.856 µs / 1.0% of the
  complete DSA region.
- Decision: reject before a formal performance campaign. Even eliminating the
  measured kernel entirely cannot satisfy the 3% containing-region gate, and
  the variants add no fusion or byte reduction.

## A4 — fail-closed mixed-context bucket

- Hypothesis: A1's mixed-context result reflects real `ks/ke` load balancing,
  but only that exact host-visible request distribution should be eligible.
- Delta: `indexer_score_balanced_mixed_bucket.py` enables `2048+2048` only for
  local M4096, K241664, batch 16, the fixed extend/context vectors, and stock
  chunks `3169+927`. Every other signature directly invokes stock. Dispatch
  reads no device value and does not synchronize.
- Required gates: three alternating series each for score, complete indexer,
  exact PCG/BCG split region, and selected TRT-LLM DSA consumption, plus
  Nsys stock/candidate captures in one GPU lease.
- Correctness: PASS in all twelve alternating mixed-context runs and in both
  stock-fallback controls. The score gate compares row-wise selected-page sets;
  the DSA gate also compares the floating attention output.
- Paired result: pooled score-only latency reached 1.03291x and all three
  score series were non-regressing. The complete indexer reached 1.00395x,
  the exact graph-split region 0.99829x, and indexer plus the selected
  TRT-LLM DSA consumer 1.00773x. All three containing-region results missed
  the 3% gate and each had at least one series below stock. The fail-closed
  main and c256 controls also missed the 3% gate, as expected for direct
  stock fallback.
- Profiler delta: the confirmation capture again reduced the mixed score
  kernels (958.559 to 863.100 µs), but the paired complete-region gate did
  not retain a deployable gain. The selected attention kernel was unchanged
  (840.670 versus 840.254 µs), making the score change progressively less
  material in the real DSA region.
- Risk: enabling a policy from the score-only result would promote a bucket
  that fails the complete, graph, and selected-consumer gates and has no
  local end-to-end server validation.
- Decision: reject promotion. Keep the exact candidate as reproducible
  evidence, leave SGLang unmodified, and retain stock dispatch for every
  bucket. The unavailable unchanged eight-rank server gate is an additional
  external blocker, not a waived requirement.
- Rollback: none required; the experiment never changed the SGLang runtime.

## A5 — four-rank diagnostic and collective failure hardening

- Hypothesis: although the broad c256 schedule lost locally, a replicated
  TP4/DP4 diagnostic should preserve its row-wise selected-page sets across
  independent rank seeds and provide a maximum-rank latency sanity check.
- First attempt: run `20260723T144048Z` exposed a runner weakness. One rank
  failed candidate correctness and entered cleanup while its peers advanced
  to the next timing barrier, masking the original exception behind a
  600-second NCCL timeout. The raw series-1 timeout and interrupted series-2
  logs are preserved.
- Runner delta: untimed reference/candidate correctness failures now
  participate in a device-side MIN handshake before any rank can enter the
  timing loop, and NCCL barriers receive the logical local device explicitly.
  This changes neither a candidate call nor the CUDA-event timing interval.
- Fresh diagnostic: all three `20260723T145846Z` series reached reference and
  candidate correctness without a collective timeout, then exited 1 before
  timing. Rank 1 had a row-wise top-k set mismatch in every series; rank 3
  also mismatched in series 2 and series 3.
- Decision: the broad schedule is incorrect across representative rank seeds.
  No four-rank latency is reported. This strengthens the no-replacement
  decision and is not relabeled as the unchanged eight-rank acceptance gate.
