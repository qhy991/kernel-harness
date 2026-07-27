# Production W2 decode attempt ledger

Date: 2026-07-22

## Frozen measurement identity

The measured leaf is SGLang's
`grouped_gemm_nt_f8f8bf16_masked` W2 call after DeepEP dispatch, fused W13,
and fused SwiGLU plus packed UE8M0 quantization. Its rank-local ABI is FP8
activation `[32,1024,2048]`, packed int32 activation scale `[32,1024,4]`, FP8
W2 weight `[32,6144,2048]`, packed int32 weight scale `[32,6144,4]`, and BF16
output `[32,1024,6144]`.

The plan workloads and current-source probes remain distinct:

| Local bucket | Plan workload / hint | Current-source workload / hint |
|---|---|---|
| M16 | `moe_w2_grouped_decode_m16` / 4 | `moe_w2_grouped_decode_m16_current_source_m5` / 5 |
| M32 | `moe_w2_grouped_decode_m32` / 8 | `moe_w2_grouped_decode_m32_current_source_m9` / 9 |

Current DeepEP source adds one full expert denominator before integer division,
which gives 5/9 for exactly divisible EP8 traffic. The deterministic masks are
fixed production-ABI test data; they are not live EP8 router observations.

## Attempt 0 — reachability and dependency provenance

- **Hypothesis:** the plan leaf remains the routed-expert production W2 symbol
  and consumes the packed int32 UE8M0 ABI.
- **Evidence:** source tracing reaches
  `DeepGemmMoeRunner -> grouped_gemm_nt_f8f8bf16_masked ->`
  `deep_gemm.fp8_m_grouped_gemm_nt_masked`. The exact single-B200 runtime call
  recorded recipe and overlap arguments as null, PDL enabled, stream 0, eager
  capture false, 148 SMs, and the complete tensor shapes/strides/dtypes.
- **Dependency delta:** none. Measurements prepend an isolated
  `sgl-deep-gemm==0.1.4.post1` overlay at upstream commit
  `edcf77b276965de8f03cdc47c23f01b08bf7c7ab`; the shared venv was not
  modified. See
  [`stock_deep_gemm_provenance.json`](stock_deep_gemm_provenance.json).
- **Correctness:** the runtime harness checks the exact SGLang wrapper and
  separately poisoned output buffers; no frozen E8/f32-scale result is used.
- **Risk:** live EP8 `packed_recv_count`, graph state, stream, recipe, signal,
  and overlap details still require the eight-rank production lane.
- **Decision:** retain the exact four named workloads and pinned stock package
  as the comparison authority.
- **Rollback:** no production implementation changed.

## Attempt 1 — DeepGEMM M-alignment portfolio

- **Hypothesis:** all fixed test-mask counts are at most 14, so BM16 can remove
  padded M/tensor/epilogue work relative to BM128 without increasing the fixed
  workloads' one-M-tile-per-expert logical task count.
- **Baseline evidence:** campaign
  `glm52-w2-alignment-2bae536257aa929b957fdb28` ran alignments
  128, 32, 64, 96, and 16 on the same physical B200. Every
  alignment/workload row has three sessions of 30 alternating pairs after five
  warmups. Stock-vs-stock alignment-128 medians were
  0.992249x--0.994566x.
- **Exact delta:** only
  `set_mk_alignment_for_contiguous_layout()` changed. Inputs, masks, valid rows,
  recipes, overlap work, PDL, SM count, output semantics, and packed ABI stayed
  fixed. The experiment restored stock alignment 128 after each run.
- **Expected low-level effect:** keep 1,536 logical tile tasks, 11 logical
  scheduler waves on 148 SMs, and a 56-task final wave while changing BM/UMMA-M
  from 128 to 16. N remains 128. The persistent launch grid remains 148.
- **Correctness:** all 20 alignment/workload rows passed pre-timing and fresh
  post-timing active-row checks. BM16 has `calc_diff=max_abs=max_rel=0` for all
  four rows, separate fresh input storage, and matching stock/candidate return
  semantics.
- **Paired result:** BM16 paired medians were 1.080470x, 1.087436x,
  1.075564x, and 1.062069x for plan-M16, source-M16, plan-M32, and source-M32.
  Each row contains 90 pairs. Full p10/p50/p90 and raw distributions are in
  [`paired_alignment_summary.json`](paired_alignment_summary.json).
- **Negative variants:** BM32 also cleared the 3% median gate on all four rows
  but was slower than BM16. BM64 cleared only the two M16 rows and missed both
  M32 rows. BM96 and BM128 cleared none. These results are preserved rather
  than collapsed into the selected row.
- **Profiler delta:** NCU duration fell from 75.520--76.256 us to
  68.576--70.112 us. Total DRAM reads changed modestly, about 414.28 MB to
  406.90 MB, while writes fell from 40.41--40.67 MB to 7.94--8.34 MB.
  Registers fell 36 to 34 with zero spills. Static MMA and input-TMA counts
  stayed 16 and 10, while PTX TMEM loads fell 32 to 4 and SASS output TMA
  stores fell 16 to 2. Nsys independently measured one target launch per
  workload at 74.687--76.544 us stock and 65.664--66.816 us BM16.
- **Remaining bottleneck:** BM16 tensor-pipe active percentage fell to about
  4.2%, while eligible warps fell and long-scoreboard ratio rose to about
  21.5. SourceCounters place the largest samples in barrier/TMA-wait paths.
  The locally faster leaf is now coordination/memory-latency limited rather
  than register- or tensor-throughput limited.
- **Risk:** alignment selection is process-global. BM16 can increase logical
  tasks and reload B for experts with more than 16 rows; the fixed performance
  result cannot be transferred to arbitrary masks or other grouped GEMMs.
- **Decision:** select BM16 only for graph, edge, and profiling experiments;
  decline production enablement.
- **Rollback:** stock alignment/config selection 128.

## Attempt 2 — evidence-directed source and scheduling portfolio

- **Source diagnosis:** the pinned SM100 grouped-masked heuristic does not use
  `expected_m` to choose among these layouts. Every fixed-mask expert is
  nonempty and has at most 14 rows, so all tested alignments produce
  `32 * (6144/128) = 1536` logical tile tasks. The 4/5 and 8/9 hints therefore
  do not change code or task count.
- **BM32 dynamic effective-M tail:** not built. The measured stock library
  already exposes BM16 directly, BM16 beats BM32 on all four rows, and a source
  rewrite would not solve process-global dispatch or TP8 acceptance.
- **SM reservation:** not attempted. The selected leaf already launches one
  persistent block on each of 148 SMs, current routed W2 overlap is disabled,
  and no production-region overlap measurement justifies taking SMs away.
- **Cluster-N=1:** not attempted. Profiling attributes the measured gain to
  padded epilogue removal; it does not establish that replacing the proven
  2-CTA tensor-core path is the next deployable bottleneck.
- **Static-M compilation:** rejected as low upside because 4/5/8/9 select the
  same config and code shape.
- **CLC/persistent rewrite:** rejected before implementation because the
  existing scheduler already supplies more than ten logical waves and the
  remaining blocker is production dispatch/acceptance, not insufficient work.
- **Static oracle contract considered:** SM100 with 148 SMs; E32/slab1024;
  N6144/K2048; packed int32 UE8M0 scales; no accumulation, recipe, or overlap;
  and exact `(forward_m, expected_m)` pairs `(16,4)`, `(16,5)`, `(32,8)`, and
  `(32,9)`. It may not read `masked_m` on the host or synchronize for dispatch.
- **Decision:** no source candidate or SM overlay was built. Attempt 1 is the
  justified configuration attempt; further source churn cannot produce a
  promotable bucket without a fail-closed dispatch design and the eight-rank
  gates.
- **Rollback:** isolated stock post1 overlay; no production source delta.

## Attempt 3 — leaf graph and edge contracts

- **Hypothesis:** BM16 preserves fixed-pointer CUDA Graph replay, active-row
  correctness, stream behavior, and return semantics, including masks that
  cross the 16-row tile boundary.
- **Exact delta:** alignment 16 only; stock PDL, 148 active SMs, ABI, and output
  pointers were preserved.
- **Graph result:** strict PASS for all four workloads. Each capture observed
  capture active during launch, replayed 30 times deterministically, matched
  eager output exactly, and preserved the return contract. Graph p50 was
  0.083968--0.084064 ms.
- **Edge result:** strict PASS for four artifacts and twelve cases covering
  counts `0,15,16,17,31,32,33,127,1024`, empty experts, front-loaded and
  scattered boundaries. Active-row `max_abs=max_rel=0` in every case.
- **Risk:** edge validation is correctness-only, not an edge-mask performance
  oracle. The graph is a single-GPU leaf graph, not the full MoE serving graph.
- **Decision:** leaf contracts pass, but they do not change the production
  no-replacement decision.
- **Rollback:** alignment 128.

## Attempt 4 — stock TP4/DP4/EP4 diagnostic

- **Scope:** attempt
  [`tp4_20260722T185932Z_1629770_10420`](tp4_diagnostic/tp4_20260722T185932Z_1629770_10420/summary.json)
  collected three stock trials for dispatch, combine, and the lower-level MoE
  region at M16 and M32, plus one Nsys result/report for each workload.
- **Result:** strict post-collection PASS for 18 stock baseline results and six
  Nsys reports. Median-of-trial-medians and full trial-median ranges are in
  [`validation.md`](validation.md).
- **Correctness boundary:** every raw result has `candidate=null`. The region
  check is eager, no-overlap EP4 structural-contract and repeatability
  validation with `independent_math_oracle=false`; it is not a candidate-vs-
  stock comparison, production CUDA Graph, or independent numerical oracle.
- **Audit history:** the raw manifest retains the first validator failure,
  caused by the Nsys `Name`-column/preamble parser. After the parser fix, the
  CPU-only strict validator passed and rechecked the same summary idempotently.
- **Diagnostic environment:** all 24 workload logs use DeepEP's default 20-SM
  communication config, report failed IBGDA transport initialization, and let
  ProcessGroupNCCL infer device IDs from global rank. The timing ranges are
  retained as fallback-environment diagnostics, including the dispatch-M16 and
  combine-M32 excursions; they are not tuned communication baselines.
- **Decision:** retain as four-rank diagnostic evidence only. It cannot satisfy
  or weaken TP8/DP8/EP8 region or SGLang end-to-end acceptance.
- **Rollback:** stock was the only implementation exercised.

## Containing-region and serving acceptance

| Gate | Raw authority | Status |
|---|---|---|
| Single-B200 exact leaf, plan 4/8 | paired summary and raw microbench JSON | passed locally; disabled in production |
| Single-B200 exact leaf, current source 5/9 | paired summary and raw microbench JSON | passed locally; disabled in production |
| CUDA Graph repeated replay and edge masks | [`leaf_validation_summary.json`](leaf_validation_summary.json) | strict PASS; single-GPU leaf scope |
| TP4/DP4/EP4 diagnostic | [exact attempt](tp4_diagnostic/tp4_20260722T185932Z_1629770_10420/summary.json) | strict PASS; stock-only eager/no-overlap diagnostic |
| TP8/DP8/EP8 dispatch -> W13 -> SwiGLU+quant -> W2 -> combine | external lane | BLOCKED on this four-GPU host |
| Eight-rank SGLang end-to-end comparison | external lane | BLOCKED on this four-GPU host |

## Final decision

The leaf experiment is real and faster, but it is not deployable as a static
production oracle: its controlling state is process-global, it has no live EP8
mask proof, and the required full-region and end-to-end topology gates cannot
run here. No production SGLang source was changed; the SGLang commit is a
test-only contract lock. Every W2 bucket remains on stock.
