# Goal: optimize the indexer score path used in prefill

Read `../COMMON_RULES.md` first and treat it as part of this goal.

## Outcome

Optimize the dense or chunked FP8 MQA logits path reached by the current SGLang
indexer during prefill.  Preserve memory-budget chunking, masking, weights, and
top-k behavior; do not substitute the decode paged kernel.

## Reachability and workload

Trace a representative local-M4096 prefill request and record query count, expanded
sequence lengths, K range, chunking decision and budget, `clean_logits` setting,
selected backend, SM reservation, PP interaction, and top-k transform.  Add an exact
`indexer_score_prefill_m4096` serving-native workload if absent.

## Work

- [ ] Prove the runtime branch, including whether logits are skipped for short
  contexts and whether full or chunked `fp8_mqa_logits` is called.
- [ ] Construct representative context distributions rather than benchmarking only
  a single dense rectangle.  Name additional fixed workloads if multiple branches
  are materially important.
- [ ] Baseline the score call, top-k handoff, complete indexer, and end-to-end
  prefill with replacement dispatch disabled.
- [ ] Profile query/key traffic, logits writes, chunk boundaries, launch count,
  masking, reductions, top-k overlap, SM use, and memory headroom.
- [ ] Make a source-level change in the reached backend or SGLang fusion path.  A
  candidate may fuse legal masking/top-k work, tune chunking, or improve the MQA
  kernel, but must preserve OOM guards and exact request semantics.
- [ ] Validate multiple context distributions, local M4096, graph/eager behavior,
  indexer outputs, DSA consumption, and SGLang prefill performance.

## Completion

Complete with a production win or a no-replacement result after exact workload,
profiling, and at least one justified source attempt.  A 1.10x dense synthetic win
does not complete the goal if the live path chunks, skips, or uses another backend.

## Deliverables

Include branch/reachability matrix, workload/tests, memory and profiler evidence,
source/build diff, context-wise paired results, correctness, indexer/DSA/e2e result,
and final policy.

