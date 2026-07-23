# Goal: optimize the indexer score backend used in decode

Read `../COMMON_RULES.md` first and treat it as part of this goal.

## Outcome

Optimize the paged MQA score/top-k path selected by the current SGLang indexer for
decode M16 and/or M32.  Do not assume stock `deep_gemm.fp8_paged_mqa_logits` is
active until `DSAPagedMQALogitsBackend.resolve` and a runtime trace prove it.

## Reachability and workload

Record the resolved backend, native versus split mode, metadata builder, cache/page
layout, sequence lengths, query/key FP8 representation, head weights, masking,
`clean_logits` behavior, top-k transform, graph mode, and any PP/SM reservation.
Add named `indexer_score_decode_m16/m32` serving-native workloads for the exact
backend and context distribution if they are missing.

## Work

- [ ] Trace both decode buckets from indexer Q/K preparation through paged score,
  masking, and top-k.  Identify whether the score computation is skipped for any
  request class.
- [ ] Establish paired score-kernel, complete indexer, and SGLang decode baselines.
- [ ] Profile paged gather traffic, logits materialization, weight application,
  masking, metadata overhead, top-k handoff, SM budget, and launch/graph behavior.
- [ ] Optimize the backend actually selected.  Source edits may target DeepGEMM,
  CuTe-DSL, SGLang metadata/fusion, or another resolved library.  Preserve the
  paged ABI and top-k-equivalent semantics.
- [ ] If a dense or synthetic paged kernel is used for development, reproduce the
  live sequence/page distribution before using its result for an oracle.
- [ ] Build separate M16/M32 variants only when paired data supports them.  Validate
  score/top-k correctness, graph replay, full indexer latency, DSA decode, and
  end-to-end behavior.

## Completion

Complete with a production win for at least one reachable bucket or a no-replacement
result.  An old 82% HBM target is diagnostic only; backend reachability and full
indexer improvement are mandatory.

## Deliverables

Provide backend resolution evidence, exact workload/tests, profiler artifacts,
source/build provenance, score and top-k correctness, paired bucket results,
indexer/DSA/e2e results, and fallback policy.

