# KV-context transfer assessment — 2026-08-01

## Decision

The original harness could not establish that a single-GPU kernel gain would
transfer to KV-cache-conditioned SGLang serving. It mixed three different scopes:

| Scope | Original coverage | What it can prove |
|---|---|---|
| frozen synthetic suite | mostly fixed `S=65536`; some dense/contiguous KV and one paged decode proxy | idea/correctness evidence only |
| serving-native decode | sparse TRT-LLM attention at fixed context 8192 | exact post-top-k attention ABI at one point |
| serving-native prefill | communication tasks at fixed local token counts | no incremental-prefill KV claim |

After this review, DSA decode accepts traced fixed or ragged logical KV lengths and
records its scope in result JSON. This still is intentionally a
`decode-sparse-attention-leaf-only` test. It does not time the full-context DSA
indexer score scan, top-k/page-table transform, scheduler, global memory pressure,
or any incremental-prefill path.

## Why prefix KV changes incremental prefill semantics

At pinned SGLang commit `2a51dee179cb3ebde8cf54185017eba65217203a`, DSA
prefill consumes real `ForwardBatch` sequence, prefix, extend, and page-table
metadata. For Blackwell with FP8 KV, automatic selection uses the observed relation

```text
total_kv_tokens < total_q_tokens * 512
```

to select `flashmla_sparse`; otherwise it selects `flashmla_kv`. The top-k
transformation can also change between ragged and paged forms. Consequently,
prefix=0 full prefill, a dense contiguous K/V tensor, and a decode attention call
are not valid substitutes for incremental prefill with an existing cache.

The minimum production axes are batch size, per-request prefix length, extend/chunk
length, total KV length, ragged distribution, cache dtype, page size, sparse top-k,
selected backend, graph mode, and TP/DP/EP topology.

## Effect of ignoring KV cache

- DSA indexer cost scales with the full logical context even when sparse attention
  consumes at most 2048 selected tokens.
- Cache length/distribution changes page-table transforms, TLB/cache locality,
  physical address range, dequantization, backend dispatch, graph buckets, memory
  pressure, and scheduling/overlap.
- GEMM/MoE leaf latency may not directly depend on KV length because KV is outside
  their ABI. Their end-to-end contribution still changes because their token/batch
  bucket, latency share, communication overlap, and critical path change.
- A single GPU omits production TP/DP/EP communication and overlap. It is a useful
  mechanism proof, not a service-level prediction.

If a leaf owns fraction `f` of end-to-end latency and has isolated speedup `s`, the
no-overlap upper bound is

```text
latency reduction = f * (1 - 1/s)
end-to-end speedup = 1 / (1 - f + f/s)
```

Overlap can reduce the realized gain to zero when another stream or communication
stage remains the critical path.

## Required promotion evidence

1. Capture one trace-backed context profile. Diagnostic powers-of-two are not
   production evidence.
2. Gate sparse decode attention across its fixed/ragged context scenarios.
3. Gate the full decode region including indexer score, top-k/page transform, and
   attention.
4. Replay each incremental-prefill `prefix_tokens x extend_tokens` scenario through
   the real SGLang containing region and record the selected backend.
5. Run the same weighted profile end-to-end on production topology and accept only
   if no relevant phase/context/mode regresses.

The atomic recipes for these experiments are `01-capture-kv-context-profile`,
`05-decode-kv-context-matrix`, `06-incremental-prefill-kv-matrix`,
`08-containing-region-promotion`, and `09-end-to-end-serving-promotion`.

## B200 implementation checks

On `verda-b200x4`, the revised reference path executed successfully for fixed 2k,
fixed 32k, a 16-request ragged 1k–16k batch, and the four diagnostic context-profile
scenarios including M32/65k. These were functional probes, not production
performance verdicts. A reference-equivalent candidate also passed pre/post and
independent graph correctness at fixed 32k; the eager timing remained unstable and
correctly returned `UNSTABLE_NO_VERDICT`.
