# GLM-5.2 indexer-score prefill final report

## Disposition

**No replacement.** The only repeatable signal was a 1.03291x pooled gain in
the exact mixed-context score/top-k micro-region. It did not survive the
complete indexer (1.00395x), exact graph-split region (0.99829x), or selected
TRT-LLM DSA-containing region (1.00773x). SGLang is therefore unchanged and
stock dispatch remains active for every production bucket.

## Production path proved

The current no-flag SM100 configuration selects FP8 DSA cache, TRT-LLM
prefill/decode backends, and the PAGED top-k transform. Non-CP long-context
prefill reaches:

`Indexer.forward_cuda`
→ fused packed-UE8M0 `wq_b` and BF16 `wk_weights_proj`
→ fused Q/K prepare, RoPE, quantize, and page-64 cache store
→ `Indexer._get_topk_ragged`
→ `GetKAndS.execute`
→ `deep_gemm.fp8_mqa_logits(clean_logits=False)`
→ `sgl_kernel.fast_topk_transform_fused`
→ FlashInfer TRT-LLM DSA.

The default balanced point is local M4096 under DP8. The exact fixtures cover
the unchunked K65536 rectangle, chunked K262144 maximum-concurrency rectangle,
and a chunked K241664 mixed-context distribution. The B200 static logits cap
is 3,063,848,894 bytes. Short eager contexts skip logits, while PCG/BCG
dispatches the eager split op and the split op independently takes its K-only
short-context branch. PP1 exposes all 148 SMs; a non-last PP rank reserves one
SM. The complete branch and ABI matrix is in `reachability.md`.

## Context-wise paired results

Campaign `20260723T134307Z` tested the broad balanced-chunk source attempt
under one flexible-GPU lease:

| Scope | main | c256 | mixed |
|---|---:|---:|---:|
| score/top-k | 0.99666x | 0.96917x | 1.04642x |
| complete indexer | 0.99642x | 0.99571x | — |
| exact graph split | 1.00638x | 0.99887x | — |
| indexer + selected DSA | 1.00054x | 1.00693x | — |

All single-rank correctness gates passed, but only the mixed score-only
distribution met the threshold without a series regression. The broad policy
was rejected; the later TP4 diagnostic found rank-specific mismatches.

Campaign `20260723T142005Z-mixed-confirmation` then tested a fail-closed
host-metadata predicate under a second single wrapper lease:

| Mixed-context scope | pooled paired p50 | correctness | 3% | no series regression |
|---|---:|---|---|---|
| score/top-k | 1.03291x | PASS | PASS | PASS |
| complete indexer | 1.00395x | PASS | FAIL | FAIL |
| exact PCG/BCG split | 0.99829x | PASS | FAIL | FAIL |
| indexer + selected TRT-LLM DSA | 1.00773x | PASS | FAIL | FAIL |

The score-only series were 1.05924x, 1.00547x, and 1.03532x. Main and c256
fallback controls directly invoked stock and missed the 3% threshold. All
twelve mixed correctness checks and both fallback controls passed.

## Binding profiler evidence

Nsys identifies one gather, one all-SM MQA launch, and one PAGED top-k launch
for the unchunked score path. The chunked paths use two MQA and two top-k
launches. Equal `2048+2048` chunks do not reduce rectangular c256 device work:
captured score kernels were 452.192 µs stock and 453.151 µs balanced.

Mixed contexts have nonuniform causal `ks/ke` ranges. Balancing those chunks
reduces captured score kernels from 958.559 to 863.100 µs, but does not remove
launches or change any kernel. The selected DSA kernel is unchanged at
840.670 versus 840.254 µs, and the alternating containing-region gates remain
below the promotion threshold.

NCU shows the reached MQA kernel occupies all 148 SMs with 384 threads/block,
168 registers/thread, 221,696 bytes shared memory/block, one block/SM, and no
local-memory loads or stores. The stock c256 MQA head and tail take
84.512+53.248 µs; two balanced launches take 2×68.992 µs. The page-cache
gather is only 8.512 µs in NCU. Those results reject a speculative DeepGEMM
fork and gather-only tuning for this goal. The installed cubin lacks line
information, so no source-line PTX/SASS claim is made.

## Source attempt and final policy

The source experiments are reproducible external candidates:

- `indexer_score_balanced_chunks.py` applies equal chunks broadly;
- `indexer_score_balanced_mixed_bucket.py` applies them only to the exact
  host-visible mixed signature;
- three gather variants exercise the reached production Triton kernel.

The focused predicate performs no device read, synchronization, copy, or
adapter work and immediately falls back for every unsupported signature. It
still fails the required containing-region gates.

**Final enable set: empty. Final fallback policy: stock SGLang for every
shape, ABI, graph mode, topology, and context distribution.** No SGLang
source or installed package was modified.

## Distributed and end-to-end status

The official gate remains one-node TP8/DP8/EP8 with the real
`nvidia/GLM-5.2-NVFP4` weights/tokenizer and normal SGLang prefill
scheduling. The local snapshot is configuration-only and the host has four
schedulable B200s, so that unchanged eight-rank model-server gate cannot run
here. It is recorded as an external validation blocker and is not weakened or
relabeled.

The independent TP4/DP4 diagnostic is also non-promotional. After scheduler
retries, the global lock became available. A first run exposed and preserved
a rank-divergent correctness failure that the old runner masked as a
600-second NCCL timeout. Untimed distributed error handling was then made
collective, without changing the timing interval.

The fresh three-series diagnostic rejected the broad candidate before timing
in every series. Rank 1 changed the row-wise top-k set three times, and rank 3
did so in the last two series. Reference correctness completed on all ranks.
Consequently there is no four-rank latency to report, and no direct CUDA
invocation, unpaired comparison, or rank-count substitution is claimed.

## Validation and artifacts

- serving-native structural test: 52 fixed workloads, PASS;
- harness structural test: 24 tasks, zero problems;
- knowledge lint: 13 entries, zero problems; indices/distillation current;
- both campaign status logs: every recorded command exit 0;
- both campaign and profiler checksum manifests: PASS;
- superseded and fresh TP4 diagnostic checksum manifests: PASS;
- fresh TP4 status: three correctness failures, no timing result;
- source compilation, shell syntax, and scoped whitespace checks: PASS.

`verify_harness.py` passes its independent compilation, structural,
knowledge, diff, and audit checks, then exits 1 at the pre-existing GPU-free
generated-task projection mismatch. This goal did not modify any forbidden
task/oracle/timing/pointer file.

Raw paired JSON, logs, source/status snapshots, and hashes are under
`campaigns/20260723T134307Z/` and
`campaigns/20260723T142005Z-mixed-confirmation/`. Nsys/NCU reports are under
the matching `profile/indexer-score-prefill-*` directories. The four-rank
failure evidence is under `four_gpu/20260723T144048Z/` and
`four_gpu/20260723T145846Z/`. The detailed causal record is split across
`attempt_ledger.md`, `profiler_summary.md`, `source_build.md`,
`production_acceptance.md`, and `validation.md`.
