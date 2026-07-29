# GLM-5.2 fused W13 decode result

## Disposition

**External-acceptance-candidate.** The exact BM16 two-SM route
`(16,128,128,12,2)` passes every required local correctness, ABI, graph,
stability, leaf, and containing-region gate for expected-M 4/5/8/9.

It is not a production win. Checkpoint-backed TP8/DP8/EP8 full-region and
serving acceptance has not run, so the production default remains stock and
the candidate remains default-off.

## Exact implementation

The candidate changes only fused W13 for E32/slab1024/K6144/N4096 with packed
`int32` UE8M0 scales and local decode buckets M16/M32. Current SGLang API-v1
dispatch reaches:

```text
grouped_gemm_nt_f8f8bf16_masked
  -> hotspot_provider.run_moe_masked
  -> provider_bm16_2sm.moe_w13
  -> infini_kernel_glm52_moe_w13_decode_em{4,5,8,9}_bm16_2sm
```

Supported calls select one expected-M-keyed symbol and return exactly `None`
after writing the caller-owned BF16 slab. Unsupported ABI, phase, recipe,
topology, expected-M, W2, prefill, and unrelated grouped-GEMM calls select
stock before candidate invocation. A selected candidate failure propagates;
there is no post-invocation stock retry.

SGLang is committed at
`5af212d00439a8990a1d64e2b7e32aa207acf2cb`, directly based on required
revision `83d313104d089bcd2af26b28453ff880f1e6a80b`. DeepGEMM is committed at
`87e0359edbb461181d3bba218442132007b9a738`, based on
`731e7c7a97d269e4b9f482ea18d0e709a948f293`. The complete source/build
identity is in `build_manifest.json`; its SHA256 is
`3c9be1ede19cee1243c9d6ea46682a77d675d48c1f4b9d6738c9c446c1e48b63`.

## Correctness and production semantics

Both BM16 topologies pass 12 eager cases spanning all expected-M points,
empty/zero/boundary/maximum/skewed and changed expert counts, deterministic
ramp/random/extreme finite/exponent-boundary data, poisoned output, repeated
eager calls, non-default streams, packed-scale byte preservation, untouched
masked regions, exact dtype/shape/stride/storage offset, output ownership, and
the `None` return contract. Both have zero failing elements and maximum
absolute error 0.

The separately captured graph leaf contains exactly one W13 node. The graph
region substitutes only W13 and retains the same ordered stock
SwiGLU/packed-quant and W2 nodes. Replay mutates activation and device
`masked_m`, re-poisons outputs, preserves pointers, and checks BM16 W13 plus
BM128 W2 store envelopes. Production traces record eight selected eager/graph
calls per topology, exact API-v1 returns, and pre-invocation fallback for all
unsupported controls.

Authoritative files are `correctness_eager.json`,
`production_trace_bm16_2sm.json`, `production_trace_bm16_1sm.json`, and the
fresh correctness sections in every fair result.

## Fair local performance

Every row is three independent same-process series of 50 alternating AB/BA
pairs on physical B200
`GPU-30b619de-87f2-1862-0d07-a595da8fe417`, with SM 1965 MHz and memory
3996 MHz recorded in the raw results. Each series independently passes finite
pooled, order-balanced, AB-median, and BA-median speedups of at least 1.03.

| Expected-M | Scope | Mode | Stock p50 (us) | Candidate p50 (us) | Weakest per-series estimator |
|---:|---|---|---:|---:|---:|
| 4 | leaf | eager | 150.560 | 144.320 | 1.042553 |
| 4 | leaf | graph | 151.552 | 145.248 | 1.042254 |
| 4 | region | eager | 231.360 | 222.336 | 1.037271 |
| 4 | region | graph | 231.456 | 223.264 | 1.036395 |
| 5 | leaf | eager | 150.624 | 144.288 | 1.043449 |
| 5 | leaf | graph | 151.552 | 145.216 | 1.042474 |
| 5 | region | eager | 231.328 | 222.336 | **1.034255** |
| 5 | region | graph | 231.424 | 223.232 | 1.036283 |
| 8 | leaf | eager | 150.592 | 144.320 | 1.043228 |
| 8 | leaf | graph | 151.552 | 143.520 | 1.043172 |
| 8 | region | eager | 231.120 | 222.240 | 1.036262 |
| 8 | region | graph | 231.456 | 223.264 | 1.036511 |
| 9 | leaf | eager | 150.624 | 144.384 | 1.042544 |
| 9 | leaf | graph | 151.552 | 145.152 | 1.042483 |
| 9 | region | eager | 231.392 | 223.232 | 1.036405 |
| 9 | region | graph | 231.456 | 223.264 | 1.035922 |

The independent `fairness_audit.json` re-audits all 16 raw files: 2400 total
pairs, 3000 candidate hits, zero fallback, zero reference delegation, one
physical GPU, and 16/16 passed lanes. Stock-vs-stock A0 controls remained
centered around 1.0 and were forced non-wins.

## Generated binary and resources

`jit_identity_audit.json` records the source, cache key, command-owned
identity, CU/PTX/cubin/SASS hashes, and resources for all eight candidate
symbols and the one stock identity. There are no duplicate candidate
contexts.

| Identity | Config | Cluster | Registers | Dynamic shared | Stack/local/spills | Key lowering |
|---|---|---:|---:|---:|---|---|
| stock | `(128,128,128,8,2)` | 2 | 36 | 213804 B | 0/0/0 | 16 `UTCQMMA.2CTA`, 32 `LDTM`, 10 `UTMALDG` |
| BM16 one-SM | `(16,128,128,11,1)` | 1 | 31 | 223020 B | 0/0/0 | 16 plain `UTCQMMA`, 4 `LDTM`, no cooperative `UCGABAR` |
| BM16 two-SM | `(16,128,128,12,2)` | 2 | 35 | 230188 B | 0/0/0 | 16 `UTCQMMA.2CTA`, 4 `LDTM`, 10 `UTMALDG`, 3 arrive/wait cluster barriers |

This proves the survivor is a genuine two-CTA `tcgen05.mma.cta_group::2`
kernel and the rejected comparison is a genuine one-CTA
`tcgen05.mma.cta_group::1` kernel.

## Nsys attribution

The clean graph-node Nsys collection contains 58 stock and 58 candidate region
launches. W13 device p50 falls from 145.744 to 139.168 us, a 1.047256x
speedup and 6.5765 us reduction. The complete device critical span falls from
226.176 to 218.816 us, a 1.033636x speedup and 7.360 us reduction. Stock
activation/quant and W2 remain the downstream nodes; sub-microsecond negative
boundary gaps show an essentially serial critical path. Graph submission API
p50 increases by 0.361 us, so the gain is not a host-submission artifact.

The report and exact analysis are
`nsys_bm16_2sm_region_graph_em4.nsys-rep.gz` and
`nsys_attribution.json`. The profiled 3×10 result is attribution-only;
unprofiled 3×50 results are the performance authority. The earlier eager
collection failed because two CUPTI subscribers conflicted and is retained as
negative evidence. NCU was intentionally not invoked because generated
PTX/SASS/resources and Nsys left no concrete unresolved survivor question.

## Negative results and provenance

The BM16 one-SM topology is correct but fails the 1.03 screening threshold for
every expected-M. Direct and by-value wrapper attempts failed compilation or
correctness/resource requirements. `Bsymbolic` alone did not isolate JIT
context statics; identical hidden visibility plus `Bsymbolic` did.
`attempt_ledger.md` retains every failure and causal decision.

The fair runs were captured while the task changes were uncommitted, so their
recorded repository provenance is dirty and should be treated as provisional
in that narrow git-status sense. This is disclosed rather than erased. Every
result binds the exact runner, workload, candidate, provider, manifest, DSO,
generated source, PTX, cubin, and SASS by SHA256; those exact sources are now
committed locally. DeepGEMM was already clean at the measured candidate
commit. No installed package, historical worktree, or remote state was
modified.

## Validation and terminal matrix

| Requirement | Status |
|---|---|
| Same-source stock/candidate build and isolated JIT state | pass |
| Exact API-v1 ABI, eager correctness, return, stream, output ownership | pass |
| Independent graph leaf and W13→activation/quant→W2 region correctness | pass |
| No fallback/retry after selected candidate invocation | pass |
| Generated one-SM and two-SM topology/resource proof | pass |
| Three-series × 50-pair × four-estimator fairness, all 16 lanes | pass |
| Nsys submission/device/critical-path attribution | pass |
| Harness selftest and fail-closed contract tests | pass |
| SGLang provider and existing hotspot-registry tests | pass |
| Checkpoint-backed TP8/DP8/EP8 full-region and serving acceptance | **not run** |
| Production default | **stock / candidate off** |

Exact future commands and acceptance requirements are in
`external_tp8_commands.md`. Until they pass, enablement is not authorized.
