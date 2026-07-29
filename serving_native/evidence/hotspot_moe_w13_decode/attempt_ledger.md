# GLM-5.2 W13 decode attempt ledger

## Frozen scope

The only optimized path is fused W13 decode with E32, expert slab 1024,
K6144, N4096, packed `int32` UE8M0 scales, local decode buckets M16/M32, and
expected-M 4/5/8/9. SGLang starts from
`83d313104d089bcd2af26b28453ff880f1e6a80b`; DeepGEMM stock starts from
`731e7c7a97d269e4b9f482ea18d0e709a948f293`.

The current production call chain is:

```text
SGLang grouped_gemm_nt_f8f8bf16_masked
  -> glm52 hotspot dispatch
  -> API-v1 provider moe_w13
  -> exact expected-M-keyed infini_kernel symbol
```

Only the selected W13 node changes. The caller-owned BF16 output slab, stock
SwiGLU/packed quant, and stock W2 remain unchanged.

## Attempts and decisions

| Attempt | Hypothesis and exact delta | Evidence | Decision and why |
|---|---|---|---|
| A0 current-checkout stock controls | Independent same-source stock should close around 1.0 in the exact leaf/region eager/graph harness. | Three 50-pair series at expected-M4 produced estimator ranges 0.989699–1.000421 leaf eager, 1.000000–1.000845 leaf graph, 1.000000–1.004494 region eager, and 0.996434–1.008431 region graph. All were forced non-wins. | Retained. The controls establish the local noise/order envelope and prove the gate does not manufacture a win. |
| Direct named-device wrapper | Expose a compiled named symbol around the generated template without changing the template ABI. | The first direct wrapper failed compilation because its parameter annotations did not preserve the required `__grid_constant__` contract. | Rejected. A wrapper that cannot preserve the launch ABI is not a candidate. |
| By-value launch-parameter wrapper | Passing the generated parameter aggregate by value might preserve compiler annotations. | The binary used 94 registers and a 752-byte stack frame; runtime validation hit an illegal address. | Rejected. The wrapper introduced material local state and was incorrect. |
| Named by-reference entry | Forward the launch parameter by reference while retaining the grid-constant annotation and exact template constants. | Compilation, exact correctness, and generated-cubin identity passed. DeepGEMM commits `153628a`, `c8a842d`, and final `87e0359` preserve the required annotations and grid constants. | Retained as the source mechanism for both BM16 topologies. |
| Side-by-side stock/candidate without ELF isolation | Separate packages and cache roots should be sufficient to keep JIT context identities independent. | Loading stock then candidate in one process produced duplicate candidate cache keys with a second generated include hash. See `jit_context_duplicate_negative.json`. | Rejected. C++ symbol interposition allowed process-global JIT/parser state to cross DSO boundaries. |
| `-Wl,-Bsymbolic` only | Binding internal DSO references locally might close the duplicate-key problem. | Duplicate candidate contexts remained. See `jit_bsymbolic_only_negative.json`. | Rejected. Local binding alone did not hide every template/JIT static. |
| Hidden C++ visibility plus `-Wl,-Bsymbolic` | Identical hidden-visibility compilation and local binding for both arms should isolate their JIT/compiler statics. | Candidate cache contains exactly eight expected identities, stock contains one, no context duplicates remain, and stock/candidate build-plan hashes are identical. | Retained. This is the final fair-build isolation contract. |
| B2 BM16 one-SM `(16,128,128,11,1)` | One-CTA UMMA should remove cooperative cluster barriers while halving the masked-M surface. | Generated code proves `cta_group::1`, cluster size 1, 31 registers, zero stack/local/spills, 16 plain `UTCQMMA`, and no `UTCQMMA.2CTA` or `UCGABAR`. Correctness passed. Three 10-pair eager-leaf probes nevertheless had a required estimator below 1.03 for every expected-M: 1.02787, 1.02913, 1.02691, and 1.02887 for 4/5/8/9. | Rejected for promotion. Barrier removal did not compensate for loss of two-SM cooperative execution. |
| B1 BM16 two-SM `(16,128,128,12,2)` | Preserve the mature cooperative pipeline but cut masked-M MMA, TMEM drain, and store granularity from stock BM128 to BM16. | Generated code proves `cta_group::2`, cluster size 2, 35 registers, zero stack/local/spills, 16 `UTCQMMA.2CTA`, 10 `UTMALDG`, and only 4 `LDTM` versus stock 32. Exact eager and graph correctness passed. | Retained as the sole survivor. |
| Short B1 probes | A 3×10 probe can reject an obviously weak route before the full matrix. | Expected-M4/5 passed; expected-M8 had one 1.02912 estimator and expected-M9 had a noisy 0.95376 series. | Not treated as a verdict. The plan requires 3×50, and the complete matrix subsequently passed every estimator. |
| Full B1 fairness matrix | The two-SM BM16 reduction should survive every expected-M, eager/graph, leaf/region lane under alternating order. | All 16 lanes passed three 50-pair series and all four estimators per series. The global weakest estimate is 1.034255 at expected-M5 region eager. See `fairness_audit.json` and all `fair_bm16_2sm_*.json` files. | Promoted to external acceptance candidate. |
| Eager Nsys collection around the profiled runner | Nsys should attribute submission and device time without changing the timed topology. | The runner's nested Torch profiler attempted a second CUPTI subscriber and failed with `CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED`. The failed report is retained as `nsys_eager_cupti_conflict.nsys-rep.gz`. | Rejected collection method; no performance conclusion was drawn. |
| Graph-node Nsys survivor collection | A clean graph collection without the nested CUPTI subscriber should isolate W13 from submission and downstream nodes. | Balanced 58/58 stock/candidate launches show W13 p50 145.744→139.168 us and device critical span 226.176→218.816 us. Submission API p50 slightly increased by 0.361 us. See `nsys_attribution.json`. | Retained as attribution only; the unprofiled 3×50 results remain the performance authority. |
| NCU survivor collection | Invoke NCU only if generated-binary/resource evidence and Nsys leave a concrete device-code question. | PTX/SASS/resource audits already prove topology, TMEM/TMA/barrier forms, spill absence, and the reduced drain; Nsys attributes the region gain to W13. | Intentionally not run. No unresolved survivor question justified another intrusive profile. |

## Harness corrections

The first A0 and graph attempts exposed stale Task-24-specific auditor
assumptions (schema 2 and BM32 store granularity). Those results were rejected.
The auditor was updated fail-closed for the schema-3 BM16 identity and
BM16-derived graph store envelope, then covered by 33 contract tests. No
failed-audit result was used as performance evidence.

## Rollback

The provider is default-off. Leave `SGLANG_GLM52_OPT=0` or omit
`SGLANG_GLM52_HOTSPOT_MODULE`; startup then loads no candidate DSO and stock
SGLang remains authoritative. Unsupported ABI/topology/phase/recipe cases
select stock before candidate invocation. Once a supported call selects the
candidate, an error propagates and there is no stock retry.
