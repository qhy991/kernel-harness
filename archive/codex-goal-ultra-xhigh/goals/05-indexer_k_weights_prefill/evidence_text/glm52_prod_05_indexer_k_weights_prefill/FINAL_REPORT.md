# GLM-5.2 fused indexer K/weights prefill: final report

## Disposition: no replacement at the validated rank-local inner gate

No candidate is promoted for rank-local prefill M4096. This is a validated
inner-gate no-replacement result, not TP8 production acceptance. The immutable
fixed-checkpoint campaign finds no repeat-stable 3% improvement in the complete
fused prepare/store subregion: direct CuTe-DSL TGV severely regresses, direct
`torch.mm` clears 1.03x in only one of three repetitions, and the exact
stock-linear single-stream control regresses in two of three. All candidate
rows pass pre-timing and post-timing correctness. Fused-region rows additionally
pass fresh-seed dual-poison full-cache replay; isolated rows pass their BF16
output contract on shared deterministic inputs. Stock dual-stream SGLang remains the only enabled
implementation and the fallback for every shape, ABI, graph mode, and topology.

SGLang trial `a75a772a2` is preserved in local history and reverted by
`2fbd443a1`. The final `dsa_indexer.py` is byte-identical to stock. No production
flag, registry entry, or dispatch guard enables any external candidate.

## Fixed-model reachability

The target is
`nvidia/GLM-5.2-NVFP4@aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa`,
architecture `GlmMoeDsaForCausalLM`, with the balanced local prefill point
M4096 under the required TP8/DP8/EP8 deployment. Static source resolution reaches
the eager CUDA path:

`Indexer.forward_cuda -> Indexer._fused_q_prepare_and_store`

The pinned config, SGLang quantization dispatch, and ranged safetensors headers
prove the exact rank-local contract:

| Value | Contract |
|---|---|
| hidden input | BF16 `[4096,6144]` |
| q-lora input | BF16 `[4096,2048]` |
| `wq_b.weight` | unquantized BF16 `[4096,2048]`; output BF16 `[4096,4096]` |
| `wk_weights_proj.weight` | replicated BF16 `[160,6144]` |
| fused narrow output | BF16 `[4096,160]`, split into key 128 + gate 32 |
| RoPE/norm | interleaved, max position 1,048,576, base 8,000,000; FP32 LayerNorm eps 1e-6 |
| returned values | FP8 E4M3 Q `[4096,32,128]`; FP32 gates `[4096,32,1]` |
| cache side effect | complete page-64 uint8 K cache, including FP8 values and FP32 scale bytes |

The ModelOpt ignore list covers `self_attn`, the fixed recipe leaves
`SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN` unset, and SGLang resolves indexer `wq_b`
to `UnquantizedLinearMethod`. The optional FP8 loader conversion list contains
`q_b_proj`, not indexer `wq_b`. This corrects the initial reconstruction, which
used a non-production FP8 `wq_b` and generic RoPE. Its prepare/store-subregion,
K-before-Q, post-revert, and Q/K profiler claims are explicitly superseded in
`SUPERSEDED_CAMPAIGN.md`; only its isolated BF16 M4096/N160/K6144 projection
measurements remain transferable.

## Exact fused prepare/store workload and stream contract

Two serving-native workloads were added without changing a frozen GLM-5.2 task:

- `indexer_wk_weights_prefill_m4096` isolates the BF16
  M4096/N160/K6144 projection.
- `indexer_fused_prepare_store_prefill_m4096_eager_dual_stream` invokes the real
  unbound SGLang scheduling method with production SGLang linear wrappers, the
  official RoPE wrapper, a `ForwardContext`, a real alternate stream, and the
  full page-64 cache contract.

The target workload is a production-shaped rank-local reconstruction of the
fused prepare/store subregion, not a full `Indexer.forward_cuda`, score/top-k,
attention-backend, model-module, or distributed request. Its stock schedule is:

```text
current:   start/wait ─ wk BF16 ─ wait(wq) ─ Q RoPE/quant/gate ─ wait(K) ─ return
alternate: start/wait ─ wq BF16 ─ wait(wk) ─ K norm/RoPE/quant/cache store ─┘
```

All immutable candidate artifacts compare reference and candidate before and
after timing. The fused-region post-check rebuilds every input from a fresh
deterministic seed; the isolated projection post-check reuses its deterministic
inputs. Region Q and gates require exact dtype/shape and
`torch.allclose(rtol=2e-2, atol=2e-2)`.
Reference caches are independently poisoned with `A5` and `5A`; candidate
caches use `3C` and `C3`. Every full uint8 cache is then compared byte-for-byte,
including scale bytes, so an unwritten or partially written cache cannot pass.
The validator recomputes every sample summary, paired ratio, 3% gate, source
hash, profile mapping, and JIT-artifact hash.

CPU contract tests also cover BF16 row fusion, the block-FP8 dequantization call
and weight/scale rendezvous, BF16 output dtype/row placement, pending pair
handling, and fail-closed loader behavior. The fixed recipe disables indexer
LoRA. Static review found a
pre-existing stale fusion-flag import in the LoRA manager, so no LoRA validation
claim is made.

## Same-GPU paired results

Speedup is the runner-recorded median of interleaved per-pair
`reference_ms / candidate_ms`; marginal medians are never divided. Each series,
its controls, and its profiler collection remain on one wrapper-selected
physical GPU.

| Attempt | Workload | Three paired speedups | Decision |
|---|---|---|---|
| Direct SGLang CuTe-DSL TGV | immutable isolated projection | 0.329284x, 0.322368x, 0.361148x | reject: stable severe regression |
| Direct SGLang CuTe-DSL TGV | immutable fused prepare/store subregion | 0.564382x, 0.562612x, 0.553539x | reject: stable severe regression |
| Direct `torch.mm` | immutable isolated projection | 0.984024x, 0.995630x, 0.996183x | reject: neutral-to-slower |
| Direct `torch.mm` | immutable fused prepare/store subregion | 1.003540x, 1.032630x, 1.002945x | reject: only one run reaches 1.03x |
| Exact stock-linear single stream | immutable fused prepare/store subregion | 1.012753x, 0.985414x, 0.978400x | reject: no run reaches 1.03x |

The immutable stock-vs-stock controls are 0.990699x isolated and 0.997625x for
the fused prepare/store subregion, 60 pairs each. Reference-only immutable
region medians are 0.144048, 0.143616, and 0.175168 ms; these are descriptive
baselines, not candidate ratios. The entire campaign, including profiler
collection, ran on physical GPU 0
(`GPU-30b619de-87f2-1862-0d07-a595da8fe417`). The earlier corrected campaign
remains provisional historical evidence; its trends agree but its numbers are
not needed for the disposition.

The earlier isolated BF16 backend sweep is still valid because it does not
construct `wq_b` or RoPE. FlashInfer auto, cuBLASLt, cuDNN, CUTLASS, and TGV
score 0.217030x, 0.109375x, 0.300118x, 0.315262x, and 0.194219x respectively;
none justifies a target-subregion retry. The raw-file index, provisional
provenance, and both
authoritative/superseded scopes are in `paired_results_summary.{json,csv,md}`.

## Immutable profiler evidence and fidelity limit

The immutable exact-BF16 stock Nsight Systems range contains four operations,
not the five operations in the superseded FP8 reconstruction:

| Operation | Stream | Grid X | Duration |
|---|---:|---:|---:|
| BF16 `wq_b` GEMM | alternate | 512 | 42.496 us |
| BF16 `wk_weights_proj` GEMM | current | 128 | 17.376 us |
| Q RoPE/quant/gate | current | 32,768 | 35.872 us |
| K norm/RoPE/quant/cache store | alternate | 1,024 | 3.168 us |

The in-profile CUDA event is 1.085568 ms, while the three unprofiled stock
medians are 0.143616-0.175168 ms. This 6.20x-7.56x perturbation means the
captured 532.223 us projected span and 1.445249 ms host range are not production
bottleneck measurements. Nsys remains valid for fixed-model kernel identity,
order, grids, stream mapping, and facts explicitly labeled as instrumented.

Under the same profiler configuration, direct `torch.mm` has 0.967773x
instrumented span, 0.979158x host range, 0.992634x narrow BF16 kernel duration,
and 0.995549x CUDA-event time. These near-unity profiler values are
descriptive; the unprofiled paired results above drive the rejection.

The immutable single-stream capture uses the exact stock linear methods and
puts all four kernels on one stream; its event time is 0.979712 ms, projected
span 501.918 us, and host range 1.349585 ms. Those instrumented values are not
treated as production latency. The unprofiled three-run ratios above drive its
rejection. The preliminary adapter-contaminated single-stream capture is
retained only as exploratory history.

Valid isolated Nsight Compute evidence for the exact BF16 projection reports a
128-CTA, 0.865-wave grid, 255 registers/thread, 224,760 bytes shared memory/CTA,
12.5% theoretical and 8.84% achieved occupancy, 38.81% SM throughput, and
41.83% DRAM throughput (3.205 TB/s). Of 648 attributed samples, 499 land on its
cooperative `NANOSLEEP.SYNCS` wait. It is a constrained narrow-N tactic, but
every measured replacement loses. The exact Q/K NCU
retry returned scheduler exit 75 three times while the four-GPU lane held priority;
old wrong-RoPE Q/K counters are not substituted.

The source has two max-gated stages: `max(wq_b, wk)` then `max(Q, K)`. In the
immutable capture, wq_b+Q is the longer-branch chain
(42.496+35.872=78.368 us), while the targeted wk+K operations occupy the
shorter overlap branches. The captured 3.168 us K duration is not divided by
the unprofiled region: severe instrumentation perturbation makes such a
cross-mode production speedup bound invalid.

No absolute unprofiled bottleneck is safely identified: CUPTI perturbs the
short region too heavily to localize launch gaps, kernel work, and overlap. The
historical independent K GEMM also is not reached; production uses the fused
BF16 N160 projection plus K normalize/RoPE/quant/cache store. All tested
replacements of the reached projection fail the unprofiled region gate. Any
future whole-region fusion needs lower-overhead evidence and real TP8
validation; this trace does not authorize it.

## Source/build record

- Kernel-Harness base: `bcd005409e65786af82c86f621507ebef12b2766`.
- Immutable validation source commit: `727cc58f39bcfb11e3f7128a4ebe0d7ef72c2a1c`.
- TP4 provenance-check fix: `95060f35adbe5e6ea7f7cb35da78cd30d0e232a2`.
- SGLang base: `f93f8867b4bc124c9809c9110ec7361ed11b6b4a`.
- SGLang trial: `a75a772a2a113e2847a4baaabba89182f78f7ae8`.
- SGLang revert/final: `2fbd443a17edb4cb7cbf7c57d22cf6f435898948`.
- Final `dsa_indexer.py` SHA-256:
  `2399c19c2ecce0c16e33fdc75eccb49383ea1fdba84da1458e191dd39173b730`,
  equal to the stock source.
- No installed package was overwritten. External Python candidates import the
  pinned environment and keep compilation/tactic setup outside timed calls.
- Final SGLang net source diff from base is empty; local history preserves the
  source experiment and its rollback.

## Distributed validation boundary

The first four-GPU locked attempt used TP4/DP1/EP1 and is invalid for the
required diagnostic lane; it aborted during distributed initialization because
free memory was unbalanced. A corrected TP4/DP4/EP4 allocation later failed
before server launch because its provenance check compared canonical installed
package paths with the logical repo-venv path. That fail-closed artifact is
preserved under `tp4_live/20260722T180336Z-corrected/`; commit `95060f3` fixes
the path check while still requiring the repo-local launcher. A fresh corrected
retry then received exit 75 on all 180 wrapper attempts and never executed or
created a run directory; the blocker record is
`tp4_live/20260722T181018Z-canonical_scheduler_blocker.json`. There is no live
TP4 route or performance evidence, and the diagnostic is not relabeled as
acceptance.

The production numerical and end-to-end completion gate remains TP8/DP8/EP8 with a real
checkpoint. This host exposes four B200s, so that gate is externally blocked
and is neither weakened nor relabeled. The blocker cannot conceal a promotable
candidate: every immutable candidate already fails the necessary rank-local
fused prepare/store inner gate. The full score/top-k, selected DSA attention,
and end-to-end acceptance lanes remain unobtained. Stock fallback remains active.

## Final validation

- The immutable artifact manifest verifies in full, including raw paired samples,
  Nsys reports/databases, source snapshots, module origins, and JIT inputs/artifacts.
- `serving_native/selftest.py` passes all 41 fixed workloads; the GLM-5.2
  structural selftest passes all 24 tasks; and all 6 TP4 fail-closed attribution
  tests pass.
- The append-only knowledge base lints 15 entries with zero problems, and its
  query indices and distilled views are current.
- `verify_harness.py --json` returned `ok=true` through the flexible-GPU wrapper,
  which made canonical tensor tables visible to the task-projection check; this
  was a structural check, not a benchmark. Its non-strict
  pointer audit retains the advisory that `runs/index.jsonl` is absent because
  this serving-native campaign did not create frozen-task runs.
- Kernel-Harness has no changes to the frozen oracle, timing/anti-cheat code,
  generated task files, or `legacy/`; SGLang is clean and has no net source diff
  from its recorded base.

## Deliverables

- Call/ABI/checkpoint evidence: `reachability.md`,
  `fixed_model_contract_cpu.json`, `loader_contract_cpu.json`
- Exact serving workloads/tests: `serving_native/{workloads.py,runner.py,selftest.py}`
- Authoritative immutable paired/profile data:
  `hardened_runs/20260722T174049Z-immutable/`
- Historical paired data: `exact_bf16_wq/`, `exact_single_stream/`, and wrapper logs
- Paired tables: `paired_results_summary.{json,csv,md}`
- Profiler report/artifacts:
  `../../profile/indexer-wk-weights-prefill-m4096-20260722/`
- Attempt/fallback/review: `attempt_ledger.md`, `validation_matrix.md`,
  `humanize_review.md`, `SUPERSEDED_CAMPAIGN.md`
- Reproduction bundles: `run_exact_bf16_wq_campaign.sh`,
  `run_exact_single_stream_campaign.sh`, `run_exact_ncu_campaign.sh`, and
  `run_tp4_live_diagnostic.sh`
- Latest append-only superseding recipe:
  `testbench/knowledge/entries/glm52-prod--indexer-k-weights-prefill--b200--20260722c.json`

Serving-native JSON does not use the frozen-task `result.json` schema, so
`audit_result.py` does not apply. The immutable serving-native validator instead
recomputes raw sample summaries and paired ratios, validates dual-poison
pre/post correctness, binds candidate/runner/SGLang/JIT source hashes, checks
module origins, and verifies the final repository allowlist. Its artifact
manifest verifies in full, and its final state records unchanged committed
HEADs with no tracked diff. Historical dirty-tree campaigns remain explicitly
provisional and are not required for the final claim.
