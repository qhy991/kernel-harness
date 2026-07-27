# GLM-5.2 DSA prefill optimization report

Date: 2026-07-22

## Outcome

**Local decision: reject the source alternatives and retain stock. Production
disposition: externally blocked.** The reached checkpoint-free B200 path remains
stock FlashInfer/TRT-LLM Q64 `PersistentKeepsAb` for every bucket. This report
does not claim the plan's formal production no-replacement disposition: a real
checkpoint-backed request, live containing region, graph/end-to-end behavior,
and eight-rank acceptance could not be run in the available environment.

The source overlay can replay either the inherited 512-page compact pool or the
513-page physical pool. The decision-bearing experiment forced FlashInfer's
shipped Q32 or Q16 `PersistentSwapsAb` tactic on the exact 513-page rank-local
M4096 FP8 workload. Across three 30-pair runs per tactic in one B200 allocation:

| Tactic | Candidate p50, median of runs | Paired speedup p50 | Decision |
|---|---:|---:|---|
| stock Q64 Keeps control | 0.887040 ms | 1.000624x | neutral |
| Q32 Swaps | 1.608112 ms | 0.541023x | reject, 45.90% speed shortfall |
| Q16 Swaps | 2.924160 ms | 0.296514x | reject, 70.35% speed shortfall |

Neither Q32 nor Q16 clears the 3% leaf gate, so neither tactic is integrated or
promoted. The clean SGLang worktree is byte-identical to the stock backend
source and remains the fallback.

This is not a live-request or eight-rank acceptance result. The configured
model directory is empty and the host has four B200s, while production requires
TP8/DP8/EP8. Those gates remain unchanged and unpassed.

## Audited evidence status

| Requirement | Status and evidence |
|---|---|
| Current source call chain | traced: `GlmMoeDsaForCausalLM` -> DSA -> `DeepseekSparseAttnBackend.forward_extend` -> `_forward_trtllm(is_prefill=True)` -> FlashInfer `trtllm-gen` |
| Real checkpoint-backed request | externally blocked by empty checkpoint directory |
| Backend-class runtime hit | passed in checkpoint-free fixture; exactly one TRTLLM leaf hit |
| Exact named serving workload | added: backend fixture plus 513-page raw-pool leaf |
| Three rank-local stock controls | passed; backend-class reference median-of-medians 0.990112 ms |
| Full checkpoint-free backend region | profiled; 0.947472 ms CUDA-event mean over 20 iterations |
| Nsight Systems / Compute | complete for stock Q64, Q32, and Q16 in the exact raw-pool bundle |
| Source/backend attempt | complete; repo-local FlashInfer 0.6.12 header overlay, no installed-package overwrite |
| Representative leaf correctness | passed pre-timing comparison for exact scattered raw pool; compact trailing distribution also passed; nonmatching shape fell back |
| Four-GPU diagnostic | complete, rank-max stock controls and nsys, reported only as DP4 |
| Live full DSA region | blocked: no model projections or live indexer without checkpoint |
| TP8/DP8/EP8 and SGLang E2E | blocked: only four physical GPUs and no model |
| Final enablement | none; stock active for all inputs |

The continuation audit corrected several inherited labels without deleting any
raw evidence; see [`AUDIT_CORRECTIONS.md`](AUDIT_CORRECTIONS.md).

## Reachability and exact ABI

Current B200 default resolution selects FP8 DSA KV storage and independently
resolves prefill to `trtllm`. The backend's FP8 branch fuses RoPE and query/key
quantization, writes current K into the raw paged pool, consumes fused physical
top-k indices, and calls:

```text
flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
    backend="trtllm-gen", sparse_mla_top_k=2048)
```

The new backend hit trace records:

| Item | Runtime value |
|---|---|
| backend class / method | `DeepseekSparseAttnBackend.forward_extend` |
| mode | eager, stream 0, not capturing |
| entry Q-nope / Q-rope | BF16 `[4096,64,512]` / `[4096,64,64]` |
| entry K-nope / K-rope | BF16 `[4096,1,512]` / `[4096,1,64]` |
| leaf query | FP8 E4M3 `[4096,1,64,576]` |
| leaf KV | raw FP8 E4M3 `[513,1,64,576]` |
| sparse physical table | int32 `[4096,1,2048]`, dummy-page offset 64 |
| sparse lengths | int32 `[4096]`, all clipped to 2048 |
| maximum context / page size | 32768 / 64 |
| workspace | 384 MiB, zeroed once before first use |
| output | finite BF16 `[4096,1,64,512]` |

The extra page in the 513-page pool is not extra context: it is SGLang's
leading dummy page. The
inherited 512-page workload is retained as a compact replay, while
`dsa_trtllm_prefill_m4096_ctx32768_rawpool` is the decision-bearing physical
ABI.

The frozen `dsa_prefill_attn` task remains operationally mismatched: it invokes
`sgl_kernel.flash_mla.flash_mla_sparse_fwd` on flat BF16 tensors rather than the
reached paged-FP8 FlashInfer backend. No frozen-task result is used here.

## Baselines and containing region

Three fresh backend-class stock-vs-stock controls, each with 20 interleaved
pairs on one physical B200, give:

| Run | Reference p50 | Second stock p50 | Paired speedup p50 |
|---|---:|---:|---:|
| 01 | 0.990112 ms | 0.994096 ms | 0.998075x |
| 02 | 0.979568 ms | 0.979088 ms | 0.999608x |
| 03 | 0.993392 ms | 0.994288 ms | 0.998658x |

The profile of the same checkpoint-free backend region attributes its
0.947472 ms CUDA-event mean to:

| Component | Mean | Event-time share |
|---|---:|---:|
| Q64 sparse attention | 0.833439 ms | 87.964% |
| fused RoPE / FP8 conversion | 0.102480 ms | 10.816% |
| raw KV write | 0.003240 ms | 0.342% |
| residual stream/event gaps | 0.008314 ms | 0.877% |

This validates the backend boundary, cache preparation, and selected attention
leaf. It is not the complete production DSA region because live indexer
score/top-k, model projections, scheduler, and collectives require a real
checkpoint-backed request.

## Profiler diagnosis

Nsight Compute on the exact 513-page leaf reports:

| Metric | Q64 Keeps | Q32 Swaps | Q16 Swaps |
|---|---:|---:|---:|
| duration | 0.837344 ms | 1.584960 ms | 2.897600 ms |
| grid / waves per SM | 4096 / 27.676 | 8192 / 55.351 | 16384 / 110.703 |
| achieved occupancy | 24.821% | 24.683% | 24.983% |
| SM instructions | 364822748 | 713532076 | 1232297777 |
| global-load instructions | 57344 | 114688 | 229376 |
| shared-memory instructions | 4262208 | 13052224 | 21580096 |
| FP8 tensor operations | 1168231104512 | same | same |
| TMEM active | 69.48% | 34.84% | 22.07% |
| DRAM reads | 203.71 MB | 203.88 MB | 203.81 MB |
| DRAM read peak | 3.17% | 1.68% | 0.92% |
| shared-store bank conflicts | 147 | 1.851M | 4.111M |
| spill instructions | 12928 | 105408 | 0 |

All tactics remain one CTA per SM at 25% theoretical occupancy. Q32/Q16 do the
same tensor math and read the same bytes, but create 2x/4x more waves and
global loads, 3.06x/5.06x more shared-memory instructions, and millions of new
shared-store bank conflicts. Q32 also increases spills 8.15x. Active-cycle
max/min stays below 1.05x, so tail imbalance is not the issue.

The compact-pool source-counter reports localize dominant long-scoreboard
sampling to cross-warp `SYNCS.PHASECHK...TRYWAIT` and `NANOSLEEP.SYNCS` sites.
The exact raw-pool cubins have the same instruction/resource signature; treating
that localization as applicable is an inference, not a second source-counter
capture. HBM bandwidth, launch count, and terminal tail are not the binding
limits. The generated producer/consumer pipeline and sparse-gather instruction
work are.

Nsight Systems independently gives target-kernel means of 0.831094 ms,
1.570242 ms, and 2.890240 ms for Q64/Q32/Q16. The profile drivers pop CPU NVTX
ranges before event synchronization; CUDA-event timing and runtime/GPU
correlation joins are authoritative, not temporal NVTX filtering.

## Attempts and source provenance

### PDL policy (historical, corrected)

The direct external `enable_pdl=False` candidate was neutral/slower. The earlier
0.996817x pooled result is valid leaf evidence but was mislabeled as timing the
temporary SGLang integration. The guarded SGLang commit existed and was tested,
then reverted; it was never performance-measured through that dispatch.

### Q32/Q16 source tactic oracle (decision-bearing)

The overlay copies two Apache-licensed headers from FlashInfer tag `v0.6.12`,
peeled commit `d768c14e7cf5dd5df45a8a1de78ae815879f108a`. Their hashes match that
commit and the installed 0.6.12 package byte-for-byte before the selector edit.
Beyond mechanical relative-to-installed include-path relocation, the only C++
behavioral delta is an exact dtype/shape/top-k predicate selecting shipped Q32
or Q16 Swaps cubins instead of Q64 Keeps.

FlashInfer JIT builds unique
`fmha_gen_glm52_dsa_swaps_q{16,32}_v2.so` libraries at candidate import time,
outside timing. The candidate clones the wrapped Python function's globals to
replace only its module getter; it neither monkey-patches the stock reference
nor overwrites the installed package. Nonmatching inputs call stock.

The workload, source overlay, bundle drivers, raw outputs, and profiler reports
are committed locally in Kernel-Harness commit
`d00181769a6041dc3803de056a1f70cafdb9d483`.

Expected effect: more head-splitting CTAs might expose additional parallelism
if Q64 were under-filled. Measured effect: occupancy is unchanged while waves,
shared-memory work, conflicts, and synchronization multiply. Both tactics are
rejected and the rollback point is simply the untouched stock FlashInfer call.

## Correctness and topology

Every persisted candidate result was written only after the serving runner's
mandatory structure/shape comparison and FP32-cast `allclose` check. The runner
does not separately require matching floating dtype or reject equal infinities.
The exact raw-pool scattered workload passed for Q32 and Q16. The compact replay
also passed the trailing distribution, and the eight-request mismatch fell back
to stock. The separate backend hit trace checked finite BF16 output and the
exact leaf ABI.

The four-GPU workload replays this unchanged rank-local leaf independently on
all DP4 ranks and reports maximum rank latency. Its stock controls and nsys
artifact are useful topology diagnostics only. The leaf event contains no
production attention collective; NCCL is used outside it for runner barriers
and rank-max reduction. This is not TP8/DP8/EP8 and does not alter the
production gate.

Three 20-pair DP4 controls report reference rank-max p50s of 0.911072,
0.940944, and 0.941648 ms, with paired speedups 1.003564x, 1.004960x, and
0.986111x. NSYS places the timed target-kernel rank-max median at 0.834369 ms
(0.832801–0.835105 ms). The larger profiled event interval includes
NSYS-amplified host launch gaps; the raw report and caveats are in the DP4
profile summary.

## External validation blocker

`/mnt/OS-oKqEXySb/models/GLM-5.2-NVFP4` contains no files, no other cached
GLM-5.2 checkpoint was found, and this host has four B200s. Consequently the
following remain unperformed and unpassed:

- a real-request trace proving launch flags and live request/top-k distribution;
- live model projections, indexer score/top-k, cache preparation, and full DSA
  region baselines;
- scheduler/stream and graph/overlap validation in the actual server;
- TP8/DP8/EP8 rank-max correctness and latency;
- complete SGLang prefill and end-to-end candidate comparison.

No gate is weakened or relabeled. A future candidate must retain the exact
eight-rank requirements in [`external_validation_blocker.md`](external_validation_blocker.md).

## Final enable/fallback policy

- `SGLANG_GLM52_OPT=0` remains the measurement reference.
- No DSA prefill bucket is enabled.
- No Q32/Q16 overlay is imported by SGLang.
- No explicit PDL override remains.
- Every shape, ABI, topology, and graph mode uses stock FlashInfer behavior.
- The SGLang worktree is clean at rollback head
  `5a444f66cf5764d2d76003a3a4c4631af152a253`; the reached backend file is
  byte-identical to stock base `f93f8867b4bc124c9809c9110ec7361ed11b6b4a`.

Raw paired JSON, environment metadata, logs, NCU reports, NSYS reports, source
overlay, workload code, and per-profile summaries are retained under the four
new backend-fixture, compact-tactic, raw-pool-tactic, and DP4 profile
directories. Source hashes and post-measurement documentation provenance are in
[`source_overlay_manifest.md`](source_overlay_manifest.md); experiment routing
is in [`prior_art.md`](prior_art.md). No branch was pushed and no remote state or
installed package was modified.
