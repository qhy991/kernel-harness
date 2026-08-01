# GLM-5.2 B300 KV-cache serving nsys bottleneck audit

Date: 2026-08-01

## Decision

The largest observed prefill loss is not an isolated NCCL or attention kernel. It is
DP-rank arrival skew that turns one request burst into multiple full-model EP passes.
The first executable task was therefore to make `PrefillDelayer` usable with the
synchronous scheduler currently required by this B300 environment. That support was
implemented and validated in SGLang commit `ae905bb23`.

The result changes the next-action decision. Coordinating DP ranks removes the
one-rank-ahead shape and improves a small BS8 cached-prefill probe, but it does not
collapse the production BS128 request into one pass: every rank still executes a
small pass followed by the remainder. A more aggressive idle-queue trigger was
tested through a fresh 8-GPU restart and rejected because it still produced a
`2 + 14` split and no end-to-end win. Do not continue tuning delay parameters without
first making the entire local burst visible to the scheduler's admission decision.

Full overlap scheduling remains a separate environment task. The installed DeepEP
binding lacks every SGLang overlap argument, so this is a build/version-alignment
problem rather than a keyword-adapter problem. `PrefillDelayer` itself does not
require overlap scheduling; it already has a synchronous device/NCCL-group path.

The highest-leverage compute-only task is the decode MoE region:

    W13 K4096->6144
      -> silu_mul_quant_varlen
      -> W2 K6144->2048

It accounts for about 41.2% of summed target-decode GPU kernel time. The two GEMMs
are already partly optimized; silu_mul_quant_varlen is the largest untouched compute
kernel and should be fused into the adjacent GEMM epilogue or prologue.

These are two different production boundaries:

1. DP scheduling / DeepEP arrival synchronization: multi-rank serving-only evidence.
2. MoE GEMM + activation + quantization: single-GPU mechanism evidence is useful,
   followed by multi-rank serving confirmation at the traced KV context.

After the scheduling experiment above, the decode MoE chain is the next code-level
optimization target with the clearest measurable upside. Restoring overlap still has
high potential, but requires rebuilding or pinning DeepEP before SGLang work can be
evaluated honestly.

## Trace and request scope

Primary trace:

    /mnt/b300-shared/home/qinhaiyan/wwxq/bench_results/
      nsys_s32768_decode_winners_20260730T182858Z/

Relevant SGLang source:

    /mnt/b300-shared/home/qinhaiyan/wwxq/SGLang-DGMK

Trace-run source commit:

    725500abe

The measured request is a production-shaped cached serving request:

| axis | value |
|---|---:|
| global batch | 128 |
| input length | 32768 |
| cache-hit ratio | 0.998 |
| existing prefix KV per request | 32704 |
| new prefill tokens per request | 64 |
| requested output tokens | 48 |
| local decode M per DP rank | 16 |
| opt0 TTFT | 2.8231 s |
| opt0 latency | 4.7854 s |
| opt0 mean ITL | 40.881 ms |

The 180-second capture contains server startup, warmup, cache population and the
target request. Whole-trace percentages therefore cannot be reported as target
decode percentages. DeepEP mode transitions and request logs identify the useful
windows:

| phase | trace window | wall duration |
|---|---:|---:|
| incremental prefill | 88.289-91.113 s | about 2.823 s |
| cached decode | 91.113-93.049 s | about 1.936 s |

All percentages below are shares of summed GPU kernel duration across the eight
GPUs inside the stated phase, not direct wall-clock shares and not Amdahl speedup
predictions.

## Executed task: synchronous PrefillDelayer

### Production test lane

The experiment used the real local weights, not a dummy or remote model:

    /mnt/b300-shared/models/GLM-5.2-FP8

The server lane was eight B300 SXM6 GPUs with TP8, DP8, EP8, DP attention, DeepEP,
FP8 KV cache, graph maximum local batch 16 and `disable_overlap_schedule`. The target
request exactly preserved the trace shape: global BS128, input 32768, prefix KV
32704, extend 64 and output 48. Results and server logs are retained under:

    /mnt/b300-shared/home/qinhaiyan/wwxq/bench_results/
      prefill_delayer_sync_20260801/

Implementation commit:

    ae905bb23 fix(scheduler): allow prefill delayer with sync scheduling

The change removes an obsolete configuration assertion and exercises the existing
device-group collective path when overlap scheduling is disabled. It does not alter
the default configuration. Verification completed on both target environments:

| check | result |
|---|---|
| B300 4-rank Gloo negotiation tests | 2/2 passed |
| Verda 4-rank Gloo negotiation tests | 2/2 passed |
| B300 8-rank NCCL device-group smoke | passed |
| Verda 4-rank NCCL device-group smoke | passed |
| real GLM-5.2 TP8/DP8/EP8 initialization | all ranks initialized |

### Fresh-restart A/B

Each side below used a new server process. The baseline and treatment used the same
patched source; only `enable_prefill_delayer` changed, avoiding a source-version
confound.

| workload | baseline | sync PrefillDelayer | change |
|---|---:|---:|---:|
| BS8, KV32768 + extend1024, latency | 0.5422 s | 0.2982 s | -45.0% |
| BS128, KV32704 + extend64, latency | 4.3209 s | 4.3065 s | -0.33% |
| BS128, KV32704 + extend64, TTFT | 2.6850 s | 2.6516 s | -1.24% |
| BS128, KV32704 + extend64, derived ITL | 34.081 ms | 34.477 ms | +1.16% |

The BS8 result confirms that the mechanism can help a small synchronized cached
burst. The production-shaped BS128 result is effectively neutral: its small latency
and TTFT movement is not enough to claim a win, and decode ITL moved in the wrong
direction within normal run-to-run noise.

Logs show why. The original request was `DP0: 1`, followed by the remainder. The
sync delayer aligns the ranks, but each rank still observes only one or two requests
on its first admission and then executes the remaining fourteen or fifteen in a
second 75-layer EP pass. Rank synchronization alone cannot batch requests that have
not yet entered the scheduler waiting queue.

Same-process repetitions later produced about 2.79 s latency and 1.15 s TTFT. They
are deliberately excluded from the table: model/JIT/cache warm state differs from
the fresh baseline, so treating them as an A/B result would create a false speedup.

### Rejected idle-burst experiment

An opt-in trigger was prototyped to wait for every rank's idle queue to reach a
fraction of local prefill capacity. It included a separate wall-clock budget and a
phase-reset fix so prior mixed-rank waiting could not consume the idle budget. Unit
tests passed, then these serving variants were exercised:

- idle caps of 20 ms, 200 ms and 500 ms;
- `batch_notify_size=128` alone and combined with the idle trigger;
- `parallel_batch` combined with notification and idle waiting;
- an explicit fresh server restart for the final phase-reset configuration.

The final fresh run still executed `2 + 14` sequences per rank and measured 4.3311 s
latency, 2.6841 s TTFT and about 34.31 ms derived ITL. Against the fresh baseline this
is no latency or TTFT improvement. The prototype was therefore reverted and was not
committed.

One hot-process probe changed the order to `14 + 2` and reported 3.2811 s latency /
1.6402 s TTFT. It is not promotion evidence: a fresh restart reverted to `2 + 14`,
showing that the apparent gain was dominated by process/JIT warm state rather than
the queue policy.

Two additional configurations are explicit no-go results:

- a 0.2 token-usage watermark bypassed coordination at the observed usage near 0.02;
- 3000 delay passes held a singleton for about 3.55 s and timed out, which is not a
  production-safe default.

The next scheduler experiment must instrument tokenizer-to-scheduler arrival time,
per-rank queue visibility and notification boundaries. Its mechanism must make all
16 local requests available before the first expensive EP pass; increasing a delay
after requests are still hidden upstream cannot meet that goal.

## Incremental-prefill result: arrival skew dominates

For prefix KV 32704 plus 64 new tokens per request:

| category | summed GPU kernel share | cross-rank CV |
|---|---:|---:|
| DeepEP | 72.44% | 0.901 |
| NCCL | 8.30% | 0.248 |
| DeepGEMM | 6.83% | 0.043 |
| FlashMLA | 5.27% | 0.039 |
| other | 3.34% | - |
| MoE auxiliary | 1.32% | - |
| MQA | 0.75% | - |
| MLA helpers | 0.68% | - |
| quantization | 0.64% | - |
| norm/RoPE | 0.39% | - |
| router | 0.05% | - |

The dominant exact kernels are:

| kernel or region | share |
|---|---:|
| DeepEP normal notify_dispatch | 61.15% |
| NCCL AllGather | about 8.4% |
| DeepEP cached_notify_combine | 5.94% |
| FlashMLA sparse attention | 5.16% |
| W13 MoE GEMM | 3.12% |
| DeepEP combine data movement | 3.11% |
| DeepEP dispatch data movement | 1.91% |
| W2 MoE GEMM | 1.70% |

### Root cause: one burst became two 75-layer EP passes

Each GPU executed 150 normal DeepEP dispatches:

    75 MoE layers x 2 prefill passes

The server logs explain the split:

1. DP0 processed one sequence with 64 new tokens and 32704 cached tokens.
2. DP0 then processed the remaining 15 sequences.
3. DP1-DP7 processed 16 sequences each in the second pass.

At the largest notify_dispatch:

| rank group | entry time |
|---|---:|
| GPU0 | 89.282843 s |
| GPU1-GPU6 | 90.7339-90.7548 s |
| GPU7 | 90.764179 s |
| common completion | about 90.764198 s |

GPU0 spun for about 1.481355 seconds in this event. The kernel name makes this look
like communication cost, but the event is principally a rank-arrival barrier caused
by uncoordinated DP batching.

Within the incremental-prefill window, estimated rank-entry wait accounts for:

| operation | fraction of its observed duration attributable to entry wait |
|---|---:|
| notify_dispatch | 99.35% |
| cached_notify_combine | 96.54% |
| NCCL | 87.92% |

Those entry waits sum to about 3.839 seconds across GPUs, or about 73.7% of all
summed prefill GPU kernel time. Removing them arithmetically leaves roughly 25.8%
residual communication share, but this adjusted number is diagnostic only; it is
not a predicted speedup.

Measured communication/compute overlap is effectively zero in this phase. NCCL has
no compute overlap, and sampled DeepEP overlap is zero or only tens of microseconds.
The waits are therefore on the serialized critical path.

### Independent 64K corroboration

The dedicated capture

    /mnt/b300-shared/home/qinhaiyan/wwxq/bench_results/nsys_prefill_64k/

shows the same structure:

- raw communication share: about 79.7%;
- DeepEP dispatch share: about 72.3%;
- DeepGEMM and FlashMLA compute are balanced across ranks;
- GPU5 is the last dispatch starter in 88.39% of 25633 aligned events;
- dispatch start skew averages 2.137 ms, with p50 2.266 ms and p90 3.074 ms;
- communication/compute overlap remains below 1%.

The full capture includes internal warmups, so its aggregate percentages are
structural corroboration rather than a pure M1024/M2048 phase estimate. The
segmented 32K cached request is the cleaner serving evidence.

## Decode result: a balanced but dominant MoE compute chain

The target decode window contains 47 generation intervals after the cached
incremental prefill.

| category | opt0 run 1 | opt0 run 2 | rank balance |
|---|---:|---:|---|
| DeepGEMM | 49.86% | 50.40% | CV about 0.004 |
| DeepEP | 14.45% | 14.18% | CV about 0.053 |
| silu_mul_quant_varlen | 8.48% | 8.57% | CV about 0.002 |
| FlashMLA | 6.52% | 6.58% | CV about 0.019 |
| norm/RoPE | 3.89% | similar | small |
| per-token quantization | 2.56% | similar | small |
| NCCL | 2.51% | 1.90% | CV about 0.343 |
| MQA | 0.98% | similar | small |
| router | 0.71% | similar | small |

Exact call counts map the main chain to every MoE layer and generation step:

    3525 calls/GPU = 75 layers x 47 decode steps

| operation | target-decode share |
|---|---:|
| W13 DeepGEMM, K4096->6144 | 21.10% |
| silu_mul_quant_varlen | 8.46% |
| W2 DeepGEMM, K6144->2048 | 11.61% |
| combined region | about 41.2% |

The top MoE GEMMs have rank CV below about 0.6% and max/min duration near 1.02.
Decode MoE compute is therefore well balanced. The opportunity is fusion and
throughput, not decode-time expert placement.

The existing combined-winners bundle improved median ITL from 40.436 ms to
38.612 ms, or 1.0472x across two runs. In one phase-aligned comparison:

| stack component | opt0 summed time | winners summed time | change |
|---|---:|---:|---:|
| FlashMLA sparse + combine | 917.2 ms | 753.5 ms | -17.8% |
| W13 GEMM | 2975.1 ms | 2788.1 ms | -6.3% |
| W2 GEMM | 1637.7 ms | 1480.4 ms | -9.6% |
| silu_mul_quant_varlen | about 1193.5 ms | essentially unchanged | no win |

This makes silu_mul_quant_varlen the clearest untouched compute target. Optimizing it
as another standalone launch leaves synchronization and global-memory traffic in
place; the preferred mechanism is a W13 epilogue or W2 input/prologue fusion.

## Decode NCCL result: control-plane skew, not collective bandwidth

Each GPU executes exactly 48 NCCL AllGather calls, matching one scheduler/DP
synchronization per output token rather than one collective per model layer.

| signal | observation |
|---|---:|
| fraction of NCCL duration before the last rank arrives | 98.59% |
| start-skew p50 | 81 us |
| start-skew p90 | 5.725 ms |
| start-skew p99 | about 15.742 ms |
| maximum start skew | 15.891 ms |
| completion spread after all ranks arrive | typically about 13 us |
| NCCL overlap with compute | zero |

Long NCCL windows show less than 0.1% non-NCCL GPU activity, including on the late
rank. The GPUs are idle; changing the NCCL ring or protocol is not the first action.

A representative step launches a CUDA graph on all workers, then performs a
128-byte device-to-host copy:

- actual GPU DMA: about 3.1-3.7 us;
- observed CUDA API interval: about 32.5-34.0 ms;
- some ranks then show another 16.24-16.25 ms host-side gap before the next CUDA API.

The trace does not contain sufficient OS scheduling tables to distinguish CFS
preemption, Python/runtime scheduling, lock contention or CPU affinity. The next
capture must include OS runtime/context-switch tracing.

## Expert-placement imbalance: important outside clean decode

The independent expert recorder contains 324508800 routed assignments over 256
experts and EP8:

| statistic | value |
|---|---:|
| per-expert CV | 28.63% |
| minimum expert / mean | 0.533 |
| maximum expert / mean | 2.052 |
| aggregate EP-rank CV | 5.826% |
| heaviest/lightest EP rank | 1.217x |
| per-layer rank-CV median | 0.351 |
| per-layer rank-CV p90 | 0.492 |
| worst layer rank CV | 0.679 |
| worst representative max/min | 4.82x |
| extreme layer max/min | 10.04x |

Normal-mode W13 and W2 GEMM rank means correlate with the independent routed-token
counts at 0.988 and 0.994. Rank6 is about 15% slower than the light ranks in that
traffic. The hot rank changes by layer, so aggregate whole-model placement hides
layer-local imbalance.

A prior static EPLB attempt regressed TTFT from 2.326 s to 2.448 s, about 5.3%.
Do not promote static EPLB from these statistics. Test layer-aware online placement,
water-filling or redundant hot experts and gate both TTFT/ITL and slow-rank GEMM.

## Source boundaries

| concern | SGLang source |
|---|---|
| DeepEP low-latency dispatch/combine | python/sglang/srt/layers/moe/token_dispatcher/deepep.py |
| DP batch synchronization | python/sglang/srt/managers/scheduler_components/dp_attn.py |
| overlap scheduling and async D2H | python/sglang/srt/managers/scheduler.py |
| global prefill readiness/delay | python/sglang/srt/managers/prefill_delayer.py |
| DP-attention collectives | python/sglang/srt/layers/dp_attention.py |
| DSA indexer communication | python/sglang/srt/layers/attention/dsa/dsa_indexer.py |

With `disable_overlap_schedule` enabled, `MLPSyncBatchInfo` selects the device/NCCL
group and the scheduler serializes the tiny D2H result path. PrefillDelayer mirrors
that choice: it uses the device group for the synchronous scheduler and the CPU
group for the overlap scheduler. The former path was already implemented; a stale
configuration assertion was the only reason the combination could not start.

The separate overlap-scheduling blocker is a substantive DeepEP/SGLang ABI mismatch
in the Blackwell combine path. The installed binding is:

    low_latency_combine(
        x, topk_idx, topk_weights, handle,
        use_logfmt=False, zero_copy=False, async_finish=False,
        return_recv_hook=False, out=None,
        combine_wait_recv_cost_stats=None,
    )

It has no `overlap`, `src_signals` or `src_signal_expect_value` capability at all.
There is consequently no safe parameter-name adapter to select. Restore overlap by
building/pinning a DeepEP revision compatible with this SGLang call contract, then
fail early on capability mismatch. Silently swallowing the arguments would disable
the intended mechanism while making the configuration look successful.

## Ranked execution plan and acceptance

### P0-A: align DeepEP and restore overlap scheduling

1. Pin or rebuild DeepEP with the overlap-capable `low_latency_combine` contract.
2. Add an explicit startup capability check that reports the installed signature.
3. Remove `disable_overlap_schedule` only after an eight-rank smoke test.
4. Verify eager and CUDA-graph correctness, shutdown, and absence of deadlock.

Acceptance:

- no unexpected-keyword or DeepEP ABI failure;
- no correctness drift;
- at least ten paired serving repetitions;
- target decode NCCL p99 arrival skew below 1 ms, or the GPU NCCL sync disappears
  from this scheduler boundary;
- ITL improves without TTFT regression.

### P0-B: coordinated DP incremental prefill -- executed, mechanism incomplete

Commit `ae905bb23` makes synchronous PrefillDelayer valid and tested, so this no
longer depends on P0-A. The production-shaped A/B shows that rank coordination alone
does not make the full request burst visible before admission. Do not ship the
reverted idle-queue prototype or continue blind delay tuning.

The follow-up should move batching to the tokenizer/DP-controller notification
boundary, or otherwise enqueue one local atomic burst before negotiation. Add
timestamps and queue-depth telemetry first so the mechanism is falsifiable.

Acceptance at prefix 32704 plus extend 64, BS128:

- one 75-layer normal DeepEP pass per rank, not two;
- exactly 16 local sequences in that pass for this balanced BS128/DP8 case;
- notify_dispatch wait and TTFT p50/p99 decrease;
- no starvation or throughput regression under mixed prefill/decode traffic.

### P0-C: fuse the decode MoE region -- next code optimization

Use the existing moe_swiglu_quant_decode contract as the local mechanism gate, but
optimize the containing W13-to-W2 region rather than only replacing the standalone
activation kernel.

Acceptance:

- exact production FP8 checkpoint scale/layout and masked-row semantics;
- eager and CUDA-graph correctness;
- no M bucket regression;
- containing-region win;
- end-to-end ITL win at KV about 32K and BS128.

### P1: host scheduling and normal-mode expert balance

- Bind each worker/scheduler process to a separate physical CPU core and GPU-local
  NUMA node.
- Capture OS runtime and context-switch data around the 11-16 ms host gaps.
- Evaluate layer-aware online EPLB or redundant experts; do not reuse the regressed
  static placement as a default.

### P2: remaining kernels

- FlashMLA remains about 5.7-6.5% after the existing sparse/combine win.
- Per-token quantization is about 2.6%.
- norm/RoPE is about 3.9%.
- router is below 1%.

These remain valid fusion candidates, but they are lower priority than restoring
overlap and fusing the 41.2% MoE chain.

## Interpretation guardrails

- Never use the whole 180-second capture as a decode-only denominator.
- Report summed multi-GPU kernel shares separately from wall-clock phase latency.
- Arrival-wait subtraction is a diagnosis, not a speedup forecast.
- A single-GPU kernel win cannot validate DP/EP synchronization.
- Compare scheduling changes with independent fresh server restarts; same-process
  JIT/cache repetitions are stability data, not a baseline/treatment pair.
- Separate the benchmark's prefix-cache population request from the measured cached
  incremental-prefill request when segmenting logs and traces.
- A decode compute win transfers more plausibly when rank times are balanced, but
  it still requires the traced KV-cache, batch, graph and topology end-to-end lane.
- Exact DeepEP LL dispatch/combine arrival fractions are not reported because ordinal
  alignment is ambiguous; raw share, rank CV, tails and measured overlap are used.
