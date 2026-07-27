# GLM-5.2 indexer K/weights prefill profiling report

**Production symbol:** `Indexer._fused_q_prepare_and_store`  
**Fixed-model point:** rank-local prefill `M=4096`  
**Projection:** BF16 `[4096,6144] @ [160,6144]^T`  
**Target:** NVIDIA B200 / SM100 (148 SMs)  
**Collection:** Nsight Systems 2025.6.3; Nsight Compute 2026.1.1  
**Date:** 2026-07-22

## Scope and provenance

The authoritative reconstruction uses
`nvidia/GLM-5.2-NVFP4@aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa`.
Its checkpoint headers and actual SGLang dispatch make `indexer.wq_b` an
unquantized BF16 linear. RoPE is interleaved with maximum position 1,048,576
and base 8,000,000; index K uses the default FP32 LayerNorm (`eps=1e-6`).
`fixed_model_contract_cpu.json` verifies these facts.

`harness/profile_indexer_region.py` invokes the real unbound SGLang method with
`ReplicatedLinear`/`UnquantizedLinearMethod`, the official RoPE wrapper, a real
alternate stream, a `ForwardContext`, and the complete page-64 uint8 K cache.
It is a world-size-1 production-shaped reconstruction on deterministic
synthetic tensors, not a full model-module request.

The authoritative immutable performance series and all three matched Nsys
captures ran in one `with_flexible_gpu.sh` allocation on physical GPU 0
(`GPU-30b619de-87f2-1862-0d07-a595da8fe417`). Source, result, profile, and JIT
inputs are SHA-bound under
`../../evidence/glm52_prod_05_indexer_k_weights_prefill/hardened_runs/20260722T174049Z-immutable/`.
The validator recomputed every paired ratio and profiler mapping. Older corrected
and preliminary bundles on other wrapper-selected GPUs are historical only;
comparisons never divide results from different GPUs.

Earlier `nsys-stock`, `nsys-best`, `nsys-schedule-stock`, and `nsys-k-first`
reports used FP8 `wq_b` and generic RoPE. They are preserved but superseded for
region, stream, and Q/K conclusions. Only their isolated BF16
`wk_weights_proj` measurements transfer.

## Profiler-fidelity gate

The immutable stock profile is strongly perturbative:

| Measurement | Region latency |
|---|---:|
| In-capture CUDA event printed by `profile_indexer_region.py` | 1.085568 ms |
| Unprofiled stock baseline 1 | 0.144048 ms |
| Unprofiled stock baseline 2 | 0.143616 ms |
| Unprofiled stock baseline 3 | 0.175168 ms |

The Nsys run is 6.20x-7.56x the unprofiled medians. Therefore its 532.223 us
projected GPU span and 1.445249 ms host range are instrumented-trace properties.
They must not be reported as a production idle fraction or as proof that launch
overhead binds the unprofiled path.

The trace remains useful for executable identity, launch order, grid, stream
assignment, and profiled kernel duration. Large matched profiler deltas can
corroborate an unprofiled A/B result, but profiler ratios do not replace that
result.

## Fixed-model operation map under Nsys

The immutable profile contains four operations, with no activation-quant
kernel before `wq_b`:

| Role | Stream | Grid X | Profiled duration | Profiled gap after prior kernel |
|---|---:|---:|---:|---:|
| BF16 `wq_b` GEMM | alternate (stream 13) | 512 | 42.496 us | — |
| BF16 `wk_weights_proj` GEMM | current (stream 7) | 128 | 17.376 us | capture-local |
| fused Q RoPE/quant/gate | current (stream 7) | 32,768 | 35.872 us | capture-local |
| fused K norm/RoPE/quant/cache store | alternate (stream 13) | 1,024 | 3.168 us | capture-local |

There is zero kernel-on-kernel overlap inside this perturbed capture. This
establishes what Nsys observed, not that unprofiled production has zero overlap.
The source-level dependency chain is still exact: alternate-stream BF16 `wq_b`
is joined before Q; current-stream BF16 `wk_weights_proj` is joined before K;
the method finally waits for K before returning.

Authoritative raw reports are
`../../evidence/glm52_prod_05_indexer_k_weights_prefill/hardened_runs/20260722T174049Z-immutable/profiles/nsys-stock.nsys-rep`
and `nsys-torch-mm.nsys-rep` in the same directory; their validated mappings are
in that run's `validation.json`. The older `reports/nsys-exact-bf16-wq-*.nsys-rep`
and `analysis/exact_bf16_wq_range_analysis.json` are provisional corroboration.

## Matched candidate profiles

### Direct `torch.mm`

Under the same Nsys configuration, direct `torch.mm` launches the same four
operation classes and introduces no adapter kernel. Its instrumented projected
span is 0.967773x stock, host range 0.979158x, narrow BF16 kernel duration
0.992634x, and CUDA-event time 0.995549x. Those near-unity values are descriptive
only; the authoritative unprofiled fused prepare/store A/B results are
1.003540x, 1.032630x, and 1.002945x, so the improvement is not repeat-stable.

### Exact stock-linear single-stream branch

The immutable candidate retains the stock `ReplicatedLinear ->
UnquantizedLinearMethod` path and changes only `enable_dual_stream=False`. Nsys
puts all four kernels on stream 7. Its perturbed CUDA-event time is 0.979712 ms,
projected span 501.918 us, and host range 1.349585 ms, versus stock 1.085568 ms,
532.223 us, and 1.445249 ms. The stream assignment is structural, but those
instrumented timing ratios do not establish an unprofiled schedule win. The
authoritative paired results are 1.012753x, 0.985414x, and 0.978400x, so the
schedule is rejected.

The earlier adapter-contaminated single-stream bundle remains historical only.
Authoritative raw data is `nsys-single-stream.nsys-rep` plus
`abi-single-stream.json` in the immutable profile directory.

## Isolated BF16 projection NCU

`reports/full-stock-wk-m4096.ncu-rep` profiles the same isolated
M4096/N160/K6144 BF16 projection and library kernel seen in the immutable Nsys
operation map:

| Dimension | NCU evidence |
|---|---|
| Launch geometry | 128 CTAs, 0.865 waves/SM on 148 SMs; 20 SMs receive no CTA |
| Resources/occupancy | 255 registers/thread, 224,760 B shared memory/CTA, 12.5% theoretical and 8.84% achieved occupancy |
| Compute | 38.81% SM throughput; tensor pipe 26.64% of elapsed peak |
| Memory | 41.83% DRAM throughput, 3.205 TB/s read bandwidth |
| Stall/SASS | 499/648 attributed samples at `NANOSLEEP.SYNCS` after `SYNCS.PHASECHK.TRANS64.TRYWAIT` |

The 128-CTA grid, composed of two-CTA cooperative groups, and synchronization explain why a different
narrow-N tactic was worth testing. They do not predict a target-subregion win:
direct TGV, every FlashInfer tactic, and direct ATen all lose or remain neutral
in unprofiled A/B measurements.

The exact-Q/K NCU bundle is
`../../evidence/glm52_prod_05_indexer_k_weights_prefill/run_exact_ncu_campaign.sh`.
Three scheduler attempts returned exit 75 while the four-GPU lane held priority;
no corrected Q/K NCU report was produced. Archived Q/K counters from the
wrong-RoPE reconstruction are not substituted.

## Critical-path conclusion

The source dependency graph has two max-gated stages:
`max(wq_b, wk_weights_proj)` followed by `max(Q, K)`. In the immutable capture,
`wq_b` (42.496 us) is longer than wk (17.376 us), and Q (35.872 us) is longer
than K (3.168 us). The capture-local longer-branch chain is therefore wq_b + Q,
78.368 us; the targeted wk + K work is on the shorter overlap branches. This is
a topology plus capture-local diagnosis, not an absolute unprofiled timing
decomposition.

Because CUPTI expands the region more than sixfold, its 3.168 us K duration
cannot be divided by the unprofiled 0.143616-0.175168 ms region to construct a
production speedup bound. The available evidence
cannot safely localize the remaining unprofiled time among launch gaps, device
work, and overlap. The absolute bottleneck diagnosis is therefore
`none-identified`, not launch-overhead.

The old independent K GEMM does not transfer for a more direct reason: it is
not the reached implementation. Production folds K into the BF16 N160
projection and a short fused normalize/RoPE/quant/cache-store kernel. Every
measured replacement of the reachable BF16 projection regresses or stays below
the gate. The exact schedule-only control also fails the repeated region gate.

No candidate is promoted. Stock dual-stream SGLang remains the only enable and
fallback policy. A future whole-region fusion or lower-overhead instrumentation
experiment would need fresh fixed-model evidence and real TP8 validation; the
current trace does not authorize either conclusion or implementation.

## Distributed boundary

The first locked four-GPU attempt used a TP4/DP1/EP1 script and failed during
distributed initialization before a request; it is not valid TP4/DP4/EP4
evidence. A corrected allocation failed closed before CUDA launch on a package
origin comparison; commit `95060f3` fixes that check while preserving canonical
repo-venv provenance. A fresh corrected retry made 180 locked wrapper attempts,
all returned exit 75 under shared-host contention, and never executed. No live
TP4 route or performance claim is made. The production TP8/DP8/EP8
real-checkpoint numerical and end-to-end gate needs eight B200s and remains
externally unavailable. Neither lane is weakened or relabeled, and stock
fallback remains active.
