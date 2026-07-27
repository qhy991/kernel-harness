# Variant and source-attempt ledger

## Reference characterization

The stock packed kernel selects `swap_ab=1`, block
`M240 × N128 × K128`, cluster `1 × 2`, load tile `120 × 128`, store tile
`16 × 128`, six pipeline stages, 148 SMs, and 256 threads. DeepGEMM reports six
internal waves and 124 tiles in the last wave.

Nsight Compute measured a compute-bound kernel: tensor-pipe activity was 96.51%
of active cycles and 91.28% of elapsed cycles, while DRAM reads consumed about
22.55% of peak. The 209.7 KB dynamic shared-memory allocation permits one CTA
per SM and 12.5% occupancy. The exact cubin uses 42 registers per thread with
zero stack, local memory, or spill traffic. Long-scoreboard and barrier stalls
were the largest sampled issue blockers.

## Attempt 1: existing compiled-dimension dispatcher

- Hypothesis: specializing the known `N` and `K` dimensions can remove dynamic
  address/control work without changing the packed ABI.
- Delta: opt-in SGLang `glm52_opt` dispatch for `o_proj`, prefill, M4096, calling
  `deep_gemm.fp8_gemm_nt(..., compiled_dims="nk")`.
- Expected effect: smaller generated control path and fewer dynamic
  instructions; identical TMA/tensor schedule, output, and scales.
- Correctness: passed at the leaf and `Fp8LinearMethod.apply` region.
- Paired result: the three leaf medians were 1.001274×, 0.994634×, and
  1.006626×; the three full-region medians were 1.009760×, 1.001332×, and
  0.997545×. No series reached the 1.03 gate.
- Profiler delta: Nsight Compute reduced average executed instructions from
  about 25.64 K to 22.54 K and registers from 42 to 41, but duration changed
  only from 256.864 µs to 256.130 µs. Tensor elapsed utilization remained near
  92%; long-scoreboard and barrier ratios increased.
- Risk: an explicit production dispatch adds a second selection/import path
  without a stable latency win.
- Decision: do not promote. Keep it available only as the pre-existing explicit
  ablation path.

## Attempt 2: supported DeepGEMM configuration sweep

The paired sweep kept the exact packed tensors and tested only public
DeepGEMM call parameters:

| Variant | Paired p50 speedup | Correct |
|---|---:|---|
| direct default | 1.035002× | yes |
| `compiled_dims="nk"` | 1.034094× | yes |
| `compiled_dims="mnk"` | 1.040949× | yes |
| PDL off | 1.057960× | yes |
| 146 SMs | 1.054310× | yes |
| 144 SMs | 1.049850× | yes |
| 144 SMs + NK | 1.059194× | yes |
| 144 SMs + MNK | 1.055712× | yes |
| 140 SMs | 1.045723× | yes |
| 136 SMs | 0.979550× | yes |
| 132 SMs | 0.972199× | yes |
| 128 SMs | 0.986559× | yes |

These are diagnostic, not deployment wins. The reference arm enters through
the SGLang production wrapper while every sweep candidate calls the DeepGEMM
leaf directly; the `direct default` row already shows a 3.5% delta without a
kernel configuration change. PDL and SM-count rows additionally use different
kernel launch policies. When the existing dispatcher is measured through the
full production region, the apparent gain disappears. A 144-SM launch fills
the final internal wave more evenly, but it did not produce a validated
production-path win and therefore was not integrated.

## Attempt 3: five-stage DeepGEMM source specialization

- Hypothesis: the stock 209.7 KB shared-memory footprint, long scoreboards, and
  barrier stalls suggested that removing one buffering stage might reduce
  synchronization/transaction pressure while preserving the tensor schedule.
- Exact source delta: in
  `third_party/DeepGEMM-GLM52/csrc/jit_kernels/heuristics/sm100.hpp`, return a
  five-stage pipeline only for normal 1D1D FP8 GEMM with
  `M=4096, N=6144, K=16384`, `swap_ab=1`, block `240 × 128 × 128`, and the
  otherwise stock six-stage configuration. The preserved patch is
  `source_experiment.patch`.
- Expected low-level effect: reduce pipeline shared memory from six stages to
  five, potentially shortening barrier waits; retain the same tile, cluster,
  TMA, tensor-core, epilogue, scale, and launch geometry.
- Build: SGLang commit
  `68e047c9a9a19f70ff10e62457ca642863f84d53`, based on vendored DeepGEMM
  v0.1.4/upstream `731e7c7a97d269e4b9f482ea18d0e709a948f293`, built as the isolated
  `deep_gemm_experimental` overlay. The installed stock package was untouched.
- Correctness: passed for the exact packed leaf and the complete
  `Fp8LinearMethod.apply` region.
- Paired result: the fair leaf regressed in all three series to 0.973239×,
  0.978281×, and 0.972641×. The full region also regressed in all three series
  to 0.961955×, 0.972547×, and 0.976800×.
- Profiler delta: dynamic shared memory fell from 209.7 KB to 176.4 KB but
  occupancy stayed at one CTA per SM/12.5%. Nsight Compute duration rose from
  251.3 µs to 259.6 µs; elapsed tensor utilization fell from 92.15% to 87.54%;
  long-scoreboard ratio rose from 17.26 to 18.32 and barrier ratio from 5.798
  to 6.111. Registers stayed at 42 and spills stayed zero. Static SASS retained
  1,967 instructions and the same 46 `UTCQMMA`, 22 `UTMALDG`, and 4 `UTMASTG`
  instructions; only small synchronization/NOP scheduling counts changed.
- Causal conclusion: five stages did not unlock a second resident CTA and
  instead under-buffered the already tensor-bound pipeline.
- Risk: exact-shape heuristic maintenance plus a separate DeepGEMM build, with
  no graph or TP8 end-to-end validation.
- Decision and rollback point: rejected. SGLang commit
  `8f450dbdf` reverts the experiment and is byte-identical to the starting
  source at `f93f8867b`; the isolated overlay and its JIT cubin remain only as
  reproducible evidence.

## Final disposition

No replacement. The existing specialization is below the noise gate, supported
configuration deltas do not survive the production wrapper/region, and the one
profiler-backed source change regresses both the exact leaf and its containing
FP8-linear region.
