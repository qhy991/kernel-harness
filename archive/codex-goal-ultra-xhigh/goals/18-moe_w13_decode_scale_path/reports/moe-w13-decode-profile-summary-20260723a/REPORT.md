# Goal 18 W13 decode profiler report

## Diagnosis

The fused W13 device kernel is not doing scale conversion. Both packed-ABI
buckets launch the same one-wave DeepGEMM SM100 kernel, and every timed
reference/candidate handoff window contains exactly one K6144/N4096 W13
launch. There is no scale-pack/layout-conversion kernel, CUDA allocation/free
API, or SM-reconfiguration API before it.

The binding device limit is the existing DeepGEMM TMA/TMEM pipeline reading
the expert weights, not arithmetic or a scale adapter. NCU measures
83.54%/83.08% memory throughput and only 37.18%/37.36% SM throughput at
M16/M32. The two buckets are effectively identical because each launches 148
blocks, exactly one block per 148-SM B200, and scans the same weights.

Nsight Systems does expose host preparation before the launch. A W13 call
performs five `cuTensorMapEncodeTiled` calls, two stream-capture checks, event
work from the measurement harness, and the kernel launch. The five tensor-map
encodes are present in both the stock wrapper and direct-call arms; they are
descriptor construction, not tensor conversion. The paired runner campaign,
not the single profiled sample, remains the latency authority.

## Measurement contract

- Collection command:
  `with_all_gpus_lock.sh bash serving_native/tools/run_moe_w13_decode_profiles.sh 20260723a`
- Physical GPU: B200 0,
  `GPU-30b619de-87f2-1862-0d07-a595da8fe417`; GPUs 0-3 were reserved for the
  entire command.
- Driver `610.43.02`; torch `2.11.0+cu130`; CUDA runtime 13.0; NCU 2026.1.1;
  Nsys 2025.6.3.
- Kernel-Harness source HEAD
  `0e004df6297ce101209804bc5ca3d727ba6cb857`; SGLang source HEAD
  `f93f8867b4bc124c9809c9110ec7361ed11b6b4a`.
- The provenance status contains only the two output files created by the
  collection script itself. Both source worktrees were clean immediately
  before invocation, and SGLang stayed clean.
- `SGLANG_GLM52_OPT=0`, PDL true, and both known GLM52 side-channel files
  absent.
- W13 inputs are FP8 E4M3 values with non-contiguous packed `torch.int32`
  UE8M0 scales. The internal symbol name contains `fp8_fp4`, but both operand
  template types in the captured symbol are `float_e4m3_t`; this is the
  reached FP8-by-FP8 masked grouped GEMM.
- Nsys collected one interleaved timed reference/candidate sample after three
  alternating warmups for each arm and bucket. These samples decompose the
  path; they do not replace the 30-pair runner results.
- NCU collected smoke, full plus PM-sampling/warp-state, and source reports
  after three same-arm warmups, with boost clocks and one exact-symbol launch.

## Nsight Systems decomposition

The strict reducer passed 10/10 reports with zero errors. All reference and
candidate windows used the byte-identical W13 mangled symbol.

| Arm | Bucket | Stock host/W13 us | Candidate host/W13 us | Stock -> W13 us | Candidate -> W13 us | Device sequence |
|---|---|---:|---:|---:|---:|---|
| Direct | M16 | 326.025 / 133.889 | 214.794 / 134.432 | 165.358 | 61.460 | W13 |
| Direct | M32 | 367.641 / 134.016 | 244.605 / 133.696 | 202.487 | 89.536 | W13 |
| Direct default args | M16 | 320.692 / 134.464 | 214.770 / 133.824 | 161.469 | 61.505 | W13 |
| Direct default args | M32 | 302.247 / 133.664 | 214.391 / 133.440 | 139.203 | 63.563 | W13 |
| Output reuse floor | M16 | 298.717 / 133.409 | 212.201 / 133.216 | 140.016 | 59.089 | W13 |
| Output reuse floor | M32 | 338.186 / 133.408 | 218.185 / 134.656 | 174.913 | 63.686 | W13 |
| Serving-safe registry | M16 | 322.945 / 134.336 | 253.152 / 134.016 | 161.039 | 99.712 | W13 |
| Serving-safe registry | M32 | 434.262 / 134.240 | 296.252 / 133.600 | 261.807 | 143.645 | W13 |
| Local compute direct | M16 | 407.341 / 135.328 | 327.303 / 134.400 | 163.475 | 92.513 | W13 -> SwiGLU+quant -> W2 |
| Local compute direct | M32 | 385.957 / 135.296 | 299.166 / 134.880 | 143.271 | 60.878 | W13 -> SwiGLU+quant -> W2 |

The host columns include profiler and measurement-event overhead and must not
be read as speedups. Their useful result is structural:

- every isolated handoff window has one device kernel;
- every local compute window has exactly W13, fused SwiGLU+quant, and W2;
- W13 device duration and symbol do not change under direct, default-argument,
  reuse, or registry arms;
- each W13 launch has five tensor-map encodes in both arms (ten for
  W13+W2);
- every window has zero timed CUDA malloc/free and zero SM reconfiguration;
- no kernel precedes W13, so there is no timed UE8M0 pack, cast, transpose, or
  layout conversion.

The line-info cache contains a
`transpose_and_pack_fp32_into_ue8m0` cubin because packed inputs are built
outside the measured window. It never appears in a timed Nsys window.

## NCU six-dimension analysis

| Dimension | M16 | M32 | Interpretation |
|---|---:|---:|---|
| Kernel duration | 136.32 us | 137.12 us | Boost-clock profiler values; nearly bucket invariant. |
| Memory throughput | 83.54% | 83.08% | Dominant speed-of-light signal. |
| DRAM read rate | 6.171 TB/s | 6.137 TB/s | About 841 MB read per launch, consistent with the expert-weight scan. |
| SM throughput | 37.18% | 37.36% | Not compute-throughput bound. |
| Tensor-pipe active, elapsed | 33.96% | 34.11% | Tensor work overlaps the memory pipeline but is not the ceiling. |
| L2 hit rate | 36.44% | 35.82% | Most required weight traffic reaches DRAM. |
| Grid / waves | 148 blocks / 1 wave | 148 blocks / 1 wave | Exactly one block per SM; no multi-wave tail to tune. |
| Registers / dynamic shared | 36 / 213.80 KiB | 36 / 213.80 KiB | Shared memory, not registers, limits one block per SM. |
| Achieved occupancy | 12.51% | 12.47% | Eight active warps per SM; appropriate for the specialized pipeline. |
| Eligible warps/scheduler | 0.050 | 0.050 | 95.06% of cycles have no eligible warp. |
| Long-scoreboard / issued | 25.68 | 25.79 | TMA/data-arrival waiting dominates scheduler stalls. |
| Barrier / issued | 8.25 | 8.22 | Cross-role pipeline synchronization is the second major wait. |
| Local/shared spilling | 0 / 0 | 0 / 0 | No register-spill or shared-spill target. |

The theoretical NCU memory-throughput lower bound is about 113.88 us for M16
and 113.92 us for M32 (`duration * achieved-memory-throughput`). That is a
speed-of-light estimate, not an expected or additive gain: reaching 100%
memory throughput is not realistic, and it says nothing about the containing
MoE region.

### Scheduler and source hotspots

PM/source sampling maps the dominant wait to CUTLASS `barrier.h:424`, the
`mbarrier.try_wait.parity` loop. It accounts for 5,487 M16 and 5,417 M32
long-scoreboard samples. DeepGEMM `barrier.cuh:18`, the cluster wait in
`cluster_sync_with_relaxed_arrive`, accounts for 1,733/1,723 barrier samples.
The next material site is the store/cleanup synchronization at
`sm100_fp8_fp4_gemm_1d1d.cuh:524`. These are the expected producer/consumer
waits of the weight-moving TMA/TMEM pipeline, not evidence of a Python scale
path.

PM timelines show repeated long-scoreboard/barrier bursts across the one wave;
there is no second-wave or partial-wave tail. Estimates above are alternatives,
not additive: removing host wrapper work cannot also claim the full theoretical
memory-throughput gap.

### PTX/SASS and resource evidence

`cuobjdump` reports 36 registers, zero stack, zero local memory, and 1,024
bytes static shared memory; NCU reports 213.80 KiB dynamic shared memory.
The exact SM100 SASS contains 16 `UTCQMMA.2CTA` tensor operations, 10
`UTMALDG`, 16 `UTMASTG`, two `UTCCP`, five `UTCBAR`, six `UCGABAR`, and 17
`NANOSLEEP` instructions. This confirms the existing two-CTA
TMA/TMEM/tensor-core specialization rather than a scalar fallback.

## Decision contribution

The profiler supports a no-scale-path diagnosis:

1. production already hands W13 packed `int32` UE8M0 scales;
2. explicit versus default `disable_ue8m0_cast` reaches the same device
   symbol and no pack kernel;
3. direct wrapper bypass changes only host preparation, not the W13 kernel;
4. the W13 kernel itself is a one-wave, memory/TMA-bound DeepGEMM kernel with
   no spills;
5. output allocation has no timed CUDA allocation API after warmup, while
   process-global reuse remains unsafe for streams and graph instances.

The paired component campaign is still the decision authority: its strongest
direct-call local-compute ceiling is below the production 3% threshold, and
the serving-safe registry regresses the containing local compute region. The
four-rank diagnostic and external eight-rank/server gates determine the final
deployment policy.

## Integrity

- Raw profile manifest (144 files) SHA256:
  `61bd68a6f1c87088918225d581afe3ae2f23a506704ae37d96cad2df0f6b3745`.
- M16 full NCU report:
  `085cf0e1d7090c14ccb301ee797e6a19abd29f673e5fd5d9797f0436c83ffdc4`.
- M32 full NCU report:
  `807a76f7cc785945675c04a0a65a0e5c5bbc89e1e316e529ebece562974d9887`.
- Strict Nsys correlation JSON:
  `aa4f921aa67ed0321d157e960744c2e15b35789c0519f452cc946aa740596e04`.
- Exact W13 SASS:
  `cfc90c36c64491ed50bd4a4ca7e4d26df1702cf33e48d513ae343b7bbc6b643e`.
- `sha256sum -c raw_profile_sha256.txt` passed after collection.
