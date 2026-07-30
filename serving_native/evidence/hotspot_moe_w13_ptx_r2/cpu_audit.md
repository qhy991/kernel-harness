# GLM-5.2 W13 PTX/SASS round-2: CPU-only audit

Performed before any source change, as required by the launch prompt.

## Environment re-audit

| Item | Required | Observed |
|---|---|---|
| GPUs | 4x B200 sm_100 | 4x NVIDIA B200, driver 610.43.02 |
| CUDA toolkit | 13.2 | `nvcc` 13.2.78 (`cuda_13.2.r13.2/compiler.37668154_0`) |
| Nsight Compute | 2026.1.1 | 2026.1.1.0 build 37634170 |
| Nsight Systems | 2025.6.3 | 2025.6.3.541 |
| PyTorch | 2.11.0+cu130 | 2.11.0, `torch.version.cuda` 13.0 |
| Triton | 3.6.0 | 3.6.0 |
| CUTLASS | 4.2.1 `f3fde58372d3` | submodule `f3fde58372d33e9a5650ba7b80fc48b3b49d40c8` |
| fmt | `553ec11ec06f` | submodule `553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28` |
| tvm-ffi | 0.1.11 | 0.1.11 |
| Free disk | > 8 GiB | 13 GiB on `/` (98% used) |

GPU snapshot at audit: GPUs 0/1/2 idle at 0 MiB / 0%, GPU 3 held 34612 MiB at
27% utilization. Consistent with the plan's warning that GPU 3 is frequently
unsafe. All CUDA work goes through `with_hotspot_gpu.sh`.

Task-local caches exported by the launcher and used exclusively by this round:

```text
SGLANG_DG_CACHE_DIR = .../cache/moe_w13_ptx_r2/deepgemm
DG_JIT_CACHE_DIR    = .../cache/moe_w13_ptx_r2/deepgemm
TRITON_CACHE_DIR    = .../cache/moe_w13_ptx_r2/triton
TORCH_EXTENSIONS_DIR= .../cache/moe_w13_ptx_r2/torch_extensions
CUDA_CACHE_PATH     = .../cache/moe_w13_ptx_r2/cuda
XDG_CACHE_HOME      = .../cache/moe_w13_ptx_r2/xdg
GLM52_TASK_BUILD_DIR= .../cache/moe_w13_ptx_r2/build
```

Round-1's cache (`cache/moe_w13_decode`, 428 MB) is read-only evidence for this
round. `build_variants.py` previously hardcoded that path; it now derives the
task cache from `GLM52_TASK_BUILD_DIR`, so round-2 builds its own artifacts
instead of reusing or overwriting another task's binary cache.

## Repository identities

| Tree | Required base | HEAD at round-2 start | Status |
|---|---|---|---|
| Kernel-Harness | `d432ea82…` | `a128c0d073a14516fb71882630eec039189f2dc7` | clean |
| SGLang | `83d31310…` | `5af212d00439a8990a1d64e2b7e32aa207acf2cb` | clean |
| DeepGEMM | `731e7c7a…` | `87e0359edbb461181d3bba218442132007b9a738` | clean |

All three are the round-1 validated heads on branch
`goal/glm52-hotspot-moe-w13-decode`, each descending from the required base.
Round-2 source materialization was re-audited twice and reproduces round-1's
tree hashes byte-for-byte:

```text
stock_source_tree_sha256     = 917592ab68ea0608c9be33208c2c609bc7f20bd9b1603f32743dd0d1ae03d0ed
candidate_source_tree_sha256 = d682daa65b8ba0ac3846d766910b8c751e0568fe62087084271bb354e46c49e4
candidate_diff_sha256        = 465c8373c0a37970225a0e93267b6c399431b23e22cf35b4511db2308df98092
```

## ABI re-confirmation from source

The frozen ABI in the plan is confirmed against the generated template
constants rather than configuration names.

For `w13_config = (16,128,128,12,2)` on `GemmType::MGroupedMasked`, swap-AB,
`compiled_dims="nk"`, `gran_k_a = gran_k_b = 128`, 148 SMs, `tc_util` 100,
PDL on:

| Derived constant | Value | Source |
|---|---:|---|
| `kIsMulticastOnA` | `cluster_n > 1` = true | `sm100_fp8_fp4_gemm_1d1d.hpp` template args |
| `LOAD_BLOCK_M` | `16 / 2` = 8 | impl line 59 |
| `LOAD_BLOCK_N` | `128 / 1` = 128 | impl line 60 |
| `UMMA_M` | `128 * 2` = 256 | impl line 56 |
| `UMMA_N` | `BLOCK_M` = 16 | impl line 57 |
| `STORE_BLOCK_M` / `STORE_BLOCK_N` | 16 / 128 | impl line 84-85 |
| threads | 256 (8 warps: TMA, MMA, UTCCP, idle, 4 epilogue) | `get_launch_config` |
| smem per stage | 1024 (A) + 16384 (B) + 512 (SFA) + 512 (SFB) = **18432** | `get_pipeline_config` |
| smem extra | 8192 (CD) + 808 (barriers) + 4 = **9004** | `get_pipeline_config` |
| dynamic smem | 9004 + 12*18432 = **230188** | matches plan and round-1 audit |
| max stages | `(232448 - 9004) / 18432` = **12** | `get_pipeline_config` |

The one-SM variant `(16,128,128,11,1)` gives 19456 B/stage and 223020 B, also
matching the plan. **Stage 12 is already the arithmetic maximum for the two-SM
BM16 layout**; the pipeline cannot be deepened without shrinking a stage.

A second confirmed fact bounds the epilogue: for `MGroupedMasked`,
`Scheduler::get_aligned_effective_m_in_block` returns `BLOCK_M`
unconditionally, so `effective_m = 16`, `num_stores = 16/16 = 1`, and the
dynamic UMMA-N update is a no-op. Each scheduled block performs exactly one
store stage.

## Round-1 generated-binary re-audit (retained artifacts)

Round-1's provider dumped PTX/SASS into its JIT cache; those artifacts are
retained and were re-read here. Instruction counts for the retained
`infini_kernel_glm52_moe_w13_decode_em4_bm16_2sm` cubin:

| Form | Two-SM BM16 | One-SM BM16 |
|---|---:|---:|
| `UTCQMMA.2CTA` | 16 | 0 |
| plain `UTCQMMA` | 0 | 16 |
| `UTMALDG.2D` | 10 | 10 |
| `LDTM.16` | 4 | 4 |
| `STSM.16.MT88.4` | 2 | 2 |
| `UTMASTG.2D` | 2 | 2 |
| `UTCCP.T.S.2CTA.4` | 2 | 2 (1CTA form) |
| `UCGABAR_WAIT` / `UCGABAR_ARV` | 3 / 3 | 0 / 0 |
| `SYNCS.PHASECHK.TRANS64.TRYWAIT` | 32 | — |
| `NANOSLEEP` | 17 | 13 |

The three `UCGABAR` pairs correspond exactly to the three source-level cluster
syncs: two in the prologue (required around 2-CTA TMEM allocation) and the one
marked `// TODO: Remove redundant synchronization` at kernel exit.

## Roofline / speed-of-light derivation (decides where round-2 headroom exists)

Per W13 call at expected-M 4 with `masked_m` around 4 per expert, so
`num_m_blocks = ceil(masked_m/16) = 1` and 32*1*32 = 1024 scheduled blocks:

| Traffic component | Bytes |
|---|---:|
| B weights `[32,4096,6144]` FP8, each expert read exactly once | 805.31 MB |
| B scales `[32,4096,12]` int32 | 6.29 MB |
| A activations actually loaded, distinct | 3.15 MB |
| C output `1024 blocks * 16*128*2 B` | 4.19 MB |
| **Total** | **~818.9 MB** |

Round-1's clean Nsys graph-node collection measured W13 device p50 of
145.744 us (stock BM128) and 139.168 us (BM16 two-SM). Stock's distinct A
footprint is `32*128*6144` = 25.17 MB instead of 3.15 MB, so stock moves about
840.9 MB.

| Arm | Traffic | Device p50 | Achieved DRAM rate | % of 8 TB/s |
|---|---:|---:|---:|---:|
| stock BM128/2SM | 840.9 MB | 145.744 us | 5.77 TB/s | 72.1% |
| BM16 two-SM | 818.9 MB | 139.168 us | 5.88 TB/s | 73.5% |

Two conclusions follow, and they reframe round-2's search space:

1. **Round-1's 1.047x came almost entirely from moving less data.** The traffic
   ratio alone is 840.9/818.9 = 1.0269 of the measured 1.0472; the residual is
   about 1.9% better bandwidth utilization. Stock and candidate sit at
   essentially the same achieved DRAM rate.
2. **Tensor cores and the epilogue are not the limit.** BM16 cut UMMA work 8x
   versus stock (`UMMA_N` 128 -> 16); at 1024 blocks * 48 k-blocks * 4 UMMA of
   256x16x32 the candidate issues roughly 51.5 GFLOP, about 12 us of B200 FP8
   tensor-core time inside a 139 us kernel. Output writes are 4.19 MB, 0.5% of
   traffic, executed by exactly 4 `LDTM` + 2 `STSM` + 2 `UTMASTG` per block.

## Consequences for the plan's three bounded hypotheses

- **H1 BM16 epilogue / TMEM store reduction.** The store surface is already at
  its instruction and byte floor: one store stage of a 16x128 BF16 tile, 4
  `LDTM.16` (the minimum to drain a 16-column accumulator through
  `16dp256b`), 2 `STSM`, 2 TMA store atoms, 4.19 MB total writes. Even
  eliminating **all** output traffic could not yield 3% of a 139 us kernel that
  moves 815 MB of reads. Recorded as a quantitatively closed route rather than
  an experiment worth one of three identities.
- **H2 barrier / mbarrier overlap.** Still open, but its ceiling is whatever
  fraction of the 26% gap to theoretical peak is exposed wait rather than
  memory-system limit. Two of the three cluster barriers are structurally
  required around 2-CTA TMEM allocation; only the `TODO`-marked terminal sync
  is a candidate, and it executes once per CTA per launch.
- **H3 BM32 rescue.** BM32 is strictly dominated by BM16 on this ABI: it moves
  more A (6.29 MB distinct versus 3.15 MB), doubles UMMA work, and doubles the
  store surface, with no compensating mechanism. Its round-1/Task-24 failure was
  a real 1.028 BA estimator, not noise. Leaving BM32 on stock, as the plan
  permits.

The one question that decides whether **any** round-2 identity can reach 1.03
is therefore: *is the BM16 two-SM survivor already at the achievable DRAM
speed-of-light for this strided FP8 weight stream, or is a material fraction of
its time exposed pipeline/barrier wait?* That is a concrete, survivor-specific
device-code question and is exactly the condition under which the plan
authorizes NCU. It is answered next.
