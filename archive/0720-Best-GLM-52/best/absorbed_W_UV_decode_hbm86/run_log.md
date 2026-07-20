# Run Log — glm52 absorbed_W_UV_decode >86% HBM (RLCR round 0)

All measurements on NVIDIA B200, Kernel-Harness commit `7d79e5e`, env
torch 2.11.0+cu130 / CUDA 13.0, CUPTI available. Idle GPU pinned via
`CUDA_VISIBLE_DEVICES` (default cuda:0 inside the harness). The box is shared;
GPU idle state was checked before/after each measurement.

Timing protocol (frozen, from `testbench/harness/timing.py` +
`evaluate_task.py`): CUPTI cold-L2 device-kernel **span** = `max(kernel.end) -
min(kernel.start)` over all kernels in the iter window; L2 flushed (265 MB
zeroed) and inputs cloned per iteration, both outside the measured window;
`cand_us` = median over `repeat=10` of [median over `iters=30` per-iter spans].

## AC-1 — seed provenance + baseline (task1)

- Seed copied: `/home/qinhaiyan/Kernel-Harness/testbench/tasks/glm52/absorbed_W_UV_decode/candidate.py` → `candidate/candidate.py`.
- SHA-256 = `d4fccc4ae38c1a942c8fbd1fb0c291b88e5ce2b2387879c433f0ebbcb1619a6c` — **matches** the pinned seed hash.
- Baseline gate on idle GPU 3 (run `20260719T155156Z-a1ad71`):

| M | cand_us | ref_us | HBM util | calc_diff | verdict |
|---:|---:|---:|---:|---:|---|
| 16 | 5.28 | 5.28 | 22.34% | 0.00e+00 | neutral (cand==ref) PASS |
| 32 | 5.47 | 5.50 | 23.95% | 0.00e+00 | neutral (cand==ref) PASS |

Reproduces the pinned baseline (~5.408/5.728 µs, ~21.81/22.88%) within noise on a
clean idle GPU. `candidate == reference call`, so speedup ~1.0 and the Harness
gate is (correctly) NOT MET — this only establishes the baseline.

## Clock-state characterization (blocking side-issue for AC-3)

- Truly-idle B200 sits at **120 MHz** SM clock (max 1965 MHz; memory clock pinned
  at 3996 MHz regardless).
- During the gate protocol the repeated 265 MB L2-flush memset keeps the GPU busy,
  so the SM clock is pinned at **1965 MHz (max boost)** throughout the measured
  loop — confirmed by a 0.1–0.2 s clock trace (`docs/floor/*clock*csv`): ~all
  samples 1965 MHz, util 63–99%, ~500 W. So candidates are gated at full boost
  even on an "idle" GPU; the boosted floor is the relevant one.
- No permission to lock clocks (`nvidia-smi -lgc` denied), so the floor is
  measured under the same boost the gate produces.

## AC-3 — single-kernel span floor (task2), BL-20260719-single-kernel-span-floor

### E1 empty/trivial single-kernel span floor (gate protocol, flush ON)

| GPU | noop<<<1,1>>> med | noop<<<1,1>>> min | global-min (any trivial kernel) |
|---|---:|---:|---:|
| 3 (idle, siblings partly busy) | 1.976 µs | 1.680 µs | **1.648 µs** |
| 1 (concurrently busy → steadier boost) | 1.552 µs | 1.344 µs | **1.280 µs** |

Trivial kernels tried: `noop<<<1,1>>>`, `noop<<<1,1024>>>`, `noop<<<148,1024>>>`,
`write_one<<<1,1>>>`. Flush-OFF control ~0.3–0.5 µs lower — the ~1.3–2 µs floor is
intrinsic to single-kernel CUPTI span measurement, not the L2-flush memset.

**The empty-kernel span floor is ~1.3–2.0 µs.** M16 ceiling (1.372 µs) sits inside
this band; M32 ceiling (1.524 µs) sits at its top. Any legal candidate launches
≥1 device kernel and does strictly more than a noop, and span only grows with
more kernels — so the empty-kernel floor is a hard lower bound on candidate span.

### E2 shortest-legal single-kernel span = move exactly bytes_hbm, cold-L2 (gate protocol, GPU 3 @ boost)

Tuned vectorized grid-stride read (independent accumulators; config swept, best kept):

| case | best span | GB/s | % of 8 TB/s |
|---|---:|---:|---:|
| read bytes_hbm M16 (9.44 MB) | 4.640 µs | 2034 | **25.4%** |
| read bytes_hbm M32 (10.49 MB) | 5.056 µs | 2074 | **25.9%** |
| read B only (8.39 MB) | 4.424 µs | 1896 | 23.7% |
| read 256 MB (asymptotic) | 53.1 µs | 5055 | 63.2% |
| memcpy B (copy engine, 2× traffic) | 5.344 µs | 3140 | 39.2% |

A legal FP8 BMM must move ≥ bytes_hbm through HBM (read A+B, write BF16 out) AND
do the dequant/MAC, so its span ≥ the pure-read span above. Even pure streaming of
this working set is ~25% HBM — ~3.4× above the 86% target. Note the simple read
kernel tops out at 63% on a huge transfer (a perfect B200 kernel reaches ~85–87%),
so the read kernel underestimates optimal BW by ~30%; even correcting for that,
the 9.44 MB cold working set lands ~35% at best — far under 86%.

## AC-4 — NCU profile of stock bmm_fp8 (task4)

Backend kernel: **`nvjet_sm100_qqtst_128x16_128x12_2x1_2cta_v_bz_TNT`** (cuBLAS
"nvjet" SM100 FP8 GEMM), a **single** kernel per call. NCU (isolated, L2-flushed
replay → cold-L2):

| M | kernel | NCU dur | DRAM bytes | DRAM %peak | grid | block | waves/SM |
|---:|---|---:|---:|---:|---:|---:|---:|
| 16 | nvjet_sm100_qqtst_128x16 | 7.38 µs | 8.93 MB | **15.87%** | 128 | 256 | 0.86 |
| 32 | nvjet_sm100_qqtst_128x32 | 7.55 µs | 9.45 MB | **16.45%** | 128 | 256 | 0.86 |

Key finding: the reference is **occupancy/latency-bound, not BW-bound** — only
128 blocks (< 148 SMs), **0.86 waves/SM**, ~16% DRAM throughput. There is real
headroom versus the reference, but it is bounded far below the needed 3.85×
(see floor_decomposition.md). (NCU absolute durations run higher than the CUPTI
gate due to instrumentation/serialised replay; utilisation %/ratios are the
signal, not the absolute µs.)

## Artifacts
- Baseline gate log: harness `runs/glm52/absorbed_W_UV_decode/20260719T155156Z-a1ad71/`, `docs/_baseline_gate_gpu3.log`
- Floor scripts: `floor/measure_floor.py`, `floor/measure_span_floor.py`, `floor/measure_stream_floor.py`
- Floor logs: `docs/floor/*.log`, clock traces `docs/floor/*clock*.csv`
- NCU logs: `docs/ncu/ncu_ref_M16_nvtx.log`, `docs/ncu/ncu_ref_M32_nvtx.log`
