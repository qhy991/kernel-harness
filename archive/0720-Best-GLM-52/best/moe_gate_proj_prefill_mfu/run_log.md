# Run Log — GLM-5.2 MoE Gate Projection Prefill B200 MFU Campaign

Task-local candidate optimization for `moe_gate/prefill`, measured only through
the frozen Kernel-Harness gate. GPU pinned to **B200 GPU 3**. This log records the
honest, reproducible starting point and every measurement command.

## Environment

| item | value |
|---|---|
| GPU (pinned) | NVIDIA B200, id 3 (`REMOTE_GPU_ID=3`, `CUDA_VISIBLE_DEVICES=3`; `cuda:0`==GPU3) |
| SM / arch | CC 10.0 (SM100), 148 SMs, 191.5 GB |
| Kernel-Harness commit | `7d79e5e` (`git_dirty: true` — 3 pre-existing dirty candidates only) |
| deep_gemm | 0.1.4 | 
| torch / CUDA | 2.11.0+cu130 / CUDA 13.0 |
| sgl_kernel | 0.4.4 |
| roofline (frozen) | FP8 peak 4.50 PFLOP/s, HBM 8.0 TB/s, ridge 562.5 FLOP/byte |
| timing protocol | CUPTI cold-L2 device-kernel median (warmup=3, repeat=10, iterations=30) |

## Harness-tree cleanliness (AC-1)

`git -C $KDA_HARNESS_ROOT status --porcelain` throughout the campaign shows ONLY
the three pre-existing dirty files, nothing else:

```
 M testbench/tasks/glm52/index_k_proj_decode/candidate.py
 M testbench/tasks/glm52/o_proj_decode/candidate.py
 M testbench/tasks/glm52/o_proj_prefill/candidate.py
```

The shared Harness is never modified. All campaign artifacts live in this worktree.

## Seed provenance (AC-1)

The campaign candidate began as a **byte-identical** copy of the clean task seed:

| file | sha256 |
|---|---|
| `Kernel-Harness/.../moe_gate_proj_prefill/candidate.py` (clean seed) | `3f96f1f1cfc3ae1211d1850bce935eb6751ad4282aff4da679af1bf30faa73dd` |
| `candidate/candidate.py` (campaign copy) | `3f96f1f1cfc3ae1211d1850bce935eb6751ad4282aff4da679af1bf30faa73dd` |

`cmp` reports identical bytes. The seed is a verbatim
`deep_gemm.fp8_m_grouped_gemm_nt_masked` reference call (speedup ~1.0, calc_diff 0).

## GPU-3 idle policy and observed contention

GPU 3 is shared: a separate campaign (`sol-execbench/.kersor`, e.g. PID 2956874/
3017153) runs **bursty** benchmark bursts on GPU 3 and releases it (observed 60-75s
idle windows between bursts). Every authoritative measurement here was bracketed by
pre/post `nvidia-smi --id=3 --query-compute-apps=pid` idle checks; contended samples
are discarded and re-run. One early CUDA-event probe was contaminated by a burst and
is not used. The workload inventory (seed=0, deterministic) is contention-independent.

## Authoritative baseline (reproduced exactly)

`./run.sh --candidate candidate/candidate.py` on idle GPU 3, exit code 1 (correct,
not faster — candidate == reference):

| M | cand us | ref us | MFU (compute_util) | target MFU | max latency | required speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 98.51 | 98.38 | 46.51% | ≥60% | 76.355 us | ~1.29x |
| 2048 | 159.56 | 159.54 | 57.42% | ≥67% | 136.755 us | ~1.17x |
| 4096 | 294.95 | 295.10 | 62.13% | ≥67% | 273.510 us | ~1.08x |

All three compute-bound (AI 1107/1518/1863 ≫ ridge 562.5). calc_diff = 0 on all.
Matches the recorded prior baseline (46.50 / 57.39 / 62.00%) within noise.

## Reference-only variance baseline (AC-7) — 3 authoritative gate runs, idle GPU 3

| M | run1 ref us | run2 ref us | run3 ref us | median | approx p10–p90 band | MFU band |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 98.44 | 98.53 | 98.47 | 98.47 | 98.30–98.72 (±0.2%) | 46.49–46.54% |
| 2048 | 159.67 | 159.74 | 159.76 | 159.74 | 159.38–159.90 (±0.2%) | 57.33–57.36% |
| 4096 | 295.56 | 295.75 | 295.86 | 295.75 | 295.25–296.29 (±0.2%) | 61.96–62.01% |

Noise band is ~±0.2–0.5% (all runs `timing_unstable=False`). The per-shape MFU
deficits to target (13.5 / 9.6 / 5.0 percentage points) are **orders of magnitude
larger than the noise band**, so they are real, not measurement artifacts.

## Workload inventory (AC-7) — deterministic, seed=0

| M | expected_m | masked_m (per expert) | total valid rows | capacity E·expected_m | capacity padding |
|---:|---:|---|---:|---:|---:|
| 1024 | 1152 | [1032,1018,1015,1027,1020,1039,1055,986] | 8192 | 9216 | 1024 (11.11%) |
| 2048 | 2176 | [2096,2070,2038,2092,1992,2082,2055,1959] | 16384 | 17408 | 1024 (5.88%) |
| 4096 | 4224 | [4140,4021,4164,4135,4046,4198,4111,3953] | 32768 | 33792 | 1024 (3.03%) |

**Tile-waste (valid vs full-slab) at BLOCK_M=128** — small, because DeepGEMM's masked
kernel walks `ceil(masked_m[e]/BLOCK_M)` valid tiles, not the full slab:

| M | valid tiles Σceil(masked_m/128) | full-slab tiles E·ceil(expected_m/128) | tile pad | waste if full-slab |
|---:|---:|---:|---:|---:|
| 1024 | 68 | 72 | 4 | 5.6% |
| 2048 | 133 | 136 | 3 | 2.2% |
| 4096 | 260 | 264 | 4 | 1.5% |

So masked-padding tail waste is NOT the dominant cost. Expert-tail imbalance is also
small (masked_m within ±3.5% of the mean at every shape).

## Per-call kernel breakdown (nsys timeline, 30 iters, idle GPU 3)

Each `fp8_m_grouped_gemm_nt_masked` call launches a preprocessing chain + the main
GEMM. Fractions applied to the authoritative run.sh totals:

| M | main GEMM `sm100_fp8_fp4_gemm` | scale-pack `transpose_and_pack_fp32_into_ue8m0` (×2) | index/schedule prep (`scatter_gather`+`div_floor`+`arange`) | preprocessing total | main-GEMM-only MFU |
|---:|---:|---:|---:|---:|---:|
| 1024 | 80.5% (~79 us) | 8.8% | 10.7% | ~19.5% | ~57.8% |
| 2048 | 87.7% (~140 us) | 5.9% | 6.4% | ~12.3% | ~65.4% |
| 4096 | 91.8% (~271 us) | 4.6% | 3.7% | ~8.2% | ~67.7% |

## Main-GEMM detailed profile (NCU, per shape) — preserved in `docs/profiles/`

Main GEMM `sm100_fp8_fp4_gemm_1d1d_impl`: BLOCK_M/N/K = 128/128/128, 2-CTA cluster,
grid = **148 (== num_sms, persistent, Waves Per SM = 1)**, 256 threads/CTA, 36 regs/thread.
Raw `.ncu-rep` + `.txt` exports and exact commands are under `docs/profiles/` (the judged
candidate is byte-identical to the reference, so this reference profile IS the candidate
profile — identical dispatch/metrics on every shape).

| M | Compute(SM) tput | Memory tput | DRAM tput | Occupancy | top pipeline | dominant stall (share) |
|---:|---:|---:|---:|---:|---|---|
| 1024 | **72.26%** | 64.19% | 30.56% | 12.47% (7.98 w/SM) | Tensor Core (72.3%) | MMA/smem scoreboard (56.1%) |
| 2048 | **81.04%** | 74.12% | 24.74% | 12.52% (8.02 w/SM) | Tensor Core (81.0%) | MMA/smem scoreboard (55.9%) |
| 4096 | **87.73%** | 73.94% | 20.53% | 12.49% (7.99 w/SM) | Tensor Core (87.7%) | MMA/smem scoreboard (55.9%) |

Per-shape reading: **Tensor-Core is the top pipeline on every shape**, rising 72→81→88%
with M; the single dominant warp stall (~56% on all three) is the MMA/shared-memory
scoreboard dependency — the signature of a well-pipelined tcgen05 GEMM, not a memory,
occupancy, or scheduling defect. DRAM falls with M (compute-bound throughout); occupancy is
a flat 12.5% (shared-mem limited by the warp-specialized design), never the limiter. M=1024's
low 72.3% is the small-per-expert-M wave/tail effect.

### Main-GEMM-only latency resolved (nsys, 60 iters, natural boost clocks)

NCU's short profiled window boosted the clock transiently (its `min` was 74.3 us at M=1024);
the representative main-GEMM latency at the boost clocks the gate actually runs at is the
nsys **median = 81.5 us** (avg 81.1 us) for M=1024. Main-GEMM-only MFU at M=1024 is therefore
**56.4%** (81.5 us), not the NCU-window 60.7%. Consequence, decisively:

| main-GEMM latency source | us | MFU | vs 60% target / 76.355 us ceiling |
|---|---:|---:|---|
| nsys median (boost, representative) | 81.5 | 56.4% | below floor AND above ceiling |
| nsys-fraction × run.sh total | 79.3 | 57.8% | below floor AND above ceiling |
| NCU window (transient high clock, outlier) | 75.5 | 60.7% | razor's edge — not reproducible at gate clocks |

So even deleting **100%** of the preprocessing overhead leaves M=1024's main GEMM alone
above the 76.355 us ceiling and below 60% MFU. Reaching M=1024 requires *beating*
DeepGEMM's own SOTA tcgen05 main GEMM by ~1.07x, not merely fusing away overhead.

## Commands (reproducible)

```bash
# authoritative gate (idle GPU 3)
cd "$KDA_HARNESS_ROOT/testbench/tasks/glm52/moe_gate_proj_prefill"
./run.sh --candidate "$CLAUDE_PROJECT_DIR/candidate/candidate.py"

# campaign tooling (bench/, never edits the Harness)
python bench/inventory.py                 # workload inventory + reference band
python bench/sweep.py [M ...]             # correctness-gated DeepGEMM knob sweep
python bench/verify.py --candidate PATH   # correctness + statelessness gate (AC-2/AC-3)
python bench/packed_probe.py             # EVIDENCE-only packed-ue8m0 probe (forbidden to ship)
nsys profile -t cuda --capture-range=cudaProfilerApi ... bench/ncu_target.py <M> baseline 30
ncu --profile-from-start off --kernel-name regex:sm100_fp8 ... bench/ncu_target.py <M> baseline 3
```
