# Production W2 decode validation log

Date: 2026-07-22

All CUDA initialization, benchmark, graph, profile, and distributed commands
for this goal run through `with_all_gpus_lock.sh`. The persisted campaign
environment exposes physical GPUs 0--3 and identifies active physical GPU 0 as
NVIDIA B200 UUID `GPU-30b619de-87f2-1862-0d07-a595da8fe417`.

## CPU and structural checks

| Check | Result |
|---|---|
| Shared rules and goal plan | read; source plan directory unchanged |
| `python3 testbench/bin/brief.py moe_down_proj_decode` | passed; no prior task run or recipe |
| `python3 testbench/bin/selftest.py` | passed: 24 tasks, 0 problems |
| `CUDA_VISIBLE_DEVICES= .venv/bin/python serving_native/selftest.py` | passed: 43 fixed workloads |
| Direct CPU-only `test_glm52_moe_overlap_contract.py` | passed: 4 tests |
| Goal Python compilation | passed |
| Goal shell syntax checks | passed |
| SGLang production source diff | none; commit `49dc279b5` adds only the contract test |
| Locked `python3 testbench/bin/verify_harness.py` | passed, exit 0 |

The four SGLang tests cover recipe/overlap bypass, argument identity, temporary
SM scoping, return-value preservation, successful replacement return, and
fail-closed stock fallback.

The final verifier run used `with_all_gpus_lock.sh` and reported selftest
24 tasks/0 problems, knowledge lint 13 entries/0 errors, current knowledge
index and distillation, all 24 task directories in sync, a clean diff check,
and an audit sweep with zero invalid results. Pointer audit remained advisory:
`runs/index.jsonl` is absent and counted as one malformed pointer. The normal
verifier does not fail on that historical-corpus pointer condition.

## Measurement identity

| Field | Value |
|---|---|
| Campaign | `glm52-w2-alignment-2bae536257aa929b957fdb28` |
| Kernel-Harness measurement HEAD | `ac9d47fe3731bca46ee2de2f77b004e124352934` |
| SGLang measurement HEAD | `49dc279b59753485c9ad6fa366d289018ecc41d3` |
| DeepGEMM | `0.1.4.post1@edcf77b276965de8f03cdc47c23f01b08bf7c7ab` |
| DeepGEMM extension SHA256 | `cd8beab174071777c972c5948af7706ae2cfb5d2adcdbb7e6fbea253ce3a81bf` |
| SM100 device-source SHA256 | `9c1e70677ede6ba09ab98e629482da7874182f8227907382efe0a81658da5a37` |
| Active B200 | `GPU-30b619de-87f2-1862-0d07-a595da8fe417`, 148 SMs |
| Measurement contract | 5 alignments x 4 workloads x 3 sessions x 30 alternating pairs; warmup 5 |

The Kernel-Harness status captured during measurement contains only untracked
in-campaign evidence, reports, and caches. SGLang was clean. The manifest pins
the exact measurement heads and every relevant harness hash.

## Paired campaign validation

Authority:

- [`alignment_campaign_manifest.json`](alignment_campaign_manifest.json)
- [`paired_alignment_summary.json`](paired_alignment_summary.json)
- [`paired_alignment_summary.csv`](paired_alignment_summary.csv)
- [`paired_alignment_summary.md`](paired_alignment_summary.md)

The summarizer validated the complete 20-row matrix. Every row has 90 paired
measurements, valid same-GPU provenance, pre-timing correctness, fresh-input
post-timing correctness, matching return semantics, and restored production
PDL/alignment state. Alignment 16 passed the 3% paired-median gate on all four
workloads with medians 1.080470x, 1.087436x, 1.075564x, and 1.062069x.

## Profile capture and diagnosis

| Role | Alignment | Attempt directory | Artifact matrix |
|---|---:|---|---|
| stock | 128 | [`profile_glm52-w2-alignment-2bae536257aa929b957fdb28_20260722T172516Z_656663_30552`](../../profile/moe-w2-packed-baseline/analysis/profiles/profile_glm52-w2-alignment-2bae536257aa929b957fdb28_20260722T172516Z_656663_30552/) | 4 Nsys + 4 NCU full/PM-sampling + 4 NCU SourceCounters reports, logs, metadata, and extracts |
| selected | 16 | [`profile_glm52-w2-alignment-2bae536257aa929b957fdb28_20260722T180259Z_1117292_24148`](../../profile/moe-w2-alignment16/analysis/profiles/profile_glm52-w2-alignment-2bae536257aa929b957fdb28_20260722T180259Z_1117292_24148/) | 4 Nsys + 4 NCU full/PM-sampling + 4 NCU SourceCounters reports, logs, metadata, and extracts |

Each NCU report contains exactly one selected
`sm100_fp8_fp4_gemm_1d1d` action. Full reports contain timing, launch, compute,
memory, scheduler, PM-sampling, tensor, eligible-warp, local-memory, and store
metric families. SourceCounters reports map sampled stalls to source lines.
The derived `metrics_key_*.json`, `raw_*.csv`, `details_*.txt`, and
`stall_hotspots_*.txt` files are retained with each attempt.

The first strict post-collection validator pass found that `nsys stats` emits
informational `Generating SQLite` and `Processing` lines before its CSV header.
The validator initially treated the first informational line as the header.
Raw `.nsys-rep` files and extracts were preserved byte-for-byte. The validator
was repaired to locate the actual header and fail closed on correlated CUDA API,
GPU, and kernel-execution trace fields. Four CPU parser tests pass, and both
attempt directories now pass strict validation idempotently. The deterministic
cross-attempt summary is
[`stock128_vs_bm16_profile_comparison.md`](../../profile/moe-w2-packed-baseline/analysis/stock128_vs_bm16_profile_comparison.md)
with an adjacent machine-readable JSON artifact.

Cross-tool checks:

| Check | Stock | BM16 |
|---|---:|---:|
| NCU kernel duration range | 75.520--76.256 us | 68.576--70.112 us |
| Nsys one-launch duration range | 74.687--76.544 us | 65.664--66.816 us |
| registers / spills | 36 / 0 | 34 / 0 |
| DRAM reads | about 414.28 MB | about 406.90 MB |
| DRAM writes | 40.41--40.67 MB | 7.94--8.34 MB |
| tensor-pipe active, elapsed | 30.41--31.00% | 4.18--4.23% |
| eligible warps per scheduler | about 0.073 | 0.059--0.060 |
| long-scoreboard ratio | 15.77--16.04 | 21.47--21.62 |

Generated config, ptxas, PTX, SASS, cubin, and hashes are in
[`profile/moe-w2-alignment16/analysis/jit_inventory.json`](../../profile/moe-w2-alignment16/analysis/jit_inventory.json).
The candidate has 34 registers, zero spills, and reduces static PTX/SASS size
and epilogue/TMA-store instructions while leaving static MMA and input-TMA
counts unchanged.

## Leaf graph and edge validation

Authority: [`leaf_validation_summary.json`](leaf_validation_summary.json).

The strict leaf validator reports `status=PASS`, alignment 16, four graph
artifacts, four edge artifacts, and twelve edge cases. Every graph:

- observed capture during launch;
- replayed 30 times with fixed pointers;
- was deterministic and finite on active rows;
- exactly matched eager output (`max_abs=0`, `calc_diff=0`); and
- preserved the stock return contract.

Graph median latencies are 0.083968--0.084064 ms. Edge cases cover
`0,15,16,17,31,32,33,127,1024`, empty experts, and scattered/front-loaded
placements. Every active-row comparison has `max_abs=max_rel=0`. The validator
scope is explicitly
`single_gpu_leaf_graph_and_edge_correctness_not_tp8_acceptance`.

## Four-rank diagnostic

Strict post-collection validation reports PASS for attempt
[`tp4_20260722T185932Z_1629770_10420`](tp4_diagnostic/tp4_20260722T185932Z_1629770_10420/summary.json): 18 stock baseline
results (six workloads x three trials), six Nsys result JSON files, and six
Nsys reports. Timings use the maximum CUDA-event latency across four ranks:

| Stock EP4 workload | Median of three trial medians (ms) | Min--max trial median (ms) |
|---|---:|---:|
| dispatch M16 | 0.057424 | 0.056736--0.737920 |
| combine M16 | 0.051312 | 0.047360--0.052240 |
| dispatch -> W13 -> SwiGLU+quant -> W2 region M16 | 0.583632 | 0.566016--0.592160 |
| dispatch M32 | 0.057072 | 0.055504--0.061568 |
| combine M32 | 0.053072 | 0.052528--0.091328 |
| dispatch -> W13 -> SwiGLU+quant -> W2 region M32 | 0.616704 | 0.594720--0.660832 |

All 24 workload logs record DeepEP's default 20-SM communication setting,
failed IBGDA transport initialization, and ProcessGroupNCCL device-ID inference
from global rank. The strict summary now verifies and records those facts, all
29 collection-command log hashes, four distinct B200 UUIDs, the all-NV18
topology, and 72 active per-GPU NVLink status records (18 per GPU). These
timings therefore characterize this fallback diagnostic environment, not tuned
production communication. The wide dispatch-M16 and combine-M32 trial ranges
are retained rather than discarded.

Every raw result has `candidate=null` and reference policy
`SGLANG_GLM52_OPT=0`. The region lane is lower-level eager, no-overlap EP4
structural-contract and fresh-repeatability validation. It conserves tokens,
checks packed scale and W2 handoff contracts, and produces finite repeatable
valid outputs, but explicitly records `independent_math_oracle=false`; it is
not candidate correctness or production graph validation.

The raw manifest preserves the first validator failure. That failure was the
Nsys CSV `Name`-column/preamble parser defect, not a benchmark or report
failure: all 18 baseline, six Nsys workload, topology, and GPU-state commands
had exit code zero. After the parser fix, the CPU-only strict validator passed
and was re-run idempotently, rechecking the same summary bytes. Four focused
TP4 validator unit tests also pass. The summary retains the prior failed
manifest tail rather than rewriting history.

The four-rank lane is always TP4/DP4/EP4 diagnostic evidence. It cannot be
renamed, extrapolated, or used as TP8/DP8/EP8 production acceptance.

## Production gates

| Gate | Result |
|---|---|
| Live EP8 `packed_recv_count` / `masked_m` | BLOCKED: eight ranks unavailable |
| Exact TP8 recipe, stream, signal, overlap, SM allocation, and selected config | BLOCKED: eight ranks unavailable |
| Three all-stock TP8 full-region baselines using rank-max latency | BLOCKED: eight ranks unavailable |
| Candidate-vs-stock TP8 full-region correctness and paired latency | BLOCKED: eight ranks unavailable |
| Production eight-rank graph/overlap replay | BLOCKED: eight ranks unavailable |
| Eight-rank SGLang end-to-end decode | BLOCKED: eight ranks unavailable |
| Goal-07 production enablement | none; stock active |

The serving-native/custom evidence JSON is not the frozen-task `result.json`
schema, so `testbench/bin/audit_result.py` does not apply. Campaign
summarization, profile-matrix checks, and the strict leaf validator are the
appropriate authorities for this evidence.
