# GLM-5.2 MoE W2 PTX + graph-only unlock: terminal report

## Disposition

**No replacement.** The graph-only dispatch integration required by the plan
was built and proven, and it removed the API-v1 Python provider tax that made
the prior campaign's selected eager leaf 0.80x. With that tax gone, the BM16
candidate clears the **graph leaf** gate for all four expected-M hints at
1.084-1.096x. It does not clear the **graph containing-region** gate for any
hint: the worst per-series estimator is 1.0256-1.0281 against a required 1.03.

The two bounded PTX hypotheses were then rejected on measured roofline
evidence rather than implemented, because the candidate kernel is already at
**99.26% of its achievable DRAM floor** and the entire remaining headroom in
the kernel is **0.50 us**, while the region gate needs about 2.5 us more.

Production default stays off. Stock SGLang/DeepGEMM remains active and is the
rollback. No checkpoint-backed TP8/DP8/EP8 acceptance command was run, and
there is no valid candidate command to hand to that lane.

## Bases

| Repository | Commit |
| --- | --- |
| Kernel-Harness | `694de1e6d1aeb0995ce3e3a7a50a087d7df4b4c6` |
| SGLang | `6e5d660d4f02bcb011bc54d817e7d965b5da3ece` |
| DeepGEMM (`0.1.4.post1` base `edcf77b2`) | `2afbc0dfca07f6bfc840f4546456a08a5c6d41d5` |

Environment re-audited CPU-only before any GPU work: four idle NVIDIA B200
(sm100, driver 610.43.02), CUDA 13.2.78, PyTorch 2.11.0+cu130, Triton 3.6.0,
CUTLASS `f3fde583` (v4.2.1), fmt `553ec11e`. Free root disk stayed between 12
and 13 GiB, above the 8 GiB stop threshold. All CUDA work ran through
`with_hotspot_gpu.sh`; every lease landed on physical GPU 0
(`GPU-30b619de-87f2-1862-0d07-a595da8fe417`).

## Required integration: graph-only dispatch

SGLang `6e5d660d4` mirrors the FlashMLA pattern for the W2 hotspot entry:

- `KernelSpec.graph_only=True` on the `moe_down_proj` hotspot spec only;
- `SGLANG_GLM52_W2_GRAPH_ONLY` defaults on, `0` restores eager selection for a
  diagnostic leaf;
- `try_dispatch_moe_masked` returns `False` before the ABI check and before the
  hit/miss lock when the current stream is not capturing, so eager decode keeps
  the unmodified stock path with no provider launch.

Five CPU contract tests pin this in `test_glm52_hotspot_registry.py`, and the
GPU contract `serving_native/moe_w2_graph_only_gpu_contract.py` proves it on
device (status `pass`, no failures):

| Property | Result |
| --- | --- |
| A. eager, graph-only on | declines; 0 provider attempts; output stays NaN-poisoned |
| B. eager, `GRAPH_ONLY=0` | selects; exactly 1 provider attempt; output written |
| C. capture, both settings | 1 graph node, kernel `infini_kernel_glm52_moe_w2_decode_bm16_auto`, no forbidden nodes |
| D. capture on vs off | identical node count, node types, and kernel identities |
| E. eager containing region | 0 provider attempts; estimators 0.9987-1.0068 |

Property D is what makes the graph timing lanes legitimate: the captured graph
is byte-identical under either setting, so a graph lane timed with
`GRAPH_ONLY=0` replays exactly what production graph-only replays. Property E
is the plan's "eager containing must stock-fallback" lane. Both of its arms run
the same stock W2 kernel, so it is reported as an identity lane against a
declared 0.97 floor and never as a candidate speedup.

## Generated-template identity, verified before any GPU timing

The signed CPU-only identity from the prior campaign re-verified with zero
drift (READY SHA-256
`17a5e23c0bd3cac16d88a1054047c4488fd9ca46b34fcca25560f83fdca7858b`,
`cpu_only_identity_gate=true`, `cuda_initialized=false`). Both arms come from
one task-local DSO, SHA-256
`395655ab609ef5037fe3bb93f4a2d813c047f1265fd5f30554948dd8d0e51780`.

| Property | Stock | Candidate |
| --- | --- | --- |
| symbol | `deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl<...>` | `infini_kernel_glm52_moe_w2_decode_bm16_auto` |
| BM/BN/BK, stages | 128/128/128, 8 | 16/128/128, 12 (auto-selected) |
| JIT key | `394687d565c010ed0cc18272659871a9` | `e8a6deeb7f7a319bbe485bfc6c351cae` |
| registers / stack / local / spills | 36 / 0 / 0 / 0 | 43 / 0 / 0 / 0 |
| launch | grid 148, block 256, cluster 2, PDL, 148 SMs | identical |
| TMA load / store | 10 / 16 | 10 / 2 |
| two-SM instructions | 25 | 25 |

This records stock as **BM128/stage-8**, confirming for the current source that
the invalidated Task-26 stage-11 premise (which assumed stock stage-12) is not
reused. The constants were checked for all four expected-M hints before the
first GPU lease.

## Fair performance gate

Every lane: 3 independent same-process alternating series, 50 AB/BA pairs each,
alternating start order, independently captured reference and candidate graphs,
all four estimators recomputed from the raw ordered pairs. Threshold 1.03 on
every estimator in every series.

### Graph leaf W2: passes

| expected-M | series passing | worst estimator | stock p50 | candidate p50 |
| ---: | :---: | ---: | ---: | ---: |
| 4 | 3/3 | 1.08977 | 79.7-80.2 us | 72.5 us |
| 5 | 3/3 | 1.09585 | — | — |
| 8 | 3/3 | 1.08457 | — | — |
| 9 | 3/3 | 1.09015 | — | — |

### Graph containing region (stock W13 -> stock SwiGLU/packed quant -> W2): fails

| expected-M | series passing | worst estimator | stock p50 | candidate p50 |
| ---: | :---: | ---: | ---: | ---: |
| 4 | 2/3 | 1.02638 | 218.4-219.0 us | 212.0-212.3 us |
| 5 | 2/3 | 1.02719 | 218.4-219.3 us | 211.7-213.0 us |
| 8 | 2/3 | 1.02810 | 220.2-220.8 us | 213.2-213.7 us |
| 9 | **0/3** | 1.02563 | 218.8-219.6 us | 212.8-212.9 us |

Across all 12 leaf series the pooled speedup geomean is **1.1008**; across all
12 region series it is **1.0314**, with the worst single estimator at 1.02563.

Each runner result self-audits as valid. The leaf win is real, uniform across
hints, and survives graph replay; it is simply too small a share of the
containing region. The region saves about 6.4 us on a 219 us denominator, and
1.03 requires 6.4 us of saving at 219 us to land exactly on the threshold, so
ordinary run-to-run spread decides each series.

## Why the two bounded PTX hypotheses were rejected

One NCU capture (`ncu/w2_em4.ncu-rep`, SHA-256
`18a231fa294759407098debdb21ea33dcd5b73d262809191ae17e28a84dbe529`) profiled
one stock and one candidate launch at expected-M 4:

| Metric | Stock | Candidate |
| --- | ---: | ---: |
| duration | 74.848 us | 68.096 us |
| DRAM read | 414.28 MB | 406.91 MB |
| DRAM write | 40.07 MB | 4.60 MB |
| achieved DRAM bandwidth | 6.070 TB/s | 6.043 TB/s |
| issue slots busy | 6.23% | 5.49% |
| registers / spills / local | 36 / 0 / 0 | 43 / 0 / 0 |

The two arms achieve the **same** DRAM bandwidth. The candidate is faster
purely because it moves fewer bytes: duration is bytes divided by 6.05 TB/s in
both arms. That fixes the ceiling:

- reads are 406.91 MB against an irreducible 402.65 MB of E32 x N6144 x K2048
  FP8 weights, a ratio of **1.0106** — the weights are read once and cannot
  shrink, so there is nothing to save on reads;
- writes are 4.60 MB against a 1.573 MB floor (128 valid rows x 6144 x BF16);
- floor duration is 67.595 us versus a measured 68.096 us, so the candidate is
  at **99.26%** of its achievable floor with **0.50 us** of total headroom.

**Hypothesis 1 — reduce padded TMEM/output stores for exact expected-M.**
Rejected before implementation. The candidate's vectorized valid-row store
already removed 89% of stock's write traffic (40.07 -> 4.60 MB). The entire
remaining write surplus is 3.03 MB, worth 0.50 us, i.e. 0.23% on the region.

**Hypothesis 2 — narrow PTX scheduling / barrier overlap on the BM16 pipeline.**
Rejected before implementation. The kernel is not issue-bound or barrier-bound:
issue slots are 5.49% busy, there are no spills and no local memory, and the
kernel already sustains the same DRAM bandwidth as stock. A scheduling or
barrier change cannot reduce bytes moved, and bytes moved fully explain the
duration. Its shared memory is also 230188 B of a ~228 KiB cap, so deeper
prefetch is unavailable. There is no concrete SASS defect to target, which the
plan requires before inline PTX.

Re-scoring each region lane's worst estimator at the physically unreachable
floor (zero-cost writes) confirms the rejection:

| expected-M | measured worst | at physical floor | passes 1.03 |
| ---: | ---: | ---: | :---: |
| 4 | 1.02638 | 1.02881 | no |
| 5 | 1.02719 | 1.02961 | no |
| 8 | 1.02810 | 1.03051 | yes |
| 9 | 1.02563 | 1.02805 | no |

Three of four hints fail the containing-region gate even at a ceiling no
implementation can reach, so no bounded PTX identity could have promoted this
candidate. Zero of the allowed two PTX identities were built; the negative
result is the deliverable.

## Correctness and routing

Each runner lane ran the full pre-timing, post-timing, fresh-input, and W2
edge-mask suites in both eager and CUDA graph, covering zero and maximum expert
counts, the 15/16/17, 31/32/33, 127/128/129 and 1024 boundaries, deterministic
ramp, random, extreme-finite, exponent-boundary and skewed masks, poisoned
caller-owned output, untouched masked tail rows, packed-scale bytes,
shape/dtype/stride/storage offset, input immutability, return identity, stream
identity, and independent graph capture with post-capture mutation and output
poison. Every lane self-audits as valid.

Selection stays fail-closed. Unsupported operator, phase, M bucket, expected-M,
ABI, recipe, topology, or execution mode declines before candidate invocation;
after selection a decline is fatal and stock is never executed. The eager
stock-fallback under graph-only is a pre-selection decline, not a post-selection
fallback, and its lane is labelled an identity lane rather than a candidate
result.

## Harness contract defect found and fixed

The first region lane failed its own audit with "candidate by_phase phase set
does not close exactly". The runner validates W2 edge masks for both the leaf
and the containing region, but the auditor and its fixtures only modelled edge
phases for the leaf. No region lane had ever been run before, so the mismatch
was latent. `audit_result.py` and `test_contract_v2.py` now derive the expected
phase set from the same predicate the runner uses. Three unrelated
`test_contract_v2.py` failures remain and are pre-existing at the parent commit
(they reference a legacy Task-26 artifact directory that no longer exists in
this worktree).

## Evidence

- `campaign_summary.json` — per-lane estimators, roofline, best-case re-scoring,
  and SHA-256 of every raw result.
- Raw results and NCU report:
  `/home/qinhaiyan/glm52-hotspot-goal-runs/cache/moe_w2_ptx_graph_only/`.
- `attempt_ledger.json` — every attempt including the two rejected hypotheses.

## Enable and fallback policy

The optimized path stays opt-in and exact-descriptor fail-closed. For this
terminal disposition, omit the hotspot environment variables and use stock
SGLang. The candidate is not eligible for external acceptance, and production
default remains `false`.
