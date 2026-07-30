# Final report: external-acceptance-candidate

## Outcome

`combine_c2_bucket_stages` is selected for **both M16 and M32**. It replaces the
stock BF16 split-combine while holding the main kernel at the round-2 survivor P1
(FlashMLA `b5af443`, `p1_consumer_scale`, byte-identical machine code). It is
**bitwise exact** against installed stock, it passes both mandatory CUDA Graph
lanes at both buckets against **both** required denominators, and the eager
containing region falls back to stock with zero provider launches as the
graph-only policy requires.

Checkpoint-backed TP8/DP8/EP8 acceptance is not runnable on this four-GPU host,
so the disposition is `external-acceptance-candidate`, not `production-win`. The
**production default stays off**: stock runs unless the hotspot profile is
explicitly enabled. Registration ceiling is **L2 external E2E**.

| Bucket | Graph containing vs stock | Graph leaf vs stock | Graph containing vs P1 | Graph leaf vs P1 |
|---|---:|---:|---:|---:|
| M16 | **1.2705–1.2753** | **1.2702–1.2748** | **1.1819–1.2155** | **1.1824–1.2086** |
| M32 | **1.1376–1.1426** | **1.1352–1.1425** | **1.0698–1.0715** | **1.0641–1.0715** |

Each cell is the minimum and maximum over all four required estimators in three
independent alternating series, 100 pairs per series, recomputed from the raw
ordered pairs in [`evidence/timing_gate_audit.json`](evidence/timing_gate_audit.json)
rather than read from the stored summaries.

Target: SGLang API-v1 `flashmla_kv` FP8 sparse decode at kernel-harness `130c1e4`,
SGLang `d7fe89a71`, FlashMLA `daef3d0` (this campaign's own source commits as
FlashMLA `a168813`) — prefixed V32 main plus a **prefixed
combine variant**, M16 Q `[16,1,64,576]` BF16 with KV `[2049,64,1,656]` FP8
E4M3FN, M32 Q `[32,1,64,576]` with KV `[4097,64,1,656]`, sparse top-k 2048, page
size 64, V dim 512, scale 0.0625.

## 1. What the combine kernel was actually doing

The stock combine reduces `my_num_splits` FP32 partial outputs per (batch, head).
It holds exactly **one** split of `o_accum` in registers and reloads split `s+1`
inside the same unrolled body that consumes split `s`, so the first FMA of `s+1`
depends on a load issued after the FMAs of `s`. The accumulation is therefore a
chain of `my_num_splits` fully exposed global-memory latencies.

Production metadata makes that chain short but unhidden. `num_splits` is exactly
8 splits per request at M16 and 4 at M32 (asserted by the correctness harness,
not assumed), and the grid is one 256-thread CTA per (batch, head-octet):

| | splits/request | combine CTAs | SMs | resident CTAs/SM at 48 regs |
|---|---:|---:|---:|---:|
| M16 | 8 | 128 | 148 | **grid-limited to 1** |
| M32 | 4 | 256 | 148 | 5 (capacity) → single wave |

Both buckets move the same 16.8 MB of `o_accum`. At M16 only eight warps are
resident per SM, so there is essentially nothing to hide eight serialized
latencies behind.

## 2. The two identities, and why the first one only half worked

**C1 — `combine_c1_stage8`.** Hold eight splits in registers, issue the whole
group's loads before consuming any of them, and hoist the first group above the
LSE reduction so the LSE math also overlaps the gather. Predicted machine-code
change: `LDG.E.128` up 8x, about 16 registers per extra stage, zero spill,
unchanged shared memory, unchanged FMA order. Delivered exactly that —
`LDG.E.128` 8 → 64, registers 48 → 184, 0 spill stores, 0 spill loads, `LOCAL 0`,
`STACK 24` and `SHARED 6144` both unchanged.

It won 1.18–1.21 against P1 at M16 and **exactly nothing** at M32: 0.9990–1.0013
on the containing lane and 0.9994–1.0007 on the leaf, both straddling 1.0 inside
an 0.085% null. Nsys says why, at the level of the kernel rather than the chain:

| M32 graph lane | main | combine | overlap | chain |
|---|---:|---:|---:|---:|
| installed stock | 24.45 | 9.50 | 3.87 | 30.08 |
| P1 + stock combine | 23.23 | 9.44 | 3.84 | 28.90 |
| P1 + C1 | 23.36 | **9.44** | 3.87 | 29.06 |

The M32 combine did not move at all. The whole M32 gain against stock was P1's
main-kernel saving, which is precisely what the plan's second denominator exists
to catch.

The discriminator is the grid crossing the SM count. On this B200 the register
file is 65536 registers per SM at 8-registers-per-thread granularity, so for a
256-thread CTA the resident-CTA ladder is 48 regs → 5, 128 regs → 2, 136 and up
→ 1. At M16 the grid is 128 CTAs against 148 SMs, so occupancy is already
grid-limited to one CTA per SM **in the stock build too** and C1's registers cost
nothing. At M32 the grid is 256 CTAs, stock has the capacity to resident all of
them in one wave, and 184 registers cuts that to one CTA per SM — two waves of
148 and 108. C1 also only ever uses four of its eight stages at M32, capping the
available gain at 4x. A capped gain paid for with a lost wave nets zero.

**C2 — `combine_c2_bucket_stages`.** Pick the shallowest depth that still covers
the bucket's split count in one group, and make the occupancy requirement
explicit in the launch bounds rather than hoping for it:

```
STAGES = (params.b <= 16) ? 8 : 4
__launch_bounds__(NUM_THREADS, STAGES <= 4 ? 2 : 1)
```

`params.b` is host metadata the ABI guard has already validated, so this adds no
device read, no synchronization and no autotuning. Depths other than the two
production buckets stay correct because the group loop handles any split count.

ptxas delivered the predeclared resources: the shallow instantiation at **112
registers** — under the 128-register threshold for two resident CTAs per SM — and
the deep one at 186, both with zero spill stores, zero spill loads, `LOCAL 0`,
and `STACK 24` / `SHARED 6144` unchanged from stock.

## 3. What C2 changed on the device

Nsys graph-lane medians over five replays, one prefixed main followed by one
prefixed combine and no other device kernel in the marked ranges. Each report
captures its own stock arm, so the stock rows come from the C2 report and the
identity rows from the identity report:

| Graph lane | main | combine | overlap | chain | combine regs |
|---|---:|---:|---:|---:|---:|
| M16 installed stock | 17.15 | 11.81 | 3.90 | 25.47 | 48 |
| M16 P1 + stock combine | 16.10 | 12.22 | 4.13 | 24.13 | 48 |
| M16 P1 + **C2** | 16.22 | **7.97** | 4.00 | **20.22** | 186 |
| M32 installed stock | 24.67 | 9.57 | 3.84 | 30.50 | 48 |
| M32 P1 + stock combine | 23.23 | 9.44 | 3.84 | 28.90 | 48 |
| M32 P1 + **C2** | 23.33 | **7.97** | 3.81 | **27.49** | 112 |

The combine lands on **7.97 µs at both buckets**. That is the informative part:
the two buckets move identical traffic through different CTA and split geometries,
and once each geometry has both enough memory-level parallelism and a single
wave, they converge on the same time. 16.8 MB in 7.97 µs is 2.1 TB/s.

## 4. Fairness, and two hazards that had to be handled rather than argued away

**The instrument.** Round-2's K=20 batching is reused unchanged. Two same-binary
nulls were measured this round, and the claim for a bucket is only read against
that bucket's own null:

| Null | half-width | the claim it bounds |
|---|---:|---|
| installed stock in both arms, M16 | ±0.29% | M16 +27.0% vs stock |
| installed stock in both arms, M32 | ±1.45% | M32 +13.5% vs stock |
| `combine_identity` in both provider arms, M16 | ±2.63% | M16 +18.2% vs P1 |
| `combine_identity` in both provider arms, M32 | ±0.085% | M32 +6.4% vs P1 |

Each "claim" column is the **minimum** of all twelve estimators for that lane, so
it is the conservative end of the range, not the mean.
The smallest margin is M32-versus-P1 at 6.4% against an 0.085% null. The widest
null is M16 provider-pair at 2.63%, against an 18% claim. A pooled null figure
would have misrepresented both, so it is reported per bucket.

**Hazard 1: the two B200s in this host are not interchangeable.** The first two
leases were served different physical GPUs, and the installed-stock M16 graph
leaf median is about 28.86 µs on one and 26.85 µs on the other at identical
nominal clocks, while the candidate arm is unchanged. A stock-relative ratio is
therefore not portable between them: C2's M16 gate initially read 1.18 on one GPU
and 1.27 on the other. **Every ratio quoted above was regenerated inside a single
lease on a single physical GPU** (`GPU-30b619de-87f2-1862-0d07-a595da8fe417`), and
[`audit_timings_combine_r1.py --require-single-gpu`](harness/audit_timings_combine_r1.py)
fails rather than reports if the decision dataset ever spans more than one.

**Hazard 2: the stock arm carries a buffer-placement term the provider arm does
not.** The provider reduces into workspaces preallocated once by `initialize()`,
while installed stock takes `o_accum` from the caching allocator, so the stock
arm's placement depends on process history. This is exactly why plan hypothesis
1's denominator is measured as a **directly paired provider-versus-provider**
comparison rather than derived from two stock-relative runs: with
`combine_identity` in the A arm both sides use identically shaped workspaces
allocated by the same code in the same process, and the placement term cancels
instead of being argued about. `combine_identity` is a legitimate stand-in for
"P1 plus stock combine" because its SASS **is** the stock combine's instruction
stream — see section 5.

## 5. Every generated-binary claim, checked

[`audit_combine_binaries.py`](harness/audit_combine_binaries.py) establishes
three things before any timing is read.

**The main kernel is unchanged P1.** The FlashMLA worktree is shared with the
concurrent round-3 main-kernel goal, whose `sm100/decode/head64/kernel.cuh` edit
(now committed as `daef3d0`) changes this campaign's source hash and therefore
its build id. It is fully guarded by `GLM52_COORD_PREFETCH_ACROSS_BUF_WAIT`,
which this campaign never defines, and the audit proves the consequence rather
than asserting it: every arm's main SASS instruction stream is **identical** to
the round-2 reference P1 extension, at the recorded 168 registers, 16 barriers
and zero spill.

**The identity control really is the stock combine.** Its SASS differs from
`flash_fwd_mla_combine_kernel<bfloat16_t,512,8,160,256>` in **exactly one
instruction out of 344**: the `__LINE__` immediate handed to `__assertfail` on
the `my_num_splits > MAX_SPLITS` branch, which cannot be taken under the frozen
ABI. The audit does not normalize that away — it requires the two immediates to
equal the device assert's line number in each source file (`0x2b` = 43 in
`combine.cu`, `0x94` = 148 in `combine.cuh`) and fails on any other difference.

**C1's measurements bind to the committed source.** C1 was built, validated and
timed before `combine.cuh` gained the `STAGES` template parameter that C2 needs.
Regenerating C1 from the final committed header changes exactly one instruction —
the same dead assert-line immediate, `0x5f` → `0x94` — out of 728
([`evidence/c1_sass_rebinding.json`](evidence/c1_sass_rebinding.json)). Its
final numbers were then re-measured from the final binary anyway.

Every device symbol this campaign compiles carries the goal prefix. The combine
variants use the documented combine-specific form the plan permits,
`infini_kernel_glm52_flashmla_sparse_decode_combine_*`, exposed through the same
API-v1 `flashmla_sparse_decode` callback.

## 6. Correctness

Bitwise exact against installed stock at both buckets for **both** candidates:
`max_abs = 0.0` and `exact = true` on every boundary — eager leaf, non-default
stream, independently captured leaf graphs before and after Q/index mutation,
deterministic repeated replay with poisoned outputs, and the containing
`_forward_flashmla_kv` in eager and graph. Also checked: output and LSE value,
shape, stride and storage ownership; input immutability for Q, KV and indices;
poison overwritten; graph replay demonstrably fresh after input mutation
(max change 2.3e-3, not stale); exactly one provider launch with zero fallback;
and wrong-page ABI rejected with zero provider launches.

The adversarial matrix passes all 17 cases per bucket per candidate, covering
random, zero, signed ramp, extreme finite, exponent-boundary and post-capture
mutated inputs, and duplicate, interleaved, sorted, unsorted, boundary-page,
`-1` and beyond-valid-length indices. Production tolerance was not loosened; the
kernel does not need it.

Correctness runs force `SGLANG_GLM52_FLASHMLA_GRAPH_ONLY=0` deliberately, so the
eager boundaries validate the candidate instead of silently degrading into
stock-versus-stock under the production graph-only default.

## 7. The eager lanes under graph-only

| Bucket | containing_eager | provider launches | leaf_eager (diagnostic) |
|---|---:|---:|---:|
| M16 | 0.859–0.875 | **0** | 1.236–1.269 |
| M32 | 0.892–0.896 | **0** | 1.116–1.140 |

Zero provider launches in the eager containing region is the graph-only
requirement, and it holds. The sub-1.0 ratio there is the API-v1 Python wrapper
cost on the **stock fallback** path — round 2 measured that tax at a constant
+17.4 µs and showed the lane is arithmetically unreachable for any Python-provider
candidate — not a candidate regression. The round-3 promotion policy explicitly
does not gate on it. `leaf_eager` is recorded with selection forced, as a
diagnostic only; it cannot promote a bucket.

## 8. What was not attempted, and the evidence that closed it

- **Loading `o_accum` before `cudaGridDependencySynchronize()`.** Illegal, not
  merely unwise: `griddepcontrol.wait` is the acquire that makes the prerequisite
  grid's stores visible, and CUTLASS states that issuing it "enforces no global
  memory access prior to this instruction". The 3.8–4.1 µs prologue overlap Nsys
  reports is a hard bound for producer-written data. The one upstream PDL
  prologue that does move real bytes (CUTLASS example 63) prefetches weights,
  which no prior kernel writes.
- **`ld.global.nc` / `__ldg` on `o_accum`.** The upstream comment in `combine.cu`
  is well founded. The non-coherent path is valid only for data no prerequisite
  grid writes during the consumer's lifetime, and under PDL that lifetime begins
  before the producer's stores land. FlashMLA's own `__ldg` uses in this kernel
  are on `num_splits_ptr` and `attn_sink`, neither of which the main kernel writes.
- **`ld.global.L1::no_allocate` / evict-first policy.** Not built. The 16.8 MB of
  `o_accum` was written by the immediately preceding kernel and fits inside this
  device's 126.5 MB L2, so these reads are L2 hits, not DRAM traffic, and the
  combine touches nothing else for L1 to protect. Documented sm_100 wins for that
  qualifier are on operand streams that displace reused data. With C2 already at
  the same 7.97 µs in both buckets there was no measured gap for it to close.
- **Reducing `num_sm_parts` to shorten the reduction.** The upstream-favoured
  lever, and out of scope here: `num_sm_parts`, `tile_scheduler_metadata` and
  `num_splits` are frozen caller inputs in plan §0.2. Recorded for whoever owns
  that ABI.
- **Changing `BLOCK_SIZE_M` or splitting `head_dim_v` across more warps.** Both
  change the combine's grid or block topology, which was deliberately preserved
  so a graph-node comparison shows only the intended kernel substitution.
- **A second NCU capture.** Not invoked. The one concrete survivor question — why
  C1 gained nothing at M32 — was answered by the Nsys per-kernel durations plus
  ptxas register counts and the documented sm_100 occupancy ladder, and that
  answer predicted C2's register target and its result in advance.

## 9. Selector and fallback policy

Both buckets promote, so the enable is the full M-bucket set. Nothing is
default-on; with no environment set, installed stock runs.

```bash
SGLANG_GLM52_OPT=1
SGLANG_GLM52_OPT_PROFILE=hotspot_candidates
SGLANG_GLM52_OPT_OPS=flashmla_sparse_decode
SGLANG_GLM52_OPT_M_BUCKETS=dsa_decode_attn:16|32
SGLANG_GLM52_HOTSPOT_MODULE=/home/qinhaiyan/glm52-hotspot-goal-runs/worktrees/flashmla-sparse-decode/kernel-harness/serving_native/candidates/flashmla_combine_decode_provider.py
GLM52_FLASHMLA_COMBINE_VARIANT=combine_c2_bucket_stages
# SGLANG_GLM52_FLASHMLA_GRAPH_ONLY=1 is the default and should stay 1.
```

Fail-closed behaviour is unchanged from the installed guard: any mismatch on M,
page count, dtype, stride, scale, phase, backend, speculative mode or device
selects stock **before** the candidate can launch, and after selection a provider
error propagates rather than falling back. Rolling back to the round-2 baseline
is `GLM52_FLASHMLA_COMBINE_VARIANT=combine_identity`; rolling back to installed
stock is unsetting `SGLANG_GLM52_OPT`.

`combine_c1_stage8` is retained as measured evidence and is **not** promotable: it
fails the P1-relative condition at M32 (0.9990–1.0013). It is dominated by C2 at
M16 as well, so there is no bucket for which it should be selected.

## 10. Repository validation

All 10 SGLang GLM-5.2 hotspot registry tests pass. All campaign Python compiles,
all 59 retained JSON artifacts parse, and `git diff --check` is clean in all three
worktrees. **The SGLang worktree is unmodified by this campaign** — the graph-only
integration it relies on was already committed as `d7fe89a71` by the prior round.

Two broader base checks remain nonzero for inherited contracts outside this
campaign. Both were re-run rather than inherited, and both reproduce exactly
([`evidence/base_check_reproduction.json`](evidence/base_check_reproduction.json)): `serving_native/selftest.py`
expects `PRODUCTION_FLASHMLA_KV_DECODE_CASES` in a mandated, unmodified SGLang
base file, and `testbench/bin/verify_harness.py` reports pre-existing stale
generated projections. Neither touches the provider, dispatch guard, FlashMLA
source, generated binary, correctness harness or measurements, and this goal
forbids regenerating them.

## 11. Provenance and limitations

Bases: kernel-harness `130c1e4`, SGLang `d7fe89a71`, FlashMLA `daef3d0` with this
campaign's variants committed as FlashMLA `a168813`, CUTLASS
`147f5673`, nvcc 13.2 V13.2.78, torch CUDA 13.0, NVIDIA B200 sm_100. Build ids:
`combine_identity` `264037f2c6872c7f`, `combine_c1_stage8` `430fa79702e17a68`,
`combine_c2_bucket_stages` `ea8c72aac9631a91`.

Measurements were taken while the FlashMLA worktree carried this campaign's
uncommitted variant sources, so the retained provider evidence records
`daef3d0…-dirty`. That is disclosed rather than relabelled, and closed rather
than merely noted: after committing as FlashMLA `a168813`, every recorded build
id recomputes exactly from the clean committed tree, for all three variants
([`evidence/build_id_reproduction_post_commit.json`](evidence/build_id_reproduction_post_commit.json)),
so the committed bytes are the measured bytes.

Coordination with the concurrent round-3 main-kernel goal held throughout: this
campaign added only new files (`api_combine.cpp`, `combine.cuh`, `combine.h`,
three `v32_combine_*.cu`, a separate provider), never edited that goal's
in-flight `kernel.cuh` or its provider, and built from a disjoint source set so
neither campaign can inherit the other's binary.

Two limitations bound this result. First, checkpoint-backed TP8/DP8/EP8 acceptance
is not locally runnable on this four-GPU host; commands are retained in
[`EXTERNAL_ACCEPTANCE.md`](EXTERNAL_ACCEPTANCE.md). Second, free space on the
shared root volume fell from about 11.8 GiB at preflight to under 200 MiB while
three other hotspot goals built and profiled concurrently, so the plan's 8 GiB
floor is **not** satisfied at the end of this campaign. This campaign's own
footprint is about 34 MiB, all measurements completed before the volume filled,
and no further variant was built afterwards; it is reported in
[`evidence/disk_pressure_disclosure.json`](evidence/disk_pressure_disclosure.json)
rather than worked around, and it constrains any future round on this host.

No remote state was modified. Production default remains off and stock remains
the fallback for every unsupported case.
