# Final report: no-replacement

## Outcome

No FlashMLA replacement is enabled for M16 or M32. One material candidate was
built. It is bitwise exact and it passes both mandatory CUDA Graph lanes at both
buckets by a clear margin, but the eager containing SGLang DSA region measures
**0.664–0.758**, so a lane that plan §6 enumerates fails and the plan's terminal
rule selects `no-replacement`. Installed SGLang/sgl-kernel FlashMLA remains the
production path and the default-off hotspot profile remains off. The
registration ceiling for this op stays **L1 (explicit diagnostic)**.

The failing lane is **not attributable to the candidate kernel**. A control whose
kernel is source-identical to upstream stock fails the same lane at 0.620–0.696
with a constant **+17.4 µs** of host-side Python in the API-v1 provider path.
Section 5 below shows the lane is arithmetically unreachable for *any*
Python-provider candidate, even one with a zero-cost guard.

Target: SGLang API-v1 `flashmla_kv` FP8 sparse decode at bases kernel-harness
`c93b342`, SGLang `c52f23b56`, FlashMLA `65293ac` — prefixed V32 main plus the
unchanged stock BF16 combine, M16 Q `[16,1,64,576]` BF16 with KV
`[2049,64,1,656]` FP8 E4M3FN, M32 Q `[32,1,64,576]` with KV `[4097,64,1,656]`,
sparse top-k 2048, page size 64, V dim 512, scale 0.0625.

## 1. What P0 established

The prior terminal result is bound to the current source. Rebuilding the prior
campaign's rejected composite recomputes its recorded build id
`24c522c90bc8583e2aa98a1e926d0bf853d1ed0eb01b59dd735f642fb68fa331` exactly, at
the recorded 3336 static main instructions; the identity control reproduces
4128 instructions, 168 registers/thread, 16 barriers and zero spills. See
[`evidence/cpu_audit.json`](evidence/cpu_audit.json),
[`evidence/preflight.json`](evidence/preflight.json) and
[`evidence/binary_manifest.json`](evidence/binary_manifest.json).

One NCU capture of the lineinfo-enabled identity main — permitted by §6 to test
a stated critical-path mechanism — confirms the plan's §0.3 mechanism and closes
two hypotheses ([`evidence/p0_mechanism_m16.json`](evidence/p0_mechanism_m16.json)):

| Stall reason | Cycles per issued instruction | Share |
|---|---:|---:|
| long scoreboard | 5.336 | 33.2% |
| barrier | 3.723 | 23.2% |
| no instruction | 1.517 | 9.4% |
| wait | 1.310 | 8.2% |
| sleeping | 1.018 | 6.3% |
| short scoreboard | 0.917 | 5.7% |
| mio throttle | 0.801 | 5.0% |

Long-scoreboard plus barrier is 56.3% of stall cycles, confirming the stated
mechanism. Three further facts shaped everything after this:

- **Shared memory has 816 free bytes** of the 232,448-byte opt-in limit
  (231,632 configured). A third KV buffer needs 106,496. The two-deep pipeline
  therefore cannot be deepened at all.
- **Shared-memory stalls total 10.7%**, so bank conflicts are not the limiter.
- **Average SM active is 68.4% of elapsed** (29,447 of 43,035 cycles; per-SM
  spread 6,892–34,531). That is tail imbalance across the 148 fixed scheduler
  partitions. `tile_scheduler_metadata` and `num_splits` are frozen caller
  inputs, so it is not addressable inside this boundary.

## 2. The instrument had to be fixed before the gate meant anything

The gate compares four estimators against 1.03, so the instrument's own null
spread bounds what it can resolve. Timing one ~30 µs call between two CUDA
events leaves per-observation event and synchronization cost comparable to the
work measured. Measured on a **stock-versus-stock null, the identical installed
binary in both arms**:

| Instrument | Null spread over 24 estimators | Half-width |
|---|---|---:|
| 1 call per observation | 0.9894–1.0123 | ±1.23% |
| 1 call per observation, prior campaign | 0.9673–1.0291 | ±2.9% |
| **20 calls per observation** | **0.9973–1.0028** | **±0.28%** |

At one call per observation the same-binary null spread is the same size as the
effect under test, so that instrument cannot resolve a 1.03 gate at this scale.
Batching 20 consecutive calls inside one event pair amortizes the overhead
symmetrically for both arms — it also exposed **2.86 µs per observation** of pure
instrument cost, which had been diluting both arms toward 1.0. The 1.03 gate
itself is unchanged; only the instrument's variance is reduced. Every decision
below uses the validated instrument, and the superseded single-call runs are
retained. See [`evidence/null_stock_stock_m16_k1.json`](evidence/null_stock_stock_m16_k1.json)
and [`evidence/null_stock_stock_m16_k20.json`](evidence/null_stock_stock_m16_k20.json).

## 3. The candidate

`p1_consumer_scale` ([`v32_p1_consumer_scale.cu`](../../../flashmla/csrc/glm52_hotspot/v32_p1_consumer_scale.cu),
macro `GLM52_CONSUMER_SCALE_GATHER` on top of the prior composite).

In stock, the index warp publishes TMA coordinates only *after* completing its
own 64 scattered FP32 scale loads, so every 64-token block serializes a full
scattered-load round trip in front of the raw-NoPE and RoPE TMA gather issue.
Under the frozen V32 contract `coord * 656` equals
`block*stride_kv_block + idx_in_block*stride_kv_row`, because `stride_kv_row ==
TMA_K_STRIDE == 656` and `stride_kv_block == 64*656` — both enforced host-side in
`api.cpp`. The candidate recovers each scale address from the already-published
coordinate with one multiply and gathers it in the 128-thread dequant consumer,
issued *before* the raw-NoPE wait, so the scattered latency overlaps the TMA
gather it no longer blocks.

This is not a rerun of the rejected B1. B1 kept the loads on the index warp and
moved only a barrier, so that warp still serialized one round trip per block
before it could accept the next block.

Preserved: two-buffer topology, gather count and coordinates, byte-identical
shared-memory footprint, 16 barriers with unchanged arrive/wait counts, the
output math including invalid-token zeroing, and the unchanged stock combine.

Generated binary: 3368 static main instructions, 168 registers/thread, 16
barriers, zero stack/local/spill. `LDG` moves 12→14 — two scattered scale loads
removed from the index warp, four added to the consumer — exactly the predicted
delta. The combine SASS is byte-identical across all four variants. The
generated main SASS is identical across both source-hash generations of this
variant, so the measurements bind to the committed source.

## 4. Results

Correctness is bitwise exact against installed stock at both buckets: 29
boolean gates per bucket plus 103 adversarial gates per bucket, zero failures,
covering leaf eager, non-default stream, independently captured leaf graphs
before and after Q/index mutation, deterministic repeated replay with poisoned
outputs, the containing `_forward_flashmla_kv` in eager and graph, output/LSE
value-shape-stride-ownership checks, input immutability, exact provider hit
count with zero fallback, wrong-page zero-launch rejection, and the full
value/index/scheduler matrix.

Nsys confirms exactly one prefixed main followed by one stock combine in eager
and in each of five graph replays, with no other device kernel in the marked
ranges. Device graph main-only medians: M16 17,152→16,192 ns (1.059), M32
24,480→23,296 ns (1.051).

Each cell is the minimum and maximum over all four required estimators in three
independent alternating series, 100 pairs per series, validated instrument:

| Bucket | Graph containing region | Graph leaf | Eager leaf | Eager containing region |
|---|---:|---:|---:|---:|
| M16 | **1.0613–1.0752** pass | **1.0739–1.0769** pass | 1.1375–1.2163 pass | 0.6683–0.7583 **fail** |
| M32 | **1.0619–1.0656** pass | **1.0628–1.0687** pass | 1.0548–1.0783 pass | 0.6637–0.7403 **fail** |

Every gate decision in [`evidence/timing_gate_audit.json`](evidence/timing_gate_audit.json)
is recomputed from the raw ordered pairs rather than trusting the stored
summaries.

## 5. Why the eager containing lane cannot pass

The identity control — kernel source-identical to upstream stock — fails the
same lane at 0.620–0.696, with a constant **+17.4 µs** tax
([`evidence/identity_eager_k20_m16.json`](evidence/identity_eager_k20_m16.json)).
Direct stage timing attributes it
([`evidence/dispatch_stage_times_before.json`](evidence/dispatch_stage_times_before.json)):

| Stage | Median host µs per call |
|---|---:|
| ABI guard (six tensor contracts) | 4.82 |
| registry lookup | 2.96 |
| NVTX context manager, **even when NVTX is disabled** | 1.73 |
| hit recording | 0.49 |
| `config.is_enabled` | 0.29 |
| phase inference | 0.13 |
| profiler range name | 0.13 |

The lane is structurally unreachable. A candidate's containing-eager time cannot
beat its own leaf-eager time plus the stock wrapper cost, and with the identity
control's measured leaf (27.50 µs) and wrapper (3.17 µs) that floor is 30.67 µs
against a stock containing region of 31.22 µs — a ceiling of **1.0179 even with a
zero-cost guard**. The candidate path also becomes host-bound in eager mode
while stock stays device-bound, which is why its absolute eager numbers vary
across processes while the graph lanes are tight and reproducible.

This is an API-v1 integration property, not a kernel property. Unblocking it
requires an integration change — a C++-side guard, or selecting the provider
only under graph capture — not a kernel change. The measured, actionable items
are handed to the integration owner in the ledger; they were deliberately not
applied here because they cannot change the disposition and editing a
fail-closed safety guard at the end of a campaign carries real risk.

## 6. The ablation, and what it corrected

A diagnostic ablation substituted a constant scale, deleting the whole scattered
chain — global gather, FP32→BF16 conversion and shared staging. Its output is
numerically wrong by construction, so it is never promotable and was never used
for a correctness run. On the validated instrument:

| Variant | M16 graph containing | M16 graph leaf | M32 graph containing | M32 graph leaf |
|---|---:|---:|---:|---:|
| chain deleted | 0.9988 | 1.0082 | 0.9970 | 0.9992 |
| chain relocated (P1) | 1.0613 | 1.0739 | 1.0619 | 1.0628 |

Deleting the chain is worth about nothing; relocating the same loads is worth
about 6%. The chain's cost is its **position in the dependency graph**, not its
instruction count or its memory traffic. Deleting it also removes an accidental
L2 prefetch — the four FP32 scales at token byte offset `[512,528)` share 128-byte
sectors with the RoPE bytes at `[528,656)` that the RoPE gather then reads — so
the two effects cancel. P1 keeps the prefetch and removes only the
serialization. This also explains the prior campaign's B1 result, and why seven
prior candidates that only changed instruction counts all measured ~1.00.

## 7. Non-attempts, with the evidence that closed them

- **P2, conflict-free dequant consumer mapping** — not built. Shared-memory
  stalls are 10.7% against 56.3% for long-scoreboard plus barrier, and the prior
  B2 already implemented this remap and lost to its own added address
  arithmetic. Rerunning it would spend an identity on a mechanism the current
  profile shows is not the limiter.
- **P3, PTX scheduling refinement** — not built. Of the three predeclared changes
  the plan allows, P1 already delivers earlier eligible TMA issue; the ablation
  bounds any further shortening of the scale dependency chain at ~1.00; and
  registers/barriers/stack/local/spill are already 168/16/0/0/0, leaving no
  live-range headroom. The plan's "near the threshold" precondition for a rescue
  refinement also does not apply, since P1 is already past 1.03 on both graph
  lanes.
- **Two-SM rewrite** — not built. The capture shows the opposite of the required
  precondition: one CTA per SM already fills all 148 SMs in a single wave, DRAM
  throughput is 13.7% of peak, and 816 free shared bytes could not stage it.
- **Second NCU capture** — not invoked. The one permitted capture answered the
  stated question; the survivor's mechanism was settled by the generated binary,
  the Nsys spans and the ablation.

## 8. Repository validation

All nine SGLang GLM-5.2 hotspot registry tests pass. All campaign Python
compiles, all 35 retained JSON artifacts parse, and `git diff --check` is clean
in all three worktrees. The SGLang worktree is unmodified by this campaign.

Two broader base checks remain nonzero for inherited contracts outside this
campaign, unchanged from the prior campaign's disclosure: `serving_native/selftest.py`
expects `PRODUCTION_FLASHMLA_KV_DECODE_CASES` in a mandated, unmodified SGLang
base file, and `testbench/bin/verify_harness.py` reports pre-existing stale
generated projections. Neither touches the provider, dispatch guard, FlashMLA
source, generated binary, correctness harness or measurements, and this goal
forbids regenerating them.

## 9. Provenance and limitations

Measurements were taken while the FlashMLA worktree carried this session's
uncommitted variant sources, so the retained provider evidence records
`65293ac…-dirty`. That is disclosed rather than relabelled, and it is closed
rather than merely noted: after committing as `b5af443`, rebuilding the
candidate from the clean tree reproduces the measured build id
`a39236323dc57a97fbfc28fa479c5cd56ce94c065824912fbd95a98a15160649` exactly
([`evidence/build_p1_post_commit.json`](evidence/build_p1_post_commit.json)), so
the committed bytes are the measured bytes. The generated main SASS is also
identical across both source-hash generations of the candidate
(`725c9b8ff2a7e332`), so the measurements bind to the committed machine code.

This host has four B200s, so checkpoint-backed TP8/DP8/EP8 acceptance is not
locally runnable. That does not affect this disposition, because a mandatory
local lane already fails. Commands are retained in
[`EXTERNAL_ACCEPTANCE.md`](EXTERNAL_ACCEPTANCE.md), which must not be used to
seek an external override for a candidate that fails a local gate.

No remote state was modified. Production default remains off and stock remains
the fallback for every bucket.
