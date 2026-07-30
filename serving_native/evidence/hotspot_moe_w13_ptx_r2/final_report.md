# GLM-5.2 fused W13 decode, PTX/SASS round-2 result

## Disposition

**`no-replacement` for round 2.** None of the plan's three bounded hypotheses
yields a deployable gain over the round-1 BM16 two-SM candidate, and the one
identity that was built is functionally incorrect.

Round-1's BM16 two-SM candidate `(16,128,128,12,2)` remains the standing
`external-acceptance-candidate` and was reconfirmed on the current checkout from
round-2's own build. **The production default stays stock and the candidate
stays off**; checkpoint-backed TP8/DP8/EP8 acceptance has still not run.

## The binding limit, measured

The round-2 plan asked for further PTX/SASS and tcgen05 scheduling gains on the
BM16 path. One NCU capture of the exact survivor answers why there are none.
Physical B200 `GPU-30b619de-87f2-1862-0d07-a595da8fe417`, exact frozen ABI,
`masked_m = 4` on all 32 experts, PDL on, 148 SMs:

| Metric | BM16 two-SM survivor | stock BM128 two-SM |
|---|---:|---:|
| `gpu__time_duration.sum` | 126.656 us | 135.744 us |
| `dram__bytes_read.sum` | 815.19 MB | 841.52 MB |
| `dram__bytes_write.sum` | 7.00 MB | 31.93 MB |
| `dram__bytes_read.sum.pct_of_peak_sustained_elapsed` | **83.92%** | 80.82% |
| achieved DRAM read rate | **6.436 TB/s** | 6.199 TB/s |
| `gpu__compute_memory_throughput` % of peak | **84.64%** | 83.89% |
| `sm__pipe_tensor_cycles_active` % of peak | **4.57%** | 34.30% |

Three facts close the search space:

1. **Traffic is 1.2% above the irreducible minimum.** Every expert's full FP8
   weight matrix must be read exactly once: `32 × 4096 × 6144` = 805.31 MB.
   Measured reads are 815.19 MB — weights plus B-scales, activations and
   overhead. There is no traffic left to remove.
2. **The kernel is at the achievable memory speed-of-light**, 83.92% of peak
   sustained DRAM read at 6.436 TB/s, which is at the top of the realistic
   sustained HBM3e band on this part. Even driving DRAM to 100% of peak would
   only reach 106.3 us, an upper bound of 1.19x that no real kernel attains.
3. **Compute is irrelevant.** Tensor pipe at 4.57% of peak. Round 1 already
   converted stock's 34.30% tensor-pipe occupancy into memory-bound time, which
   is why `long_scoreboard` stalls *rose* from 25.747 to 27.649 while duration
   fell. 88.6% of per-PC stall samples are memory-arrival waits (67.34%
   `long_scoreboard` plus 21.21% on a transaction-barrier try-wait);
   `membar` and every throttle are zero.

Round-1's 1.047x therefore came mostly from moving less data — the traffic ratio
alone is 840.9/818.9 = 1.0269 of the measured 1.0472 — not from better
scheduling. That is the ceiling round 2 ran into.

## What each hypothesis did

| Hypothesis | Outcome | Basis |
|---|---|---|
| H1 BM16 epilogue / TMEM store reduction | closed without spending an identity | Writes are 0.85% of traffic (7.00 of 822.2 MB) and the store path is already at its instruction floor: `effective_m` is unconditionally `BLOCK_M` for `MGroupedMasked`, so one store stage emits exactly 4 `LDTM.16`, 2 `STSM.16.MT88.4`, 2 `UTMASTG.2D`. Removing all output traffic could not reach 3%. |
| H2 barrier / mbarrier overlap | one identity built; **rejected, incorrect** | See below. The terminal-cluster-sync variant was not authorized (its "material time" precondition is unmet: 3 `UCGABAR` pairs, once per launch) and the arrival-count variant has nothing to remove (the 32-lane arrive is already one warp-aggregate SASS instruction). |
| H3 BM32 rescue | closed without spending an identity | In a kernel bound by bytes moved, BM32 strictly increases bytes versus BM16 and doubles the store surface. No PTX or tile change removes traffic. BM32 stays stock, as the plan permits. |

## The built identity and why it fails

**H2-SF-BYPASS**, predeclared in
`profile/w13-bm16-r2-survivor-em4-20260730/REPORT.md` before any code was
written. The UTCCP transposer warp relays every k-block's TMA arrival to the MMA
warp through `with_sf_full_barriers`, yet it rewrites shared-memory scale factors
only on one k-block in four. The experiment let the MMA warp wait
`full_barriers[stage_idx]` directly on the other three and had the transposer
participate only on scale-factor k-blocks.

The delta was emitted exactly as intended — the bypass cubin is smaller than the
control and keeps the identical topology proof (`cta_group::2`, cluster 2, 148
CTAs, 256 threads, 230,188 B dynamic shared memory, 35 registers, zero
stack/local/spills, 16 `UTCQMMA.2CTA`, 10 `UTMALDG.2D`, 4 `LDTM`) — and it is
wrong:

- `compute-sanitizer --tool synccheck` on the two-SM bypass:
  `Barrier error detected. Missing wait.` at
  `infini_kernel_glm52_moe_w13_decode_em4_bm16_2sm_sfrelaybypass+0x1b90`,
  thread (0,0,0) in block (7,0,0), barrier at shared address 0x1038408.
- The one-SM bypass hangs under synccheck and raises `Unknown Error` under
  memcheck.
- Both bypass identities die with `CUDA error: unspecified launch failure` in
  the exact-numerics gate, each in its own process.
- `synccheck` and `memcheck` both report **0 errors** on the two-SM control.

The root cause is structural. `with_sf_full_barriers[i]->init(kNumMulticast * 32)`
is arrived by **both** CTAs' transposer warps through `arrive(0u)`, which CUTLASS
implements as a *remote* arrive into the leader CTA's barrier. Reaching that
count is therefore the proof that both CTAs' TMA loads have landed — it is the
cross-CTA data-readiness handshake that a `cta_group::2` UMMA requires, not
removable relay overhead. Bypassing it lets the leader issue a two-CTA UMMA over
the peer CTA's not-yet-arrived tile.

## Controls and reconfirmation

The round-2 plumbing is provably inert when the new flag is 0. The control
identity's `kernel.sass` SHA256 is
`4b5275310bf5c96a050f8c0e868afc25a154c5f650a30f53af660aef984d1607`, **identical
to round-1's retained BM16 two-SM SASS**. Both BM16 control identities pass 20
exact-numerics cases each — expected-M 4/5/8/9 crossed with uniform,
empty-expert, tile-boundary, skewed and maximum masks, 3 repeats — with zero
mismatched elements against stock on identical input bytes and no writes beyond
the tile-aligned store envelope.

The standing candidate still clears the fairness gate on the current checkout,
built entirely from round-2's own task-local cache. Three independent
same-process 50-pair alternating series per lane, all four estimators per series:

| Lane (expected-M 4) | Stock p50 (us) | Candidate p50 (us) | Weakest per-series estimator | Hits | Fallback |
|---|---:|---:|---:|---:|---:|
| leaf eager | 150.592 | 144.352 | 1.042332 | 167 | 0 |
| leaf graph | 151.552 | 145.200 | 1.042598 | 208 | 0 |
| region eager | 230.560 | 223.232 | 1.031819 | 167 | 0 |
| region graph | 231.440 | 223.264 | 1.036098 | 208 | 0 |

All four harness self-audits are valid, on one physical GPU at SM 1965 MHz.
Round-1's full 16-lane matrix (all expected-M points, leaf and containing region,
eager and graph) remains the authority for the standing candidate and is
unchanged; these four lanes reconfirm it rather than replace it.

## Identities built (2 of the allowed 3)

| Identity | Config | Topology proof | Registers | Dynamic smem | Verdict |
|---|---|---|---:|---:|---|
| `…_em{4,5,8,9}_bm16_2sm` (control) | `(16,128,128,12,2,0)` | `cta_group::2`, 16 `UTCQMMA.2CTA`, cluster 2, 3 `UCGABAR` pairs | 35 | 230,188 | retained, SASS-identical to round 1 |
| `…_em{4,5,8,9}_bm16_1sm` (control) | `(16,128,128,11,1,0)` | plain `UTCQMMA`, cluster 1, no `UCGABAR` | 31 | 223,020 | correct; still below 1.03 (round-1 result) |
| `…_em4_bm16_2sm_sfrelaybypass` | `(16,128,128,12,2,1)` | `cta_group::2` preserved | 35 | 230,188 | **rejected: synccheck missing-wait** |
| `…_em4_bm16_1sm_sfrelaybypass` | `(16,128,128,11,1,1)` | cluster 1 preserved | 31 | 223,020 | **rejected: hang / launch failure** |

## Validation matrix

| Requirement | Status |
|---|---|
| CPU-only environment, source, ABI and negative-evidence audit before source change | pass |
| Task-local cache only; no reuse or mutation of another task's binary cache | pass |
| Same-source stock/candidate builds, identical normalized build plan | pass |
| Mandatory pre-timing identity gate on every built identity | pass |
| Round-2 control reproduces round-1 machine code byte-for-byte | pass |
| Exact-numerics gate vs production stock for both control identities | pass |
| Four-estimator, three-series fairness reconfirmation of the standing candidate | pass |
| No fallback after selected candidate invocation; zero fallback counted | pass |
| Harness contract tests (40) and 46-workload selftest | pass |
| A round-2 identity that beats the round-1 candidate | **none found** |
| Checkpoint-backed TP8/DP8/EP8 full-region and serving acceptance | **not run** |
| Production default | **stock / candidate off** |

## What a future round should and should not do

Do not reopen H1, H3, or the SF-relay bypass: each is now closed by measurement
or by a synccheck-confirmed correctness violation. Within the frozen ABI the
BM16 two-SM kernel has no meaningful in-kernel headroom left — it reads 1.2%
more than the theoretical minimum bytes at 84% of peak sustained bandwidth.

Any further W13 gain has to change how many bytes cross DRAM, which means
leaving this task's frozen boundary: a longer contiguous run per weight row
(BLOCK_K above 128, which the current UE8M0 granularity and SF pipeline
hard-code), weight reuse across more than one m-block per expert, or a fusion
that avoids re-reading the weights at all. Those are ABI-level or
fusion-boundary changes and belong to a differently scoped task, not to a
PTX/SASS round.
