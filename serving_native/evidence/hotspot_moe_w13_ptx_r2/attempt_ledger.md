# GLM-5.2 W13 PTX/SASS round-2 attempt ledger

## Frozen scope

Round-2 optimizes only the exact fused W13 decode path already validated in
round 1: E32, expert slab 1024, K6144, N4096, packed `int32` UE8M0 scales,
local decode buckets M16/M32, expected-M 4/5/8/9, current SGLang API-v1
`infini_kernel` provider. Baseline is round-1's BM16 two-SM candidate
`(16,128,128,12,2)` at DeepGEMM `87e0359`; the performance denominator is
always the production stock path.

The plan allowed at most three identities against three bounded hypotheses:

1. BM16 epilogue / TMEM store reduction;
2. barrier / mbarrier overlap;
3. BM32 rescue.

## Attempts and decisions

| Attempt | Hypothesis and exact delta | Evidence | Decision and why |
|---|---|---|---|
| R2-A0 task-local rebuild | Round-2 must build its own artifacts rather than reuse round-1's binary cache. Parameterize `build_variants.py` on `GLM52_TASK_BUILD_DIR` and rebuild same-source stock plus the round-1 candidate commit into `cache/moe_w13_ptx_r2`. | Source materialization is byte-reproducible and reproduces round-1's tree hashes exactly (`stock 917592ab…`, `candidate d682daa6…`, `diff 465c8373…`). Stock and candidate normalized build plans are identical (`2f658765…`). | Retained. Round-2 never reads or writes another task's cache. |
| R2-H1 epilogue / TMEM store reduction | Reduce output writes or TMEM drain without changing numerics. | Closed before implementation by measurement, not opinion. NCU on the survivor: `dram__bytes_write.sum` = 6,996,992 B against `dram__bytes_read.sum` = 815,190,016 B, so writes are 0.85% of traffic. The store path is already at its instruction floor: `get_aligned_effective_m_in_block` returns `BLOCK_M` unconditionally for `MGroupedMasked`, so `effective_m` = 16 and `num_stores` = 1, emitting exactly 4 `LDTM.16`, 2 `STSM.16.MT88.4` and 2 `UTMASTG.2D` per scheduled block. | Rejected without spending an identity. Eliminating **all** output traffic could not reach 3% of a kernel that moves 815 MB of reads. Recorded as a quantitatively closed route. |
| R2-H3 BM32 rescue | Lift the historical BM32 two-SM BA estimator of ~1.028 with a predeclared PTX/tile change. | Closed by the same measurement. The kernel is bound by bytes moved: 83.92% of peak sustained DRAM read, tensor pipe at 4.57% of peak. BM32 strictly increases bytes relative to BM16 — larger activation footprint and double the store surface (stock BM128 writes 31.93 MB, BM16 writes 7.00 MB). No PTX change removes traffic. | Rejected. BM32 stays on stock, as the plan permits. Its historical deficit was real, not noise. |
| R2-H2a terminal cluster sync refactor | Refactor the `// TODO: Remove redundant synchronization` cluster sync at kernel exit. | The plan authorizes this "only … if profiling attributes material time to it". It does not: the kernel contains exactly 3 `UCGABAR_ARV`/`UCGABAR_WAIT` pairs (two structurally required around 2-CTA TMEM allocation, one at exit), each executing once per launch inside ~232,400 SM cycles, while the measured `barrier` stall sits on a steady-state transaction-barrier try-wait. | Not attempted. The plan's own precondition is unmet; attempting it would be checklist-filling. |
| R2-H2c mbarrier arrival-count reduction | Replace the 32-lane `with_sf_full_barriers` arrive with a single elected lane plus `init(kNumMulticast)`. | Closed by the generated binary. The whole kernel already contains only 9 arrive instructions (`4 SYNCS.ARRIVE.TRANS64.RED.A1T0`, `4 SYNCS.ARRIVE.TRANS64`, `1 SYNCS.ARRIVE.TRANS64.RED`): the 32-lane arrive is already one warp-aggregate SASS instruction. | Rejected. There is no instruction to remove. |
| **R2-H2b SF-relay bypass (the one built identity)** | Predeclared in `profile/w13-bm16-r2-survivor-em4-20260730/REPORT.md` before implementation. The UTCCP transposer warp relays every k-block's TMA arrival to the MMA warp through `with_sf_full_barriers`, but rewrites shared-memory scale factors only when `k_block_idx % kNumSFAStagesPerLoad == 0`, i.e. one k-block in four. Delta: on the other three k-blocks the MMA warp waits `full_barriers[stage_idx]` directly and the transposer participates only on SF k-blocks. Gated by `DEEP_GEMM_W13_SF_RELAY_BYPASS`, emitted only for the named W13 entry; `w13_config` gained a sixth element carrying the flag. | Built and identity-gated successfully: two-SM bypass proves `cta_group::2`, cluster 2, 148 CTAs, 256 threads, 230,188 B dynamic shared memory, 35 registers, 0 stack/local/spill, 16 `UTCQMMA.2CTA`, 10 `UTMALDG.2D`, 4 `LDTM`, and a smaller cubin than the control, so the intended instructions really were removed. It is nevertheless **functionally broken**. `compute-sanitizer --tool synccheck`: `Barrier error detected. Missing wait.` at `infini_kernel_glm52_moe_w13_decode_em4_bm16_2sm_sfrelaybypass+0x1b90`, thread (0,0,0) in block (7,0,0), barrier at shared address 0x1038408. The one-SM bypass hangs under synccheck (240 s timeout) and raises `Unknown Error` under memcheck. In the exact-numerics gate both bypass identities died with `CUDA error: unspecified launch failure` in isolated processes. | **Rejected — mandatory failed attempt.** The relay is load-bearing, not overhead. `with_sf_full_barriers[i]->init(kNumMulticast * 32)` is arrived by *both* CTAs' transposer warps via `arrive(0u)`, which CUTLASS implements as a **remote** arrive into the leader CTA's barrier. Reaching that count is therefore the proof that *both* CTAs' TMA loads have landed — the cross-CTA data-readiness handshake the 2-CTA UMMA requires. Bypassing it on non-SF k-blocks lets the leader issue a `cta_group::2` UMMA over the peer CTA's not-yet-arrived tile. The plan's instruction to "preserve all fences until a producer-consumer proof establishes a safe replacement" is exactly right here; the proof runs the other way. |
| R2 controls | The round-2 plumbing must not perturb the validated baseline. `w13_config = (…,0)` must reproduce round-1's kernel. | The round-2 control's `kernel.sass` SHA256 is `4b5275310bf5c96a050f8c0e868afc25a154c5f650a30f53af660aef984d1607`, **identical to round-1's retained BM16 two-SM SASS**. Both BM16 control identities pass 20 exact-numerics cases each (expected-M 4/5/8/9 × uniform / empty-expert / tile-boundary / skewed / maximum masks, 3 repeats): zero mismatched elements against stock on identical input bytes and no writes beyond the tile-aligned store envelope. `synccheck` and `memcheck` report 0 errors on the two-SM control. | Retained. The widened `w13_config` and the new `#define` hook are provably inert when the flag is 0. |
| R2 baseline reconfirmation | Confirm the standing candidate still clears every estimator on the current checkout, built from round-2's own cache. | Four lanes at expected-M 4 on physical B200 `GPU-30b619de-87f2-1862-0d07-a595da8fe417`, SM 1965 MHz, three independent 50-pair alternating series each, all four estimators per series: leaf eager weakest 1.042332, leaf graph 1.042598, region eager 1.031819, region graph 1.036098. 167/208 candidate hits, zero fallback, all four harness self-audits valid. | Retained. The standing candidate remains valid on the current checkout. |
| Harness corrections | Round-1's runner, auditor and artifact auditor pinned round-1's single candidate commit, the 5-element `w13_config`, and the round-1 provider filenames, so they fail closed on any round-2 identity. | Extended fail-closed rather than loosened: candidate commits are pinned in an allowlist (`87e0359…`, `e29df03…`) each bound to its exact source-tree and diff SHA256; variant→config and variant→provider-filename maps are explicit; the artifact auditor takes `--relay-mode` and `--include-hash` so the validated portfolio and the experiment are audited separately. 40 harness contract tests and the 46-workload selftest pass. | Retained. The first reconfirmation attempt was correctly rejected by the auditor and **no failed-audit result was used as evidence**. |

## Measurement hygiene

Every CUDA command ran through `with_hotspot_gpu.sh`. The wrapper rejected
GPU 0 once ("visible compute process") and GPU 3 was busy at audit time; each
complete series stayed inside one lease on one physical GPU. NCU was invoked
once, for the single predeclared survivor question, and its 3×10-style
attribution is never used as a performance number — the 3×50 unprofiled series
are the performance authority.

The first exact-numerics run was invalid for an unrelated reason and is
disclosed rather than erased: the FP8 E4M3FN test data was drawn from a byte
range that includes the NaN encodings 0x7F/0xFF, so NaN ≠ NaN was counted as a
mismatch even for the control. The generator was restricted to finite byte
patterns and every identity was then re-run in its own process so that one
identity's launch failure could not poison another's CUDA context.

## Resource note

The shared root filesystem fell from 13 GiB free at the start of this round to
under 1 GiB during it, driven by other activity on the host (`/tmp/ray`,
`/tmp/torchinductor_*`, unrelated `nvcc` temporaries and `/var/crash` all
predate this session). Round-2's own footprint is 395 MB of task-local cache
plus 800 KB of compressed profiler reports. No further cache expansion was
performed after the threshold was crossed, and no unrelated file was deleted.

## Rollback

Nothing is enabled. `DEEP_GEMM_W13_SF_RELAY_BYPASS` is emitted only when the
sixth `w13_config` element is 1, which no validated provider or harness variant
sets. The three round-2 provider modules are default-off and reachable only via
an explicit `SGLANG_GLM52_HOTSPOT_MODULE`. Leaving `SGLANG_GLM52_OPT=0` loads no
candidate DSO at all.
