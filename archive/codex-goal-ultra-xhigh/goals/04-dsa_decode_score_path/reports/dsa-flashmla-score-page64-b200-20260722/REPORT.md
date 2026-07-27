# Goal04 re-audit: fused DSA decode score path on B200

## Disposition

**No replacement.** The locally bound FlashMLA experiment is correct and CUDA
Graph-safe, but loses every eager M16/M32 bucket and has no 3% graph win. No
SGLang dispatch was enabled; stock remains the only active path.

The earlier closeout at Kernel-Harness commit `34f612e` is provisional. Its
`page64-002` DSO had a dynamic `JUMP_SLOT` relocation for the weak SM100 V32
sparse launcher. Because stock FlashMLA was loaded globally first,
`RTLD_LOCAL` did not prove that the candidate invoked its patched launcher.
That artifact (SHA-256 `208b78ba...f1c4`) is preserved as
`flashmla_goal04_page64_ops.page64-002-preemptable.so`; none of its candidate
timings or profiler numbers support this decision.

The authoritative evidence is `attempts/page64-004/`. Its DSO was built
CPU-only with `-Wl,-Bsymbolic`, has `DT_SYMBOLIC`, has no sparse-launcher
dynamic relocation, and has SHA-256
`063882d2195c0b454523184ab0d98808e5acfc6ddfef6b8618a53f83d421c853`.

## Production reachability and ABI

The explicit serving selection is:

`DeepseekSparseAttnBackend._forward_flashmla_kv` →
`sgl_kernel.flash_mla.flash_mla_with_kvcache` →
`sgl_kernel::fwd_kvcache_mla` → FlashMLA SM100 V32 head64 split-KV plus
combine.

Required launch flags are `--attention-backend dsa`,
`--kv-cache-dtype fp8_e4m3`, `--dsa-prefill-backend flashmla_sparse`,
`--dsa-decode-backend flashmla_kv`, `--tp 8 --dp 8
--enable-dp-attention`. Attention TP is one, so the production local M remains
16 or 32 on each DP rank; the serving-native workload itself is world size one.

| input/output | exact production ABI |
|---|---|
| Q | BF16 `[M,1,64,576]`, stride `[36864,36864,576,1]` |
| paged KV | FP8 E4M3 `[1+128M,64,1,656]`, stride `[41984,656,656,1]`; page 64 |
| physical sparse indices | int32 `[M,1,2048]`, stride `[2048,2048,1]` |
| cache lengths | int32 `[M]`, value 2048 after top-k clamp |
| scheduler | int32 metadata `[148,8]`; split prefix int32 `[M+1]` |
| block table | empty int32 `[M,0]` for physical-token indexing |
| parameters | context 8192, top-k 2048, value dim 512, scale 0.0625, noncausal, FP8 cache |
| output | BF16 `[M,1,64,512]` |

The static oracle checks the named bucket, device, all relevant shapes,
strides, dtypes, head/value parameters and scale without a device-to-host read.
Every unsupported case calls `runtime.reference(inputs)`.

## Fused source mapping and experiment

The reached source is `csrc/sm100/decode/head64/instantiations/v32.cu` and
`kernel.cuh`:

- lines 218–327: score reduction and online softmax;
- lines 387–420: split result write-out;
- lines 531–552: Q/K tcgen05 accumulation (SV follows in the same main loop);
- lines 654–745: sparse-index, TMA-coordinate, scale and validity producer;
- `combine.cu` lines 18–161 and dispatch lines 165–213: split combine.

The upstream is `https://github.com/sgl-project/FlashMLA`. FlashMLA commit
`5fa2b1f63aa74a72f2db0e3797ee0ffa867d38cd` (base
`05e26647fe840b8baedae486c2d86d5ce4efeb7c`) specializes only the original V32
cache producer: physical token index becomes the direct TMA coordinate and
scale address `index*656`. Generic/extra-cache logic stays generic and invalid
index semantics are preserved. The source patch is under `source/`; the build
record is `builds/bsymbolic-001/`. The CPU-only reconstruction/build sequence
is `harness/prepare_source.sh` followed by `harness/build_candidate.sh` from
this run directory. Set fresh `GOAL04_BUILD_ID`, `GOAL04_BUILD_DIR`, and
`GOAL04_ARTIFACT` paths to reproduce without overwriting committed evidence;
the canonical artifact is
`profile/dsa-flashmla-score-page64-b200-20260722/artifacts/flashmla_goal04_page64_ops.so`.
Import resolution loads installed stock
`/mnt/OS-oKqEXySb/home/qinhaiyan/miniconda3/envs/sglang/lib/python3.12/site-packages/sgl_kernel/flashmla_ops.abi3.so`
first and then opens the custom `sgl_kernel_goal04_page64` DSO locally. The
link-time `DT_SYMBOLIC` binding keeps the custom launcher internal without
allowing it to interpose on stock.

## Correctness and fallback

Both authoritative validation JSONs pass:

- stock versus candidate output;
- 128 interspersed `-1` slots per request;
- non-default CUDA stream execution;
- unsupported softmax scale rejection and exact stock fallback.

All six graph comparisons report exact post-mutation agreement
(`max_abs_diff=0`). The graph input mutation changes stock output by
`0.0059967041` (M16) or `0.0062103271` (M32), proving replay consumes current Q
and sparse-index storage rather than capture-time outputs.

## Paired eager results

All 600 pairs were measured in the single `page64-004` wrapper invocation on
physical GPU 0. Each speedup is the median of 100 per-pair `reference/candidate`
ratios, not the ratio of the two displayed marginal medians.

| bucket | run | ref p50 (ms) | candidate p50 (ms) | paired speedup | 3% gate |
|---|---:|---:|---:|---:|---|
| M16 | 1 | 0.044336 | 0.048704 | 0.90454x | fail |
| M16 | 2 | 0.044448 | 0.047376 | 0.93343x | fail |
| M16 | 3 | 0.047600 | 0.051648 | 0.92437x | fail |
| M32 | 1 | 0.050400 | 0.056000 | 0.90813x | fail |
| M32 | 2 | 0.049584 | 0.053824 | 0.91585x | fail |
| M32 | 3 | 0.051024 | 0.055584 | 0.90966x | fail |

## Paired CUDA Graph results

Each row is another 100 alternating graph replays after correctness capture and
the live-input mutation control.

| bucket | run | ref p50 (ms) | candidate p50 (ms) | paired speedup | max diff | 3% gate |
|---|---:|---:|---:|---:|---:|---|
| M16 | 1 | 0.033136 | 0.033296 | 0.99665x | 0 | fail |
| M16 | 2 | 0.032416 | 0.032864 | 0.99165x | 0 | fail |
| M16 | 3 | 0.032800 | 0.032832 | 1.00145x | 0 | fail |
| M32 | 1 | 0.036448 | 0.036560 | 0.99334x | 0 | fail |
| M32 | 2 | 0.035888 | 0.036704 | 0.98175x | 0 | fail |
| M32 | 3 | 0.036384 | 0.036928 | 0.98670x | 0 | fail |

## Fused-region and profiler result

The score path is fused into the main split-KV kernel and is not separately
timeable without changing the kernel. Nsight Systems captured exactly 20 main
and 20 combine launches per bucket. It gives this complete attention-kernel
region decomposition (main start through combine end, including PDL overlap):

| bucket | main p50 | combine p50 | overlap p50 | fused region p50 |
|---|---:|---:|---:|---:|
| M16 | 17.552 µs | 13.312 µs | 4.192 µs | 26.688 µs |
| M32 | 24.704 µs | 10.080 µs | 4.096 µs | 30.656 µs |

NCU reports were parsed through the `ncu_report` Python API. The main kernel is
one 148-CTA wave, 384 threads/CTA, 168 registers/thread, 232,656 bytes dynamic
shared memory and no spills. M16/M32 duration is 22.528/30.016 µs; eligible
warps are only 0.209/0.287 per cycle. Long-scoreboard stall ratios are
5.465/4.465 and barrier ratios 3.735/2.102. Tensor elapsed utilization is
8.85%/13.45% and DRAM read utilization 13.35%/20.02%. The combine kernel is
underfilled (0.173/0.346 waves per SM) and long-scoreboard dominated. Removing
integer coordinate reconstruction therefore does not attack the binding
latency/synchronization limits; the measured isolated-op path regresses in
eager mode.

Stock profiler reports were collected in a different wrapped campaign/GPU, so
they are retained as stock characterization but are not used for a direct
candidate profiler delta.

## SGLang region/end-to-end status and enable policy

The fused split-KV-plus-combine attention-kernel region is the Nsight Systems
result above. The broader containing DSA pipeline—cache preparation, indexer
score/top-k and selected attention backend—and a full GLM-5.2 SGLang
request/server run were **not promoted or claimed**: the candidate failed the
prerequisite paired microbenchmark in both buckets, and this host has four
B200s while the production acceptance lane is TP8/DP8/EP8. There is no separate
four-rank DSA collective to validate—the attention backend is rank-local with
attention TP=1—so a four-GPU diagnostic cannot replace the eight-rank server
gate. This is an explicit “not run after microbenchmark rejection”
region/end-to-end result, not a weakened acceptance gate.

Enable policy: no bucket is enabled. M16, M32, every unsupported ABI, and the
complete server path remain stock. The experimental DSO is external evidence
only and no SGLang dispatch source calls it.

## Environment and provenance

Authoritative campaign: B200 SM100 (148 SMs), physical GPU 0,
UUID `GPU-30b619de-87f2-1862-0d07-a595da8fe417`, PCI `05:00.0`, driver
610.43.02. Boundary graphics clocks were 780 MHz before and 1740 MHz after;
memory remained 3996 MHz and max graphics clock is 1965 MHz. Torch is
2.11.0+cu130, CUDA runtime 13.0, nvcc 13.2.78, sgl-kernel 0.4.4, NCU 2026.1.1,
and Nsight Systems 2025.6.3. Stock extension SHA-256 is
`d8d97150bd86381c73406603cb7d6b682767535e0526053f04e3acefadb13316`.
The SGLang worktree is clean at `d33ad5bf4`; the source bundle and readable
patch record the removed task-local FlashMLA experiment at `5fa2b1f`.
The campaign intentionally records a dirty Kernel-Harness tree because its raw
artifacts were not committed at run time. This audit commit preserves those raw
artifacts and documents post-run, non-measurement hardening: stricter fail-closed
candidate guards, corrected future profiler-field labels, and reproducible source
and build tooling. The supported measured operator call is unchanged; the raw
profiler-label correction is recorded in `attempts/page64-004/analysis/README.md`.

The append-only knowledge log preserves the original `20260722a` entry and the
first audit draft `20260722b`; `20260722c` explicitly supersedes both and is the
current recipe. The older entries are retained only because knowledge records
cannot be edited or deleted.

The maintainer-corpus audit scanned 32,639 human-review threads and matched 268
threads across 112 SGLang PRs. Recurring blocking concerns were exact bucket
coverage, CUDA Graph compatibility, fail-closed dispatch, and workload-specific
benchmark evidence. GLM history notes (including TP8 graph/e2e precedent) are
under `history/model-pr-history-notes.md`.

## Experiment ledger

| item | `page64-004` record |
|---|---|
| hypothesis | Fixed V32 page64 physical indices already equal flattened TMA coordinates; removing divide/remainder and address reconstruction may reduce the sparse-index producer critical path. |
| baseline evidence | Stock source/SASS contains the runtime coordinate reconstruction; stock NCU showed low eligible warps with long-scoreboard/barrier pressure. |
| exact delta | FlashMLA `5fa2b1f`: direct coordinate and `index*656` scale offset for original V32 pages; custom Torch namespace and locally bound DSO. |
| expected effect | Remove integer divide/remainder/address instructions without changing score scale, selection, softmax, KV addressing, output, stream or graph behavior. |
| correctness | Pass at M16/M32, including interspersed `-1`, non-default stream, unsupported-scale fallback and exact graph replay after mutation. |
| paired distribution | Eager M16 runs: 0.90454x `[p10 0.83979, p90 0.98746]`, 0.93343x `[0.74683, 1.13926]`, 0.92437x `[0.80517, 1.01173]`; M32: 0.90813x `[0.79572, 0.97247]`, 0.91585x `[0.84009, 0.97917]`, 0.90966x `[0.83076, 1.01016]`. Raw 100-pair distributions are the `paired_*.json` files. |
| profiler result | Candidate remains low-eligibility and long-scoreboard/barrier limited. A direct profiler delta is deliberately not claimed because stock profiler collection used a different wrapped GPU. |
| risk | Narrow V32/page64 ABI assumptions, invalid/extra-cache semantics, CUDA Graph capture, and weak ELF symbol preemption. The oracle, validation suite and `DT_SYMBOLIC` relocation gate cover these risks. |
| decision | Reject both buckets: all eager series regress and no graph series clears 3%. |
| rollback | No production dispatch was added. Stock `sgl_kernel::fwd_kvcache_mla` remains active for M16, M32 and all fallbacks. |

## Attempt ledger

| attempt | status | reason |
|---|---|---|
| `stock-001` | invalid | initial campaign stopped on profiler-command correction |
| `stock-002` | valid stock characterization | identity, eager/graph controls, runtime checks, Nsys and NCU complete |
| `page64-001` | invalid | stopped before measurements after candidate-path/loader audit |
| `page64-002` | invalid/provisional | DSO launcher remained dynamically preemptable |
| `page64-003` | incomplete | build was stopped after discovering compilation held the GPU lease; no measurements |
| `bsymbolic-001` | valid CPU build | locally bound DSO and ELF relocation gate |
| `page64-004` | authoritative | one measurement-only wrapper; correctness, 600 eager pairs, 600 graph pairs, Nsys and NCU complete |
