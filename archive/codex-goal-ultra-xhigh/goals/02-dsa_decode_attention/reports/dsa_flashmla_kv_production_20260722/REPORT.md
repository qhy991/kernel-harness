# GLM-5.2 DSA `flashmla_kv` decode: no replacement

## Disposition

No production bucket is enabled.  The split-scheduler configuration regressed
all three eager paired series for both M16 and M32, and the source-level
raw-NoPE bank-layout candidate failed correctness before timing.  Stock
`flashmla_kv` remains active for both buckets with `SGLANG_GLM52_OPT=0`; the
isolated candidate namespace never changed production dispatch.

The authoritative measurement is immutable campaign
[`screen3`](../dsa_flashmla_kv_scheduler_campaign_20260722/runs/screen3), collected
in one flexible-GPU wrapper invocation on physical GPU 1,
`GPU-5b9be10b-5bfc-b658-9b31-f7ae8516dc54`.  Reference and candidate samples
alternate in one process.  The campaign records clocks, topology, repository
state, environment, raw samples, Nsight Systems traces, and full Nsight Compute
reports.

## Exact production path and ABI

The frozen launch contract is explicit
`--dsa-decode-backend flashmla_kv --kv-cache-dtype fp8_e4m3`; the no-flag SM100
TRT-LLM backend and `flash_mla_sparse_fwd` are out of scope.  The exercised call
chain is:

```text
DeepseekSparseAttnBackend._forward_flashmla_kv
  -> sgl_kernel.flash_mla.flash_mla_with_kvcache
  -> torch.ops.sgl_kernel.fwd_kvcache_mla
  -> sm100::decode::head64::flash_fwd_splitkv_mla_fp8_sparse_kernel<ModelType::V32>
  -> flash_fwd_mla_combine_kernel
```

The serving-native reference invokes the actual unbound backend method on the
current PyTorch CUDA stream.  It is a model-free production-ABI trace, not a
claim that an eight-rank server was launched.

| Contract | M16 | M32 |
|---|---:|---:|
| Query | `[16,1,64,576]` BF16 | `[32,1,64,576]` BF16 |
| Packed KV | `[2049,64,1,656]` uint8 | `[4097,64,1,656]` uint8 |
| Sparse indices | `[16,1,2048]` int32 | `[32,1,2048]` int32 |
| Scheduler metadata | `[148,8]` int32 | `[148,8]` int32 |
| Splits | 8/request, 128 total | 4/request, 128 total |
| Output | `[16,1,64,512]` BF16 | `[32,1,64,512]` BF16 |

Each 656-byte KV token contains 512 FP8 latent bytes, four FP32 block scales,
and 64 BF16 RoPE values.  Physical page zero is reserved, pages are interleaved
across requests, sparse positions use a deterministic coprime walk across the
full 8,192-token context, and the softmax scale is 0.0625.

## Reference and per-bucket oracle

Stock passed the independent BF16 oracle and fresh-input CUDA-graph replay in
both buckets.  Maximum oracle error was `3.0517578125e-5`; M16 also passed a
17-invalid-index-per-request check.  Stock graph medians were 30.784 us (M16)
and 36.032 us (M32), and output changed after the captured inputs were mutated,
excluding stale capture-time output.

The three reference-vs-reference controls establish the local paired noise:

| Bucket | Control 1 | Control 2 | Control 3 |
|---|---:|---:|---:|
| M16 | 0.9949x | 1.0017x | 0.9997x |
| M32 | 0.9923x | 1.0000x | 0.9963x |

| Bucket / candidate | Eager paired speedups | Graph result | Correctness | Decision |
|---|---|---|---|---|
| M16, 120 useful scheduler partitions | 0.9582x, 0.8780x, 0.9502x | 1.0354x once; p10 0.9810x, p90 1.1041x | Pass, including fresh-input replay | Reject: all repeated eager series regress; isolated graph median is noisy |
| M32, 112 useful scheduler partitions | 0.8706x, 0.8852x, 0.8747x | 0.8952x | Pass, including fresh-input replay | Reject |
| M16/M32 group-major bank layout | Not run | Not run | Fail at first M16 invocation | Reject before timing; M32 is not exposed to a known-invalid binary |

The production gate requires a repeatable paired median gain of at least 3% and
no enabled-bucket regression.  Neither bucket qualifies.

## Profile and recoverable-time diagnosis

Nsight Systems shows the exact two-kernel chain.  Its stock medians are 17.440 us
main plus 12.176 us combine at M16, and 25.088 us main plus 9.904 us combine at
M32.  Combine kernel start timestamps overlap the tail of main by a 4.064-us
median, so a missing host launch is not the binding opportunity.  Softmax is
fused inside the main kernel and cannot be timed separately; the combine kernel
is the separately measurable normalization/reduction cost.

| NCU main-kernel metric | M16 | M32 |
|---|---:|---:|
| Duration | 22.336 us | 30.432 us |
| Grid / block | 148 CTAs / 384 threads | 148 CTAs / 384 threads |
| Registers / launch shared memory | 168 / 232,656 B | 168 / 232,656 B |
| Local spills | 0 | 0 |
| Active warps | 18.41% | 18.52% |
| Eligible warps/cycle | 0.215 | 0.291 |
| Tensor-pipe active, elapsed | 8.82% | 13.23% |
| DRAM reads / rate | 23,043,072 B / 1.032 TB/s | 46,012,928 B / 1.512 TB/s |
| L2 sector hit rate | 14.11% | 14.87% |
| Long-scoreboard / issue-active | 5.403 | 4.429 |
| Barrier / issue-active | 3.669 | 2.045 |
| Shared wavefronts / excessive | 987,080 / 262,144 | 1,709,512 / 393,216 |
| Shared load / store conflicts | 152,955 / 151,122 | 325,705 / 191,740 |

Shared memory and registers both cap the launch at one CTA per SM.  The scheduler
launches one 148-CTA wave, but only 128 partitions do useful work.  Sparse gather
has low L2 reuse and substantial DRAM traffic, yet neither HBM nor tensor pipes
are close to saturation; very low eligible-warps and high long-scoreboard,
barrier, and shared-bank-conflict evidence identify the fused pipeline stalls.

The scheduler attempt reduced combine DRAM reads from 16,818,176 to 14,717,184
bytes at M16 and to 12,615,936 bytes at M32.  It nevertheless lengthened the
dominant main kernel from 22.336 to 24.096 us and from 30.432 to 35.488 us,
respectively.  Less split parallelism trades a smaller reduction for a larger
main-kernel loss.

## Source experiment and binary identity

The source-backed attempt grouped the four V32 token rows consumed by each
eight-thread dequant group and inserted 16 bytes between 2,048-byte payload
slabs to rotate shared banks.  It was built in a private operator namespace so
stock and candidate could coexist:

- SGLang base `f93f8867b4bc124c9809c9110ec7361ed11b6b4a`;
- isolated namespace hook `5db482acd95020a3113ff4c1d65e8406d71f5a14`;
- FlashMLA base `05e26647fe840b8baedae486c2d86d5ce4efeb7c`;
- experiment `beeba02f55bc577786737c2d8addad204ebccf50`;
- CUTLASS gitlink `147f5673d0c1c3dcf66f78d677fd647e4a020219`;
- artifact `flashmla_goal02_ops.so`, 15,293,968 bytes, SHA-256
  `a978dd3f307a5d25dc90595b2b2cb832fdc726c1094df32a7179ac6e82e83560`;
- namespace `sgl_kernel_goal02`, init symbol `PyInit_flashmla_goal02_ops`.

The build passed static resource checks, but the first strict M16 invocation
faulted.  Memcheck localized thread 160 in the main kernel at `+0xa420`, an
`UTMALDG.2D.GATHER4`.  Cubin inspection shows destination offsets `0x24000`,
`0x24810`, `0x25020`, ...: the experimental 2,064-byte stride violates the
instruction's 128-byte shared-address alignment after the first group.  A
zero-pad correction removes the bank rotation while keeping added coordinate
work; a 128-byte pad needs 3,584 additional bytes, but the 232,144-byte plan is
already only 304 bytes below the 227-KiB source cap.  There is no credible
corrective variant to benchmark.  The parent FlashMLA commit is the rollback.
The full 130-error sanitizer log is preserved at
[`runs/memcheck1/compute-sanitizer.log`](../dsa_flashmla_kv_bank_conflict_20260722/runs/memcheck1/compute-sanitizer.log);
that diagnostic-only reproduction ran through the flexible wrapper on physical
GPU 0 (`GPU-30b619de-87f2-1862-0d07-a595da8fe417`) and is not used in a
cross-GPU performance comparison.

## Containing region, end to end, and fallback

Candidate containing-DSA-region and end-to-end decode tests were not advanced:
the scheduler candidate fails the leaf performance gate and the source candidate
fails leaf correctness.  Reporting a containing-region speedup for either would
be invalid.  A full stock distributed baseline was also unavailable: no GLM-5.2
model weights are present on this host, and the host exposes four B200s while
production acceptance is TP8/DP8/EP8.  The four-GPU lane is diagnostic only and
was not relabeled as acceptance.  These external limitations are recorded in
[`validation_blockers.json`](validation_blockers.json); they are not the reason
for declining the candidates, which already fail on one GPU.

The exact policy is:

| Bucket | Enabled implementation | Candidate state |
|---|---|---|
| M16 | Stock `sgl_kernel::fwd_kvcache_mla` | Disabled |
| M32 | Stock `sgl_kernel::fwd_kvcache_mla` | Disabled |
| Unsupported ABI/topology | Existing stock path | Fail closed; no candidate dispatch exists |

No installed package was overwritten.  No source change routes production to
`sgl_kernel_goal02`, and the namespace macros default to the original
`sgl_kernel` / `flashmla_ops` values.

## Evidence map

- Exact raw samples, graph/oracle JSON, NSYS/NCU reports, parsed metrics, and
  environment: [`screen3`](../dsa_flashmla_kv_scheduler_campaign_20260722/runs/screen3)
- Scheduler report and reproduction script:
  [`dsa_flashmla_kv_scheduler_campaign_20260722`](../dsa_flashmla_kv_scheduler_campaign_20260722)
- Isolated build, source patch, artifact identity, source review, and failed run:
  [`dsa_flashmla_kv_bank_conflict_20260722`](../dsa_flashmla_kv_bank_conflict_20260722)
- Complete structured attempt record: [`attempt_ledger.json`](attempt_ledger.json)

`audit_result.py` is not applicable: no frozen synthetic `result.json` is used
for this production disposition.  The evidence above comes from the explicitly
named serving-native workloads and is not compared with the unrelated TRT-LLM
DSA task.
