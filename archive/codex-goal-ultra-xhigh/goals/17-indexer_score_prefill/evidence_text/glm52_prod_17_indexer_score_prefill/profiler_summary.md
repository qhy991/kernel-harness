# Profiler summary

## Collection contract

Campaign `20260723T134307Z` ran every alternating series, Nsight Systems
capture, and Nsight Compute report in one `with_flexible_gpu.sh` lease on
logical GPU 0 / physical
`GPU-30b619de-87f2-1862-0d07-a595da8fe417`. All 72 recorded steps exited 0.
The environment was B200/SM100 with 148 SMs, CUDA 13.2, Nsight Compute
2026.1.1, Nsight Systems 2025.6.3, Torch 2.11.0+cu130, DeepGEMM 0.1.4,
SGL-Kernel 0.4.4, FlashInfer 0.6.12, and SGLang 0.5.15 at
`f93f8867b4bc124c9809c9110ec7361ed11b6b4a`.

Raw reports and hashes are under
`profile/indexer-score-prefill-20260723T134307Z/` and the campaign directory.
The installed DeepGEMM cubin has no line information, so NCU could not import
source lines. Full metrics, source-counter samples, and SASS views were still
captured; no source-line attribution is claimed.

## Nsight Systems: path, launch count, and dilution

Each unchunked main-score invocation launches one K gather, one MQA-logits
kernel, and one PAGED top-k kernel. Its captured kernel time was:

| Kernel | Instances | total µs | score-kernel share |
|---|---:|---:|---:|
| PAGED top-k | 1 | 175.872 | 58.8% |
| DeepGEMM MQA logits | 1 | 111.840 | 37.4% |
| K+scale gather | 1 | 5.632 | 1.9% |

The c256 distribution takes two MQA and two top-k launches because the
4-GiB dense logits matrix exceeds the 3,063,848,894-byte budget. Equalizing
stock `2921+1175` to `2048+2048` did not reduce GPU work:

| c256 scope | stock total kernel µs | balanced total kernel µs |
|---|---:|---:|
| score/top-k | 452.192 | 453.151 |
| complete indexer | 542.400 | 545.792 |
| graph-split indexer | 540.736 | 545.151 |
| complete indexer + TRT-LLM DSA | 1,384.639 | 1,387.007 |

In the DSA scope the selected TRT-LLM attention kernel alone takes
842.047 µs (60.8% of captured kernel time); top-k is 200.640 µs (14.5%),
MQA logits 133.216 µs (9.6%), and gather 13.856 µs (1.0%). This is the
measured containing-region dilution, not an estimate.

The fixed mixed-context distribution behaves differently because rows have
nonuniform `ks/ke`. Balancing two chunks cuts MQA time from 393.088 to
309.664 µs and top-k from 345.120 to 331.903 µs; total captured score kernel
time falls from 957.312 to 862.559 µs. That explains the pooled score-only
1.04642x result and motivates the separate mixed containing-region
confirmation. It does not by itself authorize production promotion.
The individual MQA launches change from 212.640+180.448 µs at
`3169+927` rows to 88.448+221.216 µs at `2048+2048`; the gain is a
shape/scheduling effect for nonuniform query ranges, not fewer launches or a
different kernel.

## Focused mixed-context confirmation

Campaign `20260723T142005Z-mixed-confirmation` put all twelve alternating
score/complete/graph/DSA series, both fallback controls, and eight Nsys
captures in one wrapper invocation on the same physical GPU UUID as the
first campaign. Every recorded command exited 0 and both checksum manifests
verify.

The exact host-metadata predicate retained a score-only signal, but that
signal did not pass any required containing-region gate:

| Mixed-context scope | pooled paired p50 | correctness | 3% gate | no series regression |
|---|---:|---|---|---|
| score/top-k | 1.03291x | PASS | PASS | PASS |
| complete indexer | 1.00395x | PASS | FAIL | FAIL |
| exact PCG/BCG split region | 0.99829x | PASS | FAIL | FAIL |
| complete indexer + TRT-LLM DSA | 1.00773x | PASS | FAIL | FAIL |

The paired score series were 1.05924x, 1.00547x, and 1.03532x. That
distribution is why the pooled result is reported instead of selecting only
the favorable runs. The exact-signature fallback controls for the main and
c256 rectangles were 0.97495x and 1.01957x and both missed the 3% gate.

Nsys independently reproduced the chunk-scheduling effect, while also
showing its dilution. These are captured kernel sums from one instrumented
invocation per variant, not substitutes for the alternating latency gate:

| Mixed-context scope | stock kernel µs | balanced kernel µs |
|---|---:|---:|
| score/top-k | 958.559 | 863.100 |
| complete indexer | 1,046.524 | 952.379 |
| exact graph-split indexer | 1,046.845 | 947.773 |
| complete indexer + TRT-LLM DSA | 1,891.066 | 1,794.589 |

The selected TRT-LLM attention kernel itself was unchanged at 840.670 versus
840.254 µs. The candidate therefore changes score scheduling only; it does
not reduce producer, launch, synchronization, or attention-consumer work.

## Nsight Compute: resource and traffic evidence

The reached `sm100_mqa_logits` launch uses all 148 SMs, 384 threads per
block, 168 registers per thread, 221,696 bytes of shared memory per block,
one wave/SM, and no measured local-memory loads or stores. Selected full
reports:

| MQA shape | duration µs | SM throughput | tensor-pipe active | DRAM read | DRAM write |
|---|---:|---:|---:|---:|---:|
| main 4096-row | 114.016 | 66.10% | 53.94% | 26.01 MB | 75.44 MB |
| c256 stock head 2921 | 84.512 | 64.48% | 52.62% | 37.26 MB | 41.44 MB |
| c256 stock tail 1175 | 53.248 | 42.52% | 34.70% | 15.81 MB | 2.22 MB |
| c256 balanced 2048 | 68.992 | 55.27% | 45.11% | 25.99 MB | 15.80 MB |

Two balanced MQA launches therefore take essentially the same device time as
the stock c256 head plus tail. The library kernel is resource-limited to one
block/SM by both registers and shared memory, but has no spills. A speculative
DeepGEMM fork has no evidenced whole-region 3% opportunity in this fixed
rectangle.

The c256 PAGED top-k head takes 144.800 µs at 59.14% SM throughput and reads
190.32 MB from DRAM. A balanced 2048-row top-k takes 103.840 µs, so two such
launches again do not improve the c256 total. The main K gather takes only
8.512 µs in NCU, reads 8.66 MB at 1.018 TB/s, and accounts for under 2% of
the unchunked Nsys score range; launch-shape tuning cannot deliver a 3%
complete-region result.

## Binding conclusion

The score path is not one uniformly rewritable kernel. The unchunked bucket
is dominated by top-k and already uses one all-SM MQA launch. The rectangular
chunked bucket preserves nearly identical total MQA/top-k work under equal
chunks and becomes strongly diluted by producer and DSA kernels. Only the
nonuniform mixed-context schedule shows score-level load-balancing headroom.
Its complete, graph-split, and DSA regions all failed the promotion threshold,
so the binding result is no replacement rather than an isolated score-only
dispatch.
