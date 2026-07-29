# Nsight Compute report: GLM-5.2 W13 BM32 two-SM

## Scope and conclusion

This profile compares exact same-source DeepGEMM stock against the promoted
BM32 two-SM candidate for the E32/slab1024/K6144/N4096 packed-int32 W13
workload at decode M16/expected-M4. Both arms ran with PDL enabled,
`num_sms=148`, `tc_util=100`, the same tensors and stream, and separately
owned, pre-warmed JIT caches. The four full/source captures were collected in
one scheduler-wrapper invocation on physical B200 GPU 0.

The candidate reduces profiler-replay duration from 136.58 us to 128.32 us
(1.0644x), primarily by reducing the BM128 tail/epilogue footprint. This
explains the leaf improvement, but it does not override the fair performance
gate: the required graph-containing region later produced a 1.028125x
per-series BA estimate and therefore ended the campaign as `no-replacement`.

The reports answer the bounded profiling question. They do not justify a CLC
scheduler rewrite or another tuning round.

## Capture identity

| Field | Stock | BM32 two-SM |
|---|---:|---:|
| DeepGEMM config | BM128/BN128/BK128, 8 stages, cluster-N 2 | BM32/BN128/BK128, 11 stages, cluster-N 2 |
| Kernel | `sm100_fp8_fp4_gemm_1d1d_impl` | same symbol, candidate specialization |
| Grid / block | 148 x 1 x 1 / 256 x 1 x 1 | same |
| Cluster policy | 2-CTA, `PolicySpread` | same |
| Waves per SM | 1 | 1 |
| Registers/thread | 36 | 36 |
| Dynamic shared memory | 213.80 KiB | 223.02 KiB |
| Static shared / stack / local | 1024 B / 0 / 0 | 1024 B / 0 / 0 |
| Theoretical occupancy | 12.5% | 12.5% |
| Achieved occupancy | 12.50% | 12.55% |
| Spills | 0 | 0 |

The exact cubin hashes are:

- stock: `92d200dfaf0d9a25c651905be06ae991bbd88a3b8c6f18b0aa8594064a5432b7`
- BM32 two-SM:
  `c9556f1194e15c2539185d1d0e0f05f63c08225d4b0af9be8b61b894812d0ea1`
- BM32 one-SM comparison:
  `11a3daf1a42d2f6dd8c36db1b335483fa68c7edf0c4751d3eefd4abeb9f2178e`

All generated-source, cubin, PTX and SASS identities are frozen in
[`analysis/codegen_sha256.txt`](analysis/codegen_sha256.txt).

## Six-dimension analysis

### 1. Compute and tensor-core activity

Executed instructions fall from about 6.690 million to 5.785 million. NCU's
tensor-cycle percentage also falls, but that percentage is not interpreted as
reduced useful MMA work: the kernel uses tcgen05/TMEM, the elapsed-time
denominator changes, and static code generation shows the same 16 cooperative
`UTCQMMA.2CTA` instructions in the promoted specialization. The defensible
claim is reduced control/epilogue work, not fewer mathematical outputs.

### 2. Global-memory behavior

| Metric | Stock | BM32 two-SM |
|---|---:|---:|
| Compute-memory throughput | 83.27% | 84.21% |
| DRAM bytes read | 840.8 MB | 818.7 MB |
| DRAM bytes written | 31.45 MB | 10.08 MB |
| L1 sector hit rate | 98.22% | 98.22% |
| L2 sector hit rate | 36.13% | 28.06% |

The 68% reduction in DRAM writes is consistent with BM32 producing fewer
padded output tiles. The generic NCU global-load rule estimates only
4 bytes/sector and low utilization in both arms (25.14% stock, 18.59%
candidate), but this rule does not account for the TMA path and is diagnostic,
not a correctness or promotion gate.

### 3. Shared memory and TMEM

The candidate pays 9.22 KiB more dynamic shared memory and remains limited to
one block per SM. Shared-load conflict diagnostics worsen from about 1.2-way
(4.13%) to 1.7-way (9.04%), while shared-store conflict severity improves from
about 20.4-way (8.67%) to 9.3-way (6.61%). There are no local-memory spills.
The smaller output-tile footprint outweighs these mixed shared-memory changes
for the isolated kernel.

### 4. Occupancy and scheduler issue

Both arms intentionally occupy one CTA/SM and eight active warps/SM. Eligible
warps average 0.0502 versus 0.0472 per cycle; roughly 95% of cycles have no
eligible warp. Long-scoreboard stall ratio is 25.52 versus 27.87 cycles per
issue interval, and barrier ratio is 8.223 versus 8.324. These ratios do not
show a new candidate-only occupancy or spill pathology; the shorter candidate
executes less total work despite slightly worse normalized latency-hiding
ratios.

### 5. Synchronization and PC sampling

The dominant sampled PCs are the persistent wait loop and cooperative cluster
barrier:

| Hot PC | Stock samples | Candidate samples |
|---|---:|---:|
| long-scoreboard `NANOSLEEP.SYNCS` | 3,649 | 2,908 |
| barrier `UCGABAR_WAIT` | 1,677 | 1,544 |

See
[`analysis/sass_hotspots_stock.txt`](analysis/sass_hotspots_stock.txt) and
[`analysis/sass_hotspots_candidate.txt`](analysis/sass_hotspots_candidate.txt).
The source reports do not contain usable line mappings because the exact JIT
cubins lack `.debug_line`; PC-to-SASS attribution remains available and is the
honest resolution of the capture.

### 6. Launch geometry and tail behavior

Both variants launch 148 persistent CTAs on 148 SMs in one measured wave.
There is no current last-wave underfill evidence. A CLC rewrite would therefore
target a historical seven-wave observation rather than this exact compiled
launch and is excluded. Cluster synchronization remains required for both
stock and the promoted two-SM variant.

## One-SM versus two-SM code-generation proof

The separately generated BM32 one-SM cubin uses 33 registers/thread, zero
stack/local memory, plain one-CTA `UTCQMMA`, and PTX
`tcgen05.mma.cta_group::1`. It contains no `UTCQMMA.2CTA` or `UCGABAR`
instructions. The stock and BM32 two-SM cubins use 36 registers/thread,
`UTCQMMA.2CTA`, `tcgen05.mma.cta_group::2`, cluster multicast commits and
`UCGABAR` synchronization. Thus the one-SM label is proven by generated code,
not inferred from a selector name.

The genuine one-SM candidate was correct but failed the first fair eager leaf
gate (minimum required estimate 1.026873x), so it was not promoted to NCU.

## Artifacts and reproducibility

- Gzip-compressed full metric reports:
  `reports/full_{stock,candidate}.ncu-rep.gz`
- Gzip-compressed source/PC sampling reports:
  `reports/source_{stock,candidate}.ncu-rep.gz`
- Raw exports:
  [`analysis/raw_stock.csv`](analysis/raw_stock.csv) and
  [`analysis/raw_candidate.csv`](analysis/raw_candidate.csv)
- Machine-readable metric exports:
  `analysis/metrics_{all,key}_{stock,candidate}.json`
- Collection logs, including the retained failed first NVTX-filter attempt:
  `analysis/collect_*.log`
- Exact harness inputs and runtime state:
  `harness/{sanity,ncu,ncu_source}_{stock,candidate}.json`
- Analysis scripts:
  `analysis/analyze_reports.py`, `analysis/extract_stall_hotspots.py`,
  `analysis/extract_sass_hotspots.py`, and `analysis/plot_timeline.py`

Plain PTX plus NCU detail/timeline exports are retained byte-for-byte,
including tool-emitted trailing padding. They are excluded from whitespace
lint; source, tests, reports and hand-written evidence pass the check.

Compressed report SHA256 values are:

- full stock:
  `c5a5662134784f39c105a0c3acc1ca66801445485a2e9eeea4b40fcabc895f4a`
- full candidate:
  `53f0b984867ff00fc3de6bf909e1244825e32c714645c65af716ca8360873c9c`
- source stock:
  `27a870e3f549086d9154c64492db44530ba3c838886c537515a7b0811f328e38`
- source candidate:
  `f232cfb0b174c04597999b50097209b8c3cfbd1a684dc41eb54a35bbdee2a068`

The NCU replay timing is explanatory only. The unprofiled alternating AB/BA
result JSON files remain the performance authority. To reopen a report, first
run `gzip -dk <report>.ncu-rep.gz`, then pass the resulting `.ncu-rep` to NCU.
