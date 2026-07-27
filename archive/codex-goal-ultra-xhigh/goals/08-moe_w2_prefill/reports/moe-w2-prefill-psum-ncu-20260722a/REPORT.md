# PSUM W2 Nsight Compute collection

This directory preserves the PSUM contiguous DeepGEMM `GemmType=5`
collection for the single-B200 production-ABI replay. The selected full-report
kernel duration is 313.312 us. It launches 148 blocks x 256 threads, uses 50
registers/thread and 214828 bytes shared memory/block, and has no local loads,
local stores, or spills.

The full report includes PM WarpStates and aggregate counters. The separate
source report includes positive-line source/SASS mapping. Their SHA256 values
are:

- `reports/psum_full.ncu-rep`:
  `8e2c271c1830d9d5d3636bd26434f64bedfdb8ca07358d42daccea5c17258932`
- `reports/psum_source.ncu-rep`:
  `5a906be089637442654518db2f567d4064bab9783d8c370462550f8b03c91fbd`

See `../moe-w2-prefill-psum-vs-stock-20260722b/REPORT.md` for comparison,
rule interpretation, PM limitations, and the production-evidence boundary.
