# Stock W2 Nsight Compute collection

This directory preserves the stock row-wise contiguous DeepGEMM
`GemmType=1` collection for the single-B200 production-ABI replay. The selected
full-report kernel duration is 331.296 us. It launches 148 blocks x 256 threads,
uses 38 registers/thread and 214828 bytes shared memory/block, and has no local
loads, local stores, or spills.

The full report includes PM WarpStates and aggregate counters. The separate
source report includes positive-line source/SASS mapping. Their SHA256 values
are:

- `reports/stock_full.ncu-rep`:
  `b24c336323e0fefb53750e08aac581194b2eb00514d7ae8721cbdd3c820ae639`
- `reports/stock_source.ncu-rep`:
  `75b9a1cf62a5073aa254a37259d2b317bb42f4bbb72665c373642b50d3796c87`

See `../moe-w2-prefill-psum-vs-stock-20260722b/REPORT.md` for comparison,
rule interpretation, PM limitations, and the production-evidence boundary.
