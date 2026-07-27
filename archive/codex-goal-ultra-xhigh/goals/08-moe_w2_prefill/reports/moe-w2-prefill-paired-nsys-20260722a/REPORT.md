# Paired Nsight Systems collection

This one-B200 trace contains exactly five stock (`GemmType=1`) and five PSUM
(`GemmType=5`) W2 launches: correctness, three warmups, and one timed pair.
Stock/PSUM kernel medians are 332.159/311.967 us (1.064725x); means are
332.2356/312.2548 us (1.063989x). The only timed pair is
330.974/314.143 us (1.053578x).

Do not use the profiled runner JSON's 1.257373x CUDA-event ratio. Nsys shows
that PSUM host-side launch preparation overlaps the preceding untimed
output-poison fill before the start marker completes, whereas stock launch
preparation occurs after its start marker. The completion-marker spans are
399.134 us stock and 322.239 us PSUM,
but only 330.974 and 314.143 us are the selected kernels. The unprofiled
cache-prime ratio is 1.068271x and agrees with the locked component result.

Raw reports are `reports/paired.nsys-rep` (SHA256
`161f331fc268315fa36434c5f8a5cded5befabd1acf7fe21a04df2b014567a0c`)
and `reports/paired.sqlite` (SHA256
`cfe74bba11a463dea3c11d155508f8c79dc896e15915d904c9e76cb7fa4c7004`).
The CSV exports, exact launch count, JIT cache provenance, and runner results
are preserved under `analysis/` and `harness/`.

See `../moe-w2-prefill-psum-vs-stock-20260722b/REPORT.md` for the complete
six-dimension profile analysis and limitations.
