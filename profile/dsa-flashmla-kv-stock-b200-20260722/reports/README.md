# Raw profiler reports

The GPU-locked collection commands write `.nsys-rep` and `.ncu-rep` artifacts
here. A report filename always includes the fixed bucket (`m16` or `m32`) and
the profiled kernel (`main` or `combine`).

The two `build_*_tensor.log` files are the raw final ptxas/nvcc logs for the
pinned stock-source rebuild control and the rejected bound-32 candidate. Their
artifact hashes and exact commands are recorded in the corresponding manifests
under `../analysis/`.
