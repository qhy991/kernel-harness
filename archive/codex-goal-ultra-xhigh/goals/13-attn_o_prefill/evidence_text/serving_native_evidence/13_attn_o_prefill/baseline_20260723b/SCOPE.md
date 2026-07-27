# Preliminary PDL-off baseline/profile

This bundle proved the exact packed int32 UE8M0 call chain and produced a
complete Nsys/NCU device-code profile on physical GPU 1
(`GPU-5b9be10b-5bfc-b658-9b31-f7ae8516dc54`).

It did not call SGLang worker startup's `update_deep_gemm_config()`.
DeepGEMM therefore retained its library default `pdl=false`, whereas the
current SGLang production default is `SGLANG_DEEPGEMM_PDL=1`.  The kernel
body evidence remains useful, but these timings are not the primary
production denominator.  The fresh production bundle explicitly sets and
records PDL before all paired timing and profiling.
