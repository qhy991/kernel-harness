# Baseline/profile bundle `20260723a` — failed preflight

The scheduler allocated physical GPU 0
(`GPU-30b619de-87f2-1862-0d07-a595da8fe417`). Environment, topology, import
identity, exact source hashes, and the line-info DeepGEMM compiler commands were
captured successfully.

The production-shaped call executed through the packed DeepGEMM kernel, then
the metadata serializer raised `TypeError` while inspecting the native Torch
operator packet. The failure occurred before all paired timing loops and before
Nsight Systems or Nsight Compute collection. Therefore this directory contains
no performance result and is never compared with another run.

The generated line-info JIT source and cubins are preserved under
`profile/attn-o-prefill-packed-stock-20260723a/`. The serializer was hardened
and the complete campaign moved to the separately named `20260723b` run.
