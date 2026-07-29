# GLM-5.2 FlashMLA sparse-decode hotspot campaign

This directory is the retained source and evidence for the exact SGLang
`flashmla_kv` FP8 sparse-index decode ABI at local M16 and M32. The terminal
disposition is **no-replacement**: all candidates were correct, but none passed
the required four-lane, every-series, every-estimator `1.03x` gate.

The authoritative summaries are:

- [`FINAL_REPORT.md`](FINAL_REPORT.md)
- [`evidence/timing_gate_audit.json`](evidence/timing_gate_audit.json)
- [`evidence/attempt_ledger.json`](evidence/attempt_ledger.json)
- [`evidence/binary_manifest.json`](evidence/binary_manifest.json)
- [`evidence/preflight.json`](evidence/preflight.json)
- [`EXTERNAL_ACCEPTANCE.md`](EXTERNAL_ACCEPTANCE.md)
- [`knowledge_entry_draft.json`](knowledge_entry_draft.json), installed
  append-only under `testbench/knowledge/entries/`

Raw correctness, paired timing, Nsys, disassembly, and control artifacts are
under [`evidence/`](evidence/). Scripts under [`harness/`](harness/) refuse to
overwrite evidence and require the shared GPU lease for CUDA work.

The provider is
[`serving_native/candidates/flashmla_sparse_decode_provider.py`](../../serving_native/candidates/flashmla_sparse_decode_provider.py).
It requires an explicit `GLM52_FLASHMLA_VARIANT`; identity is marked as a
control and is never presented as a candidate. No variant is enabled by
default in SGLang.
