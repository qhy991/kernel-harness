# Superseded initial fused-region campaign

The root-level region JSON files, the first two Nsight Systems pairs, the
K-before-Q schedule trial, and the root-level post-revert smoke were collected
before the fixed checkpoint ABI was audited against the pinned model files.
They are preserved verbatim for provenance, but they are **not production
performance evidence** for `nvidia/GLM-5.2-NVFP4`.

The original reconstruction made two incorrect assumptions:

- it bound `indexer.wq_b` to an FP8/packed-UE8M0 path; the checkpoint stores
  BF16 `[4096,2048]` and SGLang resolves the fixed recipe to
  `UnquantizedLinearMethod` because `self_attn` is in the ModelOpt ignore list;
- it used RoPE max-position/base `4096/10000`; the pinned model configuration
  uses `1048576/8000000`, default scaling, and interleaved layout.

Consequently, the old fused prepare/store subregion trace has an extra activation-quant kernel
and an FP8 wq GEMM, so its prepare/store-subregion ratios, host gaps, stream critical path,
and K-before-Q result cannot be transferred to the fixed model. The existing
raw files are deliberately not edited or relabeled.

The root-level `best_backend.txt`, `best_selection.json`, `select_best.py`, and
`knowledge_entry_draft.json` also belong to this superseded exploration. In
particular, `best_selection.json` merely selected the least-slow backend from
that sweep; it is not an enablement decision. The draft's old launch-gap
diagnosis is withdrawn by append-only entry `20260722c` and must not be cited.

The following narrow evidence remains valid because it does not construct
`wq_b` or RoPE:

- isolated BF16 `wk_weights_proj` M4096/N160/K6144 benchmark rows;
- isolated wk kernel Nsight Compute reports and their device-resource data;
- checkpoint-loader CPU checks for the fused BF16 wk/weights parameter.

The authoritative fixed-model contract is
`fixed_model_contract_cpu.json`. Corrected region evidence lives under
`exact_bf16_wq/` and uses one new serialized GPU campaign. Final reports and
the superseding append-only knowledge entry cite only that corrected campaign
for region-level conclusions.

The older reconstruction trace JSON used the label "one rank of balanced
TP8/DP8/EP8" as a shape mapping. It actually executed at world size 1. Current
profile output records execution topology and TP8-derived shape mapping as two
separate fields.
