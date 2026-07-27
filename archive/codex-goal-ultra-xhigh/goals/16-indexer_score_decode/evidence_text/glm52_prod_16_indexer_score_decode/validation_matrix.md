# Validation matrix

| Requirement | Status | Evidence |
|---|---|---|
| Backend resolution | pass | `reachability.md`; AUTO resolves DeepGEMM on CUDA |
| Exact M16/M32 score ABI | pass | `backend_validation.json`; run contracts |
| Normal decode split/native mode | pass | `next_n=1`, split wrapper recorded |
| Page/cache/metadata layout | pass | page 64, fused uint8 K, compact table, `[149,2]` schedule |
| Masking and `clean_logits` | pass | no masks, `clean_logits=False` |
| Top-k handoff | pass | top-k-v2, exact 2048-element physical set |
| CUDA graph replay | pass | bidirectional reference and candidate captures |
| Score correctness | pass | max absolute difference `2.384185791015625e-7` |
| Three paired score series | pass collection, no winning bucket | corrected campaign |
| Nsys score/top-k profile | pass | four raw reports and exported tables |
| NCU score/top-k profile | pass | six full plus six source reports |
| Complete indexer correctness/latency | pass collection, no winning bucket | 24 paired results |
| Selected TRT-LLM DSA correctness/latency | pass collection, no winning bucket | 12 candidate/control results plus Nsys |
| Four-rank diagnostic | pass, diagnostic only | 12 rank-max paired results |
| Full SGLang real-checkpoint decode | blocked | checkpoint/tokenizer absent |
| TP8/DP8/EP8 acceptance | blocked | host has four GPUs |
| Fallback | pass | no SGLang diff; no candidate bucket enabled |

All decision-bearing performance collections were executed through the
required wrapper. The score and containing-region campaigns each held one
physical B200 for their complete alternating series and profiler collection.
The four-rank diagnostic held the all-GPU lock for its complete series.

Serving-native JSON does not use the frozen task `result.json` schema, so
`audit_result.py` does not apply. Integrity is instead established by exact
candidate and runner hashes, pre/post correctness, raw sample persistence,
repository/status capture, and verified SHA-256 artifact manifests.

The final configured-environment check output is persisted in
`final_checks.json`. The verifier returns `ok=true`; its non-strict pointer
audit reports the expected advisory that `runs/index.jsonl` is absent for this
serving-native-only campaign.
