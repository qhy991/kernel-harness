# Paired campaign summary

| Workload | Series | samples/side | chunks | ref p50 ms | cand p50 ms | paired p50 speedup | paired p10–p90 | correct | >=3% | no series regression |
|---|---:|---:|---|---:|---:|---:|---:|---|---|---|
| indexer_complete_prefill_m4096_mixed | 3 | 72 | 3169+927 -> max 2048 | 2.303024 | 2.283984 | 1.00395x | 0.96580–1.07069x | PASS | FAIL | FAIL |
| indexer_dsa_prefill_m4096_mixed | 3 | 72 | 3169+927 -> max 2048 | 3.172352 | 3.155328 | 1.00773x | 0.96928–1.04440x | PASS | FAIL | FAIL |
| indexer_graph_split_prefill_m4096_mixed | 3 | 72 | 3169+927 -> max 2048 | 2.249776 | 2.238976 | 0.99829x | 0.96807–1.03785x | PASS | FAIL | FAIL |
| indexer_score_prefill_m4096 | 1 | 16 | 4096 | 0.420176 | 0.427216 | 0.97495x | 0.89421–1.06486x | PASS | FAIL | FAIL |
| indexer_score_prefill_m4096_c256 | 1 | 16 | 2921+1175 -> max 2048 | 1.074176 | 1.081808 | 1.01957x | 0.83819–1.08110x | PASS | FAIL | PASS |
| indexer_score_prefill_m4096_mixed | 3 | 120 | 3169+927 -> max 2048 | 1.476160 | 1.419584 | 1.03291x | 0.98064–1.09042x | PASS | PASS | PASS |
