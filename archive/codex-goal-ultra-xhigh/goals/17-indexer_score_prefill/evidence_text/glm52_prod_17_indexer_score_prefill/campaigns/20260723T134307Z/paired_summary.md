# Paired campaign summary

| Workload | Series | samples/side | chunks | ref p50 ms | cand p50 ms | paired p50 speedup | paired p10–p90 | correct | >=3% | no series regression |
|---|---:|---:|---|---:|---:|---:|---:|---|---|---|
| indexer_complete_prefill_m4096 | 3 | 72 | 4096 | 1.086448 | 1.115360 | 0.99642x | 0.89599–1.05126x | PASS | FAIL | FAIL |
| indexer_complete_prefill_m4096_c256 | 3 | 72 | 2921+1175 -> max 2048 | 1.826160 | 1.778992 | 0.99571x | 0.94395–1.06919x | PASS | FAIL | FAIL |
| indexer_dsa_prefill_m4096 | 3 | 72 | 4096 | 2.060272 | 2.069456 | 1.00054x | 0.96309–1.02460x | PASS | FAIL | FAIL |
| indexer_dsa_prefill_m4096_c256 | 3 | 72 | 2921+1175 -> max 2048 | 2.840528 | 2.812816 | 1.00693x | 0.93866–1.08265x | PASS | FAIL | PASS |
| indexer_graph_split_prefill_m4096 | 3 | 72 | 4096 | 1.088832 | 1.101664 | 1.00638x | 0.91331–1.05356x | PASS | FAIL | FAIL |
| indexer_graph_split_prefill_m4096_c256 | 3 | 72 | 2921+1175 -> max 2048 | 1.886144 | 1.891200 | 0.99887x | 0.92118–1.04414x | PASS | FAIL | FAIL |
| indexer_score_prefill_m4096 | 3 | 120 | 4096 | 0.435536 | 0.440736 | 0.99666x | 0.91198–1.06632x | PASS | FAIL | FAIL |
| indexer_score_prefill_m4096_c256 | 3 | 120 | 2921+1175 -> max 2048 | 1.056240 | 1.080960 | 0.96917x | 0.87955–1.06893x | PASS | FAIL | FAIL |
| indexer_score_prefill_m4096_mixed | 3 | 120 | 3169+927 -> max 2048 | 1.477264 | 1.409616 | 1.04642x | 0.98234–1.10314x | PASS | PASS | PASS |
