# Complete indexer and selected-DSA paired summary

Every series uses 40 same-process, alternating A/B pairs under bidirectional CUDA-graph capture.

| Workload | Candidate | Series speedups | Median | Min | 3% in every series | Correct |
|---|---|---|---:|---:|---|---|
| tp4_indexer_dsa_decode_m16 | cutedsl | 1.004793x, 1.002506x, 1.018696x | 1.004793x | 1.002506x | False | True |
| tp4_indexer_dsa_decode_m16 | identity | 0.994493x, 0.980592x, 0.991134x | 0.991134x | 0.980592x | False | True |
| tp4_indexer_dsa_decode_m32 | cutedsl | 1.023810x, 1.030381x, 1.017586x | 1.023810x | 1.017586x | False | True |
| tp4_indexer_dsa_decode_m32 | identity | 1.017577x, 0.990450x, 1.014213x | 1.014213x | 0.990450x | False | True |
