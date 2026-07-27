# Complete indexer and selected-DSA paired summary

Every series uses 60 same-process, alternating A/B pairs under bidirectional CUDA-graph capture.

| Workload | Candidate | Series speedups | Median | Min | 3% in every series | Correct |
|---|---|---|---:|---:|---|---|
| indexer_complete_decode_m16 | cutedsl | 1.014806x, 1.031918x, 1.056648x | 1.031918x | 1.014806x | False | True |
| indexer_complete_decode_m16 | identity | 0.998914x, 1.006760x, 0.990560x | 0.998914x | 0.990560x | False | True |
| indexer_complete_decode_m32 | cutedsl | 1.028806x, 1.032633x, 1.015328x | 1.028806x | 1.015328x | False | True |
| indexer_complete_decode_m32 | identity | 1.035866x, 0.995117x, 0.997602x | 0.997602x | 0.995117x | False | True |
| indexer_dsa_decode_m16 | cutedsl | 1.011807x, 1.014385x, 1.011058x | 1.011807x | 1.011058x | False | True |
| indexer_dsa_decode_m16 | identity | 0.992189x, 1.001997x, 0.990996x | 0.992189x | 0.990996x | False | True |
| indexer_dsa_decode_m32 | cutedsl | 1.021669x, 1.019186x, 1.020334x | 1.020334x | 1.019186x | False | True |
| indexer_dsa_decode_m32 | identity | 0.996038x, 0.990988x, 0.981250x | 0.990988x | 0.981250x | False | True |
