# Hardened same-GPU validation summary

Every candidate row passed the runner's pre-timing comparison and post-timing replay; fused prepare/store rows use a fresh deterministic seed.

| Series | Reference medians (ms) | Candidate medians (ms) | Paired median speedups | Stable >=1.03x |
|---|---|---|---|---|
| isolated_identity | 0.025216 | 0.025664 | 0.990699 | False |
| region_identity | 0.135984 | 0.135536 | 0.997625 | False |
| isolated_torch_mm | 0.025872, 0.025712, 0.025728 | 0.026560, 0.025760, 0.025936 | 0.984024, 0.995630, 0.996183 | False |
| region_torch_mm | 0.134496, 0.141232, 0.141008 | 0.134096, 0.145504, 0.142432 | 1.003540, 1.032630, 1.002945 | False |
| isolated_tgv | 0.031760, 0.033136, 0.033008 | 0.093568, 0.095632, 0.093184 | 0.329284, 0.322368, 0.361148 | False |
| region_tgv | 0.153504, 0.158336, 0.147920 | 0.279504, 0.284624, 0.268480 | 0.564382, 0.562612, 0.553539 | False |
| region_single_stream | 0.139216, 0.136032, 0.140256 | 0.139968, 0.136080, 0.143696 | 1.012753, 0.985414, 0.978400 | False |

Stable region candidates: none.
Nsys stock, torch-mm, and single-stream reports plus ABI traces were validated as non-empty, source-hashed artifacts.
