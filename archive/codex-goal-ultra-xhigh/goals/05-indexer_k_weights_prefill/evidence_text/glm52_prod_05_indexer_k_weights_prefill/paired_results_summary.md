# Paired result summary

`candidate.speedup` is the runner-recorded median of interleaved per-pair `reference_ms / candidate_ms` ratios. The latency medians below are marginal medians and are not divided to manufacture a speedup. Reference-only baseline files are descriptive only. Results from different campaign GPUs are never combined into a comparison. The `immutable_hardened_campaign` is authoritative. The three `superseded_fp8_wq_*` campaigns used a non-production FP8 wq_b and generic RoPE in their prepare/store-subregion rows; only their isolated BF16 projection rows remain valid.

## Campaign provenance

| Campaign | Physical GPU | UUID | Results | Missing results | Missing profiler artifacts |
|---|---:|---|---:|---:|---:|
| immutable_hardened_campaign | 0 | GPU-30b619de-87f2-1862-0d07-a595da8fe417 | 23/23 | 0 | 0 |
| superseded_fp8_wq_backend_campaign | 2 | GPU-df8b1d78-b06c-39a2-54f0-66b9fabf3a99 | 20/20 | 0 | 0 |
| superseded_fp8_wq_schedule_ncu_campaign | 1 | GPU-5b9be10b-5bfc-b658-9b31-f7ae8516dc54 | 9/9 | 0 | 0 |
| superseded_fp8_wq_post_revert_smoke | 0 | GPU-30b619de-87f2-1862-0d07-a595da8fe417 | 2/2 | 0 | 0 |
| exact_bf16_wq_campaign | 1 | GPU-5b9be10b-5bfc-b658-9b31-f7ae8516dc54 | 17/17 | 0 | 0 |
| exact_single_stream_campaign | 2 | GPU-df8b1d78-b06c-39a2-54f0-66b9fabf3a99 | 3/3 | 0 | 0 |

## Recorded series

| Campaign / GPU | Evidence scope | Series | Candidate | Runs | Reference marginal medians (ms) | Candidate marginal medians (ms) | Paired median speedups | Across-run median | All runs pass 1.03x |
|---|---|---|---|---:|---|---|---|---:|---:|
| exact_bf16_wq_campaign / 1 | fixed_model_bf16_wq_exact_config | isolated_identity | reference | 1 | 0.025648 | 0.026016 | 0.982726 | 0.982726 | no |
| exact_bf16_wq_campaign / 1 | fixed_model_bf16_wq_exact_config | isolated_stock | stock_reference_only | 3 | 0.024976, 0.027856, 0.025040 | - | - | - | - |
| exact_bf16_wq_campaign / 1 | fixed_model_bf16_wq_exact_config | isolated_torch_mm | torch_mm_direct | 3 | 0.029536, 0.030864, 0.031360 | 0.029680, 0.030896, 0.032160 | 0.991200, 0.981217, 0.997276 | 0.991200 | no |
| exact_bf16_wq_campaign / 1 | fixed_model_bf16_wq_exact_config | region_identity | reference | 1 | 0.132608 | 0.133536 | 0.999768 | 0.999768 | no |
| exact_bf16_wq_campaign / 1 | fixed_model_bf16_wq_exact_config | region_stock | stock_reference_only | 3 | 0.135856, 0.140880, 0.137120 | - | - | - | - |
| exact_bf16_wq_campaign / 1 | fixed_model_bf16_wq_exact_config | region_tgv | sglang_cutedsl_tgv_direct | 3 | 0.200160, 0.149440, 0.329024 | 0.353056, 0.260080, 0.640032 | 0.568669, 0.567336, 0.517108 | 0.567336 | no |
| exact_bf16_wq_campaign / 1 | fixed_model_bf16_wq_exact_config | region_torch_mm | torch_mm_direct | 3 | 0.162368, 0.173568, 0.135040 | 0.154448, 0.176432, 0.135472 | 1.009863, 0.980541, 0.997941 | 0.997941 | no |
| exact_single_stream_campaign / 2 | fixed_model_preliminary_single_stream_with_linear_adapter | region_single_stream | stock_bf16_single_stream_schedule | 3 | 0.143696, 0.378416, 0.140160 | 0.148112, 0.262608, 0.148256 | 0.974558, 1.201739, 0.971418 | 0.974558 | no |
| immutable_hardened_campaign / 0 | authoritative_fixed_model_immutable | isolated_identity | reference | 1 | 0.025216 | 0.025664 | 0.990699 | 0.990699 | no |
| immutable_hardened_campaign / 0 | authoritative_fixed_model_immutable | isolated_stock | stock_reference_only | 3 | 0.031296, 0.026320, 0.025888 | - | - | - | - |
| immutable_hardened_campaign / 0 | authoritative_fixed_model_immutable | isolated_tgv | sglang_cutedsl_tgv_direct | 3 | 0.031760, 0.033136, 0.033008 | 0.093568, 0.095632, 0.093184 | 0.329284, 0.322368, 0.361148 | 0.329284 | no |
| immutable_hardened_campaign / 0 | authoritative_fixed_model_immutable | isolated_torch_mm | torch_mm_direct | 3 | 0.025872, 0.025712, 0.025728 | 0.026560, 0.025760, 0.025936 | 0.984024, 0.995630, 0.996183 | 0.995630 | no |
| immutable_hardened_campaign / 0 | authoritative_fixed_model_immutable | region_identity | reference | 1 | 0.135984 | 0.135536 | 0.997625 | 0.997625 | no |
| immutable_hardened_campaign / 0 | authoritative_fixed_model_immutable | region_single_stream | stock_bf16_single_stream_schedule | 3 | 0.139216, 0.136032, 0.140256 | 0.139968, 0.136080, 0.143696 | 1.012753, 0.985414, 0.978400 | 0.985414 | no |
| immutable_hardened_campaign / 0 | authoritative_fixed_model_immutable | region_stock | stock_reference_only | 3 | 0.144048, 0.143616, 0.175168 | - | - | - | - |
| immutable_hardened_campaign / 0 | authoritative_fixed_model_immutable | region_tgv | sglang_cutedsl_tgv_direct | 3 | 0.153504, 0.158336, 0.147920 | 0.279504, 0.284624, 0.268480 | 0.564382, 0.562612, 0.553539 | 0.562612 | no |
| immutable_hardened_campaign / 0 | authoritative_fixed_model_immutable | region_torch_mm | torch_mm_direct | 3 | 0.134496, 0.141232, 0.141008 | 0.134096, 0.145504, 0.142432 | 1.003540, 1.032630, 1.002945 | 1.003540 | no |
| superseded_fp8_wq_backend_campaign / 2 | production_exact_isolated_bf16_projection | isolated_baseline | stock_reference_only | 3 | 0.024960, 0.025952, 0.025904 | - | - | - | - |
| superseded_fp8_wq_backend_campaign / 2 | production_exact_isolated_bf16_projection | isolated_best | sglang_cutedsl_tgv_direct | 3 | 0.101456, 0.035120, 0.037088 | 0.245136, 0.094432, 0.121392 | 0.405225, 0.344080, 0.314668 | 0.344080 | no |
| superseded_fp8_wq_backend_campaign / 2 | production_exact_isolated_bf16_projection | isolated_cutedsl_sweep | sglang_cutedsl_tgv_direct | 1 | 0.034400 | 0.093520 | 0.370204 | 0.370204 | no |
| superseded_fp8_wq_backend_campaign / 2 | production_exact_isolated_bf16_projection | isolated_flashinfer_auto_sweep | flashinfer_mm_bf16_auto | 1 | 0.066064 | 0.363648 | 0.217030 | 0.217030 | no |
| superseded_fp8_wq_backend_campaign / 2 | production_exact_isolated_bf16_projection | isolated_flashinfer_cublaslt_sweep | flashinfer_mm_bf16_cublaslt | 1 | 0.040400 | 0.345568 | 0.109375 | 0.109375 | no |
| superseded_fp8_wq_backend_campaign / 2 | production_exact_isolated_bf16_projection | isolated_flashinfer_cudnn_sweep | flashinfer_mm_bf16_cudnn | 1 | 0.053584 | 0.178576 | 0.300118 | 0.300118 | no |
| superseded_fp8_wq_backend_campaign / 2 | production_exact_isolated_bf16_projection | isolated_flashinfer_cutlass_sweep | flashinfer_mm_bf16_cutlass | 1 | 0.043536 | 0.149600 | 0.315262 | 0.315262 | no |
| superseded_fp8_wq_backend_campaign / 2 | production_exact_isolated_bf16_projection | isolated_flashinfer_tgv_sweep | flashinfer_mm_bf16_tgv | 1 | 0.033008 | 0.182656 | 0.194219 | 0.194219 | no |
| superseded_fp8_wq_backend_campaign / 2 | production_exact_isolated_bf16_projection | isolated_reference_control | reference | 1 | 0.026576 | 0.028000 | 0.959423 | 0.959423 | no |
| superseded_fp8_wq_backend_campaign / 2 | superseded_wrong_fp8_wq_and_rope_region | region_baseline | stock_reference_only | 3 | 0.720416, 0.250208, 0.379008 | - | - | - | - |
| superseded_fp8_wq_backend_campaign / 2 | superseded_wrong_fp8_wq_and_rope_region | region_best | sglang_cutedsl_tgv_direct | 3 | 0.432000, 0.341760, 0.440928 | 0.574816, 0.443600, 0.581776 | 0.751610, 0.752319, 0.760573 | 0.752319 | no |
| superseded_fp8_wq_backend_campaign / 2 | superseded_wrong_fp8_wq_and_rope_region | region_reference_control | reference | 1 | 0.334800 | 0.342080 | 0.983630 | 0.983630 | no |
| superseded_fp8_wq_post_revert_smoke / 0 | production_exact_isolated_bf16_projection | post_revert_isolated_stock | stock_reference_only | 1 | 0.025888 | - | - | - | - |
| superseded_fp8_wq_post_revert_smoke / 0 | superseded_wrong_fp8_wq_and_rope_region | post_revert_region_stock | stock_reference_only | 1 | 0.326112 | - | - | - | - |
| superseded_fp8_wq_schedule_ncu_campaign / 1 | production_exact_isolated_bf16_projection | isolated_torch_mm | torch_mm_direct | 3 | 0.025248, 0.026288, 0.026256 | 0.025392, 0.027104, 0.026816 | 0.992703, 0.984665, 0.979441 | 0.984665 | no |
| superseded_fp8_wq_schedule_ncu_campaign / 1 | superseded_wrong_fp8_wq_and_rope_region | region_k_first | stock_bf16_gemm_k_before_q_schedule | 3 | 0.314624, 0.469776, 0.251344 | 0.317728, 0.423824, 0.252352 | 0.991256, 1.114721, 1.007711 | 1.007711 | no |
| superseded_fp8_wq_schedule_ncu_campaign / 1 | superseded_wrong_fp8_wq_and_rope_region | region_torch_mm | torch_mm_direct | 3 | 0.378464, 0.382256, 0.539168 | 0.386592, 0.383536, 0.531904 | 1.015425, 1.056410, 1.001299 | 1.015425 | no |

## Separate validation lanes

- Immutable hardened same-GPU rerun: validated (`evidence/glm52_prod_05_indexer_k_weights_prefill/hardened_runs/20260722T174049Z-immutable`)
- Exact fixed-model Q/K NCU: blocked after three scheduler exit-75 attempts.
- Corrected live TP4/DP4/EP4 trace: blocked by shared four-GPU scheduler after 180 exit-75 attempts (`evidence/glm52_prod_05_indexer_k_weights_prefill/tp4_live/20260722T181018Z-canonical_scheduler_blocker.json`); this is not the TP8 gate.

## Missing or unclassified artifacts

All artifacts expected by the completed single-GPU campaign specs are present. Q/K NCU and TP4 remain separate validation lanes.
