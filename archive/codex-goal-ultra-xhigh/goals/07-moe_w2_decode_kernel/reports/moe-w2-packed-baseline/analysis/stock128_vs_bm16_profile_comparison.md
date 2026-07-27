# Stock BM128 versus BM16 profile comparison

Campaign: `glm52-w2-alignment-2bae536257aa929b957fdb28`

Stock attempt: `profile_glm52-w2-alignment-2bae536257aa929b957fdb28_20260722T172516Z_656663_30552`

BM16 attempt: `profile_glm52-w2-alignment-2bae536257aa929b957fdb28_20260722T180259Z_1117292_24148`

> NCU and Nsys values below are one-shot profiled launches. They diagnose the mechanism; the canonical alternating paired campaign is the only latency performance claim.

## Canonical paired performance

| Workload | Stock-control paired p50 | BM16 paired p10 | BM16 paired p50 | BM16 paired p90 | BM16 gate |
|---|---:|---:|---:|---:|---|
| moe_w2_grouped_decode_m16 | 0.994507x | 1.020861x | 1.080470x | 1.139754x | True |
| moe_w2_grouped_decode_m32 | 0.992249x | 1.015899x | 1.075564x | 1.152016x | True |
| moe_w2_grouped_decode_m16_current_source_m5 | 0.992272x | 0.983099x | 1.087436x | 1.229160x | True |
| moe_w2_grouped_decode_m32_current_source_m9 | 0.994566x | 0.938654x | 1.062069x | 1.136703x | True |

## Selected configs and resources

| Field | Stock BM128 | BM16 | Delta |
|---|---:|---:|---:|
| block_m | 128 | 16 | -112 |
| block_n | 128 | 128 | 0 |
| block_k | 128 | 128 | 0 |
| load_block_m | 64 | 8 | -56 |
| store_block_m | 16 | 16 | 0 |
| smem_size | 213804 | 230188 | 16384 |
| num_stages | 8 | 12 | 4 |
| num_sms | 148 | 148 | 0 |
| num_threads | 256 | 256 | 0 |
| num_tma_threads | 32 | 32 | 0 |
| num_math_threads | 128 | 128 | 0 |
| num_non_epilogue_threads | 128 | 128 | 0 |
| num_epilogue_threads | 128 | 128 | 0 |
| num_waves | 11 | 11 | 0 |
| last_wave_util | 56 | 56 | 0 |
| cuobjdump_registers | 36 | 34 | -2 |
| ptxas_spill_store_bytes | 0 | 0 | 0 |
| ptxas_spill_load_bytes | 0 | 0 | 0 |

- Stock BM128 cache key: `32373bad60bad3f321c765b3eaa1d7a2`
- Stock BM128 cuobjdump: 36 registers, 0 stack bytes, 0 local bytes
- Stock BM128 ptxas: 0 spill-store bytes, 0 spill-load bytes, 9 barriers

- BM16 cache key: `0aaee40e9c1a31495b1fdca0df8f100c`
- BM16 cuobjdump: 34 registers, 0 stack bytes, 0 local bytes
- BM16 ptxas: 0 spill-store bytes, 0 spill-load bytes, 9 barriers

## Mask and logical-tail model

| Workload | Variant | Useful rows | Scheduled rows | Useful fraction | Logical tiles | Logical waves | Last wave CTAs |
|---|---|---:|---:|---:|---:|---:|---:|
| moe_w2_grouped_decode_m16 | BM128 | 128 | 4096 | 0.031250 | 1536 | 11 | 56 |
| moe_w2_grouped_decode_m16 | BM16 | 128 | 512 | 0.250000 | 1536 | 11 | 56 |
| moe_w2_grouped_decode_m32 | BM128 | 256 | 4096 | 0.062500 | 1536 | 11 | 56 |
| moe_w2_grouped_decode_m32 | BM16 | 256 | 512 | 0.500000 | 1536 | 11 | 56 |
| moe_w2_grouped_decode_m16_current_source_m5 | BM128 | 128 | 4096 | 0.031250 | 1536 | 11 | 56 |
| moe_w2_grouped_decode_m16_current_source_m5 | BM16 | 128 | 512 | 0.250000 | 1536 | 11 | 56 |
| moe_w2_grouped_decode_m32_current_source_m9 | BM128 | 256 | 4096 | 0.062500 | 1536 | 11 | 56 |
| moe_w2_grouped_decode_m32_current_source_m9 | BM16 | 256 | 512 | 0.500000 | 1536 | 11 | 56 |

## Nsys one-shot trace facts

| Workload | Stock kernel ns | BM16 kernel ns | Delta | Stock API ns | BM16 API ns | Stock queue ns | BM16 queue ns |
|---|---:|---:|---:|---:|---:|---:|---:|
| moe_w2_grouped_decode_m16 | 75200 | 65664 | -9536 | 78512 | 93139 | n/a | n/a |
| moe_w2_grouped_decode_m32 | 75488 | 65792 | -9696 | 97202 | 72298 | n/a | n/a |
| moe_w2_grouped_decode_m16_current_source_m5 | 74687 | 66816 | -7871 | 72755 | 77198 | n/a | n/a |
| moe_w2_grouped_decode_m32_current_source_m9 | 76544 | 66208 | -10336 | 126333 | 82124 | n/a | n/a |

## Six-dimension NCU comparison

### moe_w2_grouped_decode_m16

Profiled duration: 76160 ns stock, 69248 ns BM16. This is diagnostic, not the performance gate.

#### launch_occupancy

| Metric | Stock | BM16 | Delta | Delta % |
|---|---:|---:|---:|---:|
| `launch__grid_size` | 148 | 148 | 0 | 0 |
| `launch__block_size` | 256 | 256 | 0 | 0 |
| `launch__cluster_size` | 2 | 2 | 0 | 0 |
| `launch__waves_per_multiprocessor` | 1 | 1 | 0 | 0 |
| `launch__registers_per_thread` | 36 | 34 | -2 | -5.55556 |
| `launch__shared_mem_per_block` | 214828 | 231212 | 16384 | 7.62657 |
| `launch__shared_mem_per_block_dynamic` | 213804 | 230188 | 16384 | 7.66309 |
| `launch__occupancy_limit_registers` | 6 | 6 | 0 | 0 |
| `launch__occupancy_limit_shared_mem` | 1 | 1 | 0 | 0 |
| `launch__occupancy_limit_warps` | 8 | 8 | 0 | 0 |
| `sm__maximum_warps_per_active_cycle_pct` | 12.5 | 12.5 | 0 | 0 |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | 12.4965 | 12.5704 | 0.0739469 | 0.591743 |

#### block_balance

| Metric | Stock | BM16 | Delta | Delta % |
|---|---:|---:|---:|---:|
| `sm__cycles_active.avg` | 124763 | 110058 | -14705.6 | -11.7868 |
| `sm__cycles_active.max` | 131478 | 119023 | -12455 | -9.47307 |
| `sm__cycles_active.min` | 118533 | 98844 | -19689 | -16.6106 |
| `sm__cycles_active.sum` | 1.8465e+07 | 1.62885e+07 | -2.17644e+06 | -11.7868 |
| `derived__max_percent_above_average` | 5.38187 | 8.14594 | 2.76408 | 51.359 |
| `derived__min_percent_below_average` | 4.99377 | 10.189 | 5.1952 | 104.034 |

#### scheduler_stalls

| Metric | Stock | BM16 | Delta | Delta % |
|---|---:|---:|---:|---:|
| `smsp__warps_eligible.avg.per_cycle_active` | 0.0730065 | 0.0592677 | -0.0137388 | -18.8185 |
| `smsp__issue_active.avg.pct_of_peak_sustained_active` | 7.06424 | 5.72038 | -1.34387 | -19.0235 |
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 15.8572 | 21.5589 | 5.70173 | 35.9567 |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.804995 | 0.680356 | -0.124639 | -15.4832 |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.64655 | 1.66206 | 0.0155143 | 0.942231 |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 6.47629 | 6.8937 | 0.417408 | 6.44518 |
| `smsp__pcsamp_sample_count` | 4533 | 3977 | -556 | -12.2656 |
| `smsp__pcsamp_warps_issue_stalled_long_scoreboard` | 2656 | 2470 | -186 | -7.00301 |
| `smsp__pcsamp_warps_issue_stalled_barrier` | 981 | 788 | -193 | -19.6738 |
| `smsp__pcsamp_warps_issue_stalled_selected` | 135 | 127 | -8 | -5.92593 |
| `derived__pcsamp_long_scoreboard_percent` | 58.5925 | 62.1071 | 3.51457 | 5.99833 |
| `derived__pcsamp_barrier_percent` | 21.6413 | 19.8139 | -1.82737 | -8.44389 |
| `derived__pcsamp_selected_percent` | 2.97816 | 3.19336 | 0.215202 | 7.22599 |

#### tensor_core

| Metric | Stock | BM16 | Delta | Delta % |
|---|---:|---:|---:|---:|
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | 34.0724 | 4.82813 | -29.2442 | -85.8298 |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | 30.4053 | 4.19121 | -26.2141 | -86.2155 |
| `sm__pipe_tensor_subpipe_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed` | 30.4053 | 4.19121 | -26.2141 | -86.2155 |

#### pm_timeline

| Metric | Stock | BM16 | Delta | Delta % |
|---|---:|---:|---:|---:|
| `pmsampling:smsp__warps_issue_stalled_long_scoreboard.avg` | 139586 | 137723 | -1862.54 | -1.33433 |
| `pmsampling:smsp__warps_issue_stalled_short_scoreboard.avg` | 7043.75 | 4311.47 | -2732.28 | -38.7902 |
| `pmsampling:smsp__warps_issue_stalled_wait.avg` | 14422.6 | 10536.9 | -3885.67 | -26.9416 |
| `pmsampling:smsp__warps_issue_stalled_barrier.avg` | 56490.4 | 43823.7 | -12666.7 | -22.4228 |

PM time-series summary:

- `pmsampling:smsp__warps_issue_stalled_long_scoreboard.avg` active-mean: 1938.69 → 2118.82; active samples 72 → 65.
- `pmsampling:smsp__warps_issue_stalled_short_scoreboard.avg` active-mean: 97.8299 → 65.3253; active samples 72 → 66.
- `pmsampling:smsp__warps_issue_stalled_wait.avg` active-mean: 197.569 → 159.65; active samples 73 → 66.
- `pmsampling:smsp__warps_issue_stalled_barrier.avg` active-mean: 784.589 → 663.996; active samples 72 → 66.

#### memory_access

| Metric | Stock | BM16 | Delta | Delta % |
|---|---:|---:|---:|---:|
| `dram__bytes_read.sum` | 4.14282e+08 | 4.06904e+08 | -7.37792e+06 | -1.78089 |
| `dram__bytes_read.sum.pct_of_peak_sustained_elapsed` | 70.9598 | 76.657 | 5.69722 | 8.02879 |
| `dram__bytes_write.sum` | 4.06584e+07 | 8.34304e+06 | -3.23154e+07 | -79.4802 |
| `dram__bytes_write.sum.pct_of_peak_sustained_elapsed` | 6.96414 | 1.57175 | -5.39238 | -77.4307 |
| `l1tex__t_sector_hit_rate.pct` | 98.3693 | 98.3693 | 0 | 0 |
| `lts__t_sector_hit_rate.pct` | 33.9378 | 25.9879 | -7.94988 | -23.4248 |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | 40768 | 40768 | 0 | 0 |
| `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum` | 40768 | 40768 | 0 | 0 |
| `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio` | 4 | 4 | 0 | 0 |
| `smsp__sass_inst_executed_op_local_ld.sum` | 0 | 0 | 0 | n/a |
| `smsp__sass_inst_executed_op_local_st.sum` | 0 | 0 | 0 | n/a |
| `derived__global_load_sectors_per_request` | 1 | 1 | 0 | 0 |

Top source-correlated stalls:
- stock: barrier.h:424 (4408); barrier.cuh:18 (1754); sm100_fp8_fp4_gemm_1d1d.cuh:524 (524)
- BM16: barrier.h:424 (4629); barrier.cuh:18 (1728); sm100_fp8_fp4_gemm_1d1d.cuh:524 (409)

### moe_w2_grouped_decode_m32

Profiled duration: 75776 ns stock, 70112 ns BM16. This is diagnostic, not the performance gate.

#### launch_occupancy

| Metric | Stock | BM16 | Delta | Delta % |
|---|---:|---:|---:|---:|
| `launch__grid_size` | 148 | 148 | 0 | 0 |
| `launch__block_size` | 256 | 256 | 0 | 0 |
| `launch__cluster_size` | 2 | 2 | 0 | 0 |
| `launch__waves_per_multiprocessor` | 1 | 1 | 0 | 0 |
| `launch__registers_per_thread` | 36 | 34 | -2 | -5.55556 |
| `launch__shared_mem_per_block` | 214828 | 231212 | 16384 | 7.62657 |
| `launch__shared_mem_per_block_dynamic` | 213804 | 230188 | 16384 | 7.66309 |
| `launch__occupancy_limit_registers` | 6 | 6 | 0 | 0 |
| `launch__occupancy_limit_shared_mem` | 1 | 1 | 0 | 0 |
| `launch__occupancy_limit_warps` | 8 | 8 | 0 | 0 |
| `sm__maximum_warps_per_active_cycle_pct` | 12.5 | 12.5 | 0 | 0 |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | 12.4873 | 12.6003 | 0.112962 | 0.904615 |

#### block_balance

| Metric | Stock | BM16 | Delta | Delta % |
|---|---:|---:|---:|---:|
| `sm__cycles_active.avg` | 124385 | 110063 | -14321.7 | -11.514 |
| `sm__cycles_active.max` | 132746 | 121729 | -11017 | -8.29931 |
| `sm__cycles_active.min` | 118101 | 96792 | -21309 | -18.043 |
| `sm__cycles_active.sum` | 1.84089e+07 | 1.62893e+07 | -2.11961e+06 | -11.514 |
| `derived__max_percent_above_average` | 6.72228 | 10.5995 | 3.87725 | 57.6776 |
| `derived__min_percent_below_average` | 5.05169 | 12.0575 | 7.00583 | 138.683 |

#### scheduler_stalls

| Metric | Stock | BM16 | Delta | Delta % |
|---|---:|---:|---:|---:|
| `smsp__warps_eligible.avg.per_cycle_active` | 0.0729872 | 0.0600247 | -0.0129626 | -17.7601 |
| `smsp__issue_active.avg.pct_of_peak_sustained_active` | 7.06484 | 5.80249 | -1.26235 | -17.8681 |
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 16.0397 | 21.4723 | 5.4326 | 33.8698 |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.806597 | 0.682485 | -0.124112 | -15.3872 |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.64799 | 1.66103 | 0.0130397 | 0.79125 |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 6.48551 | 6.90107 | 0.415562 | 6.40755 |
| `smsp__pcsamp_sample_count` | 4509 | 3985 | -524 | -11.6212 |
| `smsp__pcsamp_warps_issue_stalled_long_scoreboard` | 2597 | 2423 | -174 | -6.70004 |
| `smsp__pcsamp_warps_issue_stalled_barrier` | 980 | 794 | -186 | -18.9796 |
| `smsp__pcsamp_warps_issue_stalled_selected` | 146 | 116 | -30 | -20.5479 |
| `derived__pcsamp_long_scoreboard_percent` | 57.5959 | 60.803 | 3.20709 | 5.56826 |
| `derived__pcsamp_barrier_percent` | 21.7343 | 19.9247 | -1.80959 | -8.32597 |
| `derived__pcsamp_selected_percent` | 3.23797 | 2.91092 | -0.327053 | -10.1005 |

#### tensor_core

| Metric | Stock | BM16 | Delta | Delta % |
|---|---:|---:|---:|---:|
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | 34.1761 | 4.8279 | -29.3482 | -85.8735 |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | 31.0008 | 4.17984 | -26.821 | -86.517 |
| `sm__pipe_tensor_subpipe_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed` | 31.0008 | 4.17984 | -26.821 | -86.517 |

#### pm_timeline

| Metric | Stock | BM16 | Delta | Delta % |
|---|---:|---:|---:|---:|
| `pmsampling:smsp__warps_issue_stalled_long_scoreboard.avg` | 141201 | 136890 | -4311.02 | -3.05312 |
| `pmsampling:smsp__warps_issue_stalled_short_scoreboard.avg` | 7051.96 | 4313.78 | -2738.19 | -38.8287 |
| `pmsampling:smsp__warps_issue_stalled_wait.avg` | 14417.1 | 10539.8 | -3877.36 | -26.8942 |
| `pmsampling:smsp__warps_issue_stalled_barrier.avg` | 56730.1 | 43118.9 | -13611.2 | -23.9929 |

PM time-series summary:

- `pmsampling:smsp__warps_issue_stalled_long_scoreboard.avg` active-mean: 1934.26 → 2074.09; active samples 73 → 66.
- `pmsampling:smsp__warps_issue_stalled_short_scoreboard.avg` active-mean: 96.6023 → 65.3602; active samples 73 → 66.
- `pmsampling:smsp__warps_issue_stalled_wait.avg` active-mean: 197.495 → 159.693; active samples 73 → 66.
- `pmsampling:smsp__warps_issue_stalled_barrier.avg` active-mean: 787.918 → 663.367; active samples 72 → 65.

#### memory_access

| Metric | Stock | BM16 | Delta | Delta % |
|---|---:|---:|---:|---:|
| `dram__bytes_read.sum` | 4.14309e+08 | 4.06903e+08 | -7.40582e+06 | -1.78751 |
| `dram__bytes_read.sum.pct_of_peak_sustained_elapsed` | 71.3011 | 75.6844 | 4.38331 | 6.1476 |
| `dram__bytes_write.sum` | 4.0674e+07 | 8.33254e+06 | -3.23415e+07 | -79.5139 |
| `dram__bytes_write.sum.pct_of_peak_sustained_elapsed` | 6.99987 | 1.54986 | -5.45 | -77.8587 |
| `l1tex__t_sector_hit_rate.pct` | 98.3693 | 98.3693 | 0 | 0 |
| `lts__t_sector_hit_rate.pct` | 33.886 | 25.866 | -8.01994 | -23.6674 |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | 40768 | 40768 | 0 | 0 |
| `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum` | 40768 | 40768 | 0 | 0 |
| `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio` | 4 | 4 | 0 | 0 |
| `smsp__sass_inst_executed_op_local_ld.sum` | 0 | 0 | 0 | n/a |
| `smsp__sass_inst_executed_op_local_st.sum` | 0 | 0 | 0 | n/a |
| `derived__global_load_sectors_per_request` | 1 | 1 | 0 | 0 |

Top source-correlated stalls:
- stock: barrier.h:424 (4279); barrier.cuh:18 (1753); sm100_fp8_fp4_gemm_1d1d.cuh:524 (504)
- BM16: barrier.h:424 (4609); barrier.cuh:18 (1709); sm100_fp8_fp4_gemm_1d1d.cuh:524 (471)

### moe_w2_grouped_decode_m16_current_source_m5

Profiled duration: 75520 ns stock, 68896 ns BM16. This is diagnostic, not the performance gate.

#### launch_occupancy

| Metric | Stock | BM16 | Delta | Delta % |
|---|---:|---:|---:|---:|
| `launch__grid_size` | 148 | 148 | 0 | 0 |
| `launch__block_size` | 256 | 256 | 0 | 0 |
| `launch__cluster_size` | 2 | 2 | 0 | 0 |
| `launch__waves_per_multiprocessor` | 1 | 1 | 0 | 0 |
| `launch__registers_per_thread` | 36 | 34 | -2 | -5.55556 |
| `launch__shared_mem_per_block` | 214828 | 231212 | 16384 | 7.62657 |
| `launch__shared_mem_per_block_dynamic` | 213804 | 230188 | 16384 | 7.66309 |
| `launch__occupancy_limit_registers` | 6 | 6 | 0 | 0 |
| `launch__occupancy_limit_shared_mem` | 1 | 1 | 0 | 0 |
| `launch__occupancy_limit_warps` | 8 | 8 | 0 | 0 |
| `sm__maximum_warps_per_active_cycle_pct` | 12.5 | 12.5 | 0 | 0 |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | 12.5037 | 12.4761 | -0.0276546 | -0.221171 |

#### block_balance

| Metric | Stock | BM16 | Delta | Delta % |
|---|---:|---:|---:|---:|
| `sm__cycles_active.avg` | 124660 | 110062 | -14597.1 | -11.7096 |
| `sm__cycles_active.max` | 130805 | 119132 | -11673 | -8.92397 |
| `sm__cycles_active.min` | 117714 | 99468 | -18246 | -15.5003 |
| `sm__cycles_active.sum` | 1.84496e+07 | 1.62892e+07 | -2.16037e+06 | -11.7096 |
| `derived__max_percent_above_average` | 4.92983 | 8.2404 | 3.31057 | 67.1538 |
| `derived__min_percent_below_average` | 5.57158 | 9.62583 | 4.05425 | 72.7667 |

#### scheduler_stalls

| Metric | Stock | BM16 | Delta | Delta % |
|---|---:|---:|---:|---:|
| `smsp__warps_eligible.avg.per_cycle_active` | 0.0733176 | 0.0601864 | -0.0131312 | -17.9101 |
| `smsp__issue_active.avg.pct_of_peak_sustained_active` | 7.10017 | 5.80966 | -1.29051 | -18.1757 |
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 15.8417 | 21.6216 | 5.77991 | 36.4854 |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.805196 | 0.680533 | -0.124663 | -15.4823 |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.6461 | 1.66407 | 0.0179756 | 1.09201 |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 6.47381 | 6.83421 | 0.360405 | 5.56713 |
| `smsp__pcsamp_sample_count` | 4498 | 4011 | -487 | -10.827 |
| `smsp__pcsamp_warps_issue_stalled_long_scoreboard` | 2591 | 2477 | -114 | -4.39985 |
| `smsp__pcsamp_warps_issue_stalled_barrier` | 965 | 796 | -169 | -17.513 |
| `smsp__pcsamp_warps_issue_stalled_selected` | 150 | 111 | -39 | -26 |
| `derived__pcsamp_long_scoreboard_percent` | 57.6034 | 61.7552 | 4.15179 | 7.20755 |
| `derived__pcsamp_barrier_percent` | 21.454 | 19.8454 | -1.60855 | -7.4977 |
| `derived__pcsamp_selected_percent` | 3.33482 | 2.76739 | -0.567426 | -17.0152 |

#### tensor_core

| Metric | Stock | BM16 | Delta | Delta % |
|---|---:|---:|---:|---:|
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | 34.1008 | 4.82792 | -29.2728 | -85.8422 |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | 30.7029 | 4.21253 | -26.4904 | -86.2797 |
| `sm__pipe_tensor_subpipe_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed` | 30.7029 | 4.21253 | -26.4904 | -86.2797 |

#### pm_timeline

| Metric | Stock | BM16 | Delta | Delta % |
|---|---:|---:|---:|---:|
| `pmsampling:smsp__warps_issue_stalled_long_scoreboard.avg` | 139411 | 137170 | -2240.72 | -1.60728 |
| `pmsampling:smsp__warps_issue_stalled_short_scoreboard.avg` | 7058.44 | 4312.85 | -2745.58 | -38.8979 |
| `pmsampling:smsp__warps_issue_stalled_wait.avg` | 14422.4 | 10524.1 | -3898.33 | -27.0297 |
| `pmsampling:smsp__warps_issue_stalled_barrier.avg` | 56877.5 | 43572.4 | -13305.1 | -23.3926 |

PM time-series summary:

- `pmsampling:smsp__warps_issue_stalled_long_scoreboard.avg` active-mean: 1936.26 → 2143.28; active samples 72 → 64.
- `pmsampling:smsp__warps_issue_stalled_short_scoreboard.avg` active-mean: 96.6909 → 66.3516; active samples 73 → 65.
- `pmsampling:smsp__warps_issue_stalled_wait.avg` active-mean: 200.311 → 161.909; active samples 72 → 65.
- `pmsampling:smsp__warps_issue_stalled_barrier.avg` active-mean: 779.145 → 670.345; active samples 73 → 65.

#### memory_access

| Metric | Stock | BM16 | Delta | Delta % |
|---|---:|---:|---:|---:|
| `dram__bytes_read.sum` | 4.14281e+08 | 4.06903e+08 | -7.37741e+06 | -1.78078 |
| `dram__bytes_read.sum.pct_of_peak_sustained_elapsed` | 71.5526 | 77.0247 | 5.47207 | 7.64762 |
| `dram__bytes_write.sum` | 4.04078e+07 | 7.93933e+06 | -3.24685e+07 | -80.352 |
| `dram__bytes_write.sum.pct_of_peak_sustained_elapsed` | 6.97905 | 1.50287 | -5.47617 | -78.4659 |
| `l1tex__t_sector_hit_rate.pct` | 98.3693 | 98.3693 | 0 | 0 |
| `lts__t_sector_hit_rate.pct` | 33.3138 | 25.9608 | -7.35299 | -22.0719 |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | 40768 | 40768 | 0 | 0 |
| `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum` | 40768 | 40768 | 0 | 0 |
| `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio` | 4 | 4 | 0 | 0 |
| `smsp__sass_inst_executed_op_local_ld.sum` | 0 | 0 | 0 | n/a |
| `smsp__sass_inst_executed_op_local_st.sum` | 0 | 0 | 0 | n/a |
| `derived__global_load_sectors_per_request` | 1 | 1 | 0 | 0 |

Top source-correlated stalls:
- stock: barrier.h:424 (4458); barrier.cuh:18 (1747); sm100_fp8_fp4_gemm_1d1d.cuh:524 (500)
- BM16: barrier.h:424 (4609); barrier.cuh:18 (1709); sm100_fp8_fp4_gemm_1d1d.cuh:524 (477)

### moe_w2_grouped_decode_m32_current_source_m9

Profiled duration: 76256 ns stock, 68576 ns BM16. This is diagnostic, not the performance gate.

#### launch_occupancy

| Metric | Stock | BM16 | Delta | Delta % |
|---|---:|---:|---:|---:|
| `launch__grid_size` | 148 | 148 | 0 | 0 |
| `launch__block_size` | 256 | 256 | 0 | 0 |
| `launch__cluster_size` | 2 | 2 | 0 | 0 |
| `launch__waves_per_multiprocessor` | 1 | 1 | 0 | 0 |
| `launch__registers_per_thread` | 36 | 34 | -2 | -5.55556 |
| `launch__shared_mem_per_block` | 214828 | 231212 | 16384 | 7.62657 |
| `launch__shared_mem_per_block_dynamic` | 213804 | 230188 | 16384 | 7.66309 |
| `launch__occupancy_limit_registers` | 6 | 6 | 0 | 0 |
| `launch__occupancy_limit_shared_mem` | 1 | 1 | 0 | 0 |
| `launch__occupancy_limit_warps` | 8 | 8 | 0 | 0 |
| `sm__maximum_warps_per_active_cycle_pct` | 12.5 | 12.5 | 0 | 0 |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | 12.489 | 12.5613 | 0.0722838 | 0.578778 |

#### block_balance

| Metric | Stock | BM16 | Delta | Delta % |
|---|---:|---:|---:|---:|
| `sm__cycles_active.avg` | 123623 | 109553 | -14070.1 | -11.3814 |
| `sm__cycles_active.max` | 131138 | 118315 | -12823 | -9.77825 |
| `sm__cycles_active.min` | 118284 | 98551 | -19733 | -16.6827 |
| `sm__cycles_active.sum` | 1.82962e+07 | 1.62139e+07 | -2.08237e+06 | -11.3814 |
| `derived__max_percent_above_average` | 6.07882 | 7.99786 | 1.91905 | 31.5694 |
| `derived__min_percent_below_average` | 4.31891 | 10.0427 | 5.72379 | 132.529 |

#### scheduler_stalls

| Metric | Stock | BM16 | Delta | Delta % |
|---|---:|---:|---:|---:|
| `smsp__warps_eligible.avg.per_cycle_active` | 0.0727432 | 0.0600693 | -0.0126739 | -17.4228 |
| `smsp__issue_active.avg.pct_of_peak_sustained_active` | 7.03348 | 5.79844 | -1.23504 | -17.5594 |
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 15.7688 | 21.568 | 5.79922 | 36.7766 |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.804374 | 0.681169 | -0.123205 | -15.3169 |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.64596 | 1.66085 | 0.014891 | 0.904698 |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 6.45271 | 6.85666 | 0.403947 | 6.26011 |
| `smsp__pcsamp_sample_count` | 4511 | 4007 | -504 | -11.1727 |
| `smsp__pcsamp_warps_issue_stalled_long_scoreboard` | 2622 | 2478 | -144 | -5.49199 |
| `smsp__pcsamp_warps_issue_stalled_barrier` | 962 | 801 | -161 | -16.736 |
| `smsp__pcsamp_warps_issue_stalled_selected` | 145 | 90 | -55 | -37.931 |
| `derived__pcsamp_long_scoreboard_percent` | 58.1246 | 61.8418 | 3.71719 | 6.39522 |
| `derived__pcsamp_barrier_percent` | 21.3256 | 19.99 | -1.33563 | -6.26303 |
| `derived__pcsamp_selected_percent` | 3.21436 | 2.24607 | -0.968296 | -30.124 |

#### tensor_core

| Metric | Stock | BM16 | Delta | Delta % |
|---|---:|---:|---:|---:|
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | 34.3866 | 4.85037 | -29.5363 | -85.8946 |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | 30.6572 | 4.23039 | -26.4268 | -86.201 |
| `sm__pipe_tensor_subpipe_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed` | 30.6572 | 4.23039 | -26.4268 | -86.201 |

#### pm_timeline

| Metric | Stock | BM16 | Delta | Delta % |
|---|---:|---:|---:|---:|
| `pmsampling:smsp__warps_issue_stalled_long_scoreboard.avg` | 140496 | 137352 | -3144.23 | -2.23795 |
| `pmsampling:smsp__warps_issue_stalled_short_scoreboard.avg` | 0 | 4312.51 | 4312.51 | n/a |
| `pmsampling:smsp__warps_issue_stalled_wait.avg` | 14425.2 | 10518.1 | -3907.09 | -27.0851 |
| `pmsampling:smsp__warps_issue_stalled_barrier.avg` | 56643.2 | 43745.7 | -12897.4 | -22.7696 |

PM time-series summary:

- `pmsampling:smsp__warps_issue_stalled_long_scoreboard.avg` active-mean: 1951.33 → 2113.11; active samples 72 → 65.
- `pmsampling:smsp__warps_issue_stalled_short_scoreboard.avg` active-mean: n/a → 66.3462; active samples None → 65.
- `pmsampling:smsp__warps_issue_stalled_wait.avg` active-mean: 200.35 → 161.817; active samples 72 → 65.
- `pmsampling:smsp__warps_issue_stalled_barrier.avg` active-mean: 786.711 → 673.011; active samples 72 → 65.

#### memory_access

| Metric | Stock | BM16 | Delta | Delta % |
|---|---:|---:|---:|---:|
| `dram__bytes_read.sum` | 4.14288e+08 | 4.06906e+08 | -7.38202e+06 | -1.78186 |
| `dram__bytes_read.sum.pct_of_peak_sustained_elapsed` | 70.8481 | 77.392 | 6.54387 | 9.23648 |
| `dram__bytes_write.sum` | 4.06013e+07 | 8.11366e+06 | -3.24877e+07 | -80.0163 |
| `dram__bytes_write.sum.pct_of_peak_sustained_elapsed` | 6.94331 | 1.54319 | -5.40012 | -77.7744 |
| `l1tex__t_sector_hit_rate.pct` | 98.3693 | 98.3693 | 0 | 0 |
| `lts__t_sector_hit_rate.pct` | 33.2049 | 25.9895 | -7.2154 | -21.7299 |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | 40768 | 40768 | 0 | 0 |
| `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum` | 40768 | 40768 | 0 | 0 |
| `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio` | 4 | 4 | 0 | 0 |
| `smsp__sass_inst_executed_op_local_ld.sum` | 0 | 0 | 0 | n/a |
| `smsp__sass_inst_executed_op_local_st.sum` | 0 | 0 | 0 | n/a |
| `derived__global_load_sectors_per_request` | 1 | 1 | 0 | 0 |

Top source-correlated stalls:
- stock: barrier.h:424 (4472); barrier.cuh:18 (1742); sm100_fp8_fp4_gemm_1d1d.cuh:524 (491)
- BM16: barrier.h:424 (4518); barrier.cuh:18 (1652); sm100_fp8_fp4_gemm_1d1d.cuh:524 (502)

## PTX/SASS instruction counts

### PTX

| Category | Stock | BM16 | Delta | Delta % |
|---|---:|---:|---:|---:|
| instructions_total | 1122 | 909 | -213 | -18.984 |
| tcgen05_mma | 16 | 16 | 0 | 0 |
| tcgen05_tmem_load | 32 | 4 | -28 | -87.5 |
| tcgen05_tmem_copy | 2 | 2 | 0 | 0 |
| tcgen05_alloc_dealloc | 2 | 2 | 0 | 0 |
| tcgen05_fence_commit_wait | 5 | 5 | 0 | 0 |
| tma_global_to_shared | 10 | 10 | 0 | 0 |
| tma_shared_to_global | 16 | 2 | -14 | -87.5 |
| async_wait_commit | 8 | 1 | -7 | -87.5 |
| mbarrier_all | 48 | 60 | 12 | 25 |
| mbarrier_init | 28 | 40 | 12 | 42.8571 |
| mbarrier_wait | 13 | 13 | 0 | 0 |
| global_load | 4 | 4 | 0 | 0 |
| global_store | 0 | 0 | 0 | n/a |
| shared_load | 9 | 9 | 0 | 0 |
| shared_store_or_stmatrix | 18 | 4 | -14 | -77.7778 |
| barrier | 34 | 13 | -21 | -61.7647 |
| shuffle_elect | 28 | 21 | -7 | -25 |
| branch | 98 | 77 | -21 | -21.4286 |

### SASS

| Category | Stock | BM16 | Delta | Delta % |
|---|---:|---:|---:|---:|
| instructions_total | 1416 | 1112 | -304 | -21.4689 |
| tensor_mma_utcqmma | 16 | 16 | 0 | 0 |
| tmem_load_ldtm | 32 | 4 | -28 | -87.5 |
| tma_load_utmaldg | 10 | 10 | 0 | 0 |
| tma_store_utmastg | 16 | 2 | -14 | -87.5 |
| tma_control_utmac | 13 | 6 | -7 | -53.8462 |
| global_load | 4 | 4 | 0 | 0 |
| global_store | 0 | 0 | 0 | n/a |
| local_load | 0 | 0 | 0 | n/a |
| local_store | 0 | 0 | 0 | n/a |
| shared_load | 15 | 15 | 0 | 0 |
| shared_store_sts_stsm | 21 | 7 | -14 | -66.6667 |
| epilogue_convert_f2fp | 64 | 8 | -56 | -87.5 |
| sync_barrier | 129 | 113 | -16 | -12.4031 |
| branch_control | 150 | 115 | -35 | -23.3333 |
| integer_address | 396 | 359 | -37 | -9.34343 |
