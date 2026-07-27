# Paired production-W2 summary

| Workload | Alignment | SMs | Pairs | Ref p50 (ms) | Cand p50 (ms) | Paired p10 | Paired p50 | Paired p90 | 3% gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| moe_w2_grouped_decode_m16 | 16 | None | 90 | 0.098208 | 0.091376 | 1.020861x | 1.080470x | 1.139754x | True |
| moe_w2_grouped_decode_m16_current_source_m5 | 16 | None | 90 | 0.103024 | 0.095040 | 0.983099x | 1.087436x | 1.229160x | True |
| moe_w2_grouped_decode_m32 | 16 | None | 90 | 0.103376 | 0.096368 | 1.015899x | 1.075564x | 1.152016x | True |
| moe_w2_grouped_decode_m32_current_source_m9 | 16 | None | 90 | 0.100640 | 0.096752 | 0.938654x | 1.062069x | 1.136703x | True |
| moe_w2_grouped_decode_m16 | 32 | None | 90 | 0.105584 | 0.100448 | 0.982317x | 1.056515x | 1.124097x | True |
| moe_w2_grouped_decode_m16_current_source_m5 | 32 | None | 90 | 0.099440 | 0.093840 | 1.002623x | 1.063299x | 1.127022x | True |
| moe_w2_grouped_decode_m32 | 32 | None | 90 | 0.098816 | 0.092512 | 1.000329x | 1.067834x | 1.124121x | True |
| moe_w2_grouped_decode_m32_current_source_m9 | 32 | None | 90 | 0.099312 | 0.094992 | 1.002978x | 1.052099x | 1.112546x | True |
| moe_w2_grouped_decode_m16 | 64 | None | 90 | 0.098624 | 0.095568 | 0.988940x | 1.032179x | 1.081605x | True |
| moe_w2_grouped_decode_m16_current_source_m5 | 64 | None | 90 | 0.099456 | 0.096096 | 0.998377x | 1.032888x | 1.112483x | True |
| moe_w2_grouped_decode_m32 | 64 | None | 90 | 0.098912 | 0.095760 | 0.974598x | 1.029308x | 1.091061x | False |
| moe_w2_grouped_decode_m32_current_source_m9 | 64 | None | 90 | 0.098320 | 0.096464 | 0.972524x | 1.027582x | 1.089081x | False |
| moe_w2_grouped_decode_m16 | 96 | None | 90 | 0.100048 | 0.097792 | 0.977295x | 1.022002x | 1.085235x | False |
| moe_w2_grouped_decode_m16_current_source_m5 | 96 | None | 90 | 0.099456 | 0.098496 | 0.963901x | 1.013852x | 1.088206x | False |
| moe_w2_grouped_decode_m32 | 96 | None | 90 | 0.098160 | 0.097168 | 0.971135x | 1.016296x | 1.077696x | False |
| moe_w2_grouped_decode_m32_current_source_m9 | 96 | None | 90 | 0.101872 | 0.097728 | 0.954383x | 1.021419x | 1.092208x | False |
| moe_w2_grouped_decode_m16 | 128 | None | 90 | 0.100016 | 0.100512 | 0.937728x | 0.994507x | 1.051629x | False |
| moe_w2_grouped_decode_m16_current_source_m5 | 128 | None | 90 | 0.100432 | 0.100752 | 0.939477x | 0.992272x | 1.068108x | False |
| moe_w2_grouped_decode_m32 | 128 | None | 90 | 0.106736 | 0.107408 | 0.936286x | 0.992249x | 1.067183x | False |
| moe_w2_grouped_decode_m32_current_source_m9 | 128 | None | 90 | 0.105328 | 0.105776 | 0.947805x | 0.994566x | 1.079771x | False |
