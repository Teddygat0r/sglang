# Live matched-memory cache-pressure experiment

Each interval is a two-sided Student-t 95% confidence interval across independent run repetitions.

## budget192_off (n=6)

| Metric | Mean | 95% CI |
|---|---:|---|
| requests | 192 | [192, 192] |
| duration_s | 122.68 | [121.49, 123.86] |
| token_cache_hit_rate | 0.36951 | [0.35144, 0.38758] |
| evicted_entries | 314.5 | [311, 318] |
| evicted_compressed_entries | 0 | [0, 0] |
| evicted_kv_tokens | 1.1797e+05 | [1.114e+05, 1.2455e+05] |
| recomputed_prefix_tokens | 1.1571e+05 | [1.0855e+05, 1.2287e+05] |
| request_throughput_rps | 1.5652 | [1.5501, 1.5803] |
| output_throughput_tps | 12.522 | [12.401, 12.642] |
| mean_ttft_ms | 288 | [281.24, 294.76] |
| p50_ttft_ms | 419.23 | [418.39, 420.07] |
| p95_ttft_ms | 429.06 | [428.06, 430.07] |
| p99_ttft_ms | 433.29 | [431.76, 434.82] |
| mean_tpot_ms | 50.106 | [50.003, 50.208] |
| persistent_cache_mib | 17673 | [17673, 17673] |
| peak_cache_state_mib | 17673 | [17673, 17673] |
| peak_staging_mib | 0 | [0, 0] |
| peak_process_cuda_mib | 27138 | [27138, 27139] |
| compression_pending_peak | 0 | [0, 0] |
| compression_enqueued | 0 | [0, 0] |
| compression_completed | 0 | [0, 0] |
| compression_completion_fraction | 0 | [0, 0] |
| compression_completion_per_s | 0 | [0, 0] |
| compression_committed | 0 | [0, 0] |
| compression_failed | 0 | [0, 0] |
| decompression_hits | 0 | [0, 0] |

## budget192_on (n=6)

| Metric | Mean | 95% CI |
|---|---:|---|
| requests | 192 | [192, 192] |
| duration_s | 146.56 | [145.65, 147.48] |
| token_cache_hit_rate | 0.068045 | [0.055021, 0.081069] |
| evicted_entries | 480.83 | [478.31, 483.35] |
| evicted_compressed_entries | 480.83 | [478.31, 483.35] |
| evicted_kv_tokens | 3.1097e+05 | [3.0474e+05, 3.172e+05] |
| recomputed_prefix_tokens | 2.3518e+05 | [2.3002e+05, 2.4034e+05] |
| request_throughput_rps | 1.31 | [1.3019, 1.3182] |
| output_throughput_tps | 10.48 | [10.415, 10.546] |
| mean_ttft_ms | 405.45 | [400.57, 410.33] |
| p50_ttft_ms | 429.7 | [428.84, 430.56] |
| p95_ttft_ms | 436.66 | [435.66, 437.66] |
| p99_ttft_ms | 448.59 | [442.63, 454.55] |
| mean_tpot_ms | 51.097 | [51.042, 51.153] |
| persistent_cache_mib | 17280 | [17280, 17280] |
| peak_cache_state_mib | 17400 | [17381, 17420] |
| peak_staging_mib | 120.06 | [100.55, 139.57] |
| peak_process_cuda_mib | 27139 | [27118, 27159] |
| compression_pending_peak | 2.3333 | [1.7914, 2.8753] |
| compression_enqueued | 562.83 | [560.31, 565.35] |
| compression_completed | 561.83 | [559.31, 564.35] |
| compression_completion_fraction | 0.99822 | [0.99822, 0.99823] |
| compression_completion_per_s | 3.8334 | [3.8261, 3.8407] |
| compression_committed | 561.83 | [559.31, 564.35] |
| compression_failed | 0 | [0, 0] |
| decompression_hits | 13.167 | [10.647, 15.687] |

## budget24_off (n=6)

| Metric | Mean | 95% CI |
|---|---:|---|
| requests | 192 | [192, 192] |
| duration_s | 148.99 | [148.33, 149.64] |
| token_cache_hit_rate | 0.005168 | [0.0017379, 0.008598] |
| evicted_entries | 553 | [552.34, 553.66] |
| evicted_compressed_entries | 0 | [0, 0] |
| evicted_kv_tokens | 3.7867e+05 | [3.7706e+05, 3.8029e+05] |
| recomputed_prefix_tokens | 2.601e+05 | [2.5874e+05, 2.6146e+05] |
| request_throughput_rps | 1.2887 | [1.283, 1.2945] |
| output_throughput_tps | 10.31 | [10.264, 10.356] |
| mean_ttft_ms | 423.21 | [420.31, 426.1] |
| p50_ttft_ms | 424.94 | [422.59, 427.29] |
| p95_ttft_ms | 432.51 | [429.74, 435.29] |
| p99_ttft_ms | 437.22 | [435.19, 439.26] |
| mean_tpot_ms | 50.363 | [50.268, 50.459] |
| persistent_cache_mib | 9420.2 | [9420.2, 9420.2] |
| peak_cache_state_mib | 9420.2 | [9420.2, 9420.2] |
| peak_staging_mib | 0 | [0, 0] |
| peak_process_cuda_mib | 18887 | [18887, 18887] |
| compression_pending_peak | 0 | [0, 0] |
| compression_enqueued | 0 | [0, 0] |
| compression_completed | 0 | [0, 0] |
| compression_completion_fraction | 0 | [0, 0] |
| compression_completion_per_s | 0 | [0, 0] |
| compression_committed | 0 | [0, 0] |
| compression_failed | 0 | [0, 0] |
| decompression_hits | 0 | [0, 0] |

## budget24_on (n=6)

| Metric | Mean | 95% CI |
|---|---:|---|
| requests | 192 | [192, 192] |
| duration_s | 151.39 | [151.1, 151.68] |
| token_cache_hit_rate | 0 | [0, 0] |
| evicted_entries | 568 | [568, 568] |
| evicted_compressed_entries | 568 | [568, 568] |
| evicted_kv_tokens | 3.9142e+05 | [3.9142e+05, 3.9142e+05] |
| recomputed_prefix_tokens | 2.6214e+05 | [2.6214e+05, 2.6214e+05] |
| request_throughput_rps | 1.2682 | [1.2658, 1.2707] |
| output_throughput_tps | 10.146 | [10.127, 10.165] |
| mean_ttft_ms | 430.64 | [429.39, 431.88] |
| p50_ttft_ms | 430.2 | [428.85, 431.54] |
| p95_ttft_ms | 437.9 | [435.67, 440.13] |
| p99_ttft_ms | 449.26 | [445.58, 452.94] |
| mean_tpot_ms | 51.099 | [51.012, 51.185] |
| persistent_cache_mib | 9027.1 | [9027.1, 9027.1] |
| peak_cache_state_mib | 9143.2 | [9122.6, 9163.8] |
| peak_staging_mib | 116.05 | [95.482, 136.61] |
| peak_process_cuda_mib | 18924 | [18852, 18997] |
| compression_pending_peak | 2 | [2, 2] |
| compression_enqueued | 576 | [576, 576] |
| compression_completed | 575 | [575, 575] |
| compression_completion_fraction | 0.99826 | [0.99826, 0.99826] |
| compression_completion_per_s | 3.7981 | [3.7909, 3.8054] |
| compression_committed | 575 | [575, 575] |
| compression_failed | 0 | [0, 0] |
| decompression_hits | 0 | [0, 0] |
