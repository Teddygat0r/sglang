# Live matched-memory cache-pressure experiment

Each interval is a two-sided Student-t 95% confidence interval across independent run repetitions.

## budget192_off (n=2)

| Metric | Mean | 95% CI |
|---|---:|---|
| requests | 192 | [192, 192] |
| duration_s | 123.72 | [114.34, 133.11] |
| token_cache_hit_rate | 0.35917 | [0.12935, 0.589] |
| evicted_entries | 316.5 | [272.03, 360.97] |
| evicted_compressed_entries | 0 | [0, 0] |
| evicted_kv_tokens | 1.2173e+05 | [43661, 1.9979e+05] |
| recomputed_prefix_tokens | 1.1981e+05 | [28730, 2.1089e+05] |
| request_throughput_rps | 1.5519 | [1.4342, 1.6696] |
| output_throughput_tps | 12.415 | [11.474, 13.357] |
| mean_ttft_ms | 292.39 | [221.34, 363.43] |
| p50_ttft_ms | 418.69 | [415.59, 421.79] |
| p95_ttft_ms | 428.84 | [427.36, 430.32] |
| p99_ttft_ms | 442.09 | [310.21, 573.97] |
| mean_tpot_ms | 50.259 | [47.041, 53.477] |
| persistent_cache_mib | 17673 | [17673, 17673] |
| peak_cache_state_mib | 17673 | [17673, 17673] |
| peak_staging_mib | 0 | [0, 0] |
| peak_process_cuda_mib | 27139 | [27136, 27142] |
| compression_pending_peak | 0 | [0, 0] |
| compression_enqueued | 0 | [0, 0] |
| compression_completed | 0 | [0, 0] |
| compression_completion_fraction | 0 | [0, 0] |
| compression_completion_per_s | 0 | [0, 0] |
| compression_committed | 0 | [0, 0] |
| compression_failed | 0 | [0, 0] |
| decompression_hits | 0 | [0, 0] |

## budget192_on (n=2)

| Metric | Mean | 95% CI |
|---|---:|---|
| requests | 192 | [192, 192] |
| duration_s | 146.08 | [128.35, 163.8] |
| token_cache_hit_rate | 0.072351 | [-0.12464, 0.26935] |
| evicted_entries | 478 | [439.88, 516.12] |
| evicted_compressed_entries | 478 | [439.88, 516.12] |
| evicted_kv_tokens | 3.0995e+05 | [2.4489e+05, 3.75e+05] |
| recomputed_prefix_tokens | 2.3347e+05 | [1.5541e+05, 3.1154e+05] |
| request_throughput_rps | 1.3145 | [1.155, 1.474] |
| output_throughput_tps | 10.516 | [9.2399, 11.792] |
| mean_ttft_ms | 403.77 | [318.88, 488.65] |
| p50_ttft_ms | 428.37 | [424.7, 432.05] |
| p95_ttft_ms | 439.87 | [395.35, 484.38] |
| p99_ttft_ms | 469.57 | [311.21, 627.93] |
| mean_tpot_ms | 50.981 | [49.87, 52.092] |
| persistent_cache_mib | 17526 | [17526, 17526] |
| peak_cache_state_mib | 17640 | [17563, 17716] |
| peak_staging_mib | 114.07 | [37.535, 190.61] |
| peak_process_cuda_mib | 27376 | [27376, 27376] |
| compression_pending_peak | 2.5 | [-3.8531, 8.8531] |
| compression_enqueued | 562 | [523.88, 600.12] |
| compression_completed | 561 | [522.88, 599.12] |
| compression_completion_fraction | 0.99822 | [0.9981, 0.99834] |
| compression_completion_per_s | 3.8406 | [3.6355, 4.0456] |
| compression_committed | 561 | [522.88, 599.12] |
| compression_failed | 0 | [0, 0] |
| decompression_hits | 14 | [-24.119, 52.119] |

## budget24_off (n=3)

| Metric | Mean | 95% CI |
|---|---:|---|
| requests | 192 | [192, 192] |
| duration_s | 148.81 | [147.8, 149.82] |
| token_cache_hit_rate | 0.005168 | [0.005168, 0.005168] |
| evicted_entries | 553 | [553, 553] |
| evicted_compressed_entries | 0 | [0, 0] |
| evicted_kv_tokens | 3.7833e+05 | [3.754e+05, 3.8127e+05] |
| recomputed_prefix_tokens | 2.601e+05 | [2.601e+05, 2.601e+05] |
| request_throughput_rps | 1.2902 | [1.2815, 1.299] |
| output_throughput_tps | 10.322 | [10.252, 10.392] |
| mean_ttft_ms | 422.07 | [419.41, 424.73] |
| p50_ttft_ms | 423.31 | [419.73, 426.89] |
| p95_ttft_ms | 432.56 | [430.92, 434.2] |
| p99_ttft_ms | 438.49 | [429.77, 447.21] |
| mean_tpot_ms | 50.399 | [50.037, 50.762] |
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

## budget24_on (n=2)

| Metric | Mean | 95% CI |
|---|---:|---|
| requests | 192 | [192, 192] |
| duration_s | 151.16 | [148.29, 154.04] |
| token_cache_hit_rate | 0 | [0, 0] |
| evicted_entries | 566 | [566, 566] |
| evicted_compressed_entries | 566 | [566, 566] |
| evicted_kv_tokens | 3.8935e+05 | [3.8935e+05, 3.8935e+05] |
| recomputed_prefix_tokens | 2.6214e+05 | [2.6214e+05, 2.6214e+05] |
| request_throughput_rps | 1.2701 | [1.246, 1.2943] |
| output_throughput_tps | 10.161 | [9.9676, 10.355] |
| mean_ttft_ms | 429.52 | [426.82, 432.22] |
| p50_ttft_ms | 428.79 | [424.54, 433.03] |
| p95_ttft_ms | 436.46 | [434.01, 438.9] |
| p99_ttft_ms | 446.91 | [388.06, 505.76] |
| mean_tpot_ms | 51.086 | [49.285, 52.888] |
| persistent_cache_mib | 9272.8 | [9272.8, 9272.8] |
| peak_cache_state_mib | 9386.8 | [9310.3, 9463.4] |
| peak_staging_mib | 114.07 | [37.535, 190.61] |
| peak_process_cuda_mib | 19126 | [19125, 19127] |
| compression_pending_peak | 2.5 | [-3.8531, 8.8531] |
| compression_enqueued | 576 | [576, 576] |
| compression_completed | 575 | [575, 575] |
| compression_completion_fraction | 0.99826 | [0.99826, 0.99826] |
| compression_completion_per_s | 3.8038 | [3.7314, 3.8763] |
| compression_committed | 575 | [575, 575] |
| compression_failed | 0 | [0, 0] |
| decompression_hits | 0 | [0, 0] |
