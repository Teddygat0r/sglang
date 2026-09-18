# Live matched-memory cache-pressure experiment

Each interval is a two-sided Student-t 95% confidence interval across independent run repetitions.

## budget24_off (n=1)

| Metric | Mean | 95% CI |
|---|---:|---|
| requests | 128 | not estimable |
| duration_s | 98.259 | not estimable |
| token_cache_hit_rate | 0.0077519 | not estimable |
| evicted_entries | 361 | not estimable |
| evicted_compressed_entries | 0 | not estimable |
| evicted_kv_tokens | 2.4442e+05 | not estimable |
| recomputed_prefix_tokens | 1.2902e+05 | not estimable |
| request_throughput_rps | 1.3027 | not estimable |
| output_throughput_tps | 10.421 | not estimable |
| mean_ttft_ms | 416.26 | not estimable |
| p50_ttft_ms | 418.69 | not estimable |
| p95_ttft_ms | 427.43 | not estimable |
| p99_ttft_ms | 431.76 | not estimable |
| mean_tpot_ms | 50.158 | not estimable |
| persistent_cache_mib | 9420.2 | not estimable |
| peak_cache_state_mib | 9420.2 | not estimable |
| peak_staging_mib | 0 | not estimable |
| peak_process_cuda_mib | 18886 | not estimable |
| compression_pending_peak | 0 | not estimable |
| compression_enqueued | 0 | not estimable |
| compression_completed | 0 | not estimable |
| compression_completion_fraction | 0 | not estimable |
| compression_completion_per_s | 0 | not estimable |
| compression_committed | 0 | not estimable |
| compression_failed | 0 | not estimable |
| decompression_hits | 0 | not estimable |

## budget24_on (n=1)

| Metric | Mean | 95% CI |
|---|---:|---|
| requests | 128 | not estimable |
| duration_s | 100.86 | not estimable |
| token_cache_hit_rate | 0 | not estimable |
| evicted_entries | 374 | not estimable |
| evicted_compressed_entries | 374 | not estimable |
| evicted_kv_tokens | 2.568e+05 | not estimable |
| recomputed_prefix_tokens | 1.3107e+05 | not estimable |
| request_throughput_rps | 1.2691 | not estimable |
| output_throughput_tps | 10.153 | not estimable |
| mean_ttft_ms | 430.35 | not estimable |
| p50_ttft_ms | 429.71 | not estimable |
| p95_ttft_ms | 436.12 | not estimable |
| p99_ttft_ms | 447.86 | not estimable |
| mean_tpot_ms | 51.066 | not estimable |
| persistent_cache_mib | 9272.8 | not estimable |
| peak_cache_state_mib | 9380.8 | not estimable |
| peak_staging_mib | 108.05 | not estimable |
| peak_process_cuda_mib | 19126 | not estimable |
| compression_pending_peak | 2 | not estimable |
| compression_enqueued | 384 | not estimable |
| compression_completed | 383 | not estimable |
| compression_completion_fraction | 0.9974 | not estimable |
| compression_completion_per_s | 3.7973 | not estimable |
| compression_committed | 383 | not estimable |
| compression_failed | 0 | not estimable |
| decompression_hits | 0 | not estimable |
