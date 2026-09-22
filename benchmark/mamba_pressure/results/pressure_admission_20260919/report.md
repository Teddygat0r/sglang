# Pressure-triggered compression experiment

Qwen3.5-4B, rank 16, concurrency 4, fixed ~14.19 GiB cache ceiling. 64 shared prefixes × 3 passes, 2048 prefix + 16 suffix tokens, 128 output tokens. LRU; graphs and overlap disabled. Five repetitions; reverse policy order on odd repetitions. Compare eager admission with low/high free-slot watermarks 8/12. Race fixes retained.

Ceiling includes KV and dense/compressed state pools plus staging; excludes model weights, arithmetic workspace and allocator reserve. Staging reservation is not runtime enforcement. Fresh eager baseline versus pressure-triggered admission; no micro-optimizations. No quality evaluation; no pooling with earlier studies.

## baseline (n=5)

| Metric | Mean | 95% CI |
|---|---:|---|
| requests | 192 | [192.0, 192.0] |
| duration_s | 291.283 | [290.8729771925475, 291.69362951681353] |
| token_cache_hit_rate | 0.653747 | [0.6537467700258398, 0.6537467700258398] |
| evicted_entries | 86 | [86.0, 86.0] |
| evicted_compressed_entries | 65.2 | [61.332859454451786, 69.06714054554823] |
| evicted_kv_tokens | 7103.2 | [6877.593717551342, 7328.806282448658] |
| recomputed_prefix_tokens | 3072 | [3072.0, 3072.0] |
| request_throughput_rps | 0.659153 | [0.658224666606378, 0.6600808826580242] |
| output_throughput_tps | 84.3716 | [84.25275732561639, 84.4903529802271] |
| mean_ttft_ms | 515.261 | [512.8579972986794, 517.6646881643952] |
| p50_ttft_ms | 166.305 | [165.08169529269355, 167.52913909723145] |
| p95_ttft_ms | 1594.89 | [1586.2342022610662, 1603.54325180223] |
| p99_ttft_ms | 1607.32 | [1596.4537492211978, 1618.1871172521194] |
| mean_tpot_ms | 43.7181 | [43.655382605925396, 43.780835993752234] |
| persistent_cache_mib | 13742.7 | [13742.6875, 13742.6875] |
| peak_cache_state_mib | 14059.6 | [14027.045822112277, 14092.079177887723] |
| peak_staging_mib | 316.875 | [284.35832211227716, 349.39167788772284] |
| peak_process_cuda_mib | 24427.5 | [24400.59856638261, 24454.38190236739] |
| compression_pending_peak | 6.8 | [6.244710978960441, 7.3552890210395585] |
| compression_enqueued | 448 | [448.0, 448.0] |
| compression_completed | 443.4 | [441.7341329368813, 445.0658670631187] |
| compression_completion_fraction | 0.989732 | [0.9860136895912529, 0.9934505961230328] |
| compression_completion_per_s | 1.52223 | [1.5152413037384167, 1.529224537489764] |
| compression_committed | 423.2 | [419.33285945445175, 427.0671405455482] |
| compression_failed | 0 | [0.0, 0.0] |
| decompression_hits | 128 | [128.0, 128.0] |
| admission_episodes | 0 | [0.0, 0.0] |
| admission_jobs | 0 | [0.0, 0.0] |

## pressure_8_12 (n=5)

| Metric | Mean | 95% CI |
|---|---:|---|
| requests | 192 | [192.0, 192.0] |
| duration_s | 291.043 | [290.75878102939174, 291.3277314058359] |
| token_cache_hit_rate | 0.653747 | [0.6537467700258398, 0.6537467700258398] |
| evicted_entries | 86 | [86.0, 86.0] |
| evicted_compressed_entries | 26 | [22.834365521916684, 29.165634478083316] |
| evicted_kv_tokens | 9259.8 | [9019.207934796214, 9500.392065203785] |
| recomputed_prefix_tokens | 3072 | [3072.0, 3072.0] |
| request_throughput_rps | 0.659696 | [0.6590509586542252, 0.6603412012868775] |
| output_throughput_tps | 84.4411 | [84.35852270774083, 84.52367376472031] |
| mean_ttft_ms | 497.724 | [495.2553388609927, 500.19176795364007] |
| p50_ttft_ms | 143.26 | [142.06999473684985, 144.45090621298593] |
| p95_ttft_ms | 1606.83 | [1603.3811941269103, 1610.269041444057] |
| p99_ttft_ms | 1615.59 | [1612.427537696288, 1618.7513688934584] |
| mean_tpot_ms | 43.8168 | [43.78052232699319, 43.85304236108391] |
| persistent_cache_mib | 13742.7 | [13742.6875, 13742.6875] |
| peak_cache_state_mib | 14167.6 | [14121.07284621275, 14214.14590378725] |
| peak_staging_mib | 424.922 | [378.38534621274954, 471.45840378725046] |
| peak_process_cuda_mib | 25050.9 | [24815.873897043333, 25286.00754826917] |
| compression_pending_peak | 9 | [9.0, 9.0] |
| compression_enqueued | 384.8 | [381.03384662330313, 388.5661533766969] |
| compression_completed | 384.8 | [381.03384662330313, 388.5661533766969] |
| compression_completion_fraction | 1 | [1.0, 1.0] |
| compression_completion_per_s | 1.32214 | [1.3088700324044376, 1.3354145796295338] |
| compression_committed | 384 | [380.8343655219167, 387.1656344780833] |
| compression_failed | 0 | [0.0, 0.0] |
| decompression_hits | 128 | [128.0, 128.0] |
| admission_episodes | 204.6 | [201.74147480901297, 207.45852519098702] |
| admission_jobs | 384.8 | [381.03384662330313, 388.5661533766969] |

## Paired pressure_8_12 minus baseline (n=5)

| Metric | Mean difference | 95% CI |
|---|---:|---|
| requests | 0 | [0.0, 0.0] |
| duration_s | -0.240047 | [-0.642419619763964, 0.1623253456305202] |
| token_cache_hit_rate | 0 | [0.0, 0.0] |
| evicted_entries | 0 | [0.0, 0.0] |
| evicted_compressed_entries | -39.2 | [-44.71108617637, -33.68891382363] |
| evicted_kv_tokens | 2156.6 | [1835.4400604728926, 2477.7599395271072] |
| recomputed_prefix_tokens | 0 | [0.0, 0.0] |
| request_throughput_rps | 0.000543305 | [-0.0003670481491995708, 0.0014536588258998989] |
| output_throughput_tps | 0.0695431 | [-0.04698216309754506, 0.18606832971518705] |
| mean_ttft_ms | -17.5378 | [-19.047618110426935, -16.027960538014884] |
| p50_ttft_ms | -23.045 | [-24.304450253420463, -21.785483186668763] |
| p95_ttft_ms | 11.9364 | [5.180266512444814, 18.692514995226066] |
| p99_ttft_ms | 8.26902 | [-1.268236349302022, 17.80627646573135] |
| mean_tpot_ms | 0.098673 | [0.03683487530313372, 0.16051121309634522] |
| persistent_cache_mib | 0 | [0.0, 0.0] |
| peak_cache_state_mib | 108.047 | [58.716991179305126, 157.37675882069487] |
| peak_staging_mib | 108.047 | [58.716991179305126, 157.37675882069487] |
| peak_process_cuda_mib | 623.45 | [371.0773407591061, 875.8236358033939] |
| compression_pending_peak | 2.2 | [1.6447109789604415, 2.755289021039559] |
| compression_enqueued | -63.2 | [-66.96615337669688, -59.433846623303126] |
| compression_completed | -58.6 | [-61.02044872716907, -56.179551272830935] |
| compression_completion_fraction | 0.0102679 | [0.006549403876967197, 0.013986310408747036] |
| compression_completion_per_s | -0.200091 | [-0.20804414627290163, -0.19213708292130766] |
| compression_committed | -39.2 | [-44.71108617637, -33.68891382363] |
| compression_failed | 0 | [0.0, 0.0] |
| decompression_hits | 0 | [0.0, 0.0] |
| admission_episodes | 204.6 | [201.74147480901297, 207.45852519098702] |
| admission_jobs | 384.8 | [381.03384662330313, 388.5661533766969] |
