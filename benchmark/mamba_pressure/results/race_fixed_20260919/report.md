# Race-fixed cache validation

Qwen3.5-4B, rank 16, concurrency 4, fixed ~14.19 GiB cache ceiling. 64 shared prefixes × 3 passes, 2048 prefix + 16 suffix tokens, 128 output tokens. LRU; graphs and overlap disabled. Five repetitions in order 1,0,2,3,4; allocation order alternates by execution order. Failed seed replayed first. Three performance optimizations remain reverted.

Ceiling includes KV and dense/compressed state pools plus staging; excludes model weights, arithmetic workspace and allocator reserve. Staging reservation is not runtime enforcement. Fresh race-fix validation, not pooled with earlier runs. No unsafe-code baseline or quality evaluation.

## fixed_f16 (n=5)

| Metric | Mean | 95% CI |
|---|---:|---|
| requests | 192 | [192.0, 192.0] |
| duration_s | 291.178 | [289.93003160938986, 292.4251206794345] |
| token_cache_hit_rate | 0.653747 | [0.6537467700258398, 0.6537467700258398] |
| evicted_entries | 86 | [86.0, 86.0] |
| evicted_compressed_entries | 63.8 | [60.35452223484866, 67.24547776515134] |
| evicted_kv_tokens | 7223.8 | [6967.14411696226, 7480.45588303774] |
| recomputed_prefix_tokens | 3072 | [3072.0, 3072.0] |
| request_throughput_rps | 0.659398 | [0.6565692811481184, 0.6622261667649133] |
| output_throughput_tps | 84.4029 | [84.04086798695916, 84.7649493459089] |
| mean_ttft_ms | 513.952 | [511.41078613793536, 516.4936652870966] |
| p50_ttft_ms | 166.362 | [164.97025476550832, 167.75330467367397] |
| p95_ttft_ms | 1587.69 | [1580.3397502945777, 1595.0427470533136] |
| p99_ttft_ms | 1601.37 | [1596.4927873227382, 1606.2419718440985] |
| mean_tpot_ms | 43.7113 | [43.50558049950619, 43.91703382699027] |
| persistent_cache_mib | 13742.7 | [13742.6875, 13742.6875] |
| peak_cache_state_mib | 14066.8 | [14017.427970947172, 14116.228279052828] |
| peak_staging_mib | 324.141 | [274.7404709471712, 373.5407790528288] |
| peak_process_cuda_mib | 24417.8 | [24416.68585252915, 24418.82547559585] |
| compression_pending_peak | 6.8 | [5.761149366316432, 7.838850633683568] |
| compression_enqueued | 448 | [448.0, 448.0] |
| compression_completed | 443.8 | [443.2447109789604, 444.3552890210396] |
| compression_completion_fraction | 0.990625 | [0.9893855155780367, 0.9918644844219633] |
| compression_completion_per_s | 1.52417 | [1.5175769357378803, 1.5307627866987172] |
| compression_committed | 421.8 | [418.3545222348487, 425.24547776515135] |
| compression_failed | 0 | [0.0, 0.0] |
| decompression_hits | 128 | [128.0, 128.0] |

## fixed_f32 (n=5)

| Metric | Mean | 95% CI |
|---|---:|---|
| requests | 192 | [192.0, 192.0] |
| duration_s | 292.047 | [291.39147051058484, 292.7030959195666] |
| token_cache_hit_rate | 0.648563 | [0.6420952146122618, 0.6550301083851542] |
| evicted_entries | 147 | [145.75833600179624, 148.24166399820376] |
| evicted_compressed_entries | 147 | [145.75833600179624, 148.24166399820376] |
| evicted_kv_tokens | 11131.2 | [9755.918696033556, 12506.481303966446] |
| recomputed_prefix_tokens | 5126.4 | [2563.4284082640393, 7689.3715917359605] |
| request_throughput_rps | 0.65743 | [0.6559512558981758, 0.6589077692311628] |
| output_throughput_tps | 84.151 | [83.9617607549665, 84.34019446158884] |
| mean_ttft_ms | 520.145 | [512.2034454646838, 528.0857602438572] |
| p50_ttft_ms | 166.558 | [165.81675936484305, 167.2987177500728] |
| p95_ttft_ms | 1590.53 | [1584.5975819299506, 1596.4594486658884] |
| p99_ttft_ms | 1605.1 | [1600.2659706995598, 1609.9253012388237] |
| mean_tpot_ms | 43.8056 | [43.73741594107584, 43.873789439222904] |
| persistent_cache_mib | 13738.4 | [13738.375, 13738.375] |
| peak_cache_state_mib | 14050.5 | [14004.647287364201, 14096.4402126358] |
| peak_staging_mib | 312.169 | [266.27228736420045, 358.0652126357995] |
| peak_process_cuda_mib | 24508.5 | [24315.92388635217, 24700.98451208533] |
| compression_pending_peak | 7.6 | [6.919912619341744, 8.280087380658255] |
| compression_enqueued | 449 | [447.75833600179624, 450.24166399820376] |
| compression_completed | 445 | [443.75833600179624, 446.24166399820376] |
| compression_completion_fraction | 0.991091 | [0.9910666424765912, 0.9911159148819128] |
| compression_completion_per_s | 1.52373 | [1.5215571275393336, 1.5258940449295546] |
| compression_committed | 445 | [443.75833600179624, 446.24166399820376] |
| compression_failed | 0 | [0.0, 0.0] |
| decompression_hits | 127 | [125.75833600179624, 128.24166399820376] |
