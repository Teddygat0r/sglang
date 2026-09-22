# Prefill deferral plus smaller SVD batches

Deferral enabled in all arms; caps 8, 2, 1. Five matched repetitions; reverse order on odd repetitions. Qwen3.5-4B rank16, concurrency4, full32/compressed298, ~14.19 GiB. No profiling or blocking prefill synchronization. See COMBINED_DEFER_BATCH.md.

## defer_batch1 (n=5)

| Metric | Mean | 95% CI |
|---|---:|---|
| requests | 192 | [192.0, 192.0] |
| duration_s | 290.366 | [289.94930014364445, 290.7827716263381] |
| token_cache_hit_rate | 0.648635 | [0.6422808877166863, 0.6549897841179391] |
| evicted_entries | 147 | [145.75833600179624, 148.24166399820376] |
| evicted_compressed_entries | 147 | [145.75833600179624, 148.24166399820376] |
| evicted_kv_tokens | 11934.4 | [9806.833902216053, 14061.966097783947] |
| recomputed_prefix_tokens | 5097.6 | [2579.4084314701845, 7615.791568529816] |
| request_throughput_rps | 0.661235 | [0.6602860961641406, 0.6621840379429573] |
| output_throughput_tps | 84.6381 | [84.51662030901, 84.75955685669854] |
| mean_ttft_ms | 489.469 | [482.12316045303555, 496.81437859778566] |
| p50_ttft_ms | 141.82 | [139.98113922778975, 143.65874439473737] |
| p95_ttft_ms | 1527.59 | [1522.1732124769992, 1533.00155459997] |
| p99_ttft_ms | 1544.6 | [1539.553088144333, 1549.6524479753068] |
| mean_tpot_ms | 43.7721 | [43.65520453310868, 43.888906665981644] |
| persistent_cache_mib | 13738.4 | [13738.375, 13738.375] |
| peak_cache_state_mib | 14278.4 | [14278.421875, 14278.421875] |
| peak_staging_mib | 540.047 | [540.046875, 540.046875] |
| peak_process_cuda_mib | 23769.5 | [23769.4560546875, 23769.4560546875] |
| compression_pending_peak | 11 | [11.0, 11.0] |
| compression_enqueued | 449 | [447.75833600179624, 450.24166399820376] |
| compression_completed | 445 | [443.75833600179624, 446.24166399820376] |
| compression_completion_fraction | 0.991091 | [0.9910666424765912, 0.9911159148819128] |
| compression_completion_per_s | 1.53255 | [1.526666804040306, 1.5384371481974703] |
| compression_committed | 445 | [443.75833600179624, 446.24166399820376] |
| compression_failed | 0 | [0.0, 0.0] |
| decompression_hits | 127 | [125.75833600179624, 128.24166399820376] |
| prefill_deferred_batches | 48 | [48.0, 48.0] |
| prefill_deferral_wait_ms | 27249.5 | [26763.31781251153, 27735.67411126772] |
| svd_batches | 446 | [444.75833600179624, 447.24166399820376] |
| svd_batch_items | 446 | [444.75833600179624, 447.24166399820376] |
| svd_batch_mean | 1 | [1.0, 1.0] |
| svd_batch_max | 1 | [1.0, 1.0] |

## defer_batch8 (n=5)

| Metric | Mean | 95% CI |
|---|---:|---|
| requests | 192 | [192.0, 192.0] |
| duration_s | 291.039 | [290.54833404538545, 291.53062132480227] |
| token_cache_hit_rate | 0.648635 | [0.6422808877166863, 0.6549897841179391] |
| evicted_entries | 147 | [145.75833600179624, 148.24166399820376] |
| evicted_compressed_entries | 147 | [145.75833600179624, 148.24166399820376] |
| evicted_kv_tokens | 11934.4 | [9806.833902216053, 14061.966097783947] |
| recomputed_prefix_tokens | 5097.6 | [2579.4084314701845, 7615.791568529816] |
| request_throughput_rps | 0.659705 | [0.6585908029821865, 0.6608197839157264] |
| output_throughput_tps | 84.4423 | [84.29962278171988, 84.58493234121298] |
| mean_ttft_ms | 493.644 | [488.0639849456919, 499.2234099197241] |
| p50_ttft_ms | 144.556 | [143.13997938067425, 145.97128727703583] |
| p95_ttft_ms | 1536.66 | [1533.6012158704139, 1539.7110830831193] |
| p99_ttft_ms | 1553.51 | [1543.3766559149763, 1563.6476088974932] |
| mean_tpot_ms | 43.8493 | [43.7466955442507, 43.95198869574034] |
| persistent_cache_mib | 13738.4 | [13738.375, 13738.375] |
| peak_cache_state_mib | 14314.4 | [14314.375, 14314.375] |
| peak_staging_mib | 576 | [576.0, 576.0] |
| peak_process_cuda_mib | 25923.4 | [25923.260048628326, 25923.476279496674] |
| compression_pending_peak | 11 | [11.0, 11.0] |
| compression_enqueued | 449 | [447.75833600179624, 450.24166399820376] |
| compression_completed | 445 | [443.75833600179624, 446.24166399820376] |
| compression_completion_fraction | 0.991091 | [0.9910666424765912, 0.9911159148819128] |
| compression_completion_per_s | 1.52901 | [1.5229705775816096, 1.5350422477308299] |
| compression_committed | 445 | [443.75833600179624, 446.24166399820376] |
| compression_failed | 0 | [0.0, 0.0] |
| decompression_hits | 127 | [125.75833600179624, 128.24166399820376] |
| prefill_deferred_batches | 48 | [48.0, 48.0] |
| prefill_deferral_wait_ms | 26062.6 | [25670.828966936868, 26454.342176624734] |
| svd_batches | 143.8 | [143.24471097896046, 144.35528902103957] |
| svd_batch_items | 446 | [444.75833600179624, 447.24166399820376] |
| svd_batch_mean | 3.10154 | [3.0914119201902728, 3.1116766578983053] |
| svd_batch_max | 8 | [8.0, 8.0] |

## defer_batch2 (n=5)

| Metric | Mean | 95% CI |
|---|---:|---|
| requests | 192 | [192.0, 192.0] |
| duration_s | 290.596 | [289.89010697050156, 291.302772125172] |
| token_cache_hit_rate | 0.648635 | [0.6422808877166863, 0.6549897841179391] |
| evicted_entries | 147 | [145.75833600179624, 148.24166399820376] |
| evicted_compressed_entries | 147 | [145.75833600179624, 148.24166399820376] |
| evicted_kv_tokens | 11934.4 | [9806.833902216053, 14061.966097783947] |
| recomputed_prefix_tokens | 5097.6 | [2579.4084314701845, 7615.791568529816] |
| request_throughput_rps | 0.660712 | [0.6591070560955479, 0.6623171745324281] |
| output_throughput_tps | 84.5712 | [84.36570318023013, 84.7765983401508] |
| mean_ttft_ms | 490.297 | [482.4839419995562, 498.10965786457035] |
| p50_ttft_ms | 144.129 | [141.93806440155717, 146.32029927749423] |
| p95_ttft_ms | 1528.98 | [1525.747457977663, 1532.209093820025] |
| p99_ttft_ms | 1542.45 | [1538.2478217318494, 1546.6431175932448] |
| mean_tpot_ms | 43.8033 | [43.73122024505263, 43.87528921252436] |
| persistent_cache_mib | 13738.4 | [13738.375, 13738.375] |
| peak_cache_state_mib | 14290.5 | [14290.46875, 14290.46875] |
| peak_staging_mib | 552.094 | [552.09375, 552.09375] |
| peak_process_cuda_mib | 24100.8 | [24100.644610874384, 24100.884686000616] |
| compression_pending_peak | 11 | [11.0, 11.0] |
| compression_enqueued | 449 | [447.75833600179624, 450.24166399820376] |
| compression_completed | 445 | [443.75833600179624, 446.24166399820376] |
| compression_completion_fraction | 0.991091 | [0.9910666424765912, 0.9911159148819128] |
| compression_completion_per_s | 1.53133 | [1.528993137980265, 1.5336737903268804] |
| compression_committed | 445 | [443.75833600179624, 446.24166399820376] |
| compression_failed | 0 | [0.0, 0.0] |
| decompression_hits | 127 | [125.75833600179624, 128.24166399820376] |
| prefill_deferred_batches | 48 | [48.0, 48.0] |
| prefill_deferral_wait_ms | 27008.6 | [26463.22322195968, 27554.032165495544] |
| svd_batches | 255.6 | [249.86948187172632, 261.33051812827364] |
| svd_batch_items | 446.4 | [445.7199126193417, 447.08008738065826] |
| svd_batch_mean | 1.74693 | [1.7079764194705596, 1.7858857587925479] |
| svd_batch_max | 2 | [2.0, 2.0] |

## Paired defer_batch2 minus defer_batch8

| Metric | Difference | 95% CI |
|---|---:|---|
| requests | 0 | [0.0, 0.0] |
| duration_s | -0.443038 | [-1.5204472497761903, 0.6343709752619918] |
| token_cache_hit_rate | 0 | [0.0, 0.0] |
| evicted_entries | 0 | [0.0, 0.0] |
| evicted_compressed_entries | 0 | [0.0, 0.0] |
| evicted_kv_tokens | 0 | [0.0, 0.0] |
| recomputed_prefix_tokens | 0 | [0.0, 0.0] |
| request_throughput_rps | 0.00100682 | [-0.0014396075133748187, 0.003453251243437938] |
| output_throughput_tps | 0.128873 | [-0.1842697617119768, 0.44201615916005604] |
| mean_ttft_ms | -3.3469 | [-6.557954855833058, -0.1358401454564464] |
| p50_ttft_ms | -0.426451 | [-3.637289137787454, 2.7843861591287777] |
| p95_ttft_ms | -7.67787 | [-11.189781164293183, -4.165965991551982] |
| p99_ttft_ms | -11.0667 | [-20.610994553326165, -1.522330934049096] |
| mean_tpot_ms | -0.0460874 | [-0.20015886646480546, 0.10798408405075444] |
| persistent_cache_mib | 0 | [0.0, 0.0] |
| peak_cache_state_mib | -23.9062 | [-23.90625, -23.90625] |
| peak_staging_mib | -23.9062 | [-23.90625, -23.90625] |
| peak_process_cuda_mib | -1822.6 | [-1822.7650642667948, -1822.4419669832052] |
| compression_pending_peak | 0 | [0.0, 0.0] |
| compression_enqueued | 0 | [0.0, 0.0] |
| compression_completed | 0 | [0.0, 0.0] |
| compression_completion_fraction | 0 | [0.0, 0.0] |
| compression_completion_per_s | 0.00232705 | [-0.0033438494942040512, 0.007997952488910172] |
| compression_committed | 0 | [0.0, 0.0] |
| compression_failed | 0 | [0.0, 0.0] |
| decompression_hits | 0 | [0.0, 0.0] |
| prefill_deferred_batches | 0 | [0.0, 0.0] |
| prefill_deferral_wait_ms | 946.042 | [105.88939335954456, 1786.1948505340788] |
| svd_batches | 111.8 | [106.28891382363, 117.31108617637] |
| svd_batch_items | 0.4 | [-0.28008738065825556, 1.0800873806582556] |
| svd_batch_mean | -1.35461 | [-1.3876436977912903, -1.3215827020341804] |
| svd_batch_max | -6 | [-6.0, -6.0] |

## Paired defer_batch1 minus defer_batch8

| Metric | Difference | 95% CI |
|---|---:|---|
| requests | 0 | [0.0, 0.0] |
| duration_s | -0.673442 | [-1.2200717114858777, -0.12681188871930538] |
| token_cache_hit_rate | 0 | [0.0, 0.0] |
| evicted_entries | 0 | [0.0, 0.0] |
| evicted_compressed_entries | 0 | [0.0, 0.0] |
| evicted_kv_tokens | 0 | [0.0, 0.0] |
| recomputed_prefix_tokens | 0 | [0.0, 0.0] |
| request_throughput_rps | 0.00152977 | [0.00028876616350164384, 0.0027707810456834435] |
| output_throughput_tps | 0.195811 | [0.03696206892821041, 0.35465997384748077] |
| mean_ttft_ms | -4.17493 | [-7.5934186933278625, -0.7564371212668894] |
| p50_ttft_ms | -2.73569 | [-4.943203739629453, -0.5281792955535001] |
| p95_ttft_ms | -9.06877 | [-16.119155822108276, -2.01837605445575] |
| p99_ttft_ms | -8.90936 | [-19.574273927565645, 1.7555452347360365] |
| mean_tpot_ms | -0.0772865 | [-0.16572480411206106, 0.011151763211349003] |
| persistent_cache_mib | 0 | [0.0, 0.0] |
| peak_cache_state_mib | -35.9531 | [-35.953125, -35.953125] |
| peak_staging_mib | -35.9531 | [-35.953125, -35.953125] |
| peak_process_cuda_mib | -2153.91 | [-2154.020224809174, -2153.803993940826] |
| compression_pending_peak | 0 | [0.0, 0.0] |
| compression_enqueued | 0 | [0.0, 0.0] |
| compression_completed | 0 | [0.0, 0.0] |
| compression_completion_fraction | 0 | [0.0, 0.0] |
| compression_completion_per_s | 0.00354556 | [0.0006707282956742531, 0.0064203986296627906] |
| compression_committed | 0 | [0.0, 0.0] |
| compression_failed | 0 | [0.0, 0.0] |
| decompression_hits | 0 | [0.0, 0.0] |
| prefill_deferred_batches | 0 | [0.0, 0.0] |
| prefill_deferral_wait_ms | 1186.91 | [373.00611602294543, 2000.8146641947021] |
| svd_batches | 302.2 | [301.1611493663164, 303.23885063368357] |
| svd_batch_items | 0 | [0.0, 0.0] |
| svd_batch_mean | -2.10154 | [-2.1116766578983053, -2.0914119201902728] |
| svd_batch_max | -7 | [-7.0, -7.0] |
