# Cache hot-path optimization comparison

Qwen3.5-4B, rank 16, concurrency 4, fixed ~14.19 GiB cache ceiling. 64 shared prefixes × 3 passes, 2048 prefix + 16 suffix tokens, 128 output tokens. LRU; graphs and overlap disabled. Five repetitions, reversed configuration order on odd repetitions.

Ceiling includes KV and dense/compressed state pools plus staging; excludes model weights, arithmetic workspace and allocator reserve. Staging reservation is not runtime enforcement. Frozen pre-change methods versus three optimizations together; not a quality evaluation or individual ablation.

## f16_after (n=1)

| Metric | Mean | 95% CI |
|---|---:|---|
| requests | 192 | None |
| duration_s | 290.806 | None |
| token_cache_hit_rate | 0.653747 | None |
| evicted_entries | 86 | None |
| evicted_compressed_entries | 61 | None |
| evicted_kv_tokens | 7322 | None |
| recomputed_prefix_tokens | 3072 | None |
| request_throughput_rps | 0.660233 | None |
| output_throughput_tps | 84.5099 | None |
| mean_ttft_ms | 512.15 | None |
| p50_ttft_ms | 161.183 | None |
| p95_ttft_ms | 1592.51 | None |
| p99_ttft_ms | 1606.1 | None |
| mean_tpot_ms | 43.6649 | None |
| persistent_cache_mib | 13742.7 | None |
| peak_cache_state_mib | 14066.8 | None |
| peak_staging_mib | 324.141 | None |
| peak_process_cuda_mib | 24417.7 | None |
| compression_pending_peak | 8 | None |
| compression_enqueued | 448 | None |
| compression_completed | 444 | None |
| compression_completion_fraction | 0.991071 | None |
| compression_completion_per_s | 1.52679 | None |
| compression_committed | 419 | None |
| compression_failed | 0 | None |
| decompression_hits | 128 | None |

## f16_before (n=1)

| Metric | Mean | 95% CI |
|---|---:|---|
| requests | 192 | None |
| duration_s | 291.111 | None |
| token_cache_hit_rate | 0.653747 | None |
| evicted_entries | 86 | None |
| evicted_compressed_entries | 64 | None |
| evicted_kv_tokens | 7370 | None |
| recomputed_prefix_tokens | 3072 | None |
| request_throughput_rps | 0.659541 | None |
| output_throughput_tps | 84.4213 | None |
| mean_ttft_ms | 512.917 | None |
| p50_ttft_ms | 165.684 | None |
| p95_ttft_ms | 1584.89 | None |
| p99_ttft_ms | 1595.3 | None |
| mean_tpot_ms | 43.7092 | None |
| persistent_cache_mib | 13742.7 | None |
| peak_cache_state_mib | 14030.9 | None |
| peak_staging_mib | 288.188 | None |
| peak_process_cuda_mib | 24417.4 | None |
| compression_pending_peak | 8 | None |
| compression_enqueued | 448 | None |
| compression_completed | 444 | None |
| compression_completion_fraction | 0.991071 | None |
| compression_completion_per_s | 1.52519 | None |
| compression_committed | 422 | None |
| compression_failed | 0 | None |
| decompression_hits | 128 | None |

## f32_after (n=2)

| Metric | Mean | 95% CI |
|---|---:|---|
| requests | 192 | [192.0, 192.0] |
| duration_s | 291.718 | [290.52478350664893, 292.9109993301012] |
| token_cache_hit_rate | 0.648518 | [0.5820835952303524, 0.7149529034776579] |
| evicted_entries | 147 | [134.2937952638253, 159.7062047361747] |
| evicted_compressed_entries | 147 | [134.2937952638253, 159.7062047361747] |
| evicted_kv_tokens | 10336 | [9827.751810553013, 10844.248189446987] |
| recomputed_prefix_tokens | 5144 | [-21183.256213353965, 31471.256213353965] |
| request_throughput_rps | 0.65817 | [0.6554783146929328, 0.660862064428742] |
| output_throughput_tps | 84.2458 | [83.9012242806954, 84.59034424687897] |
| mean_ttft_ms | 516.905 | [425.72561019225884, 608.0839640898483] |
| p50_ttft_ms | 162.613 | [154.32552433337727, 170.90052926216563] |
| p95_ttft_ms | 1586.94 | [1534.4136208149807, 1639.472795851229] |
| p99_ttft_ms | 1601.1 | [1509.0642914047082, 1693.1303305680315] |
| mean_tpot_ms | 43.7766 | [42.86347117235067, 44.6897415308122] |
| persistent_cache_mib | 13738.4 | [13738.375, 13738.375] |
| peak_cache_state_mib | 14068.5 | [13535.176265254167, 14601.901859745833] |
| peak_staging_mib | 330.164 | [-203.198734745833, 863.526859745833] |
| peak_process_cuda_mib | 24416.3 | [24395.522681329214, 24437.016381170786] |
| compression_pending_peak | 8 | [8.0, 8.0] |
| compression_enqueued | 449 | [436.2937952638253, 461.7062047361747] |
| compression_completed | 445 | [432.2937952638253, 457.7062047361747] |
| compression_completion_fraction | 0.991091 | [0.9908391626044409, 0.9913433770780989] |
| compression_completion_per_s | 1.52545 | [1.4756521703838261, 1.5752431026340832] |
| compression_committed | 445 | [432.2937952638253, 457.7062047361747] |
| compression_failed | 0 | [0.0, 0.0] |
| decompression_hits | 127 | [114.2937952638253, 139.7062047361747] |

## f32_before (n=2)

| Metric | Mean | 95% CI |
|---|---:|---|
| requests | 192 | [192.0, 192.0] |
| duration_s | 292.061 | [287.2919011159971, 296.8303166732952] |
| token_cache_hit_rate | 0.651102 | [0.6175001449362305, 0.6847043124281107] |
| evicted_entries | 146.5 | [140.14689763191265, 152.85310236808735] |
| evicted_compressed_entries | 146.5 | [140.14689763191265, 152.85310236808735] |
| evicted_kv_tokens | 10320 | [10015.051086331807, 10624.948913668193] |
| recomputed_prefix_tokens | 4120 | [-9196.102563511076, 17436.10256351108] |
| request_throughput_rps | 0.657398 | [0.646662786925248, 0.6681327217308343] |
| output_throughput_tps | 84.1469 | [82.77283672643175, 85.52098838154679] |
| mean_ttft_ms | 514.956 | [504.8697634578243, 525.0421221525694] |
| p50_ttft_ms | 167.441 | [154.40667576086304, 180.4753321509383] |
| p95_ttft_ms | 1587.97 | [1586.825067550168, 1589.116186182722] |
| p99_ttft_ms | 1602.89 | [1534.0160597482782, 1671.756168343677] |
| mean_tpot_ms | 43.8499 | [42.99130903804418, 44.708525843792046] |
| persistent_cache_mib | 13738.4 | [13738.375, 13738.375] |
| peak_cache_state_mib | 14026.6 | [14026.5625, 14026.5625] |
| peak_staging_mib | 288.188 | [288.1875, 288.1875] |
| peak_process_cuda_mib | 24414.8 | [24414.44503137273, 24415.09647253352] |
| compression_pending_peak | 7.5 | [1.146897631912653, 13.853102368087347] |
| compression_enqueued | 448.5 | [442.1468976319127, 454.8531023680873] |
| compression_completed | 444.5 | [438.1468976319127, 450.8531023680873] |
| compression_completion_fraction | 0.991081 | [0.9909550369398645, 0.9912077056627444] |
| compression_completion_per_s | 1.52195 | [1.475341229107315, 1.5685517316108408] |
| compression_committed | 444.5 | [438.1468976319127, 450.8531023680873] |
| compression_failed | 0 | [0.0, 0.0] |
| decompression_hits | 127.5 | [121.14689763191265, 133.85310236808735] |

## Paired f16_after minus f16_before (n=1)

| Metric | Mean difference | 95% CI |
|---|---:|---|
| requests | 0 | None |
| duration_s | -0.305223 | None |
| token_cache_hit_rate | 0 | None |
| evicted_entries | 0 | None |
| evicted_compressed_entries | -3 | None |
| evicted_kv_tokens | -48 | None |
| recomputed_prefix_tokens | 0 | None |
| request_throughput_rps | 0.000692238 | None |
| output_throughput_tps | 0.0886065 | None |
| mean_ttft_ms | -0.766828 | None |
| p50_ttft_ms | -4.50007 | None |
| p95_ttft_ms | 7.61386 | None |
| p99_ttft_ms | 10.8022 | None |
| mean_tpot_ms | -0.044296 | None |
| persistent_cache_mib | 0 | None |
| peak_cache_state_mib | 35.9531 | None |
| peak_staging_mib | 35.9531 | None |
| peak_process_cuda_mib | 0.27832 | None |
| compression_pending_peak | 0 | None |
| compression_enqueued | 0 | None |
| compression_completed | 0 | None |
| compression_completion_fraction | 0 | None |
| compression_completion_per_s | 0.0016008 | None |
| compression_committed | -3 | None |
| compression_failed | 0 | None |
| decompression_hits | 0 | None |

## Paired f32_after minus f32_before (n=2)

| Metric | Mean difference | 95% CI |
|---|---:|---|
| requests | 0 | [0.0, 0.0] |
| duration_s | -0.343217 | [-3.919317343193995, 3.232882390651809] |
| token_cache_hit_rate | -0.00258398 | [-0.03541654970587808, 0.03024859104954728] |
| evicted_entries | 0.5 | [-5.853102368087347, 6.853102368087347] |
| evicted_compressed_entries | 0.5 | [-5.853102368087347, 6.853102368087347] |
| evicted_kv_tokens | 16 | [-187.2992757787951, 219.2992757787951] |
| recomputed_prefix_tokens | 1024 | [-11987.153649842887, 14035.153649842887] |
| request_throughput_rps | 0.000772435 | [-0.007270657302092271, 0.008815527767684645] |
| output_throughput_tps | 0.0988717 | [-0.9306441346678107, 1.1283875542636346] |
| mean_ttft_ms | 1.94884 | [-79.14415326556542, 83.04184193727906] |
| p50_ttft_ms | -4.82798 | [-9.574802888772663, -0.08115142748576876] |
| p95_ttft_ms | -1.02742 | [-54.70256536774118, 52.64772830106099] |
| p99_ttft_ms | -1.7888 | [-162.69187693896882, 159.11427081975333] |
| mean_tpot_ms | -0.0733111 | [-0.12783786569350802, -0.018784312979846132] |
| persistent_cache_mib | 0 | [0.0, 0.0] |
| peak_cache_state_mib | 41.9766 | [-491.386234745833, 575.339359745833] |
| peak_staging_mib | 41.9766 | [-491.386234745833, 575.339359745833] |
| peak_process_cuda_mib | 1.49878 | [-19.573791204305344, 22.571349798055344] |
| compression_pending_peak | 0.5 | [-5.853102368087347, 6.853102368087347] |
| compression_enqueued | 0.5 | [-5.853102368087347, 6.853102368087347] |
| compression_completed | 0.5 | [-5.853102368087347, 6.853102368087347] |
| compression_completion_fraction | 9.89854e-06 | [-0.00011587433542369286, 0.00013567141535440963] |
| compression_completion_per_s | 0.00350116 | [0.0003109412765111767, 0.006691371023242167] |
| compression_committed | 0.5 | [-5.853102368087347, 6.853102368087347] |
| compression_failed | 0 | [0.0, 0.0] |
| decompression_hits | -0.5 | [-6.853102368087347, 5.853102368087347] |
