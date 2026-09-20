# Nonblocking prefill-aware compression

Batch cap 8 in both arms. Qwen3.5-4B rank16, concurrency4, full32/compressed298, ~14.19 GiB. Five paired repetitions, alternate order, no profiling or prefill synchronization. See DEFER_PREFILL.md.

## defer (n=5)

| Metric | Mean | 95% CI |
|---|---:|---|
| requests | 192 | [192.0, 192.0] |
| duration_s | 292.163 | [291.7208933691385, 292.60476708832357] |
| token_cache_hit_rate | 0.647569 | [0.6405557410621726, 0.6545831478267162] |
| evicted_entries | 147.2 | [145.8398252386835, 148.5601747613165] |
| evicted_compressed_entries | 147.2 | [145.8398252386835, 148.5601747613165] |
| evicted_kv_tokens | 11115.2 | [8840.736169821968, 13389.663830178033] |
| recomputed_prefix_tokens | 5520 | [2740.553514046288, 8299.446485953711] |
| request_throughput_rps | 0.657169 | [0.6561748687614084, 0.6581622611474391] |
| output_throughput_tps | 84.1176 | [83.99038320146028, 84.24476942687221] |
| mean_ttft_ms | 497.219 | [490.75769907198674, 503.6797789838803] |
| p50_ttft_ms | 146.434 | [143.64784879564087, 149.22066183806618] |
| p95_ttft_ms | 1541.34 | [1533.1037884146629, 1549.5824449940862] |
| p99_ttft_ms | 1559.58 | [1547.7041213501786, 1571.450519048741] |
| mean_tpot_ms | 44.0054 | [43.94499735464572, 44.06584612536625] |
| persistent_cache_mib | 13738.4 | [13738.375, 13738.375] |
| peak_cache_state_mib | 14309.6 | [14301.565838720006, 14317.696661279995] |
| peak_staging_mib | 571.256 | [563.190838720006, 579.321661279994] |
| peak_process_cuda_mib | 25923.6 | [25923.25080261727, 25923.874783320232] |
| compression_pending_peak | 11 | [11.0, 11.0] |
| compression_enqueued | 449.2 | [447.8398252386835, 450.5601747613165] |
| compression_completed | 445.2 | [443.8398252386835, 446.5601747613165] |
| compression_completion_fraction | 0.991095 | [0.9910683253883382, 0.9911221510372951] |
| compression_completion_per_s | 1.52381 | [1.5204631886838442, 1.527150657499363] |
| compression_committed | 445.2 | [443.8398252386835, 446.5601747613165] |
| compression_failed | 0 | [0.0, 0.0] |
| decompression_hits | 126.8 | [125.43982523868348, 128.1601747613165] |
| prefill_deferred_batches | 48 | [48.0, 48.0] |
| prefill_deferral_wait_ms | 26858.2 | [26247.828431482565, 27468.534754906643] |
| svd_batches | 144 | [144.0, 144.0] |
| svd_batch_items | 447.2 | [445.1597378580252, 449.24026214197477] |
| svd_batch_mean | 3.10556 | [3.0913870684585087, 3.1197240426526025] |
| svd_batch_max | 8 | [8.0, 8.0] |

## baseline (n=5)

| Metric | Mean | 95% CI |
|---|---:|---|
| requests | 192 | [192.0, 192.0] |
| duration_s | 291.007 | [289.8479484455318, 292.1656791871755] |
| token_cache_hit_rate | 0.647553 | [0.6405665533341776, 0.6545400358131093] |
| evicted_entries | 147.2 | [145.8398252386835, 148.5601747613165] |
| evicted_compressed_entries | 147.2 | [145.8398252386835, 148.5601747613165] |
| evicted_kv_tokens | 11150.4 | [8844.647406898881, 13456.152593101118] |
| recomputed_prefix_tokens | 5526.4 | [2757.6382876945663, 8295.161712305433] |
| request_throughput_rps | 0.659784 | [0.6571580135049777, 0.6624095694036951] |
| output_throughput_tps | 84.4523 | [84.11622572863715, 84.78842488367297] |
| mean_ttft_ms | 523.154 | [516.0367983265032, 530.270358490553] |
| p50_ttft_ms | 166.42 | [165.1543901293984, 167.68654873674006] |
| p95_ttft_ms | 1601.44 | [1594.8638020689637, 1608.0080749233216] |
| p99_ttft_ms | 1615.34 | [1610.2438842945342, 1620.4327957711573] |
| mean_tpot_ms | 43.6119 | [43.47172758348328, 43.752003062553875] |
| persistent_cache_mib | 13738.4 | [13738.375, 13738.375] |
| peak_cache_state_mib | 14060.1 | [14035.669008120682, 14084.54349187932] |
| peak_staging_mib | 321.731 | [297.2940081206815, 346.1684918793185] |
| peak_process_cuda_mib | 24415.5 | [24414.11917295789, 24416.95797547961] |
| compression_pending_peak | 7.6 | [6.919912619341744, 8.280087380658255] |
| compression_enqueued | 449.2 | [447.8398252386835, 450.5601747613165] |
| compression_completed | 445.2 | [443.8398252386835, 446.5601747613165] |
| compression_completion_fraction | 0.991095 | [0.9910683253883382, 0.9911221510372951] |
| compression_completion_per_s | 1.52986 | [1.527599881670169, 1.5321292117214804] |
| compression_committed | 445.2 | [443.8398252386835, 446.5601747613165] |
| compression_failed | 0 | [0.0, 0.0] |
| decompression_hits | 126.8 | [125.43982523868348, 128.1601747613165] |
| prefill_deferred_batches | 0 | [0.0, 0.0] |
| prefill_deferral_wait_ms | 0 | [0.0, 0.0] |
| svd_batches | 302.2 | [292.80921054917627, 311.5907894508237] |
| svd_batch_items | 446.6 | [444.9341329368813, 448.2658670631187] |
| svd_batch_mean | 1.47865 | [1.42743801540129, 1.529867103850651] |
| svd_batch_max | 4 | [4.0, 4.0] |

## Paired defer minus baseline

| Metric | Difference | 95% CI |
|---|---:|---|
| requests | 0 | [0.0, 0.0] |
| duration_s | 1.15602 | [0.35427062302604795, 1.957762201728667] |
| token_cache_hit_rate | 1.61499e-05 | [-5.9878742262116495e-05, 9.217848386417634e-05] |
| evicted_entries | 0 | [0.0, 0.0] |
| evicted_compressed_entries | 0 | [0.0, 0.0] |
| evicted_kv_tokens | -35.2 | [-78.26981443740641, 7.869814437406411] |
| recomputed_prefix_tokens | -6.4 | [-36.52922701357501, 23.729227013575013] |
| request_throughput_rps | -0.00261523 | [-0.004437988021421122, -0.0007924649784040185] |
| output_throughput_tps | -0.334749 | [-0.5680624667419036, -0.10143551723571437] |
| mean_ttft_ms | -25.9348 | [-32.0525260044857, -19.81715275670352] |
| p50_ttft_ms | -19.9862 | [-22.652822412176334, -17.319605820255077] |
| p95_ttft_ms | -60.0928 | [-74.13168047793877, -46.05396310559738] |
| p99_ttft_ms | -55.761 | [-70.18116487269985, -41.340874794072036] |
| mean_tpot_ms | 0.393556 | [0.2687238357223931, 0.5183889982524293] |
| persistent_cache_mib | 0 | [0.0, 0.0] |
| peak_cache_state_mib | 249.525 | [230.09306748156192, 268.9569325184381] |
| peak_staging_mib | 249.525 | [230.09306748156192, 268.9569325184381] |
| peak_process_cuda_mib | 1508.02 | [1506.6838661286158, 1509.3645713713843] |
| compression_pending_peak | 3.4 | [2.7199126193417444, 4.080087380658256] |
| compression_enqueued | 0 | [0.0, 0.0] |
| compression_completed | 0 | [0.0, 0.0] |
| compression_completion_fraction | 0 | [0.0, 0.0] |
| compression_completion_per_s | -0.00605762 | [-0.010268740714447222, -0.001846506493995037] |
| compression_committed | 0 | [0.0, 0.0] |
| compression_failed | 0 | [0.0, 0.0] |
| decompression_hits | 0 | [0.0, 0.0] |
| prefill_deferred_batches | 48 | [48.0, 48.0] |
| prefill_deferral_wait_ms | 26858.2 | [26247.828431482565, 27468.534754906643] |
| svd_batches | -158.2 | [-167.59078945082373, -148.80921054917624] |
| svd_batch_items | 0.6 | [-0.5105780420791172, 1.710578042079117] |
| svd_batch_mean | 1.6269 | [1.5830252551878976, 1.6707807366712721] |
| svd_batch_max | 4 | [4.0, 4.0] |
