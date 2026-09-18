# Allocation and eviction search

Screening and held-out validation are separate. Student-t 95% CIs are across repetitions.

Cache budget includes KV, full/compressed state pools and measured staging; excludes weights, arithmetic workspace and allocator reserve. Process CUDA allocated peak is reported separately.

## allocation/full16 (n=3)

| Metric | Mean | 95% CI |
|---|---:|---|
| requests | 192 | [192.0, 192.0] |
| duration_s | 103.638 | [102.37762748644965, 104.89928628930232] |
| token_cache_hit_rate | 0.661499 | [0.661498708010336, 0.661498708010336] |
| evicted_entries | 0 | [0.0, 0.0] |
| evicted_compressed_entries | 0 | [0.0, 0.0] |
| evicted_kv_tokens | 0 | [0.0, 0.0] |
| recomputed_prefix_tokens | 0 | [0.0, 0.0] |
| request_throughput_rps | 1.85262 | [1.8300477102078911, 1.8751999533461103] |
| output_throughput_tps | 14.821 | [14.64038168166313, 15.001599626768883] |
| mean_ttft_ms | 185.28 | [183.2597088101525, 187.30016714022] |
| p50_ttft_ms | 65.1399 | [63.988163993488556, 66.29161109454354] |
| p95_ttft_ms | 431.784 | [425.87726783852804, 437.69139681407785] |
| p99_ttft_ms | 439.656 | [428.5190211485576, 450.7939347996562] |
| mean_tpot_ms | 50.609 | [49.96881485971373, 51.24911574459085] |
| persistent_cache_mib | 17272.8 | [17272.75, 17272.75] |
| peak_cache_state_mib | 17380.8 | [17380.796875, 17380.796875] |
| peak_staging_mib | 108.047 | [108.046875, 108.046875] |
| peak_process_cuda_mib | 27121.9 | [27121.80653252492, 27122.014105495917] |
| compression_pending_peak | 2 | [2.0, 2.0] |
| compression_enqueued | 448 | [448.0, 448.0] |
| compression_completed | 447 | [447.0, 447.0] |
| compression_completion_fraction | 0.997768 | [0.9977678571428571, 0.9977678571428571] |
| compression_completion_per_s | 4.31314 | [4.260579825327747, 4.365699891383914] |
| compression_committed | 447 | [447.0, 447.0] |
| compression_failed | 0 | [0.0, 0.0] |
| decompression_hits | 128 | [128.0, 128.0] |

## allocation/full32 (n=3)

| Metric | Mean | 95% CI |
|---|---:|---|
| requests | 192 | [192.0, 192.0] |
| duration_s | 103.839 | [102.84885161104721, 104.82862377432066] |
| token_cache_hit_rate | 0.661499 | [0.661498708010336, 0.661498708010336] |
| evicted_entries | 0 | [0.0, 0.0] |
| evicted_compressed_entries | 0 | [0.0, 0.0] |
| evicted_kv_tokens | 0 | [0.0, 0.0] |
| recomputed_prefix_tokens | 0 | [0.0, 0.0] |
| request_throughput_rps | 1.84904 | [1.8314150830089513, 1.8666630905760242] |
| output_throughput_tps | 14.7923 | [14.65132066407161, 14.933304724608194] |
| mean_ttft_ms | 185.824 | [182.4160050017252, 189.23225819095867] |
| p50_ttft_ms | 64.9485 | [63.34418623375561, 66.55280472112034] |
| p95_ttft_ms | 434.274 | [426.1847734742462, 442.36292351850875] |
| p99_ttft_ms | 440.338 | [438.00893498148, 442.66703604225836] |
| mean_tpot_ms | 50.6837 | [50.29738930610432, 51.06996695047013] |
| persistent_cache_mib | 17268.4 | [17268.4375, 17268.4375] |
| peak_cache_state_mib | 17392.5 | [17323.641931324008, 17461.326818675992] |
| peak_staging_mib | 124.047 | [55.20443132400861, 192.88931867599138] |
| peak_process_cuda_mib | 27114.4 | [26984.899316971292, 27243.879979903708] |
| compression_pending_peak | 2 | [2.0, 2.0] |
| compression_enqueued | 448 | [448.0, 448.0] |
| compression_completed | 447 | [447.0, 447.0] |
| compression_completion_fraction | 0.997768 | [0.9977678571428571, 0.9977678571428571] |
| compression_completion_per_s | 4.30479 | [4.263763240130214, 4.345825007747307] |
| compression_committed | 447 | [447.0, 447.0] |
| compression_failed | 0 | [0.0, 0.0] |
| decompression_hits | 128 | [128.0, 128.0] |

## allocation/full64 (n=3)

| Metric | Mean | 95% CI |
|---|---:|---|
| requests | 192 | [192.0, 192.0] |
| duration_s | 103.915 | [103.38546267719221, 104.44516361185757] |
| token_cache_hit_rate | 0.661499 | [0.661498708010336, 0.661498708010336] |
| evicted_entries | 0 | [0.0, 0.0] |
| evicted_compressed_entries | 0 | [0.0, 0.0] |
| evicted_kv_tokens | 0 | [0.0, 0.0] |
| recomputed_prefix_tokens | 0 | [0.0, 0.0] |
| request_throughput_rps | 1.84766 | [1.8382536738351865, 1.8570734695143984] |
| output_throughput_tps | 14.7813 | [14.706029390681492, 14.856587756115188] |
| mean_ttft_ms | 185.885 | [184.60257223514856, 187.16720261877595] |
| p50_ttft_ms | 65.4556 | [64.25494204309287, 66.65621511293746] |
| p95_ttft_ms | 433.073 | [430.81479790392285, 435.3309945753935] |
| p99_ttft_ms | 438.386 | [435.93598994465424, 440.83675149478756] |
| mean_tpot_ms | 50.7325 | [50.48267522635275, 50.9824026328268] |
| persistent_cache_mib | 17273 | [17272.984375, 17272.984375] |
| peak_cache_state_mib | 17381 | [17381.03125, 17381.03125] |
| peak_staging_mib | 108.047 | [108.046875, 108.046875] |
| peak_process_cuda_mib | 27121.4 | [27120.475678593302, 27122.346261510862] |
| compression_pending_peak | 2 | [2.0, 2.0] |
| compression_enqueued | 448 | [448.0, 448.0] |
| compression_completed | 447 | [447.0, 447.0] |
| compression_completion_fraction | 0.997768 | [0.9977678571428571, 0.9977678571428571] |
| compression_completion_per_s | 4.30159 | [4.279684334397545, 4.323499171213208] |
| compression_committed | 447 | [447.0, 447.0] |
| compression_failed | 0 | [0.0, 0.0] |
| decompression_hits | 128 | [128.0, 128.0] |

## policy/lru (n=3)

| Metric | Mean | 95% CI |
|---|---:|---|
| requests | 192 | [192.0, 192.0] |
| duration_s | 150.528 | [149.32462436102054, 151.73138343029] |
| token_cache_hit_rate | 0.0137812 | [0.006369245943584046, 0.021193200223513283] |
| evicted_entries | 543.333 | [541.8991157567502, 544.7675509099165] |
| evicted_compressed_entries | 543.333 | [541.8991157567502, 544.7675509099165] |
| evicted_kv_tokens | 371461 | [368523.38906982436, 374397.944263509] |
| recomputed_prefix_tokens | 256683 | [253745.38906982436, 259619.94426350895] |
| request_throughput_rps | 1.27552 | [1.2653316141405926, 1.2857063247992333] |
| output_throughput_tps | 10.2042 | [10.12265291312474, 10.285650598393866] |
| mean_ttft_ms | 425.358 | [418.9880302759076, 431.728110032872] |
| p50_ttft_ms | 430.092 | [425.4368459985667, 434.7462315553413] |
| p95_ttft_ms | 437.472 | [433.86545029807746, 441.07824789952576] |
| p99_ttft_ms | 442.961 | [437.4278832039904, 448.4935613784521] |
| mean_tpot_ms | 51.2061 | [51.04452782949207, 51.367589313957794] |
| persistent_cache_mib | 9016.14 | [9016.140625, 9016.140625] |
| peak_cache_state_mib | 9140.19 | [9071.34505632401, 9209.02994367599] |
| peak_staging_mib | 124.047 | [55.20443132400861, 192.88931867599138] |
| peak_process_cuda_mib | 18881.9 | [18812.21448581755, 18951.62047511995] |
| compression_pending_peak | 2 | [2.0, 2.0] |
| compression_enqueued | 573.333 | [571.8991157567502, 574.7675509099165] |
| compression_completed | 572.333 | [570.8991157567502, 573.7675509099165] |
| compression_completion_fraction | 0.998256 | [0.9982514521521814, 0.9982601733978812] |
| compression_completion_per_s | 3.80219 | [3.780530585756261, 3.823850026259633] |
| compression_committed | 572.333 | [570.8991157567502, 573.7675509099165] |
| compression_failed | 0 | [0.0, 0.0] |
| decompression_hits | 2.66667 | [1.2324490900835126, 4.100884243249821] |

## policy/retain_full (n=3)

| Metric | Mean | 95% CI |
|---|---:|---|
| requests | 192 | [192.0, 192.0] |
| duration_s | 150.774 | [150.50501827436656, 151.04264213976293] |
| token_cache_hit_rate | 0.0137812 | [0.006369245943584046, 0.021193200223513283] |
| evicted_entries | 541.333 | [539.8991157567502, 542.7675509099165] |
| evicted_compressed_entries | 541.333 | [539.8991157567502, 542.7675509099165] |
| evicted_kv_tokens | 359816 | [353941.7781396487, 365690.8885270179] |
| recomputed_prefix_tokens | 256683 | [253745.38906982436, 259619.94426350895] |
| request_throughput_rps | 1.27343 | [1.2711611502682931, 1.275700803871905] |
| output_throughput_tps | 10.1874 | [10.169289202146345, 10.20560643097524] |
| mean_ttft_ms | 426.743 | [422.9849317986535, 430.5014793519431] |
| p50_ttft_ms | 431.277 | [430.5858744839977, 431.9690873314787] |
| p95_ttft_ms | 438.782 | [437.68105093942734, 439.8837081784343] |
| p99_ttft_ms | 446.893 | [434.9405310275495, 458.8446128395855] |
| mean_tpot_ms | 51.1848 | [50.82769295198908, 51.54191207016828] |
| persistent_cache_mib | 9016.14 | [9016.140625, 9016.140625] |
| peak_cache_state_mib | 9144.2 | [9058.08284145611, 9230.32340854389] |
| peak_staging_mib | 128.062 | [41.942216456108426, 214.18278354389156] |
| peak_process_cuda_mib | 18885.4 | [18821.030727022317, 18949.835483915183] |
| compression_pending_peak | 2.33333 | [0.8991157567501795, 3.7675509099164874] |
| compression_enqueued | 573.333 | [571.8991157567502, 574.7675509099165] |
| compression_completed | 572.333 | [570.8991157567502, 573.7675509099165] |
| compression_completion_fraction | 0.998256 | [0.9982514521521814, 0.9982601733978812] |
| compression_completion_per_s | 3.79597 | [3.7922484955162035, 3.799695895500992] |
| compression_committed | 570.333 | [568.8991157567502, 571.7675509099165] |
| compression_failed | 0 | [0.0, 0.0] |
| decompression_hits | 2.66667 | [1.2324490900835126, 4.100884243249821] |

## policy/reuse (n=3)

| Metric | Mean | 95% CI |
|---|---:|---|
| requests | 192 | [192.0, 192.0] |
| duration_s | 149.821 | [148.19926125014368, 151.4423137671794] |
| token_cache_hit_rate | 0.0206718 | [0.007833913634365221, 0.03350975561628078] |
| evicted_entries | 542 | [539.5158622882497, 544.4841377117503] |
| evicted_compressed_entries | 542 | [539.5158622882497, 544.4841377117503] |
| evicted_kv_tokens | 363959 | [358100.8686521307, 369817.1313478693] |
| recomputed_prefix_tokens | 253952 | [248864.4859663353, 259039.5140336647] |
| request_throughput_rps | 1.28155 | [1.267643664872639, 1.2954510675061817] |
| output_throughput_tps | 10.2524 | [10.141149318981112, 10.363608540049453] |
| mean_ttft_ms | 422.323 | [415.1992779100751, 429.44587052413544] |
| p50_ttft_ms | 429.433 | [426.75903232846076, 432.1062846287746] |
| p95_ttft_ms | 436.532 | [431.5332247666903, 441.5302477675291] |
| p99_ttft_ms | 446.65 | [434.18918212592143, 459.11168372542124] |
| mean_tpot_ms | 51.1136 | [50.82876904575011, 51.39844472550236] |
| persistent_cache_mib | 9016.14 | [9016.140625, 9016.140625] |
| peak_cache_state_mib | 9128.2 | [9110.9252851321, 9145.4809648679] |
| peak_staging_mib | 112.062 | [94.78466013209982, 129.34033986790018] |
| peak_process_cuda_mib | 18898.7 | [18756.778189150195, 19040.55644626647] |
| compression_pending_peak | 2.33333 | [0.8991157567501795, 3.7675509099164874] |
| compression_enqueued | 572 | [569.5158622882497, 574.4841377117503] |
| compression_completed | 571 | [568.5158622882497, 573.4841377117503] |
| compression_completion_fraction | 0.998252 | [0.9982441521832535, 0.9982593371957826] |
| compression_completion_per_s | 3.81125 | [3.784190830511188, 3.8383106640716735] |
| compression_committed | 571 | [568.5158622882497, 573.4841377117503] |
| compression_failed | 0 | [0.0, 0.0] |
| decompression_hits | 4 | [1.5158622882496697, 6.48413771175033] |

## validation/b192_off (n=6)

| Metric | Mean | 95% CI |
|---|---:|---|
| requests | 192 | [192.0, 192.0] |
| duration_s | 122.905 | [121.46627771994176, 124.34333379302717] |
| token_cache_hit_rate | 0.366064 | [0.34594089981350185, 0.386186576500021] |
| evicted_entries | 315.167 | [311.27289744724595, 319.0604358860874] |
| evicted_compressed_entries | 0 | [0.0, 0.0] |
| evicted_kv_tokens | 119339 | [111719.70463633002, 126957.62869700333] |
| recomputed_prefix_tokens | 117077 | [109102.89397195968, 125051.77269470698] |
| request_throughput_rps | 1.56235 | [1.543902057552866, 1.580793940702163] |
| output_throughput_tps | 12.4988 | [12.351216460422927, 12.646351525617304] |
| mean_ttft_ms | 288.893 | [281.6298183409224, 296.1569870777081] |
| p50_ttft_ms | 418.222 | [415.8987919274407, 420.54502810950703] |
| p95_ttft_ms | 428.837 | [427.4101157241511, 430.26370863706387] |
| p99_ttft_ms | 433.964 | [432.06321147755995, 435.8652791487553] |
| mean_tpot_ms | 50.1494 | [50.08402414665129, 50.21487208755036] |
| persistent_cache_mib | 17673.2 | [17673.15625, 17673.15625] |
| peak_cache_state_mib | 17673.2 | [17673.15625, 17673.15625] |
| peak_staging_mib | 0 | [0.0, 0.0] |
| peak_process_cuda_mib | 27138.6 | [27138.17666479135, 27138.94442895865] |
| compression_pending_peak | 0 | [0.0, 0.0] |
| compression_enqueued | 0 | [0.0, 0.0] |
| compression_completed | 0 | [0.0, 0.0] |
| compression_completion_fraction | 0 | [0.0, 0.0] |
| compression_completion_per_s | 0 | [0.0, 0.0] |
| compression_committed | 0 | [0.0, 0.0] |
| compression_failed | 0 | [0.0, 0.0] |
| decompression_hits | 0 | [0.0, 0.0] |

## validation/b192_on (n=6)

| Metric | Mean | 95% CI |
|---|---:|---|
| requests | 192 | [192.0, 192.0] |
| duration_s | 104.178 | [103.97501297722098, 104.38126624817812] |
| token_cache_hit_rate | 0.661499 | [0.661498708010336, 0.661498708010336] |
| evicted_entries | 0 | [0.0, 0.0] |
| evicted_compressed_entries | 0 | [0.0, 0.0] |
| evicted_kv_tokens | 0 | [0.0, 0.0] |
| recomputed_prefix_tokens | 0 | [0.0, 0.0] |
| request_throughput_rps | 1.843 | [1.8394088328662173, 1.8465957940368682] |
| output_throughput_tps | 14.744 | [14.715270662929738, 14.772766352294946] |
| mean_ttft_ms | 186.549 | [185.73508506513605, 187.3622466539489] |
| p50_ttft_ms | 65.9746 | [64.92223267363423, 67.02699307742802] |
| p95_ttft_ms | 434.609 | [432.58765951538743, 436.6299803667301] |
| p99_ttft_ms | 441.88 | [437.19552714100325, 446.5639738517916] |
| mean_tpot_ms | 50.8377 | [50.742026051488075, 50.93333822652121] |
| persistent_cache_mib | 17272.8 | [17272.75, 17272.75] |
| peak_cache_state_mib | 17382.8 | [17377.643441158136, 17387.965933841864] |
| peak_staging_mib | 110.055 | [104.89344115813645, 115.21593384186355] |
| peak_process_cuda_mib | 27093.7 | [27051.983068725847, 27135.48991304499] |
| compression_pending_peak | 2.16667 | [1.7382363607272806, 2.5950969726060524] |
| compression_enqueued | 448 | [448.0, 448.0] |
| compression_completed | 447 | [447.0, 447.0] |
| compression_completion_fraction | 0.997768 | [0.9977678571428571, 0.9977678571428571] |
| compression_completion_per_s | 4.29074 | [4.282373689016661, 4.299105832992084] |
| compression_committed | 447 | [447.0, 447.0] |
| compression_failed | 0 | [0.0, 0.0] |
| decompression_hits | 128 | [128.0, 128.0] |

## validation/b24_off (n=6)

| Metric | Mean | 95% CI |
|---|---:|---|
| requests | 192 | [192.0, 192.0] |
| duration_s | 148.979 | [148.36990901682958, 149.5882985898188] |
| token_cache_hit_rate | 0.00775194 | [0.002063786125539982, 0.013440089843452265] |
| evicted_entries | 552.5 | [551.399342615292, 553.600657384708] |
| evicted_compressed_entries | 0 | [0.0, 0.0] |
| evicted_kv_tokens | 377992 | [375737.853676118, 380246.146323882] |
| recomputed_prefix_tokens | 259072 | [256817.853676118, 261326.146323882] |
| request_throughput_rps | 1.28879 | [1.2835395852748321, 1.294035580732525] |
| output_throughput_tps | 10.3103 | [10.268316682198657, 10.3522846458602] |
| mean_ttft_ms | 422.457 | [419.33231635619364, 425.58147537309884] |
| p50_ttft_ms | 425.05 | [423.3448795738312, 426.75522698586263] |
| p95_ttft_ms | 433.3 | [431.7746355931642, 434.8244996008096] |
| p99_ttft_ms | 440.04 | [435.76769356442594, 444.3121921381543] |
| mean_tpot_ms | 50.4637 | [50.38817508507766, 50.53931420490298] |
| persistent_cache_mib | 9420.16 | [9420.15625, 9420.15625] |
| peak_cache_state_mib | 9420.16 | [9420.15625, 9420.15625] |
| peak_staging_mib | 0 | [0.0, 0.0] |
| peak_process_cuda_mib | 18887.2 | [18887.192667903324, 18887.23148574251] |
| compression_pending_peak | 0 | [0.0, 0.0] |
| compression_enqueued | 0 | [0.0, 0.0] |
| compression_completed | 0 | [0.0, 0.0] |
| compression_completion_fraction | 0 | [0.0, 0.0] |
| compression_completion_per_s | 0 | [0.0, 0.0] |
| compression_committed | 0 | [0.0, 0.0] |
| compression_failed | 0 | [0.0, 0.0] |
| decompression_hits | 0 | [0.0, 0.0] |

## validation/b24_on (n=6)

| Metric | Mean | 95% CI |
|---|---:|---|
| requests | 192 | [192.0, 192.0] |
| duration_s | 150.882 | [150.0858501010951, 151.67815521656803] |
| token_cache_hit_rate | 0.0120586 | [0.003893335567697309, 0.020223804828512854] |
| evicted_entries | 543.667 | [542.0866937656828, 545.2466395676505] |
| evicted_compressed_entries | 543.667 | [542.0866937656828, 545.2466395676505] |
| evicted_kv_tokens | 368734 | [364658.46234591806, 372809.20432074857] |
| recomputed_prefix_tokens | 257365 | [254129.5488321183, 260601.11783454838] |
| request_throughput_rps | 1.27254 | [1.2658075352153897, 1.279281409414607] |
| output_throughput_tps | 10.1804 | [10.126460281723118, 10.234251275316856] |
| mean_ttft_ms | 427.292 | [423.71296897022665, 430.87068764560973] |
| p50_ttft_ms | 431.546 | [430.3163701218273, 432.77647984058603] |
| p95_ttft_ms | 438.509 | [437.1940519519274, 439.8248233352975] |
| p99_ttft_ms | 445.08 | [442.30980550152, 447.84983972805804] |
| mean_tpot_ms | 51.1887 | [51.07490105904394, 51.302542894598744] |
| persistent_cache_mib | 9016.14 | [9016.140625, 9016.140625] |
| peak_cache_state_mib | 9126.2 | [9121.034066158136, 9131.356558841864] |
| peak_staging_mib | 110.055 | [104.89344115813645, 115.21593384186355] |
| peak_process_cuda_mib | 18866 | [18865.74345302067, 18866.302282656416] |
| compression_pending_peak | 2.16667 | [1.7382363607272806, 2.5950969726060524] |
| compression_enqueued | 573.667 | [572.0866937656828, 575.2466395676505] |
| compression_completed | 572.667 | [571.0866937656828, 574.2466395676505] |
| compression_completion_fraction | 0.998257 | [0.9982520056613815, 0.9982616291352016] |
| compression_completion_per_s | 3.7955 | [3.785127866962007, 3.805872622760706] |
| compression_committed | 572.667 | [571.0866937656828, 574.2466395676505] |
| compression_failed | 0 | [0.0, 0.0] |
| decompression_hits | 2.33333 | [0.7533604323494292, 3.913306234317238] |
