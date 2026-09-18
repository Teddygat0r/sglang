# Spark concurrent cache-pressure experiment

Means and Student-t 95% CIs across independent repetitions. Synthetic prefixes, concurrency 2/4, 128 forced output tokens. LRU; graphs and overlap disabled. Not a quality evaluation or peak-capacity claim.

Matched ceilings cover KV, Mamba pools and snapshot/result staging; exclude weights, arithmetic workspace and allocator reserve.

## b128_c2_off (n=5)

| Metric | Mean | 95% CI |
|---|---:|---|
| requests | 192 | [192.0, 192.0] |
| duration_s | 565.989 | [563.9548298236715, 568.0234770487262] |
| token_cache_hit_rate | 0.144073 | [0.12663691397284277, 0.16150908085919857] |
| evicted_entries | 422 | [418.5995630967087, 425.4004369032913] |
| evicted_compressed_entries | 0 | [0.0, 0.0] |
| evicted_kv_tokens | 270874 | [263809.42460506817, 277938.57539493183] |
| recomputed_prefix_tokens | 205050 | [198139.8893644699, 211959.3106355301] |
| request_throughput_rps | 0.339231 | [0.33801329001206626, 0.34044948497981176] |
| output_throughput_tps | 43.4216 | [43.26570112154448, 43.577534077415905] |
| mean_ttft_ms | 656.03 | [643.0574521621438, 669.001921176563] |
| p50_ttft_ms | 734.964 | [732.0912748272063, 737.8359750216363] |
| p95_ttft_ms | 800.882 | [798.6718833695327, 803.0915336807097] |
| p99_ttft_ms | 805.463 | [802.8234784827519, 808.1019761784983] |
| mean_tpot_ms | 41.2541 | [41.155023563780304, 41.35317934553565] |
| persistent_cache_mib | 14529.2 | [14529.15625, 14529.15625] |
| peak_cache_state_mib | 14529.2 | [14529.15625, 14529.15625] |
| peak_staging_mib | 0 | [0.0, 0.0] |
| peak_process_cuda_mib | 23997.8 | [23997.518691093, 23998.108652657] |
| compression_pending_peak | 0 | [0.0, 0.0] |
| compression_enqueued | 0 | [0.0, 0.0] |
| compression_completed | 0 | [0.0, 0.0] |
| compression_completion_fraction | 0 | [0.0, 0.0] |
| compression_completion_per_s | 0 | [0.0, 0.0] |
| compression_committed | 0 | [0.0, 0.0] |
| compression_failed | 0 | [0.0, 0.0] |
| decompression_hits | 0 | [0.0, 0.0] |

## b128_c2_on (n=5)

| Metric | Mean | 95% CI |
|---|---:|---|
| requests | 192 | [192.0, 192.0] |
| duration_s | 535.336 | [534.5035396221067, 536.1679981512045] |
| token_cache_hit_rate | 0.653779 | [0.6473867858109476, 0.6601713537239361] |
| evicted_entries | 149 | [147.75833600179624, 150.24166399820376] |
| evicted_compressed_entries | 149 | [147.75833600179624, 150.24166399820376] |
| evicted_kv_tokens | 11401.2 | [10008.381044411894, 12794.018955588108] |
| recomputed_prefix_tokens | 3059.2 | [526.0145754487653, 5592.385424551234] |
| request_throughput_rps | 0.358654 | [0.3580961000850022, 0.35921162349234287] |
| output_throughput_tps | 45.9077 | [45.83630081088028, 45.97908780701989] |
| mean_ttft_ms | 332.672 | [327.0881544517494, 338.25576461868815] |
| p50_ttft_ms | 134.399 | [132.96656040880188, 135.83199502494827] |
| p95_ttft_ms | 822.936 | [820.9361196101886, 824.9355400188583] |
| p99_ttft_ms | 829.235 | [825.8970788798284, 832.5732766085315] |
| mean_tpot_ms | 41.2856 | [41.195723875934654, 41.375568868514925] |
| persistent_cache_mib | 13738.4 | [13738.375, 13738.375] |
| peak_cache_state_mib | 13949.6 | [13941.457072336132, 13957.842927663867] |
| peak_staging_mib | 211.275 | [203.08207233613257, 219.46792766386744] |
| peak_process_cuda_mib | 23746.9 | [23592.681377115707, 23901.104365071795] |
| compression_pending_peak | 4 | [4.0, 4.0] |
| compression_enqueued | 449 | [447.75833600179624, 450.24166399820376] |
| compression_completed | 447 | [445.75833600179624, 448.24166399820376] |
| compression_completion_fraction | 0.995546 | [0.9955333212382955, 0.9955579574409563] |
| compression_completion_per_s | 0.834992 | [0.8319484197038202, 0.838034987082623] |
| compression_committed | 447 | [445.75833600179624, 448.24166399820376] |
| compression_failed | 0 | [0.0, 0.0] |
| decompression_hits | 127 | [125.75833600179624, 128.24166399820376] |

## b128_c4_off (n=5)

| Metric | Mean | 95% CI |
|---|---:|---|
| requests | 192 | [192.0, 192.0] |
| duration_s | 324.197 | [322.00980800787096, 326.3843347748172] |
| token_cache_hit_rate | 0.139834 | [0.12016919371228466, 0.15949811894921403] |
| evicted_entries | 422.6 | [418.7129768527231, 426.48702314727694] |
| evicted_compressed_entries | 0 | [0.0, 0.0] |
| evicted_kv_tokens | 272096 | [264340.53004953766, 279852.2699504624] |
| recomputed_prefix_tokens | 206730 | [198936.80943785387, 214522.39056214615] |
| request_throughput_rps | 0.592246 | [0.5882658727527871, 0.5962267371937179] |
| output_throughput_tps | 75.8075 | [75.29803171235675, 76.31702236079589] |
| mean_ttft_ms | 1055.39 | [1034.262480438808, 1076.525315988222] |
| p50_ttft_ms | 1098.53 | [1092.2002691959026, 1104.866220038664] |
| p95_ttft_ms | 1526.98 | [1518.9167282122153, 1535.0334324923379] |
| p99_ttft_ms | 1534.57 | [1526.134387338613, 1543.008780648853] |
| mean_tpot_ms | 44.8657 | [44.605148719553355, 45.12621498238997] |
| persistent_cache_mib | 14529.2 | [14529.15625, 14529.15625] |
| peak_cache_state_mib | 14529.2 | [14529.15625, 14529.15625] |
| peak_staging_mib | 0 | [0.0, 0.0] |
| peak_process_cuda_mib | 23998.5 | [23998.117280532675, 23998.885649154825] |
| compression_pending_peak | 0 | [0.0, 0.0] |
| compression_enqueued | 0 | [0.0, 0.0] |
| compression_completed | 0 | [0.0, 0.0] |
| compression_completion_fraction | 0 | [0.0, 0.0] |
| compression_completion_per_s | 0 | [0.0, 0.0] |
| compression_committed | 0 | [0.0, 0.0] |
| compression_failed | 0 | [0.0, 0.0] |
| decompression_hits | 0 | [0.0, 0.0] |

## b128_c4_on (n=5)

| Metric | Mean | 95% CI |
|---|---:|---|
| requests | 192 | [192.0, 192.0] |
| duration_s | 292.788 | [292.1145301629337, 293.46125336321865] |
| token_cache_hit_rate | 0.648684 | [0.6442696827594241, 0.6530978883000074] |
| evicted_entries | 147 | [146.12201096691493, 147.87798903308507] |
| evicted_compressed_entries | 147 | [146.12201096691493, 147.87798903308507] |
| evicted_kv_tokens | 10705.6 | [9568.368084910984, 11842.831915089017] |
| recomputed_prefix_tokens | 5078.4 | [3329.144041366675, 6827.655958633324] |
| request_throughput_rps | 0.655767 | [0.6542605033151329, 0.6572727372803405] |
| output_throughput_tps | 83.9381 | [83.745344424337, 84.13091037188359] |
| mean_ttft_ms | 520.136 | [515.2178070194731, 525.0547845928287] |
| p50_ttft_ms | 167.342 | [166.72120954615613, 167.96303127186755] |
| p95_ttft_ms | 1590.97 | [1585.6393417583656, 1596.304262777989] |
| p99_ttft_ms | 1608.64 | [1602.3576348536355, 1614.9221039749043] |
| mean_tpot_ms | 43.9275 | [43.815589482117794, 44.03933046171117] |
| persistent_cache_mib | 13738.4 | [13738.375, 13738.375] |
| peak_cache_state_mib | 14060.1 | [14016.20296288445, 14104.009537115551] |
| peak_staging_mib | 321.731 | [277.82796288444933, 365.63453711555064] |
| peak_process_cuda_mib | 24427.9 | [24403.736189531723, 24452.09056828078] |
| compression_pending_peak | 7.8 | [7.244710978960441, 8.355289021039559] |
| compression_enqueued | 449 | [448.1220109669149, 449.8779890330851] |
| compression_completed | 445 | [444.1220109669149, 445.8779890330851] |
| compression_completion_fraction | 0.991091 | [0.9910738759119699, 0.9911087167984625] |
| compression_completion_per_s | 1.51987 | [1.5158936905343332, 1.5238560094525984] |
| compression_committed | 445 | [444.1220109669149, 445.8779890330851] |
| compression_failed | 0 | [0.0, 0.0] |
| decompression_hits | 127 | [126.12201096691491, 127.87798903308509] |

## b192_c2_off (n=5)

| Metric | Mean | 95% CI |
|---|---:|---|
| requests | 192 | [192.0, 192.0] |
| duration_s | 553.213 | [551.8077267228688, 554.6185885303593] |
| token_cache_hit_rate | 0.333624 | [0.31070005977014686, 0.35654800224535704] |
| evicted_entries | 321.2 | [316.77507463878436, 325.6249253612156] |
| evicted_compressed_entries | 0 | [0.0, 0.0] |
| evicted_kv_tokens | 152986 | [144789.13209552216, 161182.06790447785] |
| recomputed_prefix_tokens | 129933 | [120848.30528619194, 139017.29471380805] |
| request_throughput_rps | 0.347064 | [0.34618291864834594, 0.34794606314184773] |
| output_throughput_tps | 44.4243 | [44.31141358698828, 44.53709608215651] |
| mean_ttft_ms | 521.509 | [505.7657768096774, 537.2531462188501] |
| p50_ttft_ms | 727.596 | [724.7021941283479, 730.4900469443069] |
| p95_ttft_ms | 798.806 | [796.1945049089035, 801.4181540909568] |
| p99_ttft_ms | 805.262 | [803.6714275970812, 806.8534511730318] |
| mean_tpot_ms | 41.2646 | [41.218683621550554, 41.310471536287125] |
| persistent_cache_mib | 17673.2 | [17673.15625, 17673.15625] |
| peak_cache_state_mib | 17673.2 | [17673.15625, 17673.15625] |
| peak_staging_mib | 0 | [0.0, 0.0] |
| peak_process_cuda_mib | 27141.2 | [27140.58213642161, 27141.808293265887] |
| compression_pending_peak | 0 | [0.0, 0.0] |
| compression_enqueued | 0 | [0.0, 0.0] |
| compression_completed | 0 | [0.0, 0.0] |
| compression_completion_fraction | 0 | [0.0, 0.0] |
| compression_completion_per_s | 0 | [0.0, 0.0] |
| compression_committed | 0 | [0.0, 0.0] |
| compression_failed | 0 | [0.0, 0.0] |
| decompression_hits | 0 | [0.0, 0.0] |

## b192_c2_on (n=5)

| Metric | Mean | 95% CI |
|---|---:|---|
| requests | 192 | [192.0, 192.0] |
| duration_s | 535.221 | [533.905547910745, 536.5355398433259] |
| token_cache_hit_rate | 0.658915 | [0.6589147286821705, 0.6589147286821705] |
| evicted_entries | 0 | [0.0, 0.0] |
| evicted_compressed_entries | 0 | [0.0, 0.0] |
| evicted_kv_tokens | 0 | [0.0, 0.0] |
| recomputed_prefix_tokens | 1024 | [1024.0, 1024.0] |
| request_throughput_rps | 0.358732 | [0.35785008538426377, 0.3596134118511509] |
| output_throughput_tps | 45.9177 | [45.80481092918576, 46.03051671694732] |
| mean_ttft_ms | 329.306 | [327.4236442178953, 331.1875955349921] |
| p50_ttft_ms | 134.483 | [132.3294571394785, 136.63696248282355] |
| p95_ttft_ms | 821.767 | [817.898610424214, 825.6344647296137] |
| p99_ttft_ms | 828.615 | [824.0039114243222, 833.2260512524414] |
| mean_tpot_ms | 41.3025 | [41.20021289648206, 41.404757246442784] |
| persistent_cache_mib | 16886.5 | [16886.453125, 16886.453125] |
| peak_cache_state_mib | 17100.1 | [17087.622596272344, 17112.652403727658] |
| peak_staging_mib | 213.684 | [201.16947127234326, 226.19927872765672] |
| peak_process_cuda_mib | 27061.1 | [26851.874626683573, 27270.254865503928] |
| compression_pending_peak | 4.2 | [3.6447109789604415, 4.755289021039559] |
| compression_enqueued | 448 | [448.0, 448.0] |
| compression_completed | 446 | [446.0, 446.0] |
| compression_completion_fraction | 0.995536 | [0.9955357142857143, 0.9955357142857143] |
| compression_completion_per_s | 0.833304 | [0.8312559275071961, 0.8353519879459027] |
| compression_committed | 446 | [446.0, 446.0] |
| compression_failed | 0 | [0.0, 0.0] |
| decompression_hits | 128 | [128.0, 128.0] |

## b192_c4_off (n=5)

| Metric | Mean | 95% CI |
|---|---:|---|
| requests | 192 | [192.0, 192.0] |
| duration_s | 311.731 | [309.1416542995604, 314.32128484511054] |
| token_cache_hit_rate | 0.320446 | [0.30020744193191656, 0.34068403093630045] |
| evicted_entries | 323.2 | [319.33285945445175, 327.0671405455482] |
| evicted_compressed_entries | 0 | [0.0, 0.0] |
| evicted_kv_tokens | 158434 | [149509.5013810481, 167358.4986189519] |
| recomputed_prefix_tokens | 135155 | [127135.00674831537, 143175.39325168464] |
| request_throughput_rps | 0.615937 | [0.6108177339777313, 0.6210558361150308] |
| output_throughput_tps | 78.8399 | [78.18466994914961, 79.49514702272394] |
| mean_ttft_ms | 852.511 | [830.0000406471146, 875.0223990123033] |
| p50_ttft_ms | 780.595 | [747.1781301732838, 814.0122853730143] |
| p95_ttft_ms | 1520.47 | [1515.9706671263043, 1524.97367772209] |
| p99_ttft_ms | 1530.68 | [1527.4362029953022, 1533.9185553194861] |
| mean_tpot_ms | 44.418 | [44.144151965629774, 44.691915649617016] |
| persistent_cache_mib | 17673.2 | [17673.15625, 17673.15625] |
| peak_cache_state_mib | 17673.2 | [17673.15625, 17673.15625] |
| peak_staging_mib | 0 | [0.0, 0.0] |
| peak_process_cuda_mib | 27142.3 | [27141.960563739907, 27142.54119407259] |
| compression_pending_peak | 0 | [0.0, 0.0] |
| compression_enqueued | 0 | [0.0, 0.0] |
| compression_completed | 0 | [0.0, 0.0] |
| compression_completion_fraction | 0 | [0.0, 0.0] |
| compression_completion_per_s | 0 | [0.0, 0.0] |
| compression_committed | 0 | [0.0, 0.0] |
| compression_failed | 0 | [0.0, 0.0] |
| decompression_hits | 0 | [0.0, 0.0] |

## b192_c4_on (n=5)

| Metric | Mean | 95% CI |
|---|---:|---|
| requests | 192 | [192.0, 192.0] |
| duration_s | 292.884 | [292.41089462496336, 293.3574289559573] |
| token_cache_hit_rate | 0.653747 | [0.6537467700258398, 0.6537467700258398] |
| evicted_entries | 0 | [0.0, 0.0] |
| evicted_compressed_entries | 0 | [0.0, 0.0] |
| evicted_kv_tokens | 0 | [0.0, 0.0] |
| recomputed_prefix_tokens | 3072 | [3072.0, 3072.0] |
| request_throughput_rps | 0.65555 | [0.6544907714822754, 0.6566095556152163] |
| output_throughput_tps | 83.9104 | [83.77481874973125, 84.04602311874768] |
| mean_ttft_ms | 517.362 | [514.6342792902611, 520.089778140606] |
| p50_ttft_ms | 166.398 | [164.79761522984901, 167.99794513010582] |
| p95_ttft_ms | 1589.34 | [1585.2825145891973, 1593.3902194627456] |
| p99_ttft_ms | 1621.62 | [1593.3798835397004, 1649.8515767633687] |
| mean_tpot_ms | 43.9653 | [43.88495689444228, 44.04564611861997] |
| persistent_cache_mib | 16886.5 | [16886.453125, 16886.453125] |
| peak_cache_state_mib | 17215.4 | [17163.184502513308, 17267.64049748669] |
| peak_staging_mib | 328.959 | [276.7313775133102, 381.18737248668987] |
| peak_process_cuda_mib | 27576 | [27552.300707146715, 27599.66628504079] |
| compression_pending_peak | 7.8 | [7.244710978960441, 8.355289021039559] |
| compression_enqueued | 448 | [448.0, 448.0] |
| compression_completed | 444 | [444.0, 444.0] |
| compression_completion_fraction | 0.991071 | [0.9910714285714286, 0.9910714285714286] |
| compression_completion_per_s | 1.51596 | [1.513509909052762, 1.5184095973601877] |
| compression_committed | 444 | [444.0, 444.0] |
| compression_failed | 0 | [0.0, 0.0] |
| decompression_hits | 128 | [128.0, 128.0] |

## b64_c2_off (n=5)

| Metric | Mean | 95% CI |
|---|---:|---|
| requests | 192 | [192.0, 192.0] |
| duration_s | 573.606 | [572.6199499691003, 574.5929668044574] |
| token_cache_hit_rate | 0.0360223 | [0.02597130817547232, 0.04607326546793854] |
| evicted_entries | 507 | [505.0367568385224, 508.9632431614776] |
| evicted_compressed_entries | 0 | [0.0, 0.0] |
| evicted_kv_tokens | 359915 | [356599.8277049166, 363230.97229508345] |
| recomputed_prefix_tokens | 247869 | [243885.71777424155, 251851.88222575843] |
| request_throughput_rps | 0.334725 | [0.3341487660106154, 0.33530079144061065] |
| output_throughput_tps | 42.8448 | [42.77104204935877, 42.91850130439816] |
| mean_ttft_ms | 736.275 | [728.8214417794497, 743.7284180041624] |
| p50_ttft_ms | 740.369 | [738.6796764018809, 742.0585208651746] |
| p95_ttft_ms | 802.081 | [799.2892372099756, 804.8718335987927] |
| p99_ttft_ms | 807.812 | [802.5421930860457, 813.0815706391159] |
| mean_tpot_ms | 41.2467 | [41.198838684226146, 41.294607489019214] |
| persistent_cache_mib | 11385.2 | [11385.15625, 11385.15625] |
| peak_cache_state_mib | 11385.2 | [11385.15625, 11385.15625] |
| peak_staging_mib | 0 | [0.0, 0.0] |
| peak_process_cuda_mib | 20854.5 | [20854.226648074935, 20854.851281612562] |
| compression_pending_peak | 0 | [0.0, 0.0] |
| compression_enqueued | 0 | [0.0, 0.0] |
| compression_completed | 0 | [0.0, 0.0] |
| compression_completion_fraction | 0 | [0.0, 0.0] |
| compression_completion_per_s | 0 | [0.0, 0.0] |
| compression_committed | 0 | [0.0, 0.0] |
| compression_failed | 0 | [0.0, 0.0] |
| decompression_hits | 0 | [0.0, 0.0] |

## b64_c2_on (n=5)

| Metric | Mean | 95% CI |
|---|---:|---|
| requests | 192 | [192.0, 192.0] |
| duration_s | 577.022 | [575.889392355576, 578.1541299919339] |
| token_cache_hit_rate | 0.0349887 | [0.023592306233940003, 0.04638508394693855] |
| evicted_entries | 508.2 | [505.97884391584176, 510.4211560841582] |
| evicted_compressed_entries | 508.2 | [505.97884391584176, 510.4211560841582] |
| evicted_kv_tokens | 358687 | [354508.14313323563, 362865.0568667643] |
| recomputed_prefix_tokens | 248278 | [243762.14785283562, 252794.65214716436] |
| request_throughput_rps | 0.332744 | [0.3320913213865787, 0.3333961741429914] |
| output_throughput_tps | 42.5912 | [42.507689137482075, 42.6747102903029] |
| mean_ttft_ms | 760.521 | [750.3964057563002, 770.6452152476207] |
| p50_ttft_ms | 758.05 | [754.9598151509206, 761.1408712651094] |
| p95_ttft_ms | 829.826 | [827.424963035246, 832.2266645798481] |
| p99_ttft_ms | 835.414 | [833.122717409692, 837.7053499675177] |
| mean_tpot_ms | 41.3355 | [41.302756840144596, 41.368287359012115] |
| persistent_cache_mib | 10590.3 | [10590.296875, 10590.296875] |
| peak_cache_state_mib | 10796.8 | [10790.063627574664, 10803.442622425335] |
| peak_staging_mib | 206.456 | [199.76675257466408, 213.14574742533594] |
| peak_process_cuda_mib | 20742.2 | [20651.417863117756, 20832.952840007245] |
| compression_pending_peak | 4.2 | [3.6447109789604415, 4.755289021039559] |
| compression_enqueued | 569.2 | [566.9788439158418, 571.4211560841583] |
| compression_completed | 567.2 | [564.9788439158418, 569.4211560841583] |
| compression_completion_fraction | 0.996486 | [0.9964725559376671, 0.9964999816422268] |
| compression_completion_per_s | 0.982977 | [0.9806553222702344, 0.9852987146632558] |
| compression_committed | 567.2 | [564.9788439158418, 569.4211560841583] |
| compression_failed | 0 | [0.0, 0.0] |
| decompression_hits | 6.8 | [4.578843915841766, 9.021156084158234] |

## b64_c4_off (n=5)

| Metric | Mean | 95% CI |
|---|---:|---|
| requests | 192 | [192.0, 192.0] |
| duration_s | 331.59 | [330.8769278886057, 332.30395449006187] |
| token_cache_hit_rate | 0.0326712 | [0.023137420400488863, 0.04220495686049305] |
| evicted_entries | 507.6 | [505.7169233116516, 509.48307668834843] |
| evicted_compressed_entries | 0 | [0.0, 0.0] |
| evicted_kv_tokens | 360325 | [356304.278005294, 364345.721994706] |
| recomputed_prefix_tokens | 249197 | [245418.6820556689, 252974.91794433107] |
| request_throughput_rps | 0.579029 | [0.5777818156489041, 0.5802760647094167] |
| output_throughput_tps | 74.1157 | [73.95607240305972, 74.27533628280534] |
| mean_ttft_ms | 1170.38 | [1152.0415965476077, 1188.7159230140346] |
| p50_ttft_ms | 1108.63 | [1103.8364790727026, 1113.4165763507956] |
| p95_ttft_ms | 1532.46 | [1525.576591787715, 1539.3492637967277] |
| p99_ttft_ms | 1539.07 | [1531.7632666206373, 1546.379865440129] |
| mean_tpot_ms | 45.1732 | [45.032909052606556, 45.313472678823786] |
| persistent_cache_mib | 11385.2 | [11385.15625, 11385.15625] |
| peak_cache_state_mib | 11385.2 | [11385.15625, 11385.15625] |
| peak_staging_mib | 0 | [0.0, 0.0] |
| peak_process_cuda_mib | 20855.6 | [20855.162207888596, 20856.076268673907] |
| compression_pending_peak | 0 | [0.0, 0.0] |
| compression_enqueued | 0 | [0.0, 0.0] |
| compression_completed | 0 | [0.0, 0.0] |
| compression_completion_fraction | 0 | [0.0, 0.0] |
| compression_completion_per_s | 0 | [0.0, 0.0] |
| compression_committed | 0 | [0.0, 0.0] |
| compression_failed | 0 | [0.0, 0.0] |
| decompression_hits | 0 | [0.0, 0.0] |

## b64_c4_on (n=5)

| Metric | Mean | 95% CI |
|---|---:|---|
| requests | 192 | [192.0, 192.0] |
| duration_s | 335.931 | [334.4909045245788, 337.3703937168875] |
| token_cache_hit_rate | 0.0388162 | [0.024238007514287258, 0.053394421426281216] |
| evicted_entries | 505.4 | [502.541474809013, 508.258525190987] |
| evicted_compressed_entries | 505.4 | [502.541474809013, 508.258525190987] |
| evicted_kv_tokens | 352257 | [347707.67233964393, 356805.527660356] |
| recomputed_prefix_tokens | 246762 | [240984.43152182188, 252538.76847817813] |
| request_throughput_rps | 0.571552 | [0.5690972218928582, 0.5740067753824479] |
| output_throughput_tps | 73.1587 | [72.84444440228584, 73.47286724895334] |
| mean_ttft_ms | 1220.33 | [1197.3932738117971, 1243.2588312490745] |
| p50_ttft_ms | 1156.81 | [1150.8508138087711, 1162.767567885188] |
| p95_ttft_ms | 1605.58 | [1596.5268375584462, 1614.6427145367882] |
| p99_ttft_ms | 1616.28 | [1607.3673411254479, 1625.1964560280846] |
| mean_tpot_ms | 45.4916 | [45.37880677318395, 45.60429709700291] |
| persistent_cache_mib | 10590.3 | [10590.296875, 10590.296875] |
| peak_cache_state_mib | 10878.5 | [10878.484375, 10878.484375] |
| peak_staging_mib | 288.188 | [288.1875, 288.1875] |
| peak_process_cuda_mib | 21266.6 | [21265.144430940847, 21268.033498746656] |
| compression_pending_peak | 7 | [7.0, 7.0] |
| compression_enqueued | 568.4 | [565.5414748090129, 571.258525190987] |
| compression_completed | 564.4 | [561.5414748090129, 567.258525190987] |
| compression_completion_fraction | 0.992963 | [0.9929272012347787, 0.9929980186069685] |
| compression_completion_per_s | 1.68011 | [1.6744859604029088, 1.685734844687232] |
| compression_committed | 564.4 | [561.5414748090129, 567.258525190987] |
| compression_failed | 0 | [0.0, 0.0] |
| decompression_hits | 7.6 | [4.741474809012978, 10.458525190987022] |
