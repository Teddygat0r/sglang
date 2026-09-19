# Individual cache optimization ablation

Qwen3.5-4B, rank 16, concurrency 4, fixed ~14.19 GiB cache ceiling. 64 shared prefixes × 3 passes, 2048 prefix + 16 suffix tokens, 128 output tokens. LRU; graphs and overlap disabled. Five repetitions; reverse variant order on odd repetitions. Each variant changes only one optimization. Race fixes retained in all variants.

Ceiling includes KV and dense/compressed state pools plus staging; excludes model weights, arithmetic workspace and allocator reserve. Staging reservation is not runtime enforcement. Fresh baseline versus logging-only, singleton-only and restore-nozero-only variants. No quality evaluation; no pooling with earlier studies.

## baseline (n=5)

| Metric | Mean | 95% CI |
|---|---:|---|
| requests | 192 | [192.0, 192.0] |
| duration_s | 291.051 | [290.42130248674556, 291.6801133493205] |
| token_cache_hit_rate | 0.653747 | [0.6537467700258398, 0.6537467700258398] |
| evicted_entries | 86 | [86.0, 86.0] |
| evicted_compressed_entries | 59.8 | [54.8801267388646, 64.7198732611354] |
| evicted_kv_tokens | 7417.2 | [7174.077242838799, 7660.322757161201] |
| recomputed_prefix_tokens | 3072 | [3072.0, 3072.0] |
| request_throughput_rps | 0.65968 | [0.6582535070956508, 0.6611074208553105] |
| output_throughput_tps | 84.4391 | [84.2564489082433, 84.62174986947974] |
| mean_ttft_ms | 514.096 | [509.47673540884654, 518.7152376622331] |
| p50_ttft_ms | 166.396 | [164.99359446987634, 167.79766429737086] |
| p95_ttft_ms | 1593.64 | [1584.248717708744, 1603.0284913422095] |
| p99_ttft_ms | 1608.9 | [1600.4914334505838, 1617.3121876850682] |
| mean_tpot_ms | 43.6898 | [43.61487682643833, 43.764723430676995] |
| persistent_cache_mib | 13742.7 | [13742.6875, 13742.6875] |
| peak_cache_state_mib | 14052.4 | [14027.995608392272, 14076.898141607728] |
| peak_staging_mib | 309.759 | [285.30810839227115, 334.2106416077288] |
| peak_process_cuda_mib | 24418.1 | [24417.93220455841, 24418.322287629093] |
| compression_pending_peak | 7 | [7.0, 7.0] |
| compression_enqueued | 448 | [448.0, 448.0] |
| compression_completed | 441.2 | [436.60416558862653, 445.79583441137345] |
| compression_completion_fraction | 0.984821 | [0.9745628696174701, 0.9950799875253872] |
| compression_completion_per_s | 1.51588 | [1.5024210633838353, 1.5293332292795998] |
| compression_committed | 417.8 | [412.8801267388646, 422.71987326113543] |
| compression_failed | 0 | [0.0, 0.0] |
| decompression_hits | 128 | [128.0, 128.0] |

## logging (n=5)

| Metric | Mean | 95% CI |
|---|---:|---|
| requests | 192 | [192.0, 192.0] |
| duration_s | 291.687 | [291.24531394435024, 292.12940413834593] |
| token_cache_hit_rate | 0.653747 | [0.6537467700258398, 0.6537467700258398] |
| evicted_entries | 86 | [86.0, 86.0] |
| evicted_compressed_entries | 63.2 | [58.952855669053186, 67.44714433094681] |
| evicted_kv_tokens | 7271.4 | [6983.045623912743, 7559.754376087256] |
| recomputed_prefix_tokens | 3072 | [3072.0, 3072.0] |
| request_throughput_rps | 0.65824 | [0.6572409887521045, 0.6592386127957479] |
| output_throughput_tps | 84.2547 | [84.12684656026937, 84.38254243785573] |
| mean_ttft_ms | 513.269 | [511.3343143578811, 515.2031135963393] |
| p50_ttft_ms | 164.166 | [163.07338539323214, 165.25787685254213] |
| p95_ttft_ms | 1590.29 | [1584.076056789313, 1596.5002009908812] |
| p99_ttft_ms | 1611.43 | [1604.6187886997104, 1618.2330203251004] |
| mean_tpot_ms | 43.8001 | [43.71374267725273, 43.88654401098244] |
| persistent_cache_mib | 13742.7 | [13742.6875, 13742.6875] |
| peak_cache_state_mib | 14054.8 | [14027.021183695173, 14082.616316304828] |
| peak_staging_mib | 312.131 | [284.33368369517245, 339.9288163048276] |
| peak_process_cuda_mib | 24437.6 | [24405.236978700705, 24470.062435361797] |
| compression_pending_peak | 6.8 | [6.244710978960441, 7.3552890210395585] |
| compression_enqueued | 448 | [448.0, 448.0] |
| compression_completed | 443.4 | [441.7341329368813, 445.0658670631187] |
| compression_completion_fraction | 0.989732 | [0.9860136895912529, 0.9934505961230328] |
| compression_completion_per_s | 1.52012 | [1.5129686029570537, 1.527281026430759] |
| compression_committed | 421.2 | [416.9528556690532, 425.4471443309468] |
| compression_failed | 0 | [0.0, 0.0] |
| decompression_hits | 128 | [128.0, 128.0] |

## restore_nozero (n=5)

| Metric | Mean | 95% CI |
|---|---:|---|
| requests | 192 | [192.0, 192.0] |
| duration_s | 291.588 | [291.3264223002872, 291.85027814572237] |
| token_cache_hit_rate | 0.653747 | [0.6537467700258398, 0.6537467700258398] |
| evicted_entries | 86 | [86.0, 86.0] |
| evicted_compressed_entries | 64.4 | [62.51692331165157, 66.28307668834844] |
| evicted_kv_tokens | 7262 | [7138.646446558541, 7385.353553441459] |
| recomputed_prefix_tokens | 3072 | [3072.0, 3072.0] |
| request_throughput_rps | 0.658463 | [0.6578712100559362, 0.6590543821898543] |
| output_throughput_tps | 84.2832 | [84.20751488715983, 84.35896092030136] |
| mean_ttft_ms | 514.118 | [510.95645000577133, 517.2794307841964] |
| p50_ttft_ms | 164.666 | [163.5517530811971, 165.78111939895695] |
| p95_ttft_ms | 1593.08 | [1587.7064077479486, 1598.4566829802748] |
| p99_ttft_ms | 1607.52 | [1599.3657094057864, 1615.6671200800472] |
| mean_tpot_ms | 43.7776 | [43.72261554899452, 43.83250182744873] |
| persistent_cache_mib | 13742.7 | [13742.6875, 13742.6875] |
| peak_cache_state_mib | 14047.6 | [14018.672275856163, 14076.583974143836] |
| peak_staging_mib | 304.941 | [275.98477585616286, 333.89647414383717] |
| peak_process_cuda_mib | 24439.2 | [24382.922510854798, 24495.4562000827] |
| compression_pending_peak | 7 | [7.0, 7.0] |
| compression_enqueued | 448 | [448.0, 448.0] |
| compression_completed | 444 | [444.0, 444.0] |
| compression_completion_fraction | 0.991071 | [0.9910714285714286, 0.9910714285714286] |
| compression_completion_per_s | 1.5227 | [1.5213271732543525, 1.5240632588140381] |
| compression_committed | 422.4 | [420.51692331165157, 424.2830766883484] |
| compression_failed | 0 | [0.0, 0.0] |
| decompression_hits | 128 | [128.0, 128.0] |

## singleton (n=5)

| Metric | Mean | 95% CI |
|---|---:|---|
| requests | 192 | [192.0, 192.0] |
| duration_s | 291.202 | [290.6744241035958, 291.72986725946464] |
| token_cache_hit_rate | 0.653747 | [0.6537467700258398, 0.6537467700258398] |
| evicted_entries | 86 | [86.0, 86.0] |
| evicted_compressed_entries | 63.2 | [59.23444319909037, 67.16555680090964] |
| evicted_kv_tokens | 7271.4 | [7057.277572573576, 7485.522427426423] |
| recomputed_prefix_tokens | 3072 | [3072.0, 3072.0] |
| request_throughput_rps | 0.659337 | [0.6581437912383808, 0.6605300548625376] |
| output_throughput_tps | 84.3951 | [84.24240527851275, 84.54784702240481] |
| mean_ttft_ms | 513.784 | [510.7123447053303, 516.8558926151896] |
| p50_ttft_ms | 166.28 | [165.3805393873286, 167.1802423746753] |
| p95_ttft_ms | 1590.78 | [1585.2024323630012, 1596.360802152487] |
| p99_ttft_ms | 1603.76 | [1593.7890935540797, 1613.725682280898] |
| mean_tpot_ms | 43.7169 | [43.649384678985406, 43.784373822058704] |
| persistent_cache_mib | 13742.7 | [13742.6875, 13742.6875] |
| peak_cache_state_mib | 14047.7 | [14018.63217406607, 14076.69907593393] |
| peak_staging_mib | 304.978 | [275.94467406607083, 334.0115759339291] |
| peak_process_cuda_mib | 24418.5 | [24416.590792333704, 24420.430301416298] |
| compression_pending_peak | 6.8 | [6.244710978960441, 7.3552890210395585] |
| compression_enqueued | 448 | [448.0, 448.0] |
| compression_completed | 442.2 | [440.1597378580252, 444.24026214197477] |
| compression_completion_fraction | 0.987054 | [0.9824994148616635, 0.9916077279954794] |
| compression_completion_per_s | 1.51853 | [1.5127024675314449, 1.5243604880087462] |
| compression_committed | 421.2 | [417.2344431990904, 425.1655568009096] |
| compression_failed | 0 | [0.0, 0.0] |
| decompression_hits | 128 | [128.0, 128.0] |

## Paired logging minus baseline (n=5)

| Metric | Mean difference | 95% CI |
|---|---:|---|
| requests | 0 | [0.0, 0.0] |
| duration_s | 0.636651 | [0.11527994461712998, 1.1580223020130618] |
| token_cache_hit_rate | 0 | [0.0, 0.0] |
| evicted_entries | 0 | [0.0, 0.0] |
| evicted_compressed_entries | 3.4 | [-2.194381872382126, 8.994381872382126] |
| evicted_kv_tokens | -145.8 | [-494.69697837467004, 203.09697837467002] |
| recomputed_prefix_tokens | 0 | [0.0, 0.0] |
| request_throughput_rps | -0.00144066 | [-0.002621997627648679, -0.00025932877546026205] |
| output_throughput_tps | -0.184405 | [-0.33561569633903093, -0.03319408325891354] |
| mean_ttft_ms | -0.827273 | [-6.8593070315217535, 5.204761914662567] |
| p50_ttft_ms | -2.23 | [-3.430262304049964, -1.029734217422967] |
| p95_ttft_ms | -3.35048 | [-16.442948574908428, 9.74199730414932] |
| p99_ttft_ms | 2.52409 | [-10.635228308635075, 15.6834161977938] |
| mean_tpot_ms | 0.110343 | [0.021702185082316236, 0.1989842460375339] |
| persistent_cache_mib | 0 | [0.0, 0.0] |
| peak_cache_state_mib | 2.37188 | [-34.62123931146195, 39.364989311461954] |
| peak_staging_mib | 2.37188 | [-34.62123931146195, 39.364989311461954] |
| peak_process_cuda_mib | 19.5225 | [-12.845327455619149, 51.89024933061915] |
| compression_pending_peak | -0.2 | [-0.7552890210395586, 0.35528902103955856] |
| compression_enqueued | 0 | [0.0, 0.0] |
| compression_completed | 2.2 | [-3.3110861763700026, 7.711086176370003] |
| compression_completion_fraction | 0.00491071 | [-0.0073908173579687675, 0.017212245929397327] |
| compression_completion_per_s | 0.00424767 | [-0.014623731501095551, 0.023119068225473095] |
| compression_committed | 3.4 | [-2.194381872382126, 8.994381872382126] |
| compression_failed | 0 | [0.0, 0.0] |
| decompression_hits | 0 | [0.0, 0.0] |

## Paired restore_nozero minus baseline (n=5)

| Metric | Mean difference | 95% CI |
|---|---:|---|
| requests | 0 | [0.0, 0.0] |
| duration_s | 0.537642 | [-0.21680202325465936, 1.2920866331982879] |
| token_cache_hit_rate | 0 | [0.0, 0.0] |
| evicted_entries | 0 | [0.0, 0.0] |
| evicted_compressed_entries | 4.6 | [-1.64545882671678, 10.84545882671678] |
| evicted_kv_tokens | -155.2 | [-506.75693934399175, 196.35693934399177] |
| recomputed_prefix_tokens | 0 | [0.0, 0.0] |
| request_throughput_rps | -0.00121767 | [-0.002927013033677335, 0.0004916773285065968] |
| output_throughput_tps | -0.155861 | [-0.37465766831069885, 0.06293469804884438] |
| mean_ttft_ms | 0.0219539 | [-6.363314669046542, 6.407222387934723] |
| p50_ttft_ms | -1.72919 | [-3.9284854938120852, 0.4700992067189227] |
| p95_ttft_ms | -0.557059 | [-11.665218645496287, 10.551100322766223] |
| p99_ttft_ms | -1.3854 | [-10.613315786130068, 7.842524136311647] |
| mean_tpot_ms | 0.0877586 | [-0.01686020948399017, 0.1923773288119197] |
| persistent_cache_mib | 0 | [0.0, 0.0] |
| peak_cache_state_mib | -4.81875 | [-47.630002096986765, 37.99250209698676] |
| peak_staging_mib | -4.81875 | [-47.630002096986765, 37.99250209698676] |
| peak_process_cuda_mib | 21.0621 | [-35.08346846195131, 77.20768721195131] |
| compression_pending_peak | 0 | [0.0, 0.0] |
| compression_enqueued | 0 | [0.0, 0.0] |
| compression_completed | 2.8 | [-1.7958344113734555, 7.395834411373455] |
| compression_completion_fraction | 0.00625 | [-0.0040085589539585985, 0.0165085589539586] |
| compression_completion_per_s | 0.00681807 | [-0.006886000882971972, 0.02052214028792762] |
| compression_committed | 4.6 | [-1.64545882671678, 10.84545882671678] |
| compression_failed | 0 | [0.0, 0.0] |
| decompression_hits | 0 | [0.0, 0.0] |

## Paired singleton minus baseline (n=5)

| Metric | Mean difference | 95% CI |
|---|---:|---|
| requests | 0 | [0.0, 0.0] |
| duration_s | 0.151438 | [-0.7849292404605599, 1.0878047674550266] |
| token_cache_hit_rate | 0 | [0.0, 0.0] |
| evicted_entries | 0 | [0.0, 0.0] |
| evicted_compressed_entries | 3.4 | [-3.146761005818291, 9.94676100581829] |
| evicted_kv_tokens | -145.8 | [-493.7245771867311, 202.12457718673107] |
| recomputed_prefix_tokens | 0 | [0.0, 0.0] |
| request_throughput_rps | -0.000343541 | [-0.0024630562844193695, 0.0017759744343764775] |
| output_throughput_tps | -0.0439732 | [-0.3152712044056793, 0.22732472760018912] |
| mean_ttft_ms | -0.311868 | [-6.39385695267152, 5.77012120211187] |
| p50_ttft_ms | -0.115239 | [-1.6489374173469582, 1.4184604121036568] |
| p95_ttft_ms | -2.85699 | [-11.561485116448496, 5.847510580983256] |
| p99_ttft_ms | -5.14442 | [-20.090661605147396, 9.801816304472956] |
| mean_tpot_ms | 0.0270791 | [-0.09077758567661251, 0.14493582960540324] |
| persistent_cache_mib | 0 | [0.0, 0.0] |
| peak_cache_state_mib | -4.78125 | [-58.043450929990094, 48.480950929990094] |
| peak_staging_mib | -4.78125 | [-58.043450929990094, 48.480950929990094] |
| peak_process_cuda_mib | 0.383301 | [-1.6029681971946939, 2.369569759694694] |
| compression_pending_peak | -0.2 | [-0.7552890210395586, 0.35528902103955856] |
| compression_enqueued | 0 | [0.0, 0.0] |
| compression_completed | 1 | [-4.757359109466785, 6.757359109466785] |
| compression_completion_fraction | 0.00223214 | [-0.010619105155059816, 0.01508339086934549] |
| compression_completion_per_s | 0.00265433 | [-0.013012643484600848, 0.018321306361357134] |
| compression_committed | 3.4 | [-3.146761005818291, 9.94676100581829] |
| compression_failed | 0 | [0.0, 0.0] |
| decompression_hits | 0 | [0.0, 0.0] |
