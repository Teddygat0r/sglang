# Native cache-allocation sweep

Qwen3.5-4B, rank 16, concurrency 4, fixed ~14.19 GiB cache ceiling. 64 shared prefixes × 3 passes, 2048 prefix + 16 suffix tokens, 128 output tokens. LRU; graphs and overlap disabled. Five repetitions, reversed configuration order on odd repetitions.

Ceiling includes KV and dense/compressed state pools plus staging; excludes model weights, arithmetic workspace and allocator reserve. Staging reservation is not runtime enforcement. Exploratory allocation screen, not an independently validated winner or quality evaluation.

## off (n=5)

| Metric | Mean | 95% CI |
|---|---:|---|
| requests | 192 | [192.0, 192.0] |
| duration_s | 322.644 | [320.6520751534534, 324.63613433478946] |
| token_cache_hit_rate | 0.151195 | [0.1303815417913235, 0.17200863908722946] |
| evicted_entries | 420.4 | [416.3194757160504, 424.48052428394953] |
| evicted_compressed_entries | 0 | [0.0, 0.0] |
| evicted_kv_tokens | 268883 | [260582.3006042469, 277183.6993957531] |
| recomputed_prefix_tokens | 202227 | [193979.04043340002, 210475.3595666] |
| request_throughput_rps | 0.595095 | [0.5914446380336601, 0.598744613972985] |
| output_throughput_tps | 76.1721 | [75.70491366830849, 76.63931058854207] |
| mean_ttft_ms | 1036.7 | [1005.5968741100751, 1067.8037781821095] |
| p50_ttft_ms | 1094.6 | [1091.5957304420679, 1097.6123140421898] |
| p95_ttft_ms | 1522.83 | [1518.3474926922024, 1527.3161310431062] |
| p99_ttft_ms | 1532.23 | [1529.2296366116177, 1535.23986432495] |
| mean_tpot_ms | 44.758 | [44.62485295259551, 44.89124326766947] |
| persistent_cache_mib | 14529.2 | [14529.15625, 14529.15625] |
| peak_cache_state_mib | 14529.2 | [14529.15625, 14529.15625] |
| peak_staging_mib | 0 | [0.0, 0.0] |
| peak_process_cuda_mib | 23998.8 | [23998.455897457643, 23999.063047854856] |
| compression_pending_peak | 0 | [0.0, 0.0] |
| compression_enqueued | 0 | [0.0, 0.0] |
| compression_completed | 0 | [0.0, 0.0] |
| compression_completion_fraction | 0 | [0.0, 0.0] |
| compression_completion_per_s | 0 | [0.0, 0.0] |
| compression_committed | 0 | [0.0, 0.0] |
| compression_failed | 0 | [0.0, 0.0] |
| decompression_hits | 0 | [0.0, 0.0] |

## on_f16 (n=5)

| Metric | Mean | 95% CI |
|---|---:|---|
| requests | 192 | [192.0, 192.0] |
| duration_s | 291.592 | [291.4021808028913, 291.7810302584671] |
| token_cache_hit_rate | 0.653747 | [0.6537467700258398, 0.6537467700258398] |
| evicted_entries | 86 | [86.0, 86.0] |
| evicted_compressed_entries | 63.8 | [60.03384662330312, 67.56615337669687] |
| evicted_kv_tokens | 7252.4 | [7044.171059469719, 7460.6289405302805] |
| recomputed_prefix_tokens | 3072 | [3072.0, 3072.0] |
| request_throughput_rps | 0.658455 | [0.6580274372671211, 0.6588831898918219] |
| output_throughput_tps | 84.2823 | [84.2275119701915, 84.3370483061532] |
| mean_ttft_ms | 512.999 | [510.93903052457415, 515.0584794729192] |
| p50_ttft_ms | 166.701 | [166.17030224422106, 167.2310242184054] |
| p95_ttft_ms | 1588.37 | [1584.3860506697156, 1592.3604686127446] |
| p99_ttft_ms | 1600.42 | [1591.5649531619129, 1609.2681701792899] |
| mean_tpot_ms | 43.7875 | [43.7536847825144, 43.82140819349402] |
| persistent_cache_mib | 13742.7 | [13742.6875, 13742.6875] |
| peak_cache_state_mib | 14062 | [14039.477618764386, 14084.541131235614] |
| peak_staging_mib | 319.322 | [296.79011876438517, 341.8536312356148] |
| peak_process_cuda_mib | 24499.7 | [24275.0567725713, 24724.274477428702] |
| compression_pending_peak | 7.6 | [6.919912619341744, 8.280087380658255] |
| compression_enqueued | 448 | [448.0, 448.0] |
| compression_completed | 444 | [444.0, 444.0] |
| compression_completion_fraction | 0.991071 | [0.9910714285714286, 0.9910714285714286] |
| compression_completion_per_s | 1.52268 | [1.5216884486802174, 1.5236673766248383] |
| compression_committed | 421.8 | [418.03384662330313, 425.5661533766969] |
| compression_failed | 0 | [0.0, 0.0] |
| decompression_hits | 128 | [128.0, 128.0] |

## on_f32 (n=5)

| Metric | Mean | 95% CI |
|---|---:|---|
| requests | 192 | [192.0, 192.0] |
| duration_s | 292.638 | [291.2816186930565, 293.99461508136716] |
| token_cache_hit_rate | 0.645486 | [0.6398612223079885, 0.6511109999142337] |
| evicted_entries | 147.6 | [146.48942195792088, 148.7105780420791] |
| evicted_compressed_entries | 147.6 | [146.48942195792088, 148.7105780420791] |
| evicted_kv_tokens | 12344 | [9229.562134954731, 15458.437865045269] |
| recomputed_prefix_tokens | 6345.6 | [4116.524065988124, 8574.675934011877] |
| request_throughput_rps | 0.656108 | [0.6530738763907135, 0.6591416171227676] |
| output_throughput_tps | 83.9818 | [83.59345617801132, 84.37012699171426] |
| mean_ttft_ms | 524.137 | [516.1448862943803, 532.128772578357] |
| p50_ttft_ms | 166.241 | [165.15275817602216, 167.32844356209696] |
| p95_ttft_ms | 1591.02 | [1585.5195701726975, 1596.5261896094976] |
| p99_ttft_ms | 1606.07 | [1596.8022345824618, 1615.3326091320735] |
| mean_tpot_ms | 43.871 | [43.70665234805553, 44.03528088871941] |
| persistent_cache_mib | 13738.4 | [13738.375, 13738.375] |
| peak_cache_state_mib | 14053 | [14000.913100531905, 14104.993149468095] |
| peak_staging_mib | 314.578 | [262.53810053190546, 366.61814946809454] |
| peak_process_cuda_mib | 24425.2 | [24399.04641346013, 24451.39128185237] |
| compression_pending_peak | 7.6 | [6.919912619341744, 8.280087380658255] |
| compression_enqueued | 449.6 | [448.4894219579209, 450.71057804207913] |
| compression_completed | 445.6 | [444.4894219579209, 446.71057804207913] |
| compression_completion_fraction | 0.991103 | [0.9910812311619505, 0.9911251182795559] |
| compression_completion_per_s | 1.52271 | [1.51930404250963, 1.5261115397083516] |
| compression_committed | 445.6 | [444.4894219579209, 446.71057804207913] |
| compression_failed | 0 | [0.0, 0.0] |
| decompression_hits | 126.4 | [125.2894219579209, 127.51057804207912] |

## on_f64 (n=5)

| Metric | Mean | 95% CI |
|---|---:|---|
| requests | 192 | [192.0, 192.0] |
| duration_s | 313.823 | [312.85977438188496, 314.78656671472487] |
| token_cache_hit_rate | 0.336022 | [0.32373751414214114, 0.3483070595012697] |
| evicted_entries | 327.2 | [324.8116116119, 329.5883883881] |
| evicted_compressed_entries | 327.2 | [324.8116116119, 329.5883883881] |
| evicted_kv_tokens | 139759 | [133175.42442414953, 146342.57557585047] |
| recomputed_prefix_tokens | 128982 | [124114.09200436083, 133850.70799563915] |
| request_throughput_rps | 0.611813 | [0.6099333783643858, 0.6136916270484013] |
| output_throughput_tps | 78.312 | [78.07147243064138, 78.55252826219537] |
| mean_ttft_ms | 880.793 | [858.1913797754341, 903.3938492583254] |
| p50_ttft_ms | 809.436 | [773.0165589931323, 845.8544961777613] |
| p95_ttft_ms | 1594.29 | [1592.5880852652997, 1595.9922865839392] |
| p99_ttft_ms | 1604.12 | [1601.2169521953938, 1607.0305309090736] |
| mean_tpot_ms | 44.5384 | [44.48232821401377, 44.59443609691758] |
| persistent_cache_mib | 13742.9 | [13742.921875, 13742.921875] |
| peak_cache_state_mib | 14079 | [14006.039626157108, 14152.02912384289] |
| peak_staging_mib | 336.113 | [263.1177511571089, 409.1072488428911] |
| peak_process_cuda_mib | 24421.1 | [24417.961180754988, 24424.15542080751] |
| compression_pending_peak | 7.4 | [6.719912619341745, 8.080087380658256] |
| compression_enqueued | 510.2 | [507.8116116119, 512.5883883880999] |
| compression_completed | 506.2 | [503.8116116119, 508.5883883881] |
| compression_completion_fraction | 0.99216 | [0.992123194715304, 0.9921965017739902] |
| compression_completion_per_s | 1.61301 | [1.608770915056292, 1.6172446942020489] |
| compression_committed | 506.2 | [503.8116116119, 508.5883883881] |
| compression_failed | 0 | [0.0, 0.0] |
| decompression_hits | 65.8 | [63.411611611900014, 68.18838838809998] |

## Paired on_f16 minus off (n=5)

| Metric | Mean difference | 95% CI |
|---|---:|---|
| requests | 0 | [0.0, 0.0] |
| duration_s | -31.0525 | [-33.0272795960321, -29.077718830852312] |
| token_cache_hit_rate | 0.502552 | [0.48173813093861034, 0.5233652282345164] |
| evicted_entries | -334.4 | [-338.48052428394953, -330.3194757160504] |
| evicted_compressed_entries | 63.8 | [60.03384662330312, 67.56615337669687] |
| evicted_kv_tokens | -261631 | [-269986.7513216088, -253274.44867839126] |
| recomputed_prefix_tokens | -199155 | [-207403.3595666, -190907.04043340002] |
| request_throughput_rps | 0.0633607 | [0.059745183305505184, 0.06697619184679271] |
| output_throughput_tps | 8.11017 | [7.6473834631046635, 8.572952556389467] |
| mean_ttft_ms | -523.702 | [-555.4145240458313, -491.9886182488599] |
| p50_ttft_ms | -927.903 | [-930.6331309194097, -925.1735871022215] |
| p95_ttft_ms | 65.5414 | [60.10041764479157, 70.98247790235999] |
| p99_ttft_ms | 68.1818 | [59.95649729184474, 76.40712511279021] |
| mean_tpot_ms | -0.970502 | [-1.1243232578967206, -0.8166799863598319] |
| persistent_cache_mib | -786.469 | [-786.46875, -786.46875] |
| peak_cache_state_mib | -467.147 | [-489.67863123561483, -444.6151187643852] |
| peak_staging_mib | 319.322 | [296.79011876438517, 341.8536312356148] |
| peak_process_cuda_mib | 500.906 | [276.3182160196797, 725.4940886678203] |
| compression_pending_peak | 7.6 | [6.919912619341744, 8.280087380658255] |
| compression_enqueued | 448 | [448.0, 448.0] |
| compression_completed | 444 | [444.0, 444.0] |
| compression_completion_fraction | 0.991071 | [0.9910714285714286, 0.9910714285714286] |
| compression_completion_per_s | 1.52268 | [1.5216884486802174, 1.5236673766248383] |
| compression_committed | 421.8 | [418.03384662330313, 425.5661533766969] |
| compression_failed | 0 | [0.0, 0.0] |
| decompression_hits | 128 | [128.0, 128.0] |

## Paired on_f32 minus off (n=5)

| Metric | Mean difference | 95% CI |
|---|---:|---|
| requests | 0 | [0.0, 0.0] |
| duration_s | -30.006 | [-32.91547570232583, -27.096500011493443] |
| token_cache_hit_rate | 0.494291 | [0.4712692145079087, 0.5173128268357605] |
| evicted_entries | -272.8 | [-277.3111893325981, -268.28881066740195] |
| evicted_compressed_entries | 147.6 | [146.48942195792088, 148.7105780420791] |
| evicted_kv_tokens | -256539 | [-264221.88019581913, -248856.1198041809] |
| recomputed_prefix_tokens | -195882 | [-205004.86552108987, -186758.33447891014] |
| request_throughput_rps | 0.0610131 | [0.05523041040801959, 0.06679583109881643] |
| output_throughput_tps | 7.80968 | [7.069492532226508, 8.549866380648503] |
| mean_ttft_ms | -512.563 | [-546.3503440516174, -478.7766493678301] |
| p50_ttft_ms | -928.363 | [-931.1513838097131, -925.5754589364254] |
| p95_ttft_ms | 68.1911 | [60.51328930415701, 75.86884674272943] |
| p99_ttft_ms | 73.8327 | [62.26627813286346, 85.39906464510399] |
| mean_tpot_ms | -0.887081 | [-1.1507695425331528, -0.6233934409568909] |
| persistent_cache_mib | -790.781 | [-790.78125, -790.78125] |
| peak_cache_state_mib | -476.203 | [-528.2431494680945, -424.16310053190546] |
| peak_staging_mib | 314.578 | [262.53810053190546, 366.61814946809454] |
| peak_process_cuda_mib | 426.459 | [400.3133997263548, 452.60535027364523] |
| compression_pending_peak | 7.6 | [6.919912619341744, 8.280087380658255] |
| compression_enqueued | 449.6 | [448.4894219579209, 450.71057804207913] |
| compression_completed | 445.6 | [444.4894219579209, 446.71057804207913] |
| compression_completion_fraction | 0.991103 | [0.9910812311619505, 0.9911251182795559] |
| compression_completion_per_s | 1.52271 | [1.51930404250963, 1.5261115397083516] |
| compression_committed | 445.6 | [444.4894219579209, 446.71057804207913] |
| compression_failed | 0 | [0.0, 0.0] |
| decompression_hits | 126.4 | [125.2894219579209, 127.51057804207912] |

## Paired on_f64 minus off (n=5)

| Metric | Mean difference | 95% CI |
|---|---:|---|
| requests | 0 | [0.0, 0.0] |
| duration_s | -8.82093 | [-11.069700770455809, -6.572167621177224] |
| token_cache_hit_rate | 0.184827 | [0.15712827711642782, 0.21252611564843008] |
| evicted_entries | -93.2 | [-98.5693892910592, -87.83061070894081] |
| evicted_compressed_entries | 327.2 | [324.8116116119, 329.5883883881] |
| evicted_kv_tokens | -129124 | [-141400.78872766136, -116847.21127233864] |
| recomputed_prefix_tokens | -73244.8 | [-84221.54931808506, -62268.05068191495] |
| request_throughput_rps | 0.0167179 | [0.012544249963795821, 0.02089150344234602] |
| output_throughput_tps | 2.13989 | [1.6056639953658651, 2.6741124406202905] |
| mean_ttft_ms | -155.908 | [-200.70392049447145, -111.11150276395362] |
| p50_ttft_ms | -285.168 | [-321.75726045353593, -248.5797288598281] |
| p95_ttft_ms | 71.4584 | [68.00633847118985, 74.91040964274038] |
| p99_ttft_ms | 71.889 | [68.28064018243823, 75.49734198546138] |
| mean_tpot_ms | -0.219666 | [-0.3497128229081915, -0.08961908642543448] |
| persistent_cache_mib | -786.234 | [-786.234375, -786.234375] |
| peak_cache_state_mib | -450.122 | [-523.116623842891, -377.1271261571089] |
| peak_staging_mib | 336.113 | [263.1177511571089, 409.1072488428911] |
| peak_process_cuda_mib | 422.299 | [419.0857874337944, 425.5118688162056] |
| compression_pending_peak | 7.4 | [6.719912619341745, 8.080087380658256] |
| compression_enqueued | 510.2 | [507.8116116119, 512.5883883880999] |
| compression_completed | 506.2 | [503.8116116119, 508.5883883881] |
| compression_completion_fraction | 0.99216 | [0.992123194715304, 0.9921965017739902] |
| compression_completion_per_s | 1.61301 | [1.608770915056292, 1.6172446942020489] |
| compression_committed | 506.2 | [503.8116116119, 508.5883883881] |
| compression_failed | 0 | [0.0, 0.0] |
| decompression_hits | 65.8 | [63.411611611900014, 68.18838838809998] |
