# Matched-memory rank-16 state compression: combined paper report

## 1. Answer to the research question

**Yes, increased retained state capacity can translate into live serving gains under matched cache-memory ceilings, but only with suitable allocation and enough memory to retain useful prefixes.** The evidence includes negative results: the original partition regressed, the smallest budgets still regress after reallocation, and concurrent P95/P99 TTFT worsens even where mean TTFT and throughput improve.

This document consolidates three completed studies without pooling incompatible workloads or tuning samples. All numbers below are recomputed from saved per-run results, with raw-request/memory checks. It covers **108 confirmatory runs (20,736 requests)**: 24 original-allocation runs, 24 held-out reallocation runs, and 60 concurrent runs. A separate 18-run exploratory screen is described below but excluded from confirmatory statistics. Pilots and failed attempts are excluded.

The report addresses the live capacity/overhead question. It does not re-estimate the paper's separate simulation or quality results; their numerical evidence is not substituted with these serving measurements.

## 2. Design and reproducibility

All studies use Qwen/Qwen3.5-4B and rank-16 asynchronous state compression on one NVIDIA GB10 (DGX Spark). The recorded model snapshot is `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Study A provenance records repository commit `b2982df42f5be1a21500b55545670d37ee10118a`, Python 3.12.12, and driver 580.95.05. Later study protocol files record source hashes; consult those rather than assuming an upstream release. Benchmarks use separate observation hooks and, for B/C, benchmark-only allocation/policy overrides. They do not establish a production-integrated allocator.

| Study | Purpose | Concurrency | Output tokens | Repetitions per mode/config | Cache ceilings |
|---|---|---:|---:|---:|---|
| A | Diagnose original partition | 1 | 8 | 6 | 9.20, 17.26 GiB |
| B | Validate reallocated cache on held-out seeds | 1 | 8 | 6 | 9.20, 17.26 GiB |
| C | Test concurrent serving and longer outputs | 2, 4 | 128 | 5 | 11.12, 14.19, 17.26 GiB |

Each confirmatory off/on pair uses the same budget and token trace. Mode order alternates by repetition; Study C also reverses configuration order on odd repetitions. Five repetitions give a 3:2 starting-order split, not perfect counterbalancing. Each run uses a fresh server, two disjoint warmup requests, a drain of pending warmup compression, and a cache flush. Startup and warmup are excluded. CUDA graphs, piecewise CUDA graphs, and overlap scheduling are disabled in all three studies. Context length is 4096 and chunked prefill size is 2048.

### Workload

This is a **custom synthetic shared-prefix workload**, not SGLang's built-in `generated-shared-prefix` dataset and not production traffic. Each run has 64 distinct 2048-token prefixes, three independently shuffled passes, and a unique 16-token suffix per request: 192 requests of 2064 input tokens. Temperature is zero and EOS is ignored to force the stated output length. Prefixes are random token IDs, not natural-language prompts. The prefix boundary aligns to the nominal prefill checkpoint boundary.

The first pass is cold; passes two/three offer reuse. Perfect full-prefix retention would yield `2/3 × 2048/2064 = 66.1499%` overall input-token hit rate. Thus 65–66% is near the workload ceiling, not poor reuse. Study C uses a client semaphore at concurrency 2/4, with a barrier between passes; prefixes do not overlap with themselves within a pass. Actual execution order can differ across modes due to service times.

### Memory fairness and scope

The matched ceiling covers allocated attention KV storage, full/compressed Mamba state pools, and observed owned snapshot/result staging. **It is not a total-process VRAM cap.** Model weights, arithmetic workspaces (including SVD scratch), allocator reserve, and management metadata are outside this cache-state accounting. Peak process CUDA allocated memory is reported separately; it is not reserved VRAM. Equal ceilings do not require equal observed peaks or equal useful occupancy.

One full state occupies 49.125 MiB; one rank-16 compressed state occupies 13.171875 MiB (about 3.73× smaller, including convolution state). The full pool includes an extra sentinel slot in byte accounting. Study A assigns leftover whole-token capacity to KV; B/C hold KV at 262144 tokens (8192.03125 MiB including the sentinel). Staging allowances are charged inside the ceiling: 393 MiB in A/B and 786 MiB in C. These are conservative fixed allowances, not an optimized allocation or a bound for arbitrary queue growth.

### Metric definitions and statistics

- Token hit rate: sum of server-reported cached input tokens divided by total input tokens; includes the cold pass.
- Evicted entries: lost saved Mamba states, including internal state tombstones and compressed victims. This is not the number of unique prefix groups. KV-token eviction is reported separately.
- Recomputed prefix tokens: sum of `max(0, 2048 - cached_tokens)` in passes two/three; excludes compulsory first-pass and suffix work. This is a metadata-derived proxy, not a direct GPU work counter.
- TTFT: client submission to first streaming event reporting output tokens. In C it excludes waiting for a client semaphore slot but includes server waiting after submission.
- TPOT: elapsed time from first to last output event divided by output tokens minus one (7 or 127). It is a streaming-client measure, not kernel-only decode time.
- Throughput: completed requests or generated output tokens divided by measured wall time, including pass-boundary telemetry and barriers.
- Compression activity: outstanding-job peak, completions/second, and completed/enqueued fraction at the final observation. Outstanding includes in-flight work, not only waiting queue items. Fractions below 100% do not imply failure; final jobs need not be drained before the measurement ends.
- Each cell is the mean and two-sided Student-t 95% CI across independent run repetitions: `mean ± t(0.975,n−1) × sd/√n`. Paired changes are computed from matched on-minus-off repetitions. Percentile intervals summarize variation in per-run percentiles, not pooled-request quantile uncertainty. Intervals are unadjusted for multiple comparisons; screening winners are not automatically statistically superior.

## 3. Results overview

The following percentages are ratios of mode means, not means of per-pair percentages. Positive throughput change is beneficial; negative TTFT change is beneficial. Full metric and paired-difference intervals follow.

| Study | Budget (GiB) | Concurrency | Hit rate off → on (%) | Requests/s off → on | Throughput change | Mean TTFT off → on (ms) | TTFT change |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 9.20 | 1 | 0.52 → 0.00 | 1.2887 → 1.2682 | -1.59% | 423.2 → 430.6 | +1.76% |
| A | 17.26 | 1 | 36.95 → 6.80 | 1.5652 → 1.3100 | -16.30% | 288.0 → 405.4 | +40.78% |
| B | 9.20 | 1 | 0.78 → 1.21 | 1.2888 → 1.2725 | -1.26% | 422.5 → 427.3 | +1.14% |
| B | 17.26 | 1 | 36.61 → 66.15 | 1.5623 → 1.8430 | +17.96% | 288.9 → 186.5 | -35.43% |
| C | 11.12 | 2 | 3.60 → 3.50 | 0.3347 → 0.3327 | -0.59% | 736.3 → 760.5 | +3.29% |
| C | 11.12 | 4 | 3.27 → 3.88 | 0.5790 → 0.5716 | -1.29% | 1170.4 → 1220.3 | +4.27% |
| C | 14.19 | 2 | 14.41 → 65.38 | 0.3392 → 0.3587 | +5.73% | 656.0 → 332.7 | -49.29% |
| C | 14.19 | 4 | 13.98 → 64.87 | 0.5922 → 0.6558 | +10.73% | 1055.4 → 520.1 | -50.72% |
| C | 17.26 | 2 | 33.36 → 65.89 | 0.3471 → 0.3587 | +3.36% | 521.5 → 329.3 | -36.86% |
| C | 17.26 | 4 | 32.04 → 65.37 | 0.6159 → 0.6556 | +6.43% | 852.5 → 517.4 | -39.31% |

## 4. Interpretation and paper-ready conclusion

Study A shows why compression alone is insufficient: at 17.26 GiB, the original partition provides 192 dense slots off, but only 81 compressed slots on alongside 162 dense slots. A saved on-run ends with 161 dense slots free while the compressed pool is full and 479 compressed entries have been evicted. Compression funnels cached states into a small pool while allocated dense memory remains largely unused. The hit-rate and throughput regression is consistent with this retention failure.

Study B restores the capacity benefit: at 17.26 GiB, 16 full slots plus 626 compressed slots eliminate state eviction and repeat-prefix recomputation in the sequential trace. Throughput increases about 18% and mean TTFT falls about 35%. The severe 9.20 GiB case remains approximately 1.3% slower. The held-out B configuration uses reuse-aware eviction, so it is not a pure allocation-only ablation against A; however, all three allocation-screen candidates with ordinary LRU also retained every reusable prefix, and C validates gains using ordinary LRU.

Study C establishes concurrent benefits on this execution path: at 14.19 GiB, throughput improves 5.7%/10.7% at concurrency 2/4 while mean TTFT falls approximately 49%/51%. At 17.26 GiB the corresponding throughput gains are 3.4%/6.4%. Longer generation makes end-to-end throughput gains smaller than prefill-latency gains; this is a plausible interpretation, not an isolated output-length ablation because other configuration details also change. The 11.12 GiB configurations lose 0.6–1.3% throughput and do not materially improve reuse.

**Tail-latency tradeoff:** C's P95/P99 TTFT worsens in every configuration. At 14.19 GiB/concurrency 4, P95 rises from about 1527 to 1591 ms while mean TTFT falls from 1055 to 520 ms. Do not claim uniformly improved latency. TPOT is nearly unchanged at concurrency 2 and modestly better at concurrency 4 at the two larger budgets.

**Eviction versus recomputation:** C's largest-budget on-runs have zero state evictions, but still report 1024/3072 recomputed prefix tokens per run at concurrency 2/4. Therefore zero eviction must not be described as zero recomputation or perfect token reuse. The specific checkpoint/matching cause has not been isolated. At the middle budget, on-runs still evict approximately 147–149 state entries while preserving almost all reusable prefix tokens; not every evicted state is needed again.

**Suggested paper text:** “We evaluate rank-16 state compression for Qwen3.5-4B on a single GB10 using matched cache-state memory ceilings and synthetic shared-prefix traffic. After correcting the full/compressed pool partition, compression converts increased state capacity into live serving gains: with 128-token outputs at concurrency four and a 14.19 GiB cache ceiling, token cache-hit rate increases from 13.98% to 64.87%, request throughput increases by 10.7%, and mean TTFT decreases by 50.7%. Benefits depend on available memory: the smallest tested concurrent configuration regresses slightly. P95/P99 TTFT also increases, so improvements are in average responsiveness and achieved throughput rather than all latency metrics.”

This is the desired live bridge between capacity and overhead: useful retained prefixes reduce repeated prefill enough to outweigh the compression-enabled path's costs in suitable regimes. It does not isolate SVD kernel overhead causally, establish realistic-traffic generalization, prove generation quality, or establish maximum production throughput.

## 5. Complete confirmatory tables

All entries: mean [95% CI]. Differences are on minus off; percentage metrics use percentage-point differences. Compression-off completion metrics are N/A. Memory is MiB.

### A: 9420.15625 MiB, concurrency 1, n=6 per mode

Full slots off/on: 24/14; compressed slots on: 7; on staging allowance: 393 MiB. KV tokens off/on: 262144/262337.

| Metric | Off | On | Paired change |
|---|---:|---:|---:|
| Token cache-hit rate (%) | 0.517 [0.174, 0.860] | 0.000 [0.000, 0.000] | -0.517 [-0.860, -0.174] |
| Evicted state entries | 553.000 [552.336, 553.664] | 568.000 [568.000, 568.000] | 15.000 [14.336, 15.664] |
| Evicted compressed entries | 0.000 [0.000, 0.000] | 568.000 [568.000, 568.000] | 568.000 [568.000, 568.000] |
| Evicted attention tokens | 378,674.667 [377,056.774, 380,292.559] | 391,419.000 [391,419.000, 391,419.000] | 12,744.333 [11,126.441, 14,362.226] |
| Recomputed prefix tokens | 260,096.000 [258,736.699, 261,455.301] | 262,144.000 [262,144.000, 262,144.000] | 2,048.000 [688.699, 3,407.301] |
| Request throughput (requests/s) | 1.289 [1.283, 1.294] | 1.268 [1.266, 1.271] | -0.020 [-0.028, -0.013] |
| Output throughput (tokens/s) | 10.310 [10.264, 10.356] | 10.146 [10.127, 10.165] | -0.164 [-0.220, -0.107] |
| Mean TTFT (ms) | 423.205 [420.306, 426.105] | 430.635 [429.389, 431.881] | 7.430 [3.746, 11.113] |
| P50 TTFT (ms) | 424.944 [422.595, 427.293] | 430.196 [428.855, 431.538] | 5.252 [1.880, 8.625] |
| P95 TTFT (ms) | 432.513 [429.735, 435.290] | 437.898 [435.666, 440.130] | 5.385 [1.822, 8.949] |
| P99 TTFT (ms) | 437.224 [435.189, 439.259] | 449.261 [445.577, 452.944] | 12.036 [9.525, 14.548] |
| Mean TPOT (ms) | 50.363 [50.268, 50.459] | 51.099 [51.012, 51.185] | 0.735 [0.585, 0.885] |
| Peak cache-state memory (MiB) | 9,420.156 [9,420.156, 9,420.156] | 9,143.188 [9,122.623, 9,163.752] | -276.969 [-297.533, -256.404] |
| Peak staging memory (MiB) | 0.000 [0.000, 0.000] | 116.047 [95.482, 136.612] | 116.047 [95.482, 136.612] |
| Peak process CUDA allocated (MiB) | 18,887.226 [18,887.217, 18,887.234] | 18,924.164 [18,851.763, 18,996.565] | 36.938 [-35.461, 109.337] |
| Peak outstanding compression jobs | 0.000 [0.000, 0.000] | 2.000 [2.000, 2.000] | 2.000 [2.000, 2.000] |
| Compression completions/s | N/A | 3.798 [3.791, 3.805] | N/A |
| Compression completion fraction (%) | N/A | 99.826 [99.826, 99.826] | N/A |
| Compression failures | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |

### A: 17673.15625 MiB, concurrency 1, n=6 per mode

Full slots off/on: 192/162; compressed slots on: 81; on staging allowance: 393 MiB. KV tokens off/on: 262144/262586.

| Metric | Off | On | Paired change |
|---|---:|---:|---:|
| Token cache-hit rate (%) | 36.951 [35.144, 38.758] | 6.804 [5.502, 8.107] | -30.146 [-32.413, -27.880] |
| Evicted state entries | 314.500 [311.004, 317.996] | 480.833 [478.313, 483.353] | 166.333 [161.947, 170.719] |
| Evicted compressed entries | 0.000 [0.000, 0.000] | 480.833 [478.313, 483.353] | 480.833 [478.313, 483.353] |
| Evicted attention tokens | 117,973.333 [111,395.569, 124,551.097] | 310,972.000 [304,742.899, 317,201.101] | 192,998.667 [183,939.490, 202,057.844] |
| Recomputed prefix tokens | 115,712.000 [108,551.436, 122,872.564] | 235,178.667 [230,017.496, 240,339.837] | 119,466.667 [110,484.300, 128,449.033] |
| Request throughput (requests/s) | 1.565 [1.550, 1.580] | 1.310 [1.302, 1.318] | -0.255 [-0.272, -0.238] |
| Output throughput (tokens/s) | 12.522 [12.401, 12.642] | 10.480 [10.415, 10.546] | -2.041 [-2.179, -1.903] |
| Mean TTFT (ms) | 288.000 [281.238, 294.761] | 405.449 [400.568, 410.331] | 117.450 [108.798, 126.102] |
| P50 TTFT (ms) | 419.226 [418.386, 420.065] | 429.702 [428.843, 430.560] | 10.476 [9.599, 11.353] |
| P95 TTFT (ms) | 429.063 [428.056, 430.071] | 436.661 [435.659, 437.664] | 7.598 [6.142, 9.054] |
| P99 TTFT (ms) | 433.289 [431.761, 434.817] | 448.593 [442.633, 454.553] | 15.304 [8.547, 22.061] |
| Mean TPOT (ms) | 50.106 [50.003, 50.208] | 51.097 [51.042, 51.153] | 0.992 [0.848, 1.135] |
| Peak cache-state memory (MiB) | 17,673.156 [17,673.156, 17,673.156] | 17,400.203 [17,380.694, 17,419.712] | -272.953 [-292.462, -253.444] |
| Peak staging memory (MiB) | 0.000 [0.000, 0.000] | 120.062 [100.553, 139.572] | 120.062 [100.553, 139.572] |
| Peak process CUDA allocated (MiB) | 27,138.404 [27,138.092, 27,138.716] | 27,138.963 [27,118.477, 27,159.450] | 0.559 [-20.062, 21.181] |
| Peak outstanding compression jobs | 0.000 [0.000, 0.000] | 2.333 [1.791, 2.875] | 2.333 [1.791, 2.875] |
| Compression completions/s | N/A | 3.833 [3.826, 3.841] | N/A |
| Compression completion fraction (%) | N/A | 99.822 [99.822, 99.823] | N/A |
| Compression failures | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |

### B: 9420.15625 MiB, concurrency 1, n=6 per mode

Full slots off/on: 24/8; compressed slots on: 29; on staging allowance: 393 MiB. KV tokens off/on: 262144/262144.

| Metric | Off | On | Paired change |
|---|---:|---:|---:|
| Token cache-hit rate (%) | 0.775 [0.206, 1.344] | 1.206 [0.389, 2.022] | 0.431 [-0.103, 0.964] |
| Evicted state entries | 552.500 [551.399, 553.601] | 543.667 [542.087, 545.247] | -8.833 [-9.865, -7.802] |
| Evicted compressed entries | 0.000 [0.000, 0.000] | 543.667 [542.087, 545.247] | 543.667 [542.087, 545.247] |
| Evicted attention tokens | 377,992.000 [375,737.854, 380,246.146] | 368,733.833 [364,658.462, 372,809.204] | -9,258.167 [-11,884.542, -6,631.791] |
| Recomputed prefix tokens | 259,072.000 [256,817.854, 261,326.146] | 257,365.333 [254,129.549, 260,601.118] | -1,706.667 [-3,819.787, 406.453] |
| Request throughput (requests/s) | 1.289 [1.284, 1.294] | 1.273 [1.266, 1.279] | -0.016 [-0.023, -0.010] |
| Output throughput (tokens/s) | 10.310 [10.268, 10.352] | 10.180 [10.126, 10.234] | -0.130 [-0.183, -0.077] |
| Mean TTFT (ms) | 422.457 [419.332, 425.581] | 427.292 [423.713, 430.871] | 4.835 [1.416, 8.254] |
| P50 TTFT (ms) | 425.050 [423.345, 426.755] | 431.546 [430.316, 432.776] | 6.496 [3.876, 9.116] |
| P95 TTFT (ms) | 433.300 [431.775, 434.824] | 438.509 [437.194, 439.825] | 5.210 [2.581, 7.838] |
| P99 TTFT (ms) | 440.040 [435.768, 444.312] | 445.080 [442.310, 447.850] | 5.040 [-0.421, 10.501] |
| Mean TPOT (ms) | 50.464 [50.388, 50.539] | 51.189 [51.075, 51.303] | 0.725 [0.575, 0.875] |
| Peak cache-state memory (MiB) | 9,420.156 [9,420.156, 9,420.156] | 9,126.195 [9,121.034, 9,131.357] | -293.961 [-299.122, -288.800] |
| Peak staging memory (MiB) | 0.000 [0.000, 0.000] | 110.055 [104.893, 115.216] | 110.055 [104.893, 115.216] |
| Peak process CUDA allocated (MiB) | 18,887.212 [18,887.193, 18,887.231] | 18,866.023 [18,865.743, 18,866.302] | -21.189 [-21.464, -20.915] |
| Peak outstanding compression jobs | 0.000 [0.000, 0.000] | 2.167 [1.738, 2.595] | 2.167 [1.738, 2.595] |
| Compression completions/s | N/A | 3.796 [3.785, 3.806] | N/A |
| Compression completion fraction (%) | N/A | 99.826 [99.825, 99.826] | N/A |
| Compression failures | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |

### B: 17673.15625 MiB, concurrency 1, n=6 per mode

Full slots off/on: 192/16; compressed slots on: 626; on staging allowance: 393 MiB. KV tokens off/on: 262144/262144.

| Metric | Off | On | Paired change |
|---|---:|---:|---:|
| Token cache-hit rate (%) | 36.606 [34.594, 38.619] | 66.150 [66.150, 66.150] | 29.543 [27.531, 31.556] |
| Evicted state entries | 315.167 [311.273, 319.060] | 0.000 [0.000, 0.000] | -315.167 [-319.060, -311.273] |
| Evicted compressed entries | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| Evicted attention tokens | 119,338.667 [111,719.705, 126,957.629] | 0.000 [0.000, 0.000] | -119,338.667 [-126,957.629, -111,719.705] |
| Recomputed prefix tokens | 117,077.333 [109,102.894, 125,051.773] | 0.000 [0.000, 0.000] | -117,077.333 [-125,051.773, -109,102.894] |
| Request throughput (requests/s) | 1.562 [1.544, 1.581] | 1.843 [1.839, 1.847] | 0.281 [0.262, 0.299] |
| Output throughput (tokens/s) | 12.499 [12.351, 12.646] | 14.744 [14.715, 14.773] | 2.245 [2.099, 2.391] |
| Mean TTFT (ms) | 288.893 [281.630, 296.157] | 186.549 [185.735, 187.362] | -102.345 [-109.110, -95.580] |
| P50 TTFT (ms) | 418.222 [415.899, 420.545] | 65.975 [64.922, 67.027] | -352.247 [-354.244, -350.250] |
| P95 TTFT (ms) | 428.837 [427.410, 430.264] | 434.609 [432.588, 436.630] | 5.772 [3.393, 8.151] |
| P99 TTFT (ms) | 433.964 [432.063, 435.865] | 441.880 [437.196, 446.564] | 7.916 [2.484, 13.347] |
| Mean TPOT (ms) | 50.149 [50.084, 50.215] | 50.838 [50.742, 50.933] | 0.688 [0.578, 0.798] |
| Peak cache-state memory (MiB) | 17,673.156 [17,673.156, 17,673.156] | 17,382.805 [17,377.643, 17,387.966] | -290.352 [-295.513, -285.190] |
| Peak staging memory (MiB) | 0.000 [0.000, 0.000] | 110.055 [104.893, 115.216] | 110.055 [104.893, 115.216] |
| Peak process CUDA allocated (MiB) | 27,138.561 [27,138.177, 27,138.944] | 27,093.736 [27,051.983, 27,135.490] | -44.824 [-86.568, -3.080] |
| Peak outstanding compression jobs | 0.000 [0.000, 0.000] | 2.167 [1.738, 2.595] | 2.167 [1.738, 2.595] |
| Compression completions/s | N/A | 4.291 [4.282, 4.299] | N/A |
| Compression completion fraction (%) | N/A | 99.777 [99.777, 99.777] | N/A |
| Compression failures | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |

### C: 11385.15625 MiB, concurrency 2, n=5 per mode

Full slots off/on: 64/32; compressed slots on: 59; on staging allowance: 786 MiB. KV tokens off/on: 262144/262144.

| Metric | Off | On | Paired change |
|---|---:|---:|---:|
| Token cache-hit rate (%) | 3.602 [2.597, 4.607] | 3.499 [2.359, 4.639] | -0.103 [-0.390, 0.184] |
| Evicted state entries | 507.000 [505.037, 508.963] | 508.200 [505.979, 510.421] | 1.200 [0.645, 1.755] |
| Evicted compressed entries | 0.000 [0.000, 0.000] | 508.200 [505.979, 510.421] | 508.200 [505.979, 510.421] |
| Evicted attention tokens | 359,915.400 [356,599.828, 363,230.972] | 358,686.600 [354,508.143, 362,865.057] | -1,228.800 [-2,621.619, 164.019] |
| Recomputed prefix tokens | 247,868.800 [243,885.718, 251,851.882] | 248,278.400 [243,762.148, 252,794.652] | 409.600 [-727.632, 1,546.832] |
| Request throughput (requests/s) | 0.335 [0.334, 0.335] | 0.333 [0.332, 0.333] | -0.002 [-0.003, -0.001] |
| Output throughput (tokens/s) | 42.845 [42.771, 42.919] | 42.591 [42.508, 42.675] | -0.254 [-0.322, -0.185] |
| Mean TTFT (ms) | 736.275 [728.821, 743.728] | 760.521 [750.396, 770.645] | 24.246 [17.891, 30.601] |
| P50 TTFT (ms) | 740.369 [738.680, 742.059] | 758.050 [754.960, 761.141] | 17.681 [14.090, 21.272] |
| P95 TTFT (ms) | 802.081 [799.289, 804.872] | 829.826 [827.425, 832.227] | 27.745 [24.412, 31.079] |
| P99 TTFT (ms) | 807.812 [802.542, 813.082] | 835.414 [833.123, 837.705] | 27.602 [20.791, 34.414] |
| Mean TPOT (ms) | 41.247 [41.199, 41.295] | 41.336 [41.303, 41.368] | 0.089 [0.041, 0.137] |
| Peak cache-state memory (MiB) | 11,385.156 [11,385.156, 11,385.156] | 10,796.753 [10,790.064, 10,803.443] | -588.403 [-595.093, -581.714] |
| Peak staging memory (MiB) | 0.000 [0.000, 0.000] | 206.456 [199.767, 213.146] | 206.456 [199.767, 213.146] |
| Peak process CUDA allocated (MiB) | 20,854.539 [20,854.227, 20,854.851] | 20,742.185 [20,651.418, 20,832.953] | -112.354 [-202.968, -21.740] |
| Peak outstanding compression jobs | 0.000 [0.000, 0.000] | 4.200 [3.645, 4.755] | 4.200 [3.645, 4.755] |
| Compression completions/s | N/A | 0.983 [0.981, 0.985] | N/A |
| Compression completion fraction (%) | N/A | 99.649 [99.647, 99.650] | N/A |
| Compression failures | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |

### C: 11385.15625 MiB, concurrency 4, n=5 per mode

Full slots off/on: 64/32; compressed slots on: 59; on staging allowance: 786 MiB. KV tokens off/on: 262144/262144.

| Metric | Off | On | Paired change |
|---|---:|---:|---:|
| Token cache-hit rate (%) | 3.267 [2.314, 4.220] | 3.882 [2.424, 5.339] | 0.615 [-0.526, 1.755] |
| Evicted state entries | 507.600 [505.717, 509.483] | 505.400 [502.541, 508.259] | -2.200 [-4.421, 0.021] |
| Evicted compressed entries | 0.000 [0.000, 0.000] | 505.400 [502.541, 508.259] | 505.400 [502.541, 508.259] |
| Evicted attention tokens | 360,325.000 [356,304.278, 364,345.722] | 352,256.600 [347,707.672, 356,805.528] | -8,068.400 [-11,840.172, -4,296.628] |
| Recomputed prefix tokens | 249,196.800 [245,418.682, 252,974.918] | 246,761.600 [240,984.432, 252,538.768] | -2,435.200 [-6,954.724, 2,084.324] |
| Request throughput (requests/s) | 0.579 [0.578, 0.580] | 0.572 [0.569, 0.574] | -0.007 [-0.009, -0.006] |
| Output throughput (tokens/s) | 74.116 [73.956, 74.275] | 73.159 [72.844, 73.473] | -0.957 [-1.200, -0.714] |
| Mean TTFT (ms) | 1,170.379 [1,152.042, 1,188.716] | 1,220.326 [1,197.393, 1,243.259] | 49.947 [43.542, 56.353] |
| P50 TTFT (ms) | 1,108.627 [1,103.836, 1,113.417] | 1,156.809 [1,150.851, 1,162.768] | 48.183 [40.507, 55.858] |
| P95 TTFT (ms) | 1,532.463 [1,525.577, 1,539.349] | 1,605.585 [1,596.527, 1,614.643] | 73.122 [59.349, 86.895] |
| P99 TTFT (ms) | 1,539.072 [1,531.763, 1,546.380] | 1,616.282 [1,607.367, 1,625.196] | 77.210 [64.807, 89.614] |
| Mean TPOT (ms) | 45.173 [45.033, 45.313] | 45.492 [45.379, 45.604] | 0.318 [0.177, 0.460] |
| Peak cache-state memory (MiB) | 11,385.156 [11,385.156, 11,385.156] | 10,878.484 [10,878.484, 10,878.484] | -506.672 [-506.672, -506.672] |
| Peak staging memory (MiB) | 0.000 [0.000, 0.000] | 288.188 [288.188, 288.188] | 288.188 [288.188, 288.188] |
| Peak process CUDA allocated (MiB) | 20,855.619 [20,855.162, 20,856.076] | 21,266.589 [21,265.144, 21,268.033] | 410.970 [409.286, 412.654] |
| Peak outstanding compression jobs | 0.000 [0.000, 0.000] | 7.000 [7.000, 7.000] | 7.000 [7.000, 7.000] |
| Compression completions/s | N/A | 1.680 [1.674, 1.686] | N/A |
| Compression completion fraction (%) | N/A | 99.296 [99.293, 99.300] | N/A |
| Compression failures | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |

### C: 14529.15625 MiB, concurrency 2, n=5 per mode

Full slots off/on: 128/32; compressed slots on: 298; on staging allowance: 786 MiB. KV tokens off/on: 262144/262144.

| Metric | Off | On | Paired change |
|---|---:|---:|---:|
| Token cache-hit rate (%) | 14.407 [12.664, 16.151] | 65.378 [64.739, 66.017] | 50.971 [49.285, 52.656] |
| Evicted state entries | 422.000 [418.600, 425.400] | 149.000 [147.758, 150.242] | -273.000 [-276.285, -269.715] |
| Evicted compressed entries | 0.000 [0.000, 0.000] | 149.000 [147.758, 150.242] | 149.000 [147.758, 150.242] |
| Evicted attention tokens | 270,874.000 [263,809.425, 277,938.575] | 11,401.200 [10,008.381, 12,794.019] | -259,472.800 [-266,350.403, -252,595.197] |
| Recomputed prefix tokens | 205,049.600 [198,139.889, 211,959.311] | 3,059.200 [526.015, 5,592.385] | -201,990.400 [-208,669.688, -195,311.112] |
| Request throughput (requests/s) | 0.339 [0.338, 0.340] | 0.359 [0.358, 0.359] | 0.019 [0.018, 0.021] |
| Output throughput (tokens/s) | 43.422 [43.266, 43.578] | 45.908 [45.836, 45.979] | 2.486 [2.341, 2.631] |
| Mean TTFT (ms) | 656.030 [643.057, 669.002] | 332.672 [327.088, 338.256] | -323.358 [-334.239, -312.477] |
| P50 TTFT (ms) | 734.964 [732.091, 737.836] | 134.399 [132.967, 135.832] | -600.564 [-602.869, -598.260] |
| P95 TTFT (ms) | 800.882 [798.672, 803.092] | 822.936 [820.936, 824.936] | 22.054 [21.082, 23.026] |
| P99 TTFT (ms) | 805.463 [802.823, 808.102] | 829.235 [825.897, 832.573] | 23.772 [20.378, 27.167] |
| Mean TPOT (ms) | 41.254 [41.155, 41.353] | 41.286 [41.196, 41.376] | 0.032 [-0.063, 0.126] |
| Peak cache-state memory (MiB) | 14,529.156 [14,529.156, 14,529.156] | 13,949.650 [13,941.457, 13,957.843] | -579.506 [-587.699, -571.313] |
| Peak staging memory (MiB) | 0.000 [0.000, 0.000] | 211.275 [203.082, 219.468] | 211.275 [203.082, 219.468] |
| Peak process CUDA allocated (MiB) | 23,997.814 [23,997.519, 23,998.109] | 23,746.893 [23,592.681, 23,901.104] | -250.921 [-405.270, -96.572] |
| Peak outstanding compression jobs | 0.000 [0.000, 0.000] | 4.000 [4.000, 4.000] | 4.000 [4.000, 4.000] |
| Compression completions/s | N/A | 0.835 [0.832, 0.838] | N/A |
| Compression completion fraction (%) | N/A | 99.555 [99.553, 99.556] | N/A |
| Compression failures | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |

### C: 14529.15625 MiB, concurrency 4, n=5 per mode

Full slots off/on: 128/32; compressed slots on: 298; on staging allowance: 786 MiB. KV tokens off/on: 262144/262144.

| Metric | Off | On | Paired change |
|---|---:|---:|---:|
| Token cache-hit rate (%) | 13.983 [12.017, 15.950] | 64.868 [64.427, 65.310] | 50.885 [49.190, 52.580] |
| Evicted state entries | 422.600 [418.713, 426.487] | 147.000 [146.122, 147.878] | -275.600 [-278.955, -272.245] |
| Evicted compressed entries | 0.000 [0.000, 0.000] | 147.000 [146.122, 147.878] | 147.000 [146.122, 147.878] |
| Evicted attention tokens | 272,096.400 [264,340.530, 279,852.270] | 10,705.600 [9,568.368, 11,842.832] | -261,390.800 [-268,892.712, -253,888.888] |
| Recomputed prefix tokens | 206,729.600 [198,936.809, 214,522.391] | 5,078.400 [3,329.144, 6,827.656] | -201,651.200 [-208,367.932, -194,934.468] |
| Request throughput (requests/s) | 0.592 [0.588, 0.596] | 0.656 [0.654, 0.657] | 0.064 [0.061, 0.066] |
| Output throughput (tokens/s) | 75.808 [75.298, 76.317] | 83.938 [83.745, 84.131] | 8.131 [7.776, 8.485] |
| Mean TTFT (ms) | 1,055.394 [1,034.262, 1,076.525] | 520.136 [515.218, 525.055] | -535.258 [-555.102, -515.413] |
| P50 TTFT (ms) | 1,098.533 [1,092.200, 1,104.866] | 167.342 [166.721, 167.963] | -931.191 [-937.874, -924.508] |
| P95 TTFT (ms) | 1,526.975 [1,518.917, 1,535.033] | 1,590.972 [1,585.639, 1,596.304] | 63.997 [52.472, 75.521] |
| P99 TTFT (ms) | 1,534.572 [1,526.134, 1,543.009] | 1,608.640 [1,602.358, 1,614.922] | 74.068 [67.592, 80.544] |
| Mean TPOT (ms) | 44.866 [44.605, 45.126] | 43.927 [43.816, 44.039] | -0.938 [-1.115, -0.762] |
| Peak cache-state memory (MiB) | 14,529.156 [14,529.156, 14,529.156] | 14,060.106 [14,016.203, 14,104.010] | -469.050 [-512.953, -425.147] |
| Peak staging memory (MiB) | 0.000 [0.000, 0.000] | 321.731 [277.828, 365.635] | 321.731 [277.828, 365.635] |
| Peak process CUDA allocated (MiB) | 23,998.501 [23,998.117, 23,998.886] | 24,427.913 [24,403.736, 24,452.091] | 429.412 [405.323, 453.501] |
| Peak outstanding compression jobs | 0.000 [0.000, 0.000] | 7.800 [7.245, 8.355] | 7.800 [7.245, 8.355] |
| Compression completions/s | N/A | 1.520 [1.516, 1.524] | N/A |
| Compression completion fraction (%) | N/A | 99.109 [99.107, 99.111] | N/A |
| Compression failures | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |

### C: 17673.15625 MiB, concurrency 2, n=5 per mode

Full slots off/on: 192/32; compressed slots on: 537; on staging allowance: 786 MiB. KV tokens off/on: 262144/262144.

| Metric | Off | On | Paired change |
|---|---:|---:|---:|
| Token cache-hit rate (%) | 33.362 [31.070, 35.655] | 65.891 [65.891, 65.891] | 32.529 [30.237, 34.821] |
| Evicted state entries | 321.200 [316.775, 325.625] | 0.000 [0.000, 0.000] | -321.200 [-325.625, -316.775] |
| Evicted compressed entries | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| Evicted attention tokens | 152,985.600 [144,789.132, 161,182.068] | 0.000 [0.000, 0.000] | -152,985.600 [-161,182.068, -144,789.132] |
| Recomputed prefix tokens | 129,932.800 [120,848.305, 139,017.295] | 1,024.000 [1,024.000, 1,024.000] | -128,908.800 [-137,993.295, -119,824.305] |
| Request throughput (requests/s) | 0.347 [0.346, 0.348] | 0.359 [0.358, 0.360] | 0.012 [0.011, 0.013] |
| Output throughput (tokens/s) | 44.424 [44.311, 44.537] | 45.918 [45.805, 46.031] | 1.493 [1.386, 1.601] |
| Mean TTFT (ms) | 521.509 [505.766, 537.253] | 329.306 [327.424, 331.188] | -192.204 [-208.966, -175.442] |
| P50 TTFT (ms) | 727.596 [724.702, 730.490] | 134.483 [132.329, 136.637] | -593.113 [-596.154, -590.072] |
| P95 TTFT (ms) | 798.806 [796.195, 801.418] | 821.767 [817.899, 825.634] | 22.960 [18.958, 26.962] |
| P99 TTFT (ms) | 805.262 [803.671, 806.853] | 828.615 [824.004, 833.226] | 23.353 [17.252, 29.453] |
| Mean TPOT (ms) | 41.265 [41.219, 41.310] | 41.302 [41.200, 41.405] | 0.038 [-0.079, 0.155] |
| Peak cache-state memory (MiB) | 17,673.156 [17,673.156, 17,673.156] | 17,100.138 [17,087.623, 17,112.652] | -573.019 [-585.534, -560.504] |
| Peak staging memory (MiB) | 0.000 [0.000, 0.000] | 213.684 [201.169, 226.199] | 213.684 [201.169, 226.199] |
| Peak process CUDA allocated (MiB) | 27,141.195 [27,140.582, 27,141.808] | 27,061.065 [26,851.875, 27,270.255] | -80.130 [-289.402, 129.141] |
| Peak outstanding compression jobs | 0.000 [0.000, 0.000] | 4.200 [3.645, 4.755] | 4.200 [3.645, 4.755] |
| Compression completions/s | N/A | 0.833 [0.831, 0.835] | N/A |
| Compression completion fraction (%) | N/A | 99.554 [99.554, 99.554] | N/A |
| Compression failures | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |

### C: 17673.15625 MiB, concurrency 4, n=5 per mode

Full slots off/on: 192/32; compressed slots on: 537; on staging allowance: 786 MiB. KV tokens off/on: 262144/262144.

| Metric | Off | On | Paired change |
|---|---:|---:|---:|
| Token cache-hit rate (%) | 32.045 [30.021, 34.068] | 65.375 [65.375, 65.375] | 33.330 [31.306, 35.354] |
| Evicted state entries | 323.200 [319.333, 327.067] | 0.000 [0.000, 0.000] | -323.200 [-327.067, -319.333] |
| Evicted compressed entries | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| Evicted attention tokens | 158,434.000 [149,509.501, 167,358.499] | 0.000 [0.000, 0.000] | -158,434.000 [-167,358.499, -149,509.501] |
| Recomputed prefix tokens | 135,155.200 [127,135.007, 143,175.393] | 3,072.000 [3,072.000, 3,072.000] | -132,083.200 [-140,103.393, -124,063.007] |
| Request throughput (requests/s) | 0.616 [0.611, 0.621] | 0.656 [0.654, 0.657] | 0.040 [0.034, 0.045] |
| Output throughput (tokens/s) | 78.840 [78.185, 79.495] | 83.910 [83.775, 84.046] | 5.071 [4.368, 5.773] |
| Mean TTFT (ms) | 852.511 [830.000, 875.022] | 517.362 [514.634, 520.090] | -335.149 [-358.119, -312.180] |
| P50 TTFT (ms) | 780.595 [747.178, 814.012] | 166.398 [164.798, 167.998] | -614.197 [-647.818, -580.576] |
| P95 TTFT (ms) | 1,520.472 [1,515.971, 1,524.974] | 1,589.336 [1,585.283, 1,593.390] | 68.864 [61.050, 76.678] |
| P99 TTFT (ms) | 1,530.677 [1,527.436, 1,533.919] | 1,621.616 [1,593.380, 1,649.852] | 90.938 [59.644, 122.232] |
| Mean TPOT (ms) | 44.418 [44.144, 44.692] | 43.965 [43.885, 44.046] | -0.453 [-0.746, -0.159] |
| Peak cache-state memory (MiB) | 17,673.156 [17,673.156, 17,673.156] | 17,215.412 [17,163.185, 17,267.640] | -457.744 [-509.972, -405.516] |
| Peak staging memory (MiB) | 0.000 [0.000, 0.000] | 328.959 [276.731, 381.187] | 328.959 [276.731, 381.187] |
| Peak process CUDA allocated (MiB) | 27,142.251 [27,141.961, 27,142.541] | 27,575.983 [27,552.301, 27,599.666] | 433.733 [409.865, 457.601] |
| Peak outstanding compression jobs | 0.000 [0.000, 0.000] | 7.800 [7.245, 8.355] | 7.800 [7.245, 8.355] |
| Compression completions/s | N/A | 1.516 [1.514, 1.518] | N/A |
| Compression completion fraction (%) | N/A | 99.107 [99.107, 99.107] | N/A |
| Compression failures | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |

## 6. Exploratory screen (not confirmatory evidence)

Three repetitions each: 16/32/64 full slots at 17.26 GiB with LRU; then LRU/retain-full/reuse-aware policies at 9.20 GiB with eight full slots. Allocation order and policy order rotated. Selection maximized mean request throughput, with hit rate as a tie-breaker. Validation used new seeds (20261201–20261206), separate from allocation (20261001–20261003) and policy (20261101–20261103). The three-repetition screen does not meet the five-repetition requirement and is not pooled with validation.

| Candidate | n | Hit rate (%) [95% CI] | Requests/s [95% CI] | Mean TTFT (ms) [95% CI] |
|---|---:|---:|---:|---:|
| allocation/full16 | 3 | 66.150 [66.150, 66.150] | 1.853 [1.830, 1.875] | 185.280 [183.260, 187.300] |
| allocation/full32 | 3 | 66.150 [66.150, 66.150] | 1.849 [1.831, 1.867] | 185.824 [182.416, 189.232] |
| allocation/full64 | 3 | 66.150 [66.150, 66.150] | 1.848 [1.838, 1.857] | 185.885 [184.603, 187.167] |
| policy/lru | 3 | 1.378 [0.637, 2.119] | 1.276 [1.265, 1.286] | 425.358 [418.988, 431.728] |
| policy/retain_full | 3 | 1.378 [0.637, 2.119] | 1.273 [1.271, 1.276] | 426.743 [422.985, 430.501] |
| policy/reuse | 3 | 2.067 [0.783, 3.351] | 1.282 [1.268, 1.295] | 422.323 [415.199, 429.446] |

All allocation candidates reached the ideal hit ceiling with no evictions; their close timings do not establish a universal optimum. Policy-screen intervals overlap, and no held-out direct policy-versus-policy comparison was conducted. Do not frame this as proof of a superior eviction algorithm.

## 7. Validity, exclusions, and outstanding cleanup

- This compiler independently rechecks all 108 confirmatory runs: mode pairing, identical saved trace hashes, expected repetitions, all recorded validation flags, memory/staging limits, cold-cache state, 192 unique request indices, expected lengths, no retractions, actual baseline eviction, and hit/recomputation arithmetic. Hashes and counts are saved in `combined_audit.json`.
- The original `full_20260913` attempt is excluded: a staging peak exceeded its allowance by approximately 8.66 MiB. Revised studies use fresh runs and larger allowances inside the same ceilings. Failed launch attempts and all pilots are excluded. No failed run is silently pooled into these tables.
- These are repeated runs on one device, not independent hardware samples. Temperature, clocks, and unrelated resident GPU processes are not controlled by a multi-device randomized design; alternating order mitigates but does not eliminate drift. The saved environment includes a small unrelated GPU process. No claim of exclusive hardware reservation is made.
- Headroom is conservative and allocations are not exhaustively optimized. Do not infer a universal minimum useful memory budget from three points. B and C also differ in policy, dense slots, staging reserve, output length, and concurrency; only within-study matched comparisons isolate the stated configuration change.
- Rank 16 is lossy. These synthetic forced-output serving runs provide no answer-quality evidence. Use the paper's separate quality evaluation with its own provenance.

### Qwen GPQA, compression-on, seed 789: pending

This requested cleanup was **not run**: the matching four-seed paired evaluation and exact command were not located. Older Qwen GPQA Diamond CoT zero-shot artifacts exist under `/home/joshuaz/sssm-states/eval_results/gpqa-diamond/Qwen__Qwen3.5-4B/`, but their recorded seeds are 0/1234 and their API setups differ. They do not establish the requested seed-789 pairing. To finish comparably, supply the existing run directory or original command, including the GPQA variant, sample set, generation settings, compression implementation/rank, and which RNGs seed 789 controls. No guessed replacement result is included, and no claim that the four-seed comparison is complete is made.

## 8. Source artifacts and reproduction

- [A: original-allocation report](results/revised_20260914/full/report.md), [raw-data audit](results/revised_20260914/full/audit.json), [provenance](results/revised_20260914/full/provenance.json).
- [B: allocation/policy search and validation report](results/allocation_policy_20260915_fixed/report.md), [protocol](results/allocation_policy_20260915_fixed/protocol.json).
- [C: concurrency report](results/spark_concurrency_20260916/report.md), [protocol](results/spark_concurrency_20260916/protocol.json), [method](SPARK_CONCURRENCY.md).
- [Original memory/measurement definitions](results/README.md).
- Source per-run `result.json`, `requests.jsonl`, `command.json`, telemetry, and traces are adjacent to each report in the corresponding stage directories. Existing result directories must not be overwritten.

To regenerate this combined report and its checks from existing measurements (no GPU work):

```sh
.venv/bin/python benchmark/mamba_pressure/combine_reports.py
```

To reproduce a study, use `revised.py`, `search.py`, or `spark.py` with `--results <fresh-directory> --launch`, preserving the saved environment, source hashes, and protocol. These scripts run GPU experiments; regeneration of this document does not.
