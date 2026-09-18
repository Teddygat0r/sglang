# Qwen3.5-4B: live rank-16 compression under cache pressure

24 measured runs; 6 paired repetitions per budget. Values are means [95% Student-t confidence intervals across repetitions].

See [method and metric definitions](../../README.md). Raw logs, request metadata, traces, byte accounting, and per-run validation accompany this report.

## Shared cache-state ceiling: 9420.156 MiB

Full state slots: off 24, on 14. On-mode staging reserve: 393.000 MiB, charged inside the same ceiling.

| Metric | Compression off | Rank 16 on | Paired change (on − off) |
|---|---:|---:|---:|
| Token hit rate (%) | 0.5168 [0.1738, 0.8598] | 0 [0, 0] | -0.5168 [-0.8598, -0.1738] |
| Evicted state entries | 553 [552.3, 553.7] | 568 [568, 568] | 15 [14.34, 15.66] |
| Recomputed prefix tokens | 2.601e+05 [2.587e+05, 2.615e+05] | 2.621e+05 [2.621e+05, 2.621e+05] | 2048 [688.7, 3407] |
| Requests/s | 1.289 [1.283, 1.294] | 1.268 [1.266, 1.271] | -0.02049 [-0.02755, -0.01343] |
| Output tokens/s | 10.31 [10.26, 10.36] | 10.15 [10.13, 10.17] | -0.1639 [-0.2204, -0.1074] |
| Mean TTFT (ms) | 423.2 [420.3, 426.1] | 430.6 [429.4, 431.9] | 7.43 [3.746, 11.11] |
| P50 TTFT (ms) | 424.9 [422.6, 427.3] | 430.2 [428.9, 431.5] | 5.252 [1.88, 8.625] |
| P95 TTFT (ms) | 432.5 [429.7, 435.3] | 437.9 [435.7, 440.1] | 5.385 [1.822, 8.949] |
| P99 TTFT (ms) | 437.2 [435.2, 439.3] | 449.3 [445.6, 452.9] | 12.04 [9.525, 14.55] |
| TPOT (ms) | 50.36 [50.27, 50.46] | 51.1 [51.01, 51.19] | 0.7352 [0.5851, 0.8854] |
| Peak cache state (MiB) | 9420 [9420, 9420] | 9143 [9123, 9164] | -277 [-297.5, -256.4] |
| Peak process CUDA allocated (MiB) | 1.889e+04 [1.889e+04, 1.889e+04] | 1.892e+04 [1.885e+04, 1.9e+04] | 36.94 [-35.46, 109.3] |
| Peak outstanding compression jobs | 0 [0, 0] | 2 [2, 2] | 2 [2, 2] |
| Compression completions/s | 0 [0, 0] | 3.798 [3.791, 3.805] | 3.798 [3.791, 3.805] |
| Completion fraction (%) | 0 [0, 0] | 99.83 [99.83, 99.83] | 99.83 [99.83, 99.83] |

Completion fraction for compression off is encoded as zero in raw metrics; it is not applicable to that mode. Hit-rate paired changes are percentage points.

## Shared cache-state ceiling: 17673.156 MiB

Full state slots: off 192, on 162. On-mode staging reserve: 393.000 MiB, charged inside the same ceiling.

| Metric | Compression off | Rank 16 on | Paired change (on − off) |
|---|---:|---:|---:|
| Token hit rate (%) | 36.95 [35.14, 38.76] | 6.804 [5.502, 8.107] | -30.15 [-32.41, -27.88] |
| Evicted state entries | 314.5 [311, 318] | 480.8 [478.3, 483.4] | 166.3 [161.9, 170.7] |
| Recomputed prefix tokens | 1.157e+05 [1.086e+05, 1.229e+05] | 2.352e+05 [2.3e+05, 2.403e+05] | 1.195e+05 [1.105e+05, 1.284e+05] |
| Requests/s | 1.565 [1.55, 1.58] | 1.31 [1.302, 1.318] | -0.2552 [-0.2724, -0.2379] |
| Output tokens/s | 12.52 [12.4, 12.64] | 10.48 [10.42, 10.55] | -2.041 [-2.179, -1.903] |
| Mean TTFT (ms) | 288 [281.2, 294.8] | 405.4 [400.6, 410.3] | 117.4 [108.8, 126.1] |
| P50 TTFT (ms) | 419.2 [418.4, 420.1] | 429.7 [428.8, 430.6] | 10.48 [9.599, 11.35] |
| P95 TTFT (ms) | 429.1 [428.1, 430.1] | 436.7 [435.7, 437.7] | 7.598 [6.142, 9.054] |
| P99 TTFT (ms) | 433.3 [431.8, 434.8] | 448.6 [442.6, 454.6] | 15.3 [8.547, 22.06] |
| TPOT (ms) | 50.11 [50, 50.21] | 51.1 [51.04, 51.15] | 0.9916 [0.8478, 1.135] |
| Peak cache state (MiB) | 1.767e+04 [1.767e+04, 1.767e+04] | 1.74e+04 [1.738e+04, 1.742e+04] | -273 [-292.5, -253.4] |
| Peak process CUDA allocated (MiB) | 2.714e+04 [2.714e+04, 2.714e+04] | 2.714e+04 [2.712e+04, 2.716e+04] | 0.5594 [-20.06, 21.18] |
| Peak outstanding compression jobs | 0 [0, 0] | 2.333 [1.791, 2.875] | 2.333 [1.791, 2.875] |
| Compression completions/s | 0 [0, 0] | 3.833 [3.826, 3.841] | 3.833 [3.826, 3.841] |
| Completion fraction (%) | 0 [0, 0] | 99.82 [99.82, 99.82] | 99.82 [99.82, 99.82] |

Completion fraction for compression off is encoded as zero in raw metrics; it is not applicable to that mode. Hit-rate paired changes are percentage points.

## Interpretation limits

The memory ceiling covers attention KV, full and compressed Mamba states, and owned snapshot/result staging. Arithmetic workspaces, model weights and allocator reserve are outside that cache-state definition. Peak process CUDA allocated memory is reported separately; it is not a measurement of process reserved VRAM.

This is a closed-loop concurrency-one synthetic prefix-reuse trace, with eight output tokens per request. It measures latency and achieved throughput at that concurrency, not peak serving capacity. Rank-16 generation quality is not evaluated. Latency-percentile intervals describe variation across run percentiles. The stock compressed-pool eviction policy is preserved.
