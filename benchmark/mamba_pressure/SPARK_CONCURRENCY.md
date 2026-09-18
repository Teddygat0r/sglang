# Spark concurrency follow-up

Run ` .venv/bin/python benchmark/mamba_pressure/spark.py --results <fresh-directory> --launch` from the repository root. The detached supervisor saves status and fails closed on invalid runs; it does not retry or relax budgets.

Qwen3.5-4B, rank 16, ordinary compressed LRU, concurrency 2 and 4, 128 forced output tokens. Three matched cache ceilings correspond to uncompressed full-state capacities 64, 128, and 192. KV stays fixed at 262144 tokens. Compression uses 32 dense slots; the remaining budget goes to compressed slots after reserving 16 full-state equivalents (786 MiB) for snapshot/result staging. Weights, arithmetic workspaces, and allocator reserve are excluded from this cache ceiling; process peak allocated CUDA memory is reported separately.

Four pilot runs use the smallest budget, both modes, both concurrency levels, eight prefix groups and two passes. Successful pilots are followed by 60 measured runs: three budgets times two concurrency levels times two modes times five repetitions. Each measured run uses a fresh server, disjoint warmup, cache flush, 64 synthetic 2048-token prefixes, three shuffled passes, unique 16-token suffixes, and independent repetition seeds. Matching configurations receive identical token traces. Pair and configuration order reverse on alternate repetitions. Pilots are excluded from summaries.

Requests are closed-loop with a concurrency semaphore and a barrier between passes; each group occurs once per pass, so requests with the same prefix do not overlap. Latency begins when a semaphore slot is obtained (client submission), not while waiting in the client backlog. Actual server execution and completion order may vary between modes. Throughput includes telemetry and pass barriers. Recomputed-prefix counts exclude the compulsory cold pass.

Metrics: token cache hit rate, state/KV eviction counters, recomputed prefix tokens, request/output throughput, mean and P50/P95/P99 TTFT, TPOT, cache/staging/process peaks, and compression pending/completion counters. Means and Student-t 95% confidence intervals are across repetitions, not pooled requests. Percentile intervals describe variation among per-run percentiles.

Per-run checks cover request completion, cache and staging limits, expected compressed-pool bytes, cold-cache isolation, no cold-pass hits, zero retractions, no compression failures, and baseline eviction for measured runs. Raw request streams, startup/before/after/pass telemetry, commands, process IDs, and traces are retained. Source hashes are saved in protocol.json.

Limits: synthetic prefixes, no quality evaluation, graphs and overlap scheduling disabled. This isolates concurrent allocation effects on the existing execution path; it does not establish production-optimized peak capacity. Real-prompt quality and production-optimization comparisons require separate experiments.
