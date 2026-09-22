# Matched-memory Mamba compression experiment

Status: preparing instrumentation and pilot runs; no experimental results yet.

Model: Qwen/Qwen3.5-4B. Compression rank: 16.

Compare compression off/on at two measured total GPU cache-memory budgets,
including a severely constrained cache. Include the compressed pool and
compression buffers in memory accounting. Use six paired repetitions per
budget, alternating off/on and on/off order, with identical request traces
within each pair and fresh servers for each run.

Report token cache-hit rate, evicted entries, recomputed prefix tokens,
request/output throughput, mean/P50/P95/P99 TTFT, TPOT, peak cache memory,
and compression queue depth or completion rate. Summarize per-run metrics
with means and 95% confidence intervals across repetitions.

Pilot checks must establish actual eviction without compression and verify
the memory comparison. The current implementation allocates an extra
compressed pool and can evict compressed entries while full-state slots
remain free; do not assume compression improves capacity or performance.
