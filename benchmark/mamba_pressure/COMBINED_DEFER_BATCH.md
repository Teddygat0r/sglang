# Prefill deferral plus smaller SVD batches

Fresh paired comparison: deferral + cap8 (baseline), deferral + cap2, deferral + cap1. All arms use the same nonblocking scheduler signal and preserve existing snapshot/cancellation/lifetime protections. No strict gate, profiling, or added blocking prefill synchronization. Normal server.py and production defaults are unchanged.

Qwen3.5-4B rank16, concurrency4, full32/compressed298, ~14.19 GiB ceiling including identical staging allowance. 64 prefixes × three passes; 2048+16 input, 128 output. Five repetitions, same trace/seed per repetition, reverse arm order on odd repetitions; three excluded pilots. Fresh seed base 20261130. Both configuration and observed maximum batch sizes are checked, along with all existing completion/failure/memory/retraction checks. Actual batch-size mean/max, worker deferral and completion metrics remain available.

This tests the incremental benefit of smaller batches on top of deferral, not a full factorial interaction study. Do not pool earlier standalone studies into these confidence intervals. Reports include per-arm mean/95% CI and paired differences versus deferral+cap8.

```
.venv/bin/python benchmark/mamba_pressure/combined_defer_sweep.py --launch --results benchmark/mamba_pressure/results/combined_defer_TIMESTAMP
```

18 runs total; expected 1.5–2.5 hours. Durable status.json, supervisor.log, report.md, summary.json and per-run artifacts. Abort on validation failure, no automatic retries or default promotion.
