# Smaller SVD batch experiment

Compare the existing `--mamba-svd-worker-batch` caps 8 (default), 2 and 1. This changes the maximum immediately-available snapshot batch, not a minimum or batching timeout. Record actual batch mean/max/count so a non-binding cap is visible.

Normal server.py, eager compression, unchanged safe snapshot/job identity/result lifetime handling. No prefill gate, added prefill synchronization, pressure admission or profiling. Necessary existing SVD stream synchronization remains. No production-source modifications.

Qwen3.5-4B rank16, concurrency4, full32/compressed298, ~14.19 GiB cache ceiling including the same staging allowance. CUDA graphs and overlap scheduling disabled, as in prior benchmarks; background SVD still overlaps inference. Workload: 64 prefixes, three shuffled passes, 2048 prefix + 16 suffix tokens, 128 output. Five repetitions with matched traces/seeds and reversed cap order on odd repetitions; three short pilots excluded. Original validation checks retained; additionally verify configured and observed batch caps. Report latency, throughput, hit rate, memory, compression activity and paired mean/95% CIs. Smaller batches may reduce interference but lower compression efficiency or grow the queue. No performance claim until completion.

```
.venv/bin/python benchmark/mamba_pressure/batch_sweep.py --launch --results benchmark/mamba_pressure/results/svd_batch_TIMESTAMP
```

Detached supervisor saves status.json, supervisor.log, report.md, summary.json, source hashes and per-run artifacts. Abort on validation failure, no retries. Estimated runtime 1.5–2.5 hours. Prior synchronization experiment sources are archived under retired/ and its results are untouched.
