# Prefill-aware SVD diagnostic

RETIRED: the experiment completed successfully, but strict exclusion worsened tails. Sources are archived as `.py.txt`; launch commands below are historical, not active. Results remain in `benchmark/mamba_pressure/results/prefill_20260920`. The normal server has no prefill synchronization hooks.

Checkpoint before implementation: `38662fffe`. No hooks in normal `server.py` or production code. Explicit `prefill_server.py` provides two experimental modes:

- `strict`: mutually exclude SVD execution and prefill forwards, including mixed batches. Prefill takes priority over the next waiting SVD batch. Already-running SVD must finish before prefill launches; that wait is measured.
- `sync_control`: the same prefill GPU-completion synchronization, but no SVD exclusion. Separates synchronization overhead from the exclusion policy.

The gate covers GPU completion, not merely Python function return. It does not pin radix nodes or remove the snapshot clone, job-token checks, or cross-stream lifetime protections. Cancelled jobs are rechecked after worker waiting. Shutdown interrupts a worker waiting at the gate. Exceptions release the gate after draining submitted stream work.

Scope: SVD kernels versus prefill **forward** execution. This is not a ban on all cache traffic: snapshot clones, completion commits and reconstruction are unchanged. It does not suppress compression for the entire arrival-to-prefill queue interval. It is a strict causal experiment, not necessarily the best scheduling policy; waiting for an existing SVD and starving compression during sustained prefills can worsen tails, memory pressure and throughput.

Protocol: Qwen3.5-4B rank 16, ~14.19 GiB fixed ceiling, concurrency 4, full32/compressed298, graphs and overlap disabled. Unchanged eager baseline, sync-control, strict; five repetitions each, reverse order on odd repetitions. Three short pilots first. 64 shared prefixes × three passes, 2048+16 input, 128 output. No profiling instrumentation in timing runs. All existing cache-budget/failure/retraction checks retained. Gate wait times and admission counts are deltas over the measured workload. Confidence intervals are across repetitions, not individual requests.

Launch after correctness checks:

```
.venv/bin/python benchmark/mamba_pressure/prefill_sweep.py --launch --results benchmark/mamba_pressure/results/prefill_TIMESTAMP
```

The durable supervisor stops on validation failure; status.json, supervisor.log, report.md and summary.json are in the output directory. Expected 1.5–2.5 hours, potentially longer if exclusion stalls serving. No live model run has yet validated this variant. CPU gate tests alone do not prove GPU exclusion; a GPU regression and the strict pilot must pass before trusting performance results.
