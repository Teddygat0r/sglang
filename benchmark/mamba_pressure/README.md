# Live Mamba cache-pressure benchmark

Run from the repository root with its existing Python environment:

```sh
.venv/bin/python benchmark/mamba_pressure/run.py \
  --results benchmark/mamba_pressure/results/full_20260913 \
  --budgets 24 192 --prefix 2048 --kv-tokens 262144 --repetitions 6
.venv/bin/python benchmark/mamba_pressure/report.py \
  benchmark/mamba_pressure/results/full_20260913
```

The standalone server installs observational hooks in its spawned workers.
It uses the repository's cache, compression, eviction, scheduler, model, and
HTTP implementations without changing their policies. Ordinary server
launches do not load these hooks.

## Design

- Qwen/Qwen3.5-4B, rank 16, default SVD worker batch/power iteration settings.
- Final budgets correspond to 24 and 192 uncompressed state slots plus a
  262,144-token attention pool. The 64-prefix first sweep creates about
  192 saved state entries, so the smaller budget is severely constrained
  and the larger budget is near the initial working-set size. Later unique
  suffixes continue creating entries. Each baseline must actually evict.
- Exact tensor bytes, including sentinel slots, are measured in a discovery
  server; nominal memory fractions do not define the experimental budgets.
  Compression pays for its extra pool and reserves three full states' worth
  of snapshot/result staging **inside** the same total budget. Its full-state
  pool is reduced accordingly. Remaining whole attention-token cells are
  allocated to attention KV. The run fails if observed peak cache-state
  bytes exceed the shared ceiling. This is an equal ceiling, not a claim
  that both modes occupy identical bytes at every instant.
- 64 distinct 2,048-token prefixes; three independently shuffled sweeps;
  16-token unique suffixes; eight forced output tokens; concurrency one.
  Prefixes align with the 2,048-token prefill chunk size so the Mamba cache
  can save a usable shared-prefix checkpoint. Attention capacity exceeds
  the distinct prefix working set. State-capacity pressure drives eviction.
- Six repetitions, off/on for even repetitions and on/off for odd ones.
  Each pair uses identical pre-generated token IDs and server seeds.
  Fresh process per run, two disjoint warmup requests, compression drain,
  then cache flush before the measured workload. Startup and warmup are
  excluded from throughput and request latency.

## Metric definitions

Token hit rate is summed per-request `cached_tokens` divided by summed input
tokens. Recomputed prefix tokens sum `max(0, prefix_length - cached_tokens)`
on second and subsequent sweeps only; compulsory first access and suffix
computation are excluded. Evicted entries count loss of a saved Mamba state,
including internal-node tombstoning and compressed victims; later removal
of an already-tombstoned radix node is not a second state eviction. Evicted
attention tokens are reported separately.

TTFT runs from HTTP dispatch to the first SSE event reporting generated
tokens. TPOT is the interval from first to last generated token divided by
`output_tokens - 1`. Throughput includes client overhead and the same small
telemetry query at each sweep boundary. This is a closed-loop, concurrency-one
comparison, not a saturated throughput measurement.

Peak cache-state memory includes allocated attention KV, full Mamba states,
compressed states, and live owned snapshot/result staging storage. Views
retain and are charged for their whole underlying batch storage. This is
tensor storage accounting, not process reserved VRAM: model weights,
allocator reserve, and transient SVD/inference workspaces are excluded.
Peak total process CUDA **allocated** memory is separately recorded from
PyTorch after the flush, so the report does not present cache-state memory
as the full GPU footprint. Staging peaks are allocation-event driven, not
periodic samples. Pending peak counts all outstanding compression jobs;
completion and commit rates are separate, since a finished result can be
discarded after eviction. Outstanding final jobs may complete after the
last request; the end-of-workload completion fraction is reported as observed.

Per-request values and metadata, per-sweep telemetry, before/after snapshots,
server commands/logs, byte geometry, request traces, and per-run results are
saved. Means and two-sided Student-t 95% confidence intervals use repetitions
as the sampling unit, including for each run's latency percentile. Paired
effects use the within-repetition on-minus-off differences. Pilot runs are
separate and excluded from final intervals. Failed starts and an intentionally
interrupted, unneeded pilot configuration remain available for audit.

This experiment measures performance and reuse under a particular synthetic
trace. It does not measure rank-16 output quality, maximum throughput, or
prove that a different memory partition would be optimal. The current
compressed pool may evict while the full-state pool has free slots; negative
results are retained rather than changing that policy during measurement.
