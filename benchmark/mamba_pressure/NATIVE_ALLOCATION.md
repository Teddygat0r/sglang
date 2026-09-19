# Native allocation experiment

This exploratory experiment isolates the full/compressed pool split using production
CLI controls, not the earlier source-replacement override. Compression, LRU eviction,
worker batching and scheduler behavior are unchanged.

## Design

- Qwen/Qwen3.5-4B, rank 16, one Spark GPU, concurrency 4.
- Fixed cache ceiling 15,234,924,544 bytes (~14.19 GiB).
- Fixed 262,144-token KV pool (8,589,967,360 bytes including sentinel).
- Compression off: 128 full slots. Compression on: 16, 32 or 64 full slots,
  with the remaining budget assigned to compressed slots (rounded down).
- Each compression-on configuration reserves 824,180,736 bytes for staging.
- Five repetitions per configuration: 20 measured runs plus four excluded pilots.
- Each repetition uses an identical saved trace and seed across configurations;
  configuration order is reversed on odd repetitions. Fresh server for every run.
- 64 distinct 2,048-token synthetic prefixes, three shuffled passes, 16-token unique
  suffix, 128 forced output tokens; barriers between passes. Pilots use eight groups
  and two passes. Two disjoint warmups, drain, and flush before measurement.
- Graphs and overlap scheduling disabled, as in the preceding concurrency study.

## Implementation and limits

`--mamba-svd-cache-size` decouples compressed capacity from the historical
`full_slots // 2` rule. Together with `--max-mamba-cache-size`, it opts into static
accounting of full storage (including its sentinel), compressed storage, and
`--mamba-svd-staging-reserve-bytes` before profiling available KV capacity.
The existing explicit KV token limit then caps the KV allocation. Unspecified
compressed sizing retains legacy behavior; this does not fix legacy accounting.

The initial implementation requires DP=1, PP=1, radix caching enabled and no
speculative decoding. The staging reserve is a sizing allowance, **not a queue
memory cap**. Runtime admission control and eviction-policy changes are out of scope.
The benchmark aborts if observed staging exceeds the reserve or total accounted
cache memory exceeds the ceiling. It also verifies actual KV and compressed pools
against the protocol before measurement. Weights, arithmetic workspace and allocator
reserve are excluded from the cache ceiling; process CUDA peak is reported separately.

## Launch and outputs

```bash
PYTHONPATH=python:benchmark/mamba_pressure .venv/bin/python -m unittest discover -s benchmark/mamba_pressure -p 'test_*.py'
.venv/bin/python benchmark/mamba_pressure/allocation_native.py --launch --results benchmark/mamba_pressure/results/native_allocation_20260918
```

The launcher creates a detached supervisor and a new output directory; it refuses
to overwrite existing results. Expected duration is roughly 2–3 hours on the Spark,
based on the prior concurrency-4 runs, including model startups and pilots.

`status.json` records progress or failure. `protocol.json` stores geometry, ordering,
seeds and source hashes. Per-run results include all earlier throughput, latency,
cache-hit, eviction, recomputation, memory and compression-completion metrics.
`report.md` and `summary.json` update after each measured run, with means and
Student-t 95% confidence intervals and paired differences against compression off.
There are no automatic retries. This is a one-budget allocation screen, not a
replacement for the earlier three-budget paper study or an independent validation
of whichever allocation performs best. Raw requests/traces remain locally ignored
by Git under the existing artifact policy.
