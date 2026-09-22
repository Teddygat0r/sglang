# Revised matched-memory experiment — 2026-09-14

The current experiment is in `revised_20260914`. Earlier directories are
preliminary evidence, including the budget violation, and are not pooled
with this experiment. Its `status.json` reports running, failed, or complete;
`supervisor.log` preserves progress and any exception. The supervisor runs
the raw-data audit and report automatically after all measurements finish.

## Fixed protocol

Qwen/Qwen3.5-4B, rank-16 compression, the existing cache/compression/eviction
implementation, concurrency one, and unchanged SVD settings. Benchmark-only
hooks observe events and tensor lifetimes; they do not change cache policy.

| Configuration | Shared cache ceiling (MiB) | Full-state slots | Compressed slots | Staging allowance (MiB) |
|---|---:|---:|---:|---:|
| Severe, off | 9420.15625 | 24 | 0 | 0 |
| Severe, on | 9420.15625 | 14 | 7 | 393 |
| Larger, off | 17673.15625 | 192 | 0 | 0 |
| Larger, on | 17673.15625 | 162 | 81 | 393 |

The reserve is eight full-state footprints, charged **inside** the original
cache ceilings by reducing compression-on persistent storage. Remaining
whole attention-token cells are assigned to KV storage. This is equal
available cache memory, not equal instantaneous occupancy. The original
147.375 MiB staging allowance was exceeded by a 156.046875 MiB peak. The
new fixed 393 MiB allowance is greater than 2.5 times that observed peak.
It is conservative headroom, not a proof that an unbounded queue can never
exceed it. Every run still checks actual observed peak cache-state bytes;
a violation stops the experiment without retries or budget relaxation.

The pilot runs both compression-on configurations with seed 20260915,
the seed of the previously failing repetition, for 192 requests each.
Only after both pass does the supervisor start 24 fresh measured runs:
two budgets, two modes, six repetitions. Even repetitions run off/on;
odd repetitions run on/off. Each run starts a new server, performs two
disjoint warmups, waits for pending compression to drain, and flushes the
cache. Startup/warmup are excluded. All final runs use the same protocol;
no original-run or pilot samples enter final statistics.

Each repetition has 64 distinct synthetic 2048-token prefixes and three
independently shuffled sweeps, with a unique 16-token suffix per request
and eight forced output tokens. Paired runs replay identical saved token IDs.
The prefix boundary aligns with a 2048-token prefill chunk so it has a
reusable Mamba checkpoint. The KV pool exceeds the distinct prefix working
set. The severe state pool is far smaller than the approximately 192 saved
entries created by the first sweep. Every baseline must demonstrably evict.

## Measurements and interpretation

- Token hit rate: total returned `cached_tokens` / total input tokens.
- Evicted entries: saved Mamba states lost, including internal tombstones
  and compressed victims. Later deletion of an already-tombstoned radix
  node is not counted again. Evicted attention tokens are separate.
- Recomputed prefix tokens: on sweeps 2 and 3, sum of
  `max(0, 2048 - cached_tokens)`. Compulsory first access and suffix work
  are excluded. Raw metadata are audited for absence of request retractions.
- TTFT: dispatch to first SSE event reporting generated tokens. TPOT:
  first-to-last output-token interval / 7. Throughput includes client
  overhead and an identical telemetry query at each sweep boundary.
- Peak cache-state memory: attention KV, full Mamba states, compressed
  states, and owned live snapshot/result storage. Retained views charge
  the whole owning batch. Staging peaks are allocation-event driven.
  Transient inference/SVD workspaces, model weights, allocator reserve,
  and management metadata are outside this state-storage definition.
  Peak total process CUDA allocated memory is separately reported; neither
  measure is process reserved VRAM or a hard cap on the whole GPU process.
- Compression: outstanding-job peak, completed and committed counts,
  completion fraction and completions/s. Jobs outstanding at the last
  request are reported as observed, not assumed finished.

All required per-run metrics receive means and two-sided Student-t 95%
confidence intervals across six repetitions. Paired effects use each
repetition's on-minus-off difference. Percentile intervals describe
variation in per-run percentiles, not pooled request percentiles.

The experiment measures achieved throughput at concurrency one, not maximum
serving capacity. It does not test rank-16 generation quality or establish
an optimal partition between state and attention memory. The stock policy
that can evict compressed states while full-state slots are free is retained.

## Reproduction

From the repository root, choose a new, nonexistent result directory:

```sh
.venv/bin/python benchmark/mamba_pressure/revised.py \
  --results benchmark/mamba_pressure/results/revised_new_run --launch
```

The driver checks measured pool geometry against the preserved discovery
record and saves its source hash, script hashes, protocol, and configurations.
The fresh experiment has its own pilot, full results, logs, audit, and report.
