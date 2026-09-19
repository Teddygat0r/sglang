# Compression eviction and stream-lifetime fixes

Pre-fix checkpoint: `55843d60c`. No old experiments or raw artifacts are deleted.

## Ownership and invalidation

Every snapshot receives a unique `CompressionJob` object carrying the existing node
ID and a cancellation event. Identity is the state-version token; it is never reused.
Only the scheduler reads/writes the radix tree, pending-node map and active-job map.
The worker sees snapshots and job tokens, checks cancellation, and returns results
or failures tagged with the same token. It no longer traverses the mutable tree.

Full-state eviction, internal-node tombstoning and cache reset cancel the current
job. A replacement state on the same node can immediately enqueue a new job. A late
result or failure is accepted only if its token is still the current pending job.
Cancelled work already running may finish, but cannot replace or clear newer state.
Queued cancelled work is skipped. Cancellation never waits for GPU work or pins the
original cache slot. Snapshot/result buffers remain alive as long as GPU work needs
them; that temporary storage is separate from reusable full-state pool slots.

The original snapshot clone is retained. Its gathered source is registered with the
SVD stream. Completed packed tensors are also registered with the inference stream
before copying/casting them into the compressed pool. Both boundaries are tested
with delayed GPU consumers and allocator reuse pressure. All three proposed speed
optimizations remain reverted; numerical SVD settings and LRU policy are unchanged.

## Regression tests

`test_compression_jobs.py` covers old results after eviction and same-node/same-slot
replacement, late failures after reset, cancellation during an active worker job,
worker isolation from scheduler metadata, and asynchronous result-copy lifetime.
`test_snapshot_lifetime.py` covers snapshot-source lifetime and failure cleanup.
Existing CPU/GPU restoration and numerical-equivalence tests remain enabled.
These targeted tests do not establish that every possible serving configuration is
race-free. Deferred allocator storage is not fully represented by the benchmark's
snapshot/result queue telemetry; process CUDA peak is reported separately.

## Fresh live validation

```bash
PYTHONPATH=python:benchmark/mamba_pressure PRESSURE_TEST_CUDA=1 .venv/bin/python -m unittest discover -s benchmark/mamba_pressure -p 'test_*.py'
.venv/bin/python benchmark/mamba_pressure/race_fixed.py --launch --results benchmark/mamba_pressure/results/race_fixed_20260919
```

Ten fresh measured runs: full-state allocations 16 and 32, five repetitions each,
plus two excluded pilots. Same ~14.19 GiB cache budget, concurrency 4, rank 16,
64 prefixes, three passes, 2048+16 input tokens and 128 output tokens. Repetition
order is 1,0,2,3,4 using seed base 20261001: the first measured run replays the failed
16-slot seed 20261002. Allocation order alternates. GPU scheduling may differ, so
matching trace/seed does not guarantee the same SVD batches or timing.

Fresh results are not pooled with the incomplete pre-fix comparison. There is no
unsafe-code baseline, no re-enabled speed optimization, and no new quality study.
The supervisor stops on any existing validation failure, including a compression
error. `status.json`, `protocol.json`, `summary.json` and `report.md` provide progress,
source hashes/commit, all prior metrics and per-configuration 95% confidence intervals.
Expected runtime is about 60–90 minutes on Spark. Raw traces/request logs stay local
under the existing artifact policy. The supervisor runs independently after handoff.
