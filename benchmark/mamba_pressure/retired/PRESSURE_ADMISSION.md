# Pressure-triggered compression experiment

This benchmark-only policy compares eager compression with free-slot-triggered
admission. Production defaults, SVD mathematics, snapshot isolation, both stream
lifetime fixes, job-token cancellation, LRU eviction and scheduler commit rules are
unchanged. No micro-optimization or reuse-score policy is enabled.

## Admission rule

The full-state pool has 16 usable slots. Start admitting jobs when actual free
slots are at or below 8. Once active, continue until free slots plus currently
pending jobs reach 12. Each live pending job is credited with one expected slot;
completion, cancellation and failure remain handled by the race-fixed scheduler.
Expected reclamation is not guaranteed: cancelled, failed or uncommittable jobs
do not produce a free slot, and subsequent scheduler checks can admit replacements.

Candidates are scanned from the existing full-state LRU, oldest first, skipping
missing states, compressed states, already-pending nodes and detached nodes.
Locked cached states remain eligible as in eager compression: they are independent
forks of request working states. Admission does not lock a slot or alter eviction.
It is checked on insertion and after each scheduler completion drain, so deferred
states are reconsidered rather than forgotten. Reset clears hysteresis and counters.

## Matched comparison

- Qwen3.5-4B, rank 16, concurrency 4.
- Both modes: 16 full slots, 358 compressed slots, 262144 KV tokens, ~14.19 GiB
  declared cache ceiling including 786 MiB staging reserve.
- Eager baseline versus pressure low/high watermarks 8/12; one conservative setting,
  not an optimization search or claim of optimal thresholds.
- Five repetitions per mode, alternating order, identical traces/seeds per pair:
  10 measured runs and two excluded pilots. Fresh servers, warmup/drain/flush.
- Same 64 shared-prefix groups, three passes, 2048+16 input tokens, 128 forced output
  tokens. LRU, disabled CUDA graphs and overlap scheduling. Seed base 20261020.

All existing validation checks remain active. The supervisor stops on the first
failure, without retries. Source hashes are saved; results are not pooled with
earlier studies. The server reports admission policy and watermarks for startup
verification. Added metrics count pressure episodes and jobs admitted; existing
enqueued/completed compression counts show whether actual work decreases.

Reports include cache hits, evictions/recomputation, throughput, mean/percentile
TTFT, TPOT, memory, compression completion, mean/95% CI and paired differences.
A reduction in compression work is not automatically a serving-speed improvement.
This is synthetic performance testing, not a quality evaluation. Queue telemetry
does not fully account for deferred allocator storage; process CUDA peak is separate.

## Tests and launch

Unit tests cover watermarks, hysteresis, pending credits, cancellation, failed
admission, eligible-node selection, actual server installation/reset, trace pairing
and complete supervisor reporting. The existing CPU/GPU race tests also run.

```bash
PYTHONPATH=python:benchmark/mamba_pressure PRESSURE_TEST_CUDA=1 .venv/bin/python -m unittest discover -s benchmark/mamba_pressure -p 'test_*.py'
.venv/bin/python benchmark/mamba_pressure/pressure_sweep.py --launch --results benchmark/mamba_pressure/results/pressure_admission_20260919
```

Expected runtime: 60–90 minutes including model startups. The detached supervisor
runs independently; `status.json` tracks completion/failure and `report.md` contains
results. Raw trace/request logs remain local under the existing artifact policy.
# Retired experiment

Pressure admission is no longer available in the active server or launchers. Source snapshots in this directory are historical, non-importable `.py.txt` files; commands below describe the completed experiment, not current usage. Results remain under `../results/pressure_admission_20260919` (relative to this directory).
