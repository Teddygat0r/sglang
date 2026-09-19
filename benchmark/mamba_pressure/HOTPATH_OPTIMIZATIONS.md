# Cache hot-path optimization comparison

## Current status: retired after failure

The 2026-09-19 run stopped after six valid measured runs when the seventh run
reported eight failed compression items. The three optimizations below have now
been reverted; the historical launcher refuses to start to avoid a mislabeled
comparison. Existing results and frozen baseline methods are preserved.

The snapshot clone remains intact. Its temporary source is now registered with
the SVD stream using `record_stream`, protecting allocator lifetime until the
asynchronous clone completes. A GPU reproducer demonstrated premature reuse and
corruption without that registration, and preserved snapshots with it. This
establishes a real race, not proof that it caused the recorded eigensolver failures:
those input tensors were not saved. Failed jobs also report back to the scheduler
for pending-state cleanup while preserving their full states and error reporting.

`test_snapshot_lifetime.py` covers the race, snapshot isolation, failure cleanup,
stale notifications and reset. A live replay is still required before reintroducing
optimizations. The description below records the historical experiment, not the
current production behavior.

## Changes

1. Restore logging uses lazy debug messages with CPU node IDs only. It never
   formats GPU slot tensors, avoiding the associated host readback.
2. A singleton compression batch uses `unsqueeze(0)` instead of `stack`, avoiding
   an extra full-state copy. Multi-item batches, snapshot cloning, stream ordering,
   SVD mathematics, rank and precision remain unchanged.
3. `MambaPool.alloc` defaults to zero initialization, but full-state restore callers
   explicitly pass `zero_initialize=False`. Both initial and eviction-retry restore
   allocations overwrite every temporal and convolution element before publishing
   the request's state index. Other callers retain their existing behavior.

## Validation

`test_hotpath.py` checks default zeroing, allocation exhaustion, untouched sentinel
storage, complete restoration from full and compressed states, GPU-tensor formatting
avoidance, eviction retry, input snapshot preservation, and numerical equivalence
against the original singleton/multi-item SVD batching. The same tests run on CPU
and, with `PRESSURE_TEST_CUDA=1`, on the Spark GPU. These are correctness regressions,
not an answer-quality evaluation or proof of correctness in every serving mode.

`test_hotpath_sweep.py` tests the actual server module imports in fresh subprocesses
for both implementations, all 24 supervisor runs using a fake request runner,
alternating order, shared traces and paired confidence-interval reporting.

## Experiment

- Qwen3.5-4B, rank 16, concurrency 4, fixed ~14.19 GiB cache ceiling.
- Reuse the best two prior allocations: 16 and 32 full slots; compressed capacity
  and staging reserves match the native allocation sweep exactly.
- Each allocation runs with the pre-change implementation and all three changes
  together. Five repetitions per cell: 20 measured runs and four excluded pilots.
- Fresh server per run, same trace/seed within repetitions, reverse the complete
  configuration sequence on odd repetitions. New seeds, starting at 20261001.
- Same 64-prefix, three-pass, 2048+16 input-token, 128-output-token workload; LRU,
  graphs and overlap scheduling disabled. All earlier memory and runtime checks
  remain active; the supervisor stops at the first failure without retrying.
- The unchanged baseline consists of four frozen pre-edit methods in
  `optimization_baseline.py`. The benchmark server installs them before telemetry
  hooks only when `PRESSURE_PRE_OPTIMIZATION=1`. This is not a production toggle.
- Source hashes, per-run commands and implementation labels are retained. The
  report includes all metrics, per-cell Student-t 95% CIs and paired after-minus-before
  differences at each allocation. This combined test does not identify individual
  optimization contributions and does not include a fresh compression-off baseline.

## Run

```bash
PYTHONPATH=python:benchmark/mamba_pressure PRESSURE_TEST_CUDA=1 .venv/bin/python -m unittest discover -s benchmark/mamba_pressure -p 'test_*.py'
.venv/bin/python benchmark/mamba_pressure/hotpath_sweep.py --launch --results benchmark/mamba_pressure/results/hotpath_20260919
```

The detached supervisor continues after the conversation ends. Expected runtime:
roughly 2–3 hours including model startups. `status.json` tracks progress or failure;
`report.md` and `summary.json` update after each measured run. Raw traces and request
logs remain local under the existing artifact policy. Nothing is automatically
committed or pushed.
