# Individual optimization ablation on the race-fixed baseline

The completed race-fixed baseline results are committed in `63a5fb5b1`.
This study does not change the production defaults. `individual_variant.py`
installs one narrowly scoped benchmark-only method substitution before telemetry
in each server worker. Exact source-match checks reject unexpected implementation
changes. The server reports its installed variant; startup validation checks it.

## Four independent variants

1. `baseline`: unchanged race-fixed implementation.
2. `logging`: lazy debug messages with CPU node IDs instead of formatting GPU
   destination tensors. No copy or allocation changes.
3. `singleton`: only replace singleton `torch.stack` with an unsqueezed view.
   Multi-item batching and the intentional independent snapshot clone remain intact.
4. `restore_nozero`: only skip allocation-time zeroing for the two full-state
   restoration call sites (initial allocation and eviction retry). All other
   allocations still zero; restored temporal and convolution states are fully written.

Every variant retains unique compression-job tokens, eviction/reset cancellation,
scheduler-owned metadata, snapshot-source stream registration and result-buffer
stream registration. There is no cumulative combination in this experiment.

## Protocol

Qwen3.5-4B, rank 16, concurrency 4, 16 full slots, 358 compressed slots,
262144 KV tokens, ~14.19 GiB cache ceiling including 786 MiB declared staging
reserve. Same synthetic 64 shared-prefix groups, three shuffled passes, 2048-token
prefix and 16-token suffix, 128 forced output tokens, LRU, disabled graphs/overlap.
Fresh server each run, two warmups then drain/flush before measuring.

Five repetitions per variant (20 measured runs) and four excluded short pilots.
Each repetition uses the same saved trace and server seed across all variants;
variant order reverses on odd repetitions. Seed base 20261010. The new baseline is
rerun alongside variants; earlier results are not substituted or pooled.
All existing completion, eviction/retraction, cold-cache, pool-size, staging,
budget and compression-failure checks remain active. Abort on first failure,
without automatic retries. Source hashes and Git commit are recorded.

Reports contain all prior cache, throughput, latency and compression metrics,
means with Student-t 95% confidence intervals, and paired variant-minus-baseline
differences. This is an exploratory three-way comparison, not an automatically
selected winner or a new quality evaluation. Queue storage telemetry does not
fully account for deferred allocator storage; CUDA process peak is also reported.

## Verification and launch

`check_individual_variant.py` uses the actual server entrypoint in a fresh process
for each variant. It checks race-critical methods remain unchanged, default zeroing,
full/compressed restoration, eviction retry, logging without tensor formatting,
singleton/multi-item SVD equivalence and input preservation on CPU and GPU.
`test_individual_sweep.py` also checks complete supervisor ordering and paired reports.

```bash
PYTHONPATH=python:benchmark/mamba_pressure PRESSURE_TEST_CUDA=1 .venv/bin/python -m unittest discover -s benchmark/mamba_pressure -p 'test_*.py'
.venv/bin/python benchmark/mamba_pressure/individual_sweep.py --launch --results benchmark/mamba_pressure/results/individual_20260919
```

Expected runtime: 2–3 hours including model startups. The detached supervisor
continues independently. `status.json` and `supervisor.log` track progress;
`report.md` and `summary.json` update after successful measured runs. Raw traces
and request logs remain local under the existing artifact policy.
