# Production prefill-aware Mamba compression

Compression stays opt-in (`--mamba-svd-compression`). When enabled, worker batch cap defaults to **2**, and new SVD batches defer while requests await prefill, chunked/selected prefill exists, or its GPU completion event is unfinished. The scheduler publishes readiness; the worker never reads mutable radix/scheduler data. Prefill does not wait for SVD; already-admitted compression may overlap it. Events are queried, not synchronized. Overlap scheduling records completion on the forward stream rather than the scheduling stream. No monkey-patch is needed in normal serving.

`--disable-mamba-svd-prefill-deferral` restores eager admission; combine with `--mamba-svd-worker-batch 8` for the former policy. Historical experiment runners explicitly request this opt-out to preserve their baselines. Compression requires TP=1 and rejects larger tensor-parallel groups before model loading. Prefill deferral supports CUDA PP=1 with standard autoregressive scheduling; disaggregation, speculative decoding, diffusion and PDMux modes warn and fall back to eager admission (cap remains configurable). Overlap stream routing has a unit regression; the live comparison keeps overlap and graphs disabled as in the prior studies. Do not infer performance validation of other serving modes.

Snapshots now use a fixed staging pool, bounded by `--mamba-svd-max-pending` (default **8**) across queued, running, and uncommitted jobs. Admission happens before copying; when staging is full, new states remain in the dense cache. Snapshot copies run on the scheduler stream before full-slot reuse; the worker waits for a completion event before reading them. Reset cancels old jobs while retaining any in-flight staging reservations until the worker finishes. Prefill deferral can starve compression, but cannot grow snapshot storage without bound.

Both automatic and explicit allocation reserve the dense sentinel, compressed pool, fixed snapshots, and pending result storage before sizing attention KV memory. `--mamba-svd-staging-reserve-bytes` is a minimum reserve for explicit sizing; temporary SVD workspace still needs dynamic memory headroom. Cache restoration does not log device tensor values.

Production `/server_info` exposes effective deferral and worker wait/count diagnostics under `mamba_svd_admission`. Wait totals include cancellation, but a current unfinished wait is included only when it ends.

## Matched comparison

`production_compare.py` uses ordinary benchmark `server.py` only for measurement telemetry; policies come from the production scheduler/cache and CLI defaults. No `defer_server.py`, `profile_server.py`, cap override or deferral opt-out. Startup verifies effective cap2, deferral enabled only in the on arm, exact pools and fixed memory ceiling. Full runs also require observed deferrals and valid actual batch caps.

Qwen3.5-4B rank16, concurrency4; ~14.19 GiB total cache ceiling; KV262144 tokens. Off: full128. On: full32/compressed298 with the same prior staging allowance. 64 shared prefixes × three passes, 2048+16 input, 128 output; five fresh paired seeds, alternate order, two excluded pilots. Model weights/allocator workspace are outside the cache ceiling. Report all existing metrics, per-arm mean/95% CI and paired on-minus-off differences. A CI containing zero does not establish equivalent P95; no equivalence margin is claimed.

```
.venv/bin/python benchmark/mamba_pressure/production_compare.py --launch --results benchmark/mamba_pressure/results/production_TIMESTAMP
```

12 runs; approximately 1–1.5 hours. Detached status.json/supervisor.log/report.md/summary.json. Stop on validation failure. Results from earlier policy experiments remain unchanged.

## Regression checks

Application tests live under `test/registered/unit/mem_cache/test_mamba_compression_*.py` and `test_mamba_svd_compression.py`, with CI registration. Historical test entrypoints in this benchmark directory forward to those tests. For a local CUDA check from the repository root:

```sh
PYTHONPATH=python:test/registered/unit/mem_cache .venv/bin/python -m unittest discover -s test/registered/unit/mem_cache -p 'test_mamba*.py'
```

These implementation fixes have unit and CUDA cache coverage. The saved serving comparisons above predate bounded staging and have not been rerun for the new implementation.
