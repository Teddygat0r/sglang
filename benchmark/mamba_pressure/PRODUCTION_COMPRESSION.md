# Production prefill-aware Mamba compression

Compression stays opt-in (`--mamba-svd-compression`). When enabled, worker batch cap defaults to **2**, and new SVD batches defer while requests await prefill, chunked/selected prefill exists, or its GPU completion event is unfinished. The scheduler publishes readiness; the worker never reads mutable radix/scheduler data. Prefill does not wait for SVD; already-admitted compression may overlap it. Events are queried, not synchronized. Overlap scheduling records completion on the forward stream rather than the scheduling stream. No monkey-patch is needed in normal serving.

`--disable-mamba-svd-prefill-deferral` restores eager admission; combine with `--mamba-svd-worker-batch 8` for the former policy. Historical experiment runners explicitly request this opt-out to preserve their baselines. Prefill deferral supports CUDA PP=1 with standard autoregressive scheduling; disaggregation, speculative decoding, diffusion and PDMux modes warn and fall back to eager admission (cap remains configurable). Overlap stream routing has a unit regression; the live comparison keeps overlap and graphs disabled as in the prior studies. Do not infer performance validation of other serving modes.

Snapshots now use a fixed staging pool, bounded by `--mamba-svd-max-pending` (default **8**) across queued, running, and uncommitted jobs. Admission happens before copying; when staging is full, new states remain in the dense cache. Snapshot copies run on the scheduler stream before full-slot reuse; the worker waits for a completion event before reading them. Reset cancels old jobs while retaining any in-flight staging reservations until the worker finishes. Prefill deferral can starve compression, but cannot grow snapshot storage without bound.

Both automatic and explicit allocation reserve the dense sentinel, compressed pool, fixed snapshots, and pending result storage before sizing attention KV memory. `--mamba-svd-staging-reserve-bytes` is a minimum reserve for explicit sizing; temporary SVD workspace still needs dynamic memory headroom. Cache restoration does not log device tensor values.

Production `/server_info` exposes effective deferral and worker wait/count diagnostics under `mamba_svd_admission`. Wait totals include cancellation, but a current unfinished wait is included only when it ends.

## Tensor-parallel compression

Standard autoregressive serving supports `--tp-size N` with compression when DP=1 and PP=1. Speculation, diffusion, disaggregation, PDMux and hierarchical caching are rejected for TP compression because their scheduler paths need separate collective integration. The single-rank path does not use collectives.

Each rank snapshots and compresses only its local state shard. The scheduler exchanges a bounded CPU status vector over its existing TP cache process group on every completion drain, including empty drains. Jobs commit in admission order only when every rank has finished successfully; this keeps dense-slot reclamation, compressed eviction and prefix availability identical across ranks. State tensors stay on their owning devices. The collective synchronizes scheduler ranks but does not wait for unfinished SVD work; the slowest worker determines when a result can commit.

Logical admission slots stay reserved until all ranks acknowledge completion or cancellation. A snapshot/SVD failure on any rank cancels the job everywhere and preserves its dense state if the prefix is still cached. Reset cancels old jobs without reusing their sequence numbers or in-flight staging. Pool-size and job-identity mismatches fail explicitly. Each worker has an independent random generator so asynchronous SVD batches do not consume inference RNG state.

Two-process Gloo tests use sharded CPU state and exercise unequal completion timing, different batch order, shard restoration, compressed eviction, failures, bounded admission, reset and stale notifications. CUDA regressions exercise snapshots, restoration and isolated RNG on one GPU. Multi-GPU live serving and the throughput cost of the added CPU collective have not been measured.

## Matched comparison

`production_compare.py` uses ordinary benchmark `server.py` only for measurement telemetry; policies come from the production scheduler/cache and CLI defaults. The comparison uses native compression defaults. Startup verifies effective cap2, deferral enabled only in the on arm, exact pools and fixed memory ceiling. Full runs also require observed deferrals and valid actual batch caps.

Qwen3.5-4B rank16, concurrency4; ~14.19 GiB total cache ceiling; KV262144 tokens. Off: full128. On: full32/compressed298 with the same prior staging allowance. 64 shared prefixes × three passes, 2048+16 input, 128 output; five fresh paired seeds, alternate order, two excluded pilots. Model weights/allocator workspace are outside the cache ceiling. Report all existing metrics, per-arm mean/95% CI and paired on-minus-off differences. A CI containing zero does not establish equivalent P95; no equivalence margin is claimed.

```
.venv/bin/python benchmark/mamba_pressure/production_compare.py --launch --geometry /data/mamba/geometry.json --results /data/mamba/production-run
```

12 runs; approximately 1–1.5 hours. Detached status.json/supervisor.log/report.md/summary.json. Stop on validation failure. Results from earlier policy experiments remain unchanged.

## Regression checks

Application tests live under `test/registered/unit/mem_cache/test_mamba_compression_*.py` and `test_mamba_svd_compression.py`, with CI registration. For a local CUDA check from the repository root:

```sh
PYTHONPATH=python:test/registered/unit/mem_cache .venv/bin/python -m unittest discover -s test/registered/unit/mem_cache -p 'test_mamba*.py'
```

These implementation fixes have unit and CUDA cache coverage. Historical serving results are in the external archive described in README.md. They predate bounded staging and have not been rerun for the new implementation.

All generated data and reports belong outside Git; see [README.md](README.md) for inputs, launch instructions and archive policy.
