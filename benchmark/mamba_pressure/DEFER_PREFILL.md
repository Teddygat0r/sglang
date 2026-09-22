# Nonblocking prefill-aware compression

Compare eager baseline with deferring admission of new SVD batches while scheduler waiting_queue, a chunked prefill, selected prefill, or unfinished prefill CUDA event is present. Scheduler publishes a thread-safe signal; worker does not inspect scheduler/radix metadata. GPU completion is queried, never synchronized. An SVD admitted just before the signal changes may run concurrently with prefill, deliberately: prefill never waits for the SVD worker. Resume on scheduler progress (including idle iterations).

Only the dedicated defer_server.py installs hooks. Normal server.py and production code are unchanged. Snapshot creation, restores and completion commits remain unchanged; this is not a ban on all cache traffic. Preserve cloning, job tokens and cancellation; recheck cancellation after waiting. Stop interrupts the worker wait. A persistent prefill backlog may starve compression, cause evictions or grow staging occupancy: the experiment retains the existing failure, memory and retraction checks and stops on failures. The staging allowance is not a hard runtime queue bound.

Original batch cap **8 in BOTH arms**, no cap1/cap2 optimization, no strict exclusion or synchronization-only control. Rank16 Qwen3.5-4B, full32/compressed298 at ~14.19 GiB, concurrency4, CUDA graphs and overlap scheduling disabled. Background compression remains asynchronous. 64 prefixes × three passes, 2048+16 input, 128 output. Five matched-seed repetitions, reverse order on odd repetitions, two excluded pilots. Report all existing metrics and actual SVD batch sizes plus deferred batch count and cumulative worker wait. In-progress worker waits are not counted until admitted; queue metrics provide additional context. Mean/95% CIs and paired differences retained.

Launch: `.venv/bin/python benchmark/mamba_pressure/defer_sweep.py --launch --results benchmark/mamba_pressure/results/defer_TIMESTAMP`

12 runs, approximately 1–1.5 hours (longer if compression starvation hurts throughput). Durable status.json, supervisor.log, report.md and summary.json. No default is promoted automatically.
