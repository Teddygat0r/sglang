# Tail latency diagnostic

Use `tail_compare.py` with external `--geometry` and `--results` paths, as
shown in [TAIL_COMPARE.md](TAIL_COMPARE.md). It compares eager and production
compression policies with both profiled and unprofiled controls. Historical
single-repetition screens and their policy variants live in the external archive.

Profiling runs launch `profile_server.py`; ordinary `server.py` does not install
these hooks. Application code contains no dependency on this profiling harness.

Hooks are benchmark-only, opt-in via PRESSURE_TAIL_PROFILE. CPU monotonic timestamps join client request IDs to cache lookup, full-state copy/compressed restore, forward batch, background SVD and scheduler completion-drain spans. CUDA events use the operation's stream and are harvested only after query reports completion, during existing server-info calls. No added GPU synchronization on request paths. All forwards (including decode) are recorded. Reported GPU spans include stream waiting/interference, not exclusive kernel execution. Late pending spans are counted in telemetry; failed request joins abort analysis.

Outputs: status.json, supervisor.log, report.md; per-run requests.jsonl, spans-PID.jsonl, tail_requests.json, tail_summary.json, validation result.json and normal telemetry. All outputs stay in the external result directory.

Tail = empirical slowest approximately 5%, including ties. Summaries separate tail/non-tail and miss/full/compressed paths. CPU overlap is unioned, but categories overlap and must NOT be added as a latency decomposition. Forward spans belong to batches, not exclusively to one request. GPU spans crossing first-token time are not clipped; treat them as associated operation timings. Client-to-first-forward includes network/tokenization, queueing and cache work, not pure scheduler waiting. Initial and repeated cache lookups can occur before a request is admitted.

SVD host overlap is an association, NOT proof of GPU bandwidth contention; CUDA event durations do not provide absolute GPU overlap or bandwidth counters. Completion-loop time is NOT lock-wait time. Existing telemetry also takes a short lock; this setup does not identify individual contended locks. If profiles implicate SVD or reconstruction, follow up with a controlled intervention and a short Nsight Systems kernel timeline. Compare profiled versus control latency to quantify instrumentation perturbation before interpreting tail differences. This report diagnoses the current race-fixed implementation, not the exact historical Study C binary.
