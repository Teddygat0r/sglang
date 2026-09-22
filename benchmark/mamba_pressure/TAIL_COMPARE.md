# Before/after tail profiling

Run `.venv/bin/python benchmark/mamba_pressure/tail_compare.py --launch --geometry /data/mamba/geometry.json --results /data/mamba/tail_compare-run`.

This implements a focused comparison on the original synthetic pressure workload,
not a new ShareGPT experiment or a cache-budget sweep. Both policies use the same
~14.19 GiB allocation: 32 full states, 298 compressed states, 262144 KV tokens,
and staging reserved for 16 full states. Qwen3.5-4B rank16, concurrency4,
64 prefixes × three passes, 2048+16 input tokens and 128 output tokens.

Before: eager compression, cap8, explicitly disabled prefill deferral.
After: production defaults (deferral+cap2). Current memory-lifetime protections
remain enabled in both. Each mode runs profiled and unprofiled; five repetitions,
identical requests within each repetition, reverse all four conditions on odd
repetitions. Two short profiled pilots must pass first. Abort on any failure;
no retries or automatic selection of configurations. Approximately 2–3 hours.

Profiling remains isolated in profile_server.py; ordinary serving installs no
profiling hooks. The runner allows the explicit profiling entrypoint with native
production policy defaults. It does not override policy to enable profiling.

Outputs include report.md, summary.json, protocol.json, status.json and
supervisor.log; per-run requests, spans, tail_requests.json and tail_summary.json.
Reports include unprofiled policy differences and profiling perturbation within
each policy, as well as request-index-matched profile cohorts. The eager-tail
cohort and union-of-tails cohort are fixed within each paired comparison. Report
repetition-level means and paired Student-t 95% CIs, not request-level CIs.

Read TAIL_PROFILE.md for measurement limits. CPU overlap does not prove GPU
bandwidth contention. Completion-drain timings are not lock-wait measurements.
GPU event spans include stream delays; batch forwards are not exclusive per-request
work. Categories overlap and cannot be added. This diagnoses associations and the
combined policy intervention, not the individual causal effect of cap or deferral.

All generated data and reports belong outside Git; see [README.md](README.md) for inputs, launch instructions and archive policy.
