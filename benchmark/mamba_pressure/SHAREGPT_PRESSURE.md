# Ten-turn ShareGPT replay with a constrained cache

Run `sharegpt_pressure.py` with `--dataset`, `--geometry`, external `--results`
and `--launch`. The detached supervisor continues after the launching session.

Sample 64 real conversations per seed, each with at least ten human/assistant
pairs. Replay the first ten pairs: 640 requests per mode, five paired seeds,
concurrency eight, alternating compression-off/on order. Two eight-conversation,
ten-turn pilots precede measurement. Sample deterministically without replacement
within a run; preserve duplicate prompts across different records. Only malformed
role/text sequences and empty tokenized assistant replies are excluded.

There is **no benchmark context cap, response-length cap, length-based filtering,
or token truncation**. The server receives no `--context-length` override and
uses the model's native context (262144 for the cached Qwen3.5-4B configuration).
Preflight checks all selected requests against this native limit and fails
explicitly if exceeded; it never drops or shortens a selected request. The
aggregate trace may exceed the KV pool; KV eviction and retraction are measured
outcomes rather than reasons to shorten the trace.

Use the tokenizer's native chat template with thinking disabled. Every later
prompt includes the complete recorded conversation history. Decode for the
recorded assistant reply's full token count, using `ignore_eos`; generated text
does not replace recorded history. Both modes therefore receive identical prompts
and target output lengths. Native templates can change generation-prefix suffixes;
measure prior-input reuse opportunity by the actual longest shared token prefix.
This is reference-history serving replay, not answer-quality evaluation.

The source contains only record IDs and role/text messages: **no arrival timestamps
or think times**. All selected sessions are initially eligible in seeded order.
Each conversation advances when its own preceding request completes, sharing a
eight-request semaphore. There are no global turn barriers and no invented wall-clock
timestamps. Actual dispatch times are logged and may differ between modes as a
consequence of completion times. This simulates real conversation content and
causality under closed-loop load, not the original production arrival process.

Use 128 full states with compression off; 32 full states plus rank-16 compressed
states with compression on. Both share 262144 KV tokens and a ceiling computed as
KV bytes + 129 full-state bytes (including sentinel). Compression reserves 16
full-state equivalents for staging/results inside the ceiling. For the matching
archived geometry this is about 14.19 GiB and 298 compressed slots. Weights and
runtime workspace are outside the cache ceiling. Graphs and overlap scheduling
remain disabled. Source hashes, dataset checksum, indices and exact token traces
are saved with the protocol.

Both modes explicitly use `--mamba-scheduler-strategy extra_buffer` and
`--mamba-track-interval 256`. Prefill tracks earlier aligned states; decode tracks
at 256-token intervals. Tracking buffers come from the fixed full-state pool,
not an additional unbudgeted allocation. With overlap disabled, the scheduler's
capacity rule is four full-state slots per admitted request, so 32 slots support
the requested eight in-flight requests. Startup verifies active extra-buffer
tracking, its interval, effective request concurrency, and the existing pool and
staging budget checks. Both ten-turn pilots must pass before full runs proceed.

Pilots must demonstrate >=10% token hits and >=50% followups hitting cache.
Measured baseline runs must evict entries, or stop without claiming pressure.
Full-run hit rates and retractions are outcomes. Allocation, staging, request
completion, compression-failure and production-policy checks remain enforced.
No automatic retries, resampling, or budget changes occur on failure.

Report TTFT including tails, TPOT (zero for single-token replies), throughput,
token hits, followup hits, recomputed prior-input tokens, KV/state evictions,
retractions, compression/restores and memory with paired differences and 95% CIs.
`status.json` and `supervisor.log` record progress/errors, `requests.jsonl` records
each completed request, and `protocol.json`, `summary.json` and `report.md` retain
the design and results. Runtime depends on uncapped recorded reply lengths;
estimate from preflight token totals and earlier measured throughput.

## Unsplit input provenance

The previous Aeala V4.3 input was already split upstream and is unsuitable for
studying unrestricted conversation lengths. Prepare the upstream HTML-cleaned,
unsplit parts instead:

```sh
.venv/bin/python benchmark/mamba_pressure/prepare_sharegpt_unsplit.py \
  --output /data/mamba/sharegpt-unsplit
```

The preparation script pins `anon8231489123/ShareGPT_Vicuna_unfiltered` revision
`192ab2185289094fc556ec8ce5ce1e8e587154ca`, concatenates the two
`HTML_cleaned_raw_dataset/sg_90k_part*_html_cleaned.json` files, and records source
URLs and SHA-256 checksums beside `sharegpt_unsplit.json`. It applies no additional
cleaning, language filtering, conversation splitting, or deduplication. The runner
validates and copies this provenance into `protocol.json`.

## Diagnosing prefix misses

For a saved pilot, run:

```sh
PYTHONPATH=python .venv/bin/python benchmark/mamba_pressure/diagnose_prefix_replay.py \
  --results /external/run
```

This writes `prefix_diagnosis.json` beside the results. It compares exact prompt
prefixes and exercises the production radix matcher on CPU with dense and
compressed node representations. It does not test GPU compression numerics.

Native chat serialization can replace the generation prompt's empty thinking
suffix when a recorded assistant answer becomes history. A common token prefix
then need not include the previous full-prompt checkpoint. In `no_buffer` mode,
partial radix matches cannot restore a recurrent state unless an earlier saved
endpoint is present. `mamba_track_interval=256` alone does not enable periodic
tracking: that path requires the `extra_buffer` strategy. The inherited 50%
followup-hit pilot gate is therefore not guaranteed by natural prefix repetition.
Do not silently remove the gate or alter the recorded prompts to force it to pass.
The current protocol explicitly selects `extra_buffer` in both modes to address
this checkpoint-placement issue, with the GPU pilots and startup checks above.
This changes checkpoint policy relative to the archived `no_buffer` runs; report
the new comparison separately rather than mixing their results.
