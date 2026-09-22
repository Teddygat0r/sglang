# Multi-turn ShareGPT shared-prefix evaluation

Corrects the first-turn-only workload limitation. Use a local V4.3 dataset via --dataset, preserving the first three user/assistant pairs of 64 distinct sessions per repetition. Five paired repetitions compare compression off versus production deferral+cap2 under the same 64 GiB ceiling and allocation as SHAREGPT_ROOMY.md. 192 requests/run; concurrency4; alternate mode order; two 8-session, three-turn pilots first. Each mode gets identical token sequences and output lengths. Graphs/overlap scheduling remain disabled. No production-policy changes.

Serialize raw text as incrementally tokenized `USER:\n…\nASSISTANT:\n` segments plus recorded assistant text. This intentionally preserves exact prior input IDs without chat-template thinking blocks or BPE boundary changes. It is not Qwen's native chat template and does not evaluate quality. Later prompts include the dataset's recorded assistant replies, not this run's generated replies. Thus prior input prefixes can hit exactly, while generated output states are generally not reusable. Do not describe this as teacher-forced decoding or on-policy live dialogue.

Filter sessions (never truncate): first three pairs must be human/gpt; each reference output length 2–1024; each cumulative input+output<=4096; distinct first prompts. At each turn, shuffle session order, issue at concurrency4, and wait for that round to finish before advancing. This interleaves sessions and preserves causality, but does not reproduce Marconi's inter-session/inter-request arrival distributions. It is a multi-turn reference-history cache evaluation, not an exact reproduction of that paper. The prior first-turn report remains valid as a separate low-reuse overhead test.

Report measured token hit rate, followup request-hit fraction, prior-input prefix opportunity (not a promised hit-rate ceiling including all sources of reuse), and recomputed prior-input-prefix tokens. Preserve throughput/TTFT/TPOT/memory/eviction/compression metrics and paired mean/95% CIs. Both pilots and every full run must show >=10% token hits and >=50% followups with nonzero hits; otherwise stop. Also require zero evictions/retractions/compression failures and valid memory accounting. A conservative unique-token estimate must fit below 240000 of the 262144 KV tokens; at least 88 GiB unified memory must be available at preparation.

```
.venv/bin/python benchmark/mamba_pressure/sharegpt_multiturn.py --launch --dataset /data/mamba/ShareGPT_V4.3_unfiltered_cleaned_split.json --geometry /data/mamba/geometry.json --results /data/mamba/sharegpt_multiturn-run
```

Status/protocol/log/report/summary saved under the result directory. Both pilots must pass before any full runs; no automatic retries or silent workload changes. Expect roughly 2–4 hours, depending on sampled reference output lengths. Source hashes, dataset revision/checksum and selected indices retained; all generated files stay in the external output directory.

All generated data and reports belong outside Git; see [README.md](README.md) for inputs, launch instructions and archive policy.
