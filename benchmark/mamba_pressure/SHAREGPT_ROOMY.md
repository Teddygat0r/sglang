# ShareGPT without forced cache pressure

Uses a local copy of `Aeala/ShareGPT_Vicuna_unfiltered` revision `8b0048ad6ae8c22f46a78c15559dec98feef5539`, file `ShareGPT_V4.3_unfiltered_cleaned_split.json`. The local dataset card describes it as a further-cleaned reupload of the source used by SGLang's default ShareGPT sampler. This is V4.3, not the V3 file in that default downloader. Pass its path with --dataset; dataset files stay outside the checkout. The runner records the file checksum and selected record indices.

Following SGLang's standard first-turn workload style, send the first human prompt and use the first assistant response's token count as the requested output length. Raw prompts, no chat template; force output length via ignore_eos. Filter (not truncate) to input>=2, output2..1024, total<=4096, and remove exact duplicate tokenized prompts within each run. Draw 256 prompts for each of five seeds; both modes get the exact same ordered tokens/output lengths per pair. Each prompt is sent once; this is NOT synthetic shared-prefix repetition or a multiturn replay. Natural common prefixes may hit the cache; report measured hit rate, not an assumed one. Dataset-derived requests are replayed closed-loop at concurrency4; no original request arrival timestamps exist in this dataset. This is a serving-performance workload, not answer-quality evaluation. Output cap/context filtering limit generalization to longer conversations.

Compression off versus production deferral+cap2, Qwen3.5-4B rank16. Same 64 GiB total cache ceiling, much larger than the former ~14.19 GiB constraint, while leaving model/workspace/OS headroom on the 119 GiB unified-memory device. Require >=88 GiB MemAvailable before launch. Same KV262144 pool; off uses the remaining ceiling for full states; on uses full512 and compressed slots filling the remainder after the same 16-state staging allowance. Slot rounding leaves small unused remainders. The memory budget excludes weights/workspace/allocator reserve; it is large, not unlimited. Mem-fraction-static .75. CUDA graphs and overlap scheduling remain disabled, as in the earlier comparisons.

Two pilots plus ten full runs, reverse mode order on odd repetitions. Startup verifies exact pools, defaults and ceiling. Both arms must finish with zero cache-entry/KV-token evictions, no retractions, no compression failures, and within the cache/staging budget. If they do not, stop and revise the protocol rather than call the run unconstrained. Initial flush is checked, but common-prefix hits on the first pass are allowed. Synthetic recomputed-prefix-token metric is omitted because no declared repeated prefix exists.

```
.venv/bin/python benchmark/mamba_pressure/sharegpt_sweep.py --launch --dataset /data/mamba/ShareGPT_V4.3_unfiltered_cleaned_split.json --geometry /data/mamba/geometry.json --results /data/mamba/sharegpt-run
```

Status, protocol, per-run validations, per-request logs and mean/95% CI report saved in the output directory. Raw token traces and selected-record manifests stay in the external output directory. Expect roughly 2–4 hours, refined from the sampled output-token totals. No changes to production defaults.

All generated data and reports belong outside Git; see [README.md](README.md) for inputs, launch instructions and archive policy.
