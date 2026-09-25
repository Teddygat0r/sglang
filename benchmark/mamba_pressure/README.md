# Mamba compression benchmarks

This directory contains reusable comparison runners and their tests. Production
compression lives in `python/sglang/srt`; application regressions live in
`test/registered/unit/mem_cache`. Benchmark telemetry is installed only by the
benchmark server entrypoints.

| Runner | Purpose |
| --- | --- |
| `production_compare.py` | Paired compression off/on under a matched cache ceiling |
| `sharegpt_sweep.py` | First-turn ShareGPT workload with a roomy cache |
| `sharegpt_multiturn.py` | Interleaved conversations with reusable input prefixes |
| `sharegpt_pressure.py` | Ten-turn unsplit ShareGPT replay, concurrency 8, constrained cache budget |
| `tail_compare.py` | Eager versus default compression, with profiled and unprofiled controls |

The shared runner, allocation/reporting helpers, observation hooks and profiling
analysis support these entrypoints. Historical allocation sweeps, policy
monkey-patches, one-off scripts, generated reports and raw results are archived
outside the checkout. Registered application tests own the former benchmark
regression wrappers' coverage.

## Inputs and outputs

**Keep all experimental results and datasets outside the source repository.**
The launcher rejects a `--results` directory inside this checkout, including
paths resolved through symlinks. `.gitignore` also excludes accidental local
result/artifact directories and generated files. Do not force-add those files.

Supply `--geometry /absolute/path/to/geometry.json`. This is a cache observation
from a baseline `server.py` process with the same model, KV token count and device
configuration as the intended run. Save `/server_info` outside the repository;
both its full response and the `internal_states[0].cache_observations` object are
accepted. Required geometry fields are `full_state_bytes`, `kv_pool_bytes`,
`temporal_shape` and `temporal_element_bytes`. They describe allocated tensors,
not the GPU's advertised memory capacity. The runners validate actual pool sizes
again after starting each server. Reuse archived geometry only for a matching
configuration; a fresh checkout does not depend on an old result file.

The current protocols target locally available `Qwen/Qwen3.5-4B` weights, TP=1,
and 262144 KV tokens. ShareGPT runners additionally require `--dataset` pointing
to a local JSON file of conversations; see their protocol documents for the
expected dataset revision and filtering. Inputs can alternatively be configured
with `SGLANG_MAMBA_BENCHMARK_GEOMETRY` and `SGLANG_MAMBA_BENCHMARK_DATASET`.

```sh
.venv/bin/python benchmark/mamba_pressure/production_compare.py \
  --geometry /data/mamba/geometry.json \
  --results /data/mamba/production-run --launch

.venv/bin/python benchmark/mamba_pressure/sharegpt_multiturn.py \
  --geometry /data/mamba/geometry.json \
  --dataset /data/mamba/ShareGPT_V4.3_unfiltered_cleaned_split.json \
  --results /data/mamba/multiturn-run --launch
```

`--launch` starts a detached supervisor. Read `supervisor.json` for its PID,
`status.json` for progress, and `supervisor.log` for startup or validation errors.
A PID alone does not establish a successful server startup. Reports, protocol
metadata, commands, per-request traces and telemetry all stay in the external
output directory. Full experiments take roughly 1–4 hours depending on workload.
No benchmark starts when importing these modules or requesting `--help`.

These are closed-loop serving comparisons. Graphs and overlap scheduling are
disabled in their current protocols. They do not establish maximum throughput,
lossy-compression answer quality, or multi-GPU production readiness.

## Checks

```sh
PYTHONPATH=python:benchmark/mamba_pressure .venv/bin/python -m unittest discover \
  -s benchmark/mamba_pressure -p 'test_*.py'
PYTHONPATH=python:test/registered/unit/mem_cache .venv/bin/python -m unittest discover \
  -s test/registered/unit/mem_cache -p 'test_mamba*.py'
```

Benchmark tests use synthetic geometry and temporary outputs; archived results
and model weights are unnecessary. The launcher regression exercises a detached
stand-in experiment without starting a model server.

## Archive policy

The 2026-09-22 cleanup preserved the full prior benchmark tree, including ignored
outputs and uncommitted workload changes, in a local archive under
`~/sglang-experiments/mamba-compression-20260922T224748Z/`. Its `snapshot/` retains
repository-relative paths; `manifest.json` records SHA-256 checksums;
`working-tree.patch` and `git-status.txt` preserve the original worktree state.
The archive is local and is not distributed with a Git clone. Keep historical
reports there alongside their inputs and source snapshots. Earlier commits still
contain previously committed results; this cleanup removes them from the branch's
current tree without rewriting shared history.
