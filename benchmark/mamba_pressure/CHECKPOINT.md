# Benchmark checkpoint — 2026-09-18

This checkpoint preserves the benchmark harnesses, tests, methods, paper report,
and compact JSON results/telemetry from the original-allocation, allocation-search,
and concurrency studies. Failed/pilot records remain clearly separated from
confirmatory measurements. Production pool allocation has not been changed.

## Raw evidence

All original results, including ignored request JSONL, server logs, input traces,
and process records, are preserved locally in:

`benchmark/mamba_pressure/artifacts/results_checkpoint_20260918.tar.gz`

The adjacent `.sha256` file checks archive integrity. This archive is ignored by
Git and is **not backed up by the GitHub push**. Copy it to durable external
storage for an independent backup. No raw result files were deleted.

The repository omits large repeated traces, logs, JSONL, and ephemeral PID files.
The compact results support inspection of the saved tables, but full regeneration
of `COMBINED_PAPER_REPORT.md` performs raw-data audits and requires restoring the
archive's `results/` directory under `benchmark/mamba_pressure/` in a fresh clone.
Extract only into a fresh location to avoid overwriting newer results.

## Verification

Before committing, run the benchmark unit tests and `combine_reports.py` using
the complete local evidence. The combined audit covers 108 confirmatory runs and
54 paired comparisons. These checks do not rerun GPU experiments.

The requested GPQA compression-on seed-789 cleanup is pending identification of
the matching original evaluation protocol; it is not represented as completed.
