"""Validate a completed experiment and write paired-effect report and provenance."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

from scipy.stats import t


def interval(values):
    mean = statistics.mean(values)
    width = float(t.ppf(.975, len(values) - 1)) * statistics.stdev(values) / math.sqrt(len(values))
    return {"mean": mean, "lower": mean - width, "upper": mean + width}


def formatted(result):
    return f"{result['mean']:.4g} [{result['lower']:.4g}, {result['upper']:.4g}]"


def main(directory):
    arguments = json.loads((directory / "arguments.json").read_text())
    runs = [json.loads(p.read_text()) for p in sorted(directory.glob("b*_r*_*/result.json"))]
    assert len(runs) == len(arguments["budgets"]) * arguments["repetitions"] * 2
    assert arguments["repetitions"] >= 5
    assert all(all(v for v in r["validation"].values() if v is not None) for r in runs)
    grouped = {}
    for run in runs:
        grouped.setdefault((run["config"]["baseline_slots"], run["config"]["compression"]), []).append(run)
    important = [
        ("token_cache_hit_rate", "Token hit rate", 100),
        ("evicted_entries", "Evicted state entries", 1),
        ("recomputed_prefix_tokens", "Recomputed prefix tokens", 1),
        ("request_throughput_rps", "Requests/s", 1),
        ("output_throughput_tps", "Output tokens/s", 1),
        ("mean_ttft_ms", "Mean TTFT (ms)", 1),
        ("p50_ttft_ms", "P50 TTFT (ms)", 1),
        ("p95_ttft_ms", "P95 TTFT (ms)", 1),
        ("p99_ttft_ms", "P99 TTFT (ms)", 1),
        ("mean_tpot_ms", "TPOT (ms)", 1),
        ("peak_cache_state_mib", "Peak cache state (MiB)", 1),
        ("peak_process_cuda_mib", "Peak process CUDA allocated (MiB)", 1),
        ("compression_pending_peak", "Peak outstanding compression jobs", 1),
        ("compression_completion_per_s", "Compression completions/s", 1),
        ("compression_completion_fraction", "Completion fraction (%)", 100),
    ]
    lines = ["# Qwen3.5-4B: live rank-16 compression under cache pressure", "",
             f"{len(runs)} measured runs; {arguments['repetitions']} paired repetitions per budget. "
             "Values are means [95% Student-t confidence intervals across repetitions].", "",
             "See [method and metric definitions](../../README.md). Raw logs, request metadata, traces, "
             "byte accounting, and per-run validation accompany this report.", ""]
    paired = {}
    for budget in arguments["budgets"]:
        off = sorted(grouped[(budget, False)], key=lambda r: r["repetition"])
        on = sorted(grouped[(budget, True)], key=lambda r: r["repetition"])
        assert [r["repetition"] for r in off] == list(range(arguments["repetitions"]))
        assert [r["repetition"] for r in on] == list(range(arguments["repetitions"]))
        total = off[0]["config"]["budget_bytes"]
        assert all(r["config"]["budget_bytes"] == total for r in off + on)
        lines += [f"## Shared cache-state ceiling: {total / 2**20:.3f} MiB", "",
                  f"Full state slots: off {off[0]['config']['full_slots']}, on {on[0]['config']['full_slots']}. "
                  f"On-mode staging reserve: {on[0]['config']['staging_reserve_bytes'] / 2**20:.3f} MiB, "
                  "charged inside the same ceiling.", "",
                  "| Metric | Compression off | Rank 16 on | Paired change (on − off) |",
                  "|---|---:|---:|---:|"]
        effects = {}
        for key, label, scale in important:
            a = [r["metrics"][key] * scale for r in off]
            b = [r["metrics"][key] * scale for r in on]
            change = interval([y - x for x, y in zip(a, b)])
            effects[key] = change
            if key == "token_cache_hit_rate":
                label += " (%)"
            lines.append(f"| {label} | {formatted(interval(a))} | {formatted(interval(b))} | {formatted(change)} |")
        paired[str(budget)] = effects
        lines += ["", "Completion fraction for compression off is encoded as zero in raw metrics; "
                  "it is not applicable to that mode. Hit-rate paired changes are percentage points.", ""]
    lines += ["## Interpretation limits", "",
              "The memory ceiling covers attention KV, full and compressed Mamba states, and owned "
              "snapshot/result staging. Arithmetic workspaces, model weights and allocator reserve "
              "are outside that cache-state definition. Peak process CUDA allocated memory is "
              "reported separately; it is not a measurement of process reserved VRAM.", "",
              "This is a closed-loop concurrency-one synthetic prefix-reuse trace, with eight output "
              "tokens per request. It measures latency and achieved throughput at that concurrency, "
              "not peak serving capacity. Rank-16 generation quality is not evaluated. "
              "Latency-percentile intervals describe variation across run percentiles. "
              "The stock compressed-pool eviction policy is preserved.", ""]
    (directory / "report.md").write_text("\n".join(lines))
    (directory / "paired_effects.json").write_text(json.dumps(paired, indent=2) + "\n")
    root = Path(__file__).resolve().parents[2]
    def command(args):
        return subprocess.run(args, cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    provenance = {
        "git_commit": command(["git", "rev-parse", "HEAD"]),
        "git_status": command(["git", "status", "--short"]),
        "gpu": command(["nvidia-smi"]),
        "python": sys.version,
        "script_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in Path(__file__).parent.glob("*.py")},
        "trace_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in directory.glob("trace_*.json")},
    }
    (directory / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(directory / "report.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    main(parser.parse_args().results)
