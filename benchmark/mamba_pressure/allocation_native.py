"""Fixed-budget native allocation sweep on Spark; no eviction-policy changes."""

import argparse
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
from argparse import Namespace

from scipy.stats import t
from spark_run import one_run, save, trace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def configs():
    g = json.loads((HERE / "results/full_20260913/discovery/initial.json").read_text())
    full = g["full_state_bytes"]
    layers, _, heads, dim, state = g["temporal_shape"]
    compressed = full - layers * heads * dim * state * g["temporal_element_bytes"]
    compressed += layers * heads * 16 * (dim + 1 + state) * g["temporal_element_bytes"]
    ceiling = g["kv_pool_bytes"] + 129 * full
    result = []
    for slots in (128, 16, 32, 64):
        on = slots != 128
        reserve = 16 * full if on else 0
        count = ((128 - slots) * full - reserve) // compressed if on else 0
        result.append(dict(
            label=f"on_f{slots}" if on else "off", native_allocation=True,
            baseline_slots=128, concurrency=4, compression=on, full_slots=slots,
            compressed_slots=count, compressed_state_bytes=compressed,
            kv_tokens=262144, kv_pool_bytes=g["kv_pool_bytes"],
            budget_bytes=ceiling, staging_reserve_bytes=reserve,
        ))
    return result


def estimate(values):
    mean = statistics.mean(values)
    half = float(t.ppf(.975, len(values) - 1)) * statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None
    return dict(mean=mean, ci95=[mean - half, mean + half] if half is not None else None)


def summarize(root):
    groups = {}
    for path in sorted((root / "full").glob("*/result.json")):
        row = json.loads(path.read_text())
        groups.setdefault(row["config"]["label"], {})[row["repetition"]] = row
    output = {}
    lines = ["# Native cache-allocation sweep", "",
             "Qwen3.5-4B, rank 16, concurrency 4, fixed ~14.19 GiB cache ceiling. "
             "64 shared prefixes × 3 passes, 2048 prefix + 16 suffix tokens, 128 output tokens. "
             "LRU; graphs and overlap disabled. Five repetitions, reversed configuration order on odd repetitions.", "",
             "Ceiling includes KV and dense/compressed state pools plus staging; excludes model weights, "
             "arithmetic workspace and allocator reserve. Staging reservation is not runtime enforcement. "
             "Exploratory allocation screen, not an independently validated winner or quality evaluation.", ""]
    for label, runs in groups.items():
        metrics = {name: estimate([r["metrics"][name] for r in runs.values()]) for name in next(iter(runs.values()))["metrics"]}
        output[label] = dict(n=len(runs), metrics=metrics)
        lines += [f"## {label} (n={len(runs)})", "", "| Metric | Mean | 95% CI |", "|---|---:|---|"]
        for name, item in metrics.items():
            lines.append(f"| {name} | {item['mean']:.6g} | {item['ci95']} |")
        lines.append("")
    paired = {}
    for label, runs in groups.items():
        if label == "off":
            continue
        reps = sorted(set(runs) & set(groups.get("off", {})))
        if not reps:
            continue
        paired[label] = {name: estimate([runs[r]["metrics"][name] - groups["off"][r]["metrics"][name] for r in reps]) for name in runs[reps[0]]["metrics"]}
        lines += [f"## Paired {label} minus off (n={len(reps)})", "", "| Metric | Mean difference | 95% CI |", "|---|---:|---|"]
        for name, item in paired[label].items():
            lines.append(f"| {name} | {item['mean']:.6g} | {item['ci95']} |")
        lines.append("")
    save(root / "summary.json", dict(groups=output, paired=paired))
    (root / "report.md").write_text("\n".join(lines).rstrip() + "\n")


async def experiment(root):
    design = configs()
    sources = [Path(__file__), HERE / "spark_run.py", HERE / "server.py", HERE / "instrumentation.py"]
    sources += [ROOT / "python/sglang/srt" / p for p in (
        "server_args.py", "mem_cache/mamba_allocation.py", "mem_cache/mamba_radix_cache.py",
        "model_executor/model_runner_kv_cache_mixin.py")]
    save(root / "protocol.json", dict(
        configs=design, repetitions=5, measured_runs=20, pilots=4,
        groups=64, rounds=3, prefix=2048, suffix=16, output=128, seed=20260927,
        ordering="Same trace/seed within each repetition; reverse all four configurations on odd repetitions.",
        policy="lru", native_allocation=True, staging_full_states=16,
        note="One budget to isolate allocation. Pilots excluded. Abort on first failed validation; no retries or winner selection.",
        hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    ))
    completed = 0
    for phase in ("pilot", "full"):
        (root / phase).mkdir()
        for rep in range(1 if phase == "pilot" else 5):
            # Exercise the smallest native dense pool first during startup verification.
            ordered = design[1:] + design[:1] if phase == "pilot" else (design if rep % 2 == 0 else list(reversed(design)))
            for config in ordered:
                args = Namespace(groups=8 if phase == "pilot" else 64, rounds=2 if phase == "pilot" else 3,
                                 prefix=2048, output=128, seed=20260926 if phase == "pilot" else 20260927,
                                 port=31037, concurrency=4, pilot=phase == "pilot")
                workload = trace(args, rep)
                label = f"{config['label']}_r{rep}"
                save(root / "status.json", dict(state="running", phase=phase, run=label,
                     completed_measured_runs=completed, expected_measured_runs=20,
                     updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
                print(f"Starting {phase}/{label}", flush=True)
                target = root / phase / label
                await one_run(args, config, rep, target, workload)
                save(target / "trace.json", workload)
                if phase == "full":
                    completed += 1
                    summarize(root)
    save(root / "status.json", dict(state="complete", measured_runs=completed, pilot_runs=4, report=str(root / "report.md")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    root = args.results.resolve()
    if args.launch:
        root.mkdir(parents=True, exist_ok=False)
        with (root / "supervisor.log").open("w") as log, open(os.devnull) as stdin:
            proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--results", str(root)],
                                    cwd=ROOT, stdin=stdin, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        save(root / "supervisor.json", dict(pid=proc.pid, started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
        print(json.dumps(dict(pid=proc.pid, results=str(root))))
        return
    try:
        asyncio.run(experiment(root))
    except BaseException as error:
        prior = json.loads((root / "status.json").read_text()) if (root / "status.json").exists() else {}
        save(root / "status.json", {**prior, "state": "failed", "error": repr(error)})
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
