"""Paired pre/post hot-path optimization test at the two best allocations."""

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
    from allocation_native import configs as allocation_configs
    result = []
    for base in allocation_configs():
        if base["full_slots"] not in (16, 32):
            continue
        for old in (True, False):
            result.append({**base, "pre_optimization": old,
                           "label": f"f{base['full_slots']}_{'before' if old else 'after'}"})
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
    lines = ["# Cache hot-path optimization comparison", "",
             "Qwen3.5-4B, rank 16, concurrency 4, fixed ~14.19 GiB cache ceiling. "
             "64 shared prefixes × 3 passes, 2048 prefix + 16 suffix tokens, 128 output tokens. "
             "LRU; graphs and overlap disabled. Five repetitions, reversed configuration order on odd repetitions.", "",
             "Ceiling includes KV and dense/compressed state pools plus staging; excludes model weights, "
             "arithmetic workspace and allocator reserve. Staging reservation is not runtime enforcement. "
             "Frozen pre-change methods versus three optimizations together; not a quality evaluation or individual ablation.", ""]
    for label, runs in groups.items():
        metrics = {name: estimate([r["metrics"][name] for r in runs.values()]) for name in next(iter(runs.values()))["metrics"]}
        output[label] = dict(n=len(runs), metrics=metrics)
        lines += [f"## {label} (n={len(runs)})", "", "| Metric | Mean | 95% CI |", "|---|---:|---|"]
        for name, item in metrics.items():
            lines.append(f"| {name} | {item['mean']:.6g} | {item['ci95']} |")
        lines.append("")
    paired = {}
    for label, runs in groups.items():
        if not label.endswith("_after"):
            continue
        baseline = label.replace("_after", "_before")
        reps = sorted(set(runs) & set(groups.get(baseline, {})))
        if not reps:
            continue
        paired[label] = {name: estimate([runs[r]["metrics"][name] - groups[baseline][r]["metrics"][name] for r in reps]) for name in runs[reps[0]]["metrics"]}
        lines += [f"## Paired {label} minus {baseline} (n={len(reps)})", "", "| Metric | Mean difference | 95% CI |", "|---|---:|---|"]
        for name, item in paired[label].items():
            lines.append(f"| {name} | {item['mean']:.6g} | {item['ci95']} |")
        lines.append("")
    save(root / "summary.json", dict(groups=output, paired=paired))
    (root / "report.md").write_text("\n".join(lines).rstrip() + "\n")


async def experiment(root):
    design = configs()
    sources = [Path(__file__), HERE / "spark_run.py", HERE / "server.py", HERE / "instrumentation.py", HERE / "optimization_baseline.py", HERE / "allocation_native.py"]
    sources += [ROOT / "python/sglang/srt" / p for p in (
        "server_args.py", "mem_cache/memory_pool.py", "mem_cache/mamba_allocation.py", "mem_cache/mamba_radix_cache.py",
        "model_executor/model_runner_kv_cache_mixin.py")]
    save(root / "protocol.json", dict(
        configs=design, repetitions=5, measured_runs=20, pilots=4,
        groups=64, rounds=3, prefix=2048, suffix=16, output=128, seed=20261001,
        ordering="Same trace/seed within each repetition; reverse all four configurations on odd repetitions.",
        policy="lru", native_allocation=True, staging_full_states=16,
        note="Same allocation and trace per pre/post pair. Baseline installs four frozen pre-change methods before telemetry. Three optimizations tested together. Pilots excluded; abort on failed validation.",
        hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    ))
    completed = 0
    for phase in ("pilot", "full"):
        (root / phase).mkdir()
        for rep in range(1 if phase == "pilot" else 5):
            # Reverse both pair order and allocation order on odd repetitions.
            ordered = design if rep % 2 == 0 else list(reversed(design))
            for config in ordered:
                args = Namespace(groups=8 if phase == "pilot" else 64, rounds=2 if phase == "pilot" else 3,
                                 prefix=2048, output=128, seed=20260930 if phase == "pilot" else 20261001,
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
    raise RuntimeError("This historical sweep is retired: the three optimizations were reverted after a compression failure. Use a new protocol for future comparisons.")
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
