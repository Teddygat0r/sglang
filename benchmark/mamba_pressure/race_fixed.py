"""Race-fixed validation at both best allocations; replay failing seed first."""

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
    from allocation_native import configs as original_configs
    return [{**c, "label": f"fixed_f{c['full_slots']}", "pre_optimization": False}
            for c in original_configs() if c["full_slots"] in (16, 32)]


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
    lines = ["# Race-fixed cache validation", "",
             "Qwen3.5-4B, rank 16, concurrency 4, fixed ~14.19 GiB cache ceiling. "
             "64 shared prefixes × 3 passes, 2048 prefix + 16 suffix tokens, 128 output tokens. "
             "LRU; graphs and overlap disabled. Five repetitions in order 1,0,2,3,4; allocation order alternates by execution order. Failed seed replayed first. Three performance optimizations remain reverted.", "",
             "Ceiling includes KV and dense/compressed state pools plus staging; excludes model weights, "
             "arithmetic workspace and allocator reserve. Staging reservation is not runtime enforcement. "
             "Fresh race-fix validation, not pooled with earlier runs. No unsafe-code baseline or quality evaluation.", ""]
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
    sources = [Path(__file__), HERE / "spark_run.py", HERE / "server.py", HERE / "instrumentation.py", HERE / "allocation_native.py"]
    sources += [ROOT / "python/sglang/srt" / p for p in (
        "server_args.py", "mem_cache/memory_pool.py", "mem_cache/mamba_allocation.py", "mem_cache/mamba_radix_cache.py",
        "model_executor/model_runner_kv_cache_mixin.py")]
    save(root / "protocol.json", dict(
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        configs=design, repetitions=5, measured_runs=10, pilots=2,
        groups=64, rounds=3, prefix=2048, suffix=16, output=128, seed=20261001,
        ordering="Repetitions 1,0,2,3,4; same trace/seed within each repetition; alternate allocation order. Previously failing seed first.",
        policy="lru", native_allocation=True, staging_full_states=16,
        note="Fresh post-race-fix validation. Performance optimizations remain reverted. Pilots excluded; abort on failed validation, no retries.",
        hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    ))
    completed = 0
    for phase in ("pilot", "full"):
        (root / phase).mkdir()
        for sequence, rep in enumerate([0] if phase == "pilot" else [1, 0, 2, 3, 4]):
            # Start at 16 slots; alternate allocations thereafter.
            ordered = design if sequence % 2 == 0 else list(reversed(design))
            for config in ordered:
                args = Namespace(groups=8 if phase == "pilot" else 64, rounds=2 if phase == "pilot" else 3,
                                 prefix=2048, output=128, seed=20260930 if phase == "pilot" else 20261001,
                                 port=31037, concurrency=4, pilot=phase == "pilot")
                workload = trace(args, rep)
                label = f"{config['label']}_r{rep}"
                save(root / "status.json", dict(state="running", phase=phase, run=label,
                     completed_measured_runs=completed, expected_measured_runs=10,
                     updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
                print(f"Starting {phase}/{label}", flush=True)
                target = root / phase / label
                await one_run(args, config, rep, target, workload)
                save(target / "trace.json", workload)
                if phase == "full":
                    completed += 1
                    summarize(root)
    save(root / "status.json", dict(state="complete", measured_runs=completed, pilot_runs=2, report=str(root / "report.md")))


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
