"""Shared allocation, reporting and launch helpers for maintained benchmarks."""

import argparse
import asyncio
import json
import math
import os
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

from scipy.stats import t
from spark_run import ROOT, save


def load_geometry():
    path = os.environ.get("SGLANG_MAMBA_BENCHMARK_GEOMETRY")
    if not path:
        raise ValueError(
            "Supply --geometry with a discovery server's cache_observations JSON"
        )
    geometry = json.loads(Path(path).read_text())
    if "internal_states" in geometry:
        geometry = geometry["internal_states"][0]["cache_observations"]
    for name in ("full_state_bytes", "kv_pool_bytes", "temporal_element_bytes"):
        if type(geometry.get(name)) is not int or geometry[name] <= 0:
            raise ValueError(f"Geometry requires a positive integer {name}")
    shape = geometry.get("temporal_shape", [])
    if len(shape) != 5 or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError("Geometry requires five positive temporal_shape dimensions")
    return geometry


def dataset_path():
    path = os.environ.get("SGLANG_MAMBA_BENCHMARK_DATASET")
    if not path:
        raise ValueError(
            "ShareGPT benchmarks require --dataset pointing to a local JSON file"
        )
    return Path(path)


def configs():
    g = load_geometry()
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
        result.append(
            dict(
                label=f"on_f{slots}" if on else "off",
                native_allocation=True,
                baseline_slots=128,
                concurrency=4,
                compression=on,
                full_slots=slots,
                compressed_slots=count,
                compressed_state_bytes=compressed,
                kv_tokens=262144,
                kv_pool_bytes=g["kv_pool_bytes"],
                budget_bytes=ceiling,
                staging_reserve_bytes=reserve,
            )
        )
    return result


def estimate(values):
    mean = statistics.mean(values)
    half = (
        float(t.ppf(0.975, len(values) - 1))
        * statistics.stdev(values)
        / math.sqrt(len(values))
        if len(values) > 1
        else None
    )
    return dict(
        mean=mean, ci95=[mean - half, mean + half] if half is not None else None
    )


def summarize(root, baseline="batch8", variants=("batch2", "batch1"), description=None):
    groups = {}
    for path in (root / "full").glob("*/result.json"):
        row = json.loads(path.read_text())
        if any(v is False for v in row["validation"].values()):
            continue
        groups.setdefault(row["config"]["label"], {})[row["repetition"]] = row[
            "metrics"
        ]
    summary = {
        label: {
            k: estimate([m[k] for m in runs.values()])
            for k in next(iter(runs.values()))
        }
        for label, runs in groups.items()
    }
    paired = {}
    for label in variants:
        reps = sorted(set(groups.get(label, {})) & set(groups.get(baseline, {})))
        if reps:
            paired[label] = {
                k: estimate(
                    [groups[label][r][k] - groups[baseline][r][k] for r in reps]
                )
                for k in groups[label][reps[0]]
                if k in groups[baseline][reps[0]]
            }
    save(root / "summary.json", {"groups": summary, f"paired_vs_{baseline}": paired})
    lines = ["# Paired Mamba compression benchmark", ""]
    if description:
        lines = [description, ""]
    for label, metrics in summary.items():
        lines += [
            f"## {label} (n={len(groups[label])})",
            "",
            "| Metric | Mean | 95% CI |",
            "|---|---:|---|",
        ]
        lines += [
            f"| {k} | {v['mean']:.6g} | {v['ci95']} |" for k, v in metrics.items()
        ]
        lines.append("")
    for label, metrics in paired.items():
        lines += [
            f"## Paired {label} minus {baseline}",
            "",
            "| Metric | Difference | 95% CI |",
            "|---|---:|---|",
        ]
        lines += [
            f"| {k} | {v['mean']:.6g} | {v['ci95']} |" for k, v in metrics.items()
        ]
        lines.append("")
    (root / "report.md").write_text("\n".join(lines))


def main(run_experiment, launcher_file):
    parser = argparse.ArgumentParser(
        description="Run a paired Mamba benchmark with external inputs and results"
    )
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument(
        "--geometry",
        type=Path,
        default=os.environ.get("SGLANG_MAMBA_BENCHMARK_GEOMETRY"),
    )
    parser.add_argument(
        "--dataset", type=Path, default=os.environ.get("SGLANG_MAMBA_BENCHMARK_DATASET")
    )
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Detach a supervisor and save its PID and logs",
    )
    args = parser.parse_args()
    output = args.results.resolve()
    if output.is_relative_to(ROOT):
        parser.error("--results must be outside the source repository")
    if args.geometry is None or not args.geometry.is_file():
        parser.error("--geometry must name an existing discovery JSON file")
    os.environ["SGLANG_MAMBA_BENCHMARK_GEOMETRY"] = str(args.geometry.resolve())
    load_geometry()
    if args.dataset is not None:
        if not args.dataset.is_file():
            parser.error("--dataset must name an existing local JSON file")
        os.environ["SGLANG_MAMBA_BENCHMARK_DATASET"] = str(args.dataset.resolve())
    if Path(launcher_file).stem.startswith("sharegpt_") and args.dataset is None:
        parser.error("ShareGPT benchmarks require --dataset")
    if args.launch:
        output.mkdir(parents=True, exist_ok=False)
        command = [
            sys.executable,
            str(Path(launcher_file).resolve()),
            "--results",
            str(output),
            "--geometry",
            str(args.geometry.resolve()),
        ]
        if args.dataset is not None:
            command += ["--dataset", str(args.dataset.resolve())]
        with (output / "supervisor.log").open("w") as log, open(os.devnull) as stdin:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdin=stdin,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        save(
            output / "supervisor.json",
            dict(
                pid=process.pid,
                started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            ),
        )
        print(json.dumps(dict(pid=process.pid, results=str(output))))
        return
    output.mkdir(parents=True, exist_ok=True)
    try:
        asyncio.run(run_experiment(output))
    except BaseException as error:
        prior = (
            json.loads((output / "status.json").read_text())
            if (output / "status.json").exists()
            else {}
        )
        save(output / "status.json", {**prior, "state": "failed", "error": repr(error)})
        traceback.print_exc()
        raise
