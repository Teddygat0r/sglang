"""Fixed revised protocol: pilot both on-mode budgets, then 24 fresh runs."""

import argparse
import asyncio
import contextlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from argparse import Namespace

from run import aggregate, one_run, save, trace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESERVE_FULL_STATES = 8


def configurations(geometry):
    full = geometry["full_state_bytes"]
    layers, _, heads, dim, state = geometry["temporal_shape"]
    element = geometry["temporal_element_bytes"]
    temporal = layers * heads * dim * state * element
    compressed = layers * heads * 16 * (dim + 1 + state) * element + full - temporal
    kv_tokens = 262144
    kv_cell = geometry["kv_pool_bytes"] // (kv_tokens + 1)
    assert kv_cell * (kv_tokens + 1) == geometry["kv_pool_bytes"]
    configs = []
    for baseline_slots in (24, 192):
        budget = geometry["kv_pool_bytes"] + (baseline_slots + 1) * full
        for on in (False, True):
            reserve = RESERVE_FULL_STATES * full if on else 0
            slots = baseline_slots
            if on:
                while (slots + 1) * full + max(1, slots // 2) * compressed + reserve > (baseline_slots + 1) * full:
                    slots -= 1
            assert slots >= 3
            pools = (slots + 1) * full + (max(1, slots // 2) * compressed if on else 0)
            tokens = (budget - pools - reserve) // kv_cell - 1
            assert tokens >= kv_tokens
            assert pools + (tokens + 1) * kv_cell + reserve <= budget
            configs.append(dict(baseline_slots=baseline_slots, compression=on, full_slots=slots,
                                kv_tokens=tokens, budget_bytes=budget, staging_reserve_bytes=reserve))
    return configs, compressed, kv_cell


def arguments(directory, pilot):
    return Namespace(results=directory, budgets=[24, 192], repetitions=1 if pilot else 6,
                     groups=64, rounds=3, prefix=2048, output=8, kv_tokens=262144,
                     seed=20260915 if pilot else 20260913, port=31035)


async def phase(root, pilot, geometry, configs, compressed, kv_cell):
    directory = root / ("pilot" if pilot else "full")
    directory.mkdir(exist_ok=False)
    args = arguments(directory, pilot)
    save(directory / "arguments.json", {**vars(args), "results": str(directory)})
    save(directory / "design.json", dict(geometry=geometry, configs=configs,
         full_state_bytes=geometry["full_state_bytes"], compressed_state_bytes=compressed,
         kv_bytes_per_token=kv_cell, reserve_full_states=RESERVE_FULL_STATES))
    count = 0
    for rep in range(args.repetitions):
        workload = trace(args, rep)
        save(directory / f"trace_{rep}.json", workload)
        for budget in args.budgets:
            pair = [c for c in configs if c["baseline_slots"] == budget]
            if pilot:
                pair = [c for c in pair if c["compression"]]
            elif rep % 2:
                pair.reverse()
            for config in pair:
                label = f"b{budget}_r{rep}_{'on' if config['compression'] else 'off'}"
                save(root / "status.json", dict(state="running", phase=directory.name,
                     run=label, phase_completed=count, updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
                print(f"Starting {directory.name}/{label}: {config}", flush=True)
                target = directory / label
                await one_run(args, config, rep, target, workload)
                initial = json.loads((target / "initial.json").read_text())
                assert initial["full_state_bytes"] == geometry["full_state_bytes"]
                assert initial["kv_pool_bytes"] == (config["kv_tokens"] + 1) * kv_cell
                assert initial["compressed_pool_bytes"] == (config["full_slots"] // 2 * compressed if config["compression"] else 0)
                count += 1
                aggregate(directory)
    return directory


async def experiment(root):
    geometry_source = HERE / "results/full_20260913/discovery/initial.json"
    geometry = json.loads(geometry_source.read_text())
    configs, compressed, kv_cell = configurations(geometry)
    save(root / "protocol.json", dict(
        reserve_full_states=RESERVE_FULL_STATES, staging_reserve_bytes=RESERVE_FULL_STATES * geometry["full_state_bytes"],
        prior_staging_peak_mib=156.046875, prior_reserve_mib=147.375,
        rationale="Fixed 393 MiB staging allowance (>2.5 times the observed old peak), charged inside unchanged cache ceilings; no automatic retry or budget relaxation.",
        pilot="Both compression-on budgets, seed 20260915 (the previously failing repetition), 192 requests each.",
        full="Fresh 24 runs: two budgets, six alternating off/on pairs per budget. No prior or pilot samples pooled.",
        geometry_source=str(geometry_source), geometry_sha256=hashlib.sha256(geometry_source.read_bytes()).hexdigest(),
        script_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob("*.py")},
        configs=configs))
    await phase(root, True, geometry, configs, compressed, kv_cell)
    full = await phase(root, False, geometry, configs, compressed, kv_cell)
    for script in ("audit.py", "report.py"):
        subprocess.run([sys.executable, str(HERE / script), str(full)], cwd=ROOT, check=True)
    save(root / "status.json", dict(state="complete", phase="audited_report", runs=24,
         report=str(full / "report.md"), completed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
    print(f"COMPLETE: {full / 'report.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--launch", action="store_true")
    options = parser.parse_args()
    root = options.results.resolve()
    if options.launch:
        root.mkdir(parents=True, exist_ok=False)
        with (root / "supervisor.log").open("w") as log, open(os.devnull) as stdin:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--results", str(root)],
                cwd=ROOT, stdin=stdin, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        save(root / "supervisor.json", dict(pid=process.pid, started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
        print(json.dumps(dict(pid=process.pid, results=str(root))))
        return
    try:
        asyncio.run(experiment(root))
    except BaseException as error:
        previous = json.loads((root / "status.json").read_text()) if (root / "status.json").exists() else {}
        save(root / "status.json", {**previous, "state": "failed", "error": repr(error),
             "failed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
