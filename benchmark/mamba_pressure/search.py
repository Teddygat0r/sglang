"""Sequential allocation/policy screening followed by held-out paired validation."""
import argparse
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import select
import statistics
import subprocess
import sys
import time
import traceback
from argparse import Namespace
from scipy.stats import t
import run

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
run.HERE = HERE / "search_variant"
G = json.loads((HERE / "results/full_20260913/discovery/initial.json").read_text())
FULL = G["full_state_bytes"]
L, _, H, D, S = G["temporal_shape"]
COMPRESSED = L * H * 16 * (D + 1 + S) * G["temporal_element_bytes"] + FULL - L * H * D * S * G["temporal_element_bytes"]
PRIOR = HERE / "results/revised_20260914"


def config(budget, slots=None, policy="lru"):
    ceiling = G["kv_pool_bytes"] + (budget + 1) * FULL
    on = slots is not None
    slots = slots if on else budget
    reserve = 8 * FULL if on else 0
    compressed = (ceiling - G["kv_pool_bytes"] - (slots + 1) * FULL - reserve) // COMPRESSED if on else 0
    assert not on or compressed > 0
    return dict(baseline_slots=budget, compression=on, full_slots=slots,
                compressed_slots=compressed, policy=policy, kv_tokens=262144,
                budget_bytes=ceiling, staging_reserve_bytes=reserve)


def write_summary(root):
    groups = {}
    for path in sorted(root.glob("*/*/result.json")):
        result = json.loads(path.read_text())
        key = path.parent.parent.name + "/" + path.parent.name.split("_r")[0]
        groups.setdefault(key, []).append(result)
    summary = {}
    lines = ["# Allocation and eviction search", "", "Screening and held-out validation are separate. Student-t 95% CIs are across repetitions.", "",
             "Cache budget includes KV, full/compressed state pools and measured staging; excludes weights, arithmetic workspace and allocator reserve. Process CUDA allocated peak is reported separately.", ""]
    for key, rows in groups.items():
        metrics = {}
        lines += [f"## {key} (n={len(rows)})", "", "| Metric | Mean | 95% CI |", "|---|---:|---|"]
        for name in rows[0]["metrics"]:
            values = [r["metrics"][name] for r in rows]
            mean = statistics.mean(values)
            half = float(t.ppf(.975, len(values)-1)) * statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None
            interval = [mean-half, mean+half] if half is not None else None
            metrics[name] = dict(mean=mean, ci95=interval)
            lines.append(f"| {name} | {mean:.6g} | {interval} |")
        lines.append("")
        summary[key] = dict(n=len(rows), config=rows[0]["config"], metrics=metrics)
    run.save(root / "summary.json", summary)
    (root / "report.md").write_text("\n".join(lines))


async def phase(root, name, configs, repetitions, seed):
    directory = root / name
    directory.mkdir()
    args = Namespace(groups=64, rounds=3, prefix=2048, output=8, seed=seed, port=31036)
    run.save(directory / "protocol.json", dict(arguments=vars(args), configs=configs, repetitions=repetitions))
    results = {key: [] for key in configs}
    for rep in range(repetitions):
        workload = run.trace(args, rep)
        run.save(directory / f"trace_{rep}.json", workload)
        keys = list(configs)
        if name == "validation":
            if rep % 2:
                keys.reverse()
        else:
            keys = keys[rep % len(keys):] + keys[:rep % len(keys)]
        for key in keys:
            cfg = configs[key]
            label = f"{key}_r{rep}"
            run.save(root / "status.json", dict(state="running", phase=name, run=label, updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
            os.environ["PRESSURE_COMPRESSED_SLOTS"] = str(cfg["compressed_slots"])
            os.environ["PRESSURE_POLICY"] = cfg["policy"]
            print(f"Starting {name}/{label}: {cfg}", flush=True)
            target = directory / label
            result = await run.one_run(args, cfg, rep, target, workload)
            initial = json.loads((target / "initial.json").read_text())
            before = json.loads((target / "before.json").read_text())
            after = json.loads((target / "after.json").read_text())
            rows = [json.loads(line) for line in (target / "requests.jsonl").read_text().splitlines()]
            assert initial["compressed_pool_bytes"] == cfg["compressed_slots"] * COMPRESSED
            assert initial["kv_pool_bytes"] == G["kv_pool_bytes"]
            assert initial["full_state_slots"] == cfg["full_slots"]
            assert before["compressed_entries"] == 0
            assert before["full_free_slots"] == cfg["full_slots"]
            assert before["compression_pending"] == 0
            assert after.get("staging_peak_bytes", 0) <= cfg["staging_reserve_bytes"]
            assert after["mamba_pool_bytes"] == (cfg["full_slots"] + 1) * FULL
            assert len(rows) == 192
            assert all(0 <= row["cached_tokens"] <= args.prefix for row in rows)
            assert all(row["cached_tokens"] == 0 for row in rows if row["round"] == 0)
            assert all(row["meta_info"]["total_retractions"] == 0 for row in rows)
            run.save(target / "audit.json", dict(passed=True, requests=len(rows)))
            results[key].append(result)
            write_summary(root)
    # Predeclared selection: maximum mean request throughput, then hit rate.
    winner = max(results, key=lambda key: (statistics.mean(r["metrics"]["request_throughput_rps"] for r in results[key]),
                                          statistics.mean(r["metrics"]["token_cache_hit_rate"] for r in results[key])))
    run.save(directory / "selection.json", dict(winner=winner, criterion="mean request throughput; hit rate tie-break"))
    return configs[winner]


async def experiment(root):
    allocations = {f"full{slots}": config(192, slots) for slots in (16, 32, 64)}
    run.save(root / "protocol.json", dict(
        allocation_screen=allocations, allocation_repetitions=3, policy_repetitions=3,
        policies=["lru", "retain_full", "reuse"], policy_screen_budget=24, validation_repetitions=6,
        validation_budgets=[24, 192], severe_full_slots=8,
        seeds=dict(allocation=20261001, policy=20261101, validation=20261201),
        expected_runs=42, selection="maximum mean request throughput, hit rate tie-break",
        notes="Fixed KV tokens and 393 MiB staging allowance for on. No automatic retries or budget relaxation. Policy variants are benchmark-only. Performance study, not output-quality validation.",
        hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), HERE / "run.py", HERE / "instrumentation.py", HERE / "search_variant/server.py", ROOT / "python/sglang/srt/mem_cache/mamba_radix_cache.py"]}))
    prior = json.loads((PRIOR / "supervisor.json").read_text())
    run.save(root / "status.json", dict(state="queued", predecessor_pid=prior["pid"], predecessor=str(PRIOR)))
    if json.loads((PRIOR / "status.json").read_text())["state"] != "complete":
        if not hasattr(os, "pidfd_open"):
            raise RuntimeError("Previous run is unfinished and this Python lacks pidfd_open; restart after it completes")
        try:
            descriptor = os.pidfd_open(prior["pid"])
        except ProcessLookupError:
            descriptor = None
        if descriptor is not None:
            try:
                select.select([descriptor], [], [])
            finally:
                os.close(descriptor)
    assert json.loads((PRIOR / "status.json").read_text())["state"] == "complete", "Previous benchmark did not complete; refusing overlap or silently ignoring its failure"
    allocation = await phase(root, "allocation", allocations, 3, 20261001)
    # Screen policies under severe pressure: the enlarged high-budget pool may fit the trace.
    policies = {policy: config(24, 8, policy) for policy in ("lru", "retain_full", "reuse")}
    policy_winner = await phase(root, "policy", policies, 3, 20261101)
    selected = config(192, allocation["full_slots"], policy_winner["policy"])
    validation = {"b24_off": config(24), "b24_on": config(24, 8, selected["policy"]),
                  "b192_off": config(192), "b192_on": selected}
    await phase(root, "validation", validation, 6, 20261201)
    audits = list(root.glob("*/*/audit.json"))
    assert len(audits) == 42 and all(json.loads(p.read_text())["passed"] for p in audits)
    run.save(root / "status.json", dict(state="complete", audited_runs=42, selected=selected, report=str(root / "report.md")))


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
        run.save(root / "supervisor.json", dict(pid=process.pid, started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
        print(json.dumps(dict(pid=process.pid, results=str(root))))
        return
    try:
        asyncio.run(experiment(root))
    except BaseException as error:
        run.save(root / "status.json", dict(state="failed", error=repr(error)))
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
