"""Matched eager versus production deferral+cap2 tail diagnostics."""

import hashlib
import json
import statistics
import subprocess
from argparse import Namespace
from pathlib import Path

from benchmark_utils import configs, estimate, main
from profile_analysis import analyze
from spark_run import ROOT, one_run, save, trace

SEED = 20261310
DESCRIPTION = """# Matched tail-latency profiling comparison

Qwen3.5-4B rank16; original synthetic pressure workload, 64 prefixes × three
passes, 2048+16 input tokens, 128 output tokens, concurrency4. Both policies use
32 full and 298 compressed slots, the original ~14.19 GiB cache ceiling, and
identical requests within each repetition. Eager uses cap8 and no deferral;
after uses native production deferral+cap2. Both retain current race fixes.
This is not a checkout of the historical Study C binary. Graphs/overlap are off.

Five repetitions with reversed condition order on odd repetitions. Two profiled
pilots precede all full runs. Unprofiled controls quantify instrumentation cost.
Intervals are Student-t 95% CIs over repetition-level means/differences, not
over individual requests. Each profile has 192 requests: tail estimates remain
noisy. Matched cohorts use the eager run's tail indices, then the union of both
runs' tail indices; each cohort is held fixed within its before/after pair.

Host-overlap categories overlap and cannot be summed into a decomposition.
GPU-event spans include stream delays and are not exclusive kernel timings.
SVD host overlap cannot prove memory-bandwidth interference, and completion-loop
time is not lock-wait time. Client-to-forward includes queueing, tokenization,
and network time. See TAIL_PROFILE.md for additional measurement limitations.
"""


def design():
    base = next(c for c in configs() if c["full_slots"] == 32)
    rows = []
    for after in (False, True):
        for profile in (False, True):
            rows.append(
                {
                    **base,
                    "label": ("after" if after else "eager")
                    + ("_profile" if profile else "_control"),
                    "production_defaults": after,
                    "expected_svd_worker_batch": 2 if after else 8,
                    "tail_profile": profile,
                }
            )
    return rows


def matched_metrics(before, after):
    left = {r["index"]: r for r in before}
    right = {r["index"]: r for r in after}
    if (
        len(left) != len(before)
        or len(right) != len(after)
        or left.keys() != right.keys()
    ):
        raise ValueError("Profile request indices are not a unique matched set")
    for i in left:
        if left[i]["round"] != right[i]["round"]:
            raise ValueError("Matched request rounds differ")
    cohorts = {
        "all": set(left),
        "eager_tail": {i for i, r in left.items() if r["tail"]},
        "union_tail": {i for i in left if left[i]["tail"] or right[i]["tail"]},
    }
    result = {}
    keys = sorted(k for k in before[0] if k.endswith("_ms"))
    for cohort, indices in cohorts.items():
        if not indices:
            raise ValueError("Empty matched tail cohort")
        result[f"{cohort}/requests"] = len(indices)
        for key in keys:
            valid = [
                i
                for i in indices
                if left[i].get(key) is not None and right[i].get(key) is not None
            ]
            if len(valid) != len(indices):
                raise ValueError(f"Missing paired measurement: {cohort}/{key}")
            for name, values in (
                ("eager", [left[i][key] for i in valid]),
                ("after", [right[i][key] for i in valid]),
                ("delta", [right[i][key] - left[i][key] for i in valid]),
            ):
                result[f"{cohort}/{key}/{name}"] = statistics.mean(values)
        for name, records in (("eager", left), ("after", right)):
            for path in ("miss", "full", "compressed"):
                result[f"{cohort}/{name}/{path}_fraction"] = sum(
                    records[i]["cache_path"] == path for i in indices
                ) / len(indices)
    return result


def summarize(root):
    groups = {}
    for path in sorted((root / "full").glob("*/result.json")):
        row = json.loads(path.read_text())
        if any(v is False for v in row["validation"].values()):
            raise ValueError(f"Invalid completed run: {path}")
        groups.setdefault(row["config"]["label"], {})[row["repetition"]] = row[
            "metrics"
        ]
    sections = {}
    for label, runs in groups.items():
        sections[f"{label} (n={len(runs)})"] = {
            key: estimate([r[key] for r in runs.values()])
            for key in next(iter(runs.values()))
        }
    comparisons = (
        ("after_control", "eager_control"),
        ("after_profile", "eager_profile"),
        ("eager_profile", "eager_control"),
        ("after_profile", "after_control"),
    )
    for a, b in comparisons:
        reps = sorted(set(groups.get(a, {})) & set(groups.get(b, {})))
        if reps:
            keys = groups[a][reps[0]].keys() & groups[b][reps[0]].keys()
            sections[f"{a} minus {b} (n={len(reps)})"] = {
                key: estimate([groups[a][r][key] - groups[b][r][key] for r in reps])
                for key in sorted(keys)
            }
    paired = []
    for rep in range(5):
        paths = [
            root / "full" / f"{mode}_profile_r{rep}" / "tail_requests.json"
            for mode in ("eager", "after")
        ]
        if all(p.exists() for p in paths):
            paired.append(matched_metrics(*(json.loads(p.read_text()) for p in paths)))
    if paired:
        sections[f"Matched profile cohorts (n={len(paired)})"] = {
            key: estimate([r[key] for r in paired]) for key in paired[0]
        }
    save(root / "summary.json", sections)
    lines = [DESCRIPTION]
    for title, metrics in sections.items():
        lines += [f"\n## {title}\n", "| Metric | Mean | 95% CI |", "|---|---:|---|"]
        lines += [
            f"| {key} | {v['mean']:.6g} | {v['ci95']} |" for key, v in metrics.items()
        ]
    (root / "report.md").write_text("\n".join(lines) + "\n")


async def experiment(root):
    conditions = design()
    sources = list(Path(__file__).parent.glob("*.py")) + [
        ROOT / "python/sglang/srt" / p
        for p in (
            "server_args.py",
            "managers/scheduler.py",
            "mem_cache/mamba_radix_cache.py",
            "mem_cache/compression_admission.py",
        )
    ]
    save(
        root / "protocol.json",
        dict(
            configs=conditions,
            repetitions=5,
            pilots=2,
            measured_runs=20,
            seed=SEED,
            description=DESCRIPTION,
            ordering="Reverse four conditions on odd repetitions",
            checkpoint=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            hashes={
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sources
            },
        ),
    )
    completed = 0
    for phase in ("pilot", "full"):
        for rep in range(1 if phase == "pilot" else 5):
            ordered = (
                [c for c in reversed(conditions) if c["tail_profile"]]
                if phase == "pilot"
                else conditions[:: (-1 if rep % 2 else 1)]
            )
            args = Namespace(
                groups=8 if phase == "pilot" else 64,
                rounds=2 if phase == "pilot" else 3,
                prefix=2048,
                output=128,
                seed=SEED - 1 if phase == "pilot" else SEED,
                port=31037,
                concurrency=4,
                pilot=phase == "pilot",
            )
            workload = trace(args, rep)
            save(root / f"trace_{phase}_r{rep}.json", workload)
            for cfg in ordered:
                target = root / phase / f"{cfg['label']}_r{rep}"
                save(
                    root / "status.json",
                    dict(
                        state="running", run=str(target), completed=completed, total=22
                    ),
                )
                await one_run(args, cfg, rep, target, workload)
                if cfg["tail_profile"]:
                    analyze(target)
                completed += 1
                if phase == "full":
                    summarize(root)
    save(
        root / "status.json",
        dict(
            state="complete",
            completed=completed,
            total=22,
            report=str(root / "report.md"),
        ),
    )


if __name__ == "__main__":
    main(experiment, __file__)
