"""Paired live cache-pressure experiment. All requests and telemetry are retained."""

import argparse
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import statistics
import subprocess
import sys
import time

import aiohttp
import numpy as np
from scipy.stats import t

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


async def get(session, base, path):
    async with session.get(base + path) as response:
        response.raise_for_status()
        return await response.json()


async def observe(session, base):
    return (await get(session, base, "/server_info"))["internal_states"][0]["cache_observations"]


async def request(session, base, tokens, output):
    start = time.perf_counter()
    first = last = None
    meta = None
    last_count = 0
    async with session.post(base + "/generate", json={
        "input_ids": tokens,
        "sampling_params": {"temperature": 0, "max_new_tokens": output, "ignore_eos": True},
        "stream": True,
    }) as response:
        response.raise_for_status()
        async for line in response.content:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            data = json.loads(line[6:])
            if "error" in data:
                raise RuntimeError(data)
            meta = data.get("meta_info", meta)
            count = meta.get("completion_tokens", 0) if meta else 0
            if count > last_count:
                now = time.perf_counter()
                first = now if first is None else first
                last = now
                last_count = count
    if first is None or meta is None or last_count != output:
        raise RuntimeError(f"Invalid output: {meta}")
    if "cached_tokens" not in meta:
        raise RuntimeError(f"Missing request cache telemetry: {meta}")
    return {
        "ttft_ms": (first - start) * 1000,
        "tpot_ms": (last - first) * 1000 / (output - 1),
        "latency_ms": (time.perf_counter() - start) * 1000,
        "input_tokens": len(tokens), "output_tokens": last_count,
        "cached_tokens": meta["cached_tokens"], "meta_info": meta,
    }


def trace(args, repetition):
    rng = random.Random(args.seed + repetition)
    prefixes = [[1000 + group] + [rng.randrange(2000, 20000) for _ in range(args.prefix - 1)]
                for group in range(args.groups)]
    rows = []
    for round_index in range(args.rounds):
        order = list(range(args.groups))
        rng.shuffle(order)
        for group in order:
            # Unique first suffix token prevents accidental reuse beyond the shared prefix.
            suffix = [30000 + round_index] + [rng.randrange(40000, 50000) for _ in range(15)]
            rows.append({"group": group, "round": round_index, "tokens": prefixes[group] + suffix})
    return rows


def summarize_requests(rows, duration, before, after, args):
    def delta(key):
        return after.get(key, 0) - before.get(key, 0)
    enqueued = delta("compression_enqueued")
    completed = delta("compression_completed")
    return {
        "requests": len(rows), "duration_s": duration,
        "token_cache_hit_rate": sum(r["cached_tokens"] for r in rows) / sum(r["input_tokens"] for r in rows),
        "evicted_entries": delta("evicted_entries"),
        "evicted_compressed_entries": delta("evicted_compressed_entries"),
        "evicted_kv_tokens": delta("evicted_kv_tokens"),
        "recomputed_prefix_tokens": sum(max(0, args.prefix - r["cached_tokens"]) for r in rows if r["round"] > 0),
        "request_throughput_rps": len(rows) / duration,
        "output_throughput_tps": sum(r["output_tokens"] for r in rows) / duration,
        "mean_ttft_ms": statistics.mean(r["ttft_ms"] for r in rows),
        **{f"p{p}_ttft_ms": float(np.percentile([r["ttft_ms"] for r in rows], p)) for p in (50, 95, 99)},
        "mean_tpot_ms": statistics.mean(r["tpot_ms"] for r in rows),
        "persistent_cache_mib": after["persistent_cache_bytes"] / 2**20,
        "peak_cache_state_mib": after["peak_cache_state_bytes"] / 2**20,
        "peak_staging_mib": after.get("staging_peak_bytes", 0) / 2**20,
        "peak_process_cuda_mib": after["cuda_peak_allocated_bytes"] / 2**20,
        "compression_pending_peak": after.get("compression_pending_peak", 0),
        "compression_enqueued": enqueued,
        "compression_completed": completed,
        "compression_completion_fraction": completed / enqueued if enqueued else 0,
        "compression_completion_per_s": completed / duration,
        "compression_committed": delta("compression_committed"),
        "compression_failed": delta("compression_failed"),
        "decompression_hits": delta("decompression_hits"),
    }


async def one_run(args, config, rep, directory, workload, discover=False):
    directory.mkdir(parents=True, exist_ok=False)
    base = f"http://127.0.0.1:{args.port}"
    command = [sys.executable, str(HERE / "server.py"),
        "--model-path", "Qwen/Qwen3.5-4B", "--port", str(args.port),
        "--host", "127.0.0.1", "--mem-fraction-static", "0.6",
        "--max-running-requests", "1", "--max-total-tokens", str(config["kv_tokens"]),
        "--max-mamba-cache-size", str(config["full_slots"]),
        "--context-length", "4096", "--chunked-prefill-size", "2048",
        "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
        "--disable-overlap-schedule", "--enable-metrics", "--random-seed", str(args.seed + rep),
    ]
    if config["compression"]:
        command += ["--mamba-svd-compression", "--mamba-svd-rank", "16"]
    env = dict(os.environ, CUDA_HOME="/usr/local/cuda", TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    env["PYTHONPATH"] = str(ROOT / "python") + os.pathsep + env.get("PYTHONPATH", "")
    save(directory / "command.json", {"command": command, "config": config, "repetition": rep})
    log = (directory / "server.log").open("w")
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    save(directory / "pid.json", {"pid": process.pid})
    timeout = aiohttp.ClientTimeout(total=600)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            deadline = time.monotonic() + 600
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited {process.returncode}; see {directory / 'server.log'}")
                try:
                    async with session.get(base + "/health", timeout=aiohttp.ClientTimeout(total=2)) as response:
                        if response.status == 200:
                            break
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("Server startup exceeded 600 seconds")
                await asyncio.sleep(2)
            initial = await observe(session, base)
            save(directory / "initial.json", initial)
            if discover:
                return initial
            # Warm the same shape and decode path; use a disjoint prefix, then flush.
            for warm in range(2):
                await request(session, base, [60000 + warm] + workload[0]["tokens"][1:], args.output)
            for _ in range(100):
                if (await observe(session, base))["compression_pending"] == 0:
                    break
                await asyncio.sleep(.1)
            else:
                raise RuntimeError("Compression did not drain after warmup")
            async with session.post(base + "/flush_cache") as response:
                response.raise_for_status()
                await response.read()
            before = await observe(session, base)
            save(directory / "before.json", before)
            rows = []
            # Telemetry is outside individual request latency but its overhead is
            # included in whole-run throughput. Same cadence in both modes.
            start = time.perf_counter()
            with (directory / "requests.jsonl").open("w") as raw:
                for index, item in enumerate(workload):
                    result = await request(session, base, item["tokens"], args.output)
                    result.update(group=item["group"], round=item["round"], index=index)
                    rows.append(result)
                    raw.write(json.dumps(result) + "\n")
                    raw.flush()
                    if (index + 1) % args.groups == 0:
                        metrics = await observe(session, base)
                        save(directory / f"round_{item['round']}_telemetry.json", metrics)
                        print(f"{directory.name}: {index+1}/{len(workload)} requests, "
                              f"evictions={metrics.get('evicted_entries', 0)}, "
                              f"compressed={metrics.get('compression_committed', 0)}", flush=True)
            duration = time.perf_counter() - start
            after = await observe(session, base)
            save(directory / "after.json", after)
            metrics = summarize_requests(rows, duration, before, after, args)
            validation = {
                "all_requests_completed": len(rows) == len(workload),
                "within_cache_budget": after["peak_cache_state_bytes"] <= config["budget_bytes"],
                "baseline_evicted": bool(metrics["evicted_entries"]) if not config["compression"] else None,
                "no_compression_failures": metrics["compression_failed"] == 0,
            }
            result = {"config": config, "repetition": rep, "metrics": metrics, "validation": validation}
            save(directory / "result.json", result)
            if not all(v for v in validation.values() if v is not None):
                raise RuntimeError(f"Run validation failed: {validation}")
            return result
    finally:
        # Only terminate the process group created by this invocation.
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        log.close()


def aggregate(directory):
    results = [json.loads(p.read_text()) for p in sorted(directory.glob("b*_r*_*/result.json"))]
    groups = {}
    for result in results:
        key = f"budget{result['config']['baseline_slots']}_{'on' if result['config']['compression'] else 'off'}"
        groups.setdefault(key, []).append(result)
    summary = {}
    for key, runs in groups.items():
        summary[key] = {"n": len(runs), "config": runs[0]["config"], "metrics": {}}
        for metric in runs[0]["metrics"]:
            values = [r["metrics"][metric] for r in runs]
            mean = statistics.mean(values)
            half = float(t.ppf(.975, len(values)-1)) * statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None
            summary[key]["metrics"][metric] = {"mean": mean, "ci95": [mean-half, mean+half] if half is not None else None}
    save(directory / "summary.json", summary)
    lines = ["# Live matched-memory cache-pressure experiment", "", "Each interval is a two-sided Student-t 95% confidence interval across independent run repetitions.", ""]
    for key, group in summary.items():
        lines += [f"## {key} (n={group['n']})", "", "| Metric | Mean | 95% CI |", "|---|---:|---|" ]
        for name, values in group["metrics"].items():
            ci = values["ci95"]
            lines.append(f"| {name} | {values['mean']:.5g} | " + (f"[{ci[0]:.5g}, {ci[1]:.5g}]" if ci else "not estimable") + " |")
        lines.append("")
    (directory / "summary.md").write_text("\n".join(lines) + "\n")


async def main(args):
    args.results.mkdir(parents=True, exist_ok=True)
    save(args.results / "arguments.json", {**vars(args), "results": str(args.results)})
    discovery_path = args.results / "discovery" / "initial.json"
    if discovery_path.exists():
        geometry = json.loads(discovery_path.read_text())
    else:
        geometry = await one_run(args, {"kv_tokens": args.kv_tokens, "full_slots": args.budgets[0], "compression": False}, 0,
                                 args.results / "discovery", [], discover=True)
    full = geometry["full_state_bytes"]
    layers, _, heads, dim, state = geometry["temporal_shape"]
    temporal = layers * heads * dim * state * geometry["temporal_element_bytes"]
    compressed = layers * heads * 16 * (dim + 1 + state) * geometry["temporal_element_bytes"] + full - temporal
    # Page size is 1; include the attention pool's one-token sentinel.
    kv_cell = geometry["kv_pool_bytes"] // (args.kv_tokens + 1)
    assert kv_cell * (args.kv_tokens + 1) == geometry["kv_pool_bytes"]
    configs = []
    for baseline_slots in args.budgets:
        budget = geometry["kv_pool_bytes"] + (baseline_slots + 1) * full
        reserve = 3 * full  # charge on-mode snapshot/result staging to the same ceiling
        for on in (False, True):
            slots = baseline_slots
            if on:
                while (slots + 1) * full + max(1, slots // 2) * compressed + reserve > (baseline_slots + 1) * full:
                    slots -= 1
                if slots < 3:
                    raise ValueError("Budget cannot support the full-state working slots")
            state_pool = (slots + 1) * full + (max(1, slots // 2) * compressed if on else 0)
            tokens = (budget - state_pool - (reserve if on else 0)) // kv_cell - 1
            configs.append({"baseline_slots": baseline_slots, "compression": on, "full_slots": slots,
                            "kv_tokens": tokens, "budget_bytes": budget, "staging_reserve_bytes": reserve if on else 0})
    save(args.results / "design.json", {"geometry": geometry, "configs": configs, "full_state_bytes": full,
                                         "compressed_state_bytes": compressed, "kv_bytes_per_token": kv_cell})
    for rep in range(args.repetitions):
        workload = trace(args, rep)
        save(args.results / f"trace_{rep}.json", workload)
        for budget in args.budgets:
            pair = [c for c in configs if c["baseline_slots"] == budget]
            if rep % 2:
                pair.reverse()
            for config in pair:
                label = f"b{budget}_r{rep}_{'on' if config['compression'] else 'off'}"
                directory = args.results / label
                if (directory / "result.json").exists():
                    existing = json.loads((directory / "result.json").read_text())
                    if not all(v for v in existing["validation"].values() if v is not None):
                        raise RuntimeError(f"Cannot resume failed run {directory}")
                    continue
                print(f"Starting {label}: {config}", flush=True)
                await one_run(args, config, rep, directory, workload)
                aggregate(args.results)
    aggregate(args.results)
    print(f"Completed: {args.results / 'summary.md'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[24, 48])
    parser.add_argument("--repetitions", type=int, default=6)
    parser.add_argument("--groups", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--prefix", type=int, default=1024)
    parser.add_argument("--output", type=int, default=8)
    parser.add_argument("--kv-tokens", type=int, default=131072)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--port", type=int, default=31035)
    options = parser.parse_args()
    asyncio.run(main(options))
