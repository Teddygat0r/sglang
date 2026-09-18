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
    command = [sys.executable, str(HERE / "search_variant" / "server.py"),
        "--model-path", "Qwen/Qwen3.5-4B", "--port", str(args.port),
        "--host", "127.0.0.1", "--mem-fraction-static", "0.6",
        "--max-running-requests", str(args.concurrency), "--max-total-tokens", str(config["kv_tokens"]),
        "--max-mamba-cache-size", str(config["full_slots"]),
        "--context-length", "4096", "--chunked-prefill-size", "2048",
        "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
        "--disable-overlap-schedule", "--enable-metrics", "--random-seed", str(args.seed + rep),
    ]
    if config["compression"]:
        command += ["--mamba-svd-compression", "--mamba-svd-rank", "16"]
    env = dict(os.environ, CUDA_HOME="/usr/local/cuda", TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    env["PRESSURE_COMPRESSED_SLOTS"] = str(config.get("compressed_slots", 0))
    env["PRESSURE_POLICY"] = "lru"
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
                for round_index in range(args.rounds):
                    semaphore = asyncio.Semaphore(args.concurrency)
                    async def execute(index, item):
                        async with semaphore:
                            result = await request(session, base, item["tokens"], args.output)
                            result.update(group=item["group"], round=item["round"], index=index)
                            rows.append(result)
                            raw.write(json.dumps(result) + "\n")
                            raw.flush()
                    items = [(i, item) for i, item in enumerate(workload) if item["round"] == round_index]
                    await asyncio.gather(*(execute(i, item) for i, item in items))
                    metrics = await observe(session, base)
                    save(directory / f"round_{round_index}_telemetry.json", metrics)
                    print(f"{directory.name}: {len(rows)}/{len(workload)} requests, evictions={metrics.get('evicted_entries', 0)}", flush=True)
            duration = time.perf_counter() - start
            after = await observe(session, base)
            save(directory / "after.json", after)
            metrics = summarize_requests(rows, duration, before, after, args)
            validation = {
                "all_requests_completed": len(rows) == len(workload),
                "within_cache_budget": after["peak_cache_state_bytes"] <= config["budget_bytes"],
                "baseline_evicted": bool(metrics["evicted_entries"]) if not config["compression"] and not getattr(args, "pilot", False) else None,
                "no_retractions": all(r["meta_info"]["total_retractions"] == 0 for r in rows),
                "staging_within_reserve": after.get("staging_peak_bytes", 0) <= config["staging_reserve_bytes"],
                "expected_compressed_pool": initial["compressed_pool_bytes"] == config.get("compressed_slots", 0) * config["compressed_state_bytes"],
                "cold_cache": before["full_free_slots"] == config["full_slots"] and before["compressed_entries"] == 0 and before["compression_pending"] == 0,
                "cold_pass_misses": all(r["cached_tokens"] == 0 for r in rows if r["round"] == 0),
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
