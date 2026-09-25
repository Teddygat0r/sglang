"""Paired live cache-pressure experiment. All requests and telemetry are retained."""

import asyncio
import json
import os
import random
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

import aiohttp
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


async def get(session, base, path):
    async with session.get(base + path) as response:
        response.raise_for_status()
        return await response.json()


async def observe(session, base):
    return (await get(session, base, "/server_info"))["internal_states"][0][
        "cache_observations"
    ]


async def request(session, base, tokens, output):
    start_ns = time.monotonic_ns()
    start = time.perf_counter()
    first = last = None
    meta = None
    last_count = 0
    async with session.post(
        base + "/generate",
        json={
            "input_ids": tokens,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": output,
                "ignore_eos": True,
            },
            "stream": True,
        },
    ) as response:
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
        "start_ns": start_ns,
        "first_ns": start_ns + int((first - start) * 1e9),
        "ttft_ms": (first - start) * 1000,
        "tpot_ms": (last - first) * 1000 / (output - 1) if output > 1 else 0.0,
        "latency_ms": (time.perf_counter() - start) * 1000,
        "input_tokens": len(tokens),
        "output_tokens": last_count,
        "cached_tokens": meta["cached_tokens"],
        "meta_info": meta,
    }


def trace(args, repetition):
    rng = random.Random(args.seed + repetition)
    prefixes = [
        [1000 + group] + [rng.randrange(2000, 20000) for _ in range(args.prefix - 1)]
        for group in range(args.groups)
    ]
    rows = []
    for round_index in range(args.rounds):
        order = list(range(args.groups))
        rng.shuffle(order)
        for group in order:
            # Unique first suffix token prevents accidental reuse beyond the shared prefix.
            suffix = [30000 + round_index] + [
                rng.randrange(40000, 50000) for _ in range(15)
            ]
            rows.append(
                {
                    "group": group,
                    "round": round_index,
                    "tokens": prefixes[group] + suffix,
                }
            )
    return rows


def summarize_requests(rows, duration, before, after, args):
    def delta(key):
        return after.get(key, 0) - before.get(key, 0)

    enqueued = delta("compression_enqueued")
    completed = delta("compression_completed")
    return {
        "requests": len(rows),
        "duration_s": duration,
        "token_cache_hit_rate": sum(r["cached_tokens"] for r in rows)
        / sum(r["input_tokens"] for r in rows),
        "evicted_entries": delta("evicted_entries"),
        "evicted_compressed_entries": delta("evicted_compressed_entries"),
        "evicted_kv_tokens": delta("evicted_kv_tokens"),
        "recomputed_prefix_tokens": sum(
            max(0, args.prefix - r["cached_tokens"]) for r in rows if r["round"] > 0
        ),
        "request_throughput_rps": len(rows) / duration,
        "output_throughput_tps": sum(r["output_tokens"] for r in rows) / duration,
        "mean_ttft_ms": statistics.mean(r["ttft_ms"] for r in rows),
        **{
            f"p{p}_ttft_ms": float(np.percentile([r["ttft_ms"] for r in rows], p))
            for p in (50, 95, 99)
        },
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
        "retractions": sum(
            r.get("meta_info", {}).get("total_retractions", 0) for r in rows
        ),
    }


async def replay_causal_sessions(workload, concurrency, execute):
    """Closed-loop replay without cross-conversation turn barriers."""
    groups = {}
    for index, item in enumerate(workload):
        group = groups.setdefault(item["group"], [])
        if item["round"] != len(group):
            raise ValueError("Conversation turns must be contiguous and causal")
        group.append((index, item))
    semaphore = asyncio.Semaphore(concurrency)

    async def conversation(items):
        for index, item in items:
            async with semaphore:
                await execute(index, item)

    await asyncio.gather(*(conversation(items) for items in groups.values()))


def validate_checkpoint_config(initial, config, concurrency):
    if config.get("mamba_scheduler_strategy") == "extra_buffer":
        if not initial.get("mamba_extra_buffer"):
            raise RuntimeError("Required extra_buffer checkpointing is inactive")
        if initial.get("mamba_track_interval") != config["mamba_track_interval"]:
            raise RuntimeError("Mamba tracking interval differs from protocol")
        if initial.get("max_running_requests") != concurrency:
            raise RuntimeError("Effective request concurrency differs from protocol")


async def one_run(args, config, rep, directory, workload, discover=False):
    production_defaults = config.get("production_defaults", False)
    expected_cap = config.get(
        "expected_svd_worker_batch", config.get("svd_worker_batch")
    )
    if production_defaults and (
        config.get("defer_prefill") or "svd_worker_batch" in config
    ):
        raise ValueError(
            "Production comparison must use native defaults without experimental overrides"
        )
    if any(
        config.get(name)
        for name in ("defer_prefill", "prefill_gate", "pre_optimization", "ablation")
    ):
        raise ValueError(
            "Historical policy overrides have been retired from this runner"
        )
    if not config.get("native_allocation", False):
        raise ValueError("Only native cache allocation is supported")
    if "svd_worker_batch" in config and (
        type(config["svd_worker_batch"]) is not int
        or config["svd_worker_batch"] < 1
        or not config["compression"]
    ):
        raise ValueError(
            "SVD worker batch must be a positive integer with compression enabled"
        )
    if config.get("admission", "eager") != "eager":
        raise ValueError("Pressure admission has been retired; use eager compression")
    if config.get("tail_profile") and not config.get("native_allocation"):
        raise ValueError(
            "Tail profiling requires the dedicated native profiling server"
        )
    directory.mkdir(parents=True, exist_ok=False)
    base = f"http://127.0.0.1:{args.port}"
    native = config.get("native_allocation", False)
    server = HERE / "server.py"
    if config.get("tail_profile"):
        server = HERE / "profile_server.py"
    command = [
        sys.executable,
        str(server),
        "--model-path",
        "Qwen/Qwen3.5-4B",
        "--port",
        str(args.port),
        "--host",
        "127.0.0.1",
        "--mem-fraction-static",
        str(config.get("mem_fraction_static", 0.6)),
        "--max-running-requests",
        str(args.concurrency),
        "--max-total-tokens",
        str(config["kv_tokens"]),
        "--max-mamba-cache-size",
        str(config["full_slots"]),
        "--chunked-prefill-size",
        "2048",
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--disable-overlap-schedule",
        "--enable-metrics",
        "--random-seed",
        str(args.seed + rep),
    ]
    if config.get("context_length", 4096) is not None:
        command += ["--context-length", str(config.get("context_length", 4096))]
    if "mamba_scheduler_strategy" in config:
        command += ["--mamba-scheduler-strategy", config["mamba_scheduler_strategy"]]
        command += ["--mamba-track-interval", str(config["mamba_track_interval"])]
    if config["compression"]:
        command += ["--mamba-svd-compression", "--mamba-svd-rank", "16"]
        if not production_defaults:
            # Historical experiment launchers explicitly retain their old policy.
            command += ["--disable-mamba-svd-prefill-deferral"]
            if "svd_worker_batch" not in config:
                command += ["--mamba-svd-worker-batch", "8"]
    if "svd_worker_batch" in config:
        command += ["--mamba-svd-worker-batch", str(config["svd_worker_batch"])]
    if native and config["compression"]:
        command += [
            "--mamba-svd-cache-size",
            str(config["compressed_slots"]),
            "--mamba-svd-staging-reserve-bytes",
            str(config["staging_reserve_bytes"]),
        ]
    env = dict(
        os.environ,
        CUDA_HOME="/usr/local/cuda",
        TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    for key in (
        "PRESSURE_ADMISSION",
        "PRESSURE_LOW_WATERMARK",
        "PRESSURE_HIGH_WATERMARK",
        "PRESSURE_PRE_OPTIMIZATION",
        "PRESSURE_INDIVIDUAL_VARIANT",
        "PRESSURE_COMPRESSED_SLOTS",
        "PRESSURE_POLICY",
    ):
        env.pop(key, None)
    env.pop("PRESSURE_TAIL_PROFILE", None)
    env.pop("PRESSURE_PREFILL_GATE", None)
    if config.get("tail_profile", False):
        env["PRESSURE_TAIL_PROFILE"] = str(directory.resolve())
    env["PYTHONPATH"] = str(ROOT / "python") + os.pathsep + env.get("PYTHONPATH", "")
    save(
        directory / "command.json",
        {"command": command, "config": config, "repetition": rep},
    )
    log = (directory / "server.log").open("w")
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    save(directory / "pid.json", {"pid": process.pid})
    # Long recorded replies and untruncated contexts can legitimately take longer.
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=600)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            deadline = time.monotonic() + 600
            while True:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"Server exited {process.returncode}; see {directory / 'server.log'}"
                    )
                try:
                    async with session.get(
                        base + "/health", timeout=aiohttp.ClientTimeout(total=2)
                    ) as response:
                        if response.status == 200:
                            break
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("Server startup exceeded 600 seconds")
                await asyncio.sleep(2)
            initial = await observe(session, base)
            save(directory / "initial.json", initial)
            validate_checkpoint_config(initial, config, args.concurrency)
            expected_deferral = bool(production_defaults and config["compression"])
            if bool(initial.get("defer_prefill")) != expected_deferral:
                raise RuntimeError("Prefill deferral mode differs from protocol")
            if (
                expected_cap is not None
                and initial.get("svd_worker_batch") != expected_cap
            ):
                raise RuntimeError("SVD worker batch does not match protocol")
            if config.get("tail_profile") and not initial.get("tail_profile"):
                raise RuntimeError("Tail profiling hooks not installed")
            if native:
                if initial["kv_pool_bytes"] != config["kv_pool_bytes"]:
                    raise RuntimeError(
                        "Native allocation changed the fixed KV pool size"
                    )
                if (
                    initial["compressed_pool_bytes"]
                    != config["compressed_slots"] * config["compressed_state_bytes"]
                ):
                    raise RuntimeError(
                        "Native compressed pool size differs from protocol"
                    )
                if (
                    initial["persistent_cache_bytes"] + config["staging_reserve_bytes"]
                    > config["budget_bytes"]
                ):
                    raise RuntimeError(
                        "Native allocation plus staging exceeds cache ceiling"
                    )
                print(
                    f"Startup verified: {directory.name}, native pools match protocol",
                    flush=True,
                )
            if discover:
                return initial
            # Warm the same shape and decode path; use a disjoint prefix, then flush.
            for warm in range(2):
                await request(
                    session,
                    base,
                    [60000 + warm] + workload[0]["tokens"][1:],
                    args.output,
                )
            for _ in range(100):
                if (await observe(session, base))["compression_pending"] == 0:
                    break
                await asyncio.sleep(0.1)
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

                async def execute(index, item):
                    result = await request(
                        session, base, item["tokens"], item.get("output", args.output)
                    )
                    result.update(group=item["group"], round=item["round"], index=index)
                    rows.append(result)
                    raw.write(json.dumps(result) + "\n")
                    raw.flush()

                if config.get("replay_schedule") == "causal_sessions":
                    await replay_causal_sessions(workload, args.concurrency, execute)
                else:
                    for round_index in range(args.rounds):
                        semaphore = asyncio.Semaphore(args.concurrency)

                        async def limited(index, item):
                            async with semaphore:
                                await execute(index, item)

                        items = [
                            (i, item)
                            for i, item in enumerate(workload)
                            if item["round"] == round_index
                        ]
                        await asyncio.gather(*(limited(i, item) for i, item in items))
                        metrics = await observe(session, base)
                        save(directory / f"round_{round_index}_telemetry.json", metrics)
                        print(
                            f"{directory.name}: {len(rows)}/{len(workload)} requests, evictions={metrics.get('evicted_entries', 0)}",
                            flush=True,
                        )
            duration = time.perf_counter() - start
            after = await observe(session, base)
            save(directory / "after.json", after)
            metrics = summarize_requests(rows, duration, before, after, args)
            if config.get("workload_kind") == "sharegpt_multiturn":
                by_index = {r["index"]: r for r in rows}
                expected_reuse = sum(
                    item.get("prior_input_tokens", 0) for item in workload
                )
                metrics["prior_input_prefix_tokens"] = expected_reuse
                metrics["prior_input_prefix_fraction"] = expected_reuse / sum(
                    r["input_tokens"] for r in rows
                )
                metrics["recomputed_prefix_tokens"] = sum(
                    max(
                        0,
                        item.get("prior_input_tokens", 0)
                        - by_index[i]["cached_tokens"],
                    )
                    for i, item in enumerate(workload)
                )
                followups = [r for r in rows if r["round"] > 0]
                metrics["followup_hit_request_fraction"] = (
                    sum(r["cached_tokens"] > 0 for r in followups) / len(followups)
                    if followups
                    else 0.0
                )
            if config.get("workload_kind") == "sharegpt_first_turn":
                # This workload has no declared repeated prefix to recompute.
                metrics.pop("recomputed_prefix_tokens")
            for key in ("prefill_deferred_batches", "prefill_deferral_wait_ms"):
                metrics[key] = after.get(key, 0) - before.get(key, 0)
            if expected_cap is not None:
                batches = after.get("svd_batches", 0) - before.get("svd_batches", 0)
                items = after.get("svd_batch_items", 0) - before.get(
                    "svd_batch_items", 0
                )
                metrics.update(
                    svd_batches=batches,
                    svd_batch_items=items,
                    svd_batch_mean=items / batches if batches else 0,
                    svd_batch_max=after.get("svd_batch_max", 0),
                )
            validation = {
                "all_requests_completed": len(rows) == len(workload),
                "within_cache_budget": after["peak_cache_state_bytes"]
                <= config["budget_bytes"],
                "baseline_evicted": bool(metrics["evicted_entries"])
                if not config["compression"]
                and not getattr(args, "pilot", False)
                and not config.get("require_no_evictions")
                and config.get("require_baseline_eviction", True)
                else None,
                "no_retractions": None
                if config.get("allow_retractions")
                else all(r["meta_info"]["total_retractions"] == 0 for r in rows),
                "staging_within_reserve": after.get("staging_peak_bytes", 0)
                <= config["staging_reserve_bytes"],
                "expected_compressed_pool": initial["compressed_pool_bytes"]
                == config.get("compressed_slots", 0) * config["compressed_state_bytes"],
                "cold_cache": before["full_free_slots"] == config["full_slots"]
                and before["compressed_entries"] == 0
                and before["compression_pending"] == 0,
                "cold_pass_misses": all(
                    r["cached_tokens"] == 0 for r in rows if r["round"] == 0
                )
                if not config.get("workload_kind", "").startswith("sharegpt_")
                else None,
                "no_compression_failures": metrics["compression_failed"] == 0,
            }
            result = {
                "config": config,
                "repetition": rep,
                "metrics": metrics,
                "validation": validation,
            }
            if config.get("require_no_evictions"):
                validation["no_cache_evictions"] = (
                    metrics["evicted_entries"] == 0
                    and metrics["evicted_kv_tokens"] == 0
                )
            if (
                config.get("workload_kind") == "sharegpt_multiturn"
                and config.get("require_prefix_reuse", True)
                and (
                    not config.get("prefix_reuse_pilot_only")
                    or getattr(args, "pilot", False)
                )
            ):
                validation["shared_prefix_reuse_observed"] = (
                    metrics["token_cache_hit_rate"] >= 0.10
                )
                validation["followup_requests_hit_cache"] = (
                    metrics["followup_hit_request_fraction"] >= 0.50
                )
            if expected_cap is not None:
                validation["svd_batches_observed"] = metrics["svd_batches"] > 0
                validation["svd_batch_limit_respected"] = (
                    metrics["svd_batch_max"] <= expected_cap
                )
            if production_defaults and config["compression"]:
                validation["deferral_exercised"] = (
                    metrics["prefill_deferred_batches"] > 0
                )
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
