"""Interleaved multi-turn ShareGPT reference-history prefix-cache evaluation."""

import hashlib
import json
import random
import subprocess
from argparse import Namespace
from pathlib import Path

from benchmark_utils import dataset_path, main, summarize
from sharegpt_sweep import DATASET, REVISION
from sharegpt_sweep import configs as roomy_configs
from spark_run import ROOT, one_run, save

SEED = 20261230
TURNS = 3


def sample_sessions(records, tokenizer, count, seed):
    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    sessions, seen = [], set()
    for index in order:
        record = records[index]
        messages = record.get("conversations", record.get("conversation", []))
        if len(messages) < 2 * TURNS:
            continue
        history, rows, valid = [], [], True
        for turn in range(TURNS):
            user, assistant = messages[2 * turn : 2 * turn + 2]
            if (
                user.get("from") != "human"
                or assistant.get("from") != "gpt"
                or not all(isinstance(m.get("value"), str) for m in (user, assistant))
            ):
                valid = False
                break
            # Segment-wise encoding intentionally preserves exact previous input
            # token IDs across turns, independent of BPE boundary retokenization.
            new_user = tokenizer.encode(
                "USER:\n" + user["value"] + "\nASSISTANT:\n", add_special_tokens=False
            )
            prompt = history + new_user
            reference = tokenizer.encode(assistant["value"], add_special_tokens=False)
            if not 2 <= len(reference) <= 1024 or len(prompt) + len(reference) > 4096:
                valid = False
                break
            prior = len(rows[-1]["tokens"]) if rows else 0
            if rows and prompt[:prior] != rows[-1]["tokens"]:
                raise AssertionError("Conversation prefix changed during serialization")
            rows.append(
                dict(
                    group=index,
                    round=turn,
                    tokens=prompt,
                    output=len(reference),
                    prior_input_tokens=prior,
                    dataset_index=index,
                    record_id=record.get("id"),
                )
            )
            history = (
                prompt + reference + tokenizer.encode("\n", add_special_tokens=False)
            )
        if not valid or tuple(rows[0]["tokens"]) in seen:
            continue
        seen.add(tuple(rows[0]["tokens"]))
        sessions.append(rows)
        if len(sessions) == count:
            return sessions
    raise ValueError(f"Only {len(sessions)} eligible three-turn sessions for {count}")


def interleave(sessions, seed):
    rng = random.Random(seed)
    rows = []
    for turn in range(TURNS):
        order = list(range(len(sessions)))
        rng.shuffle(order)
        rows.extend(sessions[i][turn] for i in order)
    return rows


def describe(sessions):
    rows = [r for session in sessions for r in session]
    inputs = sum(len(r["tokens"]) for r in rows)
    prior = sum(r["prior_input_tokens"] for r in rows)
    # Conservative KV bound: final reference history + every generated output.
    bound = sum(len(s[-1]["tokens"]) + sum(r["output"] for r in s) for s in sessions)
    if bound > 240000:
        raise ValueError(f"Trace may exceed roomy KV pool: upper bound {bound}")
    return dict(
        sessions=len(sessions),
        requests=len(rows),
        input_tokens=inputs,
        output_tokens=sum(r["output"] for r in rows),
        prior_input_prefix_tokens=prior,
        prior_input_prefix_fraction=prior / inputs,
        conservative_kv_tokens=bound,
    )


async def experiment(root):
    dataset = DATASET or dataset_path()
    from transformers import AutoTokenizer

    save(root / "status.json", dict(state="preparing", phase="multiturn_trace"))
    memory = dict(
        line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines()
    )
    if int(memory["MemAvailable"].split()[0]) * 1024 < 88 * 2**30:
        raise RuntimeError(
            "Need 88 GiB available unified memory for the roomy-cache comparison"
        )
    records = json.loads(dataset.read_text())
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", local_files_only=True)
    sessions = {
        rep: sample_sessions(records, tokenizer, 64, SEED + rep) for rep in range(5)
    }
    pilots = sample_sessions(records, tokenizer, 8, SEED - 1)
    stats = {str(rep): describe(s) for rep, s in sessions.items()}
    describe(pilots)
    design = [{**c, "workload_kind": "sharegpt_multiturn"} for c in roomy_configs()]
    sources = list(Path(__file__).parent.glob("*.py")) + [
        ROOT / "python/sglang/srt" / p
        for p in (
            "server_args.py",
            "managers/scheduler.py",
            "mem_cache/mamba_radix_cache.py",
            "mem_cache/compression_admission.py",
        )
    ]
    with dataset.open("rb") as data:
        digest = hashlib.file_digest(data, "sha256").hexdigest()
    save(
        root / "protocol.json",
        dict(
            configs=design,
            repetitions=5,
            pilots=2,
            measured_runs=10,
            dataset=dict(
                repo="Aeala/ShareGPT_Vicuna_unfiltered",
                revision=REVISION,
                path=str(dataset),
                sha256=digest,
            ),
            seed=SEED,
            session_count=64,
            turns=TURNS,
            concurrency=4,
            context=4096,
            trace_stats=stats,
            replay="Reference-history replay, not generated-output continuation; append encoded USER/ASSISTANT segments; ignore_eos for recorded output length",
            filtering="First three human/gpt pairs; output2..1024 each; cumulative context+output<=4096; distinct first prompts; no truncation",
            arrivals="Seed-shuffled sessions within each turn; round barriers preserve causality, closed-loop concurrency4; no real think-time/arrival timestamps",
            ordering="Reverse mode order on odd repetitions; same trace within each pair",
            cache_validation="Both pilots and full runs require >=10% token hits and >=50% followups with nonzero hits, no eviction or retraction",
            checkpoint=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            hashes={
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sources
            },
        ),
    )
    workloads = {rep: interleave(s, SEED + rep) for rep, s in sessions.items()}
    pilot_trace = interleave(pilots, SEED - 1)
    save(root / "trace_pilot.json", pilot_trace)
    for rep, workload in workloads.items():
        save(root / f"trace_r{rep}.json", workload)
        save(
            root / f"workload_r{rep}.json",
            [
                {k: v for k, v in r.items() if k != "tokens"}
                | {"input_tokens": len(r["tokens"])}
                for r in workload
            ],
        )
    completed = 0
    for phase in ("pilot", "full"):
        for rep in range(1 if phase == "pilot" else 5):
            workload = pilot_trace if phase == "pilot" else workloads[rep]
            for config in reversed(design) if phase == "pilot" or rep % 2 else design:
                target = root / phase / f"{config['label']}_r{rep}"
                save(
                    root / "status.json",
                    dict(
                        state="running", run=str(target), completed=completed, total=12
                    ),
                )
                args = Namespace(
                    groups=8 if phase == "pilot" else 64,
                    rounds=TURNS,
                    prefix=0,
                    output=128,
                    seed=SEED - 1 if phase == "pilot" else SEED,
                    port=31037,
                    concurrency=4,
                    pilot=phase == "pilot",
                )
                await one_run(args, config, rep, target, workload)
                completed += 1
                if phase == "full":
                    summarize(
                        root,
                        baseline="off",
                        variants=("deferral_cap2",),
                        description="# Multi-turn ShareGPT prefix-cache evaluation\n\n64 sessions × three turns, reference-history replay, "
                        "192 requests, concurrency4. Qwen3.5-4B rank16, matched 64 GiB ceiling, no evictions. "
                        "Five paired repetitions; reverse order on odd repetitions. "
                        "Pilots enforce actual reuse before full measurement. Not an exact Marconi trace reproduction. "
                        "See SHAREGPT_MULTITURN.md for serialization and measurement limitations.",
                    )
    save(
        root / "status.json",
        dict(state="complete", completed=completed, report=str(root / "report.md")),
    )


if __name__ == "__main__":
    main(experiment, __file__)
