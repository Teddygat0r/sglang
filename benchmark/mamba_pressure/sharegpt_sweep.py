"""ShareGPT first-turn serving with a roomy matched 64 GiB cache ceiling."""

import hashlib
import json
import random
import subprocess
from argparse import Namespace
from pathlib import Path

from benchmark_utils import configs as small_configs
from benchmark_utils import dataset_path, load_geometry, main, summarize
from spark_run import ROOT, one_run, save

DATASET = None  # Optional programmatic override; CLI uses --dataset.
REVISION = "8b0048ad6ae8c22f46a78c15559dec98feef5539"
SEED = 20261220


def configs():
    original = small_configs()
    geometry = load_geometry()
    full_bytes = geometry["full_state_bytes"]
    ceiling = 64 * 2**30
    result = []
    for on in (False, True):
        c = dict(next(c for c in original if c["compression"] == on))
        reserve = 16 * full_bytes if on else 0
        slots = 512 if on else (ceiling - c["kv_pool_bytes"]) // full_bytes - 1
        count = (
            (ceiling - c["kv_pool_bytes"] - (slots + 1) * full_bytes - reserve)
            // c["compressed_state_bytes"]
            if on
            else 0
        )
        c.update(
            label="deferral_cap2" if on else "off",
            production_defaults=True,
            full_slots=slots,
            compressed_slots=count,
            budget_bytes=ceiling,
            staging_reserve_bytes=reserve,
            mem_fraction_static=0.75,
            workload_kind="sharegpt_first_turn",
            require_no_evictions=True,
        )
        c.pop("baseline_slots", None)
        if on:
            c["expected_svd_worker_batch"] = 2
        result.append(c)
    return result


def sample(records, tokenizer, count, seed):
    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    selected, seen = [], set()
    for index in order:
        record = records[index]
        turns = record.get("conversations", record.get("conversation", []))
        if (
            len(turns) < 2
            or turns[0].get("from") != "human"
            or turns[1].get("from") != "gpt"
        ):
            continue
        prompt, answer = turns[0].get("value"), turns[1].get("value")
        if not isinstance(prompt, str) or not isinstance(answer, str):
            continue
        tokens = tokenizer.encode(prompt)
        output = len(tokenizer.encode(answer))
        # Keep real lengths, never truncate or repeat prompts to make them fit.
        if len(tokens) < 2 or not 2 <= output <= 1024 or len(tokens) + output > 4096:
            continue
        identity = tuple(tokens)
        if identity in seen:
            continue
        seen.add(identity)
        selected.append(
            dict(
                group=index,
                round=0,
                tokens=tokens,
                output=output,
                record_id=record.get("id"),
                dataset_index=index,
            )
        )
        if len(selected) == count:
            return selected
    raise ValueError(
        f"Only {len(selected)} eligible distinct prompts for {count} requests"
    )


def describe(workload):
    return dict(
        requests=len(workload),
        input_tokens=sum(len(r["tokens"]) for r in workload),
        output_tokens=sum(r["output"] for r in workload),
        max_input=max(len(r["tokens"]) for r in workload),
        max_output=max(r["output"] for r in workload),
    )


async def experiment(root):
    dataset = DATASET or dataset_path()
    from transformers import AutoTokenizer

    save(
        root / "status.json", dict(state="preparing", phase="dataset_and_memory_check")
    )
    memory = dict(
        line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines()
    )
    available = int(memory["MemAvailable"].split()[0]) * 1024
    if available < 88 * 2**30:
        raise RuntimeError(
            "Need at least 88 GiB available unified memory for the 64 GiB cache and headroom"
        )
    records = json.loads(dataset.read_text())
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", local_files_only=True)
    workloads = {rep: sample(records, tokenizer, 256, SEED + rep) for rep in range(5)}
    pilot = sample(records, tokenizer, 8, SEED - 1)
    design = configs()
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
                records=len(records),
            ),
            seed=SEED,
            available_memory_bytes=available,
            concurrency=4,
            context=4096,
            filtering="First human/gpt pair; distinct tokenized prompts; input>=2, reference output2..1024, total<=4096; no truncation, raw prompts/no chat template",
            arrivals="Closed-loop concurrency4; no original timestamps; each prompt once per repetition",
            ordering="Reverse off/on order on odd repetitions, identical request trace within each pair",
            trace_stats={
                str(rep): describe(workload) for rep, workload in workloads.items()
            },
            checkpoint=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            hashes={
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sources
            },
        ),
    )
    for rep, workload in workloads.items():
        save(root / f"trace_r{rep}.json", workload)
        save(
            root / f"workload_r{rep}.json",
            [
                dict(
                    dataset_index=r["dataset_index"],
                    record_id=r["record_id"],
                    input_tokens=len(r["tokens"]),
                    output_tokens=r["output"],
                )
                for r in workload
            ],
        )
    completed = 0
    for phase in ("pilot", "full"):
        for rep in range(1 if phase == "pilot" else 5):
            workload = pilot if phase == "pilot" else workloads[rep]
            for config in reversed(design) if phase == "pilot" or rep % 2 else design:
                target = root / phase / f"{config['label']}_r{rep}"
                save(
                    root / "status.json",
                    dict(
                        state="running", run=str(target), completed=completed, total=12
                    ),
                )
                args = Namespace(
                    groups=len(workload),
                    rounds=1,
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
                        description="# ShareGPT serving with a roomy cache\n\nShareGPT V4.3 first-turn requests; "
                        "256 distinct prompts per run, natural reference output lengths (2–1024), total<=4096. "
                        "No synthetic prefix repetition. Qwen3.5-4B rank16, concurrency4, shared 64 GiB cache ceiling. "
                        "Five paired repetitions, alternating order, no evictions required in BOTH modes. "
                        "Graphs and overlap scheduling remain disabled. See SHAREGPT_ROOMY.md for limitations.",
                    )
    save(
        root / "status.json",
        dict(state="complete", completed=completed, report=str(root / "report.md")),
    )


if __name__ == "__main__":
    main(experiment, __file__)
