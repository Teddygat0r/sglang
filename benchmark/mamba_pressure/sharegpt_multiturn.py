"""Interleaved multi-turn ShareGPT reference-history prefix-cache evaluation."""

import hashlib
import json
import random
import subprocess
from argparse import Namespace
from pathlib import Path

from benchmark_utils import configs as pressure_allocations
from benchmark_utils import dataset_path, main, summarize
from sharegpt_sweep import DATASET, REVISION
from sharegpt_sweep import configs as roomy_configs
from spark_run import ROOT, one_run, save

SEED = 20261230
TURNS = 3


def sample_sessions(
    records, tokenizer, count, seed, *, turns=TURNS, uncapped=False, native_chat=False
):
    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    sessions, seen = [], set()
    for index in order:
        record = records[index]
        messages = record.get("conversations", record.get("conversation", []))
        if len(messages) < 2 * turns:
            continue
        history, rows, valid, chat = [], [], True, []
        for turn in range(turns):
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
            if native_chat:
                chat.append({"role": "user", "content": user["value"]})
                prompt = tokenizer.apply_chat_template(
                    chat,
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                    return_dict=False,
                )
            reference = tokenizer.encode(assistant["value"], add_special_tokens=False)
            if not reference or (
                not uncapped
                and (
                    not 2 <= len(reference) <= 1024
                    or len(prompt) + len(reference) > 4096
                )
            ):
                valid = False
                break
            prior = len(rows[-1]["tokens"]) if rows else 0
            if native_chat and rows:
                # Native templates can change suffix tokens when a recorded answer
                # replaces the generation prefix. Count only the exact shared IDs.
                prior = 0
                for old, new in zip(rows[-1]["tokens"], prompt):
                    if old != new:
                        break
                    prior += 1
            elif rows and prompt[:prior] != rows[-1]["tokens"]:
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
            chat.append({"role": "assistant", "content": assistant["value"]})
        if not valid or (not uncapped and tuple(rows[0]["tokens"]) in seen):
            continue
        seen.add(tuple(rows[0]["tokens"]))
        sessions.append(rows)
        if len(sessions) == count:
            return sessions
    raise ValueError(f"Only {len(sessions)} eligible {turns}-turn sessions for {count}")


def interleave(sessions, seed):
    rng = random.Random(seed)
    rows = []
    for turn in range(len(sessions[0])):
        order = list(range(len(sessions)))
        rng.shuffle(order)
        rows.extend(sessions[i][turn] for i in order)
    return rows


def describe(sessions, *, require_roomy=True):
    rows = [r for session in sessions for r in session]
    inputs = sum(len(r["tokens"]) for r in rows)
    prior = sum(r["prior_input_tokens"] for r in rows)
    # Conservative KV bound: final reference history + every generated output.
    bound = sum(len(s[-1]["tokens"]) + sum(r["output"] for r in s) for s in sessions)
    if require_roomy and bound > 240000:
        raise ValueError(f"Trace may exceed roomy KV pool: upper bound {bound}")
    return dict(
        sessions=len(sessions),
        requests=len(rows),
        input_tokens=inputs,
        output_tokens=sum(r["output"] for r in rows),
        prior_input_prefix_tokens=prior,
        prior_input_prefix_fraction=prior / inputs,
        conservative_kv_tokens=bound,
        max_input_tokens=max(len(r["tokens"]) for r in rows),
        max_output_tokens=max(r["output"] for r in rows),
        max_request_tokens=max(len(r["tokens"]) + r["output"] for r in rows),
    )


def pressure_configs():
    return [
        {
            **c,
            "label": "deferral_cap2" if c["compression"] else "off",
            "production_defaults": True,
            "concurrency": 8,
            "mamba_scheduler_strategy": "extra_buffer",
            "mamba_track_interval": 256,
            "prefix_reuse_pilot_only": True,
            "context_length": None,
            "replay_schedule": "causal_sessions",
            "allow_retractions": True,
            **({"expected_svd_worker_batch": 2} if c["compression"] else {}),
        }
        for c in pressure_allocations()
        if c["full_slots"] in (128, 32)
    ]


async def experiment(root, *, pressure=False):
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
    turns = 10 if pressure else TURNS
    concurrency = 8 if pressure else 4
    sampling = dict(turns=turns, uncapped=pressure, native_chat=pressure)
    sessions = {
        rep: sample_sessions(records, tokenizer, 64, SEED + rep, **sampling)
        for rep in range(5)
    }
    pilots = sample_sessions(records, tokenizer, 8, SEED - 1, **sampling)
    stats = {
        str(rep): describe(s, require_roomy=not pressure) for rep, s in sessions.items()
    }
    pilot_stats = describe(pilots, require_roomy=not pressure)
    if pressure:
        from transformers import AutoConfig

        model_config = AutoConfig.from_pretrained(
            "Qwen/Qwen3.5-4B", local_files_only=True
        )
        native_context = model_config.get_text_config().max_position_embeddings
        if (
            max(s["max_request_tokens"] for s in [*stats.values(), pilot_stats])
            > native_context
        ):
            raise ValueError(
                "Selected trace exceeds native model context; no records were shortened or dropped"
            )
    design = [
        {**c, "workload_kind": "sharegpt_multiturn"}
        for c in (pressure_configs() if pressure else roomy_configs())
    ]
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
    provenance_path = dataset.with_suffix(".metadata.json")
    provenance = (
        json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    )
    if provenance.get("sha256", digest) != digest:
        raise ValueError("Dataset checksum does not match its provenance metadata")
    save(
        root / "protocol.json",
        dict(
            configs=design,
            repetitions=5,
            pilots=2,
            measured_runs=10,
            dataset=dict(
                repo=provenance.get(
                    "repo", None if pressure else "Aeala/ShareGPT_Vicuna_unfiltered"
                ),
                revision=provenance.get("revision", None if pressure else REVISION),
                path=str(dataset),
                sha256=digest,
                provenance=provenance,
            ),
            seed=SEED,
            session_count=64,
            turns=turns,
            concurrency=concurrency,
            context=None if pressure else 4096,
            native_model_context=native_context if pressure else None,
            trace_stats=stats,
            pilot_stats=pilot_stats,
            replay=(
                "Native chat template, thinking disabled; real recorded history; ignore_eos for full recorded output length; exact token-prefix opportunity measured"
                if pressure
                else "Reference-history replay, not generated-output continuation; append encoded USER/ASSISTANT segments; ignore_eos for recorded output length"
            ),
            filtering=(
                "First ten human/gpt pairs; nonempty tokenized replies; no context/output-length filter or truncation; preserve duplicate prompts"
                if pressure
                else "First three human/gpt pairs; output2..1024 each; cumulative context+output<=4096; distinct first prompts; no truncation"
            ),
            arrivals=(
                f"All sampled sessions ready at time zero, seeded initial order, closed-loop concurrency{concurrency}; each next turn eligible after its own predecessor; no global barriers or invented timestamps; dataset contains no arrival/think times"
                if pressure
                else "Seed-shuffled sessions within each turn; round barriers preserve causality, closed-loop concurrency4; no real think-time/arrival timestamps"
            ),
            ordering="Reverse mode order on odd repetitions; same trace within each pair",
            cache_validation=(
                "Pilots require >=10% token hits and >=50% followups with nonzero hits; "
                "full baseline runs require eviction; full-run hit rates are outcomes. "
                "All runs require memory accounting and no compression failures; retractions are measured outcomes."
                if pressure
                else "Both pilots and full runs require >=10% token hits and >=50% followups with nonzero hits, no eviction or retraction"
            ),
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
                    rounds=turns,
                    prefix=0,
                    output=128,
                    seed=SEED - 1 if phase == "pilot" else SEED,
                    port=31037,
                    concurrency=concurrency,
                    pilot=phase == "pilot",
                )
                await one_run(args, config, rep, target, workload)
                completed += 1
                if phase == "full":
                    summarize(
                        root,
                        baseline="off",
                        variants=("deferral_cap2",),
                        description=f"# Multi-turn ShareGPT prefix-cache evaluation\n\n64 sessions × {turns} turns, reference-history replay, "
                        f"{64 * turns} requests, concurrency{concurrency}. Qwen3.5-4B rank16. "
                        + (
                            "Matched constrained cache ceiling; baseline eviction required. "
                            if pressure
                            else "Matched 64 GiB ceiling, no evictions. "
                        )
                        + "Five paired repetitions; reverse order on odd repetitions. "
                        "Pilots enforce actual reuse before full measurement. Not an exact Marconi trace reproduction. "
                        + (
                            "See SHAREGPT_PRESSURE.md for the uncapped causal replay protocol."
                            if pressure
                            else "See SHAREGPT_MULTITURN.md for serialization and measurement limitations."
                        ),
                    )
    save(
        root / "status.json",
        dict(state="complete", completed=completed, report=str(root / "report.md")),
    )


if __name__ == "__main__":
    main(experiment, __file__)
