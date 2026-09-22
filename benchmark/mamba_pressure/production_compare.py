"""Compression off versus production default prefill deferral and batch cap 2."""

import hashlib
import subprocess
from argparse import Namespace
from pathlib import Path

from benchmark_utils import configs, main, summarize
from spark_run import ROOT, one_run, save, trace


async def experiment(root):
    allocations = configs()
    off = next(c for c in allocations if not c["compression"])
    on = next(c for c in allocations if c["full_slots"] == 32)
    design = [
        {**off, "label": "off", "production_defaults": True},
        {
            **on,
            "label": "deferral_cap2",
            "production_defaults": True,
            "expected_svd_worker_batch": 2,
        },
    ]
    sources = list(Path(__file__).parent.glob("*.py")) + [
        ROOT / "python/sglang/srt" / p
        for p in (
            "mem_cache/compression_admission.py",
            "mem_cache/mamba_radix_cache.py",
            "managers/scheduler.py",
            "server_args.py",
        )
    ]
    save(
        root / "protocol.json",
        dict(
            configs=design,
            repetitions=5,
            pilots=2,
            measured_runs=10,
            groups=64,
            rounds=3,
            prefix=2048,
            suffix=16,
            output=128,
            seed=20261210,
            checkpoint=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            hashes={
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sources
            },
            ordering="Reverse order on odd repetitions; identical traces per pair",
            interpretation="Paired on-minus-off CI; no claim of equivalence from a non-significant difference",
        ),
    )
    completed = 0
    for phase in ("pilot", "full"):
        for rep in range(1 if phase == "pilot" else 5):
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
                    rounds=2 if phase == "pilot" else 3,
                    prefix=2048,
                    output=128,
                    seed=20261209 if phase == "pilot" else 20261210,
                    port=31037,
                    concurrency=4,
                    pilot=phase == "pilot",
                )
                await one_run(args, config, rep, target, trace(args, rep))
                completed += 1
                if phase == "full":
                    summarize(
                        root,
                        baseline="off",
                        variants=("deferral_cap2",),
                        description="# Production compression off versus deferral + cap2\n\n"
                        "Qwen3.5-4B rank16; concurrency4; fixed ~14.19 GiB ceiling. "
                        "Off:128 full states. On:32 full + 298 compressed, identical KV and ceiling. "
                        "Five fresh paired repetitions; reverse order on odd repetitions. "
                        "Real default cache/scheduler path, no policy monkey-patches. "
                        "Graphs and overlap scheduling disabled to match prior studies. "
                        "See PRODUCTION_COMPRESSION.md for scope and limitations.",
                    )
    save(
        root / "status.json",
        dict(state="complete", completed=completed, report=str(root / "report.md")),
    )


if __name__ == "__main__":
    main(experiment, __file__)
