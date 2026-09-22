"""Nonblocking prefill deferral versus eager SVD, both with batch cap 8."""
import hashlib
import subprocess
from argparse import Namespace
from pathlib import Path
from allocation_native import configs
from batch_sweep import summarize
from spark_run import one_run, save, trace, ROOT
from tail_sweep import main


async def experiment(root, combined=False):
    base = {**next(c for c in configs() if c['full_slots'] == 32), 'svd_worker_batch': 8}
    design = [{**base, 'label': 'baseline'}, {**base, 'label': 'defer', 'defer_prefill': True}]
    if combined:
        design = [{**base, 'label': f'defer_batch{cap}', 'defer_prefill': True, 'svd_worker_batch': cap}
                  for cap in (8, 2, 1)]
    seed = 20261130 if combined else 20261120
    total = len(design) * 6
    sources = list(Path(__file__).parent.glob('*.py')) + [ROOT / 'python/sglang/srt/mem_cache/mamba_radix_cache.py', ROOT / 'python/sglang/srt/managers/scheduler.py']
    save(root / 'protocol.json', dict(configs=design, repetitions=5, measured_runs=len(design)*5, pilots=len(design),
        seed=seed, groups=64, rounds=3, prefix=2048, suffix=16, output=128,
        checkpoint=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        ordering='Reverse order on odd repetitions, same seed/trace per pair'))
    completed = 0
    for phase in ('pilot', 'full'):
        for rep in range(1 if phase == 'pilot' else 5):
            for config in reversed(design) if phase == 'pilot' or rep % 2 else design:
                target = root / phase / f"{config['label']}_r{rep}"
                save(root / 'status.json', dict(state='running', run=str(target), completed=completed, total=total))
                args = Namespace(groups=8 if phase == 'pilot' else 64, rounds=2 if phase == 'pilot' else 3,
                    prefix=2048, output=128, seed=seed-1 if phase == 'pilot' else seed,
                    port=31037, concurrency=4, pilot=phase == 'pilot')
                await one_run(args, config, rep, target, trace(args, rep))
                completed += 1
                if phase == 'full':
                    if combined:
                        summarize(root, baseline='defer_batch8', variants=('defer_batch2','defer_batch1'), description=
                            '# Prefill deferral plus smaller SVD batches\n\nDeferral enabled in all arms; caps 8, 2, 1. '
                            'Five matched repetitions; reverse order on odd repetitions. Qwen3.5-4B rank16, concurrency4, '
                            'full32/compressed298, ~14.19 GiB. No profiling or blocking prefill synchronization. See COMBINED_DEFER_BATCH.md.')
                        continue
                    summarize(root, baseline='baseline', variants=('defer',), description=
                        '# Nonblocking prefill-aware compression\n\nBatch cap 8 in both arms. '
                        'Qwen3.5-4B rank16, concurrency4, full32/compressed298, ~14.19 GiB. '
                        'Five paired repetitions, alternate order, no profiling or prefill synchronization. See DEFER_PREFILL.md.')
    save(root / 'status.json', dict(state='complete', completed=completed, report=str(root / 'report.md')))


if __name__ == '__main__':
    main(experiment, __file__)
