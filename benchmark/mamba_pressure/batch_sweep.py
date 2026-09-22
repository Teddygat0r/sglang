"""SVD batch-cap sweep with normal concurrent prefill and compression."""
import hashlib
import json
import subprocess
from argparse import Namespace
from pathlib import Path

from allocation_native import configs, estimate
from spark_run import one_run, save, trace, ROOT
from tail_sweep import main


def summarize(root, baseline='batch8', variants=('batch2', 'batch1'), description=None):
    groups = {}
    for path in (root / 'full').glob('*/result.json'):
        row = json.loads(path.read_text())
        if any(v is False for v in row['validation'].values()):
            continue
        groups.setdefault(row['config']['label'], {})[row['repetition']] = row['metrics']
    summary = {label: {k: estimate([m[k] for m in runs.values()]) for k in next(iter(runs.values()))}
               for label, runs in groups.items()}
    paired = {}
    for label in variants:
        reps = sorted(set(groups.get(label, {})) & set(groups.get(baseline, {})))
        if reps:
            paired[label] = {k: estimate([groups[label][r][k] - groups[baseline][r][k] for r in reps])
                             for k in groups[label][reps[0]] if k in groups[baseline][reps[0]]}
    save(root / 'summary.json', {'groups': summary, f'paired_vs_{baseline}': paired})
    lines = ['# SVD batch-cap experiment', '',
             'Qwen3.5-4B rank16, concurrency4, full32/compressed298, ~14.19 GiB ceiling. '
             'Normal SVD/prefill overlap, no experimental synchronization or profiling. '
             'Five paired repetitions; reverse cap order on odd repetitions. See SVD_BATCH_SWEEP.md.', '']
    if description:
        lines = [description, '']
    for label, metrics in summary.items():
        lines += [f'## {label} (n={len(groups[label])})', '', '| Metric | Mean | 95% CI |', '|---|---:|---|']
        lines += [f"| {k} | {v['mean']:.6g} | {v['ci95']} |" for k, v in metrics.items()]
        lines.append('')
    for label, metrics in paired.items():
        lines += [f'## Paired {label} minus {baseline}', '', '| Metric | Difference | 95% CI |', '|---|---:|---|']
        lines += [f"| {k} | {v['mean']:.6g} | {v['ci95']} |" for k, v in metrics.items()]
        lines.append('')
    (root / 'report.md').write_text('\n'.join(lines))


async def experiment(root):
    base = next(c for c in configs() if c['full_slots'] == 32)
    design = [{**base, 'label': f'batch{cap}', 'svd_worker_batch': cap} for cap in (8, 2, 1)]
    sources = list(Path(__file__).parent.glob('*.py')) + [ROOT / 'python/sglang/srt/mem_cache/mamba_radix_cache.py']
    save(root / 'protocol.json', dict(configs=design, repetitions=5, measured_runs=15, pilots=3,
         seed=20261110, groups=64, rounds=3, prefix=2048, suffix=16, output=128,
         checkpoint=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
         hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
         ordering='Reverse mode order on odd repetitions; identical trace within repetition'))
    completed = 0
    for phase in ('pilot', 'full'):
        for rep in range(1 if phase == 'pilot' else 5):
            for config in reversed(design) if phase == 'pilot' or rep % 2 else design:
                target = root / phase / f"{config['label']}_r{rep}"
                save(root / 'status.json', dict(state='running', run=str(target), completed=completed, total=18))
                args = Namespace(groups=8 if phase == 'pilot' else 64, rounds=2 if phase == 'pilot' else 3,
                                 prefix=2048, output=128, seed=20261109 if phase == 'pilot' else 20261110,
                                 port=31037, concurrency=4, pilot=phase == 'pilot')
                await one_run(args, config, rep, target, trace(args, rep))
                completed += 1
                if phase == 'full':
                    summarize(root)
    save(root / 'status.json', dict(state='complete', completed=completed, report=str(root / 'report.md')))


if __name__ == '__main__':
    main(experiment, __file__)
