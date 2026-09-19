"""Exploratory matched tail profile with a dedicated profiling server."""
import argparse
import asyncio
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from argparse import Namespace
from pathlib import Path

from allocation_native import configs
from spark_run import one_run, save, trace


def overlap_ms(spans, start, end):
    intervals = sorted((max(start, s['start_ns']), min(end, s['end_ns'])) for s in spans
                       if s['start_ns'] < end and s['end_ns'] > start)
    total = 0
    last = start
    for lo, hi in intervals:
        total += max(0, hi - max(last, lo))
        last = max(last, hi)
    return total / 1e6


def analyze(directory):
    rows = [json.loads(line) for line in (directory / 'requests.jsonl').read_text().splitlines()]
    spans = [json.loads(line) for p in directory.glob('spans-*.jsonl') for line in p.read_text().splitlines()]
    if not spans:
        raise RuntimeError('Missing profiling spans')
    threshold = sorted(r['ttft_ms'] for r in rows)[int(.95 * (len(rows) - 1))]
    details = []
    for row in rows:
        rid = row['meta_info']['id']
        own = [s for s in spans if s.get('rid') == rid or rid in s.get('rids', [])]
        if not any(s['kind'] == 'forward' for s in own):
            raise RuntimeError(f'No forward spans for request {rid}')
        restores = [s for s in own if s['kind'] == 'restore']
        start, end = row['start_ns'], row['first_ns']
        detail = dict(index=row['index'], round=row['round'], rid=rid, ttft_ms=row['ttft_ms'],
                      tail=row['ttft_ms'] >= threshold,
                      cache_path='compressed' if any(s['compressed'] for s in restores) else
                      ('full' if row['cached_tokens'] else 'miss'))
        for kind in ('cache_lookup', 'restore', 'forward'):
            relevant = [s for s in own if s['kind'] == kind and s['start_ns'] < end and s['end_ns'] > start]
            detail[kind + '_host_ms'] = overlap_ms(relevant, start, end)
            detail[kind + '_gpu_span_ms'] = sum(s.get('gpu_ms', 0) for s in relevant)
        for kind in ('svd', 'commit_loop'):
            detail[kind + '_host_overlap_ms'] = overlap_ms([s for s in spans if s['kind'] == kind], start, end)
        forward = [s['start_ns'] for s in own if s['kind'] == 'forward' and start <= s['start_ns'] <= end]
        detail['client_to_first_forward_ms'] = (min(forward) - start) / 1e6 if forward else None
        details.append(detail)
    save(directory / 'tail_requests.json', details)
    summary = {}
    for category in ('all', 'tail', 'non_tail', 'compressed', 'full', 'miss'):
        subset = [r for r in details if category == 'all' or
                  (category == 'tail' and r['tail']) or (category == 'non_tail' and not r['tail']) or r['cache_path'] == category]
        summary[category] = {'n': len(subset), 'means': {
            key: statistics.mean(r[key] for r in subset if r[key] is not None)
            for key in details[0] if key.endswith('_ms') and any(r[key] is not None for r in subset)}}
    save(directory / 'tail_summary.json', summary)
    return summary


async def experiment(root):
    base = configs()
    off = next(c for c in base if not c['compression'])
    eager = next(c for c in base if c['full_slots'] == 32)
    design = [{**off, 'label': 'off'}, {**eager, 'label': 'eager'}]
    save(root / 'protocol.json', dict(configs=design, repetitions=1, seed=20261030,
         purpose='Exploratory profiling, not a confirmatory latency comparison; Study C 32-slot allocation.',
         hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}))
    completed = 0
    for phase in ('pilot', 'full'):
        for index, config in enumerate(design):
            modes = [True] if phase == 'pilot' else ([False, True] if index % 2 == 0 else [True, False])
            for enabled in modes:
                label = config['label'] + ('_profile' if enabled else '_control')
                cfg = {**config, 'label': label, 'tail_profile': enabled}
                target = root / phase / label
                save(root / 'status.json', dict(state='running', run=str(target), completed=completed, total=6))
                args = Namespace(groups=8 if phase == 'pilot' else 64, rounds=2 if phase == 'pilot' else 3,
                                 prefix=2048, output=128, seed=20261030, port=31037, concurrency=4, pilot=phase == 'pilot')
                await one_run(args, cfg, 0, target, trace(args, 0))
                if enabled:
                    analyze(target)
                completed += 1
    lines = ['# Exploratory tail profiling', '',
             'One repetition per mode: descriptive only, no confidence intervals. See TAIL_PROFILE.md for measurement limitations.', '']
    for result in sorted((root / 'full').glob('*/result.json')):
        row = json.loads(result.read_text())
        m = row['metrics']
        lines += [f"## {result.parent.name}", '', f"Mean/P95/P99 TTFT: {m['mean_ttft_ms']:.2f}/{m['p95_ttft_ms']:.2f}/{m['p99_ttft_ms']:.2f} ms", '']
        summary = result.parent / 'tail_summary.json'
        if summary.exists():
            lines += ['```json', summary.read_text(), '```', '']
    (root / 'report.md').write_text('\n'.join(lines))
    save(root / 'status.json', dict(state='complete', completed=completed, report=str(root / 'report.md')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    root = args.results.resolve()
    if args.launch:
        root.mkdir(parents=True, exist_ok=False)
        with (root / 'supervisor.log').open('w') as log, open(os.devnull) as stdin:
            proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--results', str(root)],
                                    cwd=Path(__file__).resolve().parents[2], stdin=stdin,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        save(root / 'supervisor.json', dict(pid=proc.pid, started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
        print(json.dumps(dict(pid=proc.pid, results=str(root))))
        return
    try:
        asyncio.run(experiment(root))
    except BaseException as error:
        prior = json.loads((root / 'status.json').read_text()) if (root / 'status.json').exists() else {}
        save(root / 'status.json', {**prior, 'state': 'failed', 'error': repr(error)})
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
