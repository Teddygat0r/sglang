"""Bounded Spark concurrency/memory experiment; preserves all prior results."""
import argparse
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
from argparse import Namespace
from scipy.stats import t
from spark_run import one_run, save, trace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def configs():
    g = json.loads((HERE / 'results/full_20260913/discovery/initial.json').read_text())
    full = g['full_state_bytes']
    l, _, h, d, s = g['temporal_shape']
    compressed = full - l*h*d*s*g['temporal_element_bytes'] + l*h*16*(d+1+s)*g['temporal_element_bytes']
    result = []
    for budget in (64, 128, 192):
        for concurrency in (2, 4):
            for on in (False, True):
                slots = 32 if on else budget
                reserve = 16*full if on else 0
                ceiling = g['kv_pool_bytes'] + (budget+1)*full
                count = ((budget-slots)*full-reserve)//compressed if on else 0
                assert not on or count > 0
                result.append(dict(baseline_slots=budget, concurrency=concurrency, compression=on,
                                   full_slots=slots, compressed_slots=count, compressed_state_bytes=compressed,
                                   kv_tokens=262144, budget_bytes=ceiling, staging_reserve_bytes=reserve))
    return result


def summary(root):
    groups = {}
    for path in sorted((root/'full').glob('*/result.json')):
        r = json.loads(path.read_text())
        key = path.parent.name.rsplit('_r', 1)[0]
        groups.setdefault(key, []).append(r)
    output = {}
    lines = ['# Spark concurrent cache-pressure experiment', '',
             'Means and Student-t 95% CIs across independent repetitions. Synthetic prefixes, concurrency 2/4, 128 forced output tokens. LRU; graphs and overlap disabled. Not a quality evaluation or peak-capacity claim.', '',
             'Matched ceilings cover KV, Mamba pools and snapshot/result staging; exclude weights, arithmetic workspace and allocator reserve.', '']
    for key, runs in groups.items():
        metrics = {}
        lines += [f'## {key} (n={len(runs)})', '', '| Metric | Mean | 95% CI |', '|---|---:|---|']
        for name in runs[0]['metrics']:
            values = [r['metrics'][name] for r in runs]
            mean = statistics.mean(values)
            half = float(t.ppf(.975,len(values)-1))*statistics.stdev(values)/math.sqrt(len(values)) if len(values)>1 else None
            ci = [mean-half,mean+half] if half is not None else None
            metrics[name] = dict(mean=mean,ci95=ci)
            lines.append(f'| {name} | {mean:.6g} | {ci} |')
        output[key] = dict(n=len(runs), config=runs[0]['config'],metrics=metrics)
        lines.append('')
    save(root/'summary.json', output)
    (root/'report.md').write_text('\n'.join(lines))


async def experiment(root):
    design = configs()
    save(root/'protocol.json', dict(configs=design, repetitions=5, groups=64, rounds=3, prefix=2048,
         output=128, full_slots_on=32, staging_full_states=16, policy='lru', seed=20260920,
         expected_runs=60, pilots='Four runs at smallest budget, both modes and concurrencies; 8 groups, 2 rounds, 128 output tokens; excluded from analysis.',
         ordering='Paired off/on; reverse pair and budget/concurrency order on odd repetitions. Round barriers prevent cross-round concurrency.',
         limitations='Synthetic workload, no quality validation, graphs/overlap disabled. Closed-loop throughput, TTFT measured from client submission excluding semaphore wait.',
         hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                 [Path(__file__),HERE/'spark_run.py',HERE/'instrumentation.py',HERE/'search_variant/server.py']}))
    completed = 0
    for phase in ('pilot','full'):
        (root/phase).mkdir()
        repetitions = 1 if phase=='pilot' else 5
        for rep in range(repetitions):
            ordered = design[:4] if phase=='pilot' else (design if rep%2==0 else list(reversed(design)))
            for c in ordered:
                args = Namespace(groups=8 if phase=='pilot' else 64, rounds=2 if phase=='pilot' else 3,
                                 prefix=2048,output=128,seed=20260919 if phase=='pilot' else 20260920,
                                 port=31037,concurrency=c['concurrency'],pilot=phase=='pilot')
                workload = trace(args,rep)
                label = f"b{c['baseline_slots']}_c{c['concurrency']}_{'on' if c['compression'] else 'off'}_r{rep}"
                save(root/'status.json',dict(state='running',phase=phase,run=label,completed_measured_runs=completed,
                     updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
                target = root/phase/label
                print(f'Starting {phase}/{label}',flush=True)
                result = await one_run(args,c,rep,target,workload)
                save(target/'trace.json',workload)
                if phase=='full':
                    completed += 1
                    summary(root)
    assert completed==60
    save(root/'status.json',dict(state='complete',measured_runs=completed,pilot_runs=4,report=str(root/'report.md')))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--launch',action='store_true')
    options=p.parse_args()
    root=options.results.resolve()
    if options.launch:
        root.mkdir(parents=True,exist_ok=False)
        with (root/'supervisor.log').open('w') as log, open(os.devnull) as stdin:
            proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--results',str(root)],
                                  cwd=ROOT,stdin=stdin,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(root/'supervisor.json',dict(pid=proc.pid,started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
        print(json.dumps(dict(pid=proc.pid,results=str(root))))
        return
    try:
        asyncio.run(experiment(root))
    except BaseException as error:
        prior=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {}
        save(root/'status.json',{**prior,'state':'failed','error':repr(error)})
        traceback.print_exc()
        raise


if __name__=='__main__':
    main()
