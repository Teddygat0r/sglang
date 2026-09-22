"""Recompute paper tables and paired CIs from immutable per-run results."""
import hashlib
import json
import math
from pathlib import Path
import statistics
from scipy.stats import t

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'
OUT = HERE / 'COMBINED_PAPER_REPORT.md'
STUDIES = [
    ('A', 'Original allocation (diagnostic)', RESULTS/'revised_20260914/full', 6, 24),
    ('B', 'Reallocated cache (held-out validation)', RESULTS/'allocation_policy_20260915_fixed/validation', 6, 24),
    ('C', 'Concurrent serving (primary result)', RESULTS/'spark_concurrency_20260916/full', 5, 60),
]
METRICS = [
    ('token_cache_hit_rate','Token cache-hit rate (%)',100),
    ('evicted_entries','Evicted state entries',1),
    ('evicted_compressed_entries','Evicted compressed entries',1),
    ('evicted_kv_tokens','Evicted attention tokens',1),
    ('recomputed_prefix_tokens','Recomputed prefix tokens',1),
    ('request_throughput_rps','Request throughput (requests/s)',1),
    ('output_throughput_tps','Output throughput (tokens/s)',1),
    ('mean_ttft_ms','Mean TTFT (ms)',1),
    ('p50_ttft_ms','P50 TTFT (ms)',1),
    ('p95_ttft_ms','P95 TTFT (ms)',1),
    ('p99_ttft_ms','P99 TTFT (ms)',1),
    ('mean_tpot_ms','Mean TPOT (ms)',1),
    ('peak_cache_state_mib','Peak cache-state memory (MiB)',1),
    ('peak_staging_mib','Peak staging memory (MiB)',1),
    ('peak_process_cuda_mib','Peak process CUDA allocated (MiB)',1),
    ('compression_pending_peak','Peak outstanding compression jobs',1),
    ('compression_completion_per_s','Compression completions/s',1),
    ('compression_completion_fraction','Compression completion fraction (%)',100),
    ('compression_failed','Compression failures',1),
]


def read(p):
    return json.loads(p.read_text())


def stats(values):
    mean=statistics.mean(values)
    half=float(t.ppf(.975,len(values)-1))*statistics.stdev(values)/math.sqrt(len(values))
    return [mean,mean-half,mean+half]


def cell(values):
    mean,lo,hi=stats(values)
    return f'{mean:,.3f} [{lo:,.3f}, {hi:,.3f}]'


def main():
    groups={}
    checked=[]
    for code,title,directory,n,expected in STUDIES:
        files=sorted(directory.glob('*/result.json'))
        assert len(files)==expected
        for p in files:
            r=read(p); c=r['config']; folder=p.parent
            assert all(v for v in r['validation'].values() if v is not None), p
            rows=[json.loads(line) for line in (folder/'requests.jsonl').read_text().splitlines()]
            before=read(folder/'before.json'); after=read(folder/'after.json')
            assert len(rows)==r['metrics']['requests']==192
            assert len({row['index'] for row in rows})==192
            assert before['full_free_slots']==c['full_slots']
            assert before['compressed_entries']==before['compression_pending']==0
            assert after['peak_cache_state_bytes']<=c['budget_bytes']
            assert after.get('staging_peak_bytes',0)<=c['staging_reserve_bytes']
            assert all(row['meta_info']['total_retractions']==0 for row in rows)
            assert all(row['input_tokens']==2064 and row['output_tokens']==(128 if code=='C' else 8) for row in rows)
            assert all(0<=row['cached_tokens']<=2048 for row in rows)
            assert all(row['cached_tokens']==0 for row in rows if row['round']==0)
            assert math.isclose(r['metrics']['token_cache_hit_rate'],sum(row['cached_tokens'] for row in rows)/sum(row['input_tokens'] for row in rows),abs_tol=1e-12)
            assert r['metrics']['recomputed_prefix_tokens']==sum(2048-row['cached_tokens'] for row in rows if row['round']>0)
            if not c['compression']:
                assert r['metrics']['evicted_entries']>0
            trace_path=folder/'trace.json' if code=='C' else directory/f"trace_{r['repetition']}.json"
            trace=read(trace_path)
            for row in rows:
                item=trace[row['index']]
                assert (row['round'],row['group'])==(item['round'],item['group'])
            r['_trace_hash']=hashlib.sha256(trace_path.read_bytes()).hexdigest()
            r['_path']=str(p.relative_to(HERE))
            key=(code,c['baseline_slots'],c.get('concurrency',1))
            groups.setdefault(key,{False:[],True:[]})[c['compression']].append(r)
            checked.append(dict(path=r['_path'],sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        for key,pair in groups.items():
            if key[0]!=code: continue
            for mode in (False,True):
                pair[mode].sort(key=lambda r:r['repetition'])
                assert [r['repetition'] for r in pair[mode]]==list(range(n))
            for off,on in zip(pair[False],pair[True]):
                assert off['_trace_hash']==on['_trace_hash']
                assert off['config']['budget_bytes']==on['config']['budget_bytes']

    intro = '''# Matched-memory rank-16 state compression: combined paper report

## 1. Answer to the research question

**Yes, increased retained state capacity can translate into live serving gains under matched cache-memory ceilings, but only with suitable allocation and enough memory to retain useful prefixes.** The evidence includes negative results: the original partition regressed, the smallest budgets still regress after reallocation, and concurrent P95/P99 TTFT worsens even where mean TTFT and throughput improve.

This document consolidates three completed studies without pooling incompatible workloads or tuning samples. All numbers below are recomputed from saved per-run results, with raw-request/memory checks. It covers **108 confirmatory runs (20,736 requests)**: 24 original-allocation runs, 24 held-out reallocation runs, and 60 concurrent runs. A separate 18-run exploratory screen is described below but excluded from confirmatory statistics. Pilots and failed attempts are excluded.

The report addresses the live capacity/overhead question. It does not re-estimate the paper's separate simulation or quality results; their numerical evidence is not substituted with these serving measurements.

## 2. Design and reproducibility

All studies use Qwen/Qwen3.5-4B and rank-16 asynchronous state compression on one NVIDIA GB10 (DGX Spark). The recorded model snapshot is `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Study A provenance records repository commit `b2982df42f5be1a21500b55545670d37ee10118a`, Python 3.12.12, and driver 580.95.05. Later study protocol files record source hashes; consult those rather than assuming an upstream release. Benchmarks use separate observation hooks and, for B/C, benchmark-only allocation/policy overrides. They do not establish a production-integrated allocator.

| Study | Purpose | Concurrency | Output tokens | Repetitions per mode/config | Cache ceilings |
|---|---|---:|---:|---:|---|
| A | Diagnose original partition | 1 | 8 | 6 | 9.20, 17.26 GiB |
| B | Validate reallocated cache on held-out seeds | 1 | 8 | 6 | 9.20, 17.26 GiB |
| C | Test concurrent serving and longer outputs | 2, 4 | 128 | 5 | 11.12, 14.19, 17.26 GiB |

Each confirmatory off/on pair uses the same budget and token trace. Mode order alternates by repetition; Study C also reverses configuration order on odd repetitions. Five repetitions give a 3:2 starting-order split, not perfect counterbalancing. Each run uses a fresh server, two disjoint warmup requests, a drain of pending warmup compression, and a cache flush. Startup and warmup are excluded. CUDA graphs, piecewise CUDA graphs, and overlap scheduling are disabled in all three studies. Context length is 4096 and chunked prefill size is 2048.

### Workload

This is a **custom synthetic shared-prefix workload**, not SGLang's built-in `generated-shared-prefix` dataset and not production traffic. Each run has 64 distinct 2048-token prefixes, three independently shuffled passes, and a unique 16-token suffix per request: 192 requests of 2064 input tokens. Temperature is zero and EOS is ignored to force the stated output length. Prefixes are random token IDs, not natural-language prompts. The prefix boundary aligns to the nominal prefill checkpoint boundary.

The first pass is cold; passes two/three offer reuse. Perfect full-prefix retention would yield `2/3 × 2048/2064 = 66.1499%` overall input-token hit rate. Thus 65–66% is near the workload ceiling, not poor reuse. Study C uses a client semaphore at concurrency 2/4, with a barrier between passes; prefixes do not overlap with themselves within a pass. Actual execution order can differ across modes due to service times.

### Memory fairness and scope

The matched ceiling covers allocated attention KV storage, full/compressed Mamba state pools, and observed owned snapshot/result staging. **It is not a total-process VRAM cap.** Model weights, arithmetic workspaces (including SVD scratch), allocator reserve, and management metadata are outside this cache-state accounting. Peak process CUDA allocated memory is reported separately; it is not reserved VRAM. Equal ceilings do not require equal observed peaks or equal useful occupancy.

One full state occupies 49.125 MiB; one rank-16 compressed state occupies 13.171875 MiB (about 3.73× smaller, including convolution state). The full pool includes an extra sentinel slot in byte accounting. Study A assigns leftover whole-token capacity to KV; B/C hold KV at 262144 tokens (8192.03125 MiB including the sentinel). Staging allowances are charged inside the ceiling: 393 MiB in A/B and 786 MiB in C. These are conservative fixed allowances, not an optimized allocation or a bound for arbitrary queue growth.

### Metric definitions and statistics

- Token hit rate: sum of server-reported cached input tokens divided by total input tokens; includes the cold pass.
- Evicted entries: lost saved Mamba states, including internal state tombstones and compressed victims. This is not the number of unique prefix groups. KV-token eviction is reported separately.
- Recomputed prefix tokens: sum of `max(0, 2048 - cached_tokens)` in passes two/three; excludes compulsory first-pass and suffix work. This is a metadata-derived proxy, not a direct GPU work counter.
- TTFT: client submission to first streaming event reporting output tokens. In C it excludes waiting for a client semaphore slot but includes server waiting after submission.
- TPOT: elapsed time from first to last output event divided by output tokens minus one (7 or 127). It is a streaming-client measure, not kernel-only decode time.
- Throughput: completed requests or generated output tokens divided by measured wall time, including pass-boundary telemetry and barriers.
- Compression activity: outstanding-job peak, completions/second, and completed/enqueued fraction at the final observation. Outstanding includes in-flight work, not only waiting queue items. Fractions below 100% do not imply failure; final jobs need not be drained before the measurement ends.
- Each cell is the mean and two-sided Student-t 95% CI across independent run repetitions: `mean ± t(0.975,n−1) × sd/√n`. Paired changes are computed from matched on-minus-off repetitions. Percentile intervals summarize variation in per-run percentiles, not pooled-request quantile uncertainty. Intervals are unadjusted for multiple comparisons; screening winners are not automatically statistically superior.

## 3. Results overview

The following percentages are ratios of mode means, not means of per-pair percentages. Positive throughput change is beneficial; negative TTFT change is beneficial. Full metric and paired-difference intervals follow.
'''
    lines=[intro,'| Study | Budget (GiB) | Concurrency | Hit rate off → on (%) | Requests/s off → on | Throughput change | Mean TTFT off → on (ms) | TTFT change |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for key,pair in sorted(groups.items()):
        code,b,c=key
        def mean(mode,k): return statistics.mean(r['metrics'][k] for r in pair[mode])
        hit=[mean(m,'token_cache_hit_rate')*100 for m in (False,True)]
        rate=[mean(m,'request_throughput_rps') for m in (False,True)]
        ttft=[mean(m,'mean_ttft_ms') for m in (False,True)]
        budget=pair[False][0]['config']['budget_bytes']/2**30
        lines.append(f'| {code} | {budget:.2f} | {c} | {hit[0]:.2f} → {hit[1]:.2f} | {rate[0]:.4f} → {rate[1]:.4f} | {(rate[1]/rate[0]-1)*100:+.2f}% | {ttft[0]:.1f} → {ttft[1]:.1f} | {(ttft[1]/ttft[0]-1)*100:+.2f}% |')
    lines += ['','## 4. Interpretation and paper-ready conclusion','',
'''Study A shows why compression alone is insufficient: at 17.26 GiB, the original partition provides 192 dense slots off, but only 81 compressed slots on alongside 162 dense slots. A saved on-run ends with 161 dense slots free while the compressed pool is full and 479 compressed entries have been evicted. Compression funnels cached states into a small pool while allocated dense memory remains largely unused. The hit-rate and throughput regression is consistent with this retention failure.

Study B restores the capacity benefit: at 17.26 GiB, 16 full slots plus 626 compressed slots eliminate state eviction and repeat-prefix recomputation in the sequential trace. Throughput increases about 18% and mean TTFT falls about 35%. The severe 9.20 GiB case remains approximately 1.3% slower. The held-out B configuration uses reuse-aware eviction, so it is not a pure allocation-only ablation against A; however, all three allocation-screen candidates with ordinary LRU also retained every reusable prefix, and C validates gains using ordinary LRU.

Study C establishes concurrent benefits on this execution path: at 14.19 GiB, throughput improves 5.7%/10.7% at concurrency 2/4 while mean TTFT falls approximately 49%/51%. At 17.26 GiB the corresponding throughput gains are 3.4%/6.4%. Longer generation makes end-to-end throughput gains smaller than prefill-latency gains; this is a plausible interpretation, not an isolated output-length ablation because other configuration details also change. The 11.12 GiB configurations lose 0.6–1.3% throughput and do not materially improve reuse.

**Tail-latency tradeoff:** C's P95/P99 TTFT worsens in every configuration. At 14.19 GiB/concurrency 4, P95 rises from about 1527 to 1591 ms while mean TTFT falls from 1055 to 520 ms. Do not claim uniformly improved latency. TPOT is nearly unchanged at concurrency 2 and modestly better at concurrency 4 at the two larger budgets.

**Eviction versus recomputation:** C's largest-budget on-runs have zero state evictions, but still report 1024/3072 recomputed prefix tokens per run at concurrency 2/4. Therefore zero eviction must not be described as zero recomputation or perfect token reuse. The specific checkpoint/matching cause has not been isolated. At the middle budget, on-runs still evict approximately 147–149 state entries while preserving almost all reusable prefix tokens; not every evicted state is needed again.

**Suggested paper text:** “We evaluate rank-16 state compression for Qwen3.5-4B on a single GB10 using matched cache-state memory ceilings and synthetic shared-prefix traffic. After correcting the full/compressed pool partition, compression converts increased state capacity into live serving gains: with 128-token outputs at concurrency four and a 14.19 GiB cache ceiling, token cache-hit rate increases from 13.98% to 64.87%, request throughput increases by 10.7%, and mean TTFT decreases by 50.7%. Benefits depend on available memory: the smallest tested concurrent configuration regresses slightly. P95/P99 TTFT also increases, so improvements are in average responsiveness and achieved throughput rather than all latency metrics.”

This is the desired live bridge between capacity and overhead: useful retained prefixes reduce repeated prefill enough to outweigh the compression-enabled path's costs in suitable regimes. It does not isolate SVD kernel overhead causally, establish realistic-traffic generalization, prove generation quality, or establish maximum production throughput.
''','## 5. Complete confirmatory tables','', 'All entries: mean [95% CI]. Differences are on minus off; percentage metrics use percentage-point differences. Compression-off completion metrics are N/A. Memory is MiB.']
    for key,pair in sorted(groups.items()):
        code,b,c=key; off=pair[False]; on=pair[True]
        co=off[0]['config']; cn=on[0]['config']
        compressed=read(Path(HERE/on[0]['_path']).parent/'initial.json')['compressed_slots']
        lines += ['',f'### {code}: {co["budget_bytes"]/2**20:.5f} MiB, concurrency {c}, n={len(off)} per mode','',
                  f'Full slots off/on: {co["full_slots"]}/{cn["full_slots"]}; compressed slots on: {compressed}; on staging allowance: {cn["staging_reserve_bytes"]/2**20:g} MiB. KV tokens off/on: {co["kv_tokens"]}/{cn["kv_tokens"]}.', '',
                  '| Metric | Off | On | Paired change |','|---|---:|---:|---:|']
        for metric,label,scale in METRICS:
            x=[r['metrics'][metric]*scale for r in off]; y=[r['metrics'][metric]*scale for r in on]
            na=metric in ('compression_completion_fraction','compression_completion_per_s')
            lines.append(f'| {label} | {"N/A" if na else cell(x)} | {cell(y)} | {"N/A" if na else cell([v-u for u,v in zip(x,y)])} |')

    lines += ['','## 6. Exploratory screen (not confirmatory evidence)','',
              'Three repetitions each: 16/32/64 full slots at 17.26 GiB with LRU; then LRU/retain-full/reuse-aware policies at 9.20 GiB with eight full slots. Allocation order and policy order rotated. Selection maximized mean request throughput, with hit rate as a tie-breaker. Validation used new seeds (20261201–20261206), separate from allocation (20261001–20261003) and policy (20261101–20261103). The three-repetition screen does not meet the five-repetition requirement and is not pooled with validation.', '',
              '| Candidate | n | Hit rate (%) [95% CI] | Requests/s [95% CI] | Mean TTFT (ms) [95% CI] |','|---|---:|---:|---:|---:|']
    base=RESULTS/'allocation_policy_20260915_fixed'
    for phase in ('allocation','policy'):
        candidates={}
        for p in sorted((base/phase).glob('*/result.json')):
            candidates.setdefault(p.parent.name.rsplit('_r',1)[0],[]).append(read(p))
        for name,rs in candidates.items():
            vals=[cell([r['metrics'][m]*scale for r in rs]) for m,scale in [('token_cache_hit_rate',100),('request_throughput_rps',1),('mean_ttft_ms',1)]]
            lines.append(f'| {phase}/{name} | {len(rs)} | '+ ' | '.join(vals)+' |')
    lines += ['', 'All allocation candidates reached the ideal hit ceiling with no evictions; their close timings do not establish a universal optimum. Policy-screen intervals overlap, and no held-out direct policy-versus-policy comparison was conducted. Do not frame this as proof of a superior eviction algorithm.', '',
'''## 7. Validity, exclusions, and outstanding cleanup

- This compiler independently rechecks all 108 confirmatory runs: mode pairing, identical saved trace hashes, expected repetitions, all recorded validation flags, memory/staging limits, cold-cache state, 192 unique request indices, expected lengths, no retractions, actual baseline eviction, and hit/recomputation arithmetic. Hashes and counts are saved in `combined_audit.json`.
- The original `full_20260913` attempt is excluded: a staging peak exceeded its allowance by approximately 8.66 MiB. Revised studies use fresh runs and larger allowances inside the same ceilings. Failed launch attempts and all pilots are excluded. No failed run is silently pooled into these tables.
- These are repeated runs on one device, not independent hardware samples. Temperature, clocks, and unrelated resident GPU processes are not controlled by a multi-device randomized design; alternating order mitigates but does not eliminate drift. The saved environment includes a small unrelated GPU process. No claim of exclusive hardware reservation is made.
- Headroom is conservative and allocations are not exhaustively optimized. Do not infer a universal minimum useful memory budget from three points. B and C also differ in policy, dense slots, staging reserve, output length, and concurrency; only within-study matched comparisons isolate the stated configuration change.
- Rank 16 is lossy. These synthetic forced-output serving runs provide no answer-quality evidence. Use the paper's separate quality evaluation with its own provenance.

### Qwen GPQA, compression-on, seed 789: pending

This requested cleanup was **not run**: the matching four-seed paired evaluation and exact command were not located. Older Qwen GPQA Diamond CoT zero-shot artifacts exist under `/home/joshuaz/sssm-states/eval_results/gpqa-diamond/Qwen__Qwen3.5-4B/`, but their recorded seeds are 0/1234 and their API setups differ. They do not establish the requested seed-789 pairing. To finish comparably, supply the existing run directory or original command, including the GPQA variant, sample set, generation settings, compression implementation/rank, and which RNGs seed 789 controls. No guessed replacement result is included, and no claim that the four-seed comparison is complete is made.

## 8. Source artifacts and reproduction

- [A: original-allocation report](results/revised_20260914/full/report.md), [raw-data audit](results/revised_20260914/full/audit.json), [provenance](results/revised_20260914/full/provenance.json).
- [B: allocation/policy search and validation report](results/allocation_policy_20260915_fixed/report.md), [protocol](results/allocation_policy_20260915_fixed/protocol.json).
- [C: concurrency report](results/spark_concurrency_20260916/report.md), [protocol](results/spark_concurrency_20260916/protocol.json), [method](SPARK_CONCURRENCY.md).
- [Original memory/measurement definitions](results/README.md).
- Source per-run `result.json`, `requests.jsonl`, `command.json`, telemetry, and traces are adjacent to each report in the corresponding stage directories. Existing result directories must not be overwritten.

To regenerate this combined report and its checks from existing measurements (no GPU work):

```sh
.venv/bin/python benchmark/mamba_pressure/combine_reports.py
```

To reproduce a study, use `revised.py`, `search.py`, or `spark.py` with `--results <fresh-directory> --launch`, preserving the saved environment, source hashes, and protocol. These scripts run GPU experiments; regeneration of this document does not.
''']
    OUT.write_text('\n'.join(lines)+'\n')
    audit=dict(passed=True,confirmatory_runs=len(checked),requests=192*len(checked),pairs=sum(len(v[False]) for v in groups.values()),files=checked)
    (HERE/'combined_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(dict(report=str(OUT),passed=True,runs=len(checked),pairs=audit['pairs'])))


if __name__=='__main__':
    main()
