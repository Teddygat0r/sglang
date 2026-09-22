import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import tail_sweep as sweep


class TailTests(unittest.TestCase):
    def test_overlap_union_and_clip(self):
        spans = [{'start_ns': a, 'end_ns': b} for a, b in [(0, 10), (5, 20), (25, 40)]]
        self.assertEqual(sweep.overlap_ms(spans, 7, 30), 18 / 1e6)

    def test_schedule_and_analysis(self):
        calls = []
        async def fake(args, cfg, rep, target, workload):
            calls.append((args, cfg, workload))
            target.mkdir(parents=True)
            sweep.save(target / 'result.json', {'metrics': dict(mean_ttft_ms=1, p95_ttft_ms=1, p99_ttft_ms=1)})
            row = dict(meta_info={'id': 'abc'}, index=0, round=0, cached_tokens=0,
                       ttft_ms=1, start_ns=0, first_ns=1000000)
            (target / 'requests.jsonl').write_text(json.dumps(row) + '\n')
            span = dict(kind='forward', rids=['abc'], start_ns=1, end_ns=900000, gpu_ms=.8)
            (target / 'spans-1.jsonl').write_text(json.dumps(span) + '\n')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(sweep, 'one_run', fake):
                asyncio.run(sweep.experiment(root))
            self.assertEqual(len(calls), 6)
            full = [c for c in calls if not c[0].pilot]
            self.assertEqual(len(full), 4)
            self.assertTrue(all(c[2] == full[0][2] for c in full))
            self.assertEqual([c[1]['tail_profile'] for c in full], [False, True, True, False])
            self.assertEqual(json.loads((root / 'status.json').read_text())['state'], 'complete')


if __name__ == '__main__':
    unittest.main()
