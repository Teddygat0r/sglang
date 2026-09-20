import asyncio
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import batch_sweep as sweep
import spark_run


class BatchTests(unittest.TestCase):
    def test_schedule_and_report(self):
        calls = []
        async def fake(args, cfg, rep, target, workload):
            calls.append((args.pilot, cfg, rep, workload))
            target.mkdir(parents=True)
            sweep.save(target / 'result.json', dict(config=cfg, repetition=rep,
                metrics={'cap': cfg['svd_worker_batch']}, validation={'ok': True}))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(sweep, 'one_run', fake):
                asyncio.run(sweep.experiment(root))
            self.assertEqual(len(calls), 18)
            for rep in range(5):
                rows = [r for r in calls if not r[0] and r[2] == rep]
                self.assertEqual([r[1]['svd_worker_batch'] for r in rows], [1,2,8] if rep % 2 else [8,2,1])
                self.assertTrue(all(r[3] == rows[0][3] for r in rows))
            summary = json.loads((root / 'summary.json').read_text())
            self.assertEqual(summary['paired_vs_batch8']['batch1']['cap']['ci95'], [-7., -7.])

    def test_launcher_cap_and_no_gate(self):
        base = next(c for c in sweep.configs() if c['full_slots'] == 32)
        args = SimpleNamespace(port=31037, concurrency=4, seed=1)
        with tempfile.TemporaryDirectory() as temp:
            for cap in (8,2,1):
                def capture(command, **kw):
                    self.assertEqual(Path(command[1]).name, 'server.py')
                    self.assertEqual(command[command.index('--mamba-svd-worker-batch') + 1], str(cap))
                    self.assertNotIn('PRESSURE_PREFILL_GATE', kw['env'])
                    self.assertNotIn('PRESSURE_TAIL_PROFILE', kw['env'])
                    kw['stdout'].close()
                    raise RuntimeError('captured launcher')
                with patch.dict(os.environ, PRESSURE_PREFILL_GATE='strict'), patch.object(spark_run.subprocess, 'Popen', side_effect=capture):
                    with self.assertRaisesRegex(RuntimeError, 'captured launcher'):
                        asyncio.run(spark_run.one_run(args, {**base, 'svd_worker_batch': cap}, 0, Path(temp)/str(cap), []))
            for cfg in ({**base, 'svd_worker_batch': 0}, {**base, 'prefill_gate': 'sync_control'}):
                with self.assertRaises(ValueError):
                    asyncio.run(spark_run.one_run(args, cfg, 0, Path(temp)/'invalid', []))


if __name__ == '__main__':
    unittest.main()
