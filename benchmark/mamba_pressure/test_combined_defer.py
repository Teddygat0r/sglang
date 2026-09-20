import asyncio
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import combined_defer_sweep
import defer_sweep
import spark_run


class CombinedTests(unittest.TestCase):
    def test_schedule_and_report(self):
        calls = []
        async def fake(args, cfg, rep, target, workload):
            calls.append((args, cfg, rep, workload))
            target.mkdir(parents=True)
            spark_run.save(target/'result.json', dict(config=cfg, repetition=rep,
                metrics={'cap': cfg['svd_worker_batch']}, validation={'ok':True}))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(defer_sweep, 'one_run', fake):
                asyncio.run(combined_defer_sweep.experiment(root))
            self.assertEqual(len(calls), 18)
            self.assertTrue(all(r[1]['defer_prefill'] for r in calls))
            for rep in range(5):
                rows = [r for r in calls if not r[0].pilot and r[2] == rep]
                self.assertEqual([r[1]['svd_worker_batch'] for r in rows], [1,2,8] if rep%2 else [8,2,1])
                self.assertTrue(all(r[3] == rows[0][3] for r in rows))
                self.assertTrue(all(r[0].seed == 20261130 for r in rows))
            summary = json.loads((root/'summary.json').read_text())
            self.assertEqual(summary['paired_vs_defer_batch8']['defer_batch1']['cap']['ci95'], [-7.,-7.])
            self.assertEqual(json.loads((root/'status.json').read_text())['completed'],18)

    def test_actual_run_launcher_selects_deferral_and_cap(self):
        base = next(c for c in defer_sweep.configs() if c['full_slots']==32)
        args = SimpleNamespace(port=31037, concurrency=4, seed=1)
        with tempfile.TemporaryDirectory() as temp:
            for cap in (8,2,1):
                def capture(command, **kw):
                    self.assertEqual(Path(command[1]).name,'defer_server.py')
                    self.assertEqual(command[command.index('--mamba-svd-worker-batch')+1],str(cap))
                    self.assertNotIn('PRESSURE_PREFILL_GATE',kw['env'])
                    self.assertNotIn('PRESSURE_TAIL_PROFILE',kw['env'])
                    kw['stdout'].close()
                    raise RuntimeError('captured launcher')
                with patch.object(spark_run.subprocess,'Popen',side_effect=capture):
                    with self.assertRaisesRegex(RuntimeError,'captured launcher'):
                        asyncio.run(spark_run.one_run(args,{**base,'defer_prefill':True,'svd_worker_batch':cap},0,Path(temp)/str(cap),[]))


if __name__ == '__main__':
    unittest.main()
