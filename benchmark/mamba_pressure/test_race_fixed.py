import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import race_fixed as run


class RaceFixedTests(unittest.TestCase):
    def test_schedule_failed_seed_first_and_complete_reporting(self):
        calls = []
        async def fake(args, config, rep, target, workload):
            calls.append((args, config, rep, workload))
            target.mkdir()
            run.save(target / 'result.json', dict(config=config, repetition=rep,
                     metrics={'test': 1.0}, validation={'ok': True}))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(run, 'one_run', fake), patch.object(run, 'trace', side_effect=lambda args, rep: [args.seed, rep]):
                asyncio.run(run.experiment(root))
            self.assertEqual(len(calls), 12)
            measured = [c for c in calls if not c[0].pilot]
            self.assertEqual(len(measured), 10)
            self.assertEqual(measured[0][1]['full_slots'], 16)
            self.assertEqual(measured[0][0].seed + measured[0][2], 20261002)
            for i in range(0, 10, 2):
                first, second = measured[i:i+2]
                self.assertEqual(first[3], second[3])
                self.assertEqual(first[1]['full_slots'], 16 if i // 2 % 2 == 0 else 32)
            self.assertTrue(all(not c[1]['pre_optimization'] for c in calls))
            summary = json.loads((root / 'summary.json').read_text())
            self.assertTrue(all(g['n'] == 5 for g in summary['groups'].values()))
            self.assertEqual(summary['paired'], {})
            self.assertEqual(json.loads((root / 'status.json').read_text())['state'], 'complete')


if __name__ == '__main__':
    unittest.main()
