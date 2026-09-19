import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import individual_sweep as sweep
from individual_variant import VARIANTS


class IndividualTests(unittest.TestCase):
    def test_entrypoints_and_numerical_checks(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                env = dict(os.environ, PRESSURE_INDIVIDUAL_VARIANT=variant, PRESSURE_PRE_OPTIMIZATION='0',
                           PYTHONPATH=str(sweep.ROOT / 'python') + os.pathsep + str(sweep.HERE))
                result = subprocess.run([sys.executable, str(sweep.HERE / 'check_individual_variant.py')],
                                        cwd=sweep.ROOT, env=env, capture_output=True, text=True, timeout=90)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_complete_schedule_pairing_and_report(self):
        calls = []
        async def fake(args, config, rep, target, workload):
            calls.append((args, config, rep, workload))
            target.mkdir()
            sweep.save(target / 'result.json', dict(config=config, repetition=rep,
                       metrics={'test': float(config['ablation'] != 'baseline')}))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(sweep, 'one_run', fake), patch.object(sweep, 'trace', side_effect=lambda args, rep: [args.seed, rep]):
                asyncio.run(sweep.experiment(root))
            measured = [c for c in calls if not c[0].pilot]
            self.assertEqual(len(calls), 24)
            self.assertEqual(len(measured), 20)
            for rep in range(5):
                rows = [c for c in measured if c[2] == rep]
                self.assertEqual([c[1]['ablation'] for c in rows], list(VARIANTS if rep % 2 == 0 else VARIANTS[::-1]))
                self.assertTrue(all(c[3] == rows[0][3] for c in rows))
                self.assertTrue(all(c[1]['full_slots'] == 16 for c in rows))
            result = json.loads((root / 'summary.json').read_text())
            self.assertEqual(set(result['paired']), set(VARIANTS) - {'baseline'})
            self.assertTrue(all(x['test']['ci95'] == [1.0, 1.0] for x in result['paired'].values()))
            self.assertEqual(json.loads((root / 'status.json').read_text())['state'], 'complete')


if __name__ == '__main__':
    unittest.main()
