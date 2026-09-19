import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import hotpath_sweep as sweep


class HotpathSweepTests(unittest.TestCase):
    def test_server_entrypoint_selects_baseline_before_instrumentation(self):
        for old in (False, True):
            code = '''
import inspect
import server
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache
import os
old = os.environ['PRESSURE_PRE_OPTIMIZATION'] == '1'
assert 'zero_initialize' not in inspect.signature(MambaPool.alloc).parameters
assert (MambaRadixCache._match_post_processor.__code__.co_filename == '<pre-optimization-baseline>') == old
print('entrypoint passed')
'''
            env = dict(os.environ, PRESSURE_PRE_OPTIMIZATION=str(int(old)),
                       PYTHONPATH=str(sweep.ROOT / 'python') + os.pathsep + str(sweep.HERE))
            completed = subprocess.run([sys.executable, '-c', code], cwd=sweep.ROOT, env=env,
                                       capture_output=True, text=True, timeout=60)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn('entrypoint passed', completed.stdout)

    def test_full_schedule_pairing_and_report(self):
        calls = []
        async def fake(args, config, rep, target, workload):
            calls.append((args.pilot, config, rep, workload))
            target.mkdir()
            sweep.save(target / 'result.json', dict(config=config, repetition=rep, metrics={'test': float(not config['pre_optimization'])}))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(sweep, 'one_run', fake), patch.object(sweep, 'trace', side_effect=lambda args, rep: [dict(rep=rep, seed=args.seed)]):
                asyncio.run(sweep.experiment(root))
            self.assertEqual(len(calls), 24)
            self.assertEqual(json.loads((root / 'status.json').read_text())['state'], 'complete')
            for rep in range(5):
                rows = [c for c in calls if not c[0] and c[2] == rep]
                expected = ['f16_before', 'f16_after', 'f32_before', 'f32_after']
                self.assertEqual([r[1]['label'] for r in rows], expected if rep % 2 == 0 else expected[::-1])
                self.assertTrue(all(r[3] == rows[0][3] for r in rows))
            summary = json.loads((root / 'summary.json').read_text())
            for slots in (16, 32):
                self.assertEqual(summary['paired'][f'f{slots}_after']['test']['ci95'], [1.0, 1.0])


if __name__ == '__main__':
    unittest.main()
