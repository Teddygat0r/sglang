import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from argparse import Namespace
import spark
import spark_run


class SparkTests(unittest.TestCase):
    def test_budgets(self):
        geometry=json.loads((spark.HERE/'results/full_20260913/discovery/initial.json').read_text())
        for c in spark.configs():
            charged=geometry['kv_pool_bytes']+(c['full_slots']+1)*geometry['full_state_bytes']+c['compressed_slots']*c['compressed_state_bytes']+c['staging_reserve_bytes']
            self.assertLessEqual(charged,c['budget_bytes'])
            self.assertGreaterEqual(c['full_slots'],c['concurrency']*8)

    def test_trace(self):
        args=Namespace(seed=20260920,groups=64,rounds=3,prefix=2048)
        rows=spark_run.trace(args,0)
        self.assertEqual(len(rows),192)
        for round_index in range(3):
            batch=[r for r in rows if r['round']==round_index]
            self.assertEqual(len({r['group'] for r in batch}),64)
        self.assertEqual(rows,spark_run.trace(args,0))

    def test_supervisor_all_phases(self):
        calls=[]
        async def fake(args,c,rep,target,workload):
            calls.append((args.pilot,c,rep))
            target.mkdir()
            result=dict(config=c,repetition=rep,metrics={'request_throughput_rps':1.0},validation={'ok':True})
            spark.save(target/'result.json',result)
            return result
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with patch.object(spark,'one_run',fake),patch.object(spark,'trace',return_value=[]):
                asyncio.run(spark.experiment(root))
            self.assertEqual(json.loads((root/'status.json').read_text())['state'],'complete')
            self.assertEqual(len(calls),64)
            measured=[c for c in calls if not c[0]]
            self.assertEqual(len(measured),60)
            for rep in range(5):
                order=[c[1] for c in measured if c[2]==rep]
                self.assertEqual(order[0]['compression'],bool(rep%2))


if __name__=='__main__':
    unittest.main()
