"""CPU checks for production allocation and the complete sweep supervisor."""

import asyncio
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

import allocation_native as sweep
from sglang.srt.mem_cache.mamba_allocation import (
    compressed_state_bytes, explicit_mamba_bytes, validate_explicit_allocation,
)


def arguments(**changes):
    return NS(**dict(dict(mamba_svd_cache_size=298, mamba_svd_staging_reserve_bytes=824180736,
                         mamba_svd_compression=True, max_mamba_cache_size=32, mamba_svd_rank=16,
                         dp_size=1, pp_size=1, speculative_algorithm=None, disable_radix_cache=False), **changes))


def geometry():
    return NS(shape=NS(temporal=(32, 128, 128), conv=[(3, 8192)]),
              dtype=NS(conv=torch.bfloat16, temporal=torch.float32), layers=list(range(24)),
              mamba_cache_per_req=51511296)


class NativeAllocationTests(unittest.TestCase):
    def test_geometry_and_sentinel(self):
        params = geometry()
        self.assertEqual(compressed_state_bytes(params, 16), 13811712)
        self.assertEqual(explicit_mamba_bytes(params, 32, 298, 16, 824180736),
                         33 * 51511296 + 298 * 13811712 + 824180736)
        for rank in (0, -1, 128):
            with self.assertRaises(ValueError):
                compressed_state_bytes(params, rank)

    def test_argument_validation(self):
        validate_explicit_allocation(arguments())
        validate_explicit_allocation(NS())
        for change in (dict(mamba_svd_cache_size=0), dict(max_mamba_cache_size=None),
                       dict(mamba_svd_compression=False), dict(mamba_svd_staging_reserve_bytes=-1),
                       dict(mamba_svd_cache_size=None), dict(dp_size=2), dict(pp_size=2),
                       dict(speculative_algorithm="EAGLE"), dict(disable_radix_cache=True)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_explicit_allocation(arguments(**change))

    def test_profiler_charges_all_storage(self):
        from sglang.srt.model_executor.model_runner_kv_cache_mixin import ModelRunnerKVCacheMixin
        runner = NS(mambaish_config=NS(mamba2_cache_params=geometry()), server_args=arguments())
        expected = explicit_mamba_bytes(geometry(), 32, 298, 16, 824180736)
        actual = ModelRunnerKVCacheMixin.handle_max_mamba_cache(runner, 20)
        self.assertEqual(actual, 20 - expected / 2**30)
        with self.assertRaises(ValueError):
            ModelRunnerKVCacheMixin.handle_max_mamba_cache(runner, 1)

    def test_real_pool_initializer_uses_explicit_or_legacy_count(self):
        from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache
        for count, expected in ((7, 7), (None, 2)):
            tree = object.__new__(MambaRadixCache)
            tree.req_to_token_pool = NS(mamba_pool=NS(size=4, mamba_cache=NS(
                temporal=torch.empty(2, 5, 3, 8, 8), conv=[torch.empty(2, 5, 4, 3)])))
            tree.device = torch.device("cpu")
            tree.svd_rank = 2
            with patch("sglang.srt.mem_cache.mamba_radix_cache.get_global_server_args", return_value=NS(mamba_svd_cache_size=count)), patch("sglang.srt.mem_cache.mamba_radix_cache.threading.Thread"):
                tree._init_compression_state()
            self.assertEqual(tree.compressed_temporal.shape[0], expected)
            self.assertEqual(len(tree._compressed_free_slots), expected)

    def test_budget_and_order(self):
        g = geometry()
        for c in sweep.configs():
            charged = 8589967360 + (c['full_slots'] + 1) * g.mamba_cache_per_req
            charged += c['compressed_slots'] * compressed_state_bytes(g, 16) + c['staging_reserve_bytes']
            self.assertLessEqual(charged, c['budget_bytes'])
            self.assertLess(c['budget_bytes'] - charged, compressed_state_bytes(g, 16))
        calls = []
        async def fake(args, config, rep, target, workload):
            calls.append((args.pilot, config['label'], rep, workload))
            target.mkdir()
            sweep.save(target / 'result.json', dict(config=config, repetition=rep, metrics={'test': 1.0}))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(sweep, 'one_run', fake):
                asyncio.run(sweep.experiment(root))
            self.assertEqual(json.loads((root / 'status.json').read_text())['state'], 'complete')
            self.assertEqual(len(calls), 24)
            for rep in range(5):
                measured = [r for r in calls if not r[0] and r[2] == rep]
                expected = ['off', 'on_f16', 'on_f32', 'on_f64']
                self.assertEqual([r[1] for r in measured], expected if rep % 2 == 0 else expected[::-1])
                self.assertTrue(all(r[3] == measured[0][3] for r in measured))
            report = json.loads((root / 'summary.json').read_text())
            self.assertEqual(report['paired']['on_f16']['test']['ci95'], [0.0, 0.0])


if __name__ == '__main__':
    unittest.main()
