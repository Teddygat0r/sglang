import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.mamba_allocation import (
    compressed_state_bytes,
    explicit_mamba_bytes,
    validate_explicit_allocation,
)
from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache
from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    ModelRunnerKVCacheMixin,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="stage-b-test-1-gpu-small")


def geometry():
    return NS(
        shape=NS(temporal=(2, 8, 8), conv=[(4, 3)]),
        dtype=NS(conv=torch.float32, temporal=torch.float32),
        layers=[0, 1],
        mamba_cache_per_req=1120,
    )


def runner(**kwargs):
    return NS(
        mambaish_config=NS(mamba2_cache_params=geometry()),
        server_args=ServerArgs(
            model_path="dummy",
            mamba_svd_compression=True,
            mamba_svd_rank=2,
            **kwargs,
        ),
        spec_algorithm=NS(is_none=lambda: True),
    )


class TestMambaCompressionAllocation(unittest.TestCase):
    def test_profiler_covers_real_pools_for_all_sizing_modes(self):
        budget = 1 / 1024  # 1 MiB, expressed in GiB as the profiler expects.
        for full, compressed in ((None, None), (17, None), (17, 7)):
            with self.subTest(full=full, compressed=compressed):
                r = runner(max_mamba_cache_size=full, mamba_svd_cache_size=compressed)
                remaining = ModelRunnerKVCacheMixin.handle_max_mamba_cache(r, budget)
                args = r.server_args
                slots = args.max_mamba_cache_size
                temporal = torch.empty(2, slots + 1, 2, 8, 8)
                conv = torch.empty(2, slots + 1, 4, 3)
                tree = object.__new__(MambaRadixCache)
                tree.device = "cpu"
                tree.svd_rank = 2
                tree.req_to_token_pool = NS(
                    mamba_pool=NS(
                        size=slots, mamba_cache=NS(temporal=temporal, conv=[conv])
                    )
                )
                with (
                    patch(
                        "sglang.srt.mem_cache.mamba_radix_cache.get_global_server_args",
                        return_value=args,
                    ),
                    patch("sglang.srt.mem_cache.mamba_radix_cache.threading.Thread"),
                ):
                    tree._init_compression_state()
                expected_slots = (
                    compressed if compressed is not None else max(1, slots // 2)
                )
                self.assertEqual(tree.compressed_temporal.shape[0], expected_slots)
                tensors = [
                    temporal,
                    conv,
                    tree.compressed_temporal,
                    *tree.compressed_conv,
                    tree._compression_staging.buffer,
                ]
                persistent = sum(t.numel() * t.element_size() for t in tensors)
                # Up to eight results may each pin their two-result batch.
                retained_results = 8 * 2 * (2 * 2 * (8 + 1 + 8) * 2 * 4)
                charged = round((budget - remaining) * 2**30)
                self.assertEqual(charged, persistent + retained_results)
                self.assertGreater(remaining, 0)
                if full is None:
                    staging = tree._compression_staging.buffer.nbytes + retained_results
                    target = (budget * 2**30 - staging) * 0.9 / 1.9
                    self.assertLessEqual(
                        (slots + 1) * 1120 + (slots // 2) * 640, target
                    )
                    self.assertGreater(
                        (slots + 2) * 1120 + ((slots + 1) // 2) * 640, target
                    )

    def test_insufficient_memory_rejected_before_allocation(self):
        for full in (None, 100):
            with self.subTest(full=full), self.assertRaisesRegex(ValueError, "Mamba"):
                ModelRunnerKVCacheMixin.handle_max_mamba_cache(
                    runner(max_mamba_cache_size=full), 4096 / 2**30
                )

    def test_explicit_reserve_is_a_minimum(self):
        r = runner(
            max_mamba_cache_size=17,
            mamba_svd_cache_size=7,
            mamba_svd_staging_reserve_bytes=50000,
        )
        remaining = ModelRunnerKVCacheMixin.handle_max_mamba_cache(r, 1)
        self.assertEqual(round((1 - remaining) * 2**30), 18 * 1120 + 7 * 640 + 50000)

    def test_compression_disabled_keeps_existing_budget(self):
        r = runner(max_mamba_cache_size=17)
        r.server_args.mamba_svd_compression = False
        remaining = ModelRunnerKVCacheMixin.handle_max_mamba_cache(r, 1)
        self.assertEqual(round((1 - remaining) * 2**30), 17 * 1120)


def arguments(**changes):
    return NS(
        **dict(
            dict(
                mamba_svd_cache_size=298,
                mamba_svd_staging_reserve_bytes=824180736,
                mamba_svd_compression=True,
                max_mamba_cache_size=32,
                mamba_svd_rank=16,
                dp_size=1,
                pp_size=1,
                speculative_algorithm=None,
                disable_radix_cache=False,
            ),
            **changes,
        )
    )


def production_geometry():
    return NS(
        shape=NS(temporal=(32, 128, 128), conv=[(3, 8192)]),
        dtype=NS(conv=torch.bfloat16, temporal=torch.float32),
        layers=list(range(24)),
        mamba_cache_per_req=51511296,
    )


class NativeAllocationTests(unittest.TestCase):
    def test_geometry_and_sentinel(self):
        params = production_geometry()
        self.assertEqual(compressed_state_bytes(params, 16), 13811712)
        self.assertEqual(
            explicit_mamba_bytes(params, 32, 298, 16, 824180736),
            33 * 51511296 + 298 * 13811712 + 824180736,
        )
        for rank in (0, -1, 128):
            with self.assertRaises(ValueError):
                compressed_state_bytes(params, rank)

    def test_argument_validation(self):
        validate_explicit_allocation(arguments())
        validate_explicit_allocation(NS())
        for change in (
            dict(mamba_svd_cache_size=0),
            dict(max_mamba_cache_size=None),
            dict(mamba_svd_compression=False),
            dict(mamba_svd_staging_reserve_bytes=-1),
            dict(mamba_svd_cache_size=None),
            dict(dp_size=2),
            dict(pp_size=2),
            dict(speculative_algorithm="EAGLE"),
            dict(disable_radix_cache=True),
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_explicit_allocation(arguments(**change))

    def test_profiler_charges_all_storage(self):
        from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
            ModelRunnerKVCacheMixin,
        )

        runner = NS(
            mambaish_config=NS(mamba2_cache_params=production_geometry()),
            server_args=arguments(),
        )
        expected = explicit_mamba_bytes(production_geometry(), 32, 298, 16, 824180736)
        actual = ModelRunnerKVCacheMixin.handle_max_mamba_cache(runner, 20)
        self.assertEqual(actual, 20 - expected / 2**30)
        with self.assertRaises(ValueError):
            ModelRunnerKVCacheMixin.handle_max_mamba_cache(runner, 1)

    def test_real_pool_initializer_uses_explicit_or_legacy_count(self):
        from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache

        for count, expected in ((7, 7), (None, 2)):
            tree = object.__new__(MambaRadixCache)
            tree.req_to_token_pool = NS(
                mamba_pool=NS(
                    size=4,
                    mamba_cache=NS(
                        temporal=torch.empty(2, 5, 3, 8, 8),
                        conv=[torch.empty(2, 5, 4, 3)],
                    ),
                )
            )
            tree.device = torch.device("cpu")
            tree.svd_rank = 2
            with (
                patch(
                    "sglang.srt.mem_cache.mamba_radix_cache.get_global_server_args",
                    return_value=NS(mamba_svd_cache_size=count),
                ),
                patch("sglang.srt.mem_cache.mamba_radix_cache.threading.Thread"),
            ):
                tree._init_compression_state()
            self.assertEqual(tree.compressed_temporal.shape[0], expected)
            self.assertEqual(len(tree._compressed_free_slots), expected)


if __name__ == "__main__":
    unittest.main()
