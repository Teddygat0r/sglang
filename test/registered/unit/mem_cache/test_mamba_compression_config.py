import unittest

import torch

from sglang.srt.mem_cache.rsvd_eigh import randomized_svd_eigh
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class TestMambaCompressionConfig(unittest.TestCase):
    def test_compression_accepts_standard_tensor_parallelism(self):
        for slots in (None, 8):
            with self.subTest(compressed_slots=slots):
                args = ServerArgs(
                    model_path="dummy",
                    tp_size=2,
                    mamba_svd_compression=True,
                    max_mamba_cache_size=16,
                    mamba_svd_cache_size=slots,
                )
                self.assertEqual(args.tp_size, 2)

    def test_tp_rejects_scheduler_modes_without_collective_drain_support(self):
        for option in (
            {"pp_size": 2},
            {"dp_size": 2},
            {"speculative_algorithm": "EAGLE"},
            {"disaggregation_mode": "prefill"},
            {"enable_pdmux": True},
            {"dllm_algorithm": "test"},
            {"enable_hierarchical_cache": True},
        ):
            with self.subTest(option=option):
                with self.assertRaisesRegex(ValueError, "standard autoregressive"):
                    ServerArgs(
                        model_path="dummy",
                        tp_size=2,
                        mamba_svd_compression=True,
                        **option,
                    )

    def test_staging_capacity_must_be_positive(self):
        for capacity in (0, -1):
            with self.subTest(capacity=capacity):
                with self.assertRaisesRegex(ValueError, "max_pending must be positive"):
                    ServerArgs(
                        model_path="dummy",
                        mamba_svd_compression=True,
                        mamba_svd_max_pending=capacity,
                    )

    def test_negative_svd_parameters_rejected_for_all_pool_sizing_modes(self):
        for slots in (None, 8):
            for name in ("mamba_svd_niter", "mamba_svd_oversample"):
                with self.subTest(compressed_slots=slots, parameter=name):
                    with self.assertRaisesRegex(
                        ValueError, f"{name} must be non-negative"
                    ):
                        ServerArgs(
                            model_path="dummy",
                            mamba_svd_compression=True,
                            mamba_svd_rank=4,
                            max_mamba_cache_size=16,
                            mamba_svd_cache_size=slots,
                            **{name: -1},
                        )

    def test_zero_svd_iterations_and_oversampling_are_accepted(self):
        for slots in (None, 8):
            with self.subTest(compressed_slots=slots):
                args = ServerArgs(
                    model_path="dummy",
                    mamba_svd_compression=True,
                    max_mamba_cache_size=16,
                    mamba_svd_cache_size=slots,
                    mamba_svd_niter=0,
                    mamba_svd_oversample=0,
                )
                self.assertEqual(args.mamba_svd_niter, 0)
                self.assertEqual(args.mamba_svd_oversample, 0)

    def test_single_rank_compression_is_accepted(self):
        args = ServerArgs(model_path="dummy", mamba_svd_compression=True)
        self.assertEqual(args.tp_size, 1)

    def test_tensor_parallelism_without_compression_is_unchanged(self):
        args = ServerArgs(model_path="dummy", tp_size=2)
        self.assertEqual(args.tp_size, 2)


class TestRandomizedSVDParameters(unittest.TestCase):
    def test_negative_parameters_fail_before_advancing_rng(self):
        matrix = torch.eye(16).expand(2, 2, 16, 16)
        for name in ("n_iter", "oversample"):
            with self.subTest(parameter=name):
                generator = torch.Generator().manual_seed(123)
                before = generator.get_state().clone()
                with self.assertRaisesRegex(ValueError, f"{name} must be non-negative"):
                    randomized_svd_eigh(
                        matrix, rank=4, generator=generator, **{name: -1}
                    )
                self.assertTrue(torch.equal(before, generator.get_state()))

    def test_zero_parameters_return_requested_rank(self):
        # A rank-four input should remain reconstructible without extra sketch
        # columns or power iterations; zero is a valid performance setting.
        matrix = torch.diag(torch.tensor([4.0, 3.0, 2.0, 1.0] + [0.0] * 12))
        matrix = matrix.expand(2, 2, 16, 16)
        u, singular_values, vh = randomized_svd_eigh(
            matrix,
            rank=4,
            n_iter=0,
            oversample=0,
            generator=torch.Generator().manual_seed(123),
        )
        self.assertEqual(u.shape, (2, 2, 16, 4))
        self.assertEqual(singular_values.shape, (2, 2, 4))
        self.assertEqual(vh.shape, (2, 2, 4, 16))
        torch.testing.assert_close(
            (u * singular_values.unsqueeze(-2)) @ vh, matrix, rtol=1e-4, atol=1e-5
        )


if __name__ == "__main__":
    unittest.main()
