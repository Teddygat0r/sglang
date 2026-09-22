import unittest

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

    def test_single_rank_compression_is_accepted(self):
        args = ServerArgs(model_path="dummy", mamba_svd_compression=True)
        self.assertEqual(args.tp_size, 1)

    def test_tensor_parallelism_without_compression_is_unchanged(self):
        args = ServerArgs(model_path="dummy", tp_size=2)
        self.assertEqual(args.tp_size, 2)


if __name__ == "__main__":
    unittest.main()
