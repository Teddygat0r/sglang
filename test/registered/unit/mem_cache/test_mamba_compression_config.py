import unittest

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class TestMambaCompressionConfig(unittest.TestCase):
    def test_compression_rejects_tensor_parallelism_before_model_loading(self):
        for slots in (None, 8):
            with self.subTest(compressed_slots=slots):
                with self.assertRaisesRegex(ValueError, "requires --tp-size 1"):
                    ServerArgs(
                        model_path="dummy",
                        tp_size=2,
                        mamba_svd_compression=True,
                        max_mamba_cache_size=16,
                        mamba_svd_cache_size=slots,
                    )

    def test_single_rank_compression_is_accepted(self):
        args = ServerArgs(model_path="dummy", mamba_svd_compression=True)
        self.assertEqual(args.tp_size, 1)

    def test_tensor_parallelism_without_compression_is_unchanged(self):
        args = ServerArgs(model_path="dummy", tp_size=2)
        self.assertEqual(args.tp_size, 2)


if __name__ == "__main__":
    unittest.main()
