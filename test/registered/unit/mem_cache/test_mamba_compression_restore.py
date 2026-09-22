import unittest
from types import SimpleNamespace as NS

import torch
from torch.profiler import ProfilerActivity, profile

from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache, TreeNode
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="stage-b-test-1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
class TestMambaCompressionRestore(unittest.TestCase):
    def test_restores_do_not_read_device_scalars_on_the_host(self):
        for compressed in (False, True):
            with self.subTest(compressed=compressed):
                pool = object.__new__(MambaPool)
                pool.mamba_cache = NS(
                    temporal=torch.zeros(2, 3, 2, 4, 4, device="cuda"),
                    conv=[torch.zeros(2, 3, 4, 3, device="cuda")],
                )
                tree = object.__new__(MambaRadixCache)
                tree.enable_svd_compression = compressed
                tree.req_to_token_pool = NS(mamba_pool=pool)
                tree.svd_rank = 1
                src = TreeNode()
                src.mamba_value = torch.tensor([1], device="cuda")
                src.mamba_compressed = compressed
                src.compressed_slot = 0 if compressed else None
                dst = torch.tensor([2], device="cuda")
                expected = 3 * torch.arange(4, device="cuda", dtype=torch.float32)
                expected = expected.expand(2, 2, 4, 4)
                if compressed:
                    tree.compressed_temporal = torch.cat(
                        [torch.ones(1, 2, 2, 4, device="cuda"),
                         torch.full((1, 2, 2, 1), 3., device="cuda"),
                         torch.arange(4, device="cuda").expand(1, 2, 2, 4)],
                        dim=-1,
                    )
                    tree.compressed_conv = [torch.full((1, 2, 4, 3), 7., device="cuda")]
                else:
                    pool.mamba_cache.temporal[:, 1] = expected
                    pool.mamba_cache.conv[0][:, 1] = 7
                torch.cuda.synchronize()
                with profile(activities=[ProfilerActivity.CPU]) as p:
                    tree._decompress_from_pool(src, dst)
                ops = {event.key for event in p.key_averages()}
                self.assertNotIn("aten::item", ops)
                self.assertNotIn("aten::_local_scalar_dense", ops)
                torch.testing.assert_close(pool.mamba_cache.temporal[:, 2], expected)
                self.assertTrue((pool.mamba_cache.conv[0][:, 2] == 7).all())


if __name__ == "__main__":
    unittest.main()
