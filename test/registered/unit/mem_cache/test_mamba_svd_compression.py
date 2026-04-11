import unittest

import torch

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler


def _create_pool_resources(device):
    """Create shared pool resources (small model for testing)."""
    num_layers = 4
    global_interval = 2
    full_attention_layer_ids = [
        i for i in range(global_interval - 1, num_layers, global_interval)
    ]
    mamba_layers = [i for i in range(num_layers) if i not in full_attention_layer_ids]

    with envs.SGLANG_MAMBA_SSM_DTYPE.override("float32"):
        shape = Mamba2StateShape.create(
            tp_world_size=1,
            intermediate_size=64,
            n_groups=2,
            num_heads=4,
            head_dim=16,
            state_size=16,
            conv_kernel=4,
        )
        mamba2_cache_params = Mamba2CacheParams(shape=shape, layers=mamba_layers)

    req_to_token_pool = HybridReqToTokenPool(
        size=100,
        mamba_size=200,
        mamba_spec_state_size=100,
        max_context_len=64,
        device=device,
        enable_memory_saver=False,
        cache_params=mamba2_cache_params,
        mamba_layer_ids=mamba_layers,
        enable_mamba_extra_buffer=False,
        speculative_num_draft_tokens=3,
    )
    pool = HybridLinearKVPool(
        size=128,
        dtype=torch.float32,
        page_size=1,
        head_num=2,
        head_dim=32,
        full_attention_layer_ids=full_attention_layer_ids,
        enable_kvcache_transpose=False,
        device=device,
        enable_memory_saver=False,
        mamba_pool=req_to_token_pool.mamba_pool,
    )
    allocator = TokenToKVPoolAllocator(
        size=128,
        dtype=torch.float32,
        device=device,
        kvcache=pool,
        need_sort=False,
    )
    return req_to_token_pool, allocator


def _make_tree(req_to_token_pool, allocator, svd_compression, svd_rank):
    set_global_server_args_for_scheduler(
        ServerArgs(
            model_path="dummy",
            page_size=1,
            mamba_svd_compression=svd_compression,
            mamba_svd_rank=svd_rank,
        )
    )
    params = CacheInitParams(
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=allocator,
        page_size=1,
        disable=False,
    )
    return MambaRadixCache(params=params)


def _make_req(req_to_token_pool):
    req = Req(
        rid=0,
        origin_input_text="",
        origin_input_ids=[],
        sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
    )
    req_to_token_pool.alloc([req])
    return req


def _write_known_state(mamba_pool, pool_idx):
    """Write a rank-2 state for predictable SVD roundtrip."""
    temporal = mamba_pool.mamba_cache.temporal  # [L, pool_size+1, H, D, S]
    L, _, H, D, S = temporal.shape
    idx = int(pool_idx.item()) if isinstance(pool_idx, torch.Tensor) else int(pool_idx)

    torch.manual_seed(42)
    a = torch.randn(L, H, D, 1, device=temporal.device, dtype=temporal.dtype)
    b = torch.randn(L, H, 1, S, device=temporal.device, dtype=temporal.dtype)
    c = torch.randn(L, H, D, 1, device=temporal.device, dtype=temporal.dtype)
    d = torch.randn(L, H, 1, S, device=temporal.device, dtype=temporal.dtype)
    state = a @ b + c @ d  # [L, H, D, S]
    temporal[:, idx, :, :, :] = state
    return state.clone()


class TestMambaSVDCompression(unittest.TestCase):
    """Tests for SVD compression of mamba temporal states in the prefix cache."""

    @classmethod
    def setUpClass(cls):
        from sglang.srt.utils import get_device

        cls.device = get_device()
        cls.req_to_token_pool, cls.allocator = _create_pool_resources(cls.device)

    def setUp(self):
        """Reset tracked requests between tests to avoid pool exhaustion."""
        if hasattr(self, "_tracked_reqs"):
            for req in self._tracked_reqs:
                self.req_to_token_pool.free_mamba_cache(req)
                self.req_to_token_pool.free(req)
        self._tracked_reqs = []

    def _fresh_tree(self, svd_compression=True, svd_rank=4):
        tree = _make_tree(
            self.req_to_token_pool,
            self.allocator,
            svd_compression=svd_compression,
            svd_rank=svd_rank,
        )
        return tree

    def _req(self):
        req = _make_req(self.req_to_token_pool)
        self._tracked_reqs.append(req)
        return req

    def test_compression_flag_set_after_insert(self):
        tree = self._fresh_tree(svd_compression=True)
        req = self._req()
        _write_known_state(self.req_to_token_pool.mamba_pool, req.mamba_pool_idx)

        tree.insert(
            InsertParams(
                key=RadixKey([1, 2, 3]),
                value=self.allocator.alloc(3),
                mamba_value=req.mamba_pool_idx.unsqueeze(0),
            )
        )

        node = tree.root_node.children[1]
        self.assertTrue(node.mamba_compressed, "Node should be compressed after insert")
        self.assertIsNotNone(node.mamba_value, "Pool slot should still be allocated")

    def test_compression_disabled(self):
        tree = self._fresh_tree(svd_compression=False)
        req = self._req()

        tree.insert(
            InsertParams(
                key=RadixKey([1, 2, 3]),
                value=self.allocator.alloc(3),
                mamba_value=req.mamba_pool_idx.unsqueeze(0),
            )
        )

        node = tree.root_node.children[1]
        self.assertFalse(node.mamba_compressed)

    def test_compress_decompress_roundtrip(self):
        tree = self._fresh_tree(svd_compression=True, svd_rank=4)
        mamba_pool = self.req_to_token_pool.mamba_pool

        req1 = self._req()
        original_state = _write_known_state(mamba_pool, req1.mamba_pool_idx)

        tree.insert(
            InsertParams(
                key=RadixKey([1, 2, 3]),
                value=self.allocator.alloc(3),
                mamba_value=req1.mamba_pool_idx.unsqueeze(0),
            )
        )

        req2 = self._req()
        tree.match_prefix(
            MatchPrefixParams(key=RadixKey([1, 2, 3]), req=req2, cow_mamba=True)
        )
        self.assertIsNotNone(req2.mamba_pool_idx)

        decompressed = mamba_pool.mamba_cache.temporal[:, req2.mamba_pool_idx]
        relative_error = torch.norm(original_state.float() - decompressed.float()) / (
            torch.norm(original_state.float()) + 1e-8
        )
        self.assertLess(
            relative_error.item(),
            0.1,
            f"Relative error too high: {relative_error:.6f}",
        )

    def test_conv_states_preserved(self):
        tree = self._fresh_tree(svd_compression=True)
        mamba_pool = self.req_to_token_pool.mamba_pool

        req1 = self._req()
        torch.manual_seed(99)
        for conv in mamba_pool.mamba_cache.conv:
            conv[:, req1.mamba_pool_idx] = torch.randn_like(
                conv[:, req1.mamba_pool_idx]
            )
        original_convs = [
            conv[:, req1.mamba_pool_idx].clone() for conv in mamba_pool.mamba_cache.conv
        ]

        tree.insert(
            InsertParams(
                key=RadixKey([10, 20, 30]),
                value=self.allocator.alloc(3),
                mamba_value=req1.mamba_pool_idx.unsqueeze(0),
            )
        )

        req2 = self._req()
        tree.match_prefix(
            MatchPrefixParams(key=RadixKey([10, 20, 30]), req=req2, cow_mamba=True)
        )

        for i, conv in enumerate(mamba_pool.mamba_cache.conv):
            self.assertTrue(
                torch.all(conv[:, req2.mamba_pool_idx] == original_convs[i]),
                f"Conv {i} not preserved through COW",
            )

    def test_uncompressed_cow_exact(self):
        tree = self._fresh_tree(svd_compression=False)
        mamba_pool = self.req_to_token_pool.mamba_pool

        req1 = self._req()
        _write_known_state(mamba_pool, req1.mamba_pool_idx)

        tree.insert(
            InsertParams(
                key=RadixKey([1, 2, 3]),
                value=self.allocator.alloc(3),
                mamba_value=req1.mamba_pool_idx.unsqueeze(0),
            )
        )

        req2 = self._req()
        tree.match_prefix(
            MatchPrefixParams(key=RadixKey([1, 2, 3]), req=req2, cow_mamba=True)
        )

        self.assertTrue(
            torch.all(
                mamba_pool.mamba_cache.temporal[:, req2.mamba_pool_idx]
                == mamba_pool.mamba_cache.temporal[:, req1.mamba_pool_idx]
            ),
            "Uncompressed COW should produce exact copy",
        )

    def test_tombstone_resets_flag(self):
        tree = self._fresh_tree(svd_compression=True)

        req1 = self._req()
        tree.insert(
            InsertParams(
                key=RadixKey([1, 2, 3]),
                value=self.allocator.alloc(3),
                mamba_value=req1.mamba_pool_idx.unsqueeze(0),
            )
        )
        req2 = self._req()
        tree.insert(
            InsertParams(
                key=RadixKey([1, 2, 3, 4, 5]),
                value=self.allocator.alloc(5),
                mamba_value=req2.mamba_pool_idx.unsqueeze(0),
            )
        )

        internal = tree.root_node.children[1]
        self.assertTrue(internal.mamba_compressed)

        tree.evict(EvictParams(num_tokens=0, mamba_num=1))

        if internal.mamba_value is None:
            self.assertFalse(
                internal.mamba_compressed, "Tombstoned node flag should be False"
            )

    def test_split_preserves_child_flag(self):
        tree = self._fresh_tree(svd_compression=True)

        req1 = self._req()
        tree.insert(
            InsertParams(
                key=RadixKey([1, 2, 3, 4, 5]),
                value=self.allocator.alloc(5),
                mamba_value=req1.mamba_pool_idx.unsqueeze(0),
            )
        )

        req2 = self._req()
        tree.insert(
            InsertParams(
                key=RadixKey([1, 2, 3, 6, 7]),
                value=self.allocator.alloc(5),
                mamba_value=req2.mamba_pool_idx.unsqueeze(0),
            )
        )

        parent = tree.root_node.children[1]
        self.assertIsNone(parent.mamba_value, "Split parent should be tombstone")
        self.assertFalse(parent.mamba_compressed)

        child_45 = parent.children[4]
        self.assertTrue(
            child_45.mamba_compressed, "Child should keep compressed flag after split"
        )

        tree.sanity_check()

    def test_rank_validation_rejects_oversized(self):
        # head_dim=16, state_size=16 → slot=256. rank=100 → 100*(16+1+16)=3300 > 256
        with self.assertRaises(AssertionError) as ctx:
            self._fresh_tree(svd_compression=True, svd_rank=100)
        self.assertIn("too large", str(ctx.exception))

    def test_sanity_check_integration(self):
        tree = self._fresh_tree(svd_compression=True)

        for token_ids in [[1, 2, 3], [1, 2, 3, 4, 5], [10, 20], [10, 20, 30, 40]]:
            req = self._req()
            tree.insert(
                InsertParams(
                    key=RadixKey(token_ids),
                    value=self.allocator.alloc(len(token_ids)),
                    mamba_value=req.mamba_pool_idx.unsqueeze(0),
                )
            )

        for token_ids in [[1, 2, 3], [10, 20, 30, 40]]:
            req = self._req()
            tree.match_prefix(
                MatchPrefixParams(key=RadixKey(token_ids), req=req, cow_mamba=True)
            )

        tree.evict(EvictParams(num_tokens=0, mamba_num=1))
        tree.sanity_check()


if __name__ == "__main__":
    unittest.main()
