import time
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
from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache, TreeNode
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

    def _drain_sync(self, tree, timeout_s: float = 5.0):
        """Wait for the async SVD worker to drain its work queue, then commit completions
        on the main thread. No-op if compression is disabled."""
        if not getattr(tree, "enable_svd_compression", False):
            return
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if (
                tree._compression_queue.empty()
                and not tree._pending_compression
                and tree._compression_done_queue.empty()
            ):
                return
            if not tree._compression_done_queue.empty():
                tree.drain_compression_completions()
            else:
                time.sleep(0.01)
        tree.drain_compression_completions()

    def _req(self):
        req = _make_req(self.req_to_token_pool)
        self._tracked_reqs.append(req)
        return req

    def _stop_compression_worker(self, tree):
        tree._compression_stop_event.set()
        tree._compression_thread.join(timeout=2.0)
        self.assertFalse(tree._compression_thread.is_alive())

    def test_compression_queue_retains_more_than_worker_batch(self):
        tree = self._fresh_tree(svd_compression=True)
        self._stop_compression_worker(tree)

        for i in range(tree.svd_worker_batch + 1):
            req = self._req()
            tree.insert(
                InsertParams(
                    key=RadixKey([100 + i]),
                    value=self.allocator.alloc(1),
                    mamba_value=req.mamba_pool_idx.unsqueeze(0),
                )
            )

        self.assertEqual(tree._compression_queue.qsize(), tree.svd_worker_batch + 1)
        self.assertEqual(len(tree._pending_compression), tree.svd_worker_batch + 1)

    def test_compression_worker_rejects_detached_node(self):
        tree = self._fresh_tree(svd_compression=True)
        self._stop_compression_worker(tree)

        req = self._req()
        tree.insert(
            InsertParams(
                key=RadixKey([200]),
                value=self.allocator.alloc(1),
                mamba_value=req.mamba_pool_idx.unsqueeze(0),
            )
        )
        item = tree._compression_queue.get_nowait()
        node = tree.root_node.children.pop(200)

        self.assertFalse(tree._is_live_compression_item(item))
        self.assertNotIn(node.id, tree._pending_compression)

    def test_late_compression_completion_after_reset_is_dropped(self):
        tree = self._fresh_tree(svd_compression=True)
        self._stop_compression_worker(tree)

        req = self._req()
        tree.insert(
            InsertParams(
                key=RadixKey([205]),
                value=self.allocator.alloc(1),
                mamba_value=req.mamba_pool_idx.unsqueeze(0),
            )
        )
        node = tree.root_node.children[205]
        self.assertIn(node.id, tree._pending_compression)

        packed = tree.compressed_temporal[0].clone()
        tree.reset()

        # Simulate an in-flight worker publishing a completion after flush/reset.
        tree._compression_done_queue.put((node.id, packed))
        tree.drain_compression_completions()

        self.assertEqual(tree.mamba_evictable_size_, 0)
        self.assertEqual(tree.mamba_protected_size_, 0)
        self.assertFalse(node.mamba_compressed)
        self.assertNotIn(node.id, tree._compressed_lru)
        self.assertEqual(
            len(tree._compressed_free_slots), tree.compressed_temporal.shape[0]
        )

    def test_compressed_parent_survives_tombstone_cleanup(self):
        tree = self._fresh_tree(svd_compression=True)

        parent_req = self._req()
        tree.insert(
            InsertParams(
                key=RadixKey([210]),
                value=self.allocator.alloc(1),
                mamba_value=parent_req.mamba_pool_idx.unsqueeze(0),
            )
        )
        parent = tree.root_node.children[210]
        self._drain_sync(tree)
        self.assertTrue(parent.mamba_compressed)
        self.assertIn(parent.id, tree._compressed_lru)

        child_req = self._req()
        tree.insert(
            InsertParams(
                key=RadixKey([210, 211]),
                value=self.allocator.alloc(2),
                mamba_value=child_req.mamba_pool_idx.unsqueeze(0),
            )
        )
        child = parent.children[211]

        tree._evict_leaf_node(child, is_evict_mamba=False)

        self.assertIs(tree.root_node.children[210], parent)
        self.assertTrue(parent.mamba_compressed)
        self.assertIn(parent.id, tree._compressed_lru)
        self.assertTrue(tree.full_lru_list.in_list(parent))

    def test_evict_compressed_lru_reclaims_detached_leaf(self):
        tree = self._fresh_tree(svd_compression=True)

        req = self._req()
        tree.insert(
            InsertParams(
                key=RadixKey([220]),
                value=self.allocator.alloc(1),
                mamba_value=req.mamba_pool_idx.unsqueeze(0),
            )
        )
        node = tree.root_node.children[220]
        self._drain_sync(tree)
        self.assertTrue(node.mamba_compressed)

        slot = node.compressed_slot
        tree.root_node.children.pop(220)
        tree.full_lru_list.remove_node(node)
        tree.full_evictable_size_ -= len(node.key)

        self.assertTrue(tree._evict_compressed_lru())
        self.assertFalse(node.mamba_compressed)
        self.assertIsNone(node.compressed_slot)
        self.assertIn(slot, tree._compressed_free_slots)
        self.assertNotIn(node.id, tree._compressed_lru)

    def test_compression_flag_set_after_drain(self):
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
        # Async: right after insert the node is still full-rank.
        self.assertFalse(node.mamba_compressed)
        self.assertIsNotNone(node.mamba_value)

        self._drain_sync(tree)

        # After the worker runs and the drain commits, the node lives in the
        # compressed pool: full slot freed, compressed_slot set, flag flipped.
        self.assertTrue(node.mamba_compressed, "Node should be compressed after drain")
        self.assertIsNone(node.mamba_value, "Full-rank slot should be freed")
        self.assertIsNotNone(node.compressed_slot)
        self.assertIn(node.id, tree._compressed_lru)

    def test_compression_commits_while_locked(self):
        """Reproduction for the scheduler path: insert then inc_lock_ref *before*
        drain fires. The scheduler's cache_unfinished_req flow always inserts and
        then immediately locks the new last_node (mamba_lock_ref=1) as part of
        prepping the request for decode. If drain treats a locked node as
        'in-use' and drops the completion, compression NEVER happens in
        production — even though the unit tests above all pass because they
        never touch inc_lock_ref.

        This test pins the required behavior: committing a compressed slot for
        a locked node must succeed, and counter accounting must stay balanced.
        """
        tree = self._fresh_tree(svd_compression=True)
        req = self._req()
        _write_known_state(self.req_to_token_pool.mamba_pool, req.mamba_pool_idx)

        tree.insert(
            InsertParams(
                key=RadixKey([21, 22, 23]),
                value=self.allocator.alloc(3),
                mamba_value=req.mamba_pool_idx.unsqueeze(0),
            )
        )
        node = tree.root_node.children[21]

        # Simulate cache_unfinished_req: lock the node *before* the worker has
        # finished + drained. This is the exact sequencing that blocks production
        # compression today.
        tree.inc_lock_ref(node)
        self.assertEqual(node.mamba_lock_ref, 1)
        pre_protected = tree.mamba_protected_size_
        pre_evictable = tree.mamba_evictable_size_

        self._drain_sync(tree)

        self.assertTrue(
            node.mamba_compressed,
            "Drain must commit compression even when the node is locked",
        )
        self.assertIsNone(node.mamba_value)
        self.assertIsNotNone(node.compressed_slot)
        self.assertIn(node.id, tree._compressed_lru)

        # Accounting: mamba_protected_size_ should have dropped by 1 (the slot
        # that was locked is no longer in the full pool). Evictable unchanged.
        self.assertEqual(tree.mamba_protected_size_, pre_protected - 1)
        self.assertEqual(tree.mamba_evictable_size_, pre_evictable)

        # Releasing the lock on a compressed node must not underflow anything.
        tree.dec_lock_ref(node)
        self.assertEqual(node.mamba_lock_ref, 0)
        self.assertEqual(tree.mamba_protected_size_, pre_protected - 1)
        self.assertEqual(tree.mamba_evictable_size_, pre_evictable)

        # Subsequent inc/dec on a compressed node is also safe.
        tree.inc_lock_ref(node)
        self.assertEqual(node.mamba_lock_ref, 1)
        tree.dec_lock_ref(node)
        self.assertEqual(node.mamba_lock_ref, 0)
        self.assertEqual(tree.mamba_protected_size_, pre_protected - 1)
        self.assertEqual(tree.mamba_evictable_size_, pre_evictable)

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

        # Force the async worker to finish + commit before we probe the compressed path.
        self._drain_sync(tree)
        node = tree.root_node.children[1]
        self.assertTrue(
            node.mamba_compressed, "Node must be compressed for this test to exercise SVD"
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
        self._drain_sync(tree)

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
        self._drain_sync(tree)

        internal = tree.root_node.children[1]
        # With Option A, compressed nodes live in _compressed_lru instead of
        # mamba_lru_list, so evict_mamba can no longer reach this internal.
        # Verify it *is* compressed and remains so after evict_mamba runs.
        self.assertTrue(internal.mamba_compressed)
        self.assertIn(internal.id, tree._compressed_lru)

        tree.evict(EvictParams(num_tokens=0, mamba_num=1))

        # evict_mamba only walks mamba_lru_list → cannot see compressed nodes.
        # The internal stays compressed.
        self.assertTrue(internal.mamba_compressed)

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
        self._drain_sync(tree)  # child must be compressed before the split happens

        req2 = self._req()
        tree.insert(
            InsertParams(
                key=RadixKey([1, 2, 3, 6, 7]),
                value=self.allocator.alloc(5),
                mamba_value=req2.mamba_pool_idx.unsqueeze(0),
            )
        )
        self._drain_sync(tree)

        parent = tree.root_node.children[1]
        self.assertIsNone(parent.mamba_value, "Split parent should be tombstone")
        self.assertFalse(parent.mamba_compressed)

        child_45 = parent.children[4]
        self.assertTrue(
            child_45.mamba_compressed, "Child should keep compressed flag after split"
        )
        # The child should still be findable in the compressed LRU list.
        self.assertIn(child_45.id, tree._compressed_lru)

        tree.sanity_check()

    def test_rank_validation_rejects_oversized(self):
        # head_dim=16, state_size=16 → slot=256. rank=100 → 100*(16+1+16)=3300 > 256
        with self.assertRaises(AssertionError) as ctx:
            self._fresh_tree(svd_compression=True, svd_rank=100)
        self.assertIn("too large", str(ctx.exception))

    def test_pack_unpack_roundtrip_async(self):
        """End-to-end check of the async packing path.

        Forces a compression + commit, then verifies that:
          1. the full-rank pool slot is freed (pool available_size increased by 1),
          2. a compressed slot is allocated,
          3. _decompress_from_pool reads back a tensor whose reconstruction error
             against the *original* known-rank-2 state is within tolerance.

        This is the single check that catches a layout mismatch between the
        worker's ``torch.cat([u, s, v])`` pack and ``_decompress_from_pool``'s
        offset-based unpack — silent corruption would show up here as a huge
        relative error while the compressed path is clearly live.
        """
        tree = self._fresh_tree(svd_compression=True, svd_rank=4)
        mamba_pool = self.req_to_token_pool.mamba_pool

        req = self._req()
        original_state = _write_known_state(mamba_pool, req.mamba_pool_idx)

        avail_before = mamba_pool.available_size()
        tree.insert(
            InsertParams(
                key=RadixKey([7, 8, 9]),
                value=self.allocator.alloc(3),
                mamba_value=req.mamba_pool_idx.unsqueeze(0),
            )
        )
        self._drain_sync(tree)

        node = tree.root_node.children[7]
        self.assertTrue(node.mamba_compressed, "Drain must flip mamba_compressed")
        self.assertIsNone(node.mamba_value, "Drain must free the full-rank slot")
        self.assertIsNotNone(node.compressed_slot)

        # The full-rank slot for the compressed node was freed during drain.
        # The pool had exactly +1 slot returned relative to before drain.
        avail_after = mamba_pool.available_size()
        self.assertEqual(
            avail_after,
            avail_before + 1,
            f"Exactly one full-rank slot should be freed: before={avail_before}, after={avail_after}",
        )

        # Reconstruct into a scratch slot and compare to the original.
        dst_scratch = mamba_pool.alloc(1)
        self.assertIsNotNone(dst_scratch)
        tree._decompress_from_pool(node, dst_scratch)

        # temporal[:, slot_tensor] is [L, 1, H, D, S]; squeeze the slot dim so the
        # subtraction against [L, H, D, S] original doesn't trigger a bad broadcast.
        reconstructed = mamba_pool.mamba_cache.temporal[:, dst_scratch].squeeze(1)
        relative_error = torch.norm(
            original_state.float() - reconstructed.float()
        ) / (torch.norm(original_state.float()) + 1e-8)
        self.assertLess(
            relative_error.item(),
            0.1,
            f"Pack/unpack relative error too high: {relative_error:.6f}",
        )
        mamba_pool.free(dst_scratch)

    def test_free_mamba_state_dispatch(self):
        """Unit-ish test of the _free_mamba_state dispatcher for the three node states."""
        tree = self._fresh_tree(svd_compression=True)
        mamba_pool = self.req_to_token_pool.mamba_pool

        # --- Case 1: full-rank node -> pool.free() branch.
        req_a = self._req()
        full_slot = req_a.mamba_pool_idx.unsqueeze(0).clone()
        avail_before = mamba_pool.available_size()

        node_full = TreeNode()
        node_full.mamba_value = full_slot
        node_full.mamba_compressed = False
        tree._free_mamba_state(node_full)

        self.assertIsNone(node_full.mamba_value, "full-rank branch should clear mamba_value")
        self.assertFalse(node_full.mamba_compressed)
        self.assertEqual(
            mamba_pool.available_size(),
            avail_before + 1,
            "free_mamba_state must return the full-rank slot to the pool",
        )

        # --- Case 2: compressed node -> compressed-pool branch.
        free_slots_before = len(tree._compressed_free_slots)
        node_compressed = TreeNode()
        node_compressed.mamba_value = None
        node_compressed.mamba_compressed = True
        node_compressed.compressed_slot = tree._compressed_free_slots.pop()
        tree._compressed_lru[node_compressed.id] = node_compressed

        tree._free_mamba_state(node_compressed)

        self.assertFalse(node_compressed.mamba_compressed)
        self.assertIsNone(node_compressed.compressed_slot)
        self.assertNotIn(node_compressed.id, tree._compressed_lru)
        self.assertEqual(
            len(tree._compressed_free_slots),
            free_slots_before,
            "Compressed slot should have been returned to the free list",
        )

        # --- Case 3: empty node (no mamba state at all) -> idempotent no-op.
        node_empty = TreeNode()
        node_empty.mamba_value = None
        node_empty.mamba_compressed = False
        pre_free_slots = len(tree._compressed_free_slots)
        pre_pool_avail = mamba_pool.available_size()

        tree._free_mamba_state(node_empty)  # should not raise, should not mutate pools

        self.assertEqual(len(tree._compressed_free_slots), pre_free_slots)
        self.assertEqual(mamba_pool.available_size(), pre_pool_avail)

        # --- Case 4: idempotency — freeing an already-freed compressed node must be safe.
        tree._free_mamba_state(node_compressed)
        self.assertFalse(node_compressed.mamba_compressed)

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
