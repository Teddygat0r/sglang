"""Real two-process Gloo coordination with sharded CPU cache tensors."""

import tempfile
import time
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.compression_coordinator import (
    CompressionJob,
    TPCompressionCoordinator,
)
from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cuda_ci
from test_mamba_svd_compression import _create_pool_resources

register_cuda_ci(est_time=30, suite="stage-b-test-1-gpu-small")


def _cache(capacity=2):
    pool, allocator = _create_pool_resources("cpu", tp_world_size=2)
    set_global_server_args_for_scheduler(
        ServerArgs(
            model_path="dummy",
            tp_size=2,
            page_size=1,
            mamba_svd_compression=True,
            mamba_svd_rank=4,
            max_mamba_cache_size=200,
            mamba_svd_cache_size=1,
            mamba_svd_max_pending=capacity,
        )
    )
    # Drive completion explicitly to reproduce different per-rank timing.
    with patch("sglang.srt.mem_cache.mamba_radix_cache.threading.Thread"):
        tree = MambaRadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=pool,
                token_to_kv_pool_allocator=allocator,
                page_size=1,
                tp_cache_group=dist.group.WORLD,
            )
        )
    assert pool.mamba_pool.mamba_cache.temporal.shape[2] == 2  # 4 heads / TP=2
    return tree


def _insert(tree, key):
    pool = tree.req_to_token_pool.mamba_pool
    slot = pool.alloc(1)
    # Different ranks own different values, but identical logical pool indices.
    generator = torch.Generator().manual_seed(123 + dist.get_rank())
    shape = pool.mamba_cache.temporal[:, slot].shape
    # Rank two fits inside the configured SVD rank for a meaningful roundtrip.
    state = torch.randn((*shape[:-1], 2), generator=generator) @ torch.randn(
        (*shape[:-2], 2, shape[-1]), generator=generator
    )
    pool.mamba_cache.temporal[:, slot] = state
    for conv in pool.mamba_cache.conv:
        conv[:, slot] = float(dist.get_rank() + 1)
    tree.insert(
        InsertParams(
            key=RadixKey(key),
            value=tree.token_to_kv_pool_allocator.alloc(len(key)),
            mamba_value=slot,
        )
    )
    return tree.root_node.children[key[0]]


def _hit(tree, key):
    return len(tree.match_prefix(MatchPrefixParams(key=RadixKey(key))).device_indices)


def _assert_same_cache(tree):
    state = (
        sorted(tree.root_node.children),
        tree.full_evictable_size(),
        tree.mamba_evictable_size(),
        tree.req_to_token_pool.mamba_pool.available_size(),
        tree.token_to_kv_pool_allocator.available_size(),
        len(tree._compression_jobs),
        tree._compression_staging.available_size(),
        [n.key.token_ids for n in tree._compressed_lru.values()],
    )
    gathered = [None, None]
    dist.all_gather_object(gathered, state)
    assert gathered[0] == gathered[1], gathered


def _distributed_cases(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=15),
    )
    try:
        tree = _cache()
        tree.drain_compression_completions()  # Empty ranks still participate.
        node = _insert(tree, [1, 2, 3])
        pool = tree.req_to_token_pool.mamba_pool
        expected = pool.mamba_cache.temporal[:, node.mamba_value].clone()
        expected_conv = [c[:, node.mamba_value].clone() for c in pool.mamba_cache.conv]
        tree._process_compression_batch([tree._compression_queue.get_nowait()])
        tree.drain_compression_completions()
        assert _hit(tree, [1, 2, 3]) == 3
        destination = pool.alloc(1)
        tree._decompress_from_pool(node, destination)
        torch.testing.assert_close(
            pool.mamba_cache.temporal[:, destination], expected, rtol=1e-3, atol=1e-3
        )
        for conv, original in zip(pool.mamba_cache.conv, expected_conv):
            torch.testing.assert_close(conv[:, destination], original, rtol=0, atol=0)
        pool.free(destination)

        _insert(tree, [4, 5, 6])
        pending = tree._compression_queue.get_nowait()
        if rank == 0:
            tree._process_compression_batch([pending])
        tree.drain_compression_completions()
        assert _hit(tree, [1, 2, 3]) == 3  # No unilateral compressed eviction.
        assert tree.root_node.children[4].mamba_value is not None
        _assert_same_cache(tree)
        if rank == 1:
            tree._process_compression_batch([pending])
        tree.drain_compression_completions()
        assert _hit(tree, [1, 2, 3]) == 0
        assert tree.root_node.children[4].mamba_compressed
        _assert_same_cache(tree)

        # Different worker batch sizes/order cannot reorder committed jobs.
        _insert(tree, [7, 8, 9])
        _insert(tree, [10, 11, 12])
        first, second = (tree._compression_queue.get_nowait() for _ in range(2))
        if rank == 0:
            tree._process_compression_batch([first, second])
        else:
            tree._process_compression_batch([second])
        tree.drain_compression_completions()
        assert tree.root_node.children[4].mamba_compressed
        _assert_same_cache(tree)
        if rank == 1:
            tree._process_compression_batch([first])
        tree.drain_compression_completions(max_per_call=1)
        assert tree.root_node.children[7].mamba_compressed
        assert not tree.root_node.children[10].mamba_compressed
        tree.drain_compression_completions(max_per_call=1)
        assert tree.root_node.children[10].mamba_compressed
        assert 7 not in tree.root_node.children
        _assert_same_cache(tree)

        # A rank-local snapshot failure cancels the other worker's job.
        tree = _cache(capacity=1)
        if rank == 0:
            with patch(
                "torch.index_select",
                side_effect=RuntimeError("injected snapshot failure"),
            ):
                node = _insert(tree, [20])
        else:
            node = _insert(tree, [20])
        job = tree._compression_jobs[node.id]
        tree.drain_compression_completions()
        assert job.cancelled.is_set()
        _insert(tree, [21])
        assert not tree._compression_jobs  # Failed job still holds admission.
        if rank == 1:
            assert not tree._discard_cancelled_compression(
                [tree._compression_queue.get_nowait()]
            )
        tree.drain_compression_completions()
        assert not node.mamba_compressed and node.mamba_value is not None
        _assert_same_cache(tree)
        _insert(tree, [22])
        tree._process_compression_batch([tree._compression_queue.get_nowait()])
        tree.drain_compression_completions()
        assert tree.root_node.children[22].mamba_compressed
        _assert_same_cache(tree)

        # A worker-side numerical failure preserves dense state on all ranks.
        tree = _cache(capacity=1)
        node = _insert(tree, [30])
        item = tree._compression_queue.get_nowait()
        if rank == 0:
            with patch(
                "sglang.srt.mem_cache.mamba_radix_cache.randomized_svd_eigh",
                side_effect=RuntimeError("injected SVD failure"),
            ):
                try:
                    tree._process_compression_batch([item])
                except RuntimeError:
                    tree._compression_failed_queue.put(item[0])
        else:
            tree._process_compression_batch([item])
        tree.drain_compression_completions()
        assert not node.mamba_compressed and node.mamba_value is not None
        _assert_same_cache(tree)

        # Reset retains in-flight reservations until the slow worker acknowledges.
        tree = _cache(capacity=1)
        _insert(tree, [40])
        old = tree._compression_queue.get_nowait()
        if rank == 0:
            tree._process_compression_batch([old])
        tree.reset()
        tree.req_to_token_pool.clear()
        tree.token_to_kv_pool_allocator.clear()
        _insert(tree, [41])
        assert not tree._compression_jobs
        tree.drain_compression_completions()
        assert tree._compression_staging.available_size() == 0
        if rank == 1:
            tree._acknowledge_cancelled_compression(old[0])
        tree.drain_compression_completions()
        assert tree._compression_staging.available_size() == 1
        _insert(tree, [42])
        tree._process_compression_batch([tree._compression_queue.get_nowait()])
        # Late stale notifications must not retire a replacement's reservation.
        tree._compression_failed_queue.put(old[0])
        tree.drain_compression_completions()
        assert tree.root_node.children[42].mamba_compressed
        _assert_same_cache(tree)

        # Detect inconsistent identities/configuration before changing any cache.
        coordinator = TPCompressionCoordinator(dist.group.WORLD, 2, 8, 1)
        coordinator.register(CompressionJob(100 + rank))
        try:
            coordinator.poll(lambda job: True, 8)
            raise AssertionError("Mismatched jobs were accepted")
        except RuntimeError as error:
            assert "differs across TP ranks" in str(error)
        try:
            TPCompressionCoordinator(dist.group.WORLD, 2 + rank, 8, 1)
            raise AssertionError("Mismatched capacities were accepted")
        except ValueError as error:
            assert "differ across TP ranks" in str(error)
    finally:
        dist.destroy_process_group()


class TestMambaCompressionTP(unittest.TestCase):
    def test_two_rank_cache_consistency(self):
        with tempfile.TemporaryDirectory() as tmp:
            context = mp.spawn(
                _distributed_cases,
                args=(str(Path(tmp) / "gloo"),),
                nprocs=2,
                join=False,
            )
            deadline = time.monotonic() + 45
            try:
                while not context.join(timeout=max(0, deadline - time.monotonic())):
                    if time.monotonic() >= deadline:
                        self.fail("Distributed compression test timed out")
            finally:
                for process in context.processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=5)

    def test_worker_rng_does_not_advance_global_rng(self):
        from test_mamba_compression_staging import make_tree, node

        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                tree = make_tree(device)
                tree._svd_generator = torch.Generator(device=device).manual_seed(123)
                tree._enqueue_compression(node(1, device))
                item = tree._compression_queue.get_nowait()
                get_state = (
                    torch.cuda.get_rng_state
                    if device == "cuda"
                    else torch.get_rng_state
                )
                before = get_state().clone()
                tree._process_compression_batch([item])
                self.assertTrue(torch.equal(before, get_state()))

    def test_missing_or_incorrect_tp_group_rejected(self):
        pool, allocator = _create_pool_resources("cpu", tp_world_size=2)
        set_global_server_args_for_scheduler(
            ServerArgs(model_path="dummy", tp_size=2, mamba_svd_compression=True)
        )
        for group in (None, object()):
            with self.subTest(group=group):
                with patch("torch.distributed.get_world_size", return_value=1):
                    with self.assertRaisesRegex(ValueError, "matching TP cache"):
                        MambaRadixCache(
                            CacheInitParams(
                                disable=False,
                                req_to_token_pool=pool,
                                token_to_kv_pool_allocator=allocator,
                                page_size=1,
                                tp_cache_group=group,
                            )
                        )


if __name__ == "__main__":
    unittest.main()
