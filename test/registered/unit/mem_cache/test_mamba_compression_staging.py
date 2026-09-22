import queue
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.compression_staging import CompressionStaging
from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="stage-b-test-1-gpu-small")


def make_tree(device="cpu", capacity=3):
    tree = object.__new__(MambaRadixCache)
    tree.enable_svd_compression = True
    tree._compressed_free_slots = [0]
    tree._compressed_lru = {}
    tree._pending_compression = {}
    tree._compression_jobs = {}
    tree._compression_queue = queue.Queue()
    tree._compression_done_queue = queue.Queue()
    tree._compression_failed_queue = queue.Queue()
    tree._compression_stop_event = threading.Event()
    tree._svd_device = torch.device(device)
    tree._svd_stream = torch.cuda.Stream() if device == "cuda" else None
    tree.svd_worker_batch = 2
    tree.svd_rank = 2
    tree.svd_niter = 1
    tree.svd_oversample = 4
    tree.compressed_temporal = torch.empty(1)
    temporal = torch.randn(
        2,
        2,
        2,
        8,
        8,
        generator=torch.Generator(device=device).manual_seed(7),
        device=device,
    )
    tree.req_to_token_pool = SimpleNamespace(
        mamba_pool=SimpleNamespace(mamba_cache=SimpleNamespace(temporal=temporal))
    )
    tree._compression_staging = CompressionStaging(
        capacity, (2, 2, 8, 8), temporal.dtype, device
    )
    return tree


def node(node_id, device="cpu"):
    return SimpleNamespace(
        id=node_id,
        mamba_value=torch.tensor([1], device=device),
        mamba_compressed=False,
    )


class TestMambaCompressionStaging(unittest.TestCase):
    def test_cancelled_backlog_is_bounded_before_copy(self):
        tree = make_tree()
        with patch("torch.index_select", wraps=torch.index_select) as copy:
            for i in range(128):
                tree._enqueue_compression(node(i))
                tree._invalidate_compression(i)
        self.assertEqual(copy.call_count, 3)
        self.assertEqual(tree._compression_queue.qsize(), 3)
        self.assertFalse(tree._pending_compression)
        self.assertEqual(tree._compression_staging.available_size(), 0)
        batch = [tree._compression_queue.get_nowait() for _ in range(3)]
        self.assertEqual(tree._discard_cancelled_compression(batch), [])
        self.assertEqual(tree._compression_staging.available_size(), 3)
        tree._enqueue_compression(node(128))
        self.assertIn(128, tree._pending_compression)

    def test_results_still_occupy_staging_until_consumed(self):
        tree = make_tree(capacity=1)
        tree._enqueue_compression(node(1))
        item = tree._compression_queue.get_nowait()
        tree._process_compression_batch([item])
        tree._enqueue_compression(node(2))
        self.assertNotIn(2, tree._pending_compression)
        tree._invalidate_compression(1)
        tree.drain_compression_completions()
        tree._enqueue_compression(node(2))
        self.assertIn(2, tree._pending_compression)

    def test_reset_keeps_inflight_snapshot_reserved(self):
        tree = make_tree(capacity=1)
        tree._enqueue_compression(node(1))
        item = tree._compression_queue.get_nowait()
        tree._reset_compression_state()
        tree._enqueue_compression(node(2))
        self.assertNotIn(2, tree._pending_compression)
        # An already-running worker can publish its old result after reset.
        tree._process_compression_batch([item])
        tree.drain_compression_completions()
        tree._enqueue_compression(node(2))
        self.assertIn(2, tree._pending_compression)
        # Duplicate/stale notifications must not release another job's slot.
        tree._compression_failed_queue.put(item[0])
        tree.drain_compression_completions()
        self.assertEqual(tree._compression_staging.available_size(), 0)

    def test_reset_reclaims_queued_snapshots_and_results(self):
        tree = make_tree()
        for i in range(3):
            tree._enqueue_compression(node(i))
        tree._process_compression_batch([tree._compression_queue.get_nowait()])
        tree._reset_compression_state()
        self.assertEqual(tree._compression_staging.available_size(), 3)
        for i in range(3):
            tree._enqueue_compression(node(i))
        self.assertEqual(tree._compression_queue.qsize(), 3)

    def test_snapshot_copy_failure_releases_reservation(self):
        tree = make_tree(capacity=1)
        with patch("torch.index_select", side_effect=RuntimeError("injected failure")):
            tree._enqueue_compression(node(1))
        self.assertFalse(tree._pending_compression)
        self.assertEqual(tree._compression_staging.available_size(), 1)
        tree._enqueue_compression(node(1))
        self.assertIn(1, tree._pending_compression)

    @unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA streams")
    def test_snapshot_precedes_slot_reuse_on_another_worker_stream(self):
        tree = make_tree("cuda", capacity=1)
        expected = tree.req_to_token_pool.mamba_pool.mamba_cache.temporal[:, 1].clone()
        schedule = torch.cuda.Stream()
        schedule.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(schedule):
            torch.cuda._sleep(5_000_000)
            tree._enqueue_compression(node(1, "cuda"))
            tree.req_to_token_pool.mamba_pool.mamba_cache.temporal.fill_(9)
        item = tree._compression_queue.get_nowait()
        tree._process_compression_batch([item])
        torch.cuda.current_stream().wait_stream(schedule)
        torch.testing.assert_close(item[1], expected)
        tree._invalidate_compression(1)
        tree.drain_compression_completions()
        self.assertEqual(tree._compression_staging.available_size(), 1)


if __name__ == "__main__":
    unittest.main()
