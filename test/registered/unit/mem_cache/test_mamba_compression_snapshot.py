"""Snapshot isolation, asynchronous source lifetime and failed-job cleanup."""

import queue
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch
from sglang.srt.mem_cache.mamba_radix_cache import CompressionJob, MambaRadixCache
from sglang.srt.mem_cache.compression_staging import CompressionStaging


def make_tree(device="cpu", shape=(2, 2, 2, 8, 8)):
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
    tree.svd_worker_batch = 8
    tree.svd_rank = 2
    tree._compression_staging = CompressionStaging(
        8, (shape[0], *shape[2:]), torch.float32, device
    )
    tree.req_to_token_pool = NS(
        mamba_pool=NS(mamba_cache=NS(temporal=torch.ones(shape, device=device)))
    )
    return tree


from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="stage-b-test-1-gpu-small")


class SnapshotLifetimeTests(unittest.TestCase):
    def test_cpu_snapshot_remains_independent(self):
        tree = make_tree()
        node = NS(id=1, mamba_value=torch.tensor([1]), mamba_compressed=False)
        tree._enqueue_compression(node)
        tree.req_to_token_pool.mamba_pool.mamba_cache.temporal.fill_(9)
        job, snapshot = tree._compression_queue.get_nowait()
        self.assertEqual(job.node_id, 1)
        self.assertTrue((snapshot == 1).all())
        tree._enqueue_compression(node)
        self.assertTrue(tree._compression_queue.empty())

    def test_failure_is_cleaned_by_scheduler_without_freeing_full_state(self):
        tree = make_tree()
        node = NS(id=1, mamba_value=torch.tensor([1]), mamba_compressed=False)
        tree._pending_compression[1] = node
        job = CompressionJob(1)
        tree._compression_jobs[1] = job
        tree._compression_queue.put((job, torch.ones(2, 2, 8, 8)))

        def fail(batch):
            tree._compression_stop_event.set()
            raise RuntimeError("injected SVD failure")

        with (
            patch.object(tree, "_is_live_compression_item", return_value=True),
            patch.object(tree, "_process_compression_batch", side_effect=fail),
        ):
            tree._compression_worker()
        self.assertIs(tree._pending_compression[1], node)
        tree.drain_compression_completions()
        self.assertNotIn(1, tree._pending_compression)
        self.assertEqual(node.mamba_value.item(), 1)
        self.assertFalse(node.mamba_compressed)
        tree._enqueue_compression(node)
        self.assertIs(tree._pending_compression[1], node)
        self.assertFalse(tree._compression_queue.empty())

    def test_stale_failure_does_not_remove_new_pending_node(self):
        tree = make_tree()
        old, new = object(), object()
        tree._pending_compression[1] = new
        tree._compression_jobs[1] = CompressionJob(1)
        tree._compression_failed_queue.put(CompressionJob(1))
        tree.drain_compression_completions()
        self.assertIs(tree._pending_compression[1], new)

    def test_reset_drains_failure_notifications(self):
        tree = make_tree()
        tree.compressed_temporal = torch.empty(3)
        tree._pending_compression[1] = object()
        for q in (
            tree._compression_queue,
            tree._compression_done_queue,
            tree._compression_failed_queue,
        ):
            q.put((1, None))
        tree._reset_compression_state()
        self.assertFalse(tree._pending_compression)
        self.assertEqual(tree._compressed_free_slots, [0, 1, 2])
        self.assertTrue(tree._compression_failed_queue.empty())
        self.assertTrue(tree._compression_queue.empty())
        self.assertTrue(tree._compression_done_queue.empty())

    @unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA streams")
    def test_gpu_snapshot_orders_copy_before_full_slot_reuse(self):
        tree = make_tree("cuda")
        node = NS(
            id=1, mamba_value=torch.tensor([1], device="cuda"), mamba_compressed=False
        )
        tree._compression_staging.buffer.zero_()
        schedule = torch.cuda.Stream()
        schedule.wait_stream(torch.cuda.current_stream())
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as trace:
            with torch.cuda.stream(schedule):
                torch.cuda._sleep(20_000_000)
                tree._enqueue_compression(node)
                tree.req_to_token_pool.mamba_pool.mamba_cache.temporal.fill_(9)
        ops = {event.key for event in trace.key_averages()}
        for sync in (
            "aten::item",
            "aten::_local_scalar_dense",
            "cudaDeviceSynchronize",
            "cudaStreamSynchronize",
        ):
            self.assertNotIn(sync, ops)
        job, snapshot = tree._compression_queue.get_nowait()
        with torch.cuda.stream(tree._svd_stream):
            tree._svd_stream.wait_event(job.ready)
            observed = snapshot.clone()
        tree._svd_stream.synchronize()
        torch.cuda.current_stream().wait_stream(schedule)
        self.assertTrue((observed == 1).all())


if __name__ == "__main__":
    unittest.main()
