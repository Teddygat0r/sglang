"""Deterministic eviction/replacement interleavings and GPU result-copy lifetime."""

from collections import Counter
import os
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch
from sglang.srt.mem_cache.mamba_radix_cache import CompressionJob, TreeNode
from test_snapshot_lifetime import make_tree


def cache():
    t = make_tree()
    t._pressure = Counter()
    root, node = TreeNode(), TreeNode()
    node.parent = root
    node.key = [1]
    node.children[0] = TreeNode()
    root.children[0] = node
    node.mamba_value = torch.tensor([1])
    t.root_node = root
    t.get_child_key_fn = lambda key: 0
    t.mamba_lru_list = Mock()
    t.mamba_lru_list.in_list.return_value = False
    t.mamba_evictable_size_ = 1
    t.mamba_protected_size_ = 0
    t.req_to_token_pool.mamba_pool.free = Mock()
    t.req_to_token_pool.mamba_pool.mamba_cache.conv = [torch.full((2, 3, 3, 4), 99.)]
    t.compressed_temporal = torch.zeros(1, 2, 2, 34)
    t.compressed_conv = [torch.zeros(1, 2, 3, 4)]
    return t, node


class CompressionJobTests(unittest.TestCase):
    def test_tombstone_upgrade_rejects_old_result_and_accepts_new(self):
        t, node = cache()
        t._enqueue_compression(node)
        old, _ = t._compression_queue.get_nowait()
        t._tombstone_internal_node(node)
        self.assertTrue(old.cancelled.is_set())
        node.mamba_value = torch.tensor([1])  # Even the identical slot can be reused.
        t.mamba_evictable_size_ += 1
        t._enqueue_compression(node)
        new, _ = t._compression_queue.get_nowait()
        self.assertIsNot(old, new)
        t._compression_done_queue.put((old, torch.full((2, 2, 34), 7.)))
        t.drain_compression_completions()
        self.assertFalse(node.mamba_compressed)
        self.assertIs(t._compression_jobs[node.id], new)
        self.assertEqual(t._compressed_free_slots, [0])
        t._compression_done_queue.put((new, torch.full((2, 2, 34), 11.)))
        t.drain_compression_completions()
        self.assertTrue(node.mamba_compressed)
        self.assertTrue((t.compressed_temporal == 11).all())
        self.assertTrue((t.compressed_conv[0] == 99).all())
        self.assertFalse(t._pending_compression)

    def test_evicted_state_late_result_is_dropped(self):
        t, node = cache()
        t._enqueue_compression(node)
        job, _ = t._compression_queue.get_nowait()
        t._free_mamba_state(node)
        self.assertTrue(job.cancelled.is_set())
        self.assertIsNone(node.mamba_value)
        t.req_to_token_pool.mamba_pool.free.assert_called_once()
        t._compression_done_queue.put((job, torch.ones(2, 2, 34)))
        t.drain_compression_completions()
        self.assertFalse(node.mamba_compressed)
        self.assertEqual(t._compressed_free_slots, [0])

    def test_reset_and_delayed_failure_cannot_clear_new_job(self):
        t, node = cache()
        t._enqueue_compression(node)
        old, _ = t._compression_queue.get_nowait()
        t._reset_compression_state()
        t._enqueue_compression(node)
        new, _ = t._compression_queue.get_nowait()
        t._compression_failed_queue.put(old)
        t._compression_done_queue.put((old, torch.ones(2, 2, 34)))
        t.drain_compression_completions()
        self.assertTrue(old.cancelled.is_set())
        self.assertIs(t._compression_jobs[node.id], new)
        self.assertIs(t._pending_compression[node.id], node)

    def test_eviction_while_worker_is_running(self):
        t, node = cache()
        t._enqueue_compression(node)
        started, finish = threading.Event(), threading.Event()
        def process(batch):
            started.set()
            if not finish.wait(5):
                raise RuntimeError('test barrier timed out')
            for job, _ in batch:
                t._compression_done_queue.put((job, torch.ones(2, 2, 34)))
            t._compression_stop_event.set()
        with patch.object(t, '_process_compression_batch', side_effect=process):
            worker = threading.Thread(target=t._compression_worker, daemon=True)
            worker.start()
            try:
                self.assertTrue(started.wait(5))
                t._free_mamba_state(node)
            finally:
                finish.set()
                worker.join(5)
                t._compression_stop_event.set()
            self.assertFalse(worker.is_alive())
        t.drain_compression_completions()
        self.assertFalse(node.mamba_compressed)
        self.assertFalse(t._pending_compression)

    def test_worker_cancellation_check_never_reads_tree(self):
        t, _ = cache()
        class Forbidden(dict):
            def get(self, *args):
                raise AssertionError('Worker touched scheduler metadata')
        t._pending_compression = Forbidden()
        t._compression_jobs = Forbidden()
        job = CompressionJob(42)
        self.assertTrue(t._is_live_compression_item((job, None)))
        job.cancelled.set()
        self.assertFalse(t._is_live_compression_item((job, None)))

    @unittest.skipUnless(os.environ.get('PRESSURE_TEST_CUDA') == '1', 'Opt-in GPU result-copy race')
    def test_gpu_result_storage_not_reused_during_commit(self):
        t, node = cache()
        shape = (24, 32, 4112)
        side = torch.cuda.Stream()
        node.mamba_value = torch.tensor([1], device='cuda')
        t.req_to_token_pool.mamba_pool.mamba_cache.conv = []
        t.compressed_conv = []
        t.compressed_temporal = torch.empty((1, *shape), device='cuda')
        job = CompressionJob(node.id)
        t._compression_jobs[node.id] = job
        t._pending_compression[node.id] = node
        with torch.cuda.stream(side):
            packed = torch.ones(shape, device='cuda')
            spare = torch.empty(shape, device='cuda')
        torch.cuda.synchronize()
        del spare
        ptr = packed.data_ptr()
        t._compression_done_queue.put((job, packed))
        del packed
        torch.cuda._sleep(500_000_000)
        t.drain_compression_completions()
        done = torch.cuda.Event()
        done.record()
        pending = not done.query()
        with torch.cuda.stream(side):
            replacement = torch.empty(shape, device='cuda')
            reused = replacement.data_ptr() == ptr
            replacement.fill_(float('nan'))
        torch.cuda.synchronize()
        self.assertTrue(pending)
        self.assertFalse(reused)
        self.assertTrue(torch.isfinite(t.compressed_temporal).all())
        self.assertTrue((t.compressed_temporal == 1).all())


if __name__ == '__main__':
    unittest.main()
