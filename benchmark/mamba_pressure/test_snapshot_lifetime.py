"""Snapshot isolation, asynchronous source lifetime and failed-job cleanup."""

import os
import ast
import importlib
from pathlib import Path
import queue
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch
from sglang.srt.mem_cache.mamba_radix_cache import CompressionJob, MambaRadixCache
from sglang.srt.mem_cache.compression_staging import CompressionStaging


def make_tree(device='cpu', shape=(2, 2, 2, 8, 8)):
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
    tree._svd_stream = torch.cuda.Stream() if device == 'cuda' else None
    tree.svd_worker_batch = 8
    tree.svd_rank = 2
    tree._compression_staging = CompressionStaging(8, (shape[0], *shape[2:]), torch.float32, device)
    tree.req_to_token_pool = NS(mamba_pool=NS(mamba_cache=NS(
        temporal=torch.ones(shape, device=device))))
    return tree


class SnapshotLifetimeTests(unittest.TestCase):
    def test_pre_optimization_methods_restored_exactly(self):
        from optimization_baseline import METHODS
        for item in METHODS:
            module = importlib.import_module(item['module'])
            source = ast.parse(Path(module.__file__).read_text())
            cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == item['cls'])
            method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == item['name'])
            # Job-token type annotations changed; numerical/restore bodies did not.
            self.assertEqual([ast.dump(n) for n in method.body], [ast.dump(n) for n in ast.parse(item['source']).body[0].body])

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
            raise RuntimeError('injected SVD failure')
        with patch.object(tree, '_is_live_compression_item', return_value=True), patch.object(tree, '_process_compression_batch', side_effect=fail):
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
        for q in (tree._compression_queue, tree._compression_done_queue, tree._compression_failed_queue):
            q.put((1, None))
        tree._reset_compression_state()
        self.assertFalse(tree._pending_compression)
        self.assertEqual(tree._compressed_free_slots, [0, 1, 2])
        self.assertTrue(tree._compression_failed_queue.empty())
        self.assertTrue(tree._compression_queue.empty())
        self.assertTrue(tree._compression_done_queue.empty())

    @unittest.skipUnless(os.environ.get('PRESSURE_TEST_CUDA') == '1', 'Opt-in GPU race regression')
    def test_gpu_source_cannot_be_recycled_before_clone_finishes(self):
        shape = (24, 32, 128, 128)
        tree = make_tree('cuda', (24, 2, 32, 128, 128))
        node = NS(id=1, mamba_value=torch.tensor([1], device='cuda'), mamba_compressed=False)
        # Warm both allocators so cudaMalloc synchronization cannot mask the race.
        spare = [torch.empty(shape, device='cuda') for _ in range(2)]
        with torch.cuda.stream(tree._svd_stream):
            warm = torch.empty(shape, device='cuda')
            warm.fill_(0)
        torch.cuda.synchronize()
        del warm, spare
        clone = torch.Tensor.clone
        source_ptr = []
        copy_done = torch.cuda.Event()
        def delayed_clone(src, *args, **kwargs):
            source_ptr.append(src.data_ptr())
            torch.cuda._sleep(500_000_000)
            snapshot = clone(src, *args, **kwargs)
            copy_done.record()
            return snapshot
        with patch.object(torch.Tensor, 'clone', delayed_clone):
            tree._enqueue_compression(node)
        was_pending = not copy_done.query()
        replacement = torch.empty(shape, device='cuda')
        reused = replacement.data_ptr() == source_ptr[0]
        replacement.fill_(float('nan'))
        snapshot = tree._compression_queue.get_nowait()[1]
        torch.cuda.synchronize()
        self.assertTrue(was_pending, 'Test did not exercise an asynchronous copy')
        self.assertFalse(reused, 'Source storage was reused while the clone was pending')
        self.assertTrue(torch.isfinite(snapshot).all())
        self.assertTrue((snapshot == 1).all())


if __name__ == '__main__':
    unittest.main()
