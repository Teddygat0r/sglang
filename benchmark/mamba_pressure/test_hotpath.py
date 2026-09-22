"""Restoration and numerical checks retained after reverting hot-path optimizations."""

from collections import Counter
import importlib
import os
import queue
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache
from optimization_baseline import METHODS


def original(name):
    item = next(x for x in METHODS if x['name'] == name)
    namespace = {}
    exec(compile(item['source'], '<test-baseline>', 'exec'), importlib.import_module(item['module']).__dict__, namespace)
    return namespace[name]


def pool(device='cpu'):
    p = object.__new__(MambaPool)
    p.free_slots = torch.tensor([1, 2], device=device)
    p.mamba_cache = NS(temporal=torch.full((2, 3, 2, 8, 8), float('nan'), device=device),
                       conv=[torch.full((2, 3, 3, 4), float('nan'), device=device, dtype=torch.bfloat16)])
    return p


def tree(p):
    t = object.__new__(MambaRadixCache)
    t.req_to_token_pool = NS(mamba_pool=p)
    t.svd_rank = 2
    t._pressure = Counter()
    return t


class HotpathTests(unittest.TestCase):
    device = 'cpu'

    def test_default_zeroing_and_exhaustion(self):
        p = pool(self.device)
        before = p.free_slots.clone()
        self.assertIsNone(p.alloc(3))
        torch.testing.assert_close(p.free_slots, before)
        idx = p.alloc(1)
        self.assertEqual(p.mamba_cache.temporal[:, idx].count_nonzero().item(), 0)
        self.assertEqual(p.mamba_cache.conv[0][:, idx].count_nonzero().item(), 0)
        self.assertTrue(torch.isnan(p.mamba_cache.temporal[:, 2]).all())
        p.free(idx)
        self.assertEqual(p.available_size(), 2)

    def test_restore_overwrites_all_state(self):
        for compressed in (False, True):
            p = pool(self.device)
            t = tree(p)
            idx = p.alloc(1)
            node = NS(id=7, mamba_compressed=compressed, compressed_slot=0,
                      mamba_value=torch.tensor([2], device=self.device))
            if compressed:
                t.compressed_temporal = torch.randn(1, 2, 2, 34, device=self.device)
                t.compressed_conv = [torch.randn(1, 2, 3, 4, device=self.device).bfloat16()]
                pack = t.compressed_temporal[0]
                expected = (pack[..., :16].reshape(2, 2, 8, 2) * pack[..., 16:18].unsqueeze(-2)) @ pack[..., 18:].reshape(2, 2, 8, 2).transpose(-2, -1)
                conv = t.compressed_conv[0][0]
            else:
                expected = torch.randn(2, 2, 8, 8, device=self.device)
                conv = torch.randn(2, 3, 4, device=self.device).bfloat16()
                p.mamba_cache.temporal[:, 2] = expected
                p.mamba_cache.conv[0][:, 2] = conv
            t._decompress_from_pool(node, idx)
            torch.testing.assert_close(p.mamba_cache.temporal[:, idx].squeeze(1), expected)
            torch.testing.assert_close(p.mamba_cache.conv[0][:, idx].squeeze(1), conv)
            self.assertTrue(torch.isnan(p.mamba_cache.temporal[:, 0]).all())

    def test_match_restore_and_eviction_retry(self):
        for exhausted in (False, True):
            p = pool(self.device)
            p.mamba_cache.temporal[:, 2].fill_(7)
            p.mamba_cache.conv[0][:, 2].fill_(9)
            p.free_slots = torch.tensor([] if exhausted else [1], dtype=torch.int64, device=self.device)
            t = tree(p)
            node = NS(id=1, parent=None, has_mamba_state=True, mamba_compressed=False,
                      mamba_value=torch.tensor([2], device=self.device))
            t.root_node = node
            t.full_lru_list = Mock()
            t.mamba_lru_list = Mock()
            t.enable_svd_compression = False
            t.device = torch.device(self.device)
            t.inc_lock_ref = Mock()
            t.dec_lock_ref = Mock()
            t.evict = Mock(side_effect=lambda _: setattr(p, 'free_slots', torch.tensor([1], device=self.device)))
            req = NS(mamba_pool_idx=None)
            with patch.object(p, 'alloc', wraps=p.alloc) as allocate:
                t._match_post_processor(NS(cow_mamba=True, req=req), [], node, 0)
            self.assertEqual(allocate.call_count, 2 if exhausted else 1)
            self.assertTrue(all(c.kwargs == {} for c in allocate.call_args_list))
            self.assertEqual(req.mamba_pool_idx.item(), 1)
            self.assertTrue((p.mamba_cache.temporal[:, 1] == 7).all())
            self.assertTrue((p.mamba_cache.conv[0][:, 1] == 9).all())
            self.assertEqual(t.dec_lock_ref.call_count, int(exhausted))

    def test_batch_equivalence_and_input_ownership(self):
        for count in (1, 3):
            snapshots = [torch.randn(2, 2, 8, 8, device=self.device) for _ in range(count)]
            saved = [x.clone() for x in snapshots]
            outputs = []
            for baseline in (True, False):
                t = tree(pool(self.device))
                t.svd_niter = 1
                t.svd_oversample = 2
                t._svd_stream = torch.cuda.Stream() if self.device == 'cuda' else None
                t._compression_done_queue = queue.Queue()
                torch.manual_seed(123)
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                method = original('_process_compression_batch') if baseline else MambaRadixCache._process_compression_batch
                with patch('torch.stack', wraps=torch.stack) as stack:
                    method(t, list(enumerate(snapshots)))
                self.assertEqual(stack.call_count, 1)
                outputs.append([t._compression_done_queue.get_nowait() for _ in range(count)])
            for old, new in zip(*outputs):
                self.assertEqual(old[0], new[0])
                torch.testing.assert_close(old[1], new[1])
            for snap, before in zip(snapshots, saved):
                torch.testing.assert_close(snap, before)


@unittest.skipUnless(os.environ.get('PRESSURE_TEST_CUDA') == '1', 'Opt-in GPU correctness checks')
class HotpathCudaTests(HotpathTests):
    device = 'cuda'


if __name__ == '__main__':
    unittest.main()
