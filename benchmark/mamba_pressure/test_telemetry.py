"""CPU checks for queued-storage accounting, without loading a model."""

from collections import OrderedDict
import gc
import queue
import unittest

import torch
from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache


def fake_init(tree):
    tree.enable_svd_compression = True
    tree._compressed_lru = OrderedDict()
    tree._compression_queue = queue.Queue()
    tree._compression_done_queue = queue.Queue()
    tree._pending_compression = {}


# This isolated test process replaces only construction. The production hooks
# under test wrap actual Python queues and actual Torch storage lifetimes.
MambaRadixCache.__init__ = fake_init
from instrumentation import install

install()


class StorageAccountingTests(unittest.TestCase):
    def test_snapshot_lifetime(self):
        tree = MambaRadixCache()
        snapshot = torch.empty(16)
        tree._compression_queue.put((1, snapshot))
        self.assertEqual(tree._pressure["staging_live_bytes"], 64)
        del snapshot
        item = tree._compression_queue.get_nowait()
        gc.collect()
        self.assertEqual(tree._pressure["staging_live_bytes"], 64)
        del item
        gc.collect()
        self.assertEqual(tree._pressure["staging_live_bytes"], 0)
        self.assertEqual(tree._pressure["staging_peak_bytes"], 64)

    def test_views_charge_whole_batch_once_until_last_release(self):
        tree = MambaRadixCache()
        packed = torch.empty(2, 16)
        tree._compression_done_queue.put((1, packed[0]))
        tree._compression_done_queue.put((2, packed[1]))
        self.assertEqual(tree._pressure["staging_live_bytes"], 128)
        self.assertEqual(tree._pressure["compression_completed"], 2)
        del packed
        first = tree._compression_done_queue.get_nowait()
        del first
        gc.collect()
        self.assertEqual(tree._pressure["staging_live_bytes"], 128)
        last = tree._compression_done_queue.get_nowait()
        del last
        gc.collect()
        self.assertEqual(tree._pressure["staging_live_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
