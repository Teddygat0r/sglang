"""Fresh-process CPU/GPU correctness checks through the actual server entrypoint."""
import os
import queue
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch
from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache
from test_hotpath import HotpathTests, pool, tree

variant = os.environ['PRESSURE_INDIVIDUAL_VARIANT']
protected = ('_enqueue_compression', '_compression_worker', '_take_compression_result',
             '_invalidate_compression', 'drain_compression_completions')
before = {name: getattr(MambaRadixCache, name) for name in protected}
batch_before = MambaRadixCache._process_compression_batch
import server

assert server._variant == variant
assert all(getattr(MambaRadixCache, name) is method for name, method in before.items())

for device in (['cpu', 'cuda'] if os.environ.get('PRESSURE_TEST_CUDA') == '1' else ['cpu']):
    checks = HotpathTests()
    checks.device = device
    checks.test_default_zeroing_and_exhaustion()
    if variant == 'logging':
        with patch.object(torch.Tensor, '__format__', side_effect=AssertionError('Formatted GPU tensor')):
            checks.test_restore_overwrites_all_state()
    else:
        checks.test_restore_overwrites_all_state()

    # Exercise both allocation and eviction-retry paths, asserting full overwrites.
    for exhausted in (False, True):
        p = pool(device)
        p.mamba_cache.temporal[:, 2].fill_(7)
        p.mamba_cache.conv[0][:, 2].fill_(9)
        p.free_slots = torch.tensor([] if exhausted else [1], dtype=torch.int64, device=device)
        t = tree(p)
        node = NS(id=1, parent=None, has_mamba_state=True, mamba_compressed=False,
                  mamba_value=torch.tensor([2], device=device))
        t.root_node = node
        t.full_lru_list = Mock()
        t.mamba_lru_list = Mock()
        t.enable_svd_compression = False
        t.device = torch.device(device)
        t.inc_lock_ref = Mock()
        t.dec_lock_ref = Mock()
        t.evict = Mock(side_effect=lambda _: setattr(p, 'free_slots', torch.tensor([1], device=device)))
        req = NS(mamba_pool_idx=None)
        with patch.object(p, 'alloc', wraps=p.alloc) as allocate:
            t._match_post_processor(NS(cow_mamba=True, req=req), [], node, 0)
        assert allocate.call_count == (2 if exhausted else 1)
        expected = {'zero_initialize': False} if variant == 'restore_nozero' else {}
        assert all(c.kwargs == expected for c in allocate.call_args_list)
        assert (p.mamba_cache.temporal[:, 1] == 7).all()
        assert (p.mamba_cache.conv[0][:, 1] == 9).all()
        if variant == 'restore_nozero':
            p = pool(device)
            idx = p.alloc(1, zero_initialize=False)
            assert torch.isnan(p.mamba_cache.temporal[:, idx]).all()

    for count in (1, 3):
        snapshots = [torch.randn(2, 2, 8, 8, device=device) for _ in range(count)]
        preserved = [s.clone() for s in snapshots]
        output = []
        for original in (True, False):
            t = tree(pool(device))
            t.svd_niter, t.svd_oversample = 1, 2
            t._svd_stream = torch.cuda.Stream() if device == 'cuda' else None
            t._compression_done_queue = queue.Queue()
            torch.manual_seed(123)
            if device == 'cuda':
                torch.cuda.synchronize()
            method = batch_before if original else MambaRadixCache._process_compression_batch
            with patch('torch.stack', wraps=torch.stack) as stack:
                method(t, list(enumerate(snapshots)))
            assert stack.call_count == int(original or variant != 'singleton' or count > 1)
            output.append([t._compression_done_queue.get_nowait()[1] for _ in range(count)])
        for a, b in zip(*output):
            torch.testing.assert_close(a, b)
        for a, b in zip(snapshots, preserved):
            torch.testing.assert_close(a, b)
print(f'{variant}: entrypoint, race-fix preservation, restore and SVD checks passed')
