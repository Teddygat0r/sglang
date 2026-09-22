import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
from defer_prefill import Admission
import defer_sweep as sweep


class DeferTests(unittest.TestCase):
    def test_pause_resume_stop_and_already_admitted(self):
        gate, stop = Admission(), threading.Event()
        self.assertTrue(gate.wait(stop))
        # Publishing busy does not wait for an already-admitted batch.
        gate.publish(True)
        done = threading.Event()
        result = []
        def worker():
            result.append(gate.wait(stop))
            done.set()
        t = threading.Thread(target=worker)
        t.start()
        self.assertFalse(done.wait(.05))
        gate.publish(False)
        t.join(2)
        self.assertEqual(result, [True])
        self.assertEqual(gate.deferred_batches, 1)
        gate.publish(True)
        stop.set()
        self.assertFalse(gate.wait(stop))

    def test_schedule(self):
        calls = []
        async def fake(args, cfg, rep, target, workload):
            calls.append((args.pilot, cfg, rep, workload))
            target.mkdir(parents=True)
            sweep.save(target/'result.json', dict(config=cfg, repetition=rep,
                metrics={'test': float(cfg.get('defer_prefill', False))}, validation={'ok': True}))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(sweep, 'one_run', fake):
                asyncio.run(sweep.experiment(root))
            self.assertEqual(len(calls), 12)
            self.assertTrue(all(r[1]['svd_worker_batch'] == 8 for r in calls))
            for rep in range(5):
                rows = [r for r in calls if not r[0] and r[2] == rep]
                self.assertEqual([r[1]['label'] for r in rows], ['defer','baseline'] if rep % 2 else ['baseline','defer'])
                self.assertEqual(rows[0][3], rows[1][3])
            s = json.loads((root/'summary.json').read_text())
            self.assertEqual(s['paired_vs_baseline']['defer']['test']['ci95'], [1.,1.])

    def test_installed_hooks_do_not_wait_for_gpu(self):
        code = '''
import threading
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch, server
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache
assert not getattr(Scheduler, '_defer_prefill_installed', False)
Scheduler.process_input_requests = lambda *a: None
Scheduler.get_next_batch_to_run = lambda s: s.next
Scheduler.run_batch = lambda *a: 'forward_returned'
seen=[]
MambaRadixCache._process_compression_batch = lambda t,b: seen.append(len(b))
import defer_server
complete = threading.Event()
event = NS(record=lambda *a: None, query=complete.is_set)
s=NS(waiting_queue=[1], chunked_req=None, next=None)
Scheduler.process_input_requests(s, [])
stop=threading.Event()
cancelled=threading.Event()
tree=NS(_compression_stop_event=stop, _is_live_compression_item=lambda item: not item[0].is_set())
t=threading.Thread(target=lambda: MambaRadixCache._process_compression_batch(tree, [(cancelled,None)]))
t.start()
batch=NS(forward_mode=NS(is_extend_or_draft_extend_or_mixed=lambda: True))
s.next=batch
Scheduler.get_next_batch_to_run(s)
s.waiting_queue=[]
with patch.object(torch.cuda,'Event',return_value=event), patch.object(torch.cuda,'current_stream',return_value=None):
    assert Scheduler.run_batch(s,batch)=='forward_returned'
assert t.is_alive() and not seen
cancelled.set()
complete.set()
s.next=None
Scheduler.get_next_batch_to_run(s)
t.join(2)
assert not t.is_alive() and not seen
MambaRadixCache._process_compression_batch(tree, [(threading.Event(),None)])
assert seen == [1]
'''
        env = dict(os.environ, PRESSURE_PRE_OPTIMIZATION='0', PRESSURE_INDIVIDUAL_VARIANT='baseline',
            PYTHONPATH=str(sweep.ROOT/'python') + os.pathsep + str(Path(sweep.__file__).parent))
        r = subprocess.run([sys.executable,'-c',code], env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stdout+r.stderr)


if __name__ == '__main__':
    unittest.main()
