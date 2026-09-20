"""Nonblocking, scheduler-published prefill admission signal for the SVD worker."""
import threading
import time


class Admission:
    def __init__(self):
        self.ready = threading.Event()
        self.ready.set()
        self.deferred_batches = 0
        self.wait_ms = 0.0

    def publish(self, busy):
        self.ready.clear() if busy else self.ready.set()

    def wait(self, stop):
        start = time.monotonic()
        deferred = not self.ready.is_set()
        if deferred:
            self.deferred_batches += 1
        while not stop.is_set():
            if self.ready.wait(.05):
                if deferred:
                    self.wait_ms += (time.monotonic() - start) * 1000
                return not stop.is_set()
        return False


def install():
    import torch
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache
    if getattr(Scheduler, '_defer_prefill_installed', False):
        return
    Scheduler._defer_prefill_installed = True
    admission = Admission()
    completion = None
    selected = False
    receive = Scheduler.process_input_requests
    select = Scheduler.get_next_batch_to_run
    forward = Scheduler.run_batch
    compress = MambaRadixCache._process_compression_batch
    info = Scheduler.get_internal_state

    def is_prefill(batch):
        return batch is not None and batch.forward_mode.is_extend_or_draft_extend_or_mixed()

    def refresh(scheduler):
        nonlocal completion
        # Query only; never synchronize or read scheduler state from the worker.
        if completion is not None and completion.query():
            completion = None
        admission.publish(bool(scheduler.waiting_queue) or scheduler.chunked_req is not None
                          or selected or completion is not None)

    def process_input(scheduler, *a, **kw):
        result = receive(scheduler, *a, **kw)
        refresh(scheduler)
        return result

    def next_batch(scheduler, *a, **kw):
        nonlocal selected
        refresh(scheduler)
        result = select(scheduler, *a, **kw)
        selected = is_prefill(result)
        refresh(scheduler)
        return result

    def run(scheduler, batch, *a, **kw):
        nonlocal selected, completion
        prefill = is_prefill(batch)
        if prefill:
            selected = True
            admission.publish(True)
        try:
            return forward(scheduler, batch, *a, **kw)
        finally:
            if prefill:
                completion = torch.cuda.Event()
                completion.record(torch.cuda.current_stream())
                selected = False
            refresh(scheduler)

    def svd(tree, batch):
        if not admission.wait(tree._compression_stop_event):
            return
        # A batch admitted just before prefill becomes pending may finish
        # concurrently. Prefill NEVER waits for that batch or takes a worker lock.
        live = [item for item in batch if tree._is_live_compression_item(item)]
        if live:
            return compress(tree, live)

    def state(scheduler, request):
        result = info(scheduler, request)
        result.internal_state['cache_observations'].update(
            defer_prefill=True, prefill_deferred_batches=admission.deferred_batches,
            prefill_deferral_wait_ms=admission.wait_ms)
        return result

    Scheduler.process_input_requests = process_input
    Scheduler.get_next_batch_to_run = next_batch
    Scheduler.run_batch = run
    Scheduler.get_internal_state = state
    MambaRadixCache._process_compression_batch = svd
