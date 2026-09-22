"""Opt-in benchmark spans. No GPU synchronization added to request execution."""

import contextvars
import functools
import json
import os
import threading
import time
from pathlib import Path

REQUEST = contextvars.ContextVar("tail_request", default=None)


def install():
    destination = os.environ.get("PRESSURE_TAIL_PROFILE")
    if not destination:
        return
    import torch
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache

    if getattr(Scheduler, "_tail_profile_installed", False):
        return
    Scheduler._tail_profile_installed = True
    pending = []
    lock = threading.Lock()

    def wrap(cls, name, kind, metadata, gpu=False, svd=False):
        original = getattr(cls, name)

        @functools.wraps(original)
        def measured(obj, *args, **kwargs):
            data = metadata(obj, *args, **kwargs)
            token = REQUEST.set(data.get("rid", REQUEST.get()))
            data.update(kind=kind, rid=REQUEST.get(), thread=threading.get_ident())
            stream = getattr(obj, "_svd_stream", None) if svd else None
            events = None
            if gpu and torch.cuda.is_initialized():
                stream = stream if stream is not None else torch.cuda.current_stream()
                events = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                events[0].record(stream)
            data["start_ns"] = time.monotonic_ns()
            try:
                return original(obj, *args, **kwargs)
            finally:
                data["end_ns"] = time.monotonic_ns()
                if events:
                    events[1].record(stream)
                with lock:
                    pending.append((data, events))
                REQUEST.reset(token)

        setattr(cls, name, measured)

    wrap(
        Req,
        "init_next_round_input",
        "cache_lookup",
        lambda req, *a, **k: {"rid": req.rid},
    )
    wrap(
        MambaRadixCache,
        "_decompress_from_pool",
        "restore",
        lambda tree, node, *a, **k: {
            "compressed": bool(node.mamba_compressed),
            "node_id": node.id,
        },
        gpu=True,
    )
    wrap(
        MambaRadixCache,
        "drain_compression_completions",
        "commit_loop",
        lambda *a, **k: {},
        gpu=True,
    )
    wrap(
        MambaRadixCache,
        "_process_compression_batch",
        "svd",
        lambda tree, batch: {"items": len(batch)},
        gpu=True,
        svd=True,
    )
    wrap(
        Scheduler,
        "run_batch",
        "forward",
        lambda scheduler, batch, *a, **k: {
            "rids": [r.rid for r in batch.reqs],
            "mode": str(batch.forward_mode),
        },
        gpu=True,
    )
    original_info = Scheduler.get_internal_state

    def info(scheduler, request):
        result = original_info(scheduler, request)
        ready = []
        with lock:
            remaining = []
            for data, events in pending:
                if events and not events[1].query():
                    remaining.append((data, events))
                    continue
                if events:
                    data["gpu_ms"] = events[0].elapsed_time(events[1])
                ready.append(data)
            pending[:] = remaining
        path = Path(destination) / f"spans-{os.getpid()}.jsonl"
        with path.open("a") as out:
            for data in ready:
                out.write(json.dumps(data) + "\n")
        result.internal_state["cache_observations"]["tail_profile"] = True
        result.internal_state["cache_observations"]["tail_profile_pending_spans"] = len(
            pending
        )
        return result

    Scheduler.get_internal_state = info
