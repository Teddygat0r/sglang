"""Nonblocking admission of background compression around prefill work.

The scheduler owns completion events and publishes readiness. The compression
worker only consumes the readiness signal; it never reads mutable cache or
scheduler state. Admission is advisory: work admitted just before a prefill
arrives may finish concurrently. Prefill never waits for compression.
"""

import threading
import time
from typing import Callable, Optional


class CompressionAdmission:
    def __init__(self):
        self._ready = threading.Event()
        self._ready.set()
        self._completion = None
        self._selected = False
        self._stats_lock = threading.Lock()
        self._deferred_batches = 0
        self._wait_ms = 0.0

    def refresh(self, waiting: bool, selected: Optional[bool] = None) -> None:
        """Scheduler-only; query GPU completion without synchronizing."""
        if selected is not None:
            self._selected = selected
        if self._completion is not None and self._completion.query():
            self._completion = None
        if waiting or self._selected or self._completion is not None:
            self._ready.clear()
        else:
            self._ready.set()

    def begin_prefill(self) -> None:
        self._selected = True
        self._ready.clear()

    def finish_prefill(self, completion) -> None:
        # Consecutive forwards use the same inference stream, so the newest
        # event subsumes all earlier prefill completion events.
        self._completion = completion
        self._selected = False

    def wait(self, stop: threading.Event, cancelled: Callable[[], bool]) -> bool:
        """Worker-only; cancellation/shutdown remain responsive while deferred."""
        started = time.monotonic()
        deferred = not self._ready.is_set()
        if deferred:
            with self._stats_lock:
                self._deferred_batches += 1
        try:
            while not stop.is_set() and not cancelled():
                if self._ready.wait(timeout=0.05):
                    return not stop.is_set() and not cancelled()
            return False
        finally:
            if deferred:
                with self._stats_lock:
                    self._wait_ms += (time.monotonic() - started) * 1000

    def metrics(self) -> dict:
        with self._stats_lock:
            return {
                "prefill_deferred_batches": self._deferred_batches,
                "prefill_deferral_wait_ms": self._wait_ms,
            }
