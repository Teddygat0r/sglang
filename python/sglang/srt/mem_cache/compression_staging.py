"""Fixed storage for queued, running, and uncommitted compression jobs."""

import threading

import torch


class CompressionStaging:
    """Reserve before copying; keep the reservation until completion is consumed.

    The scheduler copies snapshots on its own stream, before a full slot can
    be reused. The worker waits for the job's event before reading a snapshot.
    Reset may release queued jobs, but must leave in-flight reservations alone.
    """

    def __init__(self, capacity, shape, dtype, device):
        if capacity < 1:
            raise ValueError("Compression staging capacity must be positive")
        self.buffer = torch.empty((capacity, *shape), dtype=dtype, device=device)
        self._free = list(range(capacity))
        self._owners = {}
        self._lock = threading.Lock()

    def acquire(self, job):
        with self._lock:
            if not self._free:
                return None
            slot = self._free.pop()
            self._owners[job] = slot
        return self.buffer[slot]

    def release(self, job):
        with self._lock:
            slot = self._owners.pop(job, None)
            if slot is not None:
                self._free.append(slot)

    def available_size(self):
        with self._lock:
            return len(self._free)
