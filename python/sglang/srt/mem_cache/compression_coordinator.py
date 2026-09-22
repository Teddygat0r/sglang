"""Scheduler-owned coordination of asynchronous compression across TP ranks."""

import threading
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.distributed as dist


@dataclass(frozen=True, eq=False)
class CompressionJob:
    """Unique state-version token; workers never inspect mutable radix metadata."""

    node_id: int
    ready: Optional[torch.cuda.Event] = None
    cancelled: threading.Event = field(default_factory=threading.Event)


@dataclass
class _Completion:
    sequence: int
    done: bool = False
    packed: Optional[torch.Tensor] = None


class TPCompressionCoordinator:
    """Keep admission and retirement identical despite local worker timing.

    Every admitted attempt occupies a logical slot on every rank, including
    failed snapshots and cancelled work. Slots retire in insertion order only
    after all workers acknowledge completion. No model state or CUDA events cross
    ranks; one CPU all-reduce at each scheduler drain agrees on job identities,
    readiness and cancellation. The scheduler never waits for an SVD to finish.
    """

    def __init__(self, group, capacity: int, full_slots: int, compressed_slots: int):
        if capacity < 1:
            raise ValueError("Compression coordination capacity must be positive")
        if dist.get_backend(group) == "nccl":
            raise ValueError("Compression coordination requires a CPU TP cache group")
        sizes = torch.tensor(
            [
                capacity,
                -capacity,
                full_slots,
                -full_slots,
                compressed_slots,
                -compressed_slots,
            ],
            dtype=torch.int64,
        )
        dist.all_reduce(sizes, op=dist.ReduceOp.MIN, group=group)
        agreed = sizes.tolist()
        if any(agreed[i] != -agreed[i + 1] for i in (0, 2, 4)):
            raise ValueError("Mamba compression pool sizes differ across TP ranks")
        self.group = group
        self.capacity = capacity
        self._pool_sizes = (full_slots, compressed_slots)
        self._entries: dict[CompressionJob, _Completion] = {}
        self._next_sequence = 0
        self._epoch = 0
        self._status = torch.empty((capacity + 2, 6), dtype=torch.int64, device="cpu")

    def register(self, job: CompressionJob) -> bool:
        if len(self._entries) == self.capacity:
            return False
        self._entries[job] = _Completion(self._next_sequence)
        self._next_sequence += 1
        return True

    def finish(self, job: CompressionJob, packed: Optional[torch.Tensor]) -> None:
        entry = self._entries.get(job)
        if entry is not None and not entry.done:
            entry.done = True
            entry.packed = packed

    def reset(self) -> None:
        # Do not reuse sequence numbers or free in-flight reservations on flush.
        self._epoch += 1
        for job in self._entries:
            job.cancelled.set()

    def poll(
        self, valid: Callable[[CompressionJob], bool], limit: int
    ) -> list[tuple[CompressionJob, Optional[torch.Tensor]]]:
        full, compressed = self._pool_sizes
        rows = [
            [
                self._next_sequence,
                -self._next_sequence,
                self._epoch,
                -self._epoch,
                limit,
                -limit,
            ],
            [self.capacity, -self.capacity, full, -full, compressed, -compressed],
        ]
        entries = list(self._entries.items())
        for job, entry in entries:
            healthy = (
                not job.cancelled.is_set()
                and valid(job)
                and (not entry.done or entry.packed is not None)
            )
            rows.append(
                [
                    entry.sequence,
                    -entry.sequence,
                    job.node_id,
                    -job.node_id,
                    int(entry.done),
                    int(healthy),
                ]
            )
        rows.extend([[-1, 1, -1, 1, 1, 1]] * (self.capacity - len(entries)))
        self._status.copy_(torch.tensor(rows, dtype=torch.int64))
        # Always participate, including when empty. A fast rank cannot choose
        # a different collective sequence based on its local completion queue.
        dist.all_reduce(self._status, op=dist.ReduceOp.MIN, group=self.group)
        agreed = self._status.tolist()
        for i, row in enumerate(agreed):
            pairs = (0, 2, 4) if i < 2 else (0, 2)
            if any(row[j] != -row[j + 1] for j in pairs):
                raise RuntimeError(
                    "Mamba compression job order or pool configuration differs across TP ranks"
                )

        # Propagate a failed snapshot, failed SVD, or cancellation immediately,
        # even if an earlier job still prevents retirement of this one.
        for (job, _), row in zip(entries, agreed[2:]):
            if not row[5]:
                job.cancelled.set()

        retired = []
        for (job, entry), row in zip(entries, agreed[2:]):
            if len(retired) >= limit or not row[4]:
                break
            retired.append((job, entry.packed if row[5] else None))
            del self._entries[job]
        return retired
