"""The GPU lease, both sides of it.

Several workers share each GPU, and a worker must hold its GPU's lease to run
GPU instructions, so no two ever touch one at the same time. :class:`LeaseClient`
is what the engine calls in the worker process; it sends messages to the parent,
where :meth:`kcoral.worker.Worker.run` answers them out of the one
:class:`GPULeases` the pool owns.

The parent arbitrates rather than the workers sharing an OS mutex: a killed
holder would leak such a mutex and deadlock its GPU for good.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:  # a type-only import: worker.py imports this module at runtime
    from .worker import Worker


class Ticket:
    """A waiter's slot, filled in by whoever gives the resource up.

    The releaser picks the slot at the head of the queue and wakes only that one,
    so order in the queue is order of service. Waking everyone to race for a
    shared condition would instead let the interpreter's scheduling decide.
    """

    __slots__ = ("event", "value")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.value: object | None = None


class Lease(Protocol):
    """A worker's claim on its GPU, as the engine sees it."""

    held: bool

    def acquire(self) -> None: ...
    def release(self) -> None: ...


class LeaseClient:
    """The worker process's half of the lease, over the pipe to the parent.

    ``acquire`` blocks on the grant; ``release`` does not wait for an answer, so
    handing the GPU on before a compile costs no round trip.
    """

    def __init__(self, conn) -> None:
        self._conn = conn
        self.held = False

    def acquire(self) -> None:
        if self.held:
            return
        self._conn.send({"__lease__": "acquire"})
        self._conn.recv()  # the grant; the parent sends nothing else meanwhile
        self.held = True

    def release(self) -> None:
        if not self.held:
            return
        self.held = False
        self._conn.send({"__lease__": "release"})


class NoopLease:
    """A worker-side lease for a process that has no GPU."""

    held = False

    def acquire(self) -> None:
        pass

    def release(self) -> None:
        pass


class NoopLeases:
    """Parent-side lease collection for CPU workers.

    It matches :class:`GPULeases` so worker lifecycle code can stay identical,
    while every reported lease timing remains zero.
    """

    def acquire(self, gpu_id: None, holder: Worker) -> float:
        return 0.0

    def release(self, gpu_id: None, holder: Worker) -> None:
        pass

    def abandon(self, gpu_id: None, holder: Worker) -> None:
        pass

    @contextmanager
    def initialization(self, gpu_id: None) -> Iterator[None]:
        yield

    def quarantine(self, gpu_ids: tuple[int, ...]) -> None:
        pass

    def depth(self, gpu_id: None) -> int:
        return 0

    def request_at(self, gpu_id: None, timestamp_ns: int) -> str | None:
        return None


class GPUUnavailable(RuntimeError):
    """A device was quarantined because job cleanup could not be verified."""


@dataclass
class _LeaseRequest:
    holder: object
    ticket: Ticket
    gpu_ids: tuple[int, ...] | None = None
    count: int = 1


class GPULeases:
    """Shared arbitration for workers bound to one GPU or a fixed set.

    A waiting set never holds a subset. Older requests reserve their place in
    the queue; independent pinned-device requests can still pass one another.
    """

    def __init__(self, gpu_ids: list[int]) -> None:
        self._lock = threading.Lock()
        self._initializers: dict[int, deque[Ticket]] = {gpu: deque() for gpu in gpu_ids}
        self._unavailable: set[int] = set()
        self._holder: dict[int, object | None] = dict.fromkeys(gpu_ids)
        self._waiters: list[_LeaseRequest] = []
        self._held_since_ns: dict[int, int | None] = dict.fromkeys(gpu_ids)
        self._history: dict[int, deque[tuple[int, int, str | None]]] = {
            gpu: deque(maxlen=256) for gpu in gpu_ids
        }

    @contextmanager
    def initialization(self, gpu_id: int) -> Iterator[None]:
        """Admit initializers FIFO, one per GPU, so requests can interleave.

        Host preparation runs before this gate; GPU work still needs a lease.
        """
        ticket = Ticket()
        with self._lock:
            pending = self._initializers[gpu_id]
            pending.append(ticket)
            if len(pending) == 1:
                ticket.event.set()
        ticket.event.wait()
        try:
            yield
        finally:
            with self._lock:
                pending = self._initializers[gpu_id]
                assert pending.popleft() is ticket
                if pending:
                    pending[0].event.set()

    def acquire(self, gpu_id: int | tuple[int, ...], holder: Worker) -> float:
        devices = (gpu_id,) if isinstance(gpu_id, int) else gpu_id
        with self._lock:
            if all(self._holder[gpu] is holder for gpu in devices):
                return 0.0
        result = self._acquire(_LeaseRequest(holder, Ticket(), devices))
        assert result is not None
        return result[1]

    def acquire_count(
        self, count: int, holder: object, *, cancelled: threading.Event | None = None
    ) -> tuple[tuple[int, ...], float] | None:
        """Wait for any complete set; cancellation removes the whole request."""
        if not 1 <= count <= len(self._holder):
            raise ValueError("GPU count exceeds this server's capacity")
        return self._acquire(_LeaseRequest(holder, Ticket(), count=count), cancelled)

    def _acquire(
        self, request: _LeaseRequest, cancelled: threading.Event | None = None
    ) -> tuple[tuple[int, ...], float] | None:
        started = time.monotonic()
        with self._lock:
            self._waiters.append(request)
            self._dispatch_locked()
        while not request.ticket.event.wait(0.1):
            if cancelled is not None and cancelled.is_set():
                with self._lock:
                    if request in self._waiters:
                        self._waiters.remove(request)
                        self._dispatch_locked()
                        return None
                    # A grant won the race; the caller will release it normally.
                    break
        if isinstance(request.ticket.value, GPUUnavailable):
            raise request.ticket.value
        return request.ticket.value, (time.monotonic() - started) * 1000

    def quarantine(self, gpu_ids: tuple[int, ...]) -> None:
        with self._lock:
            self._unavailable.update(gpu_ids)
            self._dispatch_locked()

    def release(self, gpu_id: int | tuple[int, ...], holder: Worker) -> None:
        self.release_many((gpu_id,) if isinstance(gpu_id, int) else gpu_id, holder)

    def release_many(self, gpu_ids: tuple[int, ...], holder: object) -> None:
        with self._lock:
            for gpu in gpu_ids:
                if gpu not in self._unavailable and self._holder[gpu] is holder:
                    since = self._held_since_ns[gpu]
                    if since is not None:
                        self._history[gpu].append((since, time.monotonic_ns(), _request_of(holder)))
                    self._holder[gpu] = None
                    self._held_since_ns[gpu] = None
            self._dispatch_locked()

    def abandon(self, gpu_id: int | tuple[int, ...], holder: Worker) -> None:
        with self._lock:
            self._waiters = [entry for entry in self._waiters if entry.holder is not holder]
        self.release(gpu_id, holder)

    def _dispatch_locked(self) -> None:
        blocked: set[int] = set()
        for entry in list(self._waiters):
            if (
                entry.gpu_ids is not None and self._unavailable.intersection(entry.gpu_ids)
            ) or entry.count > len(self._holder) - len(self._unavailable):
                self._waiters.remove(entry)
                entry.ticket.value = GPUUnavailable("worker cleanup failed; device unavailable")
                entry.ticket.event.set()
                continue
            if entry.gpu_ids is None:
                available = tuple(
                    gpu
                    for gpu, owner in self._holder.items()
                    if owner is None and gpu not in blocked and gpu not in self._unavailable
                )
                if len(available) < entry.count:
                    # Let active users drain instead of starving a large job.
                    blocked.update(self._holder)
                    continue
                chosen = available[: entry.count]
            else:
                chosen = entry.gpu_ids
                if any(gpu in blocked or self._holder[gpu] is not None for gpu in chosen):
                    blocked.update(chosen)
                    continue
            for gpu in chosen:
                self._holder[gpu] = entry.holder
                self._held_since_ns[gpu] = time.monotonic_ns()
            self._waiters.remove(entry)
            entry.ticket.value = chosen
            entry.ticket.event.set()

    def depth(self, gpu_id: int) -> int:
        with self._lock:
            return sum(
                entry.gpu_ids is None or gpu_id in entry.gpu_ids for entry in self._waiters
            ) + (self._holder[gpu_id] is not None)

    def request_at(self, gpu_id: int, timestamp_ns: int) -> str | None:
        with self._lock:
            holder, since = self._holder[gpu_id], self._held_since_ns[gpu_id]
            if holder is not None and since is not None and since <= timestamp_ns:
                return _request_of(holder)
            for started, ended, request_id in reversed(self._history[gpu_id]):
                if started <= timestamp_ns <= ended:
                    return request_id
            return None


def _request_of(holder: Worker) -> str | None:
    return getattr(holder, "request_id", None)
