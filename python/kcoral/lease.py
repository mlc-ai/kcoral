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

    def depth(self, gpu_id: None) -> int:
        return 0

    def request_at(self, gpu_id: None, timestamp_ns: int) -> str | None:
        return None


class GPULeases:
    """One lease per GPU, granted FIFO among the workers pinned to it.

    Keyed by GPU id rather than indexed, because ``--gpus`` may name any ids it
    likes and they need not start at zero or be consecutive.
    """

    def __init__(self, gpu_ids: list[int]) -> None:
        self._lock = threading.Lock()
        self._initializers: dict[int, deque[Ticket]] = {gpu: deque() for gpu in gpu_ids}
        self._holder: dict[int, Worker | None] = {gpu: None for gpu in gpu_ids}
        self._waiters: dict[int, deque[tuple[Worker, Ticket]]] = {gpu: deque() for gpu in gpu_ids}
        # Who held the GPU when: (from, to, request id), on the monotonic clock.
        self._held_since_ns: dict[int, int | None] = {gpu: None for gpu in gpu_ids}
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

    def acquire(self, gpu_id: int, holder: Worker) -> float:
        """Block until ``holder`` owns the GPU; returns the wait in milliseconds.

        Unbounded on purpose: the caller leaves this out of the request's
        execution timeout, so a worker is never killed for a neighbour's
        slowness. FIFO bounds it instead, to one wait per worker ahead.
        Acquiring an already-owned lease is a no-op, as with LeaseClient.
        """
        started = time.monotonic()
        with self._lock:
            if self._holder[gpu_id] is holder:
                return 0.0
            if self._holder[gpu_id] is None and not self._waiters[gpu_id]:
                self._holder[gpu_id] = holder
                self._held_since_ns[gpu_id] = time.monotonic_ns()
                return 0.0
            ticket = Ticket()
            self._waiters[gpu_id].append((holder, ticket))
        ticket.event.wait()
        return (time.monotonic() - started) * 1000

    def release(self, gpu_id: int, holder: Worker) -> None:
        """Give up the GPU and hand it to the next waiter."""
        with self._lock:
            if self._holder[gpu_id] is not holder:
                return
            self._grant_next_locked(gpu_id)

    def abandon(self, gpu_id: int, holder: Worker) -> None:
        """Drop ``holder`` whether it held the lease or was queued for it, so the
        pool can call this unconditionally after killing a worker."""
        with self._lock:
            waiters = self._waiters[gpu_id]
            # Dropped without waking: the thread that would be waiting on the
            # ticket is the one calling this, and waking it would return as
            # though it held the GPU.
            for entry in [e for e in waiters if e[0] is holder]:
                waiters.remove(entry)
            if self._holder[gpu_id] is holder:
                self._grant_next_locked(gpu_id)

    def _grant_next_locked(self, gpu_id: int) -> None:
        """Pass the GPU to the longest-waiting worker, or mark it free. Callers
        must already hold ``_lock``."""
        now = time.monotonic_ns()
        outgoing, since = self._holder[gpu_id], self._held_since_ns[gpu_id]
        if outgoing is not None and since is not None:
            self._history[gpu_id].append((since, now, _request_of(outgoing)))
        waiters = self._waiters[gpu_id]
        if waiters:
            holder, ticket = waiters.popleft()
            self._holder[gpu_id] = holder
            self._held_since_ns[gpu_id] = now
            ticket.event.set()
        else:
            self._holder[gpu_id] = None
            self._held_since_ns[gpu_id] = None

    def depth(self, gpu_id: int) -> int:
        """Workers holding or queued for this GPU."""
        with self._lock:
            return len(self._waiters[gpu_id]) + (self._holder[gpu_id] is not None)

    def request_at(self, gpu_id: int, timestamp_ns: int) -> str | None:
        """The request holding the GPU at ``timestamp_ns``, if any."""
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
