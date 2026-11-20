"""The parent-side arbitration on its own: no workers, no subprocesses.

Waiter threads are daemons, so a broken handoff fails outright rather than
stalling the suite on threads that will never be granted anything.
"""

import threading
import time
from dataclasses import dataclass

import pytest

from kcoral.runtime.lease import GPULeases
from kcoral.runtime.pool import IdleWorkers


def until(predicate, timeout=2.0):
    """Wait for a state the other thread reaches, so ordering is not a race."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition never held")


@dataclass
class FakeWorker:
    gpu_id: int
    name: str
    request_id: str | None = None
    client: str | None = None


def test_the_gpu_goes_to_waiters_in_arrival_order():
    """FIFO is the property that bounds how long a worker can wait, so it is the
    one worth pinning down: each waiter is queued before the next one starts."""
    leases = GPULeases([0])
    leases.acquire(0, "holder")
    granted: list[str] = []

    def wait_for_it(name):
        leases.acquire(0, name)
        granted.append(name)

    threads = []
    for position, name in enumerate(("first", "second", "third"), start=2):
        thread = threading.Thread(target=wait_for_it, args=(name,), daemon=True)
        thread.start()
        until(lambda p=position: leases.depth(0) == p)  # queued before the next starts
        threads.append(thread)

    leases.release(0, "holder")
    for name in ("first", "second", "third"):
        until(lambda n=name: granted and granted[-1] == n)
        leases.release(0, name)
    for thread in threads:
        thread.join()
    assert granted == ["first", "second", "third"]


def test_abandoning_the_holder_hands_the_gpu_to_the_next_waiter():
    """What the pool calls after killing a worker: the GPU must not be stranded."""
    leases = GPULeases([0])
    leases.acquire(0, "doomed")
    granted = threading.Event()
    waiter = threading.Thread(
        target=lambda: (leases.acquire(0, "next"), granted.set()), daemon=True
    )
    waiter.start()
    until(lambda: leases.depth(0) == 2)

    leases.abandon(0, "doomed")
    assert granted.wait(2.0), "the GPU stayed held by a worker that is gone"
    leases.release(0, "next")
    waiter.join()
    assert leases.depth(0) == 0


def test_request_at_names_the_request_that_held_the_gpu_then():
    """What the pool asks when a worker reports a cpu_only function touching the
    GPU: whose measurement might it have disturbed?"""
    leases = GPULeases([0])
    before = time.monotonic_ns()
    time.sleep(0.001)
    first = FakeWorker(0, "a", request_id="request-a")
    leases.acquire(0, first)
    during = time.monotonic_ns()
    assert leases.request_at(0, during) == "request-a"  # the holder right now
    assert leases.request_at(0, before) is None  # nobody held it then

    time.sleep(0.001)
    leases.release(0, first)
    time.sleep(0.001)
    between = time.monotonic_ns()
    time.sleep(0.001)
    leases.acquire(0, FakeWorker(0, "b"))  # a worker initializing serves no request
    assert leases.request_at(0, during) == "request-a"  # from the record, once handed on
    assert leases.request_at(0, between) is None
    assert leases.request_at(0, time.monotonic_ns()) is None


def test_workers_come_from_the_least_loaded_gpu():
    """Spread across GPUs before doubling up, which one flat queue would not do."""
    workers = [FakeWorker(0, "a0"), FakeWorker(0, "a1"), FakeWorker(1, "b0"), FakeWorker(1, "b1")]
    idle = IdleWorkers(workers)
    taken = [idle.acquire(1.0) for _ in range(4)]
    assert sorted(w.gpu_id for w in taken) == [0, 0, 1, 1]
    assert taken[0].gpu_id != taken[1].gpu_id  # spread first
    assert idle.acquire(0) is None  # all four are out
    assert len(taken) == len(set(id(w) for w in taken))  # never handed out twice


def test_a_returned_worker_goes_to_the_longest_waiting_request():
    """Direct handoff, not a wake-and-race: whoever waited longest is served."""
    idle = IdleWorkers([FakeWorker(0, "only")])
    held = idle.acquire(1.0)
    served: list[str] = []

    def wait_for_a_worker(name):
        got = idle.acquire(5.0)
        served.append(name)
        idle.release(got)

    threads = []
    for name in ("first", "second"):
        thread = threading.Thread(target=wait_for_a_worker, args=(name,), daemon=True)
        thread.start()
        until(lambda n=len(threads) + 1: idle.snapshot()[1] == n)  # queued, in order
        threads.append(thread)

    idle.release(held)
    for thread in threads:
        thread.join(timeout=5)
    assert served == ["first", "second"]


def test_only_one_replacement_enters_lease_fifo_while_requests_progress():
    leases = GPULeases([0])
    leases.acquire(0, "busy")
    order = []
    threads = []

    def initialize(name):
        with leases.initialization(0):
            leases.acquire(0, name)
            order.append(name)
            leases.release(0, name)

    for i in range(3):
        t = threading.Thread(target=initialize, args=(f"init{i}",), daemon=True)
        t.start()
        threads.append(t)
        until(lambda i=i: len(leases._initializers[0]) == i + 1)
    until(lambda: leases.depth(0) == 2)

    def request():
        leases.acquire(0, "request")
        order.append("request")
        leases.release(0, "request")

    t = threading.Thread(target=request, daemon=True)
    t.start()
    threads.append(t)
    until(lambda: leases.depth(0) == 3)
    leases.release(0, "busy")
    for t in threads:
        t.join(timeout=3)
        assert not t.is_alive()
    assert order == ["init0", "request", "init1", "init2"]
    assert leases.depth(0) == 0
    assert not leases._initializers[0]


def grant_order(leases, holder, waiters, maintenance=()):
    """Queue ``waiters`` behind ``holder`` one at a time, release, and return the
    order the GPU went round in. Each waiter hands it back as soon as it has it."""
    leases.acquire(0, holder)
    granted: list[str] = []

    def wait_for_it(waiter):
        leases.acquire(0, waiter, maintenance=waiter.name in maintenance)
        granted.append(waiter.name)
        leases.release(0, waiter)

    threads = []
    for position, waiter in enumerate(waiters, start=2):
        thread = threading.Thread(target=wait_for_it, args=(waiter,), daemon=True)
        thread.start()
        until(lambda p=position: leases.depth(0) == p)
        threads.append(thread)
    leases.release(0, holder)
    for thread in threads:
        thread.join(timeout=3)
        assert not thread.is_alive()
    return granted


def queue_of(*names):
    """Waiters named ``<client><n>``."""
    return [FakeWorker(0, name, client=name[0]) for name in names]


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        ("fifo", ["a1", "a2", "b1", "c1", "d1", "b2"]),
        # d was never served, then c, b and a were served last in that order;
        # each client's own requests stay in arrival order.
        ("fair", ["d1", "c1", "b1", "a1", "b2", "a2"]),
    ],
)
def test_fair_policy_takes_turns_between_clients(policy, expected):
    """Turns follow the last grant each client received, not when it queued, so
    one client's backlog no longer queues another client behind all of it."""
    leases = GPULeases([0], policy)
    for name in ("c0", "b0", "a0"):
        worker = FakeWorker(0, name, client=name[0])
        leases.acquire(0, worker)
        leases.release(0, worker)
    holder = FakeWorker(0, "holder", client="a")
    waiters = queue_of("a1", "a2", "b1", "c1", "d1", "b2")
    assert grant_order(leases, holder, waiters) == expected


@pytest.mark.parametrize(
    ("policy", "expected"),
    [("fifo", ["a1", "b1", "respawn"]), ("fair", ["respawn", "b1", "a1"])],
)
def test_fair_policy_serves_worker_maintenance_first(policy, expected):
    """A replacement waiting behind requests would leave the pool a worker short
    for longer; it does not count as any client's turn either."""
    leases = GPULeases([0], policy)
    holder = FakeWorker(0, "holder", client="a")
    waiters = [*queue_of("a1", "b1"), FakeWorker(0, "respawn", client="a")]
    assert grant_order(leases, holder, waiters, maintenance={"respawn"}) == expected


def test_fair_policy_keeps_a_waiting_gpu_set_whole():
    """Turn order changes who reserves first, not that a set waits whole."""
    leases = GPULeases([0, 1], "fair")
    busy = FakeWorker(0, "busy", client="a")
    leases.acquire(0, busy)
    big = FakeWorker(0, "big", client="b")
    got: list[tuple[int, ...]] = []
    thread = threading.Thread(
        target=lambda: got.append(leases.acquire_count(2, big)[0]), daemon=True
    )
    thread.start()
    until(lambda: leases.depth(1) == 1)
    small = FakeWorker(1, "small", client="a")
    granted = threading.Event()
    waiter = threading.Thread(target=lambda: (leases.acquire(1, small), granted.set()), daemon=True)
    waiter.start()
    until(lambda: leases.depth(1) == 2)
    assert not granted.wait(0.1), "a later client's turn took a device the set reserved"
    leases.release(0, busy)
    thread.join(timeout=3)
    assert got == [(0, 1)]
    leases.release_many((0, 1), big)
    assert granted.wait(2.0)
    leases.release(1, small)
