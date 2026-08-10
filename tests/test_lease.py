"""The parent-side arbitration on its own: no workers, no subprocesses.

Waiter threads are daemons, so a broken handoff fails outright rather than
stalling the suite on threads that will never be granted anything.
"""

import threading
import time
from dataclasses import dataclass

from benchmark_server.lease import GPULeases
from benchmark_server.pool import IdleWorkers


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
