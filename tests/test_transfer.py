"""Sealed-buffer transport through real worker lifetime boundaries."""

import array
import gc
import os
import socket
import sys
import warnings
from contextlib import ExitStack
from multiprocessing import Pipe, reduction

import pytest
from support.runtime import FakeRuntime

from kcoral.protocol import compute_blob_hash
from kcoral.runtime import transfer
from kcoral.runtime.lease import NoopLeases
from kcoral.runtime.worker import Worker, WorkerCrashed, WorkerTimeout
from kcoral.schemas import GetFunction, Program, Ref, Return, Run, Upload

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="requires Linux")
SOURCE = "def main(t): return [sum(t.data), t]"


class BufferRuntime(FakeRuntime):
    def load_tensor(self, data, dtype, shape):
        import torch

        assert isinstance(data, bytes)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="The given buffer is not writable", category=UserWarning
            )
            self.retained_tensor = torch.frombuffer(data, dtype=torch.uint8)
        return super().load_tensor(data, dtype, shape)


def tensor_program(data, source=SOURCE):
    digest = compute_blob_hash(data)
    return Program(
        instructions=[
            Upload("module", "module", source=source),
            GetFunction("fn", Ref("module"), "main", cpu_only=True),
            Upload("tensor", "tensor", blob=digest, dtype="uint8", shape=[len(data)]),
            Run("result", Ref("fn"), [Ref("tensor")]),
            Return("value", Ref("result")),
        ],
        blob_bytes={digest: data},
    )


def fd_count():
    gc.collect()
    return len(os.listdir("/proc/self/fd"))


@pytest.fixture
def worker():
    instance = Worker(
        None, BufferRuntime, sandbox="none", max_requests=0, termination_grace_seconds=0.1
    )
    try:
        yield instance
    finally:
        instance.close()


def check_result(worker, data, source=SOURCE):
    program = tensor_program(data, source)
    outcome = worker.run(program, 10, NoopLeases()).execution
    assert outcome.status == "COMPLETED", outcome.error
    values = outcome.results["value"]["value"]
    assert values[0]["value"] == 3 * len(data)
    assert list(outcome.binary_parts.values()) == [data]
    assert next(iter(program.blob_bytes.values())) is data


@pytest.mark.parametrize("size", [4096, transfer.MEMFD_THRESHOLD])
def test_large_and_small_inputs_have_identical_numeric_results(worker, size, monkeypatch):
    monkeypatch.delattr(os, "memfd_create", raising=False)
    sent, original = [], worker._conn.send

    def send(message):
        if isinstance(message, tuple):
            sent.extend(
                isinstance(value, transfer.BlobHandle) for value in message[0].blob_bytes.values()
            )
        return original(message)

    monkeypatch.setattr(worker._conn, "send", send)
    received, receive = [], transfer._recv_handle

    def recv(conn):
        fd = receive(conn)
        received.append(fd)
        return fd

    monkeypatch.setattr(transfer, "_recv_handle", recv)
    pid, before = worker.pid, fd_count()
    for _ in range(2):
        check_result(worker, b"\3" * size)
        assert worker.pid == pid and fd_count() == before
    assert sent == [size >= transfer.MEMFD_THRESHOLD] * 2
    assert len(received) == 2 * (size >= transfer.MEMFD_THRESHOLD)


def test_memfd_unavailable_preserves_the_result(worker, monkeypatch):
    def unavailable(*args):
        raise OSError("memfd unavailable")

    monkeypatch.setattr(transfer, "_memfd_create", unavailable)
    source = (
        "from kcoral.runtime import transfer\n"
        "def unavailable(*args): raise OSError('memfd unavailable')\n"
        "transfer._memfd_create = unavailable\n" + SOURCE
    )
    before = fd_count()
    check_result(worker, b"\3" * transfer.MEMFD_THRESHOLD, source)
    assert fd_count() == before


@pytest.mark.parametrize("failure", ["crash", "timeout", "ack"])
def test_large_input_failures_do_not_leak_parent_fds(worker, failure, monkeypatch):
    data = b"\3" * transfer.MEMFD_THRESHOLD
    source = (
        "import os\ndef main(t): os._exit(3)"
        if failure == "crash"
        else "import time\ndef main(t): time.sleep(10)"
    )
    check_result(worker, data)
    before = fd_count()
    pid = worker.pid
    with monkeypatch.context() as patch:
        if failure == "ack":

            def failed_ack(*args):
                raise RuntimeError("did not receive acknowledgement of fd")

            patch.setattr(reduction, "send_handle", failed_ack)
        with pytest.raises(WorkerTimeout if failure == "timeout" else WorkerCrashed):
            worker.run(tensor_program(data, source), 0.3, NoopLeases())
    assert worker.pid != pid
    check_result(worker, data)
    assert fd_count() == before


@pytest.mark.parametrize(
    "fault", ["unsealed", "wrong_size", "too_large", "extra", "ctrunc", "non_memfd"]
)
def test_hostile_handles_are_rejected_without_leaking(fault, monkeypatch):
    sender, receiver = Pipe()
    with ExitStack() as cleanup, sender, receiver:
        if fault in ("unsealed", "wrong_size", "too_large"):
            import fcntl

            stream = cleanup.enter_context(os.fdopen(transfer._memfd_create(), "wb"))
            stream.truncate(transfer.MEMFD_THRESHOLD - (fault == "wrong_size"))
            if fault != "unsealed":
                fcntl.fcntl(stream.fileno(), getattr(fcntl, "F_ADD_SEALS", 1033), transfer._seals())
            fds = [stream.fileno()]
        else:
            fds = [
                cleanup.enter_context(open(os.devnull, "rb")).fileno()
                for _ in range(1 if fault == "non_memfd" else 3 if fault == "ctrunc" else 2)
            ]
        with socket.fromfd(sender.fileno(), socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.sendmsg(
                [b"\1"], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", fds).tobytes())]
            )
        if fault == "ctrunc":
            space = socket.CMSG_SPACE
            monkeypatch.setattr(
                transfer.socket, "CMSG_SPACE", lambda _n: space(array.array("i").itemsize)
            )
        before = fd_count()
        with (
            pytest.raises(OSError),
            transfer.mapped_blobs(
                receiver,
                {"blob": transfer.BlobHandle(transfer.MEMFD_THRESHOLD)},
                max_bytes=transfer.MEMFD_THRESHOLD - 1 if fault == "too_large" else None,
            ),
        ):
            pass
        assert fd_count() == before


def test_cached_memfds_are_bounded_and_borrow_survives_eviction(monkeypatch):
    from kcoral.server.cache import ByteCache

    size = transfer.MEMFD_THRESHOLD
    before = fd_count()
    cache = ByteCache(2 * size, max_object_bytes=size)
    monkeypatch.setattr(cache, "_MAX_FDS", 1)
    cache.put("a", b"a" * size)
    fd = cache.borrow_memfd("a")
    assert fd is not None
    assert cache._fd_count == 1
    owned = cache._entries["a"].fd
    cache.pin(["a"])
    cache.put("b", b"b" * size)
    assert cache.borrow_memfd("b") is None
    cache.put("c", b"c" * size)
    cache.unpin(["a"])
    cache.put("d", b"d" * size)
    with pytest.raises(OSError):
        os.fstat(owned)
    assert os.pread(fd, 1, 0) == b"a"
    os.close(fd)
    borrowed = cache.borrow_memfd("d")
    assert borrowed is not None
    cache.close()
    assert os.pread(borrowed, 1, 0) == b"d"
    os.close(borrowed)
    cache.close()
    assert fd_count() == before
