"""Request-local sealed buffers on the existing worker control socket."""

import array
import ctypes
import mmap
import os
import socket
import sys
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from multiprocessing import reduction

if sys.platform == "linux":
    import fcntl

MEMFD_THRESHOLD = 1 << 20
_SEAL_NAMES = ("SEAL", "SHRINK", "GROW", "WRITE")


def _seals():
    return sum(getattr(fcntl, "F_SEAL_" + name, 1 << i) for i, name in enumerate(_SEAL_NAMES))


def _memfd_create():
    create = getattr(os, "memfd_create", None)
    fd = create("blob", 3) if create else ctypes.CDLL(None, use_errno=True).memfd_create(b"blob", 3)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "memfd_create failed")
    return fd


class HandleTimeout(OSError):
    pass


class ResultSizeError(OSError):
    pass


@dataclass(frozen=True)
class BlobHandle:
    size: int


@contextmanager
def pack_blobs(blobs, cache=None):
    fds, packed = [], blobs.copy()
    with ExitStack() as cleanup:
        try:
            if sys.platform == "linux":
                for key, data in blobs.items():
                    if len(data) < MEMFD_THRESHOLD:
                        continue
                    cached_fd = cache.borrow_memfd(key) if cache is not None else None
                    if cached_fd is not None:
                        cleanup.callback(os.close, cached_fd)
                        packed[key] = BlobHandle(len(data))
                        fds.append(cached_fd)
                        continue
                    fd = _memfd_create()  # MFD_CLOEXEC | MFD_ALLOW_SEALING
                    cleanup.callback(os.close, fd)
                    with os.fdopen(fd, "wb", closefd=False) as stream:
                        stream.write(data)
                    fcntl.fcntl(fd, getattr(fcntl, "F_ADD_SEALS", 1033), _seals())
                    packed[key] = BlobHandle(len(data))
                    fds.append(fd)
        except (OSError, AttributeError, NotImplementedError):
            cleanup.close()
            fds, packed = [], blobs
        yield packed, fds


def _recv_handle(conn):
    fds = array.array("i")
    try:
        with socket.fromfd(conn.fileno(), socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            data, ancillary, flags, _ = sock.recvmsg(
                1, socket.CMSG_SPACE(256 * fds.itemsize), socket.MSG_CMSG_CLOEXEC
            )
            for level, kind, payload in ancillary:
                if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                    fds.frombytes(payload[: len(payload) - len(payload) % fds.itemsize])
            if len(fds) != 1 or data != bytes([1]) or flags & socket.MSG_CTRUNC:
                raise OSError("invalid memfd handle message")
            if getattr(reduction, "ACKNOWLEDGE", True):
                sock.sendall(b"A")
        return fds.pop()
    finally:
        for fd in fds:
            os.close(fd)


@contextmanager
def mapped_blobs(conn, blobs, max_bytes=None, ready=lambda: True):
    descriptors = [(key, value) for key, value in blobs.items() if isinstance(value, BlobHandle)]
    if not descriptors:
        yield blobs
        return
    if max_bytes is not None:
        sizes = [p.size if isinstance(p, BlobHandle) else len(p) for p in blobs.values()]
        if any(type(n) is not int or n < 0 for n in sizes) or sum(sizes) > max_bytes:
            raise ResultSizeError("results exceed the configured response-size limit")
    mapped = blobs.copy()
    with ExitStack() as cleanup:
        for key, descriptor in descriptors:
            if not ready():
                raise HandleTimeout("worker result handles exceeded the execution timeout")
            fd = _recv_handle(conn)
            try:
                if (
                    descriptor.size < MEMFD_THRESHOLD
                    or fcntl.fcntl(fd, getattr(fcntl, "F_GET_SEALS", 1034)) & _seals() != _seals()
                    or os.fstat(fd).st_size != descriptor.size
                ):
                    raise OSError("invalid memfd size or seals")
                mapping = mmap.mmap(fd, descriptor.size, access=mmap.ACCESS_READ)
            finally:
                os.close(fd)
            cleanup.callback(mapping.close)
            mapped[key] = cleanup.enter_context(memoryview(mapping))
        yield mapped
