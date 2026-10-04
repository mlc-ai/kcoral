"""Operating-system process cleanup and native environment access."""

from __future__ import annotations

import ctypes
import errno
import os
import platform
import signal
import time
from pathlib import Path


def _terminate_process_tree(process, grace_seconds: float) -> None:
    """Terminate the worker and everything the submitted code spawned.

    The worker leads a process group (see :func:`worker_main`), so a killpg
    covers grandchildren a plain ``Process.kill()`` would orphan. SIGTERM first,
    ``grace_seconds`` for a clean exit, then SIGKILL the survivors.
    """
    if process.pid is None:
        return
    signaled_group = _signal_process_group(process.pid, signal.SIGTERM)
    if not signaled_group:
        process.terminate()
    process.join(timeout=grace_seconds)
    if signaled_group:
        _signal_process_group(process.pid, signal.SIGKILL)
    if process.is_alive():
        process.kill()
    process.join(timeout=5)


def _signal_process_group(process_group_id: int, sig: signal.Signals) -> bool:
    """Signal a process group; False when unsupported or the group is gone."""
    if not hasattr(os, "killpg"):
        return False
    try:
        os.killpg(process_group_id, sig)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # the group exists but a member is not signalable


def _environ() -> dict[str, str]:
    """The environment as C sees it, falling back to Python's view of it.

    ``os.environ`` only tracks Python's own writes, so a compiled library calling
    ``setenv`` leaves it stale and a restore diffed against it finds nothing to undo.
    Reading libc's ``environ`` is what makes such a change visible; writing back
    through ``os.environ`` is what makes the fix reach both.
    """
    try:
        entries = ctypes.POINTER(ctypes.c_char_p).in_dll(ctypes.CDLL(None), "environ")
        environ: dict[str, str] = {}
        index = 0
        while entries[index]:
            name, _, value = entries[index].decode("utf-8", "surrogateescape").partition("=")
            environ[name] = value
            index += 1
        return environ
    except Exception:
        return dict(os.environ)  # no libc ``environ`` to read, or it moved as we read


def _children(pid: int) -> set[int]:
    """Read every thread: native libraries can launch children off the main thread."""
    children: set[int] = set()
    for task in Path(f"/proc/{pid}/task").glob("*"):
        try:
            children.update(map(int, (task / "children").read_text().split()))
        except (FileNotFoundError, ProcessLookupError):
            pass
    return children


def _descendants(pid: int) -> set[int]:
    found: set[int] = set()
    pending = [pid]
    while pending:
        for child in _children(pending.pop()) - found:
            found.add(child)
            pending.append(child)
    return found


def _pidfd_open(pid: int) -> int:
    native = getattr(os, "pidfd_open", None)
    if native is not None:
        return native(pid)
    # Linux x86-64 and AArch64 share these syscall numbers. This fallback also
    # works when Python or libc was built against headers predating pidfds.
    if platform.machine() not in ("x86_64", "aarch64"):
        raise RuntimeError("this platform needs Python with pidfd support")
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.syscall(434, pid, 0)  # pidfd_open
    if fd < 0:
        raise OSError(ctypes.get_errno(), "pidfd_open failed")
    return fd


def _pidfd_signal(fd: int, sig: int) -> None:
    native = getattr(signal, "pidfd_send_signal", None)
    if native is not None:
        native(fd, sig)
        return
    if platform.machine() not in ("x86_64", "aarch64"):
        raise RuntimeError("this platform needs Python with pidfd support")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.syscall(424, fd, sig, None, 0) != 0:  # pidfd_send_signal
        raise OSError(ctypes.get_errno(), "pidfd_send_signal failed")


def _signal(pid: int, sig: int) -> None:
    try:
        fd = _pidfd_open(pid)
        try:
            # Pin the identity before checking ancestry, so PID reuse cannot
            # direct a cleanup signal at another request's process.
            if pid in _descendants(os.getpid()):
                _pidfd_signal(fd, sig)
        finally:
            os.close(fd)
    except OSError as exc:
        if exc.errno != errno.ESRCH:
            raise


def _reap() -> None:
    while True:
        try:
            if os.waitpid(-1, os.WNOHANG)[0] == 0:
                return
        except ChildProcessError:
            return


def _drain_children(runner, grace: float) -> bool:
    """Terminate all descendants and reap them; never acknowledge a live tree."""
    had_children = bool(_descendants(os.getpid()) - {runner.pid})
    deadline = time.monotonic() + grace
    signaled: set[int] = set()
    while True:
        # multiprocessing owns waitpid for the immediate interpreter.
        runner.join(timeout=0)
        if runner.exitcode is not None:
            _reap()
        children = _descendants(os.getpid())
        if not children:
            runner.join(timeout=0)
            return had_children
        for pid in children:
            if time.monotonic() >= deadline:
                _signal(pid, signal.SIGKILL)
            elif pid not in signaled:
                _signal(pid, signal.SIGTERM)
                signaled.add(pid)
        # An uninterruptible process must keep its GPU reservation. Do not
        # acknowledge cleanup until the complete tree has exited.
        time.sleep(0.02)


def _wait_for_tree_exit(runner, grace: float) -> None:
    """Allow normal teardown, including adopted multiprocessing helpers."""
    deadline = time.monotonic() + grace
    while True:
        runner.join(timeout=0)
        if runner.exitcode is not None:
            _reap()
            if not _descendants(os.getpid()):
                return
        if time.monotonic() >= deadline:
            return
        time.sleep(0.02)


def _enable_child_subreaper() -> bool:
    """Adopt orphaned descendants so supervision can confirm complete cleanup."""
    return ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) == 0
