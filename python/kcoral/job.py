"""One request interpreter and its complete process tree on a reserved GPU set.

A small Linux subreaper supervises the interpreter without importing CUDA.
Orphaned descendants, including torchrun ranks that start new sessions, are
adopted by this supervisor. A response is sent only after all of them exit.
"""

from __future__ import annotations

import ctypes
import errno
import multiprocessing
import os
import platform
import signal
import sys
import tempfile
import time
from pathlib import Path

from .engine import execute, read_captured_output
from .lease import NoopLease
from .schemas import ProgramOutcome
from .worker import WorkerCrashed, WorkerTimeout


class JobCleanupError(RuntimeError):
    """The supervisor disappeared before confirming that its tree had exited."""


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


def _run(conn, factory, devices: str, program, workspace: str, capture_dir: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = devices
    for name in (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "TORCHELASTIC_RESTART_COUNT",
        "TORCHELASTIC_MAX_RESTARTS",
        "TORCHELASTIC_RUN_ID",
    ):
        os.environ.pop(name, None)
    runtime = getattr(factory, "for_job", factory)()
    outcome = execute(
        program,
        runtime,
        NoopLease(),
        workspace_dir=workspace,
        progress=lambda index: conn.send(("instruction", index)),
        capture_dir=capture_dir,
    )
    conn.send(("outcome", outcome))
    conn.close()


def _failure(program, index: int | None, message: str) -> dict:
    instruction = program.instructions[index] if index is not None else None
    return {
        "kind": "runtime",
        "message": message,
        "instruction_index": index,
        "instruction_op": instruction.op if instruction is not None else None,
        "instruction_id": getattr(instruction, "id", None),
        "traceback": "",
    }


def _supervise(conn, factory, devices, program, workspace, capture_dir, timeout, grace) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    # PR_SET_CHILD_SUBREAPER: adopt orphaned grandchildren even after setsid().
    if libc.prctl(36, 1, 0, 0, 0) != 0:
        conn.send(
            ("startup_error", f"cannot supervise job descendants: errno {ctypes.get_errno()}")
        )
        return
    try:
        fd = _pidfd_open(os.getpid())
        try:
            _pidfd_signal(fd, 0)
        finally:
            os.close(fd)
    except (OSError, RuntimeError) as exc:
        conn.send(("startup_error", f"Linux pidfd process supervision is unavailable: {exc}"))
        return
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    runner = context.Process(
        target=_run, args=(sender, factory, devices, program, workspace, capture_dir)
    )
    outcome = None
    index = None
    timed_out = False
    cancelled = False
    deadline = time.monotonic() + timeout
    runner.start()
    sender.close()
    try:
        conn.send(("pid", runner.pid))
        while True:
            if receiver.poll(0.05):
                try:
                    kind, value = receiver.recv()
                except EOFError:
                    break
                if kind == "instruction":
                    index = value
                elif kind == "outcome":
                    outcome = value
            if conn.poll():
                try:
                    conn.recv()
                except EOFError:
                    pass
                cancelled = True
                break
            if not runner.is_alive():
                # The pipe is drained before detecting EOF on the next pass.
                if not receiver.poll():
                    break
            if time.monotonic() >= deadline:
                timed_out = True
                break
        if outcome is not None and not timed_out and not cancelled:
            # Allow the interpreter to complete normal Python teardown first.
            runner.join(timeout=min(grace, max(0, deadline - time.monotonic())))
        incomplete = outcome is not None and runner.is_alive()
        orphaned = _drain_children(runner, grace)
        if timed_out:
            conn.send(("timeout", None))
        elif cancelled:
            conn.send(("cancelled", None))
        elif outcome is None:
            conn.send(("crashed", (runner.exitcode, index)))
        else:
            if (orphaned or incomplete) and outcome.status == "COMPLETED":
                outcome.status = "FAILED"
                outcome.error = _failure(
                    program,
                    index,
                    "program left background processes running; they were terminated",
                )
            conn.send(("outcome", outcome))
    finally:
        # This supervisor contains no submitted code. Never exit leaving its
        # adopted descendants alive, including on a lost response connection.
        _drain_children(runner, grace)
        receiver.close()
        conn.close()


def run_job(
    program,
    factory,
    devices: str,
    timeout: float,
    grace: float,
    *,
    cancelled=None,
    capture_dir: str | None = None,
) -> ProgramOutcome:
    """Execute once; return or raise only after the GPU process tree is gone."""
    if sys.platform != "linux":
        raise RuntimeError("GPU jobs require Linux process supervision")
    with tempfile.TemporaryDirectory(prefix="kcoral-job-") as directory:
        workspace = str(Path(directory) / "workspace")
        Path(workspace).mkdir()
        captures = capture_dir or str(Path(directory) / "capture")
        Path(captures).mkdir(exist_ok=True)
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        supervisor = context.Process(
            target=_supervise,
            args=(child, factory, devices, program, workspace, captures, timeout, grace),
        )
        supervisor.start()
        child.close()
        pid = None
        sent_cancel = False
        try:
            while True:
                if cancelled is not None and cancelled.is_set() and not sent_cancel:
                    parent.send("cancel")
                    sent_cancel = True
                if not parent.poll(0.1):
                    if supervisor.is_alive():
                        continue
                    raise JobCleanupError("GPU job supervisor exited without confirming cleanup")
                try:
                    kind, value = parent.recv()
                except EOFError as exc:
                    raise JobCleanupError(
                        "GPU job supervisor exited without confirming cleanup"
                    ) from exc
                if kind == "pid":
                    pid = value
                    continue
                supervisor.join()
                tail = read_captured_output(captures, pid, 8192) if pid is not None else ""
                if kind == "outcome":
                    return value
                if kind == "timeout":
                    error = WorkerTimeout("GPU job exceeded its execution timeout")
                    error.output_tail = tail
                    raise error
                if kind == "crashed":
                    error = WorkerCrashed("GPU job interpreter exited")
                    error.exitcode, error.instruction_index = value
                    error.output_tail = tail
                    raise error
                if kind == "cancelled":
                    raise WorkerCrashed("GPU job cancelled during server shutdown")
                raise RuntimeError(value)
        finally:
            parent.close()
            # On normal paths the supervisor has already acknowledged cleanup.
            # If the client connection disappears it observes EOF and cleans up.
            supervisor.join()
