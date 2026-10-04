"""Operating-system process cleanup and native environment access."""

from __future__ import annotations

import ctypes
import os
import signal


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
