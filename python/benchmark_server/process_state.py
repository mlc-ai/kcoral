"""Process-global settings an uploaded program can change, and how to put them back.

A program runs inside the worker process, so it can flip a torch numeric global or
set an environment variable and leave it that way. Neither is state the engine
tracks, and both change what a *later* program measures - enabling TF32 alone moves
an fp32 matmul by an order of magnitude, and the next program reports the speedup as
its own.
"""

from __future__ import annotations

import ctypes
import os
from typing import Any

# Knobs read and written as an attribute, as (dotted path under ``torch``, name).
# ``fp32_precision`` rather than the ``allow_tf32`` that shadows it: the boolean
# cannot express "unset", so writing it back turns 'none' into 'ieee' and leaves
# ``get_float32_matmul_precision`` raising for every program that follows.
_TORCH_ATTRS = (
    ("backends.cuda.matmul", "fp32_precision"),
    ("backends.cudnn", "fp32_precision"),
    ("backends.cudnn.conv", "fp32_precision"),
    ("backends.cudnn.rnn", "fp32_precision"),
    ("backends.cuda.matmul", "allow_fp16_reduced_precision_reduction"),
    ("backends.cuda.matmul", "allow_bf16_reduced_precision_reduction"),
    ("backends.cudnn", "benchmark"),
    ("backends.cudnn", "deterministic"),
)

# Knobs reached through a getter/setter pair instead, as (getter, setter). Restored
# before the attributes above: ``set_float32_matmul_precision`` writes through to the
# per-backend knobs, so the finer-grained ones have to land last to survive.
_TORCH_CALLS = (
    ("get_float32_matmul_precision", "set_float32_matmul_precision"),
    ("get_default_dtype", "set_default_dtype"),
    ("get_default_device", "set_default_device"),
    ("is_grad_enabled", "set_grad_enabled"),
    ("are_deterministic_algorithms_enabled", "use_deterministic_algorithms"),
)

# The RNG is deliberately not restored: rewinding it would hand every request the
# same `builtin.randn` draw, a bigger change than the reseeding it would undo.


def snapshot() -> dict[str, Any]:
    """The settings as they stand, to hand back to :func:`restore` later.

    Take it after the runtime's warm-up, not before: the warm-up sets
    ``CUTE_DSL_LIBS`` and its loaders run once, so an earlier snapshot would restore
    the variable away with nothing left to set it again.
    """
    return {"torch": _snapshot_torch(), "environ": _environ()}


def restore(snapshot: dict[str, Any]) -> None:
    """Put the settings back as ``snapshot`` found them."""
    _restore_torch(snapshot["torch"])
    environ = snapshot["environ"]
    current = _environ()
    for name in set(current) - set(environ):
        os.unsetenv(name)  # reaches libc even for a name ``os.environ`` never saw
        os.environ.pop(name, None)
    for name, value in environ.items():
        if current.get(name) != value:
            os.environ[name] = value  # ``__setitem__`` putenv()s, so libc is fixed too


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


def _snapshot_torch() -> dict[str, Any]:
    import torch

    state: dict[str, Any] = {}
    for path, name in _TORCH_ATTRS:
        try:
            state[f"{path}.{name}"] = getattr(_reach(torch, path), name)
        except Exception:
            pass  # a torch build without this knob
    for getter, setter in _TORCH_CALLS:
        try:
            state[setter] = getattr(torch, getter)()
        except Exception:
            pass
    return state


def _restore_torch(state: dict[str, Any]) -> None:
    """Write back only the knobs that actually moved.

    Setting one to the value it already holds is not free: ``set_default_device``
    installs a torch function mode that intercepts every tensor creation afterwards.
    """
    import torch

    for getter, setter in _TORCH_CALLS:
        if setter in state:
            try:
                current = getattr(torch, getter)()
            except Exception:
                current = _UNREADABLE  # write back rather than assume it is already right
            if current != state[setter]:
                try:
                    getattr(torch, setter)(state[setter])
                except Exception:
                    pass
    for path, name in _TORCH_ATTRS:
        if f"{path}.{name}" in state:
            try:
                holder = _reach(torch, path)
                if getattr(holder, name) != state[f"{path}.{name}"]:
                    setattr(holder, name, state[f"{path}.{name}"])
            except Exception:
                pass  # a program may have made the knob itself unwritable


_UNREADABLE = object()


def _reach(root: Any, path: str) -> Any:
    for part in path.split("."):
        root = getattr(root, part)
    return root
