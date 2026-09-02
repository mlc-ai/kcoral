"""Server builtins and the registry that holds them.

A builtin is a module-level function registered as ``builtin.<name>`` by
:func:`register_builtin` and looked up with :func:`resolve`. Extend the server by
adding a builtin, not a runtime. :mod:`core` holds the toolchain-free ones; every
other module covers one kernel language.

Builtins import torch/tvm/tvm_ffi lazily, so importing this package touches no
GPU. They raise :class:`ExecutionError` tagged parse, compile, runtime,
correctness, or unavailable.
"""

from __future__ import annotations

# Imported for their registrations: a builtin exists only once its module runs.
from . import core, cuda, cutedsl, tirx, triton
from ._common import torch_dtype
from ._registry import (
    cpu_only_builtins,
    register_builtin,
    resolve,
    restore_registry,
    snapshot_registry,
)
from .cuda import CUDAModule

__all__ = [
    "CUDAModule",
    "cpu_only_builtins",
    "register_builtin",
    "resolve",
    "restore_registry",
    "snapshot_registry",
    "torch_dtype",
]
