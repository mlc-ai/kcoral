"""Builtins as module attributes, for code that runs on the worker.

An uploaded module executes in the worker process, where the server package is
importable, so it can call builtins directly rather than through ``run``
instructions::

    from kcoral import builtin


    def main(x):
        y = builtin.randn({"shape": [256], "dtype": "float32", "seed": 0})
        return builtin.check_close(x, y)

Attribute access resolves through the registry the ``run`` instruction uses,
so ``builtin.check_close`` here is the function ``"builtin.check_close"``
names on the wire — same behaviour, same error kinds. The ``compile_*``
builtins give the GPU up only when they run as their own instruction; called
from uploaded code they compile while the worker holds the GPU, so keep
compiles at the instruction level.
"""

from __future__ import annotations

from collections.abc import Callable

from .builtin_ops import resolve
from .builtin_ops._registry import _REGISTRY

_PREFIX = "builtin."


def __getattr__(name: str) -> Callable:
    fn = resolve(_PREFIX + name)
    if fn is None:
        raise AttributeError(f"no builtin named {name!r}; the registered names are {__dir__()}")
    return fn


def __dir__() -> list[str]:
    return sorted(name.removeprefix(_PREFIX) for name in _REGISTRY)
