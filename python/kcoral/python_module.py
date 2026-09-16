"""Materialize uploaded Python with source lines retained for diagnostics and compilers."""

from __future__ import annotations

import hashlib
import linecache
from dataclasses import dataclass
from typing import Any

from .errors import ExecutionError


@dataclass(frozen=True)
class LoadedPythonModule:
    namespace: dict[str, Any]

    def get_function(self, name: str) -> Any:
        try:
            return self.namespace[name]
        except KeyError:
            raise ExecutionError(
                "parse", f"the uploaded Python module defines no name {name!r}"
            ) from None


def materialize_module(source: str, seeded_fnames: list[str]) -> LoadedPythonModule:
    # A kernel is re-read from its source text at compile time, so seed
    # linecache. Key by content hash so two functions in one program don't
    # overwrite each other's source.
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:16]
    fname = f"<uploaded:{digest}>"
    linecache.cache[fname] = (len(source), None, source.splitlines(True), fname)
    seeded_fnames.append(fname)
    ns: dict = {}
    try:
        # Runs in the request worker process, with its configured device visibility.
        exec(compile(source, fname, "exec"), ns)
    except SyntaxError as exc:
        raise ExecutionError("parse", f"syntax error: {exc}") from exc
    except Exception as exc:
        raise ExecutionError("parse", f"{type(exc).__name__}: {exc}") from exc
    return LoadedPythonModule(namespace=ns)
