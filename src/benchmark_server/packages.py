"""Materialize multi-file function packages (upload kind ``"package"``).

A package's canonical bytes are compact, sorted-keys JSON:
``{"entry": "pkg/main.py:main", "files": {"pkg/main.py": "<source>", …}}``.
Materializing writes the files to a private directory, imports the entry module
through the normal import machinery (so imports between the package's files
work), and returns the entry attribute. The returned cleanup undoes the
import-system changes; runtimes run it on reset so a persistent worker cannot
leak one request's modules into the next.
"""

from __future__ import annotations

import importlib
import json
import shutil
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath

from .errors import ExecutionError


def load_package_entry(data: bytes) -> tuple[Callable, Callable[[], None]]:
    """Materialize canonical package bytes; returns ``(entry callable, cleanup)``."""
    files, entry_path, attribute = _parse(data)
    module_name = entry_path[: -len(".py")].replace("/", ".")
    root_name = module_name.split(".")[0]
    package_dir = Path(tempfile.mkdtemp(prefix="benchmark-package-"))
    for relative_path, source in files.items():
        target = package_dir / PurePosixPath(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding="utf-8")

    def cleanup() -> None:
        _purge_modules(root_name)
        try:
            sys.path.remove(str(package_dir))
        except ValueError:
            pass
        shutil.rmtree(package_dir, ignore_errors=True)

    # Re-import even if an earlier request loaded same-named modules; the front
    # of sys.path makes this package's directory win the lookup.
    _purge_modules(root_name)
    sys.path.insert(0, str(package_dir))
    try:
        module = importlib.import_module(module_name)
        entry = getattr(module, attribute, None)
        if not callable(entry):
            raise ExecutionError(
                "parse", f"entry module {module_name!r} has no callable {attribute!r}"
            )
    except ExecutionError:
        cleanup()
        raise
    except SyntaxError as exc:
        cleanup()
        raise ExecutionError("parse", f"syntax error: {exc}") from exc
    except Exception as exc:
        cleanup()
        raise ExecutionError("parse", f"{type(exc).__name__}: {exc}") from exc
    return entry, cleanup


def _parse(data: bytes) -> tuple[dict[str, str], str, str]:
    try:
        spec = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExecutionError("parse", f"malformed package payload: {exc}") from exc
    if not isinstance(spec, dict):
        raise ExecutionError("parse", "package payload must be an object")
    files = spec.get("files")
    entry = spec.get("entry")
    if not isinstance(files, dict) or not files:
        raise ExecutionError("parse", "package requires a non-empty 'files' object")
    for relative_path, source in files.items():
        if not isinstance(relative_path, str) or not isinstance(source, str):
            raise ExecutionError("parse", "package files must map string paths to string sources")
        _validate_relative_path(relative_path)
    if not isinstance(entry, str):
        raise ExecutionError("parse", "package requires a string 'entry'")
    entry_path, separator, attribute = entry.partition(":")
    if not separator or not attribute:
        raise ExecutionError("parse", "package 'entry' must look like 'path/to/mod.py:attribute'")
    if entry_path not in files:
        raise ExecutionError("parse", f"entry file {entry_path!r} is not among the package files")
    if not entry_path.endswith(".py"):
        raise ExecutionError("parse", "the entry file must be a .py file")
    module_parts = entry_path[: -len(".py")].split("/")
    if not all(part.isidentifier() for part in module_parts):
        raise ExecutionError("parse", f"entry path {entry_path!r} is not an importable module path")
    return files, entry_path, attribute


def _validate_relative_path(relative_path: str) -> None:
    path = PurePosixPath(relative_path)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in ("..", ".") for part in path.parts)
        or "\\" in relative_path
    ):
        raise ExecutionError("parse", f"unsafe package file path: {relative_path!r}")


def _purge_modules(root_name: str) -> None:
    for name in list(sys.modules):
        if name == root_name or name.startswith(root_name + "."):
            del sys.modules[name]
