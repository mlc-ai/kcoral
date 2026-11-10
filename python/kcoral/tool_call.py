"""Tool calls for coding agents: the ``kcoral run`` tools served over MCP."""

from __future__ import annotations

import argparse
import json
import os
import re
from functools import wraps
from pathlib import Path

from kcoral.tools.cli import build_program

_OUTPUT_LIMIT_BYTES = 256 * 1024**2
_MANAGED_ENV = {"CUDA_VISIBLE_DEVICES", "KCORAL_DIR"}


def run_shell(
    endpoint: str,
    shell_script: str,
    uploaded_files: dict[str, str],
    env: dict[str, str] | None = None,
    fetch: list[str] | None = None,
    out_dir: str | None = None,
    timeout_seconds: int = 300,
) -> dict:
    """Run a shell script on a remote KCoral GPU worker and return its outcome.

    The worker starts in an empty working directory that contains only the uploaded
    files, runs ``shell_script`` with ``bash -c`` there, and discards the directory
    afterwards. Nothing persists between calls: upload everything the script needs
    every time, and name every output you want back in ``fetch``.

    ``COMPLETED`` means the protocol finished, not that validation passed. Check the
    script's ``returncode``, ``missing`` artifacts and the benchmark's correctness results.

    :param endpoint: Server URL, for example ``http://127.0.0.1:8000``.
    :param shell_script: Script text; may span several lines.
    :param uploaded_files: Remote relative path -> local file or directory. A
        directory is uploaded recursively under the remote path.
    :param env: Environment variables set for the script, such as ``PATH``.
    :param fetch: Remote relative files or directories to return; requires ``out_dir``.
    :param out_dir: New local directory that receives the fetched outputs.
    :param timeout_seconds: Server-side execution limit; the server caps it.
    :returns: ``status`` (``COMPLETED``, ``FAILED`` or ``ERROR``), the script's
        ``returncode``, ``stdout`` and ``stderr`` (and whether each was truncated),
        ``missing`` fetch paths, the local ``artifacts_dir``, the remote ``error``
        when the request failed, ``request_id`` and ``elapsed_ms``. A local artifact-save
        failure appears in ``result_errors`` while the script's outcome and logs are kept.
    """
    env = dict(env or {})
    for name in env:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) or name in _MANAGED_ENV:
            raise ValueError(f"invalid or managed environment variable: {name}")
    if bool(fetch) != (out_dir is not None):
        raise ValueError("fetch and out_dir must be used together")
    if out_dir is not None and os.path.lexists(out_dir):
        raise ValueError(f"output already exists; choose a new out_dir: {out_dir}")
    args = argparse.Namespace(
        url=endpoint,
        timeout=timeout_seconds,
        output_limit_bytes=_OUTPUT_LIMIT_BYTES,
        send=[(remote, Path(local)) for remote, local in uploaded_files.items()],
        env=env,
        fetch=list(fetch or []),
        out=None if out_dir is None else Path(out_dir),
    )
    program = build_program(args, "shell", ["bash", "-c", shell_script])
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
    result, report = _execute(endpoint, program, timeout_seconds)
    report.update(returncode=None, missing=[], artifacts_dir=None, result_errors={})
    if result is None:
        return report
    if "outcome" in result.results:
        report["returncode"] = result["outcome"]["returncode"]
        report["missing"] = result["outcome"]["missing"]
    if "artifacts" in result.results:
        try:
            report["artifacts_dir"] = str(result["artifacts"].save(args.out))
        except (OSError, ValueError) as exc:
            error = {
                "kind": "save_failed",
                "message": f"cannot save artifacts to {args.out}: {exc}",
            }
            report["result_errors"]["artifacts"] = error
            if report["status"] == "COMPLETED":
                report.update(status="ERROR", error=error)
    return report


def run_python(
    endpoint: str,
    python_script: str,
    entry_function: str,
    uploaded_files: dict[str, str],
    args: list | None = None,
    timeout_seconds: int = 300,
) -> dict:
    """Run one function of a Python script on a remote KCoral GPU worker and return its value.

    The script is uploaded as a module next to the uploaded files, ``entry_function`` is
    called with ``args`` (JSON literals), and its return value comes back as ``value``. The
    function may import torch and use ``cuda``. It runs in KCoral's own Python environment
    (torch, flashinfer, CUTLASS), not in a task's benchmark environment, so use this for
    quick checks and experiments that return a value; run recorded benchmarks with
    ``kcoral_run_shell``. The script is executed as a module from the request's working
    directory, where the uploaded files are; it has no ``__file__``, so refer to uploads by
    relative path (for example ``sys.path.insert(0, "lib")``). Nothing persists between
    calls: upload everything the script needs every time.

    :param endpoint: Server URL, for example ``http://127.0.0.1:8000``.
    :param python_script: Source of the module to upload.
    :param entry_function: Name of a function defined by the script.
    :param uploaded_files: Remote relative path -> local file or directory, placed in the
        script's working directory.
    :param args: Positional arguments for the function, as JSON values.
    :param timeout_seconds: Server-side execution limit; the server caps it.
    :returns: ``status`` (``COMPLETED``, ``FAILED`` or ``ERROR``), the function's ``value``
        (which must be JSON-compatible: numbers, strings, booleans, lists, dicts; convert
        tensors with ``.tolist()`` or ``float()``), ``stdout`` and ``stderr`` (and whether
        each was truncated), the remote ``error`` when the request failed, ``request_id``
        and ``elapsed_ms``.
    """
    from kcoral.client import Program

    if not isinstance(entry_function, str) or not entry_function.isidentifier():
        raise ValueError(f"entry_function must be a Python identifier: {entry_function!r}")
    program = Program()
    for remote, local in uploaded_files.items():
        path = Path(local)
        if path.is_symlink() or not path.exists():
            raise ValueError(f"input must exist and not be a symlink: {path}")
        if path.is_dir():
            program.upload_folder(path, path=remote)
        else:
            program.upload_file(blob=path.read_bytes(), path=remote)
    module = program.upload(kind="module", source=python_script)
    function = program.get_function(module=module, name=entry_function)
    program.return_(key="value", value=program.run(fn=function, args=list(args or [])))
    result, report = _execute(endpoint, program, timeout_seconds)
    report["value"] = None
    if result is not None and "value" in result.results:
        value = result["value"]
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            report["status"] = "ERROR"
            report["error"] = {
                "kind": "unsupported_value",
                "message": "entry function must return JSON-compatible data (numbers, "
                f"strings, booleans, lists, dicts); got {type(value).__name__}",
            }
        else:
            report["value"] = value
    return report


def run_program(
    endpoint: str,
    program: list[dict],
    out_dir: str | None = None,
    timeout_seconds: int = 300,
) -> dict:
    """Run a KCoral program you wrote yourself: a list of protocol instructions.

    Use this when the shell and python tools cannot express the request: several uploaded
    modules or functions in one request, tensor or library uploads, ``cpu_only`` steps, or
    returning a file or folder written by Python. Instructions run top to bottom and link
    to earlier ones with ``{"$ref": "<id>"}``. The ops are ``upload`` (``kind`` is
    ``module`` with ``source`` text, or ``file`` / ``bytes`` / ``library`` / ``tensor``
    with ``local`` naming a local file to read, a tensor also giving ``dtype`` and
    ``shape``, or ``folder`` with ``local`` naming a local directory and ``path`` its
    remote path), ``get_function`` (``module`` ref, ``name``, optional ``cpu_only``),
    ``run`` (``fn`` ref, ``args`` as JSON values or refs) and ``return`` (``key`` with a
    ``value`` ref, or ``kind`` ``file`` / ``folder`` with a remote ``path`` or ref). Example::

        [
            {
                "op": "upload",
                "kind": "file",
                "id": "w",
                "local": "/abs/w.bin",
                "path": "w.bin",
            },
            {
                "op": "upload",
                "kind": "module",
                "id": "mod",
                "source": "def main(n): ...",
            },
            {
                "op": "get_function",
                "id": "fn",
                "module": {"$ref": "mod"},
                "name": "main",
            },
            {"op": "run", "id": "out", "fn": {"$ref": "fn"}, "args": [4]},
            {"op": "return", "key": "value", "value": {"$ref": "out"}},
            {"op": "return", "key": "report", "kind": "folder", "path": "report"},
        ]

    Returned values must be JSON-compatible (convert tensors with ``.tolist()``); a file
    or folder return is saved to ``out_dir/<key>``, so ``out_dir`` (a new directory) is
    required for those and their keys must be single filenames. Only the file and folder
    returns the program asked for are saved; anything else the server sends back as a
    file or folder is reported in ``result_errors`` and left unsaved. Scripts run as modules in
    the request's working directory with no ``__file__``, and uploaded modules cannot import
    each other: to use one module's function from another, pass its ``get_function`` ref as
    a ``run`` argument. The program runs in KCoral's own Python environment, and nothing
    persists between calls.

    :returns: ``status``, ``results`` (key -> value or saved path), ``result_errors``
        (key -> ``kind`` and ``message`` for a return that could not be delivered),
        ``stdout``, ``stderr``, the remote ``error`` when the request failed, or an
        ``invalid_program`` error with the offending ``instruction_index`` when the program
        was rejected before being sent.
    """
    from kcoral.artifacts import ReturnedFile, ReturnedFolder
    from kcoral.client import Program
    from kcoral.errors import ValidationError
    from kcoral.protocol import parse_program

    built = Program()
    for index, instruction in enumerate(program):
        op = instruction.get("op") if isinstance(instruction, dict) else None
        try:
            _translate(built, instruction, out_dir)
        except (TypeError, ValueError, OSError) as exc:
            return _invalid(str(exc), index, op)
    try:
        parse_program({"instructions": built.instructions})
    except ValidationError as exc:
        return _invalid(str(exc), None, None)
    if out_dir is not None:
        if os.path.lexists(out_dir):
            message = f"output directory already exists; choose a new out_dir: {out_dir}"
            return {"status": "ERROR", "error": {"kind": "out_dir", "message": message}}
        try:
            Path(out_dir).mkdir(parents=True)
        except OSError as exc:
            message = f"cannot create out_dir {out_dir}: {exc}"
            return {"status": "ERROR", "error": {"kind": "out_dir", "message": message}}
    # Save only the artifacts this program requested: their keys were validated above,
    # while a key invented by the server could name a path outside out_dir.
    artifact_keys = {
        item["key"]
        for item in built.instructions
        if item["op"] == "return" and item.get("kind") in {"file", "folder"}
    }
    result, report = _execute(endpoint, built, timeout_seconds)
    report.update(results={}, result_errors={})
    if result is None:
        return report
    for key, value in result.results.items():
        if isinstance(value, (ReturnedFile, ReturnedFolder)):
            if key not in artifact_keys:
                message = (
                    f"server returned a file or folder for {key!r}, which the program did "
                    "not request; not saved"
                )
                report["result_errors"][key] = {"kind": "unexpected_artifact", "message": message}
                continue
            try:
                report["results"][key] = str(value.save(Path(out_dir) / key))
            except (OSError, ValueError) as exc:
                message = f"could not save return {key!r} under {out_dir}: {exc}"
                report["result_errors"][key] = {"kind": "save_failed", "message": message}
            continue
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            message = (
                f"return {key!r} is {type(value).__name__}; return JSON-compatible data "
                "(numbers, strings, booleans, lists, dicts) or write it to a file"
            )
            report["result_errors"][key] = {"kind": "unsupported_value", "message": message}
        else:
            report["results"][key] = value
    if report["status"] == "COMPLETED" and report["result_errors"]:
        first = next(iter(report["result_errors"].values()))
        report["status"] = "ERROR"
        report["error"] = {
            "kind": first["kind"],
            "message": f"{len(report['result_errors'])} return(s) could not be delivered; "
            "see result_errors",
        }
    return report


def _invalid(message, index, op):
    error = {"kind": "invalid_program", "message": message}
    error.update(instruction_index=index, instruction_op=op)
    return {"status": "ERROR", "error": error}


def _translate(built, item, out_dir):
    """Apply one agent-written instruction to the program builder."""
    from kcoral.tools.runner import relative_path

    if not isinstance(item, dict):
        raise TypeError("instruction must be an object")
    fields = dict(item)
    op = fields.pop("op", None)
    if op == "upload":
        if "blob" in fields:
            raise ValueError("'blob' is not accepted; give 'local' with a local file path")
        kind = fields.pop("kind", None)
        ident = fields.pop("id", None)
        if kind == "module":
            _only(fields, {"source"})
            built.upload(id=ident, kind="module", source=fields.get("source"))
        elif kind == "folder":
            _only(fields, {"local", "path"})
            if ident is not None:
                raise ValueError("folder uploads take no id")
            built.upload_folder(_local(fields, directory=True), path=fields.get("path"))
        elif kind == "file":
            _only(fields, {"local", "path"})
            built.upload_file(id=ident, blob=_local(fields).read_bytes(), path=fields.get("path"))
        elif kind in {"bytes", "library", "tensor"}:
            _only(fields, {"local", "dtype", "shape"})
            value = _local(fields).read_bytes()
            built.upload(
                id=ident,
                kind=kind,
                value=value,
                dtype=fields.get("dtype"),
                shape=fields.get("shape"),
            )
        else:
            raise ValueError(f"unknown upload kind: {kind!r}")
    elif op == "get_function":
        _only(fields, {"id", "module", "name", "cpu_only"})
        built.get_function(
            id=fields.get("id"),
            module=fields.get("module"),
            name=fields.get("name"),
            cpu_only=fields.get("cpu_only", False),
        )
    elif op == "run":
        _only(fields, {"id", "fn", "args"})
        args = fields.get("args", [])
        if not isinstance(args, list):
            raise TypeError("'args' must be a list")
        built.run(
            id=fields.get("id"),
            fn=_register(fields.get("fn")),
            args=[_register(argument) if _is_ref(argument) else argument for argument in args],
        )
    elif op == "return":
        key, kind = fields.get("key"), fields.get("kind")
        if "kind" not in fields:
            _only(fields, {"key", "value"})
            built.return_(key=key, value=fields.get("value"))
        elif kind in {"file", "folder"}:
            _only(fields, {"key", "kind", "path"})
            if out_dir is None:
                raise ValueError(f"return {key!r} of kind {kind} needs out_dir")
            if not isinstance(key, str) or len(relative_path(key).parts) != 1:
                raise ValueError(f"file and folder return keys must be single filenames: {key!r}")
            path = fields.get("path")
            select = built.return_file if kind == "file" else built.return_folder
            select(key=key, path=_register(path) if _is_ref(path) else path)
        else:
            raise ValueError(f"unknown return kind: {kind!r}")
    else:
        raise ValueError(f"unknown op: {op!r}")


def _only(fields, allowed):
    extra = sorted(set(fields) - allowed)
    if extra:
        raise ValueError(f"unexpected fields: {extra}")


def _is_ref(value):
    return isinstance(value, dict) and set(value) == {"$ref"} and isinstance(value["$ref"], str)


def _register(value):
    from kcoral.client.program import Register

    if not _is_ref(value):
        raise ValueError("expected a {'$ref': id} reference")
    return Register(value["$ref"])


def _local(fields, *, directory=False):
    local = fields.get("local")
    if not isinstance(local, str) or not local:
        raise ValueError("'local' must name a local path")
    path = Path(local)
    if path.is_symlink() or not (path.is_dir() if directory else path.is_file()):
        what = "directory" if directory else "regular file"
        raise ValueError(f"local input must be an existing {what}, not a symlink: {path}")
    return path


def _execute(endpoint, program, timeout_seconds):
    """Run a program; return the result (or None) and the report fields it determines."""
    from kcoral.client import Client

    try:
        with Client(endpoint) as client:
            result = client.execute(
                program, timeout_seconds=timeout_seconds, output_limit_bytes=_OUTPUT_LIMIT_BYTES
            )
    except Exception as exc:
        return None, {"status": "ERROR", "error": {"kind": type(exc).__name__, "message": str(exc)}}
    report = {
        "status": result.status,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "stdout_truncated": result.stdout_truncated,
        "stderr_truncated": result.stderr_truncated,
        "error": result.error,
        "request_id": result.request_id,
        "elapsed_ms": result.elapsed_ms,
    }
    return result, report


def build_server():
    """Create the MCP server exposing the tools above."""
    from mcp.server.mcpserver import MCPServer

    server = MCPServer("kcoral")

    def register(name, function):
        @wraps(function)
        def call(*args, **kwargs):
            # Keep normal Python validation exceptions, but expose their explanations
            # over MCP instead of letting the SDK mask them as unexpected tool errors.
            try:
                return function(*args, **kwargs)
            except (TypeError, ValueError, OSError) as exc:
                return {
                    "status": "ERROR",
                    "error": {"kind": "invalid_arguments", "message": str(exc)},
                }

        server.tool(name=name)(call)

    register("kcoral_run_shell", run_shell)
    register("kcoral_run_python", run_python)
    register("kcoral_run_program", run_program)
    return server


def main() -> None:
    """Serve the tools to an agent over standard input and output."""
    build_server().run("stdio")
