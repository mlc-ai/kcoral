"""Tool calls for coding agents: the ``kcoral run`` tools served over MCP."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from kcoral.tools.cli import build_program

_OUTPUT_LIMIT_BYTES = 256 * 1024**2
_MANAGED_ENV = {"CUDA_VISIBLE_DEVICES", "KCORAL_DIR"}


def send_shell(
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
        when the request failed, ``request_id`` and ``elapsed_ms``.
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
    result, report = _execute(endpoint, program, timeout_seconds)
    report.update(returncode=None, missing=[], artifacts_dir=None)
    if result is None:
        return report
    if "artifacts" in result.results:
        report["artifacts_dir"] = str(result["artifacts"].save(args.out))
    if "outcome" in result.results:
        report["returncode"] = result["outcome"]["returncode"]
        report["missing"] = result["outcome"]["missing"]
    return report


def send_python(
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
    ``kcoral_send_shell``. The script is executed as a module from the request's working
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
    server.tool(name="kcoral_send_shell")(send_shell)
    server.tool(name="kcoral_send_python")(send_python)
    return server


def main() -> None:
    """Serve the tools to an agent over standard input and output."""
    build_server().run("stdio")
