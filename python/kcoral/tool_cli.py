"""Client-only command line tools for remote kernel development."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from ._tool_inputs import pack_inputs, relative_path
from .client import Client, Program

COMMANDS = ("python", "compute-sanitizer", "ncu", "run-iket", "bench", "shell")


def add_connection_args(parser):
    parser.add_argument(
        "--url", default=os.environ.get("KCORAL_URL"), help="server URL; default: KCORAL_URL"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="execution timeout in seconds (default: 300; server caps apply)",
    )
    parser.add_argument(
        "--output-limit-bytes",
        type=int,
        default=16 * 1024**2,
        help="capture limit per stream; server caps apply",
    )


def validate_connection_args(parser, args):
    if not args.url:
        parser.error("set KCORAL_URL or pass --url")
    if args.timeout <= 0 or args.output_limit_bytes <= 0:
        parser.error("--timeout and --output-limit-bytes must be positive")


def execute(args, program):
    with Client(args.url) as client:
        result = client.execute(
            program, timeout_seconds=args.timeout, output_limit_bytes=args.output_limit_bytes
        )
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.stdout_truncated or result.stderr_truncated:
        print("kcoral: remote output was truncated", file=sys.stderr)
    return result


def require_completed(result):
    if not result.completed:
        error = result.error or {}
        raise RuntimeError(
            f"{result.status}: {error.get('message', error)}\n{error.get('traceback', '')}"
        )


def validate_python_args(parser, arguments):
    remaining = iter(arguments)
    for token in remaining:
        if token == "--":
            if next(remaining, "-") != "-":
                return
            break
        if token == "-":
            break
        if not token.startswith("-"):
            return
        if token == "--check-hash-based-pycs":
            next(remaining, None)
            continue
        if token.startswith("--"):
            continue
        for index, option in enumerate(token[1:], start=1):
            if option == "i":
                parser.error("interactive Python execution is unsupported")
            if option in "cmWX":
                value = token[index + 1 :] or next(remaining, None)
                if value is None:
                    parser.error(f"Python -{option} requires an argument")
                if option in "cm":
                    return
                break
    parser.error("a script, -c command, or -m module is required; stdin execution is unsupported")


def parse_args(tool, argv):
    parser = argparse.ArgumentParser(
        prog=f"kcoral run {tool}", allow_abbrev=False, description=f"Run {tool} on a KCoral worker."
    )
    add_connection_args(parser)
    parser.add_argument(
        "--send",
        type=Path,
        action="append",
        default=[],
        help="upload a file or directory's contents; repeatable",
    )
    parser.add_argument(
        "-e",
        "--env",
        action="append",
        default=[],
        metavar="NAME[=VALUE]",
        help="set a subprocess variable; NAME copies its local value",
    )
    profiling = tool in {"ncu", "run-iket"}
    parser.add_argument(
        "--out", type=Path, required=profiling, help="new local directory for returned artifacts"
    )
    if not profiling:
        parser.add_argument(
            "--fetch",
            action="append",
            default=[],
            help="relative output file or directory to return; repeatable",
        )
    else:
        parser.set_defaults(fetch=[])
    if "--" not in argv:
        if "--help" in argv or "-h" in argv:
            parser.parse_args(["--help"])
        parser.error("separate KCoral options from tool arguments with '--'")
    boundary = argv.index("--")
    args = parser.parse_args(argv[:boundary])
    forwarded = argv[boundary + 1 :]
    if not forwarded:
        parser.error("arguments are required after '--'")
    validate_connection_args(parser, args)
    environment = {}
    for entry in args.env:
        name, separator, value = entry.partition("=")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            parser.error("environment variable names must be identifiers")
        if name in {"CUDA_VISIBLE_DEVICES", "KCORAL_DIR"}:
            parser.error(f"{name} is managed by the remote worker")
        if not separator:
            if name not in os.environ:
                parser.error(f"local environment variable {name} is not set")
            value = os.environ[name]
        if "\x00" in value:
            parser.error(f"environment variable {name} contains NUL")
        environment[name] = value
    args.env = environment
    if not profiling and bool(args.fetch) != bool(args.out):
        parser.error("--fetch and --out must be used together")
    for name in args.fetch:
        try:
            relative_path(name)
        except ValueError as exc:
            parser.error(str(exc))
    if tool == "python":
        validate_python_args(parser, forwarded)
    if profiling:
        if "--" not in forwarded:
            parser.error("separate profiler options from the application with another '--'")
        split = forwarded.index("--")
        options = forwarded[:split]
        if not forwarded[split + 1 :] or (tool == "run-iket" and "profile" not in options):
            parser.error("expected profiler options followed by '-- application [args]'")
        managed = (
            ("--output-dir", "--working-dir")
            if tool == "run-iket"
            else (
                "--export",
                "--import",
                "--mode",
                "--config-file",
                "--config-file-path",
            )
        )
        short = ("-o",) if tool == "run-iket" else ("-o", "-i")
        for token in options:
            option = token.split("=", 1)[0]
            if (option.startswith("--") and any(flag.startswith(option) for flag in managed)) or (
                not option.startswith("--") and any(option.startswith(flag) for flag in short)
            ):
                parser.error(f"{option} conflicts with managed remote capture paths or mode")
    return args, forwarded


def build_program(args, tool, forwarded):
    program = Program()
    module = program.upload(
        id="tool_runner",
        kind="module",
        source=Path(__file__).with_name("_tool_worker.py").read_text(),
    )
    runner = program.get_function(id="runner", module=module, name="run")
    inputs = program.upload(id="inputs", kind="bytes", value=pack_inputs(args.send))
    outcome = program.run(id="run", fn=runner, args=[inputs, tool, forwarded, args.env, args.fetch])
    # Preserve the exit status even when a subsequent artifact collection fails.
    program.return_(key="outcome", value=outcome)
    if args.out is not None:
        program.return_folder(key="artifacts", path="outputs")
    return program


def main(tool, argv):
    if tool == "bench":
        from .bench_cli import main as bench_main

        return bench_main(argv)
    args, forwarded = parse_args(tool, argv)
    try:
        if args.out is not None:
            if os.path.lexists(args.out):
                raise ValueError(f"output already exists; choose a new --out directory: {args.out}")
            args.out.parent.mkdir(parents=True, exist_ok=True)
        result = execute(args, build_program(args, tool, forwarded))
        if "artifacts" in result.results:
            result["artifacts"].save(args.out)
            print(f"kcoral: saved artifacts to {args.out}", file=sys.stderr)
        require_completed(result)
        outcome = result["outcome"]
        code = outcome["returncode"]
        if outcome["missing"]:
            print(f"kcoral: missing artifacts: {', '.join(outcome['missing'])}", file=sys.stderr)
            code = code or 1
        return code if code >= 0 else 128 - code
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"kcoral run {tool}: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        # Transport/protocol failures should have the same concise CLI presentation.
        print(f"kcoral run {tool}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
