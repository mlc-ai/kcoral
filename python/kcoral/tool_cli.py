"""Client-only command line tools for remote kernel development."""

from __future__ import annotations

import argparse
import ipaddress
import os
import re
import sys
from pathlib import Path

from ._tool_inputs import pack_inputs, relative_path
from .client import Client, Program

COMMANDS = ("python", "compute-sanitizer", "ncu", "run-iket", "bench", "shell")


def run_main(argv):
    parser = argparse.ArgumentParser(
        prog="kcoral run", description="Run a tool on a remote server.", allow_abbrev=False
    )
    commands = parser.add_subparsers(dest="tool", required=True)
    for name in COMMANDS:
        commands.add_parser(name, add_help=False, help=f"run {name} remotely")
    # Each tool owns its arguments, including --help and native -- separators.
    args = parser.parse_args(argv[:1])
    return main(args.tool, argv[1:])


def add_connection_args(parser):
    parser.add_argument(
        "--url", help="server URL; default: KCORAL_URL; cannot combine with --host or --port"
    )
    parser.add_argument(
        "--host",
        help="HTTP server hostname or IP; overrides KCORAL_URL (default with --port: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        help="HTTP server port; overrides KCORAL_URL (default with --host: 8000)",
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
        default=256 * 1024**2,
        help="capture limit per stream (default: 256 MiB; server caps apply)",
    )


def validate_connection_args(parser, args):
    environment_url = os.environ.get("KCORAL_URL")
    if args.host is not None or args.port is not None:
        if args.url is not None:
            parser.error("--url cannot be combined with --host or --port")
        host = args.host if args.host is not None else "127.0.0.1"
        port = args.port if args.port is not None else 8000
        if not 1 <= port <= 65535:
            parser.error("--port must be between 1 and 65535")
        if not host or re.search(r"[\s\x00-\x1f\x7f/@?#\\]", host):
            parser.error("--host must be a hostname or IP address, without a scheme, port or path")
        if ":" in host:
            # Accept IPv6 both with and without URL brackets.
            address = host[1:-1] if host.startswith("[") and host.endswith("]") else host
            try:
                ipaddress.IPv6Address(address)
            except ValueError:
                parser.error(
                    "--host must be a hostname or IP address, without a scheme, port or path"
                )
            host = f"[{address}]"
        elif re.search(r"[\[\]%]", host):
            parser.error("--host must be a hostname or IP address, without a scheme, port or path")
        args.url = f"http://{host}:{port}"
        if environment_url:
            print(
                f"kcoral: warning: --host/--port override KCORAL_URL; using {args.url}",
                file=sys.stderr,
            )
    elif args.url is None:
        args.url = environment_url
    if not args.url:
        parser.error("set KCORAL_URL or pass --url, --host or --port")
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


def parse_args(tool, argv, *, epilog=None):
    parser = argparse.ArgumentParser(
        prog=f"kcoral run {tool}",
        allow_abbrev=False,
        description=f"Run {tool} on a KCoral worker.",
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        help="set a remote environment variable; NAME copies its local value",
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
    if tool == "bench" and args.fetch and args.out is None:
        parser.error("--fetch requires --out")
    if tool != "bench" and not profiling and bool(args.fetch) != bool(args.out):
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
