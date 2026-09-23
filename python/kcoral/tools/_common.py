"""Shared tool arguments, file transfers, and local and remote execution."""

from __future__ import annotations

import argparse
import io
import ipaddress
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath


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
        "--output-limit-mbytes",
        type=float,
        default=256,
        help="capture limit per stream in MiB (1024**2 bytes; default: 256; server caps apply)",
    )


def validate_connection_args(parser, args):
    from ..config import mbytes_to_bytes

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
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    try:
        mbytes_to_bytes(args.output_limit_mbytes, "--output-limit-mbytes", positive=True)
    except ValueError as exc:
        parser.error(str(exc))


def execute(args, program):
    from ..client import Client

    with Client(args.url) as client:
        result = client.execute(
            program, timeout_seconds=args.timeout, output_limit_mbytes=args.output_limit_mbytes
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


def parse_args(tool, argv, *, validate=None, profiling=False, allow_out=False, epilog=None):
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
        help="upload a file or directory, preserving its name; repeatable",
    )
    parser.add_argument(
        "-e",
        "--env",
        action="append",
        default=[],
        metavar="NAME[=VALUE]",
        help="set a remote environment variable; NAME copies its local value",
    )
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
    if allow_out and args.fetch and args.out is None:
        parser.error("--fetch requires --out")
    if not allow_out and not profiling and bool(args.fetch) != bool(args.out):
        parser.error("--fetch and --out must be used together")
    for name in args.fetch:
        try:
            relative_path(name)
        except ValueError as exc:
            parser.error(str(exc))
    if validate is not None:
        validate(parser, forwarded)
    return args, forwarded


def validate_profiler_args(parser, forwarded, *, managed, short, subcommand=None):
    if "--" not in forwarded:
        parser.error("separate profiler options from the application with another '--'")
    split = forwarded.index("--")
    options = forwarded[:split]
    if not forwarded[split + 1 :] or (subcommand is not None and subcommand not in options):
        parser.error("expected profiler options followed by '-- application [args]'")
    for token in options:
        option = token.split("=", 1)[0]
        if (option.startswith("--") and any(flag.startswith(option) for flag in managed)) or (
            not option.startswith("--") and any(option.startswith(flag) for flag in short)
        ):
            parser.error(f"{option} conflicts with managed remote capture paths or mode")


def build_program(args, tool, forwarded):
    from ..client import Program
    from . import COMMANDS

    program = Program()
    module = program.upload(
        id="tool_runner",
        kind="module",
        source=Path(__file__).read_text(),
    )
    runner = program.get_function(id="runner", module=module, name="run")
    definition = program.upload(
        id="definition",
        kind="module",
        source=Path(__file__).with_name(COMMANDS[tool] + ".py").read_text(),
    )
    tool_run = program.get_function(id="tool_run", module=definition, name="run")
    inputs = program.upload(id="inputs", kind="bytes", value=pack_inputs(args.send))
    outcome = program.run(
        id="run", fn=runner, args=[inputs, tool_run, forwarded, args.env, args.fetch]
    )
    # Preserve the exit status even when a subsequent artifact collection fails.
    program.return_(key="outcome", value=outcome)
    if args.out is not None:
        program.return_folder(key="artifacts", path="outputs")
    return program


def run_tool(tool, args, forwarded):
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


def relative_path(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name or not path.parts:
        raise ValueError(f"expected a relative file path: {name!r}")
    return path


def pack_inputs(paths=()):
    """Stable archives: preserve each selected file or directory's basename."""
    buffer, names = io.BytesIO(), set()
    with tarfile.open(fileobj=buffer, mode="w") as tar:

        def add(name, data, executable=False):
            name = relative_path(name).as_posix()
            if name in names or any(
                name.startswith(old + "/") or old.startswith(name + "/") for old in names
            ):
                raise ValueError(f"duplicate or conflicting input path: {name}")
            names.add(name)
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o700 if executable else 0o600
            tar.addfile(info, io.BytesIO(data))

        for path in paths:
            if path.is_symlink() or not path.exists():
                raise ValueError(f"input must exist and not be a symlink: {path}")
            directory = path.is_dir()
            basename = path.resolve().name if directory else path.name
            if not basename:
                raise ValueError(f"input directory must have a name: {path}")
            for entry in sorted(path.rglob("*")) if directory else [path]:
                if "__pycache__" in entry.parts:
                    continue
                if entry.is_symlink():
                    raise ValueError(f"symlink inputs are not supported: {entry}")
                if entry.is_dir():
                    continue
                if not entry.is_file():
                    raise ValueError(f"input is not a regular file: {entry}")
                name = f"{basename}/{entry.relative_to(path).as_posix()}" if directory else basename
                add(name, entry.read_bytes(), bool(entry.stat().st_mode & 0o111))
    return buffer.getvalue()


def unpack_inputs(archive, workdir):
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
        for member in tar:
            path = relative_path(member.name)
            if not member.isfile():
                raise ValueError(f"input is not a regular file: {member.name!r}")
            destination = workdir.joinpath(*path.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, destination.open("xb") as target:
                shutil.copyfileobj(source, target)
            destination.chmod(0o700 if member.mode & 0o111 else 0o600)


def collect_files(workdir, reports, paths):
    missing = []
    for name in paths:
        source = workdir.joinpath(*relative_path(name).parts)
        # Check every ancestor before traversing or reading the selection.
        current = workdir
        for part in source.relative_to(workdir).parts:
            current /= part
            if current.is_symlink():
                raise ValueError(f"artifact is a symlink: {current}")
        if not source.exists():
            missing.append(name)
            continue
        entries = [source, *sorted(source.rglob("*"))] if source.is_dir() else [source]
        for entry in entries:
            mode = entry.lstat().st_mode
            destination = reports / entry.relative_to(workdir)
            if stat.S_ISDIR(mode):
                destination.mkdir(parents=True, exist_ok=True)
            elif stat.S_ISREG(mode):
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(entry, destination)
            else:
                raise ValueError(f"artifact is not a regular file or directory: {entry}")
    return missing


def run(archive, tool_run, arguments, overrides, fetch):
    workdir = Path("inputs").absolute()
    reports = Path("outputs").absolute()
    workdir.mkdir()
    unpack_inputs(archive, workdir)
    env = {**os.environ, **overrides, "KCORAL_DIR": str(workdir)}
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")

    def execute(command, *, create_reports=True):
        if create_reports:
            reports.mkdir(exist_ok=True)
        target = command[0]
        if "/" in target and not os.path.isabs(target):
            target = str(workdir / target)
        executable = shutil.which(target, path=env["PATH"])
        if executable is None:
            raise RuntimeError(f"{command[0]} is not installed in the remote server environment")
        # Stay in the worker's process group so its timeout also kills subprocesses.
        return subprocess.run(
            [executable, *command[1:]],
            cwd=workdir,
            env=env,
            stdin=subprocess.DEVNULL,
        ).returncode

    outcome = tool_run(arguments, env, reports, execute)
    reports.mkdir(exist_ok=True)
    outcome["missing"].extend(collect_files(workdir, reports, fetch))
    return outcome
