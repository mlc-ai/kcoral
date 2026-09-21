"""Start standalone Python servers, supervised nodes and Rust routers."""

from __future__ import annotations

import argparse
import ipaddress
import os
import shutil
import socket
import sys
from pathlib import Path

from ._server import build_parser, config_from_args, serve, server_components

# Advanced routed-node options. Rust owns their types, defaults and validation.
_SUPERVISOR_OPTIONS = (
    "health-interval-seconds",
    "health-timeout-seconds",
    "failure-threshold",
    "startup-grace-seconds",
    "stable-reset-seconds",
    "termination-grace-seconds",
    "restart-min-delay-seconds",
    "restart-max-delay-seconds",
    "restart-jitter",
)


def exec_native_binary(
    name: str, arguments: list[str], *, env: dict[str, str] | None = None
) -> None:
    """Replace this process with a Rust executable, preserving signals and exit status."""
    search_path = os.pathsep.join([str(Path(sys.executable).parent), os.environ.get("PATH", "")])
    executable = shutil.which(name, path=search_path)
    if executable is None:
        raise SystemExit(
            f"{name} was not found. Build it with `cargo build --release --locked` "
            "and add target/release to PATH. See the Router deployment guide."
        )
    try:
        os.execve(executable, [executable, *arguments], dict(os.environ) if env is None else env)
    except OSError as exc:
        raise SystemExit(f"cannot start {name}: {exc}") from exc


def router_main(argv: list[str]) -> None:
    exec_native_binary("kcoral-router", argv)


def _health_origin(host: str, port: int) -> str:
    # Probes originate on loopback, including for a specific local interface.
    # Wildcard addresses are never used as destinations.
    addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    address = ipaddress.ip_address(addresses[0][4][0])
    if address.is_unspecified:
        address = ipaddress.ip_address("::1" if address.version == 6 else "127.0.0.1")
    host = f"[{address}]" if address.version == 6 else str(address)
    return f"http://{host}:{port}/"


def _add_supervisor_options(parser: argparse.ArgumentParser) -> None:
    for name in _SUPERVISOR_OPTIONS:
        parser.add_argument("--" + name, help=argparse.SUPPRESS)


def server_parser() -> argparse.ArgumentParser:
    parser = build_parser()
    parser.description = "Start a KCoral server; add --router and --node-id to join a Router."
    _add_supervisor_options(parser)
    return parser


def _python_arguments(argv: list[str]) -> list[str]:
    """Remove launcher options while preserving the original Python server arguments."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    for name in ("router", "node-id", "node-token"):
        parser.add_argument("--" + name)
    _add_supervisor_options(parser)
    _, arguments = parser.parse_known_args(argv)
    return arguments


def server_main(argv: list[str]) -> None:
    parser = server_parser()
    args = parser.parse_args(argv)
    config_from_args(args)
    if not args.router_endpoint:
        for name in _SUPERVISOR_OPTIONS:
            if getattr(args, name.replace("-", "_")) is not None:
                parser.error(f"--{name} requires --router")
        serve(args)
        return
    server_components()
    try:
        origin = _health_origin(args.host, args.port)
    except (OSError, ValueError) as exc:
        parser.error(f"cannot resolve --host {args.host!r}: {exc}")

    node_args = [
        "--server-url",
        origin,
        "--router-endpoint",
        args.router_endpoint,
        "--node-id",
        args.node_id,
    ]
    for name in _SUPERVISOR_OPTIONS:
        value = getattr(args, name.replace("-", "_"))
        if value is not None:
            node_args.append(f"--{name}={value}")

    worker_args = _python_arguments(argv)
    # The supervisor derives child defaults from the probe URL. Preserve the
    # requested bind address, including environment defaults and wildcards.
    worker_args += [f"--host={args.host}", f"--port={args.port}"]
    env = dict(os.environ)
    for name in ("KCORAL_ROUTER_ENDPOINT", "KCORAL_NODE_ID", "KCORAL_NODE_TOKEN"):
        env.pop(name, None)
    if args.node_token is not None:
        env["KCORAL_NODE_TOKEN"] = args.node_token
    node_args += ["--", sys.executable, "-m", "kcoral._server", *worker_args]
    exec_native_binary("kcoral-node", node_args, env=env)
