"""Launch native services in place, preserving signals and exit status."""

from __future__ import annotations

import argparse
import ipaddress
import math
import os
import shutil
import socket
import sys
from pathlib import Path

from ._server import build_parser, config_from_args, server_components

# These options configure the supervisor rather than the Python worker service.
_SUPERVISOR_DEFAULTS = {
    "health-interval-seconds": 2.0,
    "health-timeout-seconds": 1.0,
    "failure-threshold": 3,
    "startup-grace-seconds": 30.0,
    "stable-reset-seconds": 60.0,
    "termination-grace-seconds": 5.0,
    "restart-min-delay-seconds": 1.0,
    "restart-max-delay-seconds": 30.0,
    "restart-jitter": 0.2,
}


def exec_service(name: str, arguments: list[str], *, env: dict[str, str] | None = None) -> None:
    # Prefer helpers installed alongside this interpreter to another environment's PATH.
    search_path = os.pathsep.join([str(Path(sys.executable).parent), os.environ.get("PATH", "")])
    executable = shutil.which(name, path=search_path)
    if executable is None:
        raise SystemExit(
            f"{name} is not installed. From the matching KCoral source checkout, run "
            f"`cargo install --locked --path rust/kcoral --root {sys.prefix}`. "
            "See the server installation guide."
        )
    try:
        os.execve(executable, [executable, *arguments], dict(os.environ) if env is None else env)
    except OSError as exc:
        raise SystemExit(f"cannot start {name}: {exc}") from exc


def _health_origin(host: str, port: int) -> str:
    # Probes originate on loopback, including when the service binds a specific
    # local interface. Wildcard addresses are never used as destinations.
    addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    address = ipaddress.ip_address(addresses[0][4][0])
    if address.is_unspecified:
        address = ipaddress.ip_address("::1" if address.version == 6 else "127.0.0.1")
    host = f"[{address}]" if address.version == 6 else str(address)
    return f"http://{host}:{port}/"


def server_parser() -> argparse.ArgumentParser:
    parser = build_parser()
    parser.description = "Start, health-check and restart a local KCoral execution service."
    supervision = parser.add_argument_group("process supervision")
    for name, default in _SUPERVISOR_DEFAULTS.items():
        supervision.add_argument(
            "--" + name,
            type=int if name == "failure-threshold" else float,
            default=default,
            help=f"supervisor {name.replace('-', ' ')} (default: {default})",
        )
    return parser


def _validate_supervision(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    nonnegative = {"startup-grace-seconds", "stable-reset-seconds", "termination-grace-seconds"}
    for name in _SUPERVISOR_DEFAULTS:
        value = getattr(args, name.replace("-", "_"))
        if not math.isfinite(value) or value < 0:
            parser.error(f"--{name} must be finite and non-negative")
        if value == 0 and name not in nonnegative | {"restart-jitter"}:
            parser.error(f"--{name} must be positive")
    if args.restart_min_delay_seconds > args.restart_max_delay_seconds:
        parser.error("--restart-min-delay-seconds must not exceed --restart-max-delay-seconds")
    if args.restart_jitter > 0.5:
        parser.error("--restart-jitter must be between 0 and 0.5")


def server_main(argv: list[str]) -> None:
    parser = server_parser()
    args = parser.parse_args(argv)
    config_from_args(args)
    _validate_supervision(parser, args)
    server_components()
    try:
        origin = _health_origin(args.host, args.port)
    except (OSError, ValueError) as exc:
        parser.error(f"cannot resolve --host {args.host!r}: {exc}")

    native_args = ["--server-url", origin]
    if args.router_endpoint:
        native_args += ["--router-endpoint", args.router_endpoint, "--node-id", args.node_id]
    for name in _SUPERVISOR_DEFAULTS:
        native_args += ["--" + name, str(getattr(args, name.replace("-", "_")))]

    # Render the parsed worker options once. This handles --option=value, repeated
    # arguments and explicit values overriding environment defaults without shell quoting.
    worker_args = []
    for action in build_parser()._actions:
        if action.dest in {"help", "router_endpoint", "node_id", "node_token"}:
            continue
        value = getattr(args, action.dest)
        option = action.option_strings[0]
        if isinstance(action, argparse._StoreFalseAction):
            if not value:
                worker_args.append(option)
        elif isinstance(action, argparse._AppendAction):
            worker_args.extend(f"{option}={item}" for item in value)
        elif value is not None:
            worker_args.append(f"{option}={value}")

    # The supervisor owns routing defaults. Tokens travel in the environment,
    # not command lines; the child uses this exact Python installation.
    env = dict(os.environ)
    for name in ("KCORAL_ROUTER_ENDPOINT", "KCORAL_NODE_ID", "KCORAL_NODE_TOKEN"):
        env.pop(name, None)
    if args.node_token is not None:
        env["KCORAL_NODE_TOKEN"] = args.node_token
    native_args += ["--", sys.executable, "-m", "kcoral._server", *worker_args]
    exec_service("kcoral-node", native_args, env=env)
