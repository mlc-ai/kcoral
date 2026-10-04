"""Command-line configuration and standalone or supervised service startup."""

from __future__ import annotations

import argparse
import ipaddress
import os
import shutil
import socket
import sys
from pathlib import Path

from kcoral.config import ServerConfig

_DEFAULTS = ServerConfig()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kcoral server",
        allow_abbrev=False,
        description="KCoral: stateless kernel benchmark server (instruction protocol).",
        epilog="Ctrl+C stops new work and reports remaining benchmarks while they finish.",
    )
    parser.add_argument("--host", default=os.environ.get("KCORAL_SERVER_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("KCORAL_SERVER_PORT", "8000"))
    )
    parser.add_argument(
        "--router",
        dest="router_endpoint",
        default=os.environ.get("KCORAL_ROUTER_ENDPOINT") or None,
        help="Router HTTP(S) origin for outbound gRPC data slots",
    )
    parser.add_argument(
        "--node-id",
        default=os.environ.get("KCORAL_NODE_ID") or None,
        help="stable node identifier used to register outbound data slots",
    )
    parser.add_argument(
        "--node-token",
        default=os.environ.get("KCORAL_NODE_TOKEN") or None,
        help="node bearer token; prefer KCORAL_NODE_TOKEN over this command-line flag",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "gpu"),
        default=os.environ.get("KCORAL_SERVER_DEVICE", _DEFAULTS.device),
        help="worker type: gpu or cpu (default: gpu)",
    )
    parser.add_argument(
        "--gpus",
        default=os.environ.get("KCORAL_SERVER_GPUS", "0"),
        help="comma-separated physical GPU ids; used only with --device gpu (default: 0)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=_DEFAULTS.num_workers,
        help="CPU worker processes; used only with --device cpu (default: 1)",
    )
    parser.add_argument("--cache-capacity-bytes", type=int, default=_DEFAULTS.cache_capacity_bytes)
    parser.add_argument(
        "--disk-cache-dir",
        default=str(_DEFAULTS.disk_cache_dir),
        help="persistent file cache directory; empty disables file caching",
    )
    parser.add_argument(
        "--disk-cache-capacity-mbytes",
        type=int,
        default=_DEFAULTS.disk_cache_capacity_mbytes,
        help="file cache budget in MiB (1024**2 bytes; default: 16384); 0 disables file caching",
    )
    parser.add_argument(
        "--log-dir",
        default=os.environ.get("KCORAL_LOG_DIR", "logs"),
        help="structured event log directory; empty disables logging (default: logs)",
    )
    parser.add_argument(
        "--no-log-console",
        dest="log_console",
        action="store_false",
        help="stop mirroring events to stderr; the log file is unaffected",
    )
    parser.add_argument(
        "--no-log-programs",
        dest="log_programs",
        action="store_false",
        help="stop keeping each request's program JSON beside the log",
    )
    parser.add_argument(
        "--default-timeout-seconds", type=float, default=_DEFAULTS.default_timeout_seconds
    )
    parser.add_argument("--max-timeout-seconds", type=float, default=_DEFAULTS.max_timeout_seconds)
    parser.add_argument(
        "--worker-wait-timeout-seconds",
        type=float,
        default=_DEFAULTS.worker_wait_timeout_seconds,
        help="how long a request waits for a free worker before 503",
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=_DEFAULTS.workers_per_gpu,
        help="workers sharing each GPU, so one can compile while another measures",
    )
    parser.add_argument(
        "--max-requests-per-worker",
        type=int,
        default=_DEFAULTS.max_requests_per_worker,
        help="retire a worker after this many requests; 0 allows unlimited reuse",
    )
    parser.add_argument(
        "--worker-termination-grace-seconds",
        type=float,
        default=_DEFAULTS.worker_termination_grace_seconds,
        help="SIGTERM-to-SIGKILL window when killing a hung or crashed worker",
    )
    parser.add_argument(
        "--sandbox",
        choices=("none", "bubblewrap"),
        default=_DEFAULTS.sandbox,
        help="isolate trusted workers' files with bubblewrap (Linux; requires bwrap)",
    )
    parser.add_argument(
        "--sandbox-readonly-path",
        action="append",
        type=Path,
        default=[],
        help="additional runtime dependency visible read-only to isolated workers; repeatable",
    )
    parser.add_argument("--max-request-bytes", type=int, default=_DEFAULTS.max_request_bytes)
    parser.add_argument("--max-response-bytes", type=int, default=_DEFAULTS.max_response_bytes)
    parser.add_argument(
        "--output-limit-bytes",
        type=int,
        default=_DEFAULTS.output_limit_bytes,
        help="default request-level stdout/stderr capture cap",
    )
    parser.add_argument(
        "--max-output-limit-bytes", type=int, default=_DEFAULTS.max_output_limit_bytes
    )
    return parser


def config_from_args(args: argparse.Namespace) -> ServerConfig:
    if args.device == "cpu":
        gpus = []
    else:
        gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
        if not gpus:
            raise SystemExit("--gpus needs at least one GPU id")
    if args.num_workers < 1:
        raise SystemExit("--num-workers must be at least 1")
    if args.workers_per_gpu < 1:
        raise SystemExit("--workers-per-gpu must be at least 1")
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535")
    if args.max_requests_per_worker < 0:
        raise SystemExit("--max-requests-per-worker must be non-negative")
    if args.disk_cache_capacity_mbytes < 0:
        raise SystemExit("--disk-cache-capacity-mbytes must be non-negative")
    if bool(args.router_endpoint) != bool(args.node_id):
        raise SystemExit("--router and --node-id must be configured together")
    if args.sandbox_readonly_path and args.sandbox == "none":
        raise SystemExit("--sandbox-readonly-path requires --sandbox bubblewrap")
    return ServerConfig(
        device=args.device,
        gpus=gpus,
        num_workers=args.num_workers,
        cache_capacity_bytes=args.cache_capacity_bytes,
        disk_cache_dir=Path(args.disk_cache_dir) if args.disk_cache_dir else None,
        disk_cache_capacity_mbytes=args.disk_cache_capacity_mbytes,
        log_dir=Path(args.log_dir) if args.log_dir else None,
        log_console=args.log_console,
        log_programs=args.log_programs,
        default_timeout_seconds=args.default_timeout_seconds,
        max_timeout_seconds=args.max_timeout_seconds,
        worker_wait_timeout_seconds=args.worker_wait_timeout_seconds,
        workers_per_gpu=args.workers_per_gpu,
        max_requests_per_worker=args.max_requests_per_worker,
        sandbox=args.sandbox,
        sandbox_readonly_paths=args.sandbox_readonly_path,
        worker_termination_grace_seconds=args.worker_termination_grace_seconds,
        max_request_bytes=args.max_request_bytes,
        max_response_bytes=args.max_response_bytes,
        output_limit_bytes=args.output_limit_bytes,
        max_output_limit_bytes=args.max_output_limit_bytes,
        router_endpoint=args.router_endpoint,
        node_id=args.node_id,
        node_token=args.node_token,
    )


def main(argv: list[str] | None = None) -> None:
    serve(build_parser().parse_args(argv))


def serve(args: argparse.Namespace) -> None:
    """Run the Python server in the current process."""
    config = config_from_args(args)
    if config.device == "gpu":
        _warn_if_visible_devices_set()
    uvicorn, create_app, shutdown_server = server_components()
    app = create_app(config)
    shutdown_server(uvicorn.Config(app, host=args.host, port=args.port), app).run()


def server_components():
    """Load the Python server components and report missing dependencies."""
    try:
        import uvicorn

        from kcoral.server.app import ShutdownServer, create_app
    except ImportError as exc:
        raise SystemExit(
            f"cannot start the server: {exc}. Install the front-end: pip install 'kcoral[server]'"
        ) from exc
    return uvicorn, create_app, ShutdownServer


def _warn_if_visible_devices_set() -> None:
    """A worker overwrites it with the card ``--gpus`` gave it, so setting it
    here selects nothing and only misleads."""
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        print(
            "warning: CUDA_VISIBLE_DEVICES is set but does not restrict this server; "
            "workers are pinned by --gpus (physical GPU ids). Unset it to avoid confusion.",
            file=sys.stderr,
        )


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
        if sys.platform != "linux":
            raise SystemExit(
                f"{name} was not found. Router deployments (`kcoral router` and "
                "`kcoral server --router`) run only on Linux."
            )
        raise SystemExit(
            f"{name} was not found. Reinstall KCoral with KCORAL_BUILD_RUST=1, or build it "
            "with `cargo build --release --locked` and add target/release to PATH. See the "
            "Router deployment guide."
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
    node_args += ["--", sys.executable, "-m", "kcoral.server.cli", *worker_args]
    exec_native_binary("kcoral-node", node_args, env=env)


if __name__ == "__main__":
    main()
