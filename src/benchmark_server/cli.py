from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from .models import ServerConfig
from .server import create_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="benchmark-server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--cache-capacity-bytes", type=int, required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("work"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--default-timeout-seconds", type=float, default=60)
    parser.add_argument("--max-timeout-seconds", type=float, default=3600)
    parser.add_argument("--default-stdout-limit-bytes", type=int, default=1048576)
    parser.add_argument("--default-stderr-limit-bytes", type=int, default=1048576)
    parser.add_argument("--max-stdout-limit-bytes", type=int, default=16777216)
    parser.add_argument("--max-stderr-limit-bytes", type=int, default=16777216)
    parser.add_argument("--worker-termination-grace-seconds", type=float, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535")
    config = ServerConfig(
        devices=tuple(part.strip() for part in args.devices.split(",") if part.strip()),
        cache_dir=args.cache_dir,
        cache_capacity_bytes=args.cache_capacity_bytes,
        work_dir=args.work_dir,
        log_dir=args.log_dir,
        default_timeout_seconds=args.default_timeout_seconds,
        max_timeout_seconds=args.max_timeout_seconds,
        default_stdout_limit_bytes=args.default_stdout_limit_bytes,
        default_stderr_limit_bytes=args.default_stderr_limit_bytes,
        max_stdout_limit_bytes=args.max_stdout_limit_bytes,
        max_stderr_limit_bytes=args.max_stderr_limit_bytes,
        worker_termination_grace_seconds=args.worker_termination_grace_seconds,
    )
    try:
        config.validated()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    uvicorn.run(create_app(config), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
