"""Run the benchmark server: ``benchmark-server`` or ``python -m benchmark_server``.

Launch it where a TIRX-enabled tvm is importable — either pip-installed
(``pip install apache-tvm``) or a from-source build (put its Python tree on
``PYTHONPATH`` and point ``TVM_LIBRARY_PATH`` at the built library directory). The
front-end process itself touches no GPU; each worker process imports torch/tvm
and is pinned to one GPU, which ``--workers-per-gpu`` of them share by taking
turns through its lease.

    benchmark-server --gpus 1,2,3

Every ``ServerConfig`` field has a flag (see ``benchmark-server --help``). A few
flags default from the environment, so env-only deployments keep working:

  BENCH_GPUS       comma-separated physical GPU ids the workers pin (default "0")
  BENCH_HOST       bind host (default 127.0.0.1)
  BENCH_PORT       bind port (default 8000)
  BENCH_LOG_DIR    directory for structured event logs (default "logs"; empty disables)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .app import create_app
from .config import ServerConfig
from .gpu_runtime import gpu_runtime_factory

_DEFAULTS = ServerConfig()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmark-server",
        description="Stateless GPU kernel benchmark server (instruction protocol).",
    )
    parser.add_argument("--host", default=os.environ.get("BENCH_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("BENCH_PORT", "8000")))
    parser.add_argument(
        "--gpus",
        default=os.environ.get("BENCH_GPUS", "0"),
        help="comma-separated physical GPU ids the workers pin (default: 0)",
    )
    parser.add_argument("--cache-capacity-bytes", type=int, default=_DEFAULTS.cache_capacity_bytes)
    parser.add_argument(
        "--log-dir",
        default=os.environ.get("BENCH_LOG_DIR", "logs"),
        help="structured event log directory; empty disables logging (default: logs)",
    )
    parser.add_argument(
        "--default-timeout-seconds", type=float, default=_DEFAULTS.default_timeout_seconds
    )
    parser.add_argument("--max-timeout-seconds", type=float, default=_DEFAULTS.max_timeout_seconds)
    parser.add_argument(
        "--worker-wait-timeout-seconds",
        type=float,
        default=_DEFAULTS.worker_wait_timeout_seconds,
        help="how long a request waits for a free GPU worker before 503",
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
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    if not gpus:
        raise SystemExit("--gpus needs at least one GPU id")
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535")
    if args.max_requests_per_worker < 0:
        raise SystemExit("--max-requests-per-worker must be non-negative")
    return ServerConfig(
        gpus=gpus,
        cache_capacity_bytes=args.cache_capacity_bytes,
        log_dir=Path(args.log_dir) if args.log_dir else None,
        default_timeout_seconds=args.default_timeout_seconds,
        max_timeout_seconds=args.max_timeout_seconds,
        worker_wait_timeout_seconds=args.worker_wait_timeout_seconds,
        workers_per_gpu=args.workers_per_gpu,
        max_requests_per_worker=args.max_requests_per_worker,
        worker_termination_grace_seconds=args.worker_termination_grace_seconds,
        max_request_bytes=args.max_request_bytes,
        max_response_bytes=args.max_response_bytes,
        output_limit_bytes=args.output_limit_bytes,
        max_output_limit_bytes=args.max_output_limit_bytes,
    )


def main() -> None:
    import uvicorn

    args = build_parser().parse_args()
    config = config_from_args(args)
    app = create_app(config, runtime_factory=gpu_runtime_factory)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
