"""Run KCoral: ``kcoral`` or ``python -m kcoral``.

The front-end process itself touches no GPU. GPU workers import torch/tvm and
are pinned to one GPU, which ``--workers-per-gpu`` of them share by taking turns
through its lease. CPU workers need only TVM FFI and a CUDA toolchain, and
compile uploaded source without importing a GPU runtime.

    kcoral --gpus 1,2,3
    kcoral --device cpu --num-workers 16

Every ``ServerConfig`` field has a flag (see ``kcoral --help``). A few
flags default from the environment, so env-only deployments keep working:

  KCORAL_SERVER_DEVICE  worker type: gpu or cpu (default "gpu")
  KCORAL_SERVER_GPUS    comma-separated physical GPU ids the GPU workers pin (default "0")
  KCORAL_SERVER_HOST    bind host (default 127.0.0.1)
  KCORAL_SERVER_PORT    bind port (default 8000)
  KCORAL_LOG_DIR        directory for structured event logs (default "logs"; empty disables)

The log is one JSONL stream per run under ``<log dir>/runs/``, uncapped, and it
is mirrored to stderr unless ``--no-log-console`` says otherwise. Its first line
names the run directory.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .app import create_app
from .config import ServerConfig

_DEFAULTS = ServerConfig()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kcoral",
        description="KCoral: stateless kernel benchmark server (instruction protocol).",
    )
    parser.add_argument("--host", default=os.environ.get("KCORAL_SERVER_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("KCORAL_SERVER_PORT", "8000"))
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
    return ServerConfig(
        device=args.device,
        gpus=gpus,
        num_workers=args.num_workers,
        cache_capacity_bytes=args.cache_capacity_bytes,
        log_dir=Path(args.log_dir) if args.log_dir else None,
        log_console=args.log_console,
        log_programs=args.log_programs,
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
    if config.device == "gpu":
        _warn_if_visible_devices_set()
    app = create_app(config)
    uvicorn.run(app, host=args.host, port=args.port)


def _warn_if_visible_devices_set() -> None:
    """A worker overwrites it with the card ``--gpus`` gave it, so setting it
    here selects nothing and only misleads."""
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        print(
            "warning: CUDA_VISIBLE_DEVICES is set but does not restrict this server; "
            "workers are pinned by --gpus (physical GPU ids). Unset it to avoid confusion.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
