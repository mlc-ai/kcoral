"""Server configuration (limits and scheduling)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _default_disk_cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME")
    return (Path(base) if base else Path.home() / ".cache") / "kcoral" / "files"


@dataclass(frozen=True)
class ServerConfig:
    gpus: list[int] = field(default_factory=lambda: [0])
    log_dir: Path | None = None  # structured event logs; None disables logging
    log_console: bool = True  # mirror events to stderr as well as the log file
    log_programs: bool = True  # keep each request's program JSON beside the log
    cache_capacity_bytes: int = 16 * 1024**3  # 16 GB byte cache
    disk_cache_dir: Path | None = field(default_factory=_default_disk_cache_dir)
    disk_cache_capacity_mbytes: int = 16 * 1024  # MiB (1024**2 bytes); 0 disables file caching
    default_timeout_seconds: float = 300.0  # per-request execution timeout
    max_timeout_seconds: float = 3600.0
    worker_wait_timeout_seconds: float = 30.0  # wait for a free worker before 503
    workers_per_gpu: int = 8
    max_requests_per_worker: int = 1  # fresh process/context per request; 0 = unlimited
    worker_termination_grace_seconds: float = 5.0  # SIGTERM-to-SIGKILL window on kill
    max_request_bytes: int = 256 * 1024**2  # 256 MB request cap
    max_response_bytes: int = 256 * 1024**2  # cap on the serialized results payload
    output_limit_bytes: int = 1024**2  # per-request stdout/stderr capture cap
    max_output_limit_bytes: int = 16 * 1024**2  # cap on a client-requested limit
    device: str = "gpu"
    num_workers: int = 1  # CPU workers; ignored in GPU mode
