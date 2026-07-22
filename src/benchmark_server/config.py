"""Server configuration (limits and scheduling)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ServerConfig:
    gpus: list[int] = field(default_factory=lambda: [0])
    cache_capacity_bytes: int = 16 * 1024**3          # 16 GB byte cache
    default_timeout_seconds: float = 300.0            # per-request execution timeout
    max_timeout_seconds: float = 3600.0
    worker_wait_timeout_seconds: float = 30.0         # wait for a free worker before 503
    max_request_bytes: int = 256 * 1024**2            # 256 MB request cap
