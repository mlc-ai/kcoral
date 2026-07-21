from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any


class EventLogger:
    def __init__(self, log_root: Path) -> None:
        parent = log_root.resolve() / "runs"
        parent.mkdir(parents=True, exist_ok=True)
        base = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        suffix = 0
        while True:
            name = base if suffix == 0 else f"{base}-{suffix}"
            run_dir = parent / name
            try:
                run_dir.mkdir()
                break
            except FileExistsError:
                suffix += 1
        self.run_dir = run_dir
        self.server_run = name
        self.requests_dir = run_dir / "requests"
        self.requests_dir.mkdir()
        (run_dir / "server.stdout.log").touch()
        (run_dir / "server.stderr.log").touch()
        self._stream = (run_dir / "events.jsonl").open("a", encoding="utf-8")
        self._lock = Lock()

    def request_paths(self, request_id: str) -> tuple[Path, Path]:
        request_dir = self.requests_dir / request_id
        request_dir.mkdir(parents=True, exist_ok=True)
        stdout = request_dir / "stdout.log"
        stderr = request_dir / "stderr.log"
        stdout.touch()
        stderr.touch()
        return stdout, stderr

    def emit(self, event: str, *, level: str = "INFO", **fields: Any) -> None:
        payload = {
            "timestamp": datetime.now(timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "level": level,
            "event": event,
            "server_run": self.server_run,
            **fields,
        }
        line = (
            json.dumps(
                payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            )
            + "\n"
        )
        with self._lock:
            self._stream.write(line)
            self._stream.flush()

    def close(self) -> None:
        self._stream.close()
