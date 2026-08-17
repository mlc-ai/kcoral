"""Structured event log: one JSONL stream per server run.

Each server start creates ``<log_dir>/runs/<utc timestamp>/events.jsonl`` and
appends one JSON object per event (server lifecycle, request lifecycle, worker
restarts, and GPU-access violations). Constructing with ``log_dir=None`` disables
logging: ``emit`` becomes a no-op, so call sites never need to branch.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO


class EventLogger:
    def __init__(self, log_dir: Path | None) -> None:
        self._lock = threading.Lock()
        self._stream: TextIO | None = None
        self.run_dir: Path | None = None
        if log_dir is None:
            return
        runs_dir = Path(log_dir).resolve() / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        base = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        suffix = 0
        while True:  # a unique directory per server run
            name = base if suffix == 0 else f"{base}-{suffix}"
            run_dir = runs_dir / name
            try:
                run_dir.mkdir()
                break
            except FileExistsError:
                suffix += 1
        self.run_dir = run_dir
        self._stream = (run_dir / "events.jsonl").open("a", encoding="utf-8")

    def emit(self, event: str, *, level: str = "INFO", **fields: Any) -> None:
        if self._stream is None:
            return
        payload = {
            "timestamp": datetime.now(timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "level": level,
            "event": event,
            **fields,
        }
        # Logging is best-effort: a failed serialization or write (e.g. disk
        # full) must never fail the request being served.
        try:
            line = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            with self._lock:
                self._stream.write(line + "\n")
                self._stream.flush()
        except (OSError, ValueError):
            pass

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None
