"""Structured event log: one JSONL stream per server run, optionally mirrored
to the console.

Each server start creates ``<log_dir>/runs/<utc timestamp>/events.jsonl`` and
appends one JSON object per event - server lifecycle, request lifecycle, worker
lifecycle. Constructing with ``log_dir=None`` and ``console=False`` disables
logging: ``emit`` becomes a no-op, so call sites never need to branch.

Only the front-end process writes here: it routes every request and supervises
every worker, so one ordered stream covers both, with no per-worker file to join
against.
"""

from __future__ import annotations

import json
import sys
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from kcoral.schemas import FileUpload, Program, Upload

# Kept on disk but left off the console line, where they would drown the rest.
_CONSOLE_SKIP = frozenset({"traceback", "versions", "config", "ops", "uploads"})
_CONSOLE_VALUE_LIMIT = 120
_UNENCODABLE_LIMIT = 2000


class EventLogger:
    def __init__(self, log_dir: Path | None, *, console: bool = False) -> None:
        self._lock = threading.Lock()
        self._stream: TextIO | None = None
        self._console = console
        self.run_dir: Path | None = None
        if log_dir is None:
            return
        runs_dir = Path(log_dir).expanduser().resolve() / "runs"
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

    @property
    def enabled(self) -> bool:
        """Whether an ``emit`` would go anywhere. Worth checking only where
        building the fields costs something."""
        return self._stream is not None or self._console

    def subdir(self, name: str) -> Path | None:
        """A directory beside ``events.jsonl`` for payloads too large to inline,
        or None when nothing is being written to disk."""
        if self.run_dir is None:
            return None
        path = self.run_dir / name
        path.mkdir(exist_ok=True)
        return path

    def emit(self, event: str, *, level: str = "INFO", **fields: Any) -> None:
        if self._stream is None and not self._console:
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
            line = _encode(payload)
        except (TypeError, ValueError):
            # One field that will not serialize must not cost the whole event,
            # and ``repr`` of them cannot hold the non-finite number that would
            # fail a second time.
            line = _encode(
                {
                    **{key: payload[key] for key in ("timestamp", "level", "event")},
                    "unencodable_fields": repr(fields)[:_UNENCODABLE_LIMIT],
                }
            )
        with self._lock:
            if self._stream is not None:
                try:
                    self._stream.write(line + "\n")
                    self._stream.flush()  # buffered writes die with a crashing process
                except OSError:
                    pass
            if self._console:
                try:
                    print(_console_line(payload), file=sys.stderr, flush=True)
                except (OSError, ValueError):
                    pass

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None


def _encode(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"), default=repr
    )


def _console_line(payload: dict[str, Any]) -> str:
    parts = [payload["timestamp"], f"{payload['level']:<7}", payload["event"]]
    for key, value in payload.items():
        if key in ("timestamp", "level", "event") or key in _CONSOLE_SKIP or value is None:
            continue
        text = f"{value:.1f}" if isinstance(value, float) else str(value)
        if len(text) > _CONSOLE_VALUE_LIMIT:
            text = text[:_CONSOLE_VALUE_LIMIT] + "..."
        parts.append(f"{key}={text}")
    return " ".join(parts)


def _program_shape(program: Program) -> dict[str, object]:
    """The shape of the workload, for reading the log without opening the
    program it describes."""
    ops: Counter[str] = Counter()
    uploads: Counter[str] = Counter()
    for instruction in program.instructions:
        ops[instruction.op] += 1
        if isinstance(instruction, (Upload, FileUpload)):
            kind = instruction.kind
            uploads[kind] += 1
    return {
        "instructions": len(program.instructions),
        "ops": dict(ops),
        "uploads": dict(uploads) or None,
        "blob_bytes": sum(len(data) for data in program.blob_bytes.values()) or None,
    }


def _keep_program(events: EventLogger, request_id: str, program_bytes: bytes) -> str | None:
    """Write the program beside the log and answer with its name. The bytes
    arrived over the wire, so nothing is re-serialized."""
    directory = events.subdir("programs")
    if directory is None:
        return None
    name = f"{request_id}.json"
    try:
        (directory / name).write_bytes(program_bytes)
    except OSError:
        return None  # best-effort, like every other write the log makes
    return name
