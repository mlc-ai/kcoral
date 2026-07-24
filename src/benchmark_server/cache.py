"""Content-addressed byte cache, persisted on disk (front-end).

Objects live under ``<cache_dir>/objects/<shard>/<name>`` and survive server
restarts; an in-memory index tracks sizes, last use, and per-request pins
(refcount > 0) so a concurrent request cannot evict entries mid-flight. LRU +
a byte budget, as before; last-use order is approximated across restarts from
file mtimes. A ``.lock`` file (flock) prevents two server processes from
sharing one directory. Without an explicit ``cache_dir`` the cache uses a
private temporary directory (nothing persists), which keeps embedded and test
use zero-config. The cache stores canonical bytes only; it is a transparent
optimization — the front-end executes inline uploads even when the cache
declines to store them.
"""

from __future__ import annotations

import base64
import fcntl
import hashlib
import os
import re
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

_SHA256_KEY_RE = re.compile(r"^sha256:([0-9a-f]{64})$")
_RAW_NAME_PREFIX = "raw-"


@dataclass
class _Entry:
    path: Path
    size: int
    refcount: int = 0
    last_used: float = 0.0


class ByteCache:
    def __init__(
        self,
        capacity_bytes: int,
        cache_dir: Path | None = None,
        max_object_bytes: int | None = None,
    ) -> None:
        self._cap = capacity_bytes
        # Don't cache an object larger than this (default: 25% of capacity) so one
        # huge blob can't evict the whole working set. It still works, just isn't
        # cached (the front-end keeps using its inline bytes).
        self._max_obj = max_object_bytes if max_object_bytes is not None else capacity_bytes // 4
        self._owned_dir = (
            tempfile.TemporaryDirectory(prefix="benchmark-server-cache-")
            if cache_dir is None
            else None
        )
        self._root = Path(self._owned_dir.name if self._owned_dir else cache_dir).resolve()
        self._objects_dir = self._root / "objects"
        self._staging_dir = self._root / "tmp"
        self._objects_dir.mkdir(parents=True, exist_ok=True)
        self._staging_dir.mkdir(parents=True, exist_ok=True)
        self._lock_file = (self._root / ".lock").open("a+b")
        try:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._lock_file.close()
            raise RuntimeError(f"cache directory is already in use: {self._root}") from exc
        self._entries: dict[str, _Entry] = {}
        self._size = 0
        self._lock = threading.Lock()
        self._load_existing()

    def get(self, key: str) -> bytes | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            try:
                data = entry.path.read_bytes()
            except OSError:  # the file vanished under us — drop the entry
                self._size -= entry.size
                del self._entries[key]
                return None
            entry.last_used = time.time()
            try:
                os.utime(entry.path, (entry.last_used, entry.last_used))
            except OSError:
                pass
            return data

    def put(self, key: str, data: bytes) -> None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                entry.last_used = time.time()
                return
            if len(data) > self._max_obj:
                return  # too big to cache; the caller keeps using it this request
            path = self._object_path(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            staging = self._staging_dir / f"put-{uuid.uuid4()}.part"
            staging.write_bytes(data)
            os.replace(staging, path)  # atomic publish
            self._entries[key] = _Entry(path=path, size=len(data), last_used=time.time())
            self._size += len(data)
            self._evict_locked()

    def missing(self, keys: list[str]) -> list[str]:
        with self._lock:
            return [k for k in keys if k not in self._entries]

    def pin(self, keys: list[str]) -> None:
        with self._lock:
            for k in keys:
                entry = self._entries.get(k)
                if entry is not None:
                    entry.refcount += 1

    def unpin(self, keys: list[str]) -> None:
        with self._lock:
            for k in keys:
                entry = self._entries.get(k)
                if entry is not None and entry.refcount > 0:
                    entry.refcount -= 1

    def close(self) -> None:
        try:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            self._lock_file.close()
        if self._owned_dir is not None:
            self._owned_dir.cleanup()

    def _load_existing(self) -> None:
        for leftover in self._staging_dir.iterdir():  # from a crashed run
            leftover.unlink(missing_ok=True)
        for path in sorted(self._objects_dir.glob("*/*")):
            if not path.is_file():
                continue
            key = _key_from_name(path.name)
            if key is None:
                continue
            stat = path.stat()
            self._entries[key] = _Entry(path=path, size=stat.st_size, last_used=stat.st_mtime)
            self._size += stat.st_size
        with self._lock:
            self._evict_locked()

    def _object_path(self, key: str) -> Path:
        name = _name_for_key(key)
        shard = hashlib.sha256(key.encode("utf-8")).hexdigest()[:2]
        return self._objects_dir / shard / name

    def _evict_locked(self) -> None:
        # Drop least-recently-used entries with refcount == 0 until under budget.
        # If everything left is pinned, temporarily exceed the budget.
        if self._size <= self._cap:
            return
        for key, entry in sorted(self._entries.items(), key=lambda item: item[1].last_used):
            if self._size <= self._cap:
                break
            if entry.refcount == 0:
                entry.path.unlink(missing_ok=True)
                self._size -= entry.size
                del self._entries[key]

    # introspection (tests)
    @property
    def size_bytes(self) -> int:
        return self._size

    def __contains__(self, key: str) -> bool:
        return key in self._entries


def _name_for_key(key: str) -> str:
    """A safe, reversible file name: canonical sha256 keys keep their hex digest
    (inspectable on disk); anything else is URL-safe base64 of the key."""
    match = _SHA256_KEY_RE.match(key)
    if match:
        return match.group(1)
    return _RAW_NAME_PREFIX + base64.urlsafe_b64encode(key.encode("utf-8")).decode("ascii")


def _key_from_name(name: str) -> str | None:
    if name.startswith(_RAW_NAME_PREFIX):
        try:
            return base64.urlsafe_b64decode(name[len(_RAW_NAME_PREFIX) :]).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            return None
    if re.fullmatch(r"[0-9a-f]{64}", name):
        return "sha256:" + name
    return None
