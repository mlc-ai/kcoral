"""Content-addressed memory and disk caches (front-end).

The memory cache pins referenced keys while requests run. File uploads use
only the disk cache; requests retain their own bytes, independently of eviction.
Both caches are transparent optimizations over content-addressed uploads.
"""

from __future__ import annotations

import fcntl
import os
import stat
import threading
import uuid
from collections import OrderedDict
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

from .keys import compute_blob_hash, is_blob_hash, verify_blob


@dataclass
class _Entry:
    data: bytes
    refcount: int = 0


class ByteCache:
    def __init__(self, capacity_bytes: int, max_object_bytes: int | None = None) -> None:
        self._cap = capacity_bytes
        # Don't cache an object larger than this (default: 25% of capacity) so one
        # huge blob can't evict the whole working set. It still works, just isn't
        # cached (re-uploaded each time).
        self._max_obj = max_object_bytes if max_object_bytes is not None else capacity_bytes // 4
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._size = 0
        self._lock = threading.Lock()

    def get(self, key: str) -> bytes | None:
        with self._lock:
            e = self._entries.get(key)
            if e is None:
                return None
            self._entries.move_to_end(key)  # most-recently-used
            return e.data

    def put(self, key: str, data: bytes) -> None:
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                return
            if len(data) > self._max_obj:
                return  # too big to cache; caller keeps using it this request
            self._entries[key] = _Entry(data=data)
            self._size += len(data)
            self._evict_locked()

    def pin(self, keys: list[str]) -> None:
        with self._lock:
            for k in keys:
                e = self._entries.get(k)
                if e is not None:
                    e.refcount += 1

    def unpin(self, keys: list[str]) -> None:
        with self._lock:
            for k in keys:
                e = self._entries.get(k)
                if e is not None and e.refcount > 0:
                    e.refcount -= 1

    def _evict_locked(self) -> None:
        # Drop least-recently-used entries with refcount == 0 until under budget.
        # If everything left is pinned, temporarily exceed the budget.
        for key in list(self._entries.keys()):
            if self._size <= self._cap:
                break
            e = self._entries[key]
            if e.refcount == 0:
                del self._entries[key]
                self._size -= len(e.data)

    def __contains__(self, key: str) -> bool:
        return key in self._entries


class DiskFileCache:
    """Disposable file content, with the filesystem as its only persistent index.

    Keys map to ``<directory>/v1/<first two hex digits>/<sha256>``. Explicitly
    updated mtimes approximate LRU. A scan at startup and after each write batch
    enforces the byte budget; no index, config, or lock file is stored. Writers
    and eviction take an advisory lock on the version directory itself, shared
    by all instances/processes using that directory.

    Readers open immutable objects and return independent bytes, so eviction
    needs no pins and cannot invalidate an accepted request. Disk errors are
    cache misses/skipped writes, never a reason to reject supplied request data.
    """

    _DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW

    def __init__(self, directory: Path | None, capacity_bytes: int) -> None:
        if capacity_bytes < 0:
            raise ValueError("disk cache capacity must be non-negative")
        self.directory = (
            Path(directory).expanduser().absolute() / "v1"
            if directory is not None and capacity_bytes > 0
            else None
        )
        self._cap = capacity_bytes
        if self.directory is not None:
            with suppress(OSError), self._writer() as root:
                self._evict(root)

    def get(self, key: str) -> bytes | None:
        if self.directory is None or not is_blob_hash(key):
            return None
        try:
            root = os.open(self.directory, self._DIRECTORY_FLAGS)
            try:
                shard = os.open(key[:2], self._DIRECTORY_FLAGS, dir_fd=root)
                try:
                    fd = os.open(key, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=shard)
                finally:
                    os.close(shard)
            finally:
                os.close(root)
            with os.fdopen(fd, "rb") as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    return None
                data = stream.read()
                if compute_blob_hash(data) != key:
                    return None
                # Use mtime rather than atime: mounts often disable atime updates.
                with suppress(OSError):
                    os.utime(stream.fileno(), None)
            return data
        except OSError:
            return None

    def put(self, key: str, data: bytes) -> None:
        self.put_many({key: data})

    def put_many(self, blobs: Mapping[str, bytes]) -> None:
        if self.directory is None or not blobs:
            return
        for key, data in blobs.items():
            verify_blob(key, data)
        with suppress(OSError), self._writer() as root:
            for key, data in blobs.items():
                if len(data) > self._cap:
                    continue
                with suppress(OSError):
                    self._publish(root, key, data)
            self._evict(root)

    @contextmanager
    def _writer(self) -> Iterator[int]:
        assert self.directory is not None
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        root = os.open(self.directory, self._DIRECTORY_FLAGS)
        try:
            # Each operation opens a new descriptor, so flock also serializes
            # independent threads in one process, not only separate processes.
            fcntl.flock(root, fcntl.LOCK_EX)
            yield root
        finally:
            os.close(root)

    def _publish(self, root: int, key: str, data: bytes) -> None:
        with suppress(FileExistsError):
            os.mkdir(key[:2], mode=0o700, dir_fd=root)
        shard = os.open(key[:2], self._DIRECTORY_FLAGS, dir_fd=root)
        temporary = f".tmp-{uuid.uuid4().hex}"
        try:
            fd = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=shard,
            )
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
            os.replace(temporary, key, src_dir_fd=shard, dst_dir_fd=shard)
        finally:
            with suppress(OSError):
                os.unlink(temporary, dir_fd=shard)
            os.close(shard)

    def _evict(self, root: int) -> None:
        entries = []
        for prefix in os.listdir(root):
            if len(prefix) != 2 or any(char not in "0123456789abcdef" for char in prefix):
                continue
            with suppress(OSError):
                shard = os.open(prefix, self._DIRECTORY_FLAGS, dir_fd=root)
                try:
                    for name in os.listdir(shard):
                        # The writer lock means no live writer owns these files.
                        # This also cleans up interrupted writes after a restart.
                        if name.startswith(".tmp-"):
                            with suppress(OSError):
                                os.unlink(name, dir_fd=shard)
                        elif is_blob_hash(name) and name.startswith(prefix):
                            info = os.stat(name, dir_fd=shard, follow_symlinks=False)
                            if stat.S_ISREG(info.st_mode):
                                entries.append((info.st_mtime_ns, prefix, name, info.st_size))
                finally:
                    os.close(shard)
        size = sum(entry[3] for entry in entries)
        for _, prefix, name, entry_size in sorted(entries):
            if size <= self._cap:
                break
            with suppress(OSError):
                shard = os.open(prefix, self._DIRECTORY_FLAGS, dir_fd=root)
                try:
                    os.unlink(name, dir_fd=shard)
                    size -= entry_size
                finally:
                    os.close(shard)
