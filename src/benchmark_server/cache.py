from __future__ import annotations

import asyncio
import fcntl
import hashlib
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass
class CacheEntry:
    digest: str
    path: Path
    size: int
    refs: int = 0
    last_used: float = 0.0


class BlobCache:
    def __init__(self, root: Path, capacity: int) -> None:
        self.root = root.resolve()
        self.capacity = capacity
        self.objects = self.root / "objects"
        self.tmp = self.root / "tmp"
        self._entries: dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
        self._lock_file = None

    async def start(self) -> None:
        self.objects.mkdir(parents=True, exist_ok=True)
        self.tmp.mkdir(parents=True, exist_ok=True)
        self._lock_file = (self.root / ".lock").open("a+b")
        try:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._lock_file.close()
            self._lock_file = None
            raise RuntimeError(
                f"cache directory is already in use: {self.root}"
            ) from exc
        for child in self.tmp.iterdir():
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)
        for path in self.objects.glob("[0-9a-f][0-9a-f]/[0-9a-f]*"):
            digest = path.name
            if len(digest) == 64 and path.is_file():
                stat = path.stat()
                self._entries[digest] = CacheEntry(
                    digest, path, stat.st_size, 0, stat.st_mtime
                )
        async with self._lock:
            self._evict_locked()

    async def close(self) -> None:
        if self._lock_file is not None:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
            self._lock_file.close()
            self._lock_file = None

    def object_path(self, digest: str) -> Path:
        return self.objects / digest[:2] / digest

    @staticmethod
    def hash_path(path: Path) -> tuple[str, int]:
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
        return digest.hexdigest(), size

    async def check(self, digests: list[str]) -> list[str]:
        async with self._lock:
            return [digest for digest in digests if digest not in self._entries]

    async def upload(self, sources: Mapping[str, Path]) -> tuple[list[str], list[str]]:
        verified: list[tuple[str, Path, int]] = []
        for declared, source in sources.items():
            actual, size = self.hash_path(source)
            if actual != declared:
                raise ValueError(
                    f"blob content does not match declared hash {declared}"
                )
            verified.append((declared, source, size))
        stored: list[str] = []
        already: list[str] = []
        async with self._lock:
            for digest, source, size in verified:
                if digest in self._entries:
                    already.append(digest)
                    continue
                self._publish_locked(digest, source, size)
                stored.append(digest)
            self._evict_locked()
        return stored, already

    async def ingest_and_acquire(
        self, manifest_digests: set[str], inline: Mapping[str, Path]
    ) -> tuple[list[str], list[str]]:
        verified: list[tuple[str, Path, int]] = []
        for declared, source in inline.items():
            actual, size = self.hash_path(source)
            if actual != declared:
                raise ValueError(
                    f"blob content does not match declared hash {declared}"
                )
            verified.append((declared, source, size))

        unused = sorted(set(inline) - manifest_digests)
        acquired: list[str] = []
        async with self._lock:
            for digest, source, size in verified:
                if digest in manifest_digests and digest not in self._entries:
                    self._publish_locked(digest, source, size)
            missing = sorted(
                digest for digest in manifest_digests if digest not in self._entries
            )
            if not missing:
                now = time.time()
                for digest in manifest_digests:
                    entry = self._entries[digest]
                    entry.refs += 1
                    entry.last_used = now
                    try:
                        os.utime(entry.path, (now, now))
                    except OSError:
                        pass
                    acquired.append(digest)
            self._evict_locked()
        return missing, unused

    async def release(self, digests: set[str]) -> None:
        async with self._lock:
            for digest in digests:
                entry = self._entries.get(digest)
                if entry is not None and entry.refs > 0:
                    entry.refs -= 1
            self._evict_locked()

    async def materialize(self, files: Mapping[str, str], destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=False)
        for remote_path, digest in files.items():
            target = destination.joinpath(*remote_path.split("/"))
            target.parent.mkdir(parents=True, exist_ok=True)
            entry = self._entries[digest]
            shutil.copyfile(entry.path, target)

    def _publish_locked(self, digest: str, source: Path, size: int) -> None:
        destination = self.object_path(digest)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp = self.tmp / f"upload-{uuid.uuid4()}.part"
        shutil.copyfile(source, temp)
        try:
            os.chmod(temp, 0o444)
            try:
                os.link(temp, destination)
            except FileExistsError:
                pass
        finally:
            temp.unlink(missing_ok=True)
        stat = destination.stat()
        self._entries[digest] = CacheEntry(
            digest, destination, stat.st_size, 0, time.time()
        )

    def _evict_locked(self) -> None:
        total = sum(entry.size for entry in self._entries.values())
        if total <= self.capacity:
            return
        candidates = sorted(
            (entry for entry in self._entries.values() if entry.refs == 0),
            key=lambda entry: entry.last_used,
        )
        for entry in candidates:
            if total <= self.capacity:
                break
            try:
                entry.path.unlink()
            except FileNotFoundError:
                pass
            total -= entry.size
            self._entries.pop(entry.digest, None)
