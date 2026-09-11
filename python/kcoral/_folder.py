"""Local directory traversal for the Program.upload_folder sugar."""

from __future__ import annotations

import os
import stat
from collections.abc import Iterator
from pathlib import Path

from .schemas import normalize_file_path


def folder_files(folder: str | os.PathLike[str], destination: str) -> Iterator[tuple[str, bytes]]:
    """Yield snapshots of regular files, never following links or recursing in Python.

    Directory descriptors anchor each descent even if a local path is replaced
    during traversal. Only the active ancestry stays open, so a wide tree does
    not exhaust descriptors. File reads are bounded by their initial size.
    """
    source = Path(folder)
    if source.is_symlink():
        raise ValueError(f"upload_folder rejects symbolic links: {source}")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    seen = set()
    stack = []

    def enter(fd: int, remote: str) -> None:
        try:
            info = os.fstat(fd)
            identity = (info.st_dev, info.st_ino)
            if identity in seen:
                raise ValueError(f"upload_folder encountered a repeated directory: {remote}")
            seen.add(identity)
            stack.append((fd, os.scandir(fd), remote))
        except BaseException:
            os.close(fd)
            raise

    enter(os.open(source, flags), destination)
    try:
        while stack:
            parent_fd, entries, remote = stack[-1]
            entry = next(entries, None)
            if entry is None:
                entries.close()
                os.close(parent_fd)
                stack.pop()
                continue
            path = normalize_file_path(f"{remote}/{entry.name}")
            mode = entry.stat(follow_symlinks=False).st_mode
            if stat.S_ISDIR(mode):
                enter(os.open(entry.name, flags, dir_fd=parent_fd), path)
            elif stat.S_ISREG(mode):
                fd = os.open(
                    entry.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent_fd
                )
                with os.fdopen(fd, "rb") as stream:
                    before = os.fstat(stream.fileno())
                    if not stat.S_ISREG(before.st_mode):
                        raise ValueError(f"upload_folder requires a regular file: {path}")
                    data = stream.read(before.st_size)
                    after = os.fstat(stream.fileno())
                    if (
                        len(data) != before.st_size
                        or after.st_size != before.st_size
                        or after.st_mtime_ns != before.st_mtime_ns
                        or after.st_ctime_ns != before.st_ctime_ns
                    ):
                        raise ValueError(f"upload_folder file changed while reading: {path}")
                yield path, data
            else:
                raise ValueError(f"upload_folder rejects symbolic links and special files: {path}")
    finally:
        for fd, entries, _ in reversed(stack):
            entries.close()
            os.close(fd)
