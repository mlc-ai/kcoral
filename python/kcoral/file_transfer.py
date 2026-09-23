"""Bounded snapshots of files beneath a request workspace (not a sandbox)."""

from __future__ import annotations

import os
import stat

from .artifacts import _DIRECTORY_FLAGS, ReturnedFile, ReturnedFolder, _open_directory
from .schemas import normalize_file_path


def _read_file(parent_fd: int, name: str, max_bytes: int) -> ReturnedFile:
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent_fd)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"file return requires a regular file: {name!r}")
        if before.st_size > max_bytes:
            raise ValueError("file return exceeds max_response_mbytes binary budget")
        # Never chase an append-only writer or allocate from an unbounded read.
        data = stream.read(before.st_size)
        after = os.fstat(stream.fileno())
        if (
            len(data) != before.st_size
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
            or after.st_ctime_ns != before.st_ctime_ns
        ):
            raise ValueError(f"file changed while reading: {name!r}")
        return ReturnedFile(data)


def collect(
    workspace: str, path: str, kind: str, *, max_bytes: int
) -> ReturnedFile | ReturnedFolder:
    """Capture a selection; caller commits results only after success."""
    parent, _, name = normalize_file_path(path).rpartition("/")
    with _open_directory(workspace, parent) as parent_fd:
        if kind == "file":
            return _read_file(parent_fd, name, max_bytes)
        root_fd = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
    return _collect_folder(root_fd, max_bytes)


def _collect_folder(root_fd: int, max_bytes: int) -> ReturnedFolder:
    files = {}
    directories = []
    seen = set()
    stack = []

    def enter(fd: int, relative: str) -> None:
        try:
            info = os.fstat(fd)
            identity = (info.st_dev, info.st_ino)
            if identity in seen:
                raise ValueError("file return encountered a repeated directory")
            seen.add(identity)
            stack.append((fd, os.scandir(fd), relative))
        except BaseException:
            os.close(fd)
            raise

    enter(root_fd, "")
    try:
        while stack:
            parent_fd, entries, relative = stack[-1]
            entry = next(entries, None)
            if entry is None:
                entries.close()
                os.close(parent_fd)
                stack.pop()
                continue
            name = normalize_file_path(f"{relative}/{entry.name}" if relative else entry.name)
            mode = entry.stat(follow_symlinks=False).st_mode
            if stat.S_ISDIR(mode):
                directories.append(name)
                enter(os.open(entry.name, _DIRECTORY_FLAGS, dir_fd=parent_fd), name)
            elif stat.S_ISREG(mode):
                file = _read_file(parent_fd, entry.name, max_bytes)
                max_bytes -= len(file.read_bytes())
                files[name] = file
            else:
                raise ValueError(f"file return rejects symbolic links and special files: {name!r}")
    finally:
        for fd, entries, _ in reversed(stack):
            entries.close()
            os.close(fd)
    return ReturnedFolder(dict(sorted(files.items())), tuple(sorted(directories)))
