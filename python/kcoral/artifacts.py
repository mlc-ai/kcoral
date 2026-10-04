"""File and folder snapshots, returned artifacts, and explicit local saving."""

from __future__ import annotations

import os
import secrets
import stat
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from kcoral.protocol import normalize_file_path

_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def validate_manifest(files: Mapping, directories: tuple[str, ...] | list[str]) -> None:
    """Validate before decoding payloads or writing any filesystem entries."""
    file_names = list(files)
    names = [*file_names, *directories]
    for name in names:
        if normalize_file_path(name) != name:
            raise ValueError("folder paths must be canonical relative POSIX paths")
    if file_names != sorted(file_names) or list(directories) != sorted(directories):
        raise ValueError("folder entries must be sorted")
    if len(set(names)) != len(names):
        raise ValueError("folder contains duplicate or conflicting paths")
    directory_set = set(directories)
    for name in names:
        parent = name.rpartition("/")[0]
        if parent and parent not in directory_set:
            raise ValueError(f"folder entry has a missing or conflicting parent: {name!r}")


@dataclass(frozen=True)
class ReturnedFile:
    """An in-memory snapshot, independent of the client and remote workspace."""

    _data: bytes

    def __post_init__(self) -> None:
        if not isinstance(self._data, bytes):
            raise TypeError("ReturnedFile contents must be bytes")

    def read_bytes(self) -> bytes:
        """Return the captured file contents without writing to disk."""
        return self._data

    def save(self, destination: str | os.PathLike[str], *, overwrite: bool = False) -> Path:
        """Save to an exact path with an existing parent; refuse symlink traversal.

        The destination must be new unless ``overwrite=True``, which permits
        replacing an existing regular file. Return the destination as a ``Path``.
        """
        with _destination_parent(destination) as (parent_fd, name, destination_path):
            _check_destination(parent_fd, name, overwrite=overwrite)
            staging = f".kcoral-{secrets.token_hex(16)}"
            fd = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=parent_fd)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(self._data)
                if overwrite:
                    _check_destination(parent_fd, name, overwrite=True)
                    # A symlink substituted after the check is replaced, never followed.
                    os.replace(staging, name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
                else:
                    # Atomic exclusive publication, even if the destination appeared
                    # after the initial check. Both names are in the same filesystem.
                    os.link(staging, name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
            finally:
                try:
                    os.unlink(staging, dir_fd=parent_fd)
                except FileNotFoundError:
                    pass
        return destination_path


@dataclass(frozen=True)
class ReturnedFolder:
    """A validated tree; paths are relative to the selected root, which is implicit."""

    files: Mapping[str, ReturnedFile]
    """Read-only mapping of relative paths to captured files, including hidden files."""

    directories: tuple[str, ...] = ()
    """Sorted relative directory paths, including empty directories; excludes the root."""

    def __post_init__(self) -> None:
        files = dict(self.files)
        directories = tuple(self.directories)
        validate_manifest(files, directories)
        if any(not isinstance(value, ReturnedFile) for value in files.values()):
            raise TypeError("ReturnedFolder files must contain ReturnedFile values")
        object.__setattr__(self, "files", MappingProxyType(files))
        object.__setattr__(self, "directories", directories)

    def save(self, destination: str | os.PathLike[str]) -> Path:
        """Create a new folder beneath an existing parent; refuse symlink traversal.

        The destination must not exist. Output is visible while writing and is
        removed on failure. Return the destination as a ``Path``.
        """
        with _destination_parent(destination) as (parent_fd, name, destination_path):
            os.mkdir(name, 0o700, dir_fd=parent_fd)
            try:
                with _open_directory(parent_fd, name) as root_fd:
                    for directory in self.directories:
                        parent, _, leaf = directory.rpartition("/")
                        with _open_directory(root_fd, parent) as fd:
                            os.mkdir(leaf, 0o700, dir_fd=fd)
                    for relative, file in self.files.items():
                        parent, _, leaf = relative.rpartition("/")
                        with _open_directory(root_fd, parent) as fd:
                            file_fd = os.open(
                                leaf, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=fd
                            )
                            with os.fdopen(file_fd, "wb") as stream:
                                stream.write(file.read_bytes())
            except BaseException:
                _remove_partial_folder(parent_fd, name)
                raise
        return destination_path


@contextmanager
def _destination_parent(destination: str | os.PathLike[str]) -> Iterator[tuple[int, str, Path]]:
    path = Path(destination)
    if ".." in path.parts or not path.name or path.name in (".", ".."):
        raise ValueError("save destination must name a file or folder without '..'")
    parts = path.parts[1:] if path.is_absolute() else path.parts
    with _open_directory(path.anchor or ".", "/".join(parts[:-1])) as fd:
        yield fd, path.name, path


def _check_destination(parent_fd: int, name: str, *, overwrite: bool) -> None:
    try:
        info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if not overwrite:
        raise FileExistsError(f"save destination already exists: {name!r}")
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("overwrite requires a regular file destination")


@contextmanager
def _open_directory(root: int | str, path: str) -> Iterator[int]:
    """Open a relative directory without following links; close it on exit."""
    fd = os.dup(root) if isinstance(root, int) else os.open(root, _DIRECTORY_FLAGS)
    try:
        for component in path.split("/") if path else ():
            child_fd = os.open(component, _DIRECTORY_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = child_fd
        yield fd
    finally:
        os.close(fd)


def _remove_partial_folder(parent_fd: int, name: str) -> None:
    """Remove partial output without recursion or following symbolic links."""
    try:
        root_fd = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
    except FileNotFoundError:
        return
    stack = []

    def enter(fd: int, parent: int, leaf: str) -> None:
        try:
            stack.append((fd, os.scandir(fd), parent, leaf))
        except BaseException:
            os.close(fd)
            raise

    enter(root_fd, parent_fd, name)
    try:
        while stack:
            fd, entries, parent, leaf = stack[-1]
            entry = next(entries, None)
            if entry is None:
                os.rmdir(leaf, dir_fd=parent)
                entries.close()
                os.close(fd)
                stack.pop()
            elif entry.is_dir(follow_symlinks=False):
                enter(os.open(entry.name, _DIRECTORY_FLAGS, dir_fd=fd), fd, entry.name)
            else:
                os.unlink(entry.name, dir_fd=fd)
    finally:
        for fd, entries, _, _ in reversed(stack):
            entries.close()
            os.close(fd)


def _read_file(parent_fd: int, name: str, max_bytes: int) -> ReturnedFile:
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent_fd)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"file return requires a regular file: {name!r}")
        if before.st_size > max_bytes:
            raise ValueError("file return exceeds max_response_bytes binary budget")
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


def _folder_files(folder: str | os.PathLike[str], destination: str) -> Iterator[tuple[str, bytes]]:
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
