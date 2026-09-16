"""Received file contents and explicit local saving on Linux."""

from __future__ import annotations

import os
import secrets
import stat
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from .schemas import normalize_file_path

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
        return self._data

    def save(self, destination: str | os.PathLike[str], *, overwrite: bool = False) -> Path:
        """Save to an exact path with an existing parent; refuse symlink traversal."""
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
    directories: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        files = dict(self.files)
        directories = tuple(self.directories)
        validate_manifest(files, directories)
        if any(not isinstance(value, ReturnedFile) for value in files.values()):
            raise TypeError("ReturnedFolder files must contain ReturnedFile values")
        object.__setattr__(self, "files", MappingProxyType(files))
        object.__setattr__(self, "directories", directories)

    def save(self, destination: str | os.PathLike[str]) -> Path:
        """Create a new folder, visible while writing; remove partial output on failure."""
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
