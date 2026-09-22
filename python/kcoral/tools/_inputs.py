"""Snapshot command inputs, preserving executable bits in a stable archive."""

import io
import tarfile
from pathlib import PurePosixPath


def relative_path(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name or not path.parts:
        raise ValueError(f"expected a relative file path: {name!r}")
    return path


def pack_inputs(paths=()):
    """Stable archives: preserve each selected file or directory's basename."""
    buffer, names = io.BytesIO(), set()
    with tarfile.open(fileobj=buffer, mode="w") as tar:

        def add(name, data, executable=False):
            name = relative_path(name).as_posix()
            if name in names or any(
                name.startswith(old + "/") or old.startswith(name + "/") for old in names
            ):
                raise ValueError(f"duplicate or conflicting input path: {name}")
            names.add(name)
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o700 if executable else 0o600
            tar.addfile(info, io.BytesIO(data))

        for path in paths:
            if path.is_symlink() or not path.exists():
                raise ValueError(f"input must exist and not be a symlink: {path}")
            directory = path.is_dir()
            basename = path.resolve().name if directory else path.name
            if not basename:
                raise ValueError(f"input directory must have a name: {path}")
            for entry in sorted(path.rglob("*")) if directory else [path]:
                if "__pycache__" in entry.parts:
                    continue
                if entry.is_symlink():
                    raise ValueError(f"symlink inputs are not supported: {entry}")
                if entry.is_dir():
                    continue
                if not entry.is_file():
                    raise ValueError(f"input is not a regular file: {entry}")
                name = f"{basename}/{entry.relative_to(path).as_posix()}" if directory else basename
                add(name, entry.read_bytes(), bool(entry.stat().st_mode & 0o111))
    return buffer.getvalue()
