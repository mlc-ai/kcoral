"""Deterministic, uncompressed folder archives containing only regular files."""

from __future__ import annotations

import io
import tarfile
from collections.abc import Mapping

from .schemas import normalize_file_path, validate_and_add_file_paths


def pack_files(files: Mapping[str, bytes]) -> bytes:
    """Ignore host metadata so equal relative paths and contents share a cache key."""
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for path, data in sorted(files.items()):
            member = tarfile.TarInfo(path)
            member.size = len(data)
            member.mode = 0o600
            archive.addfile(member, io.BytesIO(data))
    return output.getvalue()


def archive_files(data: bytes) -> list[tuple[str, int, int]]:
    """Validate all entries before extraction; return paths and bounded byte ranges.

    Only regular files are allowed. Parents are created implicitly, as with
    individual file uploads. No tar extraction API or archived metadata is used.
    Uncompressed, non-sparse contents cannot expand beyond the received blob.
    """
    entries = []
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as archive:
            for member in archive:
                if (
                    not member.isreg()
                    or member.sparse is not None
                    or any(key.startswith("GNU.sparse") for key in member.pax_headers)
                ):
                    raise ValueError("folder archive entries must be non-sparse regular files")
                path = normalize_file_path(member.name)
                if member.size < 0 or member.offset_data + member.size > len(data):
                    raise ValueError("folder archive contains truncated or invalid file contents")
                entries.append((path, member.offset_data, member.size))
    except (tarfile.TarError, EOFError, UnicodeError) as exc:
        raise ValueError(f"invalid uncompressed folder archive: {exc}") from exc
    validate_and_add_file_paths([path for path, _, _ in entries], set())
    return entries
