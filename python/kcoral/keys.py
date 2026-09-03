"""SHA-256 helpers for content-addressed binary blobs."""

from __future__ import annotations

import hashlib
import re

from .errors import ValidationError

_BLOB_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def is_blob_hash(value: object) -> bool:
    return isinstance(value, str) and _BLOB_HASH_RE.fullmatch(value) is not None


def compute_blob_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_blob(blob_hash: str, data: bytes) -> None:
    if not is_blob_hash(blob_hash):
        raise ValidationError("blob name must contain a lowercase SHA-256 digest")
    actual = compute_blob_hash(data)
    if actual != blob_hash:
        raise ValidationError(f"blob hash mismatch: declared {blob_hash}, computed {actual}")
