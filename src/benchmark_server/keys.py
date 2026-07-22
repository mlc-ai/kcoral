"""Content addressing: canonical byte serialization + `sha256` keys.

An uploaded object is identified by ``key = "sha256:" + sha256(canonical_bytes)``.
The client computes the key over the exact bytes it would send; the server
recomputes and *verifies* so the two sides can never disagree on identity.

Canonical forms (fixed per kind):
- ``function``: the UTF-8 source bytes.
- ``tensor``: a JSON ``{dtype, shape}`` header + ``\\0`` + raw row-major bytes.

The byte cache stores exactly ``canonical_bytes``; the worker's Runtime knows how
to materialize each kind back from those bytes.
"""

from __future__ import annotations

import base64
import hashlib
import json

from .errors import ValidationError


def canonical_bytes(kind: str, inline: dict) -> bytes:
    """Serialize an upload payload to its canonical byte form."""
    if not isinstance(inline, dict):
        raise ValidationError("upload 'inline' must be an object")
    if kind == "function":
        src = inline.get("source")
        if not isinstance(src, str):
            raise ValidationError("function upload requires string 'source'")
        return src.encode("utf-8")
    if kind == "tensor":
        try:
            dtype = str(inline["dtype"])
            shape = [int(d) for d in inline["shape"]]
            raw = base64.b64decode(inline["data_b64"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError(f"malformed tensor upload: {exc}") from exc
        header = json.dumps({"dtype": dtype, "shape": shape}, separators=(",", ":")).encode()
        return header + b"\x00" + raw
    raise ValidationError(f"unknown upload kind: {kind!r}")


def compute_key(kind: str, inline: dict) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(kind, inline)).hexdigest()


def verify_key(key: str, kind: str, inline: dict) -> bytes:
    """Verify a declared key against its payload; return the canonical bytes."""
    payload = canonical_bytes(kind, inline)
    actual = "sha256:" + hashlib.sha256(payload).hexdigest()
    if actual != key:
        raise ValidationError(f"key mismatch: declared {key}, computed {actual}")
    return payload
