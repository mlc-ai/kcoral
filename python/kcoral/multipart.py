"""Strict multipart/form-data parsing and encoding for protocol payloads."""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from email.parser import BytesParser
from email.policy import default

from .errors import ValidationError


@dataclass(frozen=True)
class MultipartPart:
    name: str
    content_type: str
    data: bytes


def parse_multipart(content_type: str | None, body: bytes) -> list[MultipartPart]:
    if not content_type or not content_type.lower().startswith("multipart/form-data"):
        raise ValidationError("expected a multipart/form-data request")
    message = BytesParser(policy=default).parsebytes(
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("ascii") + body
    )
    if message.defects or not message.is_multipart():
        raise ValidationError("malformed multipart/form-data body")

    parts: list[MultipartPart] = []
    for item in message.iter_parts():
        if len(item.get_all("content-disposition", [])) != 1:
            raise ValidationError("each multipart part must have one Content-Disposition")
        if len(item.get_all("content-type", [])) != 1:
            raise ValidationError("each multipart part must have one Content-Type")
        if item["content-disposition"].defects:
            raise ValidationError("multipart part has a malformed Content-Disposition")
        if item["content-type"].defects:
            raise ValidationError("multipart part has a malformed Content-Type")
        if item.is_multipart():
            raise ValidationError("nested multipart parts are not supported")
        if item.get_content_disposition() != "form-data":
            raise ValidationError("multipart parts must use form-data disposition")
        names = [
            value
            for key, value in item.get_params(header="content-disposition", unquote=True)[1:]
            if key.lower() == "name"
        ]
        if len(names) != 1 or not isinstance(names[0], str) or not names[0]:
            raise ValidationError("multipart part is missing its name")
        name = names[0]
        transfer_encoding = item.get("content-transfer-encoding")
        if transfer_encoding and transfer_encoding.lower() not in ("binary", "8bit"):
            raise ValidationError("encoded multipart parts are not supported")
        payload = item.get_payload(decode=True)
        if not isinstance(payload, bytes):
            raise ValidationError(f"multipart part {name!r} has an invalid payload")
        parts.append(MultipartPart(name=name, content_type=item.get_content_type(), data=payload))
    if not parts:
        raise ValidationError("multipart body has no parts")
    return parts


def encode_multipart(parts: list[MultipartPart]) -> tuple[bytes, str]:
    if not parts:
        raise ValueError("at least one multipart part is required")
    while True:
        boundary = f"kcoral-{secrets.token_hex(16)}"
        delimiter = f"\r\n--{boundary}".encode("ascii")
        if all(delimiter not in part.data for part in parts):
            break

    chunks: list[bytes] = []
    for part in parts:
        if not part.name or any(character in part.name for character in '"\r\n'):
            raise ValueError(f"invalid multipart part name: {part.name!r}")
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("ascii"),
                f'Content-Disposition: form-data; name="{part.name}"\r\n'.encode("ascii"),
                f"Content-Type: {part.content_type}\r\n\r\n".encode("ascii"),
                part.data,
                b"\r\n",
            ]
        )
    chunks.append(f"--{boundary}--\r\n".encode("ascii"))
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"
