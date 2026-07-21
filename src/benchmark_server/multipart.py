from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from fastapi import Request
from python_multipart.multipart import MultipartParser, parse_options_header


@dataclass
class ParsedPart:
    name: str
    content_type: str | None
    path: Path
    size: int

    def read(self, maximum: int | None = None) -> bytes:
        if maximum is not None and self.size > maximum:
            raise ValueError(f"multipart part {self.name!r} is too large")
        return self.path.read_bytes()


class MultipartBody:
    def __init__(self, parts: list[ParsedPart]) -> None:
        self.parts = parts

    def close(self) -> None:
        for part in self.parts:
            try:
                part.path.unlink()
            except FileNotFoundError:
                pass


async def parse_multipart_request(
    request: Request, temporary_dir: Path
) -> MultipartBody:
    content_type = request.headers.get("content-type", "")
    media_type, options = parse_options_header(content_type)
    if media_type.lower() != b"multipart/form-data" or b"boundary" not in options:
        raise ValueError("content type must be multipart/form-data with a boundary")
    temporary_dir.mkdir(parents=True, exist_ok=True)

    parts: list[ParsedPart] = []
    current_headers: list[tuple[bytes, bytes]] = []
    header_field = bytearray()
    header_value = bytearray()
    current_file = None
    current_path: Path | None = None
    current_size = 0
    current_name: str | None = None
    current_type: str | None = None
    ended = False

    def part_begin() -> None:
        nonlocal \
            current_headers, \
            current_file, \
            current_path, \
            current_size, \
            current_name, \
            current_type
        current_headers = []
        current_size = 0
        current_name = None
        current_type = None
        fd, filename = tempfile.mkstemp(
            prefix="multipart-", suffix=".part", dir=temporary_dir
        )
        current_path = Path(filename)
        current_file = os.fdopen(fd, "wb")

    def on_header_field(data: bytes, start: int, end: int) -> None:
        header_field.extend(data[start:end])

    def on_header_value(data: bytes, start: int, end: int) -> None:
        header_value.extend(data[start:end])

    def on_header_end() -> None:
        current_headers.append((bytes(header_field).lower(), bytes(header_value)))
        header_field.clear()
        header_value.clear()

    def headers_finished() -> None:
        nonlocal current_name, current_type
        if len({name for name, _ in current_headers}) != len(current_headers):
            raise ValueError("multipart part contains a duplicate header")
        headers = dict(current_headers)
        disposition, disposition_options = parse_options_header(
            headers.get(b"content-disposition", b"")
        )
        if disposition.lower() != b"form-data" or b"name" not in disposition_options:
            raise ValueError("multipart part is missing a valid form-data name")
        try:
            current_name = disposition_options[b"name"].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("multipart part name must be UTF-8") from exc
        raw_type = headers.get(b"content-type")
        if raw_type is not None:
            try:
                current_type = raw_type.decode("ascii").split(";", 1)[0].strip().lower()
            except UnicodeDecodeError as exc:
                raise ValueError("invalid multipart content type") from exc

    def part_data(data: bytes, start: int, end: int) -> None:
        nonlocal current_size
        assert current_file is not None
        chunk = data[start:end]
        current_file.write(chunk)
        current_size += len(chunk)

    def part_end() -> None:
        nonlocal current_file
        assert (
            current_file is not None
            and current_path is not None
            and current_name is not None
        )
        current_file.close()
        current_file = None
        parts.append(ParsedPart(current_name, current_type, current_path, current_size))

    def body_end() -> None:
        nonlocal ended
        ended = True

    callbacks = {
        "on_part_begin": part_begin,
        "on_header_field": on_header_field,
        "on_header_value": on_header_value,
        "on_header_end": on_header_end,
        "on_headers_finished": headers_finished,
        "on_part_data": part_data,
        "on_part_end": part_end,
        "on_end": body_end,
    }
    parser = MultipartParser(options[b"boundary"], callbacks)
    try:
        async for chunk in request.stream():
            parser.write(chunk)
        parser.finalize()
        if not ended:
            raise ValueError("multipart body is incomplete")
        if current_file is not None:
            current_file.close()
            current_file = None
        return MultipartBody(parts)
    except Exception:
        if current_file is not None:
            current_file.close()
        if current_path is not None and all(
            part.path != current_path for part in parts
        ):
            try:
                current_path.unlink()
            except FileNotFoundError:
                pass
        MultipartBody(parts).close()
        raise


def encode_multipart(
    metadata: bytes, binaries: list[tuple[str, bytes]]
) -> tuple[bytes, str]:
    import secrets

    boundary = f"benchmark-server-{secrets.token_hex(16)}"
    body = bytearray()

    def add(name: str, content_type: str, data: bytes) -> None:
        body.extend(f"--{boundary}\r\n".encode("ascii"))
        body.extend(
            f'Content-Disposition: form-data; name="{name}"\r\n'.encode("ascii")
        )
        body.extend(f"Content-Type: {content_type}\r\n\r\n".encode("ascii"))
        body.extend(data)
        body.extend(b"\r\n")

    add("result", "application/json", metadata)
    for name, data in binaries:
        add(name, "application/octet-stream", data)
    body.extend(f"--{boundary}--\r\n".encode("ascii"))
    return bytes(body), boundary
