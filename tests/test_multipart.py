import pytest

from benchmark_server.errors import ValidationError
from benchmark_server.multipart import MultipartPart, encode_multipart, parse_multipart


def test_multipart_round_trip_preserves_binary_data():
    parts = [
        MultipartPart("program", "application/json", b'{"instructions":[]}'),
        MultipartPart("blob:" + "0" * 64, "application/octet-stream", b"\x00\xff\r\nbytes"),
    ]
    body, content_type = encode_multipart(parts)
    assert parse_multipart(content_type, body) == parts


def test_multipart_rejects_wrong_type_and_missing_close_boundary():
    with pytest.raises(ValidationError, match="multipart/form-data"):
        parse_multipart("application/json", b"{}")
    body, content_type = encode_multipart([MultipartPart("program", "application/json", b"{}")])
    with pytest.raises(ValidationError, match="malformed"):
        parse_multipart(content_type, body.rsplit(b"--", 1)[0])


def test_multipart_rejects_duplicate_part_headers():
    body = (
        b"--boundary\r\n"
        b'Content-Disposition: form-data; name="program"\r\n'
        b"Content-Type: application/json\r\n"
        b"Content-Type: application/json\r\n\r\n"
        b"{}\r\n"
        b"--boundary--\r\n"
    )
    with pytest.raises(ValidationError, match="one Content-Type"):
        parse_multipart("multipart/form-data; boundary=boundary", body)

    duplicate_name = body.replace(b'name="program"', b'name="program"; name="other"').replace(
        b"Content-Type: application/json\r\nContent-Type: application/json\r\n",
        b"Content-Type: application/json\r\n",
    )
    with pytest.raises(ValidationError, match="malformed Content-Disposition"):
        parse_multipart("multipart/form-data; boundary=boundary", duplicate_name)
