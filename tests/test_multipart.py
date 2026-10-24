import pytest

from kcoral.errors import ValidationError
from kcoral.protocol import MultipartPart, encode_multipart, parse_multipart


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


def test_binary_payload_preserves_boundary_prefixes():
    payload = bytes(range(256)) + b"\r\n--boundaryX\r\n--boundary--X\r\n--boundary \tX\n"
    body = (
        b"--boundary\r\n"
        b'Content-Disposition: form-data; name="blob"\r\n'
        b"Content-Type: application/octet-stream\r\n\r\n" + payload + b"\r\n--boundary--\r\n"
    )
    assert parse_multipart("multipart/form-data; boundary=boundary", body) == [
        MultipartPart("blob", "application/octet-stream", payload)
    ]


def test_encoded_payloads_are_aligned_and_parse_as_views_of_a_writable_body():
    parts = [
        MultipartPart("result", "application/json", b"{}"),
        MultipartPart("return:0", "application/octet-stream", bytes(range(100))),
        MultipartPart("return:1", "application/octet-stream", b"xyz"),
    ]
    body, content_type = encode_multipart(parts)
    buffer = bytearray(body)
    parsed = parse_multipart(content_type, buffer)
    assert parsed == parts
    for part in parsed:
        assert isinstance(part.data, memoryview) and part.data.obj is buffer
        offset = body.index(bytes(part.data) + b"\r\n--")
        assert offset % 8 == 0


def test_reencoded_views_avoid_a_boundary_inside_their_payload(monkeypatch):
    payload = b"A\r\n--kcoral-" + b"a" * 32 + b"\r\nB"
    part = MultipartPart("blob", "application/octet-stream", payload)
    body, content_type = encode_multipart([part])
    parsed = parse_multipart(content_type, bytearray(body))
    assert isinstance(parsed[0].data, memoryview)
    candidates = iter(["a" * 32, "b" * 32])
    monkeypatch.setattr("kcoral.protocol.secrets.token_hex", lambda size: next(candidates))
    body, content_type = encode_multipart(parsed)
    assert content_type.endswith("b" * 32)
    assert parse_multipart(content_type, body) == [part]


@pytest.mark.parametrize("kind", [bytes, bytearray, memoryview])
@pytest.mark.parametrize(
    "payload",
    [
        b"A\r\n--kcoral-" + b"a" * 32 + b"\r\nB",
        b"A\n--kcoral-" + b"a" * 32 + b"\nB",
        b"--kcoral-" + b"a" * 32 + b"\r\nB",
        b"A\n--kcoral-" + b"a" * 32 + b"--\r\nB",
    ],
    ids=["crlf", "lf", "payload_start", "closing"],
)
def test_encoder_avoids_every_delimiter_form_in_a_payload(monkeypatch, kind, payload):
    candidates = iter(["a" * 32, "b" * 32])
    monkeypatch.setattr("kcoral.protocol.secrets.token_hex", lambda size: next(candidates))
    part = MultipartPart("blob", "application/octet-stream", kind(payload))
    body, content_type = encode_multipart([part])
    assert content_type.endswith("b" * 32)
    assert parse_multipart(content_type, body) == [
        MultipartPart("blob", "application/octet-stream", payload)
    ]
