import base64

import pytest

from benchmark_server.errors import ValidationError
from benchmark_server.keys import canonical_bytes, compute_key, verify_key


def _tensor(dtype, shape, raw):
    return {"dtype": dtype, "shape": shape, "data_b64": base64.b64encode(raw).decode()}


def test_function_key_stable_and_prefixed():
    inline = {"source": "def f(): pass\n"}
    assert compute_key("function", inline) == compute_key("function", inline)
    assert compute_key("function", inline).startswith("sha256:")
    assert canonical_bytes("function", inline) == b"def f(): pass\n"


def test_tensor_key_depends_on_content_dtype_shape():
    raw = b"\x00\x01\x02\x03"
    k = compute_key("tensor", _tensor("float16", [2], raw))
    assert k != compute_key("tensor", _tensor("float16", [2], b"\x00\x01\x02\x04"))  # content
    assert k != compute_key("tensor", _tensor("int16", [2], raw))                    # dtype
    assert k != compute_key("tensor", _tensor("float16", [4], raw))                  # shape


def test_verify_key_roundtrip_and_mismatch():
    inline = {"source": "x=1\n"}
    key = compute_key("function", inline)
    assert verify_key(key, "function", inline) == b"x=1\n"
    with pytest.raises(ValidationError, match="key mismatch"):
        verify_key("sha256:deadbeef", "function", inline)


def test_unknown_kind_and_malformed():
    with pytest.raises(ValidationError):
        compute_key("weird", {})
    with pytest.raises(ValidationError):
        compute_key("function", {})  # no source
