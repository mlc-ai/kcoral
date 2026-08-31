import pytest

from kcoral.errors import ValidationError
from kcoral.keys import compute_blob_hash, is_blob_hash, verify_blob


def test_blob_hash_format_is_strict():
    digest = compute_blob_hash(b"data")
    assert is_blob_hash(digest)
    assert not is_blob_hash(f"sha256:{digest}")
    assert not is_blob_hash(digest.upper())
    assert not is_blob_hash("0" * 63)


def test_verify_blob_rejects_bad_name_and_content():
    digest = compute_blob_hash(b"data")
    verify_blob(digest, b"data")
    with pytest.raises(ValidationError, match="digest"):
        verify_blob("bad", b"data")
    with pytest.raises(ValidationError, match="mismatch"):
        verify_blob(digest, b"changed")
