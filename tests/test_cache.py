import pytest

from benchmark_server.cache import ByteCache

CANONICAL_KEY = "sha256:" + "0" * 64


def test_put_get_missing():
    c = ByteCache(1000)
    assert c.get("k") is None
    c.put("k", b"abc")
    assert c.get("k") == b"abc"
    assert c.missing(["k", "z"]) == ["z"]
    c.close()


def test_lru_eviction_over_budget():
    c = ByteCache(10, max_object_bytes=10)
    c.put("a", b"aaaaa")  # 5
    c.put("b", b"bbbbb")  # 5 -> total 10 (at budget)
    c.get("a")  # a becomes most-recently-used
    c.put("c", b"ccccc")  # 15 > 10 -> evict LRU unpinned (b)
    assert "a" in c and "c" in c and "b" not in c
    c.close()


def test_pinned_entry_not_evicted():
    c = ByteCache(10, max_object_bytes=10)
    c.put("a", b"aaaaa")
    c.pin(["a"])  # a is now pinned
    c.put("b", b"bbbbb")
    c.put("c", b"ccccc")  # over budget, but a is pinned -> b is evicted instead
    assert "a" in c and "c" in c and "b" not in c
    c.unpin(["a"])
    c.close()


def test_per_object_cap_skips_giant_objects():
    c = ByteCache(100, max_object_bytes=10)
    c.put("big", b"x" * 50)
    assert "big" not in c
    c.put("ok", b"x" * 8)
    assert "ok" in c
    c.close()


# --- disk persistence -------------------------------------------------------


def test_persists_across_instances(tmp_path):
    first = ByteCache(1000, cache_dir=tmp_path / "cache")
    first.put(CANONICAL_KEY, b"hello")
    first.close()
    second = ByteCache(1000, cache_dir=tmp_path / "cache")
    assert second.get(CANONICAL_KEY) == b"hello"
    second.close()


def test_arbitrary_key_roundtrips_safely(tmp_path):
    # Non-canonical keys (e.g. containing path separators) must not escape the
    # cache directory and must survive a reload.
    c = ByteCache(1000, cache_dir=tmp_path / "cache")
    c.put("../weird/key:1", b"data")
    c.close()
    assert not (tmp_path / "weird").exists()
    reloaded = ByteCache(1000, cache_dir=tmp_path / "cache")
    assert reloaded.get("../weird/key:1") == b"data"
    reloaded.close()


def test_lock_prevents_sharing_a_directory(tmp_path):
    c = ByteCache(1000, cache_dir=tmp_path / "cache")
    with pytest.raises(RuntimeError):
        ByteCache(1000, cache_dir=tmp_path / "cache")
    c.close()
    reopened = ByteCache(1000, cache_dir=tmp_path / "cache")  # lock released on close
    reopened.close()


def test_eviction_deletes_files(tmp_path):
    c = ByteCache(10, cache_dir=tmp_path / "cache", max_object_bytes=10)
    c.put("a", b"aaaaa")
    c.put("b", b"bbbbb")
    c.get("a")
    c.put("c", b"ccccc")
    assert "b" not in c
    files = list((tmp_path / "cache" / "objects").glob("*/*"))
    assert len(files) == 2
    c.close()


def test_reload_respects_capacity(tmp_path):
    c = ByteCache(1000, cache_dir=tmp_path / "cache", max_object_bytes=1000)
    c.put("a", b"a" * 400)
    c.put("b", b"b" * 400)
    c.close()
    shrunk = ByteCache(500, cache_dir=tmp_path / "cache")
    assert shrunk.size_bytes <= 500
    c2_files = list((tmp_path / "cache" / "objects").glob("*/*"))
    assert len(c2_files) == 1
    shrunk.close()
