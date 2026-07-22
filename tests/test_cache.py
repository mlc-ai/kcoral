from benchmark_server.cache import ByteCache


def test_put_get_missing():
    c = ByteCache(1000)
    assert c.get("k") is None
    c.put("k", b"abc")
    assert c.get("k") == b"abc"
    assert c.missing(["k", "z"]) == ["z"]


def test_lru_eviction_over_budget():
    c = ByteCache(10, max_object_bytes=10)
    c.put("a", b"aaaaa")  # 5
    c.put("b", b"bbbbb")  # 5 -> total 10 (at budget)
    c.get("a")            # a becomes most-recently-used
    c.put("c", b"ccccc")  # 15 > 10 -> evict LRU unpinned (b)
    assert "a" in c and "c" in c and "b" not in c


def test_pinned_entry_not_evicted():
    c = ByteCache(10, max_object_bytes=10)
    c.put("a", b"aaaaa")
    c.pin(["a"])          # a is now pinned
    c.put("b", b"bbbbb")
    c.put("c", b"ccccc")  # over budget, but a is pinned -> b is evicted instead
    assert "a" in c and "c" in c and "b" not in c
    c.unpin(["a"])


def test_per_object_cap_skips_giant_objects():
    c = ByteCache(100, max_object_bytes=10)
    c.put("big", b"x" * 50)
    assert "big" not in c
    c.put("ok", b"x" * 8)
    assert "ok" in c
