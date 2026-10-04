import multiprocessing as mp
import os
from concurrent.futures import ThreadPoolExecutor

import pytest

from kcoral.protocol import compute_blob_hash
from kcoral.server.cache import ByteCache, DiskFileCache


def test_lru_eviction_over_budget():
    c = ByteCache(10, max_object_bytes=10)
    c.put("a", b"aaaaa")  # 5
    c.put("b", b"bbbbb")  # 5 -> total 10 (at budget)
    assert c.get("a") == b"aaaaa"  # and a becomes most-recently-used
    c.put("c", b"ccccc")  # 15 > 10 -> evict LRU unpinned (b)
    assert "a" in c and "c" in c and "b" not in c


def test_pinned_entry_not_evicted():
    c = ByteCache(10, max_object_bytes=10)
    c.put("a", b"aaaaa")
    c.pin(["a"])  # a is now pinned
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


def _disk_path(cache, key):
    return cache.directory / key[:2] / key


def test_disk_cache_survives_restart_without_sidecar_files(tmp_path):
    data = b"persistent file"
    key = compute_blob_hash(data)
    first = DiskFileCache(tmp_path, 100)
    first.put(key, data)
    second = DiskFileCache(tmp_path, 100)
    assert second.get(key) == data
    assert [p for p in tmp_path.rglob("*") if p.is_file()] == [_disk_path(first, key)]


def test_disk_lru_eviction_and_restart_with_smaller_budget(tmp_path):
    cache = DiskFileCache(tmp_path, 8)
    a, b, c = (compute_blob_hash(data) for data in (b"aaaa", b"bbbb", b"cccc"))
    cache.put_many({a: b"aaaa", b: b"bbbb"})
    os.utime(_disk_path(cache, a), ns=(1, 1))
    os.utime(_disk_path(cache, b), ns=(2, 2))
    assert cache.get(a) == b"aaaa"  # explicitly touches mtime, even on noatime mounts
    cache.put(c, b"cccc")
    assert cache.get(b) is None
    os.utime(_disk_path(cache, c), ns=(1, 1))
    assert cache.get(a) == b"aaaa"
    smaller = DiskFileCache(tmp_path, 4)
    assert smaller.get(c) is None and smaller.get(a) == b"aaaa"


@pytest.mark.parametrize("disabled", ["directory", "capacity"])
def test_disabled_disk_cache_never_writes(tmp_path, disabled):
    cache = DiskFileCache(
        None if disabled == "directory" else tmp_path, 0 if disabled == "capacity" else 10
    )
    key = compute_blob_hash(b"x")
    cache.put(key, b"x")
    assert cache.get(key) is None and list(tmp_path.iterdir()) == []


def test_disk_skips_oversized_objects_and_rejects_wrong_hashes(tmp_path):
    cache = DiskFileCache(tmp_path, 4)
    large = compute_blob_hash(b"large")
    cache.put(large, b"large")
    assert cache.get(large) is None
    with pytest.raises(ValueError, match="hash mismatch"):
        cache.put(large, b"oops")
    with pytest.raises(ValueError):
        cache.put("../escape", b"x")
    assert cache.get("../escape") is None
    with pytest.raises(ValueError, match="non-negative"):
        DiskFileCache(tmp_path, -1)


def test_corrupt_disk_object_is_a_miss_and_can_be_replaced(tmp_path):
    cache = DiskFileCache(tmp_path, 100)
    key = compute_blob_hash(b"good")
    cache.put(key, b"good")
    _disk_path(cache, key).write_bytes(b"evil")
    assert cache.get(key) is None
    cache.put(key, b"good")
    assert cache.get(key) == b"good"


@pytest.mark.parametrize("location", ["blob", "shard"])
def test_disk_does_not_follow_symlinks(tmp_path, location):
    outside = tmp_path / "outside"
    outside.mkdir()
    data = b"outside content"
    key = compute_blob_hash(data)
    target = outside / key
    target.write_bytes(data)
    cache = DiskFileCache(tmp_path / "cache", 100)
    path = _disk_path(cache, key)
    if location == "shard":
        path.parent.symlink_to(outside, target_is_directory=True)
    else:
        path.parent.mkdir()
        path.symlink_to(target)
    assert cache.get(key) is None
    cache.put(key, data)
    assert target.read_bytes() == data
    assert not list(outside.glob(".tmp-*"))


def test_failed_publish_keeps_existing_object_and_removes_temporary(tmp_path, monkeypatch):
    cache = DiskFileCache(tmp_path, 100)
    key = compute_blob_hash(b"stable")
    cache.put(key, b"stable")

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("kcoral.server.cache.os.replace", fail)
    cache.put(key, b"stable")
    assert cache.get(key) == b"stable"
    assert not list(tmp_path.rglob(".tmp-*"))


def test_disk_recovers_interrupted_writes(tmp_path):
    cache = DiskFileCache(tmp_path, 100)
    key = compute_blob_hash(b"complete")
    cache.put(key, b"complete")
    temporary = _disk_path(cache, key).parent / ".tmp-interrupted"
    temporary.write_bytes(b"partial")
    restarted = DiskFileCache(tmp_path, 100)
    assert restarted.get(key) == b"complete" and not temporary.exists()


def test_unusable_disk_directory_is_best_effort(tmp_path):
    occupied = tmp_path / "file"
    occupied.write_bytes(b"occupied")
    cache = DiskFileCache(occupied, 10)
    cache.put(compute_blob_hash(b"x"), b"x")
    assert cache.get(compute_blob_hash(b"x")) is None
    assert occupied.read_bytes() == b"occupied"


def _concurrent_disk_writer(directory, payload):
    cache = DiskFileCache(directory, 100_000)
    key = compute_blob_hash(payload)
    for _ in range(5):
        cache.put(key, payload)
        assert cache.get(key) == payload


def test_disk_cache_atomic_writes_across_threads_and_processes(tmp_path):
    payload = b"shared" * 1000
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda _: _concurrent_disk_writer(tmp_path, payload), range(4)))
    processes = [
        mp.get_context("spawn").Process(target=_concurrent_disk_writer, args=(tmp_path, payload))
        for _ in range(3)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=15)
        if process.is_alive():
            process.kill()
            process.join()
            pytest.fail("disk writer did not finish")
        assert process.exitcode == 0
    assert not list(tmp_path.rglob(".tmp-*"))
