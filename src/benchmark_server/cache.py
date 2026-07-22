"""Content-addressed byte cache (front-end).

LRU + reference-counting + a byte budget. A request's referenced keys are
*pinned* (refcount > 0) for its whole run so a concurrent request cannot evict
them mid-flight. Stores canonical bytes only; it is a transparent optimization.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class _Entry:
    data: bytes
    refcount: int = 0


class ByteCache:
    def __init__(self, capacity_bytes: int, max_object_bytes: int | None = None) -> None:
        self._cap = capacity_bytes
        # Don't cache an object larger than this (default: 25% of capacity) so one
        # huge blob can't evict the whole working set. It still works, just isn't
        # cached (re-uploaded each time).
        self._max_obj = max_object_bytes if max_object_bytes is not None else capacity_bytes // 4
        self._entries: "OrderedDict[str, _Entry]" = OrderedDict()
        self._size = 0
        self._lock = threading.Lock()

    def get(self, key: str) -> bytes | None:
        with self._lock:
            e = self._entries.get(key)
            if e is None:
                return None
            self._entries.move_to_end(key)  # most-recently-used
            return e.data

    def put(self, key: str, data: bytes) -> None:
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                return
            if len(data) > self._max_obj:
                return  # too big to cache; caller keeps using it this request
            self._entries[key] = _Entry(data=data)
            self._size += len(data)
            self._evict_locked()

    def missing(self, keys: list[str]) -> list[str]:
        with self._lock:
            return [k for k in keys if k not in self._entries]

    def pin(self, keys: list[str]) -> None:
        with self._lock:
            for k in keys:
                e = self._entries.get(k)
                if e is not None:
                    e.refcount += 1

    def unpin(self, keys: list[str]) -> None:
        with self._lock:
            for k in keys:
                e = self._entries.get(k)
                if e is not None and e.refcount > 0:
                    e.refcount -= 1

    def _evict_locked(self) -> None:
        # Drop least-recently-used entries with refcount == 0 until under budget.
        # If everything left is pinned, temporarily exceed the budget.
        for key in list(self._entries.keys()):
            if self._size <= self._cap:
                break
            e = self._entries[key]
            if e.refcount == 0:
                del self._entries[key]
                self._size -= len(e.data)

    # introspection (tests)
    @property
    def size_bytes(self) -> int:
        return self._size

    def __contains__(self, key: str) -> bool:
        return key in self._entries
