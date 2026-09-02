"""In-memory TTL cache for per-user recommendation data."""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class CachedUserData:
    history: list[int]
    seen_items: set[int]
    cached_at: float


class UserCache:
    """LRU cache with TTL for user history and seen-item sets."""

    def __init__(self, *, max_size: int = 1000, ttl_seconds: int = 300) -> None:
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._entries: OrderedDict[int, CachedUserData] = OrderedDict()

    def get(self, user_id: int) -> CachedUserData | None:
        entry = self._entries.get(user_id)
        if entry is None:
            return None
        if time.monotonic() - entry.cached_at > self.ttl_seconds:
            del self._entries[user_id]
            return None
        self._entries.move_to_end(user_id)
        return entry

    def set(self, user_id: int, history: list[int], seen_items: set[int]) -> None:
        self._entries[user_id] = CachedUserData(
            history=list(history),
            seen_items=set(seen_items),
            cached_at=time.monotonic(),
        )
        self._entries.move_to_end(user_id)
        while len(self._entries) > self.max_size:
            self._entries.popitem(last=False)

    def invalidate(self, user_id: int) -> None:
        self._entries.pop(user_id, None)

    def clear(self) -> None:
        self._entries.clear()

    @property
    def size(self) -> int:
        return len(self._entries)
