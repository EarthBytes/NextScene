import time

from app.services.user_cache import UserCache


def test_user_cache_stores_and_returns_data():
    cache = UserCache(max_size=10, ttl_seconds=60)
    cache.set(1, [10, 20], {10, 20, 30})
    entry = cache.get(1)
    assert entry is not None
    assert entry.history == [10, 20]
    assert entry.seen_items == {10, 20, 30}


def test_user_cache_expires_entries():
    cache = UserCache(max_size=10, ttl_seconds=0)
    cache.set(1, [10], {10})
    time.sleep(0.01)
    assert cache.get(1) is None


def test_user_cache_invalidates_user():
    cache = UserCache(max_size=10, ttl_seconds=60)
    cache.set(1, [10], {10})
    cache.invalidate(1)
    assert cache.get(1) is None


def test_user_cache_evicts_oldest_when_full():
    cache = UserCache(max_size=2, ttl_seconds=60)
    cache.set(1, [1], {1})
    cache.set(2, [2], {2})
    cache.set(3, [3], {3})
    assert cache.get(1) is None
    assert cache.get(2) is not None
    assert cache.get(3) is not None
