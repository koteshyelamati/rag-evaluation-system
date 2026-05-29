"""Tests for app/cache.py — QueryCache LRU cache."""

import time
import pytest
from app.cache import QueryCache


class TestQueryCacheBasics:
      """Core get/set behaviour."""

    def test_miss_returns_none(self):
              cache = QueryCache(maxsize=10)
              assert cache.get("something not cached") is None

    def test_set_then_get_returns_value(self):
              cache = QueryCache(maxsize=10)
              payload = {"answer": "42", "source_documents": []}
              cache.set("What is the answer?", payload)
              result = cache.get("What is the answer?")
              assert result == payload

    def test_case_insensitive_normalisation(self):
              cache = QueryCache(maxsize=10)
              cache.set("What is RAG?", {"answer": "retrieval augmented generation"})
              # Different capitalisation should still hit
              assert cache.get("what is rag?") is not None
              assert cache.get("WHAT IS RAG?") is not None

    def test_whitespace_normalisation(self):
              cache = QueryCache(maxsize=10)
              cache.set("  What is RAG?  ", {"answer": "rag"})
              assert cache.get("What is RAG?") is not None

    def test_different_questions_dont_collide(self):
              cache = QueryCache(maxsize=10)
              cache.set("What is RAG?", {"answer": "retrieval augmented generation"})
              cache.set("What is a transformer?", {"answer": "attention is all you need"})
              r1 = cache.get("What is RAG?")
              r2 = cache.get("What is a transformer?")
              assert r1["answer"] != r2["answer"]

    def test_overwrite_existing_key(self):
              cache = QueryCache(maxsize=10)
              cache.set("What is ML?", {"answer": "v1"})
              cache.set("What is ML?", {"answer": "v2"})
              assert cache.get("What is ML?")["answer"] == "v2"


class TestQueryCacheLRUEviction:
      """LRU eviction when maxsize is exceeded."""

    def test_lru_evicts_oldest_when_full(self):
              cache = QueryCache(maxsize=3)
              cache.set("q1", "a1")
              cache.set("q2", "a2")
              cache.set("q3", "a3")
              # Add a fourth entry — q1 should be evicted
              cache.set("q4", "a4")
              assert cache.get("q1") is None
              assert cache.get("q2") == "a2"
              assert cache.get("q3") == "a3"
              assert cache.get("q4") == "a4"

    def test_access_updates_recency(self):
              cache = QueryCache(maxsize=3)
              cache.set("q1", "a1")
              cache.set("q2", "a2")
              cache.set("q3", "a3")
              # Touch q1 so it becomes most-recently used
              cache.get("q1")
              # Adding q4 should evict q2 (now the LRU)
              cache.set("q4", "a4")
              assert cache.get("q1") == "a1"  # still present
        assert cache.get("q2") is None  # evicted
        assert cache.get("q3") == "a3"
        assert cache.get("q4") == "a4"

    def test_maxsize_one(self):
              cache = QueryCache(maxsize=1)
              cache.set("q1", "a1")
              cache.set("q2", "a2")
              assert cache.get("q1") is None
              assert cache.get("q2") == "a2"

    def test_invalid_maxsize_raises(self):
              with pytest.raises(ValueError):
                            QueryCache(maxsize=0)


class TestQueryCacheTTL:
      """TTL expiry behaviour."""

    def test_entry_expired_after_ttl(self):
              cache = QueryCache(maxsize=10, ttl_seconds=0.05)
              cache.set("q1", "v1")
              time.sleep(0.1)
              assert cache.get("q1") is None

    def test_entry_valid_before_ttl(self):
              cache = QueryCache(maxsize=10, ttl_seconds=5)
              cache.set("q1", "v1")
              assert cache.get("q1") == "v1"

    def test_none_ttl_never_expires(self):
              cache = QueryCache(maxsize=10, ttl_seconds=None)
              cache.set("q1", "v1")
              # No sleep needed — just verify the flag works
              assert cache.get("q1") == "v1"

    def test_expired_entry_counted_as_miss(self):
              cache = QueryCache(maxsize=10, ttl_seconds=0.05)
              cache.set("q1", "v1")
              time.sleep(0.1)
              cache.get("q1")  # miss (expired)
        stats = cache.stats()
        assert stats["misses"] >= 1


class TestQueryCacheInvalidateAndClear:
      """invalidate() and clear() operations."""

    def test_invalidate_removes_entry(self):
              cache = QueryCache(maxsize=10)
              cache.set("q1", "v1")
              assert cache.invalidate("q1") is True
              assert cache.get("q1") is None

    def test_invalidate_missing_key_returns_false(self):
              cache = QueryCache(maxsize=10)
              assert cache.invalidate("nonexistent") is False

    def test_clear_removes_all_entries(self):
              cache = QueryCache(maxsize=10)
              for i in range(5):
                            cache.set(f"q{i}", f"v{i}")
                        cache.clear()
        for i in range(5):
                      assert cache.get(f"q{i}") is None

    def test_clear_resets_stats(self):
              cache = QueryCache(maxsize=10)
        cache.set("q1", "v1")
        cache.get("q1")
        cache.clear()
        stats = cache.stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["size"] == 0


class TestQueryCacheStats:
      """stats() accuracy."""

    def test_initial_stats(self):
              cache = QueryCache(maxsize=32, ttl_seconds=600)
        stats = cache.stats()
        assert stats["size"] == 0
        assert stats["maxsize"] == 32
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0
        assert stats["ttl_seconds"] == 600

    def test_hit_and_miss_counted(self):
              cache = QueryCache(maxsize=10)
        cache.set("q1", "v1")
        cache.get("q1")   # hit
        cache.get("q2")   # miss
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_size_reflects_entries(self):
              cache = QueryCache(maxsize=10)
        for i in range(4):
                      cache.set(f"q{i}", f"v{i}")
                  assert cache.stats()["size"] == 4
