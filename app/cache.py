"""
app/cache.py — In-memory LRU query cache for the RAG pipeline.

Caches (question -> RAG result) to avoid redundant LLM and embedding
API calls for identical queries.  Thread-safe via a simple RLock.

Usage
-----
    from app.cache import QueryCache

        cache = QueryCache(maxsize=256, ttl_seconds=3600)

            result = cache.get("What is RAG?")      # None on a miss
                cache.set("What is RAG?", rag_result)   # store the result
                    cache.invalidate("What is RAG?")        # remove one entry
                        cache.clear()                           # flush everything
                            stats = cache.stats()                   # hit/miss counters
                            """

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Optional

logger = logging.getLogger(__name__)


class QueryCache:
      """Thread-safe LRU cache with optional TTL for RAG query results.

          Parameters
              ----------
                  maxsize:
                          Maximum number of entries to keep.  Oldest entries are evicted
                                  when the cache is full (LRU policy).
                                      ttl_seconds:
                                              Time-to-live in seconds.  ``None`` means entries never expire.
                                                  """

    def __init__(self, maxsize: int = 256, ttl_seconds: Optional[float] = 3600) -> None:
              if maxsize < 1:
                            raise ValueError("maxsize must be >= 1")
                        self._maxsize = maxsize
        self._ttl = ttl_seconds
        self._store: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, question: str) -> Optional[Any]:
              """Return the cached result for *question*, or ``None`` on a miss."""
        key = self._make_key(question)
        with self._lock:
                      if key not in self._store:
                                        self._misses += 1
                                        return None

                      value, stored_at = self._store[key]

            if self._is_expired(stored_at):
                              del self._store[key]
                              self._misses += 1
                              logger.debug("Cache TTL expired for key %s", key[:8])
                              return None

            # Move to end (most-recently used)
            self._store.move_to_end(key)
            self._hits += 1
            logger.debug("Cache hit for key %s", key[:8])
            return value

    def set(self, question: str, result: Any) -> None:
              """Store *result* under *question*, evicting LRU entry if needed."""
        key = self._make_key(question)
        with self._lock:
                      if key in self._store:
                                        self._store.move_to_end(key)
                                    self._store[key] = (result, time.monotonic())

            while len(self._store) > self._maxsize:
                              evicted_key, _ = self._store.popitem(last=False)
                              logger.debug("Cache evicted LRU key %s", evicted_key[:8])

    def invalidate(self, question: str) -> bool:
              """Remove a single entry.  Returns ``True`` if the entry existed."""
        key = self._make_key(question)
        with self._lock:
                      if key in self._store:
                                        del self._store[key]
                                        return True
                                    return False

    def clear(self) -> None:
              """Remove all cached entries and reset counters."""
        with self._lock:
                      self._store.clear()
            self._hits = 0
            self._misses = 0
            logger.info("Query cache cleared")

    def stats(self) -> dict:
              """Return a snapshot of cache statistics."""
        with self._lock:
                      total = self._hits + self._misses
            hit_rate = self._hits / total if total else 0.0
            return {
                              "size": len(self._store),
                              "maxsize": self._maxsize,
                              "hits": self._hits,
                              "misses": self._misses,
                              "hit_rate": round(hit_rate, 4),
                              "ttl_seconds": self._ttl,
            }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_key(question: str) -> str:
              """Normalise and hash a question string into a stable cache key."""
        normalised = question.strip().lower()
        return hashlib.sha256(normalised.encode()).hexdigest()

    def _is_expired(self, stored_at: float) -> bool:
              if self._ttl is None:
                            return False
                        return (time.monotonic() - stored_at) > self._ttl
