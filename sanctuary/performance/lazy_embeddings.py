"""Lazy embedding computation — only compute embeddings when actually needed.

Instead of computing embeddings for every piece of text eagerly, this module
provides a cache-on-demand pattern. Embeddings are computed only when
first accessed, then cached for reuse. Includes LRU eviction and TTL-based
expiration.

This avoids wasting compute on embeddings that are never used (e.g., percepts
that get processed before any similarity query needs them).
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class LazyEmbeddingConfig:
    """Configuration for lazy embedding cache."""

    max_cache_size: int = 1000  # Maximum cached embeddings
    ttl_seconds: float = 3600.0  # Time-to-live for cache entries (1 hour)
    enabled: bool = True


@dataclass
class _CacheEntry:
    """Internal cache entry."""

    embedding: list[float]
    created_at: float
    access_count: int = 0


class LazyEmbeddingCache:
    """Cache-on-demand embedding computation with LRU eviction.

    Usage::

        def compute_embedding(text: str) -> list[float]:
            return model.encode(text)

        cache = LazyEmbeddingCache(compute_fn=compute_embedding)

        # Embeddings computed lazily on first access
        emb1 = cache.get("Hello world")  # Computes
        emb2 = cache.get("Hello world")  # Cache hit
    """

    def __init__(
        self,
        compute_fn: Optional[Callable[[str], list[float]]] = None,
        config: Optional[LazyEmbeddingConfig] = None,
    ):
        self.config = config or LazyEmbeddingConfig()
        if compute_fn is None:
            # No silent fallback. A hash-based pseudo-embedding would let
            # production compute meaningless vectors and corrupt every
            # similarity / retrieval built on this cache without ever
            # failing -- the worst kind of bug, because it looks like it
            # works. Require a real embedder explicitly. Tests that only
            # exercise cache mechanics (eviction, TTL, hit-rate) should
            # pass compute_fn=LazyEmbeddingCache.deterministic_test_embedding.
            raise ValueError(
                "LazyEmbeddingCache requires an explicit compute_fn (a real "
                "embedding function). There is no default: a fabricated "
                "embedding would silently corrupt similarity. For tests, pass "
                "compute_fn=LazyEmbeddingCache.deterministic_test_embedding."
            )
        self._compute_fn = compute_fn
        self._cache: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0

    def get(self, text: str) -> list[float]:
        """Get embedding for text, computing lazily if not cached."""
        if not self.config.enabled:
            return self._compute_fn(text)

        key = self._cache_key(text)

        # Check cache
        if key in self._cache:
            entry = self._cache[key]
            # Check TTL
            if time.time() - entry.created_at < self.config.ttl_seconds:
                entry.access_count += 1
                self._cache.move_to_end(key)  # LRU: most recent at end
                self._hits += 1
                return entry.embedding
            else:
                # Expired
                del self._cache[key]

        # Cache miss — compute
        self._misses += 1
        embedding = self._compute_fn(text)

        # Store in cache
        self._cache[key] = _CacheEntry(
            embedding=embedding,
            created_at=time.time(),
        )

        # Evict if over capacity
        while len(self._cache) > self.config.max_cache_size:
            self._cache.popitem(last=False)  # Remove oldest
            self._evictions += 1

        return embedding

    def get_batch(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for multiple texts."""
        return [self.get(text) for text in texts]

    def precompute(self, texts: list[str]) -> int:
        """Eagerly compute and cache embeddings for a list of texts.

        Returns the number of new computations (cache misses).
        """
        computed = 0
        for text in texts:
            key = self._cache_key(text)
            if key not in self._cache:
                self.get(text)
                computed += 1
        return computed

    def invalidate(self, text: str) -> bool:
        """Remove a specific entry from the cache."""
        key = self._cache_key(text)
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> int:
        """Clear the entire cache. Returns number of entries removed."""
        count = len(self._cache)
        self._cache.clear()
        return count

    def get_hit_rate(self) -> float:
        """Get cache hit rate (0 to 1)."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def get_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "cache_size": len(self._cache),
            "max_size": self.config.max_cache_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.get_hit_rate(), 3),
            "evictions": self._evictions,
            "enabled": self.config.enabled,
        }

    # -- Internal --

    @staticmethod
    def _cache_key(text: str) -> str:
        """Generate a cache key from text."""
        return hashlib.md5(text.encode()).hexdigest()

    @staticmethod
    def deterministic_test_embedding(text: str) -> list[float]:
        """Deterministic hash-based pseudo-embedding -- FOR TESTS ONLY.

        This is NOT a real embedding: it is an MD5-derived 16-dim vector
        with no semantic meaning. It exists so cache-mechanics tests
        (eviction, TTL, hit-rate) have a cheap, deterministic embedder
        without loading a model. It must be passed *explicitly* as
        ``compute_fn`` -- it is never a silent default, because using it
        in production would fabricate similarity. Naming it loudly so no
        one mistakes it for a usable embedder.
        """
        h = hashlib.md5(text.encode()).digest()
        return [b / 255.0 for b in h]  # 16-dimensional normalized vector
