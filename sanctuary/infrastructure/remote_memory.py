"""Remote memory storage — ChromaDB on a separate server.

Implements the same store interface as InMemoryStore and legacy MemoryManager,
but connects to a remote ChromaDB instance over HTTP. Falls back to a local
cache when the remote server is unreachable, ensuring the cognitive cycle
never stalls on a network failure.

Phase 8 of Sanctuary development.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import chromadb.errors

logger = logging.getLogger(__name__)

# Errors expected when the remote ChromaDB server is unreachable or the
# request fails for transport/operational reasons. Caught at fallback
# boundaries so the cognitive cycle can degrade to local cache instead
# of crashing. Programming errors (AttributeError, KeyError, etc.) are
# intentionally NOT in this set — those should propagate.
_REMOTE_TRANSIENT_ERRORS = (
    ConnectionError,
    TimeoutError,
    OSError,
    chromadb.errors.ChromaError,
)


@dataclass
class RemoteMemoryConfig:
    """Configuration for remote ChromaDB memory storage."""

    host: str = "localhost"
    port: int = 8100
    collection_prefix: str = "sanctuary"
    use_ssl: bool = False
    auth_token: Optional[str] = None
    timeout_seconds: float = 10.0
    max_retries: int = 2
    local_cache_enabled: bool = True
    local_cache_max_entries: int = 500
    health_check_interval: float = 30.0


class LocalCache:
    """In-memory write-ahead cache for when the remote store is unreachable.

    Entries written here are replayed to the remote store when connectivity
    returns.  Read queries fall back to this cache on remote failure.
    """

    def __init__(self, max_entries: int = 500):
        self._entries: list[dict] = []
        self._pending_writes: list[dict] = []
        self._max_entries = max_entries

    def store(self, entry: dict) -> None:
        """Cache an entry locally and mark it as pending for remote sync."""
        self._entries.append(entry)
        self._pending_writes.append(entry)
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries:]

    async def recall(
        self,
        query: str,
        n_results: int = 5,
        min_significance: Optional[int] = None,
    ) -> list[dict]:
        """Substring-match recall from local cache (same as InMemoryStore)."""
        query_lower = query.lower()
        results = []
        for entry in reversed(self._entries):
            sig = entry.get("significance", entry.get("significance_score", 5))
            if min_significance and sig < min_significance:
                continue
            content = str(entry.get("content", "")).lower()
            tags = " ".join(str(t) for t in entry.get("tags", []))
            if query_lower in content or query_lower in tags.lower():
                results.append(entry)
            if len(results) >= n_results:
                break
        return results

    def drain_pending(self) -> list[dict]:
        """Return and clear all entries pending remote sync."""
        pending = list(self._pending_writes)
        self._pending_writes.clear()
        return pending

    @property
    def pending_count(self) -> int:
        return len(self._pending_writes)

    @property
    def entry_count(self) -> int:
        return len(self._entries)


class RemoteMemoryStore:
    """ChromaDB-backed remote memory store with local cache fallback.

    Usage::

        store = RemoteMemoryStore(RemoteMemoryConfig(host="memory-server", port=8100))
        await store.connect()

        # Use like any other store
        store.store({"content": "...", "significance": 5, "tags": ["test"]})
        results = await store.recall("search query", n_results=5)

        # Sync cached writes when remote comes back
        await store.sync_pending()
    """

    def __init__(self, config: Optional[RemoteMemoryConfig] = None):
        self._config = config or RemoteMemoryConfig()
        self._client = None
        self._collections: dict = {}
        self._connected = False
        self._last_health_check = 0.0
        self._consecutive_failures = 0

        self._cache = LocalCache(
            max_entries=self._config.local_cache_max_entries,
        ) if self._config.local_cache_enabled else None

        logger.info(
            "RemoteMemoryStore initialized (host=%s:%d, cache=%s)",
            self._config.host,
            self._config.port,
            "enabled" if self._cache else "disabled",
        )

    async def connect(self) -> bool:
        """Connect to the remote ChromaDB server.

        Returns True if connection succeeds, False otherwise.
        Does not raise — failure is handled via fallback to local cache.
        """
        try:
            import chromadb

            protocol = "https" if self._config.use_ssl else "http"
            settings = chromadb.config.Settings(anonymized_telemetry=False)

            headers = {}
            if self._config.auth_token:
                headers["Authorization"] = f"Bearer {self._config.auth_token}"

            self._client = chromadb.HttpClient(
                host=self._config.host,
                port=self._config.port,
                ssl=self._config.use_ssl,
                headers=headers if headers else None,
                settings=settings,
            )

            # Verify connectivity
            self._client.heartbeat()

            # Get or create collections
            prefix = self._config.collection_prefix
            for name in ("episodic_memory", "semantic_memory", "procedural_memory"):
                collection_name = f"{prefix}_{name}"
                self._collections[name] = self._client.get_or_create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"},
                )

            self._connected = True
            self._consecutive_failures = 0
            self._last_health_check = time.time()
            logger.info(
                "Connected to remote ChromaDB at %s://%s:%d",
                protocol, self._config.host, self._config.port,
            )
            return True

        except _REMOTE_TRANSIENT_ERRORS as e:
            self._connected = False
            logger.warning("Failed to connect to remote ChromaDB: %s", e)
            return False

    def store(self, entry: dict) -> None:
        """Store a memory entry.

        Writes to the remote server if connected, always caches locally
        when the cache is enabled.
        """
        if self._connected:
            try:
                self._store_remote(entry)
                self._consecutive_failures = 0
            except _REMOTE_TRANSIENT_ERRORS as e:
                self._consecutive_failures += 1
                logger.warning("Remote store failed, caching locally: %s", e)
                if self._consecutive_failures >= self._config.max_retries:
                    self._connected = False
                    logger.error(
                        "Remote store marked disconnected after %d failures",
                        self._consecutive_failures,
                    )

        if self._cache is not None:
            self._cache.store(entry)

    async def recall(
        self,
        query: str,
        n_results: int = 5,
        min_significance: Optional[int] = None,
    ) -> list[dict]:
        """Retrieve memories matching the query.

        Tries the remote server first, falls back to local cache.
        """
        if self._connected:
            try:
                results = self._recall_remote(query, n_results, min_significance)
                self._consecutive_failures = 0
                return results
            except _REMOTE_TRANSIENT_ERRORS as e:
                self._consecutive_failures += 1
                logger.warning("Remote recall failed, falling back to cache: %s", e)
                if self._consecutive_failures >= self._config.max_retries:
                    self._connected = False

        if self._cache is not None:
            return await self._cache.recall(query, n_results, min_significance)

        return []

    async def sync_pending(self) -> int:
        """Replay locally cached writes to the remote server.

        Returns the number of entries successfully synced.
        """
        if self._cache is None or not self._connected:
            return 0

        pending = self._cache.drain_pending()
        if not pending:
            return 0

        synced = 0
        failed = []
        for entry in pending:
            try:
                self._store_remote(entry)
                synced += 1
            except _REMOTE_TRANSIENT_ERRORS as e:
                logger.warning("Sync failed for entry: %s", e)
                failed.append(entry)

        # Re-queue anything that failed
        for entry in failed:
            self._cache._pending_writes.append(entry)

        if synced:
            logger.info("Synced %d/%d pending entries to remote", synced, len(pending))
        return synced

    async def health_check(self) -> dict:
        """Check remote server health and return status."""
        now = time.time()
        self._last_health_check = now

        status = {
            "connected": self._connected,
            "host": f"{self._config.host}:{self._config.port}",
            "consecutive_failures": self._consecutive_failures,
            "cache_entries": self._cache.entry_count if self._cache else 0,
            "pending_sync": self._cache.pending_count if self._cache else 0,
        }

        if self._client is not None:
            try:
                self._client.heartbeat()
                status["remote_healthy"] = True
                if not self._connected:
                    self._connected = True
                    self._consecutive_failures = 0
                    logger.info("Remote ChromaDB reconnected")
            except _REMOTE_TRANSIENT_ERRORS:
                status["remote_healthy"] = False

        # Collection counts
        if self._connected:
            try:
                status["collections"] = {
                    name: col.count()
                    for name, col in self._collections.items()
                }
            except _REMOTE_TRANSIENT_ERRORS:
                status["collections"] = {}

        return status

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def entry_count(self) -> int:
        """Total entries across all remote collections (or cache if disconnected)."""
        if self._connected:
            try:
                return sum(col.count() for col in self._collections.values())
            except _REMOTE_TRANSIENT_ERRORS:
                pass
        return self._cache.entry_count if self._cache else 0

    # -----------------------------------------------------------------
    # Internal: remote operations
    # -----------------------------------------------------------------

    def _store_remote(self, entry: dict) -> None:
        """Write an entry to the appropriate remote collection."""
        entry_type = entry.get("type", "episodic")
        collection_key = f"{entry_type}_memory"
        collection = self._collections.get(collection_key)
        if collection is None:
            collection = self._collections.get("episodic_memory")

        import uuid

        doc_id = entry.get("id", str(uuid.uuid4()))
        content = entry.get("content", "")

        metadata = {
            k: v for k, v in entry.items()
            if k not in ("content", "id") and isinstance(v, (str, int, float, bool))
        }
        # ChromaDB doesn't support list metadata; serialize tags
        if "tags" in entry and isinstance(entry["tags"], list):
            metadata["tags"] = ",".join(str(t) for t in entry["tags"])

        collection.upsert(
            documents=[content],
            metadatas=[metadata],
            ids=[doc_id],
        )

    def _recall_remote(
        self,
        query: str,
        n_results: int,
        min_significance: Optional[int],
    ) -> list[dict]:
        """Query the remote collections and return results."""
        all_results = []

        for name, collection in self._collections.items():
            try:
                if collection.count() == 0:
                    continue
                response = collection.query(
                    query_texts=[query],
                    n_results=min(n_results, collection.count()),
                )

                docs = response.get("documents", [[]])[0]
                metas = response.get("metadatas", [[]])[0]
                ids = response.get("ids", [[]])[0]

                for doc, meta, doc_id in zip(docs, metas, ids):
                    sig = meta.get("significance", meta.get("significance_score", 5))
                    if isinstance(sig, str):
                        sig = int(sig)
                    if min_significance and sig < min_significance:
                        continue

                    result = {
                        "id": doc_id,
                        "content": doc,
                        "significance": sig,
                        **{k: v for k, v in meta.items() if k != "content"},
                    }
                    # Deserialize tags
                    if "tags" in result and isinstance(result["tags"], str):
                        result["tags"] = result["tags"].split(",")
                    all_results.append(result)
            except _REMOTE_TRANSIENT_ERRORS as e:
                logger.warning("Query failed for collection %s: %s", name, e)

        # Sort by significance descending, take top n
        all_results.sort(
            key=lambda r: r.get("significance", 0), reverse=True,
        )
        return all_results[:n_results]
