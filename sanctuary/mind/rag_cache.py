"""
Caching system for Sanctuary's RAG context
"""
from typing import Dict, Any, Optional, List
import time
from pathlib import Path
import json
import asyncio
from dataclasses import dataclass
import threading

@dataclass
class CacheEntry:
    """Cache entry with data and metadata."""
    data: Any
    timestamp: float
    ttl: int  # Time to live in seconds

class RAGCache:
    """Caching system for RAG queries and contexts."""
    
    def __init__(self, cache_dir: Path, max_size: int = 1000, default_ttl: int = 3600):
        self.cache_dir = cache_dir
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.lock = threading.Lock()
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load persistent cache
        self._load_persistent_cache()
        
        # Start cleanup task
        self._start_cleanup_task()
    
    def _load_persistent_cache(self):
        """Load cached entries from disk.

        Uses JSON (not pickle) so loading an on-disk cache can never execute
        arbitrary code. Cached ``data`` must be JSON-serializable; anything that
        is not will simply not have been persisted (see _save_persistent_cache).
        """
        try:
            cache_file = self.cache_dir / "rag_cache.json"
            if cache_file.exists():
                with open(cache_file, 'r', encoding='utf-8') as f:
                    saved_cache = json.load(f)
                # Only load non-expired entries
                current_time = time.time()
                for key, entry in saved_cache.items():
                    if current_time - entry["timestamp"] < entry["ttl"]:
                        self.memory_cache[key] = CacheEntry(
                            data=entry["data"],
                            timestamp=entry["timestamp"],
                            ttl=entry["ttl"],
                        )
        except Exception as e:
            print(f"Error loading cache: {e}")

    def _save_persistent_cache(self):
        """Save cache entries to disk as JSON.

        JSON is used instead of pickle to eliminate arbitrary-code-execution on
        load. If any cached ``data`` is not JSON-serializable, the write fails
        closed here (logged, not persisted) rather than falling back to pickle.
        """
        try:
            cache_file = self.cache_dir / "rag_cache.json"
            serializable = {
                key: {
                    "data": entry.data,
                    "timestamp": entry.timestamp,
                    "ttl": entry.ttl,
                }
                for key, entry in self.memory_cache.items()
            }
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(serializable, f)
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def _generate_cache_key(self, query: str, context: Dict[str, Any]) -> str:
        """Generate a unique cache key."""
        context_str = json.dumps(context, sort_keys=True)
        return f"{query}::{context_str}"
    
    def get(self, query: str, context: Dict[str, Any] = None) -> Optional[Any]:
        """Get cached result for a query."""
        if context is None:
            context = {}
            
        key = self._generate_cache_key(query, context)
        
        with self.lock:
            entry = self.memory_cache.get(key)
            if entry is None:
                return None
                
            # Check if entry is expired
            if time.time() - entry.timestamp > entry.ttl:
                del self.memory_cache[key]
                return None
                
            return entry.data
    
    def set(self, query: str, data: Any, context: Dict[str, Any] = None, ttl: int = None):
        """Cache result for a query."""
        if context is None:
            context = {}
        if ttl is None:
            ttl = self.default_ttl
            
        key = self._generate_cache_key(query, context)
        
        with self.lock:
            # Enforce cache size limit
            if len(self.memory_cache) >= self.max_size:
                # Remove oldest entry
                oldest_key = min(
                    self.memory_cache.keys(),
                    key=lambda k: self.memory_cache[k].timestamp
                )
                del self.memory_cache[oldest_key]
            
            # Add new entry
            self.memory_cache[key] = CacheEntry(
                data=data,
                timestamp=time.time(),
                ttl=ttl
            )
            
            # Save to disk
            self._save_persistent_cache()
    
    def _cleanup_expired(self):
        """Remove expired entries."""
        current_time = time.time()
        with self.lock:
            expired_keys = [
                key for key, entry in self.memory_cache.items()
                if current_time - entry.timestamp > entry.ttl
            ]
            for key in expired_keys:
                del self.memory_cache[key]
            
            if expired_keys:
                self._save_persistent_cache()
    
    def _start_cleanup_task(self):
        """Start periodic cleanup of expired entries."""
        def cleanup_loop():
            while True:
                time.sleep(300)  # Run every 5 minutes
                self._cleanup_expired()
        
        cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        cleanup_thread.start()