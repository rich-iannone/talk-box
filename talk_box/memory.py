"""Memory tiers: working, short-term, and long-term memory for conversations."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class MemoryTier(Enum):
    """Which memory tier an entry belongs to.

    Attributes
    ----------
    WORKING
        In-conversation memory, lost when the conversation ends.
    SHORT_TERM
        Recent session memory with TTL and max entries.
    LONG_TERM
        Persistent memory backed by SQLite.
    """

    WORKING = "working"
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"


@dataclass(frozen=True)
class MemoryEntry:
    """A single memory entry across any tier.

    Parameters
    ----------
    key
        Unique identifier for this memory within its tier.
    value
        The stored value (any JSON-serializable object).
    tier
        Which memory tier this entry belongs to.
    timestamp
        Unix timestamp when the entry was created or last updated.
    ttl
        Time-to-live in seconds. ``None`` means no expiration.
    tags
        Tags for categorization and search.
    metadata
        Additional metadata about this entry.
    """

    key: str
    value: Any
    tier: MemoryTier
    timestamp: float = 0.0
    ttl: float | None = None
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Whether this entry has exceeded its TTL."""
        if self.ttl is None:
            return False
        return (time.time() - self.timestamp) > self.ttl


# ---------------------------------------------------------------------------
# Working Memory
# ---------------------------------------------------------------------------


class WorkingMemory:
    """In-conversation key-value memory, lost when the conversation ends.

    This is the fastest tier — a simple dict-backed store for transient
    data needed during a single conversation turn or session.

    Examples
    --------
    ```python
    import talk_box as tb

    wm = tb.WorkingMemory()
    wm.set("user_name", "Alice")
    wm.get("user_name")  # "Alice"
    wm.keys()             # ["user_name"]
    wm.clear()
    ```
    """

    def __init__(self) -> None:
        self._store: dict[str, MemoryEntry] = {}

    def set(self, key: str, value: Any, *, tags: tuple[str, ...] = ()) -> None:
        """Store a value in working memory.

        Parameters
        ----------
        key
            The key to store under.
        value
            The value to store.
        tags
            Optional tags for categorization.
        """
        self._store[key] = MemoryEntry(
            key=key,
            value=value,
            tier=MemoryTier.WORKING,
            timestamp=time.time(),
            tags=tags,
        )

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from working memory.

        Parameters
        ----------
        key
            The key to look up.
        default
            Value to return if key is not found.

        Returns
        -------
        Any
            The stored value, or ``default`` if not found.
        """
        entry = self._store.get(key)
        if entry is None:
            return default
        return entry.value

    def has(self, key: str) -> bool:
        """Check if a key exists in working memory."""
        return key in self._store

    def delete(self, key: str) -> bool:
        """Remove a key from working memory.

        Returns
        -------
        bool
            ``True`` if the key existed and was removed, ``False`` otherwise.
        """
        if key in self._store:
            del self._store[key]
            return True
        return False

    def keys(self) -> list[str]:
        """List all keys in working memory."""
        return list(self._store.keys())

    def entries(self) -> list[MemoryEntry]:
        """List all entries in working memory."""
        return list(self._store.values())

    def search(self, *, tags: list[str] | None = None) -> list[MemoryEntry]:
        """Search working memory by tags.

        Parameters
        ----------
        tags
            Filter to entries that have all of these tags.

        Returns
        -------
        list[MemoryEntry]
            Matching entries.
        """
        results = list(self._store.values())
        if tags:
            tag_set = set(tags)
            results = [e for e in results if tag_set.issubset(set(e.tags))]
        return results

    def clear(self) -> None:
        """Remove all entries from working memory."""
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


# ---------------------------------------------------------------------------
# Short-Term Memory
# ---------------------------------------------------------------------------


class ShortTermMemory:
    """Recent session memory with TTL and max-entry eviction.

    Entries expire after their TTL elapses. When the store exceeds
    ``max_entries``, the oldest entries are evicted first.

    Parameters
    ----------
    max_entries
        Maximum number of entries to retain (default 100).
    default_ttl
        Default time-to-live in seconds for new entries (default 3600 = 1 hour).

    Examples
    --------
    ```python
    import talk_box as tb

    stm = tb.ShortTermMemory(max_entries=50, default_ttl=1800)
    stm.set("last_topic", "Python decorators")
    stm.get("last_topic")  # "Python decorators"
    ```
    """

    def __init__(
        self,
        *,
        max_entries: int = 100,
        default_ttl: float = 3600.0,
    ) -> None:
        self._store: dict[str, MemoryEntry] = {}
        self._max_entries = max_entries
        self._default_ttl = default_ttl

    @property
    def max_entries(self) -> int:
        """Maximum number of entries."""
        return self._max_entries

    @property
    def default_ttl(self) -> float:
        """Default TTL in seconds."""
        return self._default_ttl

    def set(
        self,
        key: str,
        value: Any,
        *,
        ttl: float | None = None,
        tags: tuple[str, ...] = (),
    ) -> None:
        """Store a value in short-term memory.

        Parameters
        ----------
        key
            The key to store under.
        value
            The value to store.
        ttl
            Time-to-live in seconds. Uses ``default_ttl`` if not specified.
        tags
            Optional tags for categorization.
        """
        effective_ttl = ttl if ttl is not None else self._default_ttl

        self._store[key] = MemoryEntry(
            key=key,
            value=value,
            tier=MemoryTier.SHORT_TERM,
            timestamp=time.time(),
            ttl=effective_ttl,
            tags=tags,
        )
        self._evict()

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from short-term memory.

        Expired entries are treated as missing and automatically removed.

        Parameters
        ----------
        key
            The key to look up.
        default
            Value to return if key is not found or expired.

        Returns
        -------
        Any
            The stored value, or ``default`` if not found/expired.
        """
        entry = self._store.get(key)
        if entry is None:
            return default
        if entry.is_expired:
            del self._store[key]
            return default
        return entry.value

    def has(self, key: str) -> bool:
        """Check if a key exists and is not expired."""
        entry = self._store.get(key)
        if entry is None:
            return False
        if entry.is_expired:
            del self._store[key]
            return False
        return True

    def delete(self, key: str) -> bool:
        """Remove a key from short-term memory.

        Returns
        -------
        bool
            ``True`` if the key existed and was removed.
        """
        if key in self._store:
            del self._store[key]
            return True
        return False

    def keys(self) -> list[str]:
        """List all non-expired keys."""
        self._purge_expired()
        return list(self._store.keys())

    def entries(self) -> list[MemoryEntry]:
        """List all non-expired entries."""
        self._purge_expired()
        return list(self._store.values())

    def search(self, *, tags: list[str] | None = None) -> list[MemoryEntry]:
        """Search short-term memory by tags.

        Parameters
        ----------
        tags
            Filter to entries that have all of these tags.

        Returns
        -------
        list[MemoryEntry]
            Matching non-expired entries.
        """
        self._purge_expired()
        results = list(self._store.values())
        if tags:
            tag_set = set(tags)
            results = [e for e in results if tag_set.issubset(set(e.tags))]
        return results

    def clear(self) -> None:
        """Remove all entries."""
        self._store.clear()

    def _purge_expired(self) -> None:
        """Remove all expired entries."""
        expired = [k for k, v in self._store.items() if v.is_expired]
        for k in expired:
            del self._store[k]

    def _evict(self) -> None:
        """Evict oldest entries if over max_entries (after purging expired)."""
        self._purge_expired()
        if len(self._store) <= self._max_entries:
            return
        # Sort by timestamp, evict oldest
        sorted_keys = sorted(self._store.keys(), key=lambda k: self._store[k].timestamp)
        to_remove = len(self._store) - self._max_entries
        for k in sorted_keys[:to_remove]:
            del self._store[k]

    def __len__(self) -> int:
        self._purge_expired()
        return len(self._store)


# ---------------------------------------------------------------------------
# Long-Term Memory
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    timestamp REAL NOT NULL,
    tags TEXT NOT NULL DEFAULT '[]',
    metadata TEXT NOT NULL DEFAULT '{}'
)
"""

_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories (timestamp)
"""


class LongTermMemory:
    """Persistent memory backed by SQLite.

    Entries persist across sessions and are stored in a SQLite database file.
    Supports search by key prefix, tags, and full listing.

    Parameters
    ----------
    path
        Path to the SQLite database file. Use ``":memory:"`` for an in-memory
        database (useful for testing).

    Examples
    --------
    ```python
    import talk_box as tb

    ltm = tb.LongTermMemory("my_memory.db")
    ltm.set("preference", {"theme": "dark", "language": "en"})
    ltm.get("preference")  # {"theme": "dark", "language": "en"}
    ltm.search(tags=["user"])
    ltm.close()
    ```
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._path = str(path)
        self._conn = sqlite3.connect(self._path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_CREATE_TABLE_SQL)
        self._conn.execute(_CREATE_INDEX_SQL)
        self._conn.commit()

    @property
    def path(self) -> str:
        """Path to the SQLite database file."""
        return self._path

    def set(
        self,
        key: str,
        value: Any,
        *,
        tags: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store a value in long-term memory.

        Parameters
        ----------
        key
            The key to store under.
        value
            The value to store (must be JSON-serializable).
        tags
            Optional tags for categorization.
        metadata
            Optional metadata dictionary.
        """
        now = time.time()
        value_json = json.dumps(value)
        tags_json = json.dumps(list(tags))
        meta_json = json.dumps(metadata or {})

        self._conn.execute(
            "INSERT OR REPLACE INTO memories (key, value, timestamp, tags, metadata) "
            "VALUES (?, ?, ?, ?, ?)",
            (key, value_json, now, tags_json, meta_json),
        )
        self._conn.commit()

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from long-term memory.

        Parameters
        ----------
        key
            The key to look up.
        default
            Value to return if key is not found.

        Returns
        -------
        Any
            The stored value, or ``default`` if not found.
        """
        cursor = self._conn.execute("SELECT value FROM memories WHERE key = ?", (key,))
        row = cursor.fetchone()
        if row is None:
            return default
        return json.loads(row[0])

    def has(self, key: str) -> bool:
        """Check if a key exists in long-term memory."""
        cursor = self._conn.execute("SELECT 1 FROM memories WHERE key = ?", (key,))
        return cursor.fetchone() is not None

    def delete(self, key: str) -> bool:
        """Remove a key from long-term memory.

        Returns
        -------
        bool
            ``True`` if the key existed and was removed.
        """
        cursor = self._conn.execute("DELETE FROM memories WHERE key = ?", (key,))
        self._conn.commit()
        return cursor.rowcount > 0

    def keys(self) -> list[str]:
        """List all keys in long-term memory."""
        cursor = self._conn.execute("SELECT key FROM memories ORDER BY key")
        return [row[0] for row in cursor.fetchall()]

    def entries(self) -> list[MemoryEntry]:
        """List all entries in long-term memory."""
        cursor = self._conn.execute(
            "SELECT key, value, timestamp, tags, metadata FROM memories ORDER BY timestamp"
        )
        return [self._row_to_entry(row) for row in cursor.fetchall()]

    def search(
        self,
        *,
        tags: list[str] | None = None,
        prefix: str | None = None,
    ) -> list[MemoryEntry]:
        """Search long-term memory by tags and/or key prefix.

        Parameters
        ----------
        tags
            Filter to entries that have all of these tags.
        prefix
            Filter to entries whose key starts with this prefix.

        Returns
        -------
        list[MemoryEntry]
            Matching entries, sorted by timestamp.
        """
        query = "SELECT key, value, timestamp, tags, metadata FROM memories"
        conditions: list[str] = []
        params: list[Any] = []

        if prefix is not None:
            conditions.append("key LIKE ?")
            # Escape existing % and _ in prefix for LIKE pattern
            escaped = prefix.replace("%", r"\%").replace("_", r"\_")
            params.append(f"{escaped}%")

        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY timestamp"

        cursor = self._conn.execute(query, params)
        results = [self._row_to_entry(row) for row in cursor.fetchall()]

        # Tag filtering in Python (SQLite JSON support varies)
        if tags:
            tag_set = set(tags)
            results = [e for e in results if tag_set.issubset(set(e.tags))]

        return results

    def clear(self) -> None:
        """Remove all entries from long-term memory."""
        self._conn.execute("DELETE FROM memories")
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def _row_to_entry(self, row: tuple[Any, ...]) -> MemoryEntry:
        """Convert a database row to a MemoryEntry."""
        key, value_json, timestamp, tags_json, meta_json = row
        return MemoryEntry(
            key=key,
            value=json.loads(value_json),
            tier=MemoryTier.LONG_TERM,
            timestamp=timestamp,
            tags=tuple(json.loads(tags_json)),
            metadata=json.loads(meta_json),
        )

    def __len__(self) -> int:
        cursor = self._conn.execute("SELECT COUNT(*) FROM memories")
        return cursor.fetchone()[0]


# ---------------------------------------------------------------------------
# Unified Memory Store
# ---------------------------------------------------------------------------


class MemoryStore:
    """Unified memory interface across all three tiers.

    Provides a single API for storing and retrieving memories, with automatic
    tier selection and cross-tier search.

    Parameters
    ----------
    short_term_max_entries
        Maximum entries for the short-term tier (default 100).
    short_term_default_ttl
        Default TTL for short-term entries in seconds (default 3600).
    long_term_path
        Path to the SQLite database for long-term storage.
        Use ``":memory:"`` for testing (default).

    Examples
    --------
    ```python
    import talk_box as tb

    store = tb.MemoryStore(long_term_path="my_bot.db")

    # Store in different tiers
    store.remember("user_query", "What is Python?", tier=tb.MemoryTier.WORKING)
    store.remember("session_topic", "Python basics", tier=tb.MemoryTier.SHORT_TERM)
    store.remember("user_pref", {"theme": "dark"}, tier=tb.MemoryTier.LONG_TERM)

    # Recall from any tier
    store.recall("user_query")       # Searches working first
    store.recall("user_pref")        # Found in long-term

    # Search across tiers
    store.search(tags=["user"])

    store.close()
    ```
    """

    def __init__(
        self,
        *,
        short_term_max_entries: int = 100,
        short_term_default_ttl: float = 3600.0,
        long_term_path: str | Path = ":memory:",
    ) -> None:
        self.working = WorkingMemory()
        self.short_term = ShortTermMemory(
            max_entries=short_term_max_entries,
            default_ttl=short_term_default_ttl,
        )
        self.long_term = LongTermMemory(path=long_term_path)

    def remember(
        self,
        key: str,
        value: Any,
        *,
        tier: MemoryTier = MemoryTier.WORKING,
        ttl: float | None = None,
        tags: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store a memory in the specified tier.

        Parameters
        ----------
        key
            The key to store under.
        value
            The value to store.
        tier
            Which memory tier to use (default ``WORKING``).
        ttl
            Time-to-live in seconds (only used for ``SHORT_TERM``).
        tags
            Optional tags for categorization.
        metadata
            Optional metadata (only used for ``LONG_TERM``).
        """
        if tier == MemoryTier.WORKING:
            self.working.set(key, value, tags=tags)
        elif tier == MemoryTier.SHORT_TERM:
            kwargs: dict[str, Any] = {"tags": tags}
            if ttl is not None:
                kwargs["ttl"] = ttl
            self.short_term.set(key, value, **kwargs)
        elif tier == MemoryTier.LONG_TERM:
            self.long_term.set(key, value, tags=tags, metadata=metadata)

    def recall(self, key: str, *, tier: MemoryTier | None = None, default: Any = None) -> Any:
        """Retrieve a memory, searching tiers if no specific tier is given.

        When ``tier`` is ``None``, searches working → short-term → long-term
        and returns the first match found.

        Parameters
        ----------
        key
            The key to look up.
        tier
            Specific tier to search. ``None`` searches all tiers.
        default
            Value to return if not found in any tier.

        Returns
        -------
        Any
            The stored value, or ``default`` if not found.
        """
        if tier is not None:
            return self._get_from_tier(key, tier, default)

        # Search working → short-term → long-term
        sentinel = object()
        for t in (MemoryTier.WORKING, MemoryTier.SHORT_TERM, MemoryTier.LONG_TERM):
            result = self._get_from_tier(key, t, sentinel)
            if result is not sentinel:
                return result
        return default

    def forget(self, key: str, *, tier: MemoryTier | None = None) -> bool:
        """Remove a memory from the specified tier, or from all tiers.

        Parameters
        ----------
        key
            The key to remove.
        tier
            Specific tier to remove from. ``None`` removes from all tiers.

        Returns
        -------
        bool
            ``True`` if the key was found and removed from at least one tier.
        """
        if tier is not None:
            return self._delete_from_tier(key, tier)

        removed = False
        for t in MemoryTier:
            if self._delete_from_tier(key, t):
                removed = True
        return removed

    def search(
        self,
        *,
        tags: list[str] | None = None,
        tier: MemoryTier | None = None,
        prefix: str | None = None,
    ) -> list[MemoryEntry]:
        """Search memories across tiers.

        Parameters
        ----------
        tags
            Filter to entries with all of these tags.
        tier
            Specific tier to search. ``None`` searches all tiers.
        prefix
            Key prefix filter (only supported for long-term tier).

        Returns
        -------
        list[MemoryEntry]
            Matching entries from the specified tier(s).
        """
        results: list[MemoryEntry] = []

        if tier is None or tier == MemoryTier.WORKING:
            results.extend(self.working.search(tags=tags))
        if tier is None or tier == MemoryTier.SHORT_TERM:
            results.extend(self.short_term.search(tags=tags))
        if tier is None or tier == MemoryTier.LONG_TERM:
            results.extend(self.long_term.search(tags=tags, prefix=prefix))

        return results

    def close(self) -> None:
        """Close underlying resources (SQLite connection)."""
        self.long_term.close()

    def _get_from_tier(self, key: str, tier: MemoryTier, default: Any) -> Any:
        """Retrieve from a specific tier."""
        if tier == MemoryTier.WORKING:
            return self.working.get(key, default)
        elif tier == MemoryTier.SHORT_TERM:
            return self.short_term.get(key, default)
        else:
            return self.long_term.get(key, default)

    def _delete_from_tier(self, key: str, tier: MemoryTier) -> bool:
        """Delete from a specific tier."""
        if tier == MemoryTier.WORKING:
            return self.working.delete(key)
        elif tier == MemoryTier.SHORT_TERM:
            return self.short_term.delete(key)
        else:
            return self.long_term.delete(key)
