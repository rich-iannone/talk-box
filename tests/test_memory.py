"""Tests for talk_box.memory module."""

import time

from talk_box.memory import (
    LongTermMemory,
    MemoryEntry,
    MemoryStore,
    MemoryTier,
    ShortTermMemory,
    WorkingMemory,
)


# ---------------------------------------------------------------------------
# MemoryTier enum
# ---------------------------------------------------------------------------


class TestMemoryTier:
    def test_values(self):
        assert MemoryTier.WORKING.value == "working"
        assert MemoryTier.SHORT_TERM.value == "short_term"
        assert MemoryTier.LONG_TERM.value == "long_term"

    def test_all_members(self):
        assert len(MemoryTier) == 3


# ---------------------------------------------------------------------------
# MemoryEntry
# ---------------------------------------------------------------------------


class TestMemoryEntry:
    def test_basic(self):
        e = MemoryEntry(key="k", value="v", tier=MemoryTier.WORKING)
        assert e.key == "k"
        assert e.value == "v"
        assert e.tier == MemoryTier.WORKING
        assert e.ttl is None
        assert e.tags == ()
        assert e.metadata == {}

    def test_not_expired_without_ttl(self):
        e = MemoryEntry(key="k", value="v", tier=MemoryTier.WORKING, timestamp=time.time())
        assert e.is_expired is False

    def test_not_expired_within_ttl(self):
        e = MemoryEntry(
            key="k", value="v", tier=MemoryTier.SHORT_TERM, timestamp=time.time(), ttl=3600
        )
        assert e.is_expired is False

    def test_expired_past_ttl(self):
        e = MemoryEntry(
            key="k", value="v", tier=MemoryTier.SHORT_TERM, timestamp=time.time() - 100, ttl=1
        )
        assert e.is_expired is True

    def test_frozen(self):
        e = MemoryEntry(key="k", value="v", tier=MemoryTier.WORKING)
        try:
            e.key = "new"  # type: ignore[misc]
            assert False, "Should be frozen"
        except Exception:
            pass


# ---------------------------------------------------------------------------
# WorkingMemory
# ---------------------------------------------------------------------------


class TestWorkingMemory:
    def test_set_and_get(self):
        wm = WorkingMemory()
        wm.set("name", "Alice")
        assert wm.get("name") == "Alice"

    def test_get_missing_returns_default(self):
        wm = WorkingMemory()
        assert wm.get("missing") is None
        assert wm.get("missing", "fallback") == "fallback"

    def test_has(self):
        wm = WorkingMemory()
        assert wm.has("x") is False
        wm.set("x", 1)
        assert wm.has("x") is True

    def test_delete(self):
        wm = WorkingMemory()
        wm.set("x", 1)
        assert wm.delete("x") is True
        assert wm.has("x") is False
        assert wm.delete("x") is False

    def test_keys(self):
        wm = WorkingMemory()
        wm.set("a", 1)
        wm.set("b", 2)
        assert sorted(wm.keys()) == ["a", "b"]

    def test_entries(self):
        wm = WorkingMemory()
        wm.set("x", 42)
        entries = wm.entries()
        assert len(entries) == 1
        assert entries[0].key == "x"
        assert entries[0].value == 42
        assert entries[0].tier == MemoryTier.WORKING

    def test_search_by_tags(self):
        wm = WorkingMemory()
        wm.set("a", 1, tags=("user", "pref"))
        wm.set("b", 2, tags=("system",))
        wm.set("c", 3, tags=("user",))
        results = wm.search(tags=["user"])
        assert len(results) == 2
        keys = {e.key for e in results}
        assert keys == {"a", "c"}

    def test_search_no_filter(self):
        wm = WorkingMemory()
        wm.set("a", 1)
        wm.set("b", 2)
        assert len(wm.search()) == 2

    def test_clear(self):
        wm = WorkingMemory()
        wm.set("a", 1)
        wm.set("b", 2)
        wm.clear()
        assert len(wm) == 0

    def test_len(self):
        wm = WorkingMemory()
        assert len(wm) == 0
        wm.set("a", 1)
        assert len(wm) == 1

    def test_overwrite(self):
        wm = WorkingMemory()
        wm.set("x", 1)
        wm.set("x", 2)
        assert wm.get("x") == 2
        assert len(wm) == 1

    def test_complex_values(self):
        wm = WorkingMemory()
        wm.set("data", {"nested": [1, 2, 3]})
        assert wm.get("data") == {"nested": [1, 2, 3]}


# ---------------------------------------------------------------------------
# ShortTermMemory
# ---------------------------------------------------------------------------


class TestShortTermMemory:
    def test_set_and_get(self):
        stm = ShortTermMemory()
        stm.set("topic", "Python")
        assert stm.get("topic") == "Python"

    def test_get_missing_returns_default(self):
        stm = ShortTermMemory()
        assert stm.get("missing") is None
        assert stm.get("missing", "fallback") == "fallback"

    def test_has(self):
        stm = ShortTermMemory()
        assert stm.has("x") is False
        stm.set("x", 1)
        assert stm.has("x") is True

    def test_delete(self):
        stm = ShortTermMemory()
        stm.set("x", 1)
        assert stm.delete("x") is True
        assert stm.has("x") is False
        assert stm.delete("x") is False

    def test_expired_entry_returns_default(self):
        stm = ShortTermMemory()
        stm.set("x", 1, ttl=0.001)
        time.sleep(0.01)
        assert stm.get("x") is None

    def test_expired_entry_has_returns_false(self):
        stm = ShortTermMemory()
        stm.set("x", 1, ttl=0.001)
        time.sleep(0.01)
        assert stm.has("x") is False

    def test_keys_excludes_expired(self):
        stm = ShortTermMemory()
        stm.set("alive", 1, ttl=3600)
        stm.set("dead", 2, ttl=0.001)
        time.sleep(0.01)
        assert stm.keys() == ["alive"]

    def test_entries(self):
        stm = ShortTermMemory()
        stm.set("x", 42)
        entries = stm.entries()
        assert len(entries) == 1
        assert entries[0].tier == MemoryTier.SHORT_TERM
        assert entries[0].ttl is not None

    def test_max_entries_eviction(self):
        stm = ShortTermMemory(max_entries=3)
        stm.set("a", 1)
        stm.set("b", 2)
        stm.set("c", 3)
        stm.set("d", 4)  # Should evict oldest
        assert len(stm) == 3
        assert stm.has("a") is False  # Oldest evicted
        assert stm.has("d") is True

    def test_default_ttl(self):
        stm = ShortTermMemory(default_ttl=1800)
        assert stm.default_ttl == 1800
        stm.set("x", 1)
        entries = stm.entries()
        assert entries[0].ttl == 1800

    def test_custom_ttl_overrides_default(self):
        stm = ShortTermMemory(default_ttl=3600)
        stm.set("x", 1, ttl=60)
        entries = stm.entries()
        assert entries[0].ttl == 60

    def test_search_by_tags(self):
        stm = ShortTermMemory()
        stm.set("a", 1, tags=("topic",))
        stm.set("b", 2, tags=("meta",))
        results = stm.search(tags=["topic"])
        assert len(results) == 1
        assert results[0].key == "a"

    def test_clear(self):
        stm = ShortTermMemory()
        stm.set("a", 1)
        stm.clear()
        assert len(stm) == 0

    def test_max_entries_property(self):
        stm = ShortTermMemory(max_entries=50)
        assert stm.max_entries == 50


# ---------------------------------------------------------------------------
# LongTermMemory
# ---------------------------------------------------------------------------


class TestLongTermMemory:
    def test_set_and_get(self):
        ltm = LongTermMemory(":memory:")
        ltm.set("pref", {"theme": "dark"})
        assert ltm.get("pref") == {"theme": "dark"}
        ltm.close()

    def test_get_missing_returns_default(self):
        ltm = LongTermMemory(":memory:")
        assert ltm.get("missing") is None
        assert ltm.get("missing", "fallback") == "fallback"
        ltm.close()

    def test_has(self):
        ltm = LongTermMemory(":memory:")
        assert ltm.has("x") is False
        ltm.set("x", 1)
        assert ltm.has("x") is True
        ltm.close()

    def test_delete(self):
        ltm = LongTermMemory(":memory:")
        ltm.set("x", 1)
        assert ltm.delete("x") is True
        assert ltm.has("x") is False
        assert ltm.delete("x") is False
        ltm.close()

    def test_keys(self):
        ltm = LongTermMemory(":memory:")
        ltm.set("b", 2)
        ltm.set("a", 1)
        assert ltm.keys() == ["a", "b"]  # Sorted by key
        ltm.close()

    def test_entries(self):
        ltm = LongTermMemory(":memory:")
        ltm.set("x", 42, tags=("test",))
        entries = ltm.entries()
        assert len(entries) == 1
        assert entries[0].key == "x"
        assert entries[0].value == 42
        assert entries[0].tier == MemoryTier.LONG_TERM
        assert entries[0].tags == ("test",)
        ltm.close()

    def test_overwrite(self):
        ltm = LongTermMemory(":memory:")
        ltm.set("x", 1)
        ltm.set("x", 2)
        assert ltm.get("x") == 2
        assert len(ltm) == 1
        ltm.close()

    def test_search_by_tags(self):
        ltm = LongTermMemory(":memory:")
        ltm.set("a", 1, tags=("user", "pref"))
        ltm.set("b", 2, tags=("system",))
        ltm.set("c", 3, tags=("user",))
        results = ltm.search(tags=["user"])
        assert len(results) == 2
        keys = {e.key for e in results}
        assert keys == {"a", "c"}
        ltm.close()

    def test_search_by_prefix(self):
        ltm = LongTermMemory(":memory:")
        ltm.set("user:name", "Alice")
        ltm.set("user:age", 30)
        ltm.set("system:version", "1.0")
        results = ltm.search(prefix="user:")
        assert len(results) == 2
        keys = {e.key for e in results}
        assert keys == {"user:name", "user:age"}
        ltm.close()

    def test_search_by_prefix_and_tags(self):
        ltm = LongTermMemory(":memory:")
        ltm.set("user:name", "Alice", tags=("pii",))
        ltm.set("user:age", 30, tags=("stats",))
        results = ltm.search(prefix="user:", tags=["pii"])
        assert len(results) == 1
        assert results[0].key == "user:name"
        ltm.close()

    def test_metadata(self):
        ltm = LongTermMemory(":memory:")
        ltm.set("x", 1, metadata={"source": "test"})
        entries = ltm.entries()
        assert entries[0].metadata == {"source": "test"}
        ltm.close()

    def test_clear(self):
        ltm = LongTermMemory(":memory:")
        ltm.set("a", 1)
        ltm.set("b", 2)
        ltm.clear()
        assert len(ltm) == 0
        ltm.close()

    def test_len(self):
        ltm = LongTermMemory(":memory:")
        assert len(ltm) == 0
        ltm.set("a", 1)
        assert len(ltm) == 1
        ltm.close()

    def test_path_property(self):
        ltm = LongTermMemory(":memory:")
        assert ltm.path == ":memory:"
        ltm.close()

    def test_complex_values(self):
        ltm = LongTermMemory(":memory:")
        ltm.set("data", {"list": [1, "two", 3.0], "nested": {"a": True}})
        result = ltm.get("data")
        assert result == {"list": [1, "two", 3.0], "nested": {"a": True}}
        ltm.close()

    def test_file_persistence(self, tmp_path):
        db_path = tmp_path / "test.db"
        ltm1 = LongTermMemory(db_path)
        ltm1.set("key", "value")
        ltm1.close()

        ltm2 = LongTermMemory(db_path)
        assert ltm2.get("key") == "value"
        ltm2.close()


# ---------------------------------------------------------------------------
# MemoryStore
# ---------------------------------------------------------------------------


class TestMemoryStore:
    def test_remember_working(self):
        store = MemoryStore()
        store.remember("x", 1, tier=MemoryTier.WORKING)
        assert store.working.get("x") == 1
        store.close()

    def test_remember_short_term(self):
        store = MemoryStore()
        store.remember("x", 1, tier=MemoryTier.SHORT_TERM)
        assert store.short_term.get("x") == 1
        store.close()

    def test_remember_long_term(self):
        store = MemoryStore()
        store.remember("x", 1, tier=MemoryTier.LONG_TERM)
        assert store.long_term.get("x") == 1
        store.close()

    def test_recall_searches_all_tiers(self):
        store = MemoryStore()
        store.remember("w", 1, tier=MemoryTier.WORKING)
        store.remember("s", 2, tier=MemoryTier.SHORT_TERM)
        store.remember("l", 3, tier=MemoryTier.LONG_TERM)
        assert store.recall("w") == 1
        assert store.recall("s") == 2
        assert store.recall("l") == 3
        store.close()

    def test_recall_priority_working_first(self):
        store = MemoryStore()
        store.remember("x", "working", tier=MemoryTier.WORKING)
        store.remember("x", "long", tier=MemoryTier.LONG_TERM)
        assert store.recall("x") == "working"
        store.close()

    def test_recall_priority_short_term_second(self):
        store = MemoryStore()
        store.remember("x", "short", tier=MemoryTier.SHORT_TERM)
        store.remember("x", "long", tier=MemoryTier.LONG_TERM)
        assert store.recall("x") == "short"
        store.close()

    def test_recall_specific_tier(self):
        store = MemoryStore()
        store.remember("x", "working", tier=MemoryTier.WORKING)
        store.remember("x", "long", tier=MemoryTier.LONG_TERM)
        assert store.recall("x", tier=MemoryTier.LONG_TERM) == "long"
        store.close()

    def test_recall_missing_returns_default(self):
        store = MemoryStore()
        assert store.recall("missing") is None
        assert store.recall("missing", default="fallback") == "fallback"
        store.close()

    def test_forget_specific_tier(self):
        store = MemoryStore()
        store.remember("x", 1, tier=MemoryTier.WORKING)
        assert store.forget("x", tier=MemoryTier.WORKING) is True
        assert store.recall("x") is None
        store.close()

    def test_forget_all_tiers(self):
        store = MemoryStore()
        store.remember("x", 1, tier=MemoryTier.WORKING)
        store.remember("x", 2, tier=MemoryTier.LONG_TERM)
        assert store.forget("x") is True
        assert store.recall("x") is None
        store.close()

    def test_forget_missing_returns_false(self):
        store = MemoryStore()
        assert store.forget("missing") is False
        store.close()

    def test_search_all_tiers(self):
        store = MemoryStore()
        store.remember("a", 1, tier=MemoryTier.WORKING, tags=("t",))
        store.remember("b", 2, tier=MemoryTier.SHORT_TERM, tags=("t",))
        store.remember("c", 3, tier=MemoryTier.LONG_TERM, tags=("t",))
        results = store.search(tags=["t"])
        assert len(results) == 3
        store.close()

    def test_search_specific_tier(self):
        store = MemoryStore()
        store.remember("a", 1, tier=MemoryTier.WORKING, tags=("t",))
        store.remember("b", 2, tier=MemoryTier.LONG_TERM, tags=("t",))
        results = store.search(tags=["t"], tier=MemoryTier.WORKING)
        assert len(results) == 1
        assert results[0].key == "a"
        store.close()

    def test_remember_with_ttl(self):
        store = MemoryStore()
        store.remember("x", 1, tier=MemoryTier.SHORT_TERM, ttl=0.001)
        time.sleep(0.01)
        assert store.recall("x") is None
        store.close()

    def test_remember_with_metadata(self):
        store = MemoryStore()
        store.remember("x", 1, tier=MemoryTier.LONG_TERM, metadata={"src": "test"})
        entries = store.long_term.entries()
        assert entries[0].metadata == {"src": "test"}
        store.close()

    def test_config_passthrough(self):
        store = MemoryStore(
            short_term_max_entries=50,
            short_term_default_ttl=1800,
        )
        assert store.short_term.max_entries == 50
        assert store.short_term.default_ttl == 1800
        store.close()

    def test_direct_tier_access(self):
        store = MemoryStore()
        store.working.set("direct", "access")
        assert store.working.get("direct") == "access"
        store.close()
