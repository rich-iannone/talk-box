"""Tests for talk_box.forgetting module."""

import time

from talk_box.forgetting import (
    PolicyResult,
    compress_after_n_turns,
    forget_after_resolution,
    retain_only,
)
from talk_box.memory import MemoryStore, MemoryTier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store(**kwargs):
    """Create a MemoryStore with in-memory long-term storage."""
    return MemoryStore(long_term_path=":memory:", **kwargs)


# ---------------------------------------------------------------------------
# PolicyResult
# ---------------------------------------------------------------------------


class TestPolicyResult:
    def test_counts(self):
        r = PolicyResult(removed=["a", "b"], retained=["c"], policy="test")
        assert r.removed_count == 2
        assert r.retained_count == 1
        assert r.policy == "test"

    def test_frozen(self):
        r = PolicyResult(removed=[], retained=[], policy="test")
        try:
            r.policy = "new"  # type: ignore[misc]
            assert False, "Should be frozen"
        except Exception:
            pass


# ---------------------------------------------------------------------------
# forget_after_resolution
# ---------------------------------------------------------------------------


class TestForgetAfterResolution:
    def test_removes_resolved_keys(self):
        store = _make_store()
        store.remember("q1", "pending", tier=MemoryTier.WORKING)
        store.remember("q2", "pending", tier=MemoryTier.WORKING)
        store.remember("user", "Alice", tier=MemoryTier.WORKING)

        result = forget_after_resolution(store, ["q1", "q2"])

        assert "q1" in result.removed
        assert "q2" in result.removed
        assert result.removed_count == 2
        assert "user" in result.retained
        assert store.recall("q1") is None
        assert store.recall("user") == "Alice"

    def test_nonexistent_key_not_in_removed(self):
        store = _make_store()
        store.remember("a", "1", tier=MemoryTier.WORKING)

        result = forget_after_resolution(store, ["nonexistent"])

        assert result.removed == []
        assert "a" in result.retained

    def test_specific_tier(self):
        store = _make_store()
        store.remember("key", "working", tier=MemoryTier.WORKING)
        store.remember("key", "short", tier=MemoryTier.SHORT_TERM)

        result = forget_after_resolution(store, ["key"], tier=MemoryTier.WORKING)

        assert "key" in result.removed
        # Still exists in short-term
        assert store.recall("key", tier=MemoryTier.SHORT_TERM) == "short"

    def test_removes_from_all_tiers(self):
        store = _make_store()
        store.remember("key", "w", tier=MemoryTier.WORKING)
        store.remember("key", "s", tier=MemoryTier.SHORT_TERM)
        store.remember("key", "l", tier=MemoryTier.LONG_TERM)

        result = forget_after_resolution(store, ["key"])

        assert result.removed_count == 1  # forget() returns True once for cross-tier
        assert store.recall("key") is None

    def test_empty_resolved_keys(self):
        store = _make_store()
        store.remember("a", "1", tier=MemoryTier.WORKING)

        result = forget_after_resolution(store, [])

        assert result.removed == []
        assert "a" in result.retained

    def test_policy_name(self):
        store = _make_store()
        result = forget_after_resolution(store, [])
        assert result.policy == "forget_after_resolution"

    def test_long_term_resolution(self):
        store = _make_store()
        store.remember("fact", "value", tier=MemoryTier.LONG_TERM)

        result = forget_after_resolution(store, ["fact"], tier=MemoryTier.LONG_TERM)

        assert "fact" in result.removed
        assert store.recall("fact", tier=MemoryTier.LONG_TERM) is None


# ---------------------------------------------------------------------------
# compress_after_n_turns
# ---------------------------------------------------------------------------


class TestCompressAfterNTurns:
    def test_no_compression_below_threshold(self):
        store = _make_store()
        for i in range(10):
            store.remember(f"k{i}", f"v{i}", tier=MemoryTier.WORKING)

        result = compress_after_n_turns(store, turn_count=3, max_entries=5)

        assert result.removed == []
        assert result.retained_count == 10  # Nothing removed

    def test_compresses_when_over_max(self):
        store = _make_store()
        for i in range(10):
            store.remember(f"k{i}", f"v{i}", tier=MemoryTier.WORKING)
            time.sleep(0.001)  # Ensure distinct timestamps

        result = compress_after_n_turns(store, turn_count=15, max_entries=5)

        assert result.removed_count == 5
        assert result.retained_count == 5
        assert len(store.working) == 5

    def test_removes_oldest_first(self):
        store = _make_store()
        store.remember("old", "first", tier=MemoryTier.WORKING)
        time.sleep(0.01)
        store.remember("mid", "second", tier=MemoryTier.WORKING)
        time.sleep(0.01)
        store.remember("new", "third", tier=MemoryTier.WORKING)

        result = compress_after_n_turns(store, turn_count=10, max_entries=1)

        assert "old" in result.removed
        assert "mid" in result.removed
        assert "new" in result.retained

    def test_preserve_tags(self):
        store = _make_store()
        store.remember("important", "keep", tier=MemoryTier.WORKING, tags=("critical",))
        time.sleep(0.001)
        store.remember("temp1", "discard", tier=MemoryTier.WORKING)
        time.sleep(0.001)
        store.remember("temp2", "discard", tier=MemoryTier.WORKING)

        result = compress_after_n_turns(
            store, turn_count=10, max_entries=1, preserve_tags=["critical"]
        )

        assert "important" not in result.removed
        assert store.recall("important") == "keep"

    def test_specific_tier(self):
        store = _make_store()
        for i in range(5):
            store.remember(f"w{i}", f"v{i}", tier=MemoryTier.WORKING)
        for i in range(5):
            store.remember(f"s{i}", f"v{i}", tier=MemoryTier.SHORT_TERM)

        result = compress_after_n_turns(
            store, turn_count=10, max_entries=3, tier=MemoryTier.WORKING
        )

        # Only working memory compressed
        assert len(store.working) == 3
        assert len(store.short_term) == 5  # Untouched

    def test_excludes_long_term_by_default(self):
        store = _make_store()
        for i in range(5):
            store.remember(f"lt{i}", f"v{i}", tier=MemoryTier.LONG_TERM)
        for i in range(5):
            store.remember(f"w{i}", f"v{i}", tier=MemoryTier.WORKING)

        result = compress_after_n_turns(store, turn_count=10, max_entries=3)

        # Long-term entries untouched
        assert len(store.long_term) == 5

    def test_long_term_when_explicit(self):
        store = _make_store()
        for i in range(5):
            store.remember(f"lt{i}", f"v{i}", tier=MemoryTier.LONG_TERM)
            time.sleep(0.001)

        result = compress_after_n_turns(
            store, turn_count=10, max_entries=2, tier=MemoryTier.LONG_TERM
        )

        assert len(store.long_term) == 2
        assert result.removed_count == 3

    def test_already_under_max(self):
        store = _make_store()
        store.remember("a", "1", tier=MemoryTier.WORKING)

        result = compress_after_n_turns(store, turn_count=10, max_entries=5)

        assert result.removed == []

    def test_policy_name(self):
        store = _make_store()
        result = compress_after_n_turns(store, turn_count=1, max_entries=5)
        assert result.policy == "compress_after_n_turns"


# ---------------------------------------------------------------------------
# retain_only
# ---------------------------------------------------------------------------


class TestRetainOnly:
    def test_retain_by_keys(self):
        store = _make_store()
        store.remember("keep1", "a", tier=MemoryTier.WORKING)
        store.remember("keep2", "b", tier=MemoryTier.WORKING)
        store.remember("remove1", "c", tier=MemoryTier.WORKING)

        result = retain_only(store, keys=["keep1", "keep2"])

        assert "remove1" in result.removed
        assert "keep1" in result.retained
        assert "keep2" in result.retained
        assert store.recall("keep1") == "a"
        assert store.recall("remove1") is None

    def test_retain_by_tags(self):
        store = _make_store()
        store.remember("user", "Alice", tier=MemoryTier.WORKING, tags=("identity",))
        store.remember("pref", "dark", tier=MemoryTier.WORKING, tags=("identity",))
        store.remember("temp", "scratch", tier=MemoryTier.WORKING)

        result = retain_only(store, tags=["identity"])

        assert "temp" in result.removed
        assert "user" in result.retained
        assert "pref" in result.retained

    def test_retain_by_keys_and_tags(self):
        store = _make_store()
        store.remember("k1", "v1", tier=MemoryTier.WORKING, tags=("keep",))
        store.remember("k2", "v2", tier=MemoryTier.WORKING)
        store.remember("k3", "v3", tier=MemoryTier.WORKING)

        result = retain_only(store, keys=["k2"], tags=["keep"])

        assert "k3" in result.removed
        assert "k1" in result.retained  # Kept by tag
        assert "k2" in result.retained  # Kept by key

    def test_specific_tier(self):
        store = _make_store()
        store.remember("w1", "a", tier=MemoryTier.WORKING)
        store.remember("w2", "b", tier=MemoryTier.WORKING)
        store.remember("s1", "c", tier=MemoryTier.SHORT_TERM)

        result = retain_only(store, keys=["w1"], tier=MemoryTier.WORKING)

        assert "w2" in result.removed
        assert len(store.working) == 1
        # Short-term untouched
        assert store.recall("s1", tier=MemoryTier.SHORT_TERM) == "c"

    def test_raises_without_keys_or_tags(self):
        store = _make_store()
        try:
            retain_only(store)
            assert False, "Should raise ValueError"
        except ValueError as e:
            assert "keys" in str(e).lower() or "tags" in str(e).lower()

    def test_all_tiers(self):
        store = _make_store()
        store.remember("w", "1", tier=MemoryTier.WORKING)
        store.remember("s", "2", tier=MemoryTier.SHORT_TERM)
        store.remember("l", "3", tier=MemoryTier.LONG_TERM)
        store.remember("keep", "4", tier=MemoryTier.WORKING, tags=("keep",))

        result = retain_only(store, tags=["keep"])

        assert "w" in result.removed
        assert "s" in result.removed
        assert "l" in result.removed
        assert "keep" in result.retained

    def test_empty_store(self):
        store = _make_store()
        result = retain_only(store, keys=["anything"])

        assert result.removed == []
        assert result.retained == []

    def test_policy_name(self):
        store = _make_store()
        result = retain_only(store, keys=["x"])
        assert result.policy == "retain_only"

    def test_long_term_retain(self):
        store = _make_store()
        store.remember("fact1", "important", tier=MemoryTier.LONG_TERM, tags=("core",))
        store.remember("fact2", "ephemeral", tier=MemoryTier.LONG_TERM)

        result = retain_only(store, tags=["core"], tier=MemoryTier.LONG_TERM)

        assert "fact2" in result.removed
        assert "fact1" in result.retained
        assert len(store.long_term) == 1
