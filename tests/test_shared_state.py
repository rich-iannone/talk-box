import threading

from talk_box.shared_state import SharedState, StateChange


# ---------------------------------------------------------------------------
# StateChange
# ---------------------------------------------------------------------------


class TestStateChange:
    def test_frozen(self):
        sc = StateChange(
            namespace="ns",
            key="k",
            old_value=None,
            new_value=1,
            timestamp=0.0,
            agent="bot",
        )
        try:
            sc.key = "other"  # type: ignore[misc]
            assert False, "Should be frozen"
        except Exception:
            pass

    def test_fields(self):
        sc = StateChange(
            namespace="ns",
            key="k",
            old_value="old",
            new_value="new",
            timestamp=1.0,
            agent="a",
        )
        assert sc.namespace == "ns"
        assert sc.key == "k"
        assert sc.old_value == "old"
        assert sc.new_value == "new"
        assert sc.timestamp == 1.0
        assert sc.agent == "a"

    def test_default_agent(self):
        sc = StateChange(namespace="ns", key="k", old_value=None, new_value=1, timestamp=0.0)
        assert sc.agent == ""


# ---------------------------------------------------------------------------
# Core get/set/delete
# ---------------------------------------------------------------------------


class TestCoreOperations:
    def test_set_and_get(self):
        s = SharedState()
        s.set("key", "value")
        assert s.get("key") == "value"

    def test_get_default(self):
        s = SharedState()
        assert s.get("missing") is None
        assert s.get("missing", "fallback") == "fallback"

    def test_set_overwrites(self):
        s = SharedState()
        s.set("key", "v1")
        s.set("key", "v2")
        assert s.get("key") == "v2"

    def test_delete_existing(self):
        s = SharedState()
        s.set("key", "value")
        assert s.delete("key") is True
        assert s.get("key") is None

    def test_delete_missing(self):
        s = SharedState()
        assert s.delete("missing") is False

    def test_has(self):
        s = SharedState()
        assert s.has("key") is False
        s.set("key", "value")
        assert s.has("key") is True

    def test_keys(self):
        s = SharedState()
        s.set("a", 1)
        s.set("b", 2)
        assert sorted(s.keys()) == ["a", "b"]

    def test_len(self):
        s = SharedState()
        assert len(s) == 0
        s.set("a", 1)
        s.set("b", 2)
        assert len(s) == 2

    def test_complex_values(self):
        s = SharedState()
        s.set("data", {"nested": [1, 2, 3]})
        assert s.get("data") == {"nested": [1, 2, 3]}


# ---------------------------------------------------------------------------
# Namespaces
# ---------------------------------------------------------------------------


class TestNamespaces:
    def test_namespace_isolation(self):
        s = SharedState()
        s.set("key", "global_val")
        s.set("key", "agent_val", namespace="agent1")
        assert s.get("key") == "global_val"
        assert s.get("key", namespace="agent1") == "agent_val"

    def test_delete_from_namespace(self):
        s = SharedState()
        s.set("key", "v1", namespace="ns1")
        s.set("key", "v2", namespace="ns2")
        s.delete("key", namespace="ns1")
        assert s.get("key", namespace="ns1") is None
        assert s.get("key", namespace="ns2") == "v2"

    def test_has_in_namespace(self):
        s = SharedState()
        s.set("key", "value", namespace="ns")
        assert s.has("key", namespace="ns") is True
        assert s.has("key") is False  # Not in global

    def test_keys_per_namespace(self):
        s = SharedState()
        s.set("a", 1)
        s.set("b", 2, namespace="ns")
        assert s.keys() == ["a"]
        assert s.keys(namespace="ns") == ["b"]

    def test_namespaces_list(self):
        s = SharedState()
        assert "__global__" in s.namespaces()
        s.set("x", 1, namespace="agent1")
        s.set("y", 2, namespace="agent2")
        ns = s.namespaces()
        assert "agent1" in ns
        assert "agent2" in ns

    def test_len_across_namespaces(self):
        s = SharedState()
        s.set("a", 1)
        s.set("b", 2, namespace="ns")
        assert len(s) == 2


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


class TestSnapshot:
    def test_snapshot_returns_copy(self):
        s = SharedState()
        s.set("key", "value")
        snap = s.snapshot()
        assert snap["__global__"]["key"] == "value"
        # Modifying snapshot shouldn't affect state
        snap["__global__"]["key"] = "modified"
        assert s.get("key") == "value"

    def test_snapshot_multiple_namespaces(self):
        s = SharedState()
        s.set("g", 1)
        s.set("a", 2, namespace="agent")
        snap = s.snapshot()
        assert snap["__global__"]["g"] == 1
        assert snap["agent"]["a"] == 2

    def test_snapshot_empty(self):
        s = SharedState()
        snap = s.snapshot()
        assert snap == {"__global__": {}}


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------


class TestClear:
    def test_clear_all(self):
        s = SharedState()
        s.set("a", 1)
        s.set("b", 2, namespace="ns")
        s.clear()
        assert len(s) == 0
        assert s.get("a") is None
        assert s.get("b", namespace="ns") is None

    def test_clear_namespace(self):
        s = SharedState()
        s.set("a", 1)
        s.set("b", 2, namespace="ns")
        s.clear(namespace="ns")
        assert s.get("a") == 1
        assert s.get("b", namespace="ns") is None

    def test_clear_unknown_namespace(self):
        s = SharedState()
        s.clear(namespace="nonexistent")  # Should not raise


# ---------------------------------------------------------------------------
# Change history
# ---------------------------------------------------------------------------


class TestHistory:
    def test_set_tracked(self):
        s = SharedState()
        s.set("key", "value", agent="bot")
        h = s.history
        assert len(h) == 1
        assert h[0].key == "key"
        assert h[0].old_value is None
        assert h[0].new_value == "value"
        assert h[0].agent == "bot"
        assert h[0].namespace == "__global__"

    def test_overwrite_tracked(self):
        s = SharedState()
        s.set("key", "v1")
        s.set("key", "v2")
        h = s.history
        assert len(h) == 2
        assert h[1].old_value == "v1"
        assert h[1].new_value == "v2"

    def test_delete_tracked(self):
        s = SharedState()
        s.set("key", "value")
        s.delete("key", agent="cleanup")
        h = s.history
        assert len(h) == 2
        assert h[1].old_value == "value"
        assert h[1].new_value is None
        assert h[1].agent == "cleanup"

    def test_history_disabled(self):
        s = SharedState(track_history=False)
        s.set("key", "value")
        s.delete("key")
        assert s.history == []

    def test_history_returns_copy(self):
        s = SharedState()
        s.set("key", "value")
        h1 = s.history
        h1.clear()
        assert len(s.history) == 1  # Original unaffected

    def test_namespace_in_history(self):
        s = SharedState()
        s.set("key", "v", namespace="agent1", agent="agent1")
        h = s.history
        assert h[0].namespace == "agent1"


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_writes(self):
        s = SharedState()
        errors = []

        def writer(ns, count):
            try:
                for i in range(count):
                    s.set(f"key_{i}", i, namespace=ns)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(f"ns_{t}", 50)) for t in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(s) == 500  # 10 threads × 50 keys

    def test_concurrent_read_write(self):
        s = SharedState()
        s.set("counter", 0)
        errors = []

        def reader():
            try:
                for _ in range(100):
                    s.get("counter")
            except Exception as e:
                errors.append(e)

        def writer():
            try:
                for i in range(100):
                    s.set("counter", i)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=reader),
            threading.Thread(target=writer),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors


# ---------------------------------------------------------------------------
# Top-level imports
# ---------------------------------------------------------------------------


class TestTopLevelImport:
    def test_import_shared_state(self):
        import talk_box as tb

        assert hasattr(tb, "SharedState")
        assert hasattr(tb, "StateChange")

    def test_in_all(self):
        import talk_box

        assert "SharedState" in talk_box.__all__
        assert "StateChange" in talk_box.__all__
