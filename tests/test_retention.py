from talk_box.memory import MemoryStore, MemoryTier
from talk_box.retention import RetentionPolicy, apply_retention


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store(**kwargs):
    """Create a MemoryStore with in-memory long-term storage."""
    return MemoryStore(long_term_path=":memory:", **kwargs)


# ---------------------------------------------------------------------------
# RetentionPolicy
# ---------------------------------------------------------------------------


class TestRetentionPolicy:
    def test_defaults(self):
        p = RetentionPolicy()
        assert p.remember_tags == []
        assert p.remember_keys == []
        assert p.forget_tags == []
        assert p.forget_keys == []
        assert p.tier is None

    def test_frozen(self):
        p = RetentionPolicy()
        try:
            p.tier = MemoryTier.WORKING  # type: ignore[misc]
            assert False, "Should be frozen"
        except Exception:
            pass

    def test_custom_values(self):
        p = RetentionPolicy(
            remember_tags=["identity"],
            remember_keys=["user_name"],
            forget_tags=["scratch"],
            forget_keys=["temp"],
            tier=MemoryTier.WORKING,
        )
        assert p.remember_tags == ["identity"]
        assert p.remember_keys == ["user_name"]
        assert p.forget_tags == ["scratch"]
        assert p.forget_keys == ["temp"]
        assert p.tier == MemoryTier.WORKING


# ---------------------------------------------------------------------------
# apply_retention — forget rules
# ---------------------------------------------------------------------------


class TestApplyRetentionForget:
    def test_forget_by_tag(self):
        store = _make_store()
        store.remember("keep_me", "v1", tier=MemoryTier.WORKING, tags=("identity",))
        store.remember("scratch", "v2", tier=MemoryTier.WORKING, tags=("scratch",))

        policy = RetentionPolicy(forget_tags=["scratch"])
        result = apply_retention(store, policy)

        assert "scratch" in result.removed
        assert "keep_me" in result.retained
        assert result.policy == "apply_retention"

    def test_forget_by_key(self):
        store = _make_store()
        store.remember("temp", "data", tier=MemoryTier.WORKING)
        store.remember("important", "data", tier=MemoryTier.WORKING)

        policy = RetentionPolicy(forget_keys=["temp"])
        result = apply_retention(store, policy)

        assert "temp" in result.removed
        assert "important" in result.retained

    def test_forget_across_tiers(self):
        store = _make_store()
        store.remember("w_scratch", "v", tier=MemoryTier.WORKING, tags=("scratch",))
        store.remember("s_scratch", "v", tier=MemoryTier.SHORT_TERM, tags=("scratch",))
        store.remember("w_keep", "v", tier=MemoryTier.WORKING, tags=("keep",))

        policy = RetentionPolicy(forget_tags=["scratch"])
        result = apply_retention(store, policy)

        assert "w_scratch" in result.removed
        assert "s_scratch" in result.removed
        assert "w_keep" in result.retained

    def test_forget_scoped_to_tier(self):
        store = _make_store()
        store.remember("scratch1", "v", tier=MemoryTier.WORKING, tags=("scratch",))
        store.remember("scratch2", "v", tier=MemoryTier.SHORT_TERM, tags=("scratch",))

        policy = RetentionPolicy(forget_tags=["scratch"], tier=MemoryTier.WORKING)
        result = apply_retention(store, policy)

        assert "scratch1" in result.removed
        # scratch2 is in SHORT_TERM, not affected
        assert "scratch2" not in result.removed


# ---------------------------------------------------------------------------
# apply_retention — remember rules
# ---------------------------------------------------------------------------


class TestApplyRetentionRemember:
    def test_remember_by_tag(self):
        store = _make_store()
        store.remember("name", "Alice", tier=MemoryTier.WORKING, tags=("identity",))
        store.remember("pref", "dark", tier=MemoryTier.WORKING, tags=("preference",))
        store.remember("scratch", "tmp", tier=MemoryTier.WORKING)

        policy = RetentionPolicy(remember_tags=["identity", "preference"])
        result = apply_retention(store, policy)

        assert "scratch" in result.removed
        assert "name" in result.retained
        assert "pref" in result.retained

    def test_remember_by_key(self):
        store = _make_store()
        store.remember("user_id", "123", tier=MemoryTier.WORKING)
        store.remember("temp", "data", tier=MemoryTier.WORKING)

        policy = RetentionPolicy(remember_keys=["user_id"])
        result = apply_retention(store, policy)

        assert "temp" in result.removed
        assert "user_id" in result.retained

    def test_remember_scoped_to_tier(self):
        store = _make_store()
        store.remember("w_name", "v", tier=MemoryTier.WORKING, tags=("identity",))
        store.remember("w_scratch", "v", tier=MemoryTier.WORKING)
        store.remember("s_other", "v", tier=MemoryTier.SHORT_TERM)

        policy = RetentionPolicy(
            remember_tags=["identity"],
            tier=MemoryTier.WORKING,
        )
        result = apply_retention(store, policy)

        assert "w_scratch" in result.removed
        assert "w_name" in result.retained
        # s_other is in SHORT_TERM, not in scope
        assert "s_other" not in result.removed


# ---------------------------------------------------------------------------
# apply_retention — combined forget + remember
# ---------------------------------------------------------------------------


class TestApplyRetentionCombined:
    def test_forget_then_remember(self):
        store = _make_store()
        store.remember("name", "Alice", tier=MemoryTier.WORKING, tags=("identity",))
        store.remember("debug", "trace", tier=MemoryTier.WORKING, tags=("debug",))
        store.remember("other", "data", tier=MemoryTier.WORKING)

        policy = RetentionPolicy(
            remember_tags=["identity"],
            forget_tags=["debug"],
        )
        result = apply_retention(store, policy)

        assert "debug" in result.removed
        assert "other" in result.removed
        assert "name" in result.retained

    def test_empty_policy_no_changes(self):
        store = _make_store()
        store.remember("a", "1", tier=MemoryTier.WORKING)
        store.remember("b", "2", tier=MemoryTier.WORKING)

        policy = RetentionPolicy()
        result = apply_retention(store, policy)

        assert result.removed == []
        assert set(result.retained) == {"a", "b"}
        assert result.policy == "apply_retention"


# ---------------------------------------------------------------------------
# PersonaDefinition with retention
# ---------------------------------------------------------------------------


class TestPersonaRetention:
    def test_persona_has_no_retention_by_default(self):
        from talk_box.personas import create_persona

        p = create_persona("test", persona_role="test role")
        assert p.retention is None

    def test_persona_with_retention_policy(self):
        from talk_box.personas import create_persona

        policy = RetentionPolicy(
            remember_tags=["identity", "preference"],
            forget_tags=["scratch"],
        )
        p = create_persona(
            "support_agent",
            persona_role="support specialist",
            retention=policy,
        )
        assert p.retention is policy
        assert p.retention.remember_tags == ["identity", "preference"]

    def test_apply_persona_retention(self):
        from talk_box.personas import create_persona

        policy = RetentionPolicy(
            remember_tags=["identity"],
            forget_tags=["debug"],
        )
        persona = create_persona(
            "helper",
            persona_role="helpful assistant",
            retention=policy,
        )

        store = _make_store()
        store.remember("user_name", "Alice", tier=MemoryTier.WORKING, tags=("identity",))
        store.remember("debug_info", "trace", tier=MemoryTier.WORKING, tags=("debug",))
        store.remember("scratch", "tmp", tier=MemoryTier.WORKING)

        result = apply_retention(store, persona.retention)

        assert "debug_info" in result.removed
        assert "scratch" in result.removed
        assert "user_name" in result.retained


# ---------------------------------------------------------------------------
# YAML round-trip
# ---------------------------------------------------------------------------


class TestRetentionYaml:
    def test_load_persona_with_retention(self, tmp_path):
        yaml_content = """\
name: yaml_test
display_name: "YAML Test"
category: technical
description: "A test persona with retention."
persona_role: "test role"
retention:
  remember_tags:
    - identity
    - preference
  forget_tags:
    - scratch
    - debug
"""
        yaml_file = tmp_path / "test_persona.yaml"
        yaml_file.write_text(yaml_content, encoding="utf-8")

        from talk_box.personas import load_persona

        persona = load_persona(yaml_file)
        assert persona.retention is not None
        assert persona.retention.remember_tags == ["identity", "preference"]
        assert persona.retention.forget_tags == ["scratch", "debug"]
        assert persona.retention.remember_keys == []
        assert persona.retention.forget_keys == []

    def test_load_persona_without_retention(self, tmp_path):
        yaml_content = """\
name: no_retention
display_name: "No Retention"
category: general
description: "A test persona without retention."
persona_role: "basic role"
"""
        yaml_file = tmp_path / "no_retention.yaml"
        yaml_file.write_text(yaml_content, encoding="utf-8")

        from talk_box.personas import load_persona

        persona = load_persona(yaml_file)
        assert persona.retention is None


# ---------------------------------------------------------------------------
# Import from top-level
# ---------------------------------------------------------------------------


class TestImport:
    def test_import_retention_policy(self):
        import talk_box.retention as retention

        assert hasattr(retention, "RetentionPolicy")
        assert hasattr(retention, "apply_retention")

    def test_in_all(self):
        import talk_box.retention

        assert "RetentionPolicy" in talk_box.retention.__all__
        assert "apply_retention" in talk_box.retention.__all__
