import pytest

from talk_box.models import (
    CostTier,
    ModelProfile,
    _PROFILES,
    get_model_profile,
    list_models,
    model_profiles_table,
    register_model,
)


# ---------------------------------------------------------------------------
# ModelProfile tests
# ---------------------------------------------------------------------------


class TestModelProfile:
    def test_creation(self):
        p = ModelProfile(provider="openai", model="gpt-4o")
        assert p.provider == "openai"
        assert p.model == "gpt-4o"

    def test_key(self):
        p = ModelProfile(provider="anthropic", model="claude-sonnet-4-6")
        assert p.key == "anthropic:claude-sonnet-4-6"

    def test_name_with_display_name(self):
        p = ModelProfile(provider="openai", model="gpt-4o", display_name="GPT-4o")
        assert p.name == "GPT-4o"

    def test_name_fallback(self):
        p = ModelProfile(provider="openai", model="gpt-4o")
        assert p.name == "gpt-4o"

    def test_frozen(self):
        p = ModelProfile(provider="openai", model="gpt-4o")
        with pytest.raises(AttributeError):
            p.provider = "anthropic"  # type: ignore[misc]

    def test_supports_known(self):
        p = ModelProfile(
            provider="openai",
            model="gpt-4o",
            supports_tools=True,
            supports_vision=True,
            supports_structured_output=True,
            supports_streaming=True,
        )
        assert p.supports("tools") is True
        assert p.supports("vision") is True
        assert p.supports("structured_output") is True
        assert p.supports("streaming") is True

    def test_supports_false(self):
        p = ModelProfile(
            provider="deepseek",
            model="deepseek-reasoner",
            supports_tools=False,
            supports_vision=False,
        )
        assert p.supports("tools") is False
        assert p.supports("vision") is False

    def test_supports_none(self):
        p = ModelProfile(provider="custom", model="unknown")
        assert p.supports("tools") is None
        assert p.supports("vision") is None

    def test_supports_invalid(self):
        p = ModelProfile(provider="openai", model="gpt-4o")
        with pytest.raises(ValueError, match="Unknown capability"):
            p.supports("telepathy")


# ---------------------------------------------------------------------------
# CostTier tests
# ---------------------------------------------------------------------------


class TestCostTier:
    def test_values(self):
        assert CostTier.FREE.value == "free"
        assert CostTier.LOW.value == "low"
        assert CostTier.MEDIUM.value == "medium"
        assert CostTier.HIGH.value == "high"
        assert CostTier.PREMIUM.value == "premium"


# ---------------------------------------------------------------------------
# get_model_profile tests
# ---------------------------------------------------------------------------


class TestGetModelProfile:
    def test_exact_match(self):
        p = get_model_profile("anthropic:claude-sonnet-4-6")
        assert p is not None
        assert p.provider == "anthropic"
        assert p.model == "claude-sonnet-4-6"

    def test_exact_match_openai(self):
        p = get_model_profile("openai:gpt-4o")
        assert p is not None
        assert p.provider == "openai"
        assert p.context_window == 128_000

    def test_bare_model_name(self):
        p = get_model_profile("gpt-4o")
        assert p is not None
        assert p.model == "gpt-4o"

    def test_not_found(self):
        p = get_model_profile("nonexistent:model-x")
        assert p is None

    def test_bare_name_not_found(self):
        p = get_model_profile("totally-unknown-model")
        assert p is None

    def test_github_model(self):
        p = get_model_profile("github:gpt-4o")
        assert p is not None
        assert p.provider == "github"
        assert p.cost_tier == CostTier.FREE

    def test_anthropic_capabilities(self):
        p = get_model_profile("anthropic:claude-opus-4-7")
        assert p is not None
        assert p.supports_tools is True
        assert p.supports_vision is True
        assert p.context_window == 200_000
        assert p.cost_tier == CostTier.PREMIUM


# ---------------------------------------------------------------------------
# list_models tests
# ---------------------------------------------------------------------------


class TestListModels:
    def test_all_models(self):
        models = list_models()
        assert len(models) > 10  # we registered a bunch

    def test_filter_by_provider(self):
        models = list_models(provider="anthropic")
        assert all(p.provider == "anthropic" for p in models)
        assert len(models) == 3  # opus, sonnet, haiku

    def test_filter_by_tools(self):
        models = list_models(supports_tools=True)
        assert all(p.supports_tools is True for p in models)

    def test_filter_no_tools(self):
        models = list_models(supports_tools=False)
        assert all(p.supports_tools is False for p in models)

    def test_filter_by_vision(self):
        models = list_models(supports_vision=True)
        assert all(p.supports_vision is True for p in models)
        assert len(models) >= 5

    def test_filter_by_cost_tier(self):
        models = list_models(cost_tier=CostTier.FREE)
        assert all(p.cost_tier == CostTier.FREE for p in models)

    def test_combined_filters(self):
        models = list_models(supports_tools=True, cost_tier=CostTier.FREE)
        assert all(p.supports_tools is True and p.cost_tier == CostTier.FREE for p in models)

    def test_sorted_output(self):
        models = list_models()
        keys = [(p.provider, p.model) for p in models]
        assert keys == sorted(keys)

    def test_empty_filter(self):
        models = list_models(provider="nonexistent_provider")
        assert models == []


# ---------------------------------------------------------------------------
# register_model tests
# ---------------------------------------------------------------------------


class TestRegisterModel:
    def test_register_custom(self):
        custom = ModelProfile(
            provider="custom",
            model="test-model-register",
            display_name="Test Model",
            context_window=4096,
            supports_tools=False,
            supports_vision=False,
            cost_tier=CostTier.LOW,
        )
        register_model(custom)

        found = get_model_profile("custom:test-model-register")
        assert found is not None
        assert found.display_name == "Test Model"
        assert found.context_window == 4096

        # Cleanup
        del _PROFILES["custom:test-model-register"]

    def test_register_overrides(self):
        original = get_model_profile("openai:gpt-4o")
        assert original is not None

        override = ModelProfile(
            provider="openai",
            model="gpt-4o",
            display_name="My Custom GPT-4o",
            context_window=64_000,
        )
        register_model(override)

        found = get_model_profile("openai:gpt-4o")
        assert found is not None
        assert found.display_name == "My Custom GPT-4o"

        # Restore original
        _PROFILES["openai:gpt-4o"] = original


# ---------------------------------------------------------------------------
# Built-in registry sanity checks
# ---------------------------------------------------------------------------


class TestBuiltInRegistry:
    def test_all_profiles_have_provider_and_model(self):
        for key, p in _PROFILES.items():
            assert p.provider, f"{key} missing provider"
            assert p.model, f"{key} missing model"
            assert p.key == key, f"key mismatch: {p.key} != {key}"

    def test_all_profiles_have_context_window(self):
        for key, p in _PROFILES.items():
            assert p.context_window is not None, f"{key} missing context_window"
            assert p.context_window > 0, f"{key} has invalid context_window"

    def test_all_profiles_have_tool_support_defined(self):
        for key, p in _PROFILES.items():
            assert p.supports_tools is not None, f"{key} missing supports_tools"

    def test_known_providers(self):
        providers = {p.provider for p in _PROFILES.values()}
        expected = {
            "anthropic",
            "openai",
            "google",
            "github",
            "deepseek",
            "groq",
            "mistral",
            "ollama",
        }
        assert expected.issubset(providers)


# ---------------------------------------------------------------------------
# model_profiles_table tests
# ---------------------------------------------------------------------------


class TestModelProfilesTable:
    def test_renders(self):
        pytest.importorskip("great_tables")
        pytest.importorskip("pandas")

        table = model_profiles_table()
        assert table is not None
        html = table.as_raw_html()
        assert "Model Capability Profiles" in html
        assert "claude-sonnet-4-6" in html or "Claude Sonnet" in html
