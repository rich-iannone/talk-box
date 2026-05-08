import pytest

from talk_box.models import (
    CostTier,
    ModelProfile,
    OllamaStatus,
    _PROFILES,
    _parse_ollama_model_details,
    detect_ollama,
    get_model_profile,
    list_models,
    list_ollama_models,
    model_profiles_table,
    register_model,
    sync_ollama_models,
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


# ---------------------------------------------------------------------------
# Ollama detection tests
# ---------------------------------------------------------------------------

# Sample Ollama API responses for mocking
_MOCK_VERSION_RESPONSE = b'{"version": "0.6.2"}'
_MOCK_TAGS_RESPONSE = b"""{
    "models": [
        {
            "name": "llama3.3:latest",
            "details": {
                "family": "llama",
                "families": ["llama"],
                "parameter_size": "70B"
            }
        },
        {
            "name": "gemma3:27b",
            "details": {
                "family": "gemma3",
                "families": ["gemma3", "vision"],
                "parameter_size": "27B"
            }
        },
        {
            "name": "qwen2.5-coder:7b",
            "details": {
                "family": "qwen2",
                "families": ["qwen2"],
                "parameter_size": "7B"
            }
        }
    ]
}"""


class TestOllamaStatus:
    def test_available(self):
        status = OllamaStatus(
            available=True, url="http://localhost:11434", version="0.6.2", models=["llama3.3"]
        )
        assert status.available
        assert status.version == "0.6.2"
        assert status.models == ["llama3.3"]

    def test_unavailable(self):
        status = OllamaStatus(
            available=False, url="http://localhost:11434", error="Connection refused"
        )
        assert not status.available
        assert "refused" in status.error


class TestDetectOllama:
    def test_ollama_available(self, monkeypatch):
        """detect_ollama returns status when Ollama is reachable."""
        import io
        import urllib.request

        call_count = {"n": 0}

        def mock_urlopen(req, timeout=None):
            call_count["n"] += 1
            url = req.full_url if hasattr(req, "full_url") else str(req)
            if "/api/version" in url:
                resp = io.BytesIO(_MOCK_VERSION_RESPONSE)
            else:
                resp = io.BytesIO(_MOCK_TAGS_RESPONSE)
            resp.status = 200
            resp.__enter__ = lambda s: s
            resp.__exit__ = lambda s, *a: None
            resp.read = resp.read
            return resp

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

        status = detect_ollama()
        assert status.available is True
        assert status.version == "0.6.2"
        assert status.url == "http://localhost:11434"
        assert "llama3.3:latest" in status.models
        assert "gemma3:27b" in status.models
        assert len(status.models) == 3

    def test_ollama_unavailable(self, monkeypatch):
        """detect_ollama returns failure when Ollama is not reachable."""
        import urllib.error
        import urllib.request

        def mock_urlopen(req, timeout=None):
            raise urllib.error.URLError("Connection refused")

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

        status = detect_ollama()
        assert status.available is False
        assert "refused" in status.error.lower()

    def test_custom_url(self, monkeypatch):
        """detect_ollama uses custom URL."""
        import io
        import urllib.request

        captured_urls = []

        def mock_urlopen(req, timeout=None):
            url = req.full_url if hasattr(req, "full_url") else str(req)
            captured_urls.append(url)
            resp = io.BytesIO(_MOCK_VERSION_RESPONSE if "/version" in url else _MOCK_TAGS_RESPONSE)
            resp.__enter__ = lambda s: s
            resp.__exit__ = lambda s, *a: None
            return resp

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

        detect_ollama(url="http://myhost:9999")
        assert any("myhost:9999" in u for u in captured_urls)

    def test_env_var_url(self, monkeypatch):
        """detect_ollama reads OLLAMA_HOST env var."""
        import io
        import urllib.request

        captured_urls = []

        def mock_urlopen(req, timeout=None):
            url = req.full_url if hasattr(req, "full_url") else str(req)
            captured_urls.append(url)
            resp = io.BytesIO(_MOCK_VERSION_RESPONSE if "/version" in url else _MOCK_TAGS_RESPONSE)
            resp.__enter__ = lambda s: s
            resp.__exit__ = lambda s, *a: None
            return resp

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)
        monkeypatch.setenv("OLLAMA_HOST", "http://remote:8080")

        detect_ollama()
        assert any("remote:8080" in u for u in captured_urls)


class TestListOllamaModels:
    def test_returns_profiles(self, monkeypatch):
        import io
        import urllib.request

        def mock_urlopen(req, timeout=None):
            resp = io.BytesIO(_MOCK_TAGS_RESPONSE)
            resp.__enter__ = lambda s: s
            resp.__exit__ = lambda s, *a: None
            return resp

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

        profiles = list_ollama_models()
        assert len(profiles) == 3
        assert all(p.provider == "ollama" for p in profiles)
        assert all(p.cost_tier == CostTier.FREE for p in profiles)

    def test_vision_detection(self, monkeypatch):
        import io
        import urllib.request

        def mock_urlopen(req, timeout=None):
            resp = io.BytesIO(_MOCK_TAGS_RESPONSE)
            resp.__enter__ = lambda s: s
            resp.__exit__ = lambda s, *a: None
            return resp

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

        profiles = list_ollama_models()
        gemma = next(p for p in profiles if "gemma3" in p.model)
        llama = next(p for p in profiles if "llama3" in p.model)
        assert gemma.supports_vision is True
        assert llama.supports_vision is False

    def test_tool_support_detection(self, monkeypatch):
        import io
        import urllib.request

        def mock_urlopen(req, timeout=None):
            resp = io.BytesIO(_MOCK_TAGS_RESPONSE)
            resp.__enter__ = lambda s: s
            resp.__exit__ = lambda s, *a: None
            return resp

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

        profiles = list_ollama_models()
        llama = next(p for p in profiles if "llama3" in p.model)
        qwen = next(p for p in profiles if "qwen" in p.model)
        assert llama.supports_tools is True
        assert qwen.supports_tools is True

    def test_context_window_from_family(self, monkeypatch):
        import io
        import urllib.request

        def mock_urlopen(req, timeout=None):
            resp = io.BytesIO(_MOCK_TAGS_RESPONSE)
            resp.__enter__ = lambda s: s
            resp.__exit__ = lambda s, *a: None
            return resp

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

        profiles = list_ollama_models()
        llama = next(p for p in profiles if "llama3" in p.model)
        qwen = next(p for p in profiles if "qwen" in p.model)
        assert llama.context_window == 128_000
        assert qwen.context_window == 32_768

    def test_unreachable_returns_empty(self, monkeypatch):
        import urllib.error
        import urllib.request

        def mock_urlopen(req, timeout=None):
            raise urllib.error.URLError("Connection refused")

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

        profiles = list_ollama_models()
        assert profiles == []

    def test_sorted_output(self, monkeypatch):
        import io
        import urllib.request

        def mock_urlopen(req, timeout=None):
            resp = io.BytesIO(_MOCK_TAGS_RESPONSE)
            resp.__enter__ = lambda s: s
            resp.__exit__ = lambda s, *a: None
            return resp

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

        profiles = list_ollama_models()
        names = [p.model for p in profiles]
        assert names == sorted(names)


class TestSyncOllamaModels:
    def test_registers_in_global(self, monkeypatch):
        import io
        import urllib.request

        def mock_urlopen(req, timeout=None):
            resp = io.BytesIO(_MOCK_TAGS_RESPONSE)
            resp.__enter__ = lambda s: s
            resp.__exit__ = lambda s, *a: None
            return resp

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

        # Remove any existing ollama:llama3.3:latest from built-in registry
        _PROFILES.pop("ollama:llama3.3:latest", None)

        profiles = sync_ollama_models()
        assert len(profiles) == 3

        # Should now be findable
        found = get_model_profile("ollama:llama3.3:latest")
        assert found is not None
        assert found.provider == "ollama"

        # Cleanup
        for p in profiles:
            _PROFILES.pop(p.key, None)

    def test_unreachable_noop(self, monkeypatch):
        import urllib.error
        import urllib.request

        def mock_urlopen(req, timeout=None):
            raise urllib.error.URLError("Connection refused")

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

        profiles = sync_ollama_models()
        assert profiles == []


class TestParseOllamaModelDetails:
    def test_llama_family(self):
        info = {"details": {"family": "llama", "families": ["llama"], "parameter_size": "70B"}}
        caps = _parse_ollama_model_details(info)
        assert caps["supports_tools"] is True
        assert caps["supports_vision"] is False
        assert caps["context_window"] == 128_000

    def test_vision_family(self):
        info = {
            "details": {
                "family": "gemma3",
                "families": ["gemma3", "vision"],
                "parameter_size": "27B",
            }
        }
        caps = _parse_ollama_model_details(info)
        assert caps["supports_vision"] is True
        assert caps["context_window"] == 128_000

    def test_unknown_family(self):
        info = {"details": {"family": "exotic", "families": ["exotic"], "parameter_size": "3B"}}
        caps = _parse_ollama_model_details(info)
        assert caps["supports_tools"] is False
        assert caps["context_window"] == 8_192  # conservative default

    def test_empty_details(self):
        info = {}
        caps = _parse_ollama_model_details(info)
        assert caps["supports_tools"] is False
        assert caps["supports_vision"] is False
        assert caps["context_window"] == 8_192
