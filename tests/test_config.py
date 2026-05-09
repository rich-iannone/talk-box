"""Tests for talk_box.config module."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from yaml12 import format_yaml, write_yaml

from talk_box.config import (
    AutonomousConfig,
    CommitStrategy,
    KnowledgeConfig,
    KnowledgeSource,
    NotificationChannel,
    NotificationsConfig,
    OnFailure,
    ProfileConfig,
    ResolvedConfig,
    RetryBackoff,
    TUIMode,
    TalkBoxConfig,
    list_profiles,
    load_config,
    load_profile,
    save_config,
    save_profile,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config_dir(tmp_path):
    """Create a temporary config directory structure."""
    global_dir = tmp_path / "global"
    global_dir.mkdir()
    profiles_dir = global_dir / "profiles"
    profiles_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    return {"global": global_dir, "profiles": profiles_dir, "project": project_dir}


@pytest.fixture
def sample_project_config():
    """A sample project config dict."""
    return {
        "default_model": "ollama:llama3.3",
        "default_persona": "code_reviewer",
        "guardrails": ["pii", "off_topic"],
        "temperature": 0.7,
        "allow_cloud": False,
        "mode": "simple",
        "profiles": {
            "review": {
                "model": "anthropic:claude-sonnet-4-6",
                "persona": "code_reviewer",
                "temperature": 0.3,
            },
            "support": {
                "model": "ollama:llama3.3",
                "persona": "customer_support_tier1",
                "guardrails": ["pii", "off_topic", "disclaimer"],
            },
        },
        "knowledge": {
            "sources": [
                {"path": "./docs/", "type": "markdown"},
                {"path": "./data/", "type": "directory"},
            ]
        },
        "trusted_commands": ["Rscript", "quarto", "pb"],
        "autonomous": {
            "auto_approve": True,
            "max_retries": 5,
            "retry_backoff": "exponential",
            "commit_strategy": "per_file",
            "on_failure": "abort",
            "checkpoint": False,
        },
        "notifications": {
            "on_complete": ["desktop", "webhook"],
            "on_failure": ["webhook"],
            "webhook_url": "https://hooks.slack.com/test",
        },
    }


# ---------------------------------------------------------------------------
# TalkBoxConfig defaults
# ---------------------------------------------------------------------------


class TestDefaults:
    """Test default config values."""

    def test_default_config(self):
        cfg = TalkBoxConfig()
        assert cfg.default_model is None
        assert cfg.default_persona is None
        assert cfg.guardrails == []
        assert cfg.temperature is None
        assert cfg.allow_cloud is True
        assert cfg.mode == TUIMode.FULL
        assert cfg.profiles == {}
        assert cfg.knowledge.sources == []
        assert cfg.trusted_commands == ["python", "pytest", "ruff", "mypy", "make"]

    def test_default_autonomous(self):
        cfg = TalkBoxConfig()
        assert cfg.autonomous.auto_approve is False
        assert cfg.autonomous.max_retries == 3
        assert cfg.autonomous.retry_backoff == RetryBackoff.LINEAR
        assert cfg.autonomous.commit_strategy == CommitStrategy.PER_TASK
        assert cfg.autonomous.on_failure == OnFailure.SKIP_DEPENDENTS
        assert cfg.autonomous.checkpoint is True

    def test_default_notifications(self):
        cfg = TalkBoxConfig()
        assert cfg.notifications.on_complete == [NotificationChannel.TERMINAL_BELL]
        assert cfg.notifications.on_failure == [NotificationChannel.DESKTOP]
        assert cfg.notifications.webhook_url is None


# ---------------------------------------------------------------------------
# Cloud model detection
# ---------------------------------------------------------------------------


class TestCloudModel:
    """Test cloud model detection and validation."""

    @pytest.mark.parametrize(
        "model,expected",
        [
            ("anthropic:claude-sonnet-4-6", True),
            ("openai:gpt-4o", True),
            ("google:gemini-2.0-flash", True),
            ("ollama:llama3.3", False),
            ("ollama:qwen3:32b", False),
            ("localmodel", False),
        ],
    )
    def test_is_cloud_model(self, model, expected):
        cfg = TalkBoxConfig()
        assert cfg.is_cloud_model(model) == expected

    def test_validate_model_allows_cloud_by_default(self):
        cfg = TalkBoxConfig()
        cfg.validate_model("anthropic:claude-sonnet-4-6")  # Should not raise

    def test_validate_model_blocks_cloud_when_disabled(self):
        cfg = TalkBoxConfig(allow_cloud=False)
        with pytest.raises(ValueError, match="Cloud model.*blocked"):
            cfg.validate_model("anthropic:claude-sonnet-4-6")

    def test_validate_model_allows_local_when_cloud_disabled(self):
        cfg = TalkBoxConfig(allow_cloud=False)
        cfg.validate_model("ollama:llama3.3")  # Should not raise


# ---------------------------------------------------------------------------
# Profile lookup
# ---------------------------------------------------------------------------


class TestProfile:
    """Test profile access."""

    def test_get_profile(self):
        cfg = TalkBoxConfig(
            profiles={
                "review": ProfileConfig(
                    name="review",
                    model="anthropic:claude-sonnet-4-6",
                    persona="code_reviewer",
                    temperature=0.3,
                )
            }
        )
        p = cfg.get_profile("review")
        assert p.model == "anthropic:claude-sonnet-4-6"
        assert p.persona == "code_reviewer"
        assert p.temperature == 0.3

    def test_get_profile_missing(self):
        cfg = TalkBoxConfig()
        with pytest.raises(KeyError, match="Unknown profile"):
            cfg.get_profile("nonexistent")


# ---------------------------------------------------------------------------
# Resolve
# ---------------------------------------------------------------------------


class TestResolve:
    """Test config resolution with layered overrides."""

    def test_resolve_defaults(self):
        cfg = TalkBoxConfig(default_model="ollama:llama3.3", default_persona="code_reviewer")
        resolved = cfg.resolve()
        assert resolved.model == "ollama:llama3.3"
        assert resolved.persona == "code_reviewer"

    def test_resolve_profile_overrides_defaults(self):
        cfg = TalkBoxConfig(
            default_model="ollama:llama3.3",
            default_persona="code_reviewer",
            profiles={
                "review": ProfileConfig(
                    name="review",
                    model="anthropic:claude-sonnet-4-6",
                    persona="data_analyst",
                    temperature=0.3,
                )
            },
        )
        resolved = cfg.resolve(profile="review")
        assert resolved.model == "anthropic:claude-sonnet-4-6"
        assert resolved.persona == "data_analyst"
        assert resolved.temperature == 0.3

    def test_resolve_cli_overrides_profile(self):
        cfg = TalkBoxConfig(
            profiles={
                "review": ProfileConfig(
                    name="review",
                    model="anthropic:claude-sonnet-4-6",
                    persona="code_reviewer",
                )
            }
        )
        resolved = cfg.resolve(profile="review", model="ollama:qwen3:32b", persona="data_analyst")
        assert resolved.model == "ollama:qwen3:32b"
        assert resolved.persona == "data_analyst"

    def test_resolve_env_overrides_defaults(self, monkeypatch):
        monkeypatch.setenv("TALK_BOX_MODEL", "openai:gpt-4o")
        monkeypatch.setenv("TALK_BOX_PERSONA", "writing_coach")
        monkeypatch.setenv("TALK_BOX_TEMPERATURE", "0.9")
        cfg = TalkBoxConfig(default_model="ollama:llama3.3", default_persona="code_reviewer")
        resolved = cfg.resolve()
        assert resolved.model == "openai:gpt-4o"
        assert resolved.persona == "writing_coach"
        assert resolved.temperature == 0.9

    def test_resolve_cli_overrides_env(self, monkeypatch):
        monkeypatch.setenv("TALK_BOX_MODEL", "openai:gpt-4o")
        cfg = TalkBoxConfig()
        resolved = cfg.resolve(model="ollama:llama3.3")
        assert resolved.model == "ollama:llama3.3"

    def test_resolve_invalid_env_temperature_ignored(self, monkeypatch):
        monkeypatch.setenv("TALK_BOX_TEMPERATURE", "not_a_number")
        cfg = TalkBoxConfig(temperature=0.5)
        resolved = cfg.resolve()
        assert resolved.temperature == 0.5

    def test_resolve_blocks_cloud_model(self):
        cfg = TalkBoxConfig(allow_cloud=False)
        with pytest.raises(ValueError, match="Cloud model.*blocked"):
            cfg.resolve(model="anthropic:claude-sonnet-4-6")

    def test_resolve_profile_guardrails(self):
        cfg = TalkBoxConfig(
            guardrails=["pii"],
            profiles={
                "strict": ProfileConfig(
                    name="strict",
                    guardrails=["pii", "off_topic", "disclaimer"],
                )
            },
        )
        resolved = cfg.resolve(profile="strict")
        assert resolved.guardrails == ["pii", "off_topic", "disclaimer"]

    def test_resolve_no_profile_keeps_config_guardrails(self):
        cfg = TalkBoxConfig(guardrails=["pii", "off_topic"])
        resolved = cfg.resolve()
        assert resolved.guardrails == ["pii", "off_topic"]


# ---------------------------------------------------------------------------
# Load config from files
# ---------------------------------------------------------------------------


class TestLoadConfig:
    """Test loading config from YAML files."""

    def test_load_empty(self, config_dir):
        cfg = load_config(
            project_dir=config_dir["project"],
            global_path=config_dir["global"] / "config.yml",
        )
        assert cfg.default_model is None
        assert cfg.allow_cloud is True

    def test_load_global_only(self, config_dir):
        global_file = config_dir["global"] / "config.yml"
        write_yaml({"default_model": "ollama:llama3.3", "temperature": 0.5}, global_file)

        cfg = load_config(
            project_dir=config_dir["project"],
            global_path=global_file,
        )
        assert cfg.default_model == "ollama:llama3.3"
        assert cfg.temperature == 0.5

    def test_load_project_overrides_global(self, config_dir):
        global_file = config_dir["global"] / "config.yml"
        write_yaml({"default_model": "ollama:llama3.3", "temperature": 0.5}, global_file)

        project_file = config_dir["project"] / "talk-box.yml"
        write_yaml(
            {"default_model": "ollama:qwen3:32b", "default_persona": "data_analyst"}, project_file
        )

        cfg = load_config(
            project_dir=config_dir["project"],
            global_path=global_file,
        )
        assert cfg.default_model == "ollama:qwen3:32b"
        assert cfg.default_persona == "data_analyst"
        assert cfg.temperature == 0.5  # Inherited from global

    def test_load_full_project_config(self, config_dir, sample_project_config):
        project_file = config_dir["project"] / "talk-box.yml"
        write_yaml(sample_project_config, project_file)

        cfg = load_config(
            project_dir=config_dir["project"],
            global_path=config_dir["global"] / "config.yml",
        )
        assert cfg.default_model == "ollama:llama3.3"
        assert cfg.default_persona == "code_reviewer"
        assert cfg.guardrails == ["pii", "off_topic"]
        assert cfg.temperature == 0.7
        assert cfg.allow_cloud is False
        assert cfg.mode == TUIMode.SIMPLE
        assert "review" in cfg.profiles
        assert "support" in cfg.profiles
        assert len(cfg.knowledge.sources) == 2
        assert cfg.trusted_commands == ["Rscript", "quarto", "pb"]
        assert cfg.autonomous.auto_approve is True
        assert cfg.autonomous.max_retries == 5
        assert cfg.autonomous.retry_backoff == RetryBackoff.EXPONENTIAL
        assert cfg.autonomous.commit_strategy == CommitStrategy.PER_FILE
        assert cfg.autonomous.on_failure == OnFailure.ABORT
        assert cfg.autonomous.checkpoint is False
        assert NotificationChannel.DESKTOP in cfg.notifications.on_complete
        assert cfg.notifications.webhook_url == "https://hooks.slack.com/test"

    def test_load_malformed_yaml_returns_defaults(self, config_dir):
        project_file = config_dir["project"] / "talk-box.yml"
        project_file.write_text(":::not valid yaml:::", encoding="utf-8")

        cfg = load_config(
            project_dir=config_dir["project"],
            global_path=config_dir["global"] / "config.yml",
        )
        assert cfg.default_model is None  # Falls back to defaults

    def test_load_searches_parent_dirs(self, config_dir):
        project_file = config_dir["project"] / "talk-box.yml"
        write_yaml({"default_model": "ollama:llama3.3"}, project_file)

        subdir = config_dir["project"] / "src" / "deep"
        subdir.mkdir(parents=True)

        cfg = load_config(
            project_dir=subdir,
            global_path=config_dir["global"] / "config.yml",
        )
        assert cfg.default_model == "ollama:llama3.3"


# ---------------------------------------------------------------------------
# Save config
# ---------------------------------------------------------------------------


class TestSaveConfig:
    """Test saving config to YAML files."""

    def test_save_and_reload(self, tmp_path):
        cfg = TalkBoxConfig(
            default_model="ollama:llama3.3",
            default_persona="code_reviewer",
            temperature=0.7,
        )
        path = tmp_path / "talk-box.yml"
        save_config(cfg, path)

        loaded = load_config(project_dir=tmp_path, global_path=tmp_path / "nonexistent.yml")
        assert loaded.default_model == "ollama:llama3.3"
        assert loaded.default_persona == "code_reviewer"

    def test_save_creates_parent_dirs(self, tmp_path):
        cfg = TalkBoxConfig(default_model="ollama:llama3.3")
        path = tmp_path / "deep" / "nested" / "talk-box.yml"
        save_config(cfg, path)
        assert path.is_file()


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------


class TestProfiles:
    """Test profile save/load/list."""

    def test_save_and_load_profile(self, tmp_path):
        profile = ProfileConfig(
            name="review",
            model="anthropic:claude-sonnet-4-6",
            persona="code_reviewer",
            temperature=0.3,
            guardrails=["pii"],
        )
        save_profile(profile, profiles_dir=tmp_path)

        loaded = load_profile("review", profiles_dir=tmp_path)
        assert loaded.name == "review"
        assert loaded.model == "anthropic:claude-sonnet-4-6"
        assert loaded.persona == "code_reviewer"
        assert loaded.temperature == 0.3
        assert loaded.guardrails == ["pii"]

    def test_load_profile_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Profile not found"):
            load_profile("nonexistent", profiles_dir=tmp_path)

    def test_list_profiles_empty(self, tmp_path):
        assert list_profiles(profiles_dir=tmp_path) == []

    def test_list_profiles(self, tmp_path):
        for name in ["review", "support", "alpha"]:
            save_profile(
                ProfileConfig(name=name, model=f"ollama:{name}"),
                profiles_dir=tmp_path,
            )
        names = list_profiles(profiles_dir=tmp_path)
        assert names == ["alpha", "review", "support"]


# ---------------------------------------------------------------------------
# to_dict round-trip
# ---------------------------------------------------------------------------


class TestToDict:
    """Test serialization to dict."""

    def test_profile_to_dict(self):
        p = ProfileConfig(
            name="review",
            model="anthropic:claude-sonnet-4-6",
            persona="code_reviewer",
            temperature=0.3,
        )
        d = p.to_dict()
        assert d["model"] == "anthropic:claude-sonnet-4-6"
        assert d["persona"] == "code_reviewer"
        assert d["temperature"] == 0.3
        assert "name" not in d  # name is not serialized

    def test_profile_to_dict_minimal(self):
        p = ProfileConfig(name="empty")
        assert p.to_dict() == {}

    def test_autonomous_to_dict(self):
        a = AutonomousConfig(auto_approve=True, max_retries=5)
        d = a.to_dict()
        assert d["auto_approve"] is True
        assert d["max_retries"] == 5
        assert d["retry_backoff"] == "linear"

    def test_config_to_dict_minimal(self):
        cfg = TalkBoxConfig()
        assert cfg.to_dict() == {}

    def test_config_to_dict_with_values(self):
        cfg = TalkBoxConfig(
            default_model="ollama:llama3.3",
            allow_cloud=False,
            mode=TUIMode.SIMPLE,
        )
        d = cfg.to_dict()
        assert d["default_model"] == "ollama:llama3.3"
        assert d["allow_cloud"] is False
        assert d["mode"] == "simple"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_unknown_mode_defaults_to_full(self):
        from talk_box.config import _parse_mode

        assert _parse_mode("unknown_mode") == TUIMode.FULL

    def test_unknown_notification_channel_skipped(self):
        from talk_box.config import _parse_notification_channels

        channels = _parse_notification_channels(["terminal_bell", "carrier_pigeon", "desktop"])
        assert channels == [NotificationChannel.TERMINAL_BELL, NotificationChannel.DESKTOP]

    def test_unknown_backoff_defaults_to_linear(self):
        from talk_box.config import _parse_autonomous

        auto = _parse_autonomous({"retry_backoff": "cubic"})
        assert auto.retry_backoff == RetryBackoff.LINEAR

    def test_non_dict_profiles_ignored(self):
        from talk_box.config import _parse_config_dict

        cfg = _parse_config_dict({"profiles": "not_a_dict"})
        assert cfg.profiles == {}

    def test_non_dict_knowledge_ignored(self):
        from talk_box.config import _parse_config_dict

        cfg = _parse_config_dict({"knowledge": "not_a_dict"})
        assert cfg.knowledge.sources == []

    def test_empty_yaml_file(self, tmp_path):
        path = tmp_path / "talk-box.yml"
        path.write_text("", encoding="utf-8")
        cfg = load_config(project_dir=tmp_path, global_path=tmp_path / "nonexistent.yml")
        assert cfg.default_model is None

    def test_profiles_merged_across_layers(self, config_dir):
        global_file = config_dir["global"] / "config.yml"
        write_yaml(
            {"profiles": {"global_profile": {"model": "ollama:llama3.3"}}},
            global_file,
        )

        project_file = config_dir["project"] / "talk-box.yml"
        write_yaml(
            {"profiles": {"project_profile": {"model": "ollama:qwen3:32b"}}},
            project_file,
        )

        cfg = load_config(
            project_dir=config_dir["project"],
            global_path=global_file,
        )
        assert "global_profile" in cfg.profiles
        assert "project_profile" in cfg.profiles


class TestAllowCloudEnforcement:
    """Tests for allow_cloud config enforcement."""

    def test_is_cloud_model(self):
        from talk_box.config import TalkBoxConfig

        cfg = TalkBoxConfig()
        assert cfg.is_cloud_model("anthropic:claude-sonnet-4-6") is True
        assert cfg.is_cloud_model("openai:gpt-4o") is True
        assert cfg.is_cloud_model("google:gemini-2.5-flash") is True
        assert cfg.is_cloud_model("ollama:llama3.3") is False
        assert cfg.is_cloud_model("local:my-model") is False

    def test_validate_model_blocks_cloud(self):
        from talk_box.config import TalkBoxConfig

        cfg = TalkBoxConfig(allow_cloud=False)
        with pytest.raises(ValueError, match="blocked by allow_cloud"):
            cfg.validate_model("anthropic:claude-sonnet-4-6")

    def test_validate_model_allows_local(self):
        from talk_box.config import TalkBoxConfig

        cfg = TalkBoxConfig(allow_cloud=False)
        cfg.validate_model("ollama:llama3.3")  # Should not raise

    def test_validate_model_allows_cloud_when_enabled(self):
        from talk_box.config import TalkBoxConfig

        cfg = TalkBoxConfig(allow_cloud=True)
        cfg.validate_model("anthropic:claude-sonnet-4-6")  # Should not raise

    def test_resolve_rejects_cloud_model(self):
        from talk_box.config import TalkBoxConfig

        cfg = TalkBoxConfig(allow_cloud=False)
        with pytest.raises(ValueError, match="blocked by allow_cloud"):
            cfg.resolve(model="openai:gpt-4o")

    def test_resolve_accepts_local_model(self):
        from talk_box.config import TalkBoxConfig

        cfg = TalkBoxConfig(allow_cloud=False)
        resolved = cfg.resolve(model="ollama:llama3.3")
        assert resolved.model == "ollama:llama3.3"

    def test_allow_cloud_from_project_config(self, tmp_path):
        from talk_box.config import load_config

        # Create a project config with allow_cloud=false
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        write_yaml({"allow_cloud": False}, project_dir / "talk-box.yml")

        # Also need a global config path that doesn't exist
        global_path = tmp_path / "global" / "config.yml"

        cfg = load_config(project_dir=project_dir, global_path=global_path)
        assert cfg.allow_cloud is False
