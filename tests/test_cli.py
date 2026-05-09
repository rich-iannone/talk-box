"""Tests for talk_box.cli module."""

from __future__ import annotations

from click.testing import CliRunner

from talk_box.cli import main


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _runner() -> CliRunner:
    """Create a Click test runner."""
    return CliRunner()


# ---------------------------------------------------------------------------
# TestMainGroup
# ---------------------------------------------------------------------------


class TestMainGroup:
    """Tests for the top-level CLI group."""

    def test_help(self):
        result = _runner().invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Talk Box" in result.output
        assert "COMMAND" in result.output

    def test_version(self):
        result = _runner().invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "version" in result.output

    def test_no_args_shows_help(self):
        result = _runner().invoke(main, [])
        assert "Usage" in result.output


# ---------------------------------------------------------------------------
# TestInfo
# ---------------------------------------------------------------------------


class TestInfo:
    """Tests for 'talk-box info' command."""

    def test_default_summary(self):
        result = _runner().invoke(main, ["info"])
        assert result.exit_code == 0
        assert "Talk Box" in result.output
        assert "Version" in result.output
        assert "Personas" in result.output
        assert "Models" in result.output

    def test_personas_flag(self):
        result = _runner().invoke(main, ["info", "--personas"])
        assert result.exit_code == 0
        assert "Available Personas" in result.output
        assert "code_reviewer" in result.output

    def test_models_flag(self):
        result = _runner().invoke(main, ["info", "--models"])
        assert result.exit_code == 0
        assert "Model Profiles" in result.output
        assert "anthropic" in result.output

    def test_help(self):
        result = _runner().invoke(main, ["info", "--help"])
        assert result.exit_code == 0
        assert "--personas" in result.output
        assert "--models" in result.output


# ---------------------------------------------------------------------------
# TestPersonas
# ---------------------------------------------------------------------------


class TestPersonas:
    """Tests for 'talk-box personas' command."""

    def test_list_all(self):
        result = _runner().invoke(main, ["personas"])
        assert result.exit_code == 0
        assert "Available Personas" in result.output
        assert "code_reviewer" in result.output

    def test_detail_view(self):
        result = _runner().invoke(main, ["personas", "code_reviewer"])
        assert result.exit_code == 0
        assert "Code Reviewer" in result.output
        assert "Role" in result.output
        assert "Category" in result.output

    def test_unknown_persona(self):
        result = _runner().invoke(main, ["personas", "nonexistent_xyz"])
        assert result.exit_code == 1
        assert "Unknown persona" in result.output

    def test_help(self):
        result = _runner().invoke(main, ["personas", "--help"])
        assert result.exit_code == 0
        assert "List personas" in result.output


# ---------------------------------------------------------------------------
# TestModels
# ---------------------------------------------------------------------------


class TestModels:
    """Tests for 'talk-box models' command."""

    def test_list_all(self):
        result = _runner().invoke(main, ["models"])
        assert result.exit_code == 0
        assert "Model Profiles" in result.output
        assert "model(s)" in result.output

    def test_filter_by_provider(self):
        result = _runner().invoke(main, ["models", "--provider", "anthropic"])
        assert result.exit_code == 0
        assert "anthropic" in result.output

    def test_filter_no_results(self):
        result = _runner().invoke(main, ["models", "--provider", "nonexistent_provider"])
        assert result.exit_code == 0
        assert "No matching" in result.output

    def test_tools_only(self):
        result = _runner().invoke(main, ["models", "--tools-only"])
        assert result.exit_code == 0

    def test_vision_only(self):
        result = _runner().invoke(main, ["models", "--vision-only"])
        assert result.exit_code == 0

    def test_help(self):
        result = _runner().invoke(main, ["models", "--help"])
        assert result.exit_code == 0
        assert "--provider" in result.output
        assert "--tools-only" in result.output


# ---------------------------------------------------------------------------
# TestTest
# ---------------------------------------------------------------------------


class TestTest:
    """Tests for 'talk-box test' command."""

    def test_help(self):
        result = _runner().invoke(main, ["test", "--help"])
        assert result.exit_code == 0
        assert "persona" in result.output.lower()
        assert "--model" in result.output

    def test_unknown_persona(self):
        result = _runner().invoke(
            main, ["test", "nonexistent_xyz", "-m", "anthropic:claude-sonnet-4-6"]
        )
        assert result.exit_code == 1
        assert "Unknown persona" in result.output

    def test_missing_model(self):
        result = _runner().invoke(main, ["test", "code_reviewer"])
        assert result.exit_code != 0
        assert "model" in result.output.lower() or "required" in result.output.lower()


# ---------------------------------------------------------------------------
# TestConfig
# ---------------------------------------------------------------------------


class TestConfig:
    """Tests for 'talk-box config' commands."""

    def test_config_help(self):
        result = _runner().invoke(main, ["config", "--help"])
        assert result.exit_code == 0
        assert "show" in result.output
        assert "get" in result.output
        assert "set" in result.output

    def test_config_show(self):
        result = _runner().invoke(main, ["config", "show"])
        assert result.exit_code == 0
        assert "Configuration" in result.output
        assert "Model" in result.output
        assert "Allow Cloud" in result.output

    def test_config_get_allow_cloud(self):
        result = _runner().invoke(main, ["config", "get", "allow_cloud"])
        assert result.exit_code == 0
        assert "True" in result.output

    def test_config_get_unknown_key(self):
        result = _runner().invoke(main, ["config", "get", "nonexistent_key"])
        assert result.exit_code == 1
        assert "Unknown key" in result.output

    def test_config_set_and_get(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = _runner().invoke(main, ["config", "set", "default_model", "ollama:llama3.3"])
        assert result.exit_code == 0
        assert "Set" in result.output
        assert "ollama:llama3.3" in result.output

        result = _runner().invoke(main, ["config", "get", "default_model"])
        assert result.exit_code == 0
        assert "ollama:llama3.3" in result.output

    def test_config_set_bool(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = _runner().invoke(main, ["config", "set", "allow_cloud", "false"])
        assert result.exit_code == 0

    def test_config_set_float(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = _runner().invoke(main, ["config", "set", "temperature", "0.42"])
        assert result.exit_code == 0

    def test_config_set_invalid_float(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = _runner().invoke(main, ["config", "set", "temperature", "hot"])
        assert result.exit_code == 1

    def test_config_list_empty(self):
        result = _runner().invoke(main, ["config", "list"])
        assert result.exit_code == 0
