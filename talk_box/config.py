"""Talk Box configuration system: layered config loading, merging, and validation.

Provides a layered configuration system that resolves settings from multiple sources
in precedence order (last wins):

1. Built-in defaults
2. Global config (``~/.config/talk-box/config.yml``)
3. Project config (``./talk-box.yml``)
4. Active profile
5. Environment variables (``TALK_BOX_MODEL``, ``TALK_BOX_PERSONA``, etc.)
6. CLI flags (``--model``, ``--persona``, etc.)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from yaml12 import read_yaml, write_yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GLOBAL_CONFIG_DIR = Path.home() / ".config" / "talk-box"
_GLOBAL_CONFIG_PATH = _GLOBAL_CONFIG_DIR / "config.yml"
_PROFILES_DIR = _GLOBAL_CONFIG_DIR / "profiles"
_PROJECT_CONFIG_NAME = "talk-box.yml"

# Environment variable prefix
_ENV_PREFIX = "TALK_BOX_"

# Cloud providers — used for ``allow_cloud`` enforcement
_CLOUD_PROVIDERS = frozenset(
    {"anthropic", "openai", "google", "mistral", "github", "azure", "bedrock", "together"}
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TUIMode(Enum):
    """TUI display mode."""

    FULL = "full"
    SIMPLE = "simple"


class RetryBackoff(Enum):
    """Retry backoff strategy for autonomous mode."""

    LINEAR = "linear"
    EXPONENTIAL = "exponential"


class OnFailure(Enum):
    """Failure handling strategy for autonomous mode."""

    SKIP_DEPENDENTS = "skip_dependents"
    ABORT = "abort"
    CONTINUE = "continue"


class CommitStrategy(Enum):
    """Git commit strategy for autonomous mode."""

    PER_TASK = "per_task"
    PER_FILE = "per_file"
    BATCH = "batch"


class NotificationChannel(Enum):
    """Notification delivery channel."""

    TERMINAL_BELL = "terminal_bell"
    DESKTOP = "desktop"
    WEBHOOK = "webhook"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class KnowledgeSource:
    """A knowledge source entry in the configuration."""

    path: str
    type: str = "markdown"

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "type": self.type}


@dataclass
class ProfileConfig:
    """A named profile bundling model + persona + settings."""

    name: str
    model: str | None = None
    persona: str | None = None
    temperature: float | None = None
    guardrails: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.model is not None:
            d["model"] = self.model
        if self.persona is not None:
            d["persona"] = self.persona
        if self.temperature is not None:
            d["temperature"] = self.temperature
        if self.guardrails:
            d["guardrails"] = self.guardrails
        return d


@dataclass
class AutonomousConfig:
    """Settings for autonomous (unattended) execution mode."""

    auto_approve: bool = False
    max_retries: int = 3
    retry_backoff: RetryBackoff = RetryBackoff.LINEAR
    commit_strategy: CommitStrategy = CommitStrategy.PER_TASK
    on_failure: OnFailure = OnFailure.SKIP_DEPENDENTS
    checkpoint: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "auto_approve": self.auto_approve,
            "max_retries": self.max_retries,
            "retry_backoff": self.retry_backoff.value,
            "commit_strategy": self.commit_strategy.value,
            "on_failure": self.on_failure.value,
            "checkpoint": self.checkpoint,
        }


@dataclass
class NotificationsConfig:
    """Notification settings for autonomous runs."""

    on_complete: list[NotificationChannel] = field(
        default_factory=lambda: [NotificationChannel.TERMINAL_BELL]
    )
    on_failure: list[NotificationChannel] = field(
        default_factory=lambda: [NotificationChannel.DESKTOP]
    )
    webhook_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "on_complete": [c.value for c in self.on_complete],
            "on_failure": [c.value for c in self.on_failure],
        }
        if self.webhook_url is not None:
            d["webhook_url"] = self.webhook_url
        return d


@dataclass
class KnowledgeConfig:
    """Knowledge graph source configuration."""

    sources: list[KnowledgeSource] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"sources": [s.to_dict() for s in self.sources]}


@dataclass
class TalkBoxConfig:
    """Complete Talk Box configuration.

    This is the merged result of all config layers. Individual fields
    use ``None`` to indicate "not set" (auto-detect or inherit from
    a lower-priority layer).
    """

    # Core settings
    default_model: str | None = None
    default_persona: str | None = None
    guardrails: list[str] = field(default_factory=list)
    temperature: float | None = None

    # Favorites
    favorite_models: list[str] = field(default_factory=list)
    favorite_personas: list[str] = field(default_factory=list)

    # Model restrictions
    allow_cloud: bool = True

    # TUI mode
    mode: TUIMode = TUIMode.FULL

    # Named profiles
    profiles: dict[str, ProfileConfig] = field(default_factory=dict)

    # Knowledge
    knowledge: KnowledgeConfig = field(default_factory=KnowledgeConfig)

    # Trusted commands for workspace shell_exec
    trusted_commands: list[str] = field(
        default_factory=lambda: ["python", "pytest", "ruff", "mypy", "make"]
    )

    # Autonomous mode
    autonomous: AutonomousConfig = field(default_factory=AutonomousConfig)

    # Notifications
    notifications: NotificationsConfig = field(default_factory=NotificationsConfig)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.default_model is not None:
            d["default_model"] = self.default_model
        if self.default_persona is not None:
            d["default_persona"] = self.default_persona
        if self.guardrails:
            d["guardrails"] = self.guardrails
        if self.temperature is not None:
            d["temperature"] = self.temperature
        if self.favorite_models:
            d["favorite_models"] = self.favorite_models
        if self.favorite_personas:
            d["favorite_personas"] = self.favorite_personas
        if not self.allow_cloud:
            d["allow_cloud"] = False
        if self.mode != TUIMode.FULL:
            d["mode"] = self.mode.value
        if self.profiles:
            d["profiles"] = {name: p.to_dict() for name, p in self.profiles.items()}
        if self.knowledge.sources:
            d["knowledge"] = self.knowledge.to_dict()
        if self.trusted_commands != ["python", "pytest", "ruff", "mypy", "make"]:
            d["trusted_commands"] = self.trusted_commands
        if self.autonomous.auto_approve or self.autonomous.max_retries != 3:
            d["autonomous"] = self.autonomous.to_dict()
        if self.notifications.webhook_url is not None:
            d["notifications"] = self.notifications.to_dict()
        return d

    def get_profile(self, name: str) -> ProfileConfig:
        """Get a named profile by name.

        Raises
        ------
        KeyError
            If the profile does not exist.
        """
        if name not in self.profiles:
            raise KeyError(f"Unknown profile: {name!r}. Available: {list(self.profiles.keys())}")
        return self.profiles[name]

    def is_cloud_model(self, model_string: str) -> bool:
        """Check whether a model string refers to a cloud provider.

        Parameters
        ----------
        model_string
            A ``provider:model`` string (e.g., ``"anthropic:claude-sonnet-4-6"``).
        """
        provider = model_string.split(":")[0].lower() if ":" in model_string else model_string
        return provider in _CLOUD_PROVIDERS

    def validate_model(self, model_string: str) -> None:
        """Validate that a model string is allowed under the current config.

        Raises
        ------
        ValueError
            If the model is a cloud model and ``allow_cloud`` is ``False``.
        """
        if not self.allow_cloud and self.is_cloud_model(model_string):
            raise ValueError(
                f"Cloud model {model_string!r} is blocked by allow_cloud=false. "
                f"Use a local model (e.g., ollama:*) or set allow_cloud: true in config."
            )

    def resolve(
        self,
        *,
        model: str | None = None,
        persona: str | None = None,
        profile: str | None = None,
        temperature: float | None = None,
    ) -> ResolvedConfig:
        """Resolve effective settings from config + profile + overrides.

        Parameters
        ----------
        model
            CLI/API model override.
        persona
            CLI/API persona override.
        profile
            Named profile to activate (applies its settings before CLI overrides).
        temperature
            CLI/API temperature override.

        Returns
        -------
        ResolvedConfig
            The final resolved settings.

        Raises
        ------
        ValueError
            If the resolved model is blocked by ``allow_cloud``.
        KeyError
            If the named profile does not exist.
        """
        # Start from config defaults
        resolved_model = self.default_model
        resolved_persona = self.default_persona
        resolved_temp = self.temperature
        resolved_guards = list(self.guardrails)

        # Layer 4: Active profile (overrides config defaults)
        if profile is not None:
            prof = self.get_profile(profile)
            if prof.model is not None:
                resolved_model = prof.model
            if prof.persona is not None:
                resolved_persona = prof.persona
            if prof.temperature is not None:
                resolved_temp = prof.temperature
            if prof.guardrails:
                resolved_guards = list(prof.guardrails)

        # Layer 5: Environment variables
        env_model = os.environ.get(f"{_ENV_PREFIX}MODEL")
        env_persona = os.environ.get(f"{_ENV_PREFIX}PERSONA")
        env_temp = os.environ.get(f"{_ENV_PREFIX}TEMPERATURE")

        if env_model is not None:
            resolved_model = env_model
        if env_persona is not None:
            resolved_persona = env_persona
        if env_temp is not None:
            try:
                resolved_temp = float(env_temp)
            except ValueError:
                pass  # Ignore invalid env var values

        # Layer 6: CLI flags (highest priority)
        if model is not None:
            resolved_model = model
        if persona is not None:
            resolved_persona = persona
        if temperature is not None:
            resolved_temp = temperature

        # Validate the resolved model
        if resolved_model is not None:
            self.validate_model(resolved_model)

        return ResolvedConfig(
            model=resolved_model,
            persona=resolved_persona,
            temperature=resolved_temp,
            guardrails=resolved_guards,
        )


@dataclass(frozen=True)
class ResolvedConfig:
    """The final resolved configuration after all layers are merged."""

    model: str | None = None
    persona: str | None = None
    temperature: float | None = None
    guardrails: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_knowledge_source(data: dict[str, Any]) -> KnowledgeSource:
    return KnowledgeSource(
        path=str(data.get("path", "")),
        type=str(data.get("type", "markdown")),
    )


def _parse_profile(name: str, data: dict[str, Any]) -> ProfileConfig:
    return ProfileConfig(
        name=name,
        model=data.get("model"),
        persona=data.get("persona"),
        temperature=data.get("temperature"),
        guardrails=data.get("guardrails", []),
    )


def _parse_autonomous(data: dict[str, Any]) -> AutonomousConfig:
    backoff_str = data.get("retry_backoff", "linear")
    try:
        backoff = RetryBackoff(backoff_str)
    except ValueError:
        backoff = RetryBackoff.LINEAR

    commit_str = data.get("commit_strategy", "per_task")
    try:
        commit = CommitStrategy(commit_str)
    except ValueError:
        commit = CommitStrategy.PER_TASK

    failure_str = data.get("on_failure", "skip_dependents")
    try:
        on_failure = OnFailure(failure_str)
    except ValueError:
        on_failure = OnFailure.SKIP_DEPENDENTS

    return AutonomousConfig(
        auto_approve=bool(data.get("auto_approve", False)),
        max_retries=int(data.get("max_retries", 3)),
        retry_backoff=backoff,
        commit_strategy=commit,
        on_failure=on_failure,
        checkpoint=bool(data.get("checkpoint", True)),
    )


def _parse_notification_channels(raw: list[str]) -> list[NotificationChannel]:
    channels: list[NotificationChannel] = []
    for item in raw:
        try:
            channels.append(NotificationChannel(item))
        except ValueError:
            pass  # Skip unknown channels
    return channels


def _parse_notifications(data: dict[str, Any]) -> NotificationsConfig:
    on_complete = _parse_notification_channels(data.get("on_complete", ["terminal_bell"]))
    on_failure = _parse_notification_channels(data.get("on_failure", ["desktop"]))
    return NotificationsConfig(
        on_complete=on_complete,
        on_failure=on_failure,
        webhook_url=data.get("webhook_url"),
    )


def _parse_mode(value: str) -> TUIMode:
    try:
        return TUIMode(value)
    except ValueError:
        return TUIMode.FULL


# ---------------------------------------------------------------------------
# Config parsing from dict
# ---------------------------------------------------------------------------


def _parse_config_dict(data: dict[str, Any]) -> TalkBoxConfig:
    """Parse a raw YAML dict into a TalkBoxConfig.

    Parameters
    ----------
    data
        Parsed YAML dictionary.
    """
    profiles: dict[str, ProfileConfig] = {}
    raw_profiles = data.get("profiles", {})
    if isinstance(raw_profiles, dict):
        for name, pdata in raw_profiles.items():
            if isinstance(pdata, dict):
                profiles[name] = _parse_profile(name, pdata)

    knowledge_sources: list[KnowledgeSource] = []
    raw_knowledge = data.get("knowledge", {})
    if isinstance(raw_knowledge, dict):
        for src in raw_knowledge.get("sources", []):
            if isinstance(src, dict):
                knowledge_sources.append(_parse_knowledge_source(src))

    autonomous = AutonomousConfig()
    raw_auto = data.get("autonomous", {})
    if isinstance(raw_auto, dict):
        autonomous = _parse_autonomous(raw_auto)

    notifications = NotificationsConfig()
    raw_notif = data.get("notifications", {})
    if isinstance(raw_notif, dict):
        notifications = _parse_notifications(raw_notif)

    trusted = data.get("trusted_commands")
    if not isinstance(trusted, list):
        trusted = ["python", "pytest", "ruff", "mypy", "make"]

    return TalkBoxConfig(
        default_model=data.get("default_model"),
        default_persona=data.get("default_persona"),
        guardrails=data.get("guardrails", []),
        temperature=data.get("temperature"),
        favorite_models=data.get("favorite_models", []),
        favorite_personas=data.get("favorite_personas", []),
        allow_cloud=data.get("allow_cloud", True),
        mode=_parse_mode(str(data.get("mode", "full"))),
        profiles=profiles,
        knowledge=KnowledgeConfig(sources=knowledge_sources),
        trusted_commands=trusted,
        autonomous=autonomous,
        notifications=notifications,
    )


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------


def _merge_configs(base: TalkBoxConfig, override: TalkBoxConfig) -> TalkBoxConfig:
    """Merge two configs, with *override* winning for set fields.

    Scalar fields in *override* win if they differ from the default.
    Collections are replaced wholesale (not merged element-by-element).
    """
    return TalkBoxConfig(
        default_model=override.default_model
        if override.default_model is not None
        else base.default_model,
        default_persona=override.default_persona
        if override.default_persona is not None
        else base.default_persona,
        guardrails=override.guardrails if override.guardrails else base.guardrails,
        temperature=override.temperature if override.temperature is not None else base.temperature,
        favorite_models=override.favorite_models
        if override.favorite_models
        else base.favorite_models,
        favorite_personas=override.favorite_personas
        if override.favorite_personas
        else base.favorite_personas,
        allow_cloud=override.allow_cloud if not override.allow_cloud else base.allow_cloud,
        mode=override.mode if override.mode != TUIMode.FULL else base.mode,
        profiles={**base.profiles, **override.profiles},
        knowledge=override.knowledge if override.knowledge.sources else base.knowledge,
        trusted_commands=override.trusted_commands
        if override.trusted_commands != ["python", "pytest", "ruff", "mypy", "make"]
        else base.trusted_commands,
        autonomous=override.autonomous
        if override.autonomous.auto_approve or override.autonomous.max_retries != 3
        else base.autonomous,
        notifications=override.notifications
        if override.notifications.webhook_url is not None
        else base.notifications,
    )


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def _load_yaml_file(path: Path) -> dict[str, Any]:
    """Load and parse a YAML file, returning an empty dict on any failure."""
    try:
        data = read_yaml(path)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _find_project_config(start: Path | None = None) -> Path | None:
    """Walk up from *start* (default: cwd) looking for ``talk-box.yml``.

    Returns the path if found, or ``None``.
    """
    current = (start or Path.cwd()).resolve()
    for _ in range(50):  # safety limit
        candidate = current / _PROJECT_CONFIG_NAME
        if candidate.is_file():
            return candidate
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_config(
    *,
    project_dir: Path | None = None,
    global_path: Path | None = None,
) -> TalkBoxConfig:
    """Load and merge the full config stack.

    Resolution order (last wins):
    1. Built-in defaults (``TalkBoxConfig()`` with no arguments)
    2. Global config (``~/.config/talk-box/config.yml``)
    3. Project config (``./talk-box.yml``, searched upward from *project_dir*)

    Layers 4–6 (profile, env vars, CLI flags) are applied later via
    ``TalkBoxConfig.resolve()``.

    Parameters
    ----------
    project_dir
        Directory to start searching for ``talk-box.yml``. Defaults to cwd.
    global_path
        Override the global config path (useful for testing).
    """
    # Layer 1: Built-in defaults
    config = TalkBoxConfig()

    # Layer 2: Global config
    gpath = global_path if global_path is not None else _GLOBAL_CONFIG_PATH
    global_data = _load_yaml_file(gpath)
    if global_data:
        global_config = _parse_config_dict(global_data)
        config = _merge_configs(config, global_config)

    # Layer 3: Project config
    project_file = _find_project_config(project_dir)
    if project_file is not None:
        project_data = _load_yaml_file(project_file)
        if project_data:
            project_config = _parse_config_dict(project_data)
            config = _merge_configs(config, project_config)

    return config


def load_profile(name: str, *, profiles_dir: Path | None = None) -> ProfileConfig:
    """Load a named profile from the profiles directory.

    Parameters
    ----------
    name
        Profile name (filename without ``.yml`` extension).
    profiles_dir
        Override the profiles directory (useful for testing).

    Raises
    ------
    FileNotFoundError
        If the profile file does not exist.
    """
    pdir = profiles_dir if profiles_dir is not None else _PROFILES_DIR
    path = pdir / f"{name}.yml"
    if not path.is_file():
        raise FileNotFoundError(f"Profile not found: {path}")
    data = _load_yaml_file(path)
    return _parse_profile(name, data)


def save_config(config: TalkBoxConfig, path: Path) -> None:
    """Write a config to a YAML file.

    Parameters
    ----------
    config
        The configuration to save.
    path
        Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    write_yaml(config.to_dict(), path)


def save_profile(profile: ProfileConfig, *, profiles_dir: Path | None = None) -> Path:
    """Save a named profile to the profiles directory.

    Parameters
    ----------
    profile
        The profile to save.
    profiles_dir
        Override the profiles directory (useful for testing).

    Returns
    -------
    Path
        The file path where the profile was saved.
    """
    pdir = profiles_dir if profiles_dir is not None else _PROFILES_DIR
    pdir.mkdir(parents=True, exist_ok=True)
    path = pdir / f"{profile.name}.yml"
    write_yaml(profile.to_dict(), path)
    return path


def list_profiles(*, profiles_dir: Path | None = None) -> list[str]:
    """List available profile names from the profiles directory.

    Parameters
    ----------
    profiles_dir
        Override the profiles directory (useful for testing).
    """
    pdir = profiles_dir if profiles_dir is not None else _PROFILES_DIR
    if not pdir.is_dir():
        return []
    return sorted(p.stem for p in pdir.glob("*.yml"))


def global_config_dir() -> Path:
    """Return the global config directory path."""
    return _GLOBAL_CONFIG_DIR


def project_config_path(start: Path | None = None) -> Path | None:
    """Find the nearest ``talk-box.yml`` searching upward from *start*."""
    return _find_project_config(start)


def persist_defaults(
    *,
    model: str | None = None,
    persona: str | None = None,
) -> None:
    """Persist default model and/or persona to the global config.

    Reads the existing global config, updates the specified fields,
    and writes it back. Only fields with non-None values are updated.
    """
    config = _load_yaml_file(_GLOBAL_CONFIG_PATH) if _GLOBAL_CONFIG_PATH.is_file() else {}
    if model is not None:
        config["default_model"] = model
    if persona is not None:
        config["default_persona"] = persona
    _GLOBAL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    write_yaml(config, _GLOBAL_CONFIG_PATH)


def toggle_favorite_model(model: str) -> bool:
    """Toggle a model in the favorites list. Returns True if added, False if removed."""
    config = _load_yaml_file(_GLOBAL_CONFIG_PATH) if _GLOBAL_CONFIG_PATH.is_file() else {}
    favs = config.get("favorite_models", [])
    if not isinstance(favs, list):
        favs = []
    if model in favs:
        favs.remove(model)
        added = False
    else:
        favs.append(model)
        added = True
    config["favorite_models"] = favs
    _GLOBAL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    write_yaml(config, _GLOBAL_CONFIG_PATH)
    return added


def toggle_favorite_persona(persona: str) -> bool:
    """Toggle a persona in the favorites list. Returns True if added, False if removed."""
    config = _load_yaml_file(_GLOBAL_CONFIG_PATH) if _GLOBAL_CONFIG_PATH.is_file() else {}
    favs = config.get("favorite_personas", [])
    if not isinstance(favs, list):
        favs = []
    if persona in favs:
        favs.remove(persona)
        added = False
    else:
        favs.append(persona)
        added = True
    config["favorite_personas"] = favs
    _GLOBAL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    write_yaml(config, _GLOBAL_CONFIG_PATH)
    return added


def get_favorites() -> tuple[list[str], list[str]]:
    """Return (favorite_models, favorite_personas) from global config."""
    config = _load_yaml_file(_GLOBAL_CONFIG_PATH) if _GLOBAL_CONFIG_PATH.is_file() else {}
    models = config.get("favorite_models", [])
    personas = config.get("favorite_personas", [])
    return (
        models if isinstance(models, list) else [],
        personas if isinstance(personas, list) else [],
    )
