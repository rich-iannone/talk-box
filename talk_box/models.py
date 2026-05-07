from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import great_tables as gt


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class CostTier(Enum):
    """Relative cost tier for a model."""

    FREE = "free"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    PREMIUM = "premium"


@dataclass(frozen=True)
class ModelProfile:
    """Capability profile for a specific LLM model.

    Parameters
    ----------
    provider
        Provider name (e.g., `"anthropic"`, `"openai"`, `"ollama"`).
    model
        Model identifier (e.g., `"claude-sonnet-4-6"`, `"gpt-4o"`).
    display_name
        Human-readable name for display in tables and reports.
    context_window
        Maximum context window size in tokens.
    max_output_tokens
        Maximum output tokens the model can generate, if known.
    supports_tools
        Whether the model supports tool/function calling.
    supports_vision
        Whether the model can process image inputs.
    supports_structured_output
        Whether the model supports structured (JSON schema) output.
    supports_streaming
        Whether the model supports streaming responses.
    cost_tier
        Relative cost tier for the model.
    knowledge_cutoff
        Knowledge cutoff date string (e.g., "2025-04"), if known.
    notes
        Any additional notes about the model.
    """

    provider: str
    model: str
    display_name: str = ""
    context_window: int | None = None
    max_output_tokens: int | None = None
    supports_tools: bool | None = None
    supports_vision: bool | None = None
    supports_structured_output: bool | None = None
    supports_streaming: bool | None = None
    cost_tier: CostTier | None = None
    knowledge_cutoff: str | None = None
    notes: str = ""

    @property
    def key(self) -> str:
        """Canonical ``provider:model`` key."""
        return f"{self.provider}:{self.model}"

    @property
    def name(self) -> str:
        """Display name, falling back to model identifier."""
        return self.display_name or self.model

    def supports(self, capability: str) -> bool | None:
        """Check whether a capability is supported.

        Parameters
        ----------
        capability
            One of `"tools"``, `"vision"``, `"structured_output"``, `"streaming"`.

        Returns
        -------
        bool | None
            `True`/`False` if known, `None` if unknown.

        Raises
        ------
        ValueError
            If the capability name is not recognised.
        """
        mapping = {
            "tools": self.supports_tools,
            "vision": self.supports_vision,
            "structured_output": self.supports_structured_output,
            "streaming": self.supports_streaming,
        }
        if capability not in mapping:
            raise ValueError(
                f"Unknown capability {capability!r}. Valid options: {', '.join(sorted(mapping))}"
            )
        return mapping[capability]


# ---------------------------------------------------------------------------
# Built-in model registry
# ---------------------------------------------------------------------------

_PROFILES: dict[str, ModelProfile] = {}


def _register(*profiles: ModelProfile) -> None:
    """Add profiles to the registry."""
    for p in profiles:
        _PROFILES[p.key] = p


# ── Anthropic ──────────────────────────────────────────────────────────────

_register(
    ModelProfile(
        provider="anthropic",
        model="claude-opus-4-7",
        display_name="Claude Opus 4 (July)",
        context_window=200_000,
        max_output_tokens=32_000,
        supports_tools=True,
        supports_vision=True,
        supports_structured_output=True,
        supports_streaming=True,
        cost_tier=CostTier.PREMIUM,
        knowledge_cutoff="2025-03",
    ),
    ModelProfile(
        provider="anthropic",
        model="claude-sonnet-4-6",
        display_name="Claude Sonnet 4 (June)",
        context_window=200_000,
        max_output_tokens=16_000,
        supports_tools=True,
        supports_vision=True,
        supports_structured_output=True,
        supports_streaming=True,
        cost_tier=CostTier.HIGH,
        knowledge_cutoff="2025-03",
    ),
    ModelProfile(
        provider="anthropic",
        model="claude-haiku-4-5",
        display_name="Claude Haiku 4 (May)",
        context_window=200_000,
        max_output_tokens=8_192,
        supports_tools=True,
        supports_vision=True,
        supports_structured_output=True,
        supports_streaming=True,
        cost_tier=CostTier.LOW,
        knowledge_cutoff="2025-03",
    ),
)

# ── OpenAI ─────────────────────────────────────────────────────────────────

_register(
    ModelProfile(
        provider="openai",
        model="gpt-4o",
        display_name="GPT-4o",
        context_window=128_000,
        max_output_tokens=16_384,
        supports_tools=True,
        supports_vision=True,
        supports_structured_output=True,
        supports_streaming=True,
        cost_tier=CostTier.HIGH,
        knowledge_cutoff="2024-06",
    ),
    ModelProfile(
        provider="openai",
        model="gpt-4o-mini",
        display_name="GPT-4o Mini",
        context_window=128_000,
        max_output_tokens=16_384,
        supports_tools=True,
        supports_vision=True,
        supports_structured_output=True,
        supports_streaming=True,
        cost_tier=CostTier.LOW,
        knowledge_cutoff="2024-06",
    ),
    ModelProfile(
        provider="openai",
        model="o3",
        display_name="o3",
        context_window=200_000,
        max_output_tokens=100_000,
        supports_tools=True,
        supports_vision=True,
        supports_structured_output=True,
        supports_streaming=True,
        cost_tier=CostTier.PREMIUM,
        knowledge_cutoff="2025-03",
    ),
    ModelProfile(
        provider="openai",
        model="o3-mini",
        display_name="o3 Mini",
        context_window=200_000,
        max_output_tokens=100_000,
        supports_tools=True,
        supports_vision=False,
        supports_structured_output=True,
        supports_streaming=True,
        cost_tier=CostTier.MEDIUM,
        knowledge_cutoff="2025-01",
    ),
    ModelProfile(
        provider="openai",
        model="o4-mini",
        display_name="o4 Mini",
        context_window=200_000,
        max_output_tokens=100_000,
        supports_tools=True,
        supports_vision=True,
        supports_structured_output=True,
        supports_streaming=True,
        cost_tier=CostTier.MEDIUM,
        knowledge_cutoff="2025-03",
    ),
)

# ── Google ─────────────────────────────────────────────────────────────────

_register(
    ModelProfile(
        provider="google",
        model="gemini-2.5-pro",
        display_name="Gemini 2.5 Pro",
        context_window=1_000_000,
        max_output_tokens=65_536,
        supports_tools=True,
        supports_vision=True,
        supports_structured_output=True,
        supports_streaming=True,
        cost_tier=CostTier.HIGH,
        knowledge_cutoff="2025-03",
    ),
    ModelProfile(
        provider="google",
        model="gemini-2.5-flash",
        display_name="Gemini 2.5 Flash",
        context_window=1_000_000,
        max_output_tokens=65_536,
        supports_tools=True,
        supports_vision=True,
        supports_structured_output=True,
        supports_streaming=True,
        cost_tier=CostTier.LOW,
        knowledge_cutoff="2025-03",
    ),
)

# ── GitHub (Copilot Models) ────────────────────────────────────────────────

_register(
    ModelProfile(
        provider="github",
        model="gpt-4o",
        display_name="GPT-4o (GitHub)",
        context_window=128_000,
        max_output_tokens=16_384,
        supports_tools=True,
        supports_vision=True,
        supports_structured_output=True,
        supports_streaming=True,
        cost_tier=CostTier.FREE,
        knowledge_cutoff="2024-06",
        notes="Free via GitHub Copilot; requires GITHUB_TOKEN with copilot scope",
    ),
    ModelProfile(
        provider="github",
        model="o4-mini",
        display_name="o4 Mini (GitHub)",
        context_window=200_000,
        max_output_tokens=100_000,
        supports_tools=True,
        supports_vision=True,
        supports_structured_output=True,
        supports_streaming=True,
        cost_tier=CostTier.FREE,
        knowledge_cutoff="2025-03",
        notes="Free via GitHub Copilot; requires GITHUB_TOKEN with copilot scope",
    ),
)

# ── DeepSeek ───────────────────────────────────────────────────────────────

_register(
    ModelProfile(
        provider="deepseek",
        model="deepseek-chat",
        display_name="DeepSeek V3",
        context_window=64_000,
        max_output_tokens=8_192,
        supports_tools=True,
        supports_vision=False,
        supports_structured_output=True,
        supports_streaming=True,
        cost_tier=CostTier.LOW,
        knowledge_cutoff="2025-02",
    ),
    ModelProfile(
        provider="deepseek",
        model="deepseek-reasoner",
        display_name="DeepSeek R1",
        context_window=64_000,
        max_output_tokens=8_192,
        supports_tools=False,
        supports_vision=False,
        supports_structured_output=False,
        supports_streaming=True,
        cost_tier=CostTier.LOW,
        knowledge_cutoff="2025-02",
    ),
)

# ── Groq ───────────────────────────────────────────────────────────────────

_register(
    ModelProfile(
        provider="groq",
        model="llama-3.3-70b-versatile",
        display_name="Llama 3.3 70B (Groq)",
        context_window=128_000,
        max_output_tokens=32_768,
        supports_tools=True,
        supports_vision=False,
        supports_structured_output=True,
        supports_streaming=True,
        cost_tier=CostTier.FREE,
        notes="Free tier with rate limits",
    ),
    ModelProfile(
        provider="groq",
        model="gemma2-9b-it",
        display_name="Gemma 2 9B (Groq)",
        context_window=8_192,
        max_output_tokens=8_192,
        supports_tools=True,
        supports_vision=False,
        supports_structured_output=True,
        supports_streaming=True,
        cost_tier=CostTier.FREE,
        notes="Free tier with rate limits",
    ),
)

# ── Mistral ────────────────────────────────────────────────────────────────

_register(
    ModelProfile(
        provider="mistral",
        model="mistral-large-latest",
        display_name="Mistral Large",
        context_window=128_000,
        max_output_tokens=8_192,
        supports_tools=True,
        supports_vision=False,
        supports_structured_output=True,
        supports_streaming=True,
        cost_tier=CostTier.MEDIUM,
    ),
    ModelProfile(
        provider="mistral",
        model="mistral-small-latest",
        display_name="Mistral Small",
        context_window=128_000,
        max_output_tokens=8_192,
        supports_tools=True,
        supports_vision=False,
        supports_structured_output=True,
        supports_streaming=True,
        cost_tier=CostTier.LOW,
    ),
)

# ── Ollama (common local models) ──────────────────────────────────────────

_register(
    ModelProfile(
        provider="ollama",
        model="llama3.3",
        display_name="Llama 3.3 (Ollama)",
        context_window=128_000,
        max_output_tokens=None,
        supports_tools=True,
        supports_vision=False,
        supports_structured_output=True,
        supports_streaming=True,
        cost_tier=CostTier.FREE,
        notes="Local model; performance depends on hardware",
    ),
    ModelProfile(
        provider="ollama",
        model="qwen2.5-coder:32b",
        display_name="Qwen 2.5 Coder 32B (Ollama)",
        context_window=32_768,
        max_output_tokens=None,
        supports_tools=True,
        supports_vision=False,
        supports_structured_output=True,
        supports_streaming=True,
        cost_tier=CostTier.FREE,
        notes="Local model; strong at code generation",
    ),
    ModelProfile(
        provider="ollama",
        model="gemma3:27b",
        display_name="Gemma 3 27B (Ollama)",
        context_window=128_000,
        max_output_tokens=None,
        supports_tools=True,
        supports_vision=True,
        supports_structured_output=True,
        supports_streaming=True,
        cost_tier=CostTier.FREE,
        notes="Local model; Google's open-weight model",
    ),
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_model_profile(model_key: str) -> ModelProfile | None:
    """Look up a model's capability profile.

    Parameters
    ----------
    model_key
        Either a `"provider:model"` string (e.g., `"anthropic:claude-sonnet-4-6"`) or a bare model
        name (e.g., `"gpt-4o"`). Bare names are searched across all providers; the first match is
        returned.

    Returns
    -------
    ModelProfile | None
        The profile if found, otherwise `None`.

    Examples
    --------
    ```python
    import talk_box as tb

    profile = tb.get_model_profile("anthropic:claude-sonnet-4-6")
    profile.context_window   # 200_000
    profile.supports_tools   # True
    profile.cost_tier        # CostTier.HIGH
    ```
    """
    # Exact match
    if model_key in _PROFILES:
        return _PROFILES[model_key]

    # Bare model name — search across providers
    if ":" not in model_key:
        for key, profile in _PROFILES.items():
            if key.endswith(f":{model_key}"):
                return profile

    return None


def list_models(
    *,
    provider: str | None = None,
    supports_tools: bool | None = None,
    supports_vision: bool | None = None,
    cost_tier: CostTier | None = None,
) -> list[ModelProfile]:
    """List known model profiles, optionally filtered.

    Parameters
    ----------
    provider
        Filter to a specific provider (e.g., `"anthropic"`).
    supports_tools
        Filter to models that support (or don't support) tool calling.
    supports_vision
        Filter to models that support (or don't support) vision.
    cost_tier
        Filter to a specific cost tier.

    Returns
    -------
    list[ModelProfile]
        Matching profiles, sorted by provider then model name.

    Examples
    --------
    ```python
    import talk_box as tb

    # All Anthropic models
    tb.list_models(provider="anthropic")

    # Free models with tool support
    tb.list_models(supports_tools=True, cost_tier=tb.CostTier.FREE)

    # Models that support vision
    tb.list_models(supports_vision=True)
    ```
    """
    results = list(_PROFILES.values())

    if provider is not None:
        results = [p for p in results if p.provider == provider]
    if supports_tools is not None:
        results = [p for p in results if p.supports_tools == supports_tools]
    if supports_vision is not None:
        results = [p for p in results if p.supports_vision == supports_vision]
    if cost_tier is not None:
        results = [p for p in results if p.cost_tier == cost_tier]

    results.sort(key=lambda p: (p.provider, p.model))
    return results


def register_model(profile: ModelProfile) -> None:
    """Register a custom model profile.

    Use this to add profiles for models not included in the built-in registry (e.g., fine-tuned
    models or new releases).

    Parameters
    ----------
    profile
        The model profile to register.

    Examples
    --------
    ```python
    import talk_box as tb

    tb.register_model(tb.ModelProfile(
        provider="openai",
        model="ft:gpt-4o:my-org:custom:id",
        display_name="My Fine-tuned GPT-4o",
        context_window=128_000,
        supports_tools=True,
        supports_vision=True,
        supports_structured_output=True,
        supports_streaming=True,
        cost_tier=tb.CostTier.HIGH,
    ))
    ```
    """
    _PROFILES[profile.key] = profile


def model_profiles_table() -> "gt.GT":
    """Render all known model profiles as a Great Table.

    Returns
    -------
    gt.GT
        A formatted table with one row per model, showing capabilities, context window, and cost
        tier.

    Raises
    ------
    ImportError
        If great_tables or pandas is not installed.

    Examples
    --------
    ```python
    import talk_box as tb

    tb.model_profiles_table()  # renders in notebook / Quarto
    ```
    """
    try:
        import great_tables as gt
    except ImportError:
        raise ImportError(
            "great_tables is required for model_profiles_table(). "
            "Install with: pip install great_tables"
        )
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for model_profiles_table(). Install with: pip install pandas"
        )

    profiles = list_models()
    if not profiles:
        return gt.GT(pd.DataFrame({"Status": ["No models registered"]}))

    def _bool_icon(val: bool | None) -> str:
        if val is True:
            return "✓"
        elif val is False:
            return "✗"
        return "?"

    rows: list[dict[str, Any]] = []
    for p in profiles:
        rows.append(
            {
                "Provider": p.provider,
                "Model": p.name,
                "Context": p.context_window,
                "Max Output": p.max_output_tokens,
                "Tools": _bool_icon(p.supports_tools),
                "Vision": _bool_icon(p.supports_vision),
                "Structured": _bool_icon(p.supports_structured_output),
                "Cost": p.cost_tier.value if p.cost_tier else "?",
            }
        )

    df = pd.DataFrame(rows)

    table = (
        gt.GT(df, rowname_col="Model", groupname_col="Provider")
        .tab_header(
            title="Model Capability Profiles",
            subtitle=f"{len(profiles)} models across {len({p.provider for p in profiles})} providers",
        )
        .fmt_number(columns="Context", use_seps=True, decimals=0)
        .fmt_number(columns="Max Output", use_seps=True, decimals=0)
        .sub_missing(missing_text="—")
        .tab_spanner(label="Capabilities", columns=["Tools", "Vision", "Structured"])
        .tab_style(style=gt.style.text(weight="bold"), locations=gt.loc.column_labels())
    )

    return table
