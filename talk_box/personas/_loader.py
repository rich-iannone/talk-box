"""Persona loader and registry for Talk Box."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@dataclass
class ModelRecommendation:
    """A recommended model configuration for a persona."""

    provider_model: str
    context: str = ""
    temperature: float | None = None


@dataclass
class PersonaDefinition:
    """Represent a complete persona loaded from a YAML definition file.

    A persona bundles everything needed for a production AI assistant:
    system prompt configuration, recommended models, tools, avoid topics,
    and test expectations. Load personas with `get_persona()` or browse
    the full catalog with `list_personas()`.

    Attributes
    ----------
    name
        Machine-readable identifier (e.g., `"customer_support_tier1"`).
    display_name
        Human-readable name (e.g., `"Customer Support (Tier 1)"`).
    category
        Grouping category (e.g., `"business"`, `"technical"`, `"creative"`).
    description
        One-line description of what this persona does.
    persona_role
        Role string passed to `PromptBuilder.persona()` (e.g.,
        `"senior customer support specialist"`).
    expertise
        Expertise string passed to `PromptBuilder.persona()`
        (e.g., `"customer service, conflict resolution"`).
    task_context
        Optional task context for `PromptBuilder.task_context()`.
    critical_constraints
        List of non-negotiable constraints for
        `PromptBuilder.critical_constraint()`.
    constraints
        List of important constraints for `PromptBuilder.constraint()`.
    core_analysis
        Analysis tasks for `PromptBuilder.core_analysis()`.
    output_format
        Output formatting instructions for `PromptBuilder.output_format()`.
    final_emphasis
        Recency-bias emphasis for `PromptBuilder.final_emphasis()`.
    avoid_topics
        Topics the persona must avoid discussing.
    tools
        Tool names the persona should have enabled.
    recommended_models
        List of recommended model configurations.
    temperature
        Default temperature for this persona.
    max_tokens
        Default max token limit for this persona.
    default_guards
        Guardrail specs applied automatically by ``persona_pack()``.
        Each entry is a guard name string or a dict mapping a guard name
        to its keyword arguments.
    tags
        Free-form tags for filtering and discovery.
    test_queries
        Sample queries for automated testing.

    Examples
    --------
    Load a persona by name and inspect its fields:

    ```python
    import talk_box as tb

    persona = tb.get_persona("python_mentor")
    persona.display_name
    ```

    Build a `PromptBuilder` from a persona:

    ```python
    builder = persona.build_prompt_builder()
    print(str(builder)[:200])
    ```

    %seealso get_persona, list_personas, persona_categories, PromptBuilder
    """

    name: str
    display_name: str
    category: str
    description: str

    # PromptBuilder fields
    persona_role: str
    expertise: str = ""
    task_context: str = ""
    critical_constraints: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    core_analysis: list[str] = field(default_factory=list)
    output_format: list[str] = field(default_factory=list)
    final_emphasis: str = ""

    # ChatBot fields
    avoid_topics: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    recommended_models: list[ModelRecommendation] = field(default_factory=list)
    temperature: float | None = None
    max_tokens: int | None = None
    default_guards: list[str | dict[str, Any]] = field(default_factory=list)

    # Metadata
    tags: list[str] = field(default_factory=list)
    test_queries: list[str] = field(default_factory=list)

    def build_prompt_builder(self):
        """
        Build an attention-optimized `PromptBuilder` from this persona's fields.

        Returns the `PromptBuilder` instance so callers can inspect, extend, or
        further customize the prompt before converting to a string.

        Returns
        -------
        PromptBuilder
            A configured `PromptBuilder` instance ready to use or extend.

        Examples
        --------
        ```python
        from talk_box.personas import get_persona

        persona = get_persona("code_reviewer")
        builder = persona.build_prompt_builder()

        # Extend before use
        builder.constraint("Always suggest type annotations")
        print(builder)
        ```
        """
        from talk_box.prompt_builder import PromptBuilder

        builder = PromptBuilder()

        # Persona (primacy position)
        builder.persona(self.persona_role, self.expertise or None)

        # Task context
        if self.task_context:
            builder.task_context(self.task_context)

        # Critical constraints (high-attention position)
        for constraint in self.critical_constraints:
            builder.critical_constraint(constraint)

        # Regular constraints
        for constraint in self.constraints:
            builder.constraint(constraint)

        # Core analysis tasks
        if self.core_analysis:
            builder.core_analysis(self.core_analysis)

        # Output format
        if self.output_format:
            builder.output_format(self.output_format)

        # Avoid topics
        if self.avoid_topics:
            builder.avoid_topics(self.avoid_topics)

        # Final emphasis (recency position)
        if self.final_emphasis:
            builder.final_emphasis(self.final_emphasis)

        return builder

    def build_system_prompt(self) -> str:
        """
        Build an attention-optimized system prompt using `PromptBuilder`.

        Convenience wrapper around `build_prompt_builder()` that returns the
        final string.

        Returns
        -------
        str
            The fully assembled system prompt.
        """
        return str(self.build_prompt_builder())


# ---------------------------------------------------------------------------
# YAML parsing
# ---------------------------------------------------------------------------


def _parse_model_recommendation(data: dict[str, Any]) -> ModelRecommendation:
    """Parse a model recommendation from YAML data."""
    return ModelRecommendation(
        provider_model=data["provider_model"],
        context=data.get("context", ""),
        temperature=data.get("temperature"),
    )


def _parse_persona(data: dict[str, Any]) -> PersonaDefinition:
    """Parse a PersonaDefinition from a YAML dict."""
    models = [_parse_model_recommendation(m) for m in data.get("recommended_models", [])]

    return PersonaDefinition(
        name=data["name"],
        display_name=data.get("display_name", data["name"].replace("_", " ").title()),
        category=data.get("category", "general"),
        description=data.get("description", ""),
        persona_role=data["persona_role"],
        expertise=data.get("expertise", ""),
        task_context=data.get("task_context", ""),
        critical_constraints=data.get("critical_constraints", []),
        constraints=data.get("constraints", []),
        core_analysis=data.get("core_analysis", []),
        output_format=data.get("output_format", []),
        final_emphasis=data.get("final_emphasis", ""),
        avoid_topics=data.get("avoid_topics", []),
        tools=data.get("tools", []),
        recommended_models=models,
        temperature=data.get("temperature"),
        max_tokens=data.get("max_tokens"),
        default_guards=data.get("default_guards", []),
        tags=data.get("tags", []),
        test_queries=data.get("test_queries", []),
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_PERSONAS_DIR = Path(__file__).parent / "packs"

# Lazily populated cache: name -> PersonaDefinition
_cache: dict[str, PersonaDefinition] = {}
_scanned = False


def _scan_packs() -> None:
    """Scan the packs/ directory and populate the cache."""
    global _scanned
    if _scanned:
        return
    _scanned = True

    if not _PERSONAS_DIR.is_dir():
        return

    for yaml_file in sorted(_PERSONAS_DIR.glob("*.yaml")):
        try:
            raw = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
            if raw is None:
                continue
            persona = _parse_persona(raw)
            _cache[persona.name] = persona
        except Exception:
            # Skip malformed files silently during scan
            continue


def _reset_cache() -> None:
    """Reset the cache (for testing)."""
    global _scanned
    _cache.clear()
    _scanned = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_personas() -> list[str]:
    """
    List all available persona names.

    Returns
    -------
    list[str]
        Sorted list of persona names.

    Examples
    --------
    >>> from talk_box.personas import list_personas
    >>> names = list_personas()
    >>> "customer_support_tier1" in names
    True
    """
    _scan_packs()
    return sorted(_cache.keys())


def get_persona(name: str) -> PersonaDefinition:
    """
    Get a persona definition by name.

    Parameters
    ----------
    name
        The persona name (e.g., `"code_reviewer"`).

    Returns
    -------
    PersonaDefinition
        The full persona definition.

    Raises
    ------
    KeyError
        If no persona with that name exists.

    Examples
    --------
    >>> from talk_box.personas import get_persona
    >>> persona = get_persona("code_reviewer")
    >>> persona.category
    'technical'
    """
    _scan_packs()
    if name not in _cache:
        available = ", ".join(sorted(_cache.keys())) or "(none)"
        raise KeyError(f"Persona '{name}' not found. Available personas: {available}")
    return _cache[name]


def load_persona(path: str | Path) -> PersonaDefinition:
    """
    Load a persona from an arbitrary YAML file path.

    This is useful for loading custom personas that aren't in the built-in
    packs directory.

    Parameters
    ----------
    path
        Path to a persona YAML file.

    Returns
    -------
    PersonaDefinition
        The parsed persona definition.

    Raises
    ------
    FileNotFoundError
        If the file doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Persona file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _parse_persona(raw)


def persona_categories() -> dict[str, list[str]]:
    """
    Get all personas grouped by category.

    Returns
    -------
    dict[str, list[str]]
        Mapping of category name to list of persona names.

    Examples
    --------
    >>> from talk_box.personas import persona_categories
    >>> cats = persona_categories()
    >>> "technical" in cats
    True
    """
    _scan_packs()
    categories: dict[str, list[str]] = {}
    for persona in _cache.values():
        categories.setdefault(persona.category, []).append(persona.name)
    for names in categories.values():
        names.sort()
    return dict(sorted(categories.items()))


def create_persona(
    name: str,
    persona_role: str,
    *,
    display_name: str | None = None,
    category: str = "custom",
    description: str = "",
    expertise: str = "",
    task_context: str = "",
    critical_constraints: list[str] | None = None,
    constraints: list[str] | None = None,
    core_analysis: list[str] | None = None,
    output_format: list[str] | None = None,
    final_emphasis: str = "",
    avoid_topics: list[str] | None = None,
    tools: list[str] | None = None,
    recommended_models: list[dict[str, Any]] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    tags: list[str] | None = None,
    test_queries: list[str] | None = None,
) -> PersonaDefinition:
    """
    Create a persona definition entirely in Python (no YAML needed).

    This is the recommended approach when you want full programmatic control
    over persona creation, or when you're building personas dynamically.

    Parameters
    ----------
    name
        Machine-readable identifier (e.g., `"onboarding_specialist"`).
    persona_role
        Role string for `PromptBuilder.persona()` (e.g.,
        `"senior onboarding specialist"`).
    display_name
        Human-readable name. Defaults to `name` with underscores replaced
        by spaces and title-cased.
    category
        Grouping category. Defaults to `"custom"`.
    description
        One-line description of what this persona does.
    expertise
        Expertise string for `PromptBuilder.persona()`.
    task_context
        Task context for `PromptBuilder.task_context()`.
    critical_constraints
        Non-negotiable constraints for `PromptBuilder.critical_constraint()`.
    constraints
        Important constraints for `PromptBuilder.constraint()`.
    core_analysis
        Analysis tasks for `PromptBuilder.core_analysis()`.
    output_format
        Formatting instructions for `PromptBuilder.output_format()`.
    final_emphasis
        Recency-bias emphasis for `PromptBuilder.final_emphasis()`.
    avoid_topics
        Topics the persona must avoid discussing.
    tools
        Tool names the persona should have enabled.
    recommended_models
        List of dicts with keys `provider_model`, `context`, `temperature`.
    temperature
        Default temperature for this persona.
    max_tokens
        Default max token limit for this persona.
    tags
        Free-form tags for filtering and discovery.
    test_queries
        Sample queries for automated testing.

    Returns
    -------
    PersonaDefinition
        A fully configured persona definition that can be used with
        `ChatBot.system_prompt(persona.build_prompt_builder())` or
        registered for use with `persona_pack()`.

    Examples
    --------
    ```python
    from talk_box.personas import create_persona, register_persona
    import talk_box as tb

    onboarding = create_persona(
        "onboarding_specialist",
        persona_role="senior employee onboarding specialist",
        expertise="HR processes, company culture, new hire integration",
        task_context="Guide new employees through their first 90 days.",
        critical_constraints=[
            "Never share salary information for other employees",
            "Always direct legal questions to the legal team",
        ],
        constraints=[
            "Be warm and encouraging",
            "Provide step-by-step guidance",
        ],
        core_analysis=[
            "Identify what stage of onboarding the employee is in",
            "Determine what resources they need next",
        ],
        final_emphasis="Make every new hire feel welcome and supported.",
        temperature=0.5,
    )

    # Use directly with ChatBot
    bot = tb.ChatBot().system_prompt(onboarding.build_prompt_builder())

    # Or register it for use with persona_pack()
    register_persona(onboarding)
    bot = tb.ChatBot().persona_pack("onboarding_specialist")
    ```
    """
    models = [_parse_model_recommendation(m) for m in (recommended_models or [])]

    return PersonaDefinition(
        name=name,
        display_name=display_name or name.replace("_", " ").title(),
        category=category,
        description=description,
        persona_role=persona_role,
        expertise=expertise,
        task_context=task_context,
        critical_constraints=critical_constraints or [],
        constraints=constraints or [],
        core_analysis=core_analysis or [],
        output_format=output_format or [],
        final_emphasis=final_emphasis,
        avoid_topics=avoid_topics or [],
        tools=tools or [],
        recommended_models=models,
        temperature=temperature,
        max_tokens=max_tokens,
        tags=tags or [],
        test_queries=test_queries or [],
    )


def register_persona(persona: PersonaDefinition) -> None:
    """
    Register a persona definition so it can be loaded via `persona_pack()`.

    This adds the persona to the in-memory registry. It does not write
    a YAML file. Registered personas persist for the lifetime of the
    process.

    Parameters
    ----------
    persona
        The persona definition to register.

    Raises
    ------
    ValueError
        If a persona with the same name already exists.

    Examples
    --------
    ```python
    from talk_box.personas import create_persona, register_persona
    import talk_box as tb

    custom = create_persona(
        "my_helper",
        persona_role="friendly assistant",
        description="A custom helper persona.",
    )
    register_persona(custom)

    # Now usable via persona_pack()
    bot = tb.ChatBot().persona_pack("my_helper")
    ```
    """
    _scan_packs()
    if persona.name in _cache:
        raise ValueError(
            f"Persona '{persona.name}' already exists. "
            "Use a different name or call _reset_cache() first."
        )
    _cache[persona.name] = persona
