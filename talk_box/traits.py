"""Composable persona traits: reusable modifiers applied to PersonaDefinitions."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@dataclass
class TraitDefinition:
    """A composable modifier that can be applied to any persona.

    Traits act as mixins — they add constraints, expertise, tools, avoid
    topics, or tags to an existing ``PersonaDefinition`` without replacing
    its core identity.  Multiple traits can be stacked.

    Parameters
    ----------
    name
        Machine-readable identifier (e.g., ``"security_focused"``).
    display_name
        Human-readable label (e.g., ``"Security-Focused"``).
    category
        Grouping category (e.g., ``"tone"``, ``"compliance"``).
    description
        One-line summary of what this trait adds.
    constraints
        Constraints appended to the persona's ``constraints`` list.
    critical_constraints
        Critical constraints appended to the persona's list.
    expertise_extra
        Text appended to the persona's ``expertise`` field.
    avoid_topics
        Topics appended to the persona's ``avoid_topics`` list.
    tools
        Tools appended to the persona's ``tools`` list.
    tags
        Tags appended to the persona's ``tags`` list.
    temperature
        If set, overrides the persona's temperature.
    task_context_extra
        Text appended to the persona's ``task_context``.
    output_format
        Output format directives appended to the persona's list.
    final_emphasis
        If set, replaces the persona's ``final_emphasis``.
    metadata
        Arbitrary metadata for the trait.

    Examples
    --------
    Create a trait programmatically:

    ```python
    import talk_box as tb

    sec = tb.create_trait(
        "security_focused",
        description="Adds security-related constraints.",
        constraints=["Flag potential security issues"],
        critical_constraints=["Never suggest insecure patterns"],
    )
    ```

    Apply a trait to a persona:

    ```python
    persona = tb.get_persona("code_reviewer")
    secure_reviewer = tb.apply_trait(persona, sec)
    secure_reviewer.constraints  # includes the security constraint
    ```
    """

    name: str
    display_name: str = ""
    category: str = "general"
    description: str = ""

    # Additive fields
    constraints: list[str] = field(default_factory=list)
    critical_constraints: list[str] = field(default_factory=list)
    expertise_extra: str = ""
    avoid_topics: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    # Optional overrides
    temperature: float | None = None
    task_context_extra: str = ""
    output_format: list[str] = field(default_factory=list)
    final_emphasis: str = ""

    # Extra
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.display_name:
            self.display_name = self.name.replace("_", " ").title()


# ---------------------------------------------------------------------------
# apply_trait
# ---------------------------------------------------------------------------


def apply_trait(
    persona: Any,
    trait: TraitDefinition,
) -> Any:
    """Apply a trait to a persona, returning a new modified persona.

    The original persona is not mutated.  List fields are merged
    (de-duplicated), string fields are appended, and ``temperature``
    is overridden only when the trait specifies one.

    Parameters
    ----------
    persona
        A ``PersonaDefinition`` instance.
    trait
        The ``TraitDefinition`` to apply.

    Returns
    -------
    PersonaDefinition
        A *new* persona with the trait's modifications merged in.

    Examples
    --------
    ```python
    import talk_box as tb

    persona = tb.get_persona("data_analyst")
    concise = tb.get_trait("concise")
    modified = tb.apply_trait(persona, concise)
    ```
    """
    new = copy.deepcopy(persona)

    # Merge list fields (de-duplicate while preserving order)
    def _merge_list(existing: list[str], additions: list[str]) -> list[str]:
        seen = set(existing)
        merged = list(existing)
        for item in additions:
            if item not in seen:
                merged.append(item)
                seen.add(item)
        return merged

    new.constraints = _merge_list(new.constraints, trait.constraints)
    new.critical_constraints = _merge_list(new.critical_constraints, trait.critical_constraints)
    new.avoid_topics = _merge_list(new.avoid_topics, trait.avoid_topics)
    new.tools = _merge_list(new.tools, trait.tools)
    new.tags = _merge_list(new.tags, trait.tags)
    new.output_format = _merge_list(new.output_format, trait.output_format)

    # Append expertise
    if trait.expertise_extra:
        if new.expertise:
            new.expertise = f"{new.expertise}, {trait.expertise_extra}"
        else:
            new.expertise = trait.expertise_extra

    # Append task context
    if trait.task_context_extra:
        if new.task_context:
            new.task_context = f"{new.task_context} {trait.task_context_extra}"
        else:
            new.task_context = trait.task_context_extra

    # Override temperature if trait specifies one
    if trait.temperature is not None:
        new.temperature = trait.temperature

    # Override final emphasis if trait specifies one
    if trait.final_emphasis:
        new.final_emphasis = trait.final_emphasis

    return new


# ---------------------------------------------------------------------------
# YAML parsing
# ---------------------------------------------------------------------------


def _parse_trait(data: dict[str, Any]) -> TraitDefinition:
    """Parse a TraitDefinition from a YAML dict."""
    return TraitDefinition(
        name=data["name"],
        display_name=data.get("display_name", ""),
        category=data.get("category", "general"),
        description=data.get("description", ""),
        constraints=data.get("constraints", []),
        critical_constraints=data.get("critical_constraints", []),
        expertise_extra=data.get("expertise_extra", ""),
        avoid_topics=data.get("avoid_topics", []),
        tools=data.get("tools", []),
        tags=data.get("tags", []),
        temperature=data.get("temperature"),
        task_context_extra=data.get("task_context_extra", ""),
        output_format=data.get("output_format", []),
        final_emphasis=data.get("final_emphasis", ""),
        metadata=data.get("metadata", {}),
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_TRAITS_DIR = Path(__file__).parent / "trait_packs"

_cache: dict[str, TraitDefinition] = {}
_scanned = False


def _scan_packs() -> None:
    """Scan the trait_packs/ directory and populate the cache."""
    global _scanned
    if _scanned:
        return
    _scanned = True

    if not _TRAITS_DIR.is_dir():
        return

    for yaml_file in sorted(_TRAITS_DIR.glob("*.yaml")):
        try:
            raw = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
            if raw is None:
                continue
            trait = _parse_trait(raw)
            _cache[trait.name] = trait
        except Exception:
            continue


def _reset_cache() -> None:
    """Reset the cache (for testing)."""
    global _scanned
    _cache.clear()
    _scanned = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_trait(
    name: str,
    *,
    display_name: str = "",
    category: str = "custom",
    description: str = "",
    constraints: list[str] | None = None,
    critical_constraints: list[str] | None = None,
    expertise_extra: str = "",
    avoid_topics: list[str] | None = None,
    tools: list[str] | None = None,
    tags: list[str] | None = None,
    temperature: float | None = None,
    task_context_extra: str = "",
    output_format: list[str] | None = None,
    final_emphasis: str = "",
    metadata: dict[str, Any] | None = None,
) -> TraitDefinition:
    """Create a trait definition programmatically.

    Parameters
    ----------
    name
        Machine-readable identifier (e.g., ``"security_focused"``).
    display_name
        Human-readable label. Defaults to *name* title-cased.
    category
        Grouping category. Defaults to ``"custom"``.
    description
        One-line description of what this trait adds.
    constraints
        Constraints to append to a persona.
    critical_constraints
        Critical constraints to append.
    expertise_extra
        Text appended to a persona's expertise.
    avoid_topics
        Topics to add to a persona's avoid list.
    tools
        Tool names to add.
    tags
        Tags to add.
    temperature
        Temperature override (applied when the trait is applied).
    task_context_extra
        Text appended to a persona's task context.
    output_format
        Output format directives to append.
    final_emphasis
        Replaces a persona's final emphasis when the trait is applied.
    metadata
        Arbitrary metadata.

    Returns
    -------
    TraitDefinition
        A new trait definition.

    Examples
    --------
    ```python
    import talk_box as tb

    friendly = tb.create_trait(
        "junior_friendly",
        description="Adjusts output for beginners.",
        constraints=["Explain jargon when first used"],
    )
    ```
    """
    return TraitDefinition(
        name=name,
        display_name=display_name,
        category=category,
        description=description,
        constraints=constraints or [],
        critical_constraints=critical_constraints or [],
        expertise_extra=expertise_extra,
        avoid_topics=avoid_topics or [],
        tools=tools or [],
        tags=tags or [],
        temperature=temperature,
        task_context_extra=task_context_extra,
        output_format=output_format or [],
        final_emphasis=final_emphasis,
        metadata=metadata or {},
    )


def register_trait(trait: TraitDefinition) -> None:
    """Register a trait so it can be retrieved with ``get_trait()``.

    Parameters
    ----------
    trait
        The trait definition to register.

    Raises
    ------
    ValueError
        If a trait with the same name is already registered.

    Examples
    --------
    ```python
    import talk_box as tb

    t = tb.create_trait("my_trait", description="Custom trait.")
    tb.register_trait(t)
    tb.get_trait("my_trait")
    ```
    """
    _scan_packs()
    if trait.name in _cache:
        raise ValueError(
            f"Trait '{trait.name}' already exists. "
            "Use a different name or call _reset_cache() first."
        )
    _cache[trait.name] = trait


def get_trait(name: str) -> TraitDefinition:
    """Get a registered trait by name.

    Parameters
    ----------
    name
        The trait name (e.g., ``"concise"``).

    Returns
    -------
    TraitDefinition
        The trait definition.

    Raises
    ------
    KeyError
        If no trait with that name exists.

    Examples
    --------
    ```python
    import talk_box as tb

    concise = tb.get_trait("concise")
    concise.description
    ```
    """
    _scan_packs()
    if name not in _cache:
        available = ", ".join(sorted(_cache.keys())) or "(none)"
        raise KeyError(f"Trait '{name}' not found. Available traits: {available}")
    return _cache[name]


def list_traits() -> list[str]:
    """List all available trait names.

    Returns
    -------
    list[str]
        Sorted list of trait names.

    Examples
    --------
    ```python
    import talk_box as tb

    names = tb.list_traits()
    "concise" in names  # True
    ```
    """
    _scan_packs()
    return sorted(_cache.keys())


def load_trait(path: str | Path) -> TraitDefinition:
    """Load a trait from an arbitrary YAML file path.

    Parameters
    ----------
    path
        Path to a trait YAML file.

    Returns
    -------
    TraitDefinition
        The parsed trait definition.

    Raises
    ------
    FileNotFoundError
        If the file doesn't exist.

    Examples
    --------
    ```python
    import talk_box as tb

    t = tb.load_trait("my_traits/strict_security.yaml")
    ```
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Trait file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _parse_trait(raw)


def trait_categories() -> dict[str, list[str]]:
    """Get all traits grouped by category.

    Returns
    -------
    dict[str, list[str]]
        Mapping of category to sorted list of trait names.

    Examples
    --------
    ```python
    import talk_box as tb

    cats = tb.trait_categories()
    cats["tone"]  # ["concise", "formal", "verbose"]
    ```
    """
    _scan_packs()
    categories: dict[str, list[str]] = {}
    for trait in _cache.values():
        categories.setdefault(trait.category, []).append(trait.name)
    for names in categories.values():
        names.sort()
    return dict(sorted(categories.items()))
