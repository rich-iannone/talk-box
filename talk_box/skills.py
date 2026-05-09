"""Skill system for Talk Box: loadable capability packs for agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from yaml12 import parse_yaml

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@dataclass
class SkillDefinition:
    """A loadable skill pack that adds capabilities to an agent.

    A skill bundles instructions, constraints, tools, and context that can be
    attached to any `Agent` or `ChatBot` to grant it domain-specific capabilities
    without modifying the agent's persona.

    Parameters
    ----------
    name
        Machine-readable identifier (e.g., `"sql_analysis"`).
    display_name
        Human-readable name (e.g., `"SQL Analysis"`).
    category
        Grouping category (e.g., `"data"`, `"engineering"`, `"writing"`).
    description
        One-line description of what this skill enables.
    instructions
        Detailed instructions appended to the system prompt when the skill is
        active.  This is the core of the skill — it tells the agent *how* to
        perform the capability.
    constraints
        Constraints the agent must follow when this skill is active.
    tools
        Tool names the skill requires or recommends.
    tags
        Free-form tags for filtering and discovery.
    metadata
        Arbitrary metadata (e.g., version, author).

    Examples
    --------
    Create a skill programmatically:

    ```python
    import talk_box as tb

    skill = tb.create_skill(
        "code_review",
        description="Review code for quality and security issues",
        instructions="Analyze code for bugs, security issues, and style. "
                     "Prioritize: security > correctness > performance > style.",
        constraints=["Always explain the 'why' behind suggestions"],
        tools=["file_reader"],
    )
    skill.name  # "code_review"
    ```

    Register and retrieve:

    ```python
    tb.register_skill(skill)
    tb.get_skill("code_review")  # same skill
    tb.list_skills()             # ["code_review"]
    ```
    """

    name: str
    display_name: str
    category: str
    description: str
    instructions: str = ""
    constraints: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# YAML parsing
# ---------------------------------------------------------------------------


def _parse_skill(data: dict[str, Any]) -> SkillDefinition:
    """Parse a SkillDefinition from a YAML dict."""
    return SkillDefinition(
        name=data["name"],
        display_name=data.get("display_name", data["name"].replace("_", " ").title()),
        category=data.get("category", "general"),
        description=data.get("description", ""),
        instructions=data.get("instructions", ""),
        constraints=data.get("constraints", []),
        tools=data.get("tools", []),
        tags=data.get("tags", []),
        metadata=data.get("metadata", {}),
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_SKILLS_DIR = Path(__file__).parent / "skill_packs"

# Lazily populated cache: name -> SkillDefinition
_cache: dict[str, SkillDefinition] = {}
_scanned = False


def _scan_packs() -> None:
    """Scan the skill_packs/ directory and populate the cache."""
    global _scanned
    if _scanned:
        return
    _scanned = True

    if not _SKILLS_DIR.is_dir():
        return

    for yaml_file in sorted(_SKILLS_DIR.glob("*.yaml")):
        try:
            raw = parse_yaml(yaml_file.read_text(encoding="utf-8"))
            if raw is None:
                continue
            skill = _parse_skill(raw)
            _cache[skill.name] = skill
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


def list_skills() -> list[str]:
    """List all available skill names.

    Returns
    -------
    list[str]
        Sorted list of skill names (built-in + registered).

    Examples
    --------
    ```python
    import talk_box as tb

    names = tb.list_skills()
    ```
    """
    _scan_packs()
    return sorted(_cache.keys())


def get_skill(name: str) -> SkillDefinition:
    """Get a skill definition by name.

    Parameters
    ----------
    name
        The skill name (e.g., `"sql_analysis"`).

    Returns
    -------
    SkillDefinition
        The full skill definition.

    Raises
    ------
    KeyError
        If no skill with that name exists.

    Examples
    --------
    ```python
    import talk_box as tb

    skill = tb.get_skill("sql_analysis")
    skill.description
    ```
    """
    _scan_packs()
    if name not in _cache:
        available = ", ".join(sorted(_cache.keys())) or "(none)"
        raise KeyError(f"Skill '{name}' not found. Available skills: {available}")
    return _cache[name]


def register_skill(skill: SkillDefinition) -> None:
    """Register a skill definition in the global registry.

    Overwrites any existing skill with the same name.

    Parameters
    ----------
    skill
        The skill definition to register.

    Examples
    --------
    ```python
    import talk_box as tb

    skill = tb.create_skill("my_skill", description="Custom skill")
    tb.register_skill(skill)
    tb.get_skill("my_skill")  # works
    ```
    """
    _scan_packs()
    _cache[skill.name] = skill


def load_skill(path: str | Path) -> SkillDefinition:
    """Load a skill from an arbitrary YAML file.

    Parameters
    ----------
    path
        Path to a skill YAML file.

    Returns
    -------
    SkillDefinition
        The parsed skill definition.

    Raises
    ------
    FileNotFoundError
        If the file doesn't exist.

    Examples
    --------
    ```python
    import talk_box as tb

    skill = tb.load_skill("my_skills/summarizer.yaml")
    ```
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Skill file not found: {path}")
    raw = parse_yaml(path.read_text(encoding="utf-8"))
    return _parse_skill(raw)


def skill_categories() -> dict[str, list[str]]:
    """Get all skills grouped by category.

    Returns
    -------
    dict[str, list[str]]
        Mapping of category name to sorted list of skill names.

    Examples
    --------
    ```python
    import talk_box as tb

    cats = tb.skill_categories()
    # {"data": ["sql_analysis"], "engineering": ["code_review"], ...}
    ```
    """
    _scan_packs()
    categories: dict[str, list[str]] = {}
    for skill in _cache.values():
        categories.setdefault(skill.category, []).append(skill.name)
    for names in categories.values():
        names.sort()
    return dict(sorted(categories.items()))


def create_skill(
    name: str,
    *,
    display_name: str | None = None,
    category: str = "custom",
    description: str = "",
    instructions: str = "",
    constraints: list[str] | None = None,
    tools: list[str] | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> SkillDefinition:
    """Create a skill definition programmatically.

    Parameters
    ----------
    name
        Machine-readable identifier (e.g., `"summarizer"`).
    display_name
        Human-readable name. Defaults to *name* with underscores replaced
        by spaces and title-cased.
    category
        Grouping category. Defaults to `"custom"`.
    description
        One-line description of what this skill enables.
    instructions
        Detailed instructions for the agent when this skill is active.
    constraints
        Constraints the agent must follow.
    tools
        Tool names the skill requires.
    tags
        Free-form tags for filtering and discovery.
    metadata
        Arbitrary metadata.

    Returns
    -------
    SkillDefinition
        The new skill definition.

    Examples
    --------
    ```python
    import talk_box as tb

    skill = tb.create_skill(
        "summarizer",
        description="Summarize text into key points",
        instructions="Read the provided text and produce a concise summary. "
                     "Use bullet points for key takeaways.",
        constraints=["Keep summaries under 200 words"],
        tags=["writing", "productivity"],
    )
    ```
    """
    return SkillDefinition(
        name=name,
        display_name=display_name or name.replace("_", " ").title(),
        category=category,
        description=description,
        instructions=instructions,
        constraints=constraints or [],
        tools=tools or [],
        tags=tags or [],
        metadata=metadata or {},
    )
