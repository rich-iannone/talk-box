"""Skill system for Talk Box: loadable capability packs for agents."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

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


def discover_skills(
    *search_paths: str | Path,
    scan_cwd: bool = True,
) -> list[SkillDefinition]:
    """Discover skills from ``SKILL.md`` files across multiple directories.

    Scans the following locations (in order):

    1. ``~/.config/talk-box/skills/``
    2. ``./skills/`` (current working directory)
    3. Any additional *search_paths* provided

    Each ``SKILL.md`` file should have a YAML frontmatter block (delimited
    by ``---``) with at minimum a ``name`` field, followed by Markdown
    body used as the skill's ``instructions``.

    Discovered skills are automatically registered in the global registry.

    Parameters
    ----------
    *search_paths
        Additional directories to scan.
    scan_cwd
        Whether to scan ``./skills/`` in the current directory.

    Returns
    -------
    list[SkillDefinition]
        Newly discovered skill definitions.

    Examples
    --------
    ```python
    import talk_box as tb

    skills = tb.discover_skills()
    for s in skills:
        print(s.name, s.description)
    ```
    """
    import os

    dirs: list[Path] = []

    # Global skills dir
    global_dir = Path(os.path.expanduser("~/.config/talk-box/skills"))
    if global_dir.is_dir():
        dirs.append(global_dir)

    # CWD skills dir
    if scan_cwd:
        cwd_skills = Path.cwd() / "skills"
        if cwd_skills.is_dir():
            dirs.append(cwd_skills)

    # Extra search paths
    for p in search_paths:
        d = Path(p)
        if d.is_dir():
            dirs.append(d)

    discovered: list[SkillDefinition] = []
    seen_names: set[str] = set()

    for d in dirs:
        for md_file in sorted(d.rglob("SKILL.md")):
            try:
                skill = _parse_skill_md(md_file)
                if skill.name not in seen_names:
                    seen_names.add(skill.name)
                    discovered.append(skill)
                    _cache[skill.name] = skill
            except Exception:
                continue

    return discovered


def _parse_skill_md(path: Path) -> SkillDefinition:
    """Parse a ``SKILL.md`` file with YAML frontmatter + Markdown body.

    Expected format::

        ---
        name: my_skill
        display_name: My Skill
        category: custom
        description: Does something useful
        constraints:
          - Keep output concise
        tools:
          - file_reader
        tags:
          - writing
        ---

        # Instructions

        Detailed instructions as markdown body...
    """
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")

    # Find YAML frontmatter delimiters
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"No YAML frontmatter in {path}")

    end_idx = -1
    for i, line in enumerate(lines[1:], 1):
        if line.strip() == "---":
            end_idx = i
            break

    if end_idx < 0:
        raise ValueError(f"Unclosed YAML frontmatter in {path}")

    frontmatter = "\n".join(lines[1:end_idx])
    body = "\n".join(lines[end_idx + 1 :]).strip()

    data = parse_yaml(frontmatter)
    if not isinstance(data, dict) or "name" not in data:
        raise ValueError(f"Frontmatter must be a dict with 'name' key in {path}")

    data.setdefault("instructions", "")
    if body:
        # Append body as instructions
        existing = data.get("instructions", "")
        data["instructions"] = f"{existing}\n\n{body}".strip() if existing else body

    # Store source path in metadata
    meta = data.get("metadata", {})
    meta["source"] = str(path)
    data["metadata"] = meta

    return _parse_skill(data)


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


# ---------------------------------------------------------------------------
# Cross-tool orchestration: wrap_callable
# ---------------------------------------------------------------------------

# Type annotation → JSON Schema type mapping
_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _params_schema_from_callable(fn: Callable[..., Any]) -> dict[str, Any]:
    """Extract a JSON-Schema-style parameters description from *fn*'s signature.

    Uses :mod:`inspect` to read parameter names, type annotations, and
    defaults.  Parameters with no annotation default to ``"string"``.
    """
    sig = inspect.signature(fn)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue

        ann = param.annotation
        json_type = "string"
        if ann is not inspect.Parameter.empty:
            json_type = _TYPE_MAP.get(ann, "string")

        prop: dict[str, Any] = {"type": json_type}

        if param.default is not inspect.Parameter.empty:
            prop["default"] = param.default
        else:
            required.append(name)

        properties[name] = prop

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required
    return schema


def wrap_callable(
    fn: Callable[..., Any],
    name: str | None = None,
    *,
    description: str = "",
    params_schema: dict[str, Any] | None = None,
    category: str = "wrapped",
    tags: list[str] | None = None,
    register: bool = True,
) -> SkillDefinition:
    """Wrap any Python callable as a :class:`SkillDefinition`.

    This turns an arbitrary function into a skill that agents can discover
    and invoke.  The function's signature is introspected to build a
    JSON-Schema parameter description, or you can supply one explicitly.

    Parameters
    ----------
    fn
        The Python callable to wrap.
    name
        Skill name.  Defaults to ``fn.__name__``.
    description
        One-line description of what this callable does.
    params_schema
        Explicit JSON-Schema dict for the function's parameters.
        If ``None``, the schema is derived from the function's
        type annotations and defaults.
    category
        Grouping category.  Defaults to ``"wrapped"``.
    tags
        Free-form tags for filtering and discovery.
    register
        Whether to register the skill in the global registry.

    Returns
    -------
    SkillDefinition
        A skill definition wrapping *fn*.

    Examples
    --------
    ```python
    import talk_box as tb

    def word_count(text: str) -> int:
        return len(text.split())

    skill = tb.wrap_callable(word_count, description="Count words in text")
    ```
    """
    skill_name = name or fn.__name__
    schema = params_schema or _params_schema_from_callable(fn)

    # Build instructions from docstring if available
    doc = inspect.getdoc(fn) or ""
    instructions = description
    if doc:
        instructions = f"{description}\n\n{doc}".strip() if description else doc

    skill = SkillDefinition(
        name=skill_name,
        display_name=skill_name.replace("_", " ").title(),
        category=category,
        description=description or (doc.split("\n")[0] if doc else ""),
        instructions=instructions,
        constraints=[],
        tools=[skill_name],
        tags=tags or [],
        metadata={
            "wrapped_callable": fn.__qualname__,
            "params_schema": schema,
        },
    )

    if register:
        _cache[skill_name] = skill

    return skill
