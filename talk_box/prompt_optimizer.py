"""Prompt size optimizer for reducing token usage without losing semantic meaning.

Applies text-level compression techniques to prompts — stripping formatting,
condensing whitespace, compacting structure, and shortening boilerplate — so
they fit within smaller context windows (especially local models).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from talk_box.prompt_builder import PromptBuilder


# ── Optimization levels ────────────────────────────────────────────────────


class OptimizationLevel(Enum):
    """How aggressively to compress prompts.

    Attributes
    ----------
    LIGHT
        Remove unnecessary formatting only. Safe for all models.
    MODERATE
        Condense whitespace, compact examples, shorten boilerplate.
    AGGRESSIVE
        Maximum compression: strip all formatting, merge sections,
        truncate long content.
    """

    LIGHT = "light"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


# ── Result ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class OptimizeResult:
    """Result of prompt optimization.

    Parameters
    ----------
    text
        The optimized prompt text.
    original_tokens
        Estimated tokens before optimization.
    optimized_tokens
        Estimated tokens after optimization.
    reduction_pct
        Percentage of tokens saved.
    level
        The optimization level that was applied.
    """

    text: str
    original_tokens: int
    optimized_tokens: int
    reduction_pct: float
    level: OptimizationLevel


# ── Core optimizer ─────────────────────────────────────────────────────────


def optimize_prompt(
    prompt: str | PromptBuilder,
    *,
    level: str | OptimizationLevel = OptimizationLevel.MODERATE,
) -> OptimizeResult:
    """Optimize a prompt for reduced token usage.

    Applies text-level compression techniques that preserve semantic meaning
    while reducing token count. Useful when targeting models with small
    context windows (8K–32K).

    Parameters
    ----------
    prompt
        A prompt string or a :class:`~talk_box.prompt_builder.PromptBuilder`
        instance. If a PromptBuilder, its built output is used.
    level
        How aggressively to compress. ``"light"`` is formatting-only,
        ``"moderate"`` (default) condenses structure, ``"aggressive"``
        maximizes compression.

    Returns
    -------
    OptimizeResult
        The optimized text with before/after token counts.

    Examples
    --------
    ```python
    import talk_box as tb

    builder = (
        tb.PromptBuilder()
        .persona("analyst", "data science")
        .task_context("Analyze sales data")
        .constraint("Be concise")
        .example("Q: Revenue?", "A: $1.2M")
    )
    result = tb.optimize_prompt(builder)
    print(f"Saved {result.reduction_pct:.0f}% tokens")
    print(result.text)
    ```
    """
    from talk_box.context_window import estimate_tokens

    if isinstance(level, str):
        level = OptimizationLevel(level)

    # Get text from PromptBuilder or use directly
    text = str(prompt)
    original_tokens = estimate_tokens(text)

    # Apply optimizations in order
    if level == OptimizationLevel.LIGHT:
        text = _strip_markdown(text)
        text = _normalize_whitespace(text)
    elif level == OptimizationLevel.MODERATE:
        text = _strip_markdown(text)
        text = _compact_persona(text)
        text = _compact_section_headers(text)
        text = _compact_examples(text)
        text = _compact_constraints(text)
        text = _normalize_whitespace(text)
    elif level == OptimizationLevel.AGGRESSIVE:
        text = _strip_markdown(text)
        text = _compact_persona(text)
        text = _compact_section_headers(text)
        text = _compact_examples(text)
        text = _compact_constraints(text)
        text = _compact_vocabulary(text)
        text = _truncate_long_lines(text)
        text = _collapse_blank_lines(text)
        text = _normalize_whitespace(text)

    optimized_tokens = estimate_tokens(text)
    saved = original_tokens - optimized_tokens
    pct = (saved / original_tokens * 100) if original_tokens > 0 else 0.0

    return OptimizeResult(
        text=text,
        original_tokens=original_tokens,
        optimized_tokens=optimized_tokens,
        reduction_pct=round(pct, 1),
        level=level,
    )


# ── Individual optimization passes ────────────────────────────────────────


def _strip_markdown(text: str) -> str:
    """Remove markdown formatting that models don't need.

    Strips bold (**text**), italic (*text*), and inline code (`text`)
    markers while preserving the content.
    """
    # Bold: **text** or __text__
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    # Italic: *text* or _text_ (but not inside words like snake_case)
    text = re.sub(r"(?<!\w)\*([^*\n]+?)\*(?!\w)", r"\1", text)
    # Inline code: `text`
    text = re.sub(r"`([^`\n]+?)`", r"\1", text)
    return text


def _compact_persona(text: str) -> str:
    """Shorten the verbose persona format.

    "You are a X with expertise in Y." → "Role: X. Expertise: Y."
    """
    text = re.sub(
        r"You are an? (.+?) with expertise in (.+?)\.",
        r"Role: \1. Expertise: \2.",
        text,
    )
    # Simpler form without expertise
    text = re.sub(
        r"You are an? (.+?)\.\n",
        r"Role: \1.\n",
        text,
    )
    return text


def _compact_section_headers(text: str) -> str:
    """Remove decorative section header formatting.

    "CRITICAL REQUIREMENTS:" stays (semantic), but leading blank lines
    before headers are collapsed.
    """
    # Remove blank line before section headers (ALL-CAPS followed by colon)
    text = re.sub(r"\n\n([A-Z][A-Z _]+:)", r"\n\1", text)
    return text


def _compact_examples(text: str) -> str:
    """Condense example blocks.

    Merges "Example N:\\nInput: X\\nOutput: Y" into a single-line format:
    "Example N: X → Y"
    """
    text = re.sub(
        r"Example (\d+):\n\s*Input:\s*(.+?)\n\s*Output:\s*(.+?)(?=\n|$)",
        r"Ex\1: \2 → \3",
        text,
    )
    # Remove the "EXAMPLES:" header if examples are now compact
    text = re.sub(r"\n?EXAMPLES:\n+", r"\n", text)
    return text.lstrip("\n")


def _compact_constraints(text: str) -> str:
    """Merge constraint lists into comma-separated form when short enough.

    If all constraints are short (< 60 chars), merge them into one line.
    Otherwise, keep the list format but strip the header.
    """
    # Find constraint blocks
    for header in ("CRITICAL REQUIREMENTS:", "ADDITIONAL CONSTRAINTS:"):
        pattern = re.compile(rf"{re.escape(header)}\n((?:- .+\n?)+)", re.MULTILINE)
        match = pattern.search(text)
        if match:
            items = re.findall(r"- (.+)", match.group(1))
            if items and all(len(item) < 60 for item in items):
                merged = "; ".join(items)
                replacement = f"{header} {merged}"
                text = text[: match.start()] + replacement + text[match.end() :]
    return text


def _compact_vocabulary(text: str) -> str:
    """Truncate long vocabulary definitions.

    Keeps the term and first sentence of the definition only.
    """

    def _truncate_def(m: re.Match) -> str:
        term = m.group(1)
        definition = m.group(2)
        # Keep first sentence only
        first_sentence = re.split(r"(?<=[.!?])\s", definition, maxsplit=1)[0]
        if len(first_sentence) < len(definition):
            return f"- {term}: {first_sentence}"
        return m.group(0)

    text = re.sub(r"- ([^:]+): (.+)", _truncate_def, text)
    return text


def _truncate_long_lines(text: str, max_chars: int = 200) -> str:
    """Truncate individual lines that exceed max_chars.

    Lines that are very long are truncated at a word boundary.
    Code blocks and section headers are preserved.
    """
    lines = text.split("\n")
    result = []
    for line in lines:
        if len(line) > max_chars and not line.startswith(("```", "#", "TASK:", "Role:")):
            # Truncate at word boundary
            truncated = line[:max_chars].rsplit(" ", 1)[0] + "..."
            result.append(truncated)
        else:
            result.append(line)
    return "\n".join(result)


def _collapse_blank_lines(text: str) -> str:
    """Collapse multiple consecutive blank lines into a single one."""
    return re.sub(r"\n{3,}", "\n\n", text)


def _normalize_whitespace(text: str) -> str:
    """Clean up trailing whitespace and normalize line endings."""
    lines = [line.rstrip() for line in text.split("\n")]
    # Remove leading/trailing blank lines
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)
