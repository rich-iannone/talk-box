from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Protocol

# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------


class GuardAction(Enum):
    """The outcome of a guard check."""

    PASSED = "passed"
    BLOCKED = "blocked"
    REWRITTEN = "rewritten"


@dataclass(frozen=True)
class GuardResult:
    """The result of evaluating a guardrail against a message.

    Use the class methods `passed()`, `blocked()`, and `rewrite()` to
    construct results rather than calling the constructor directly.

    Attributes
    ----------
    action
        Whether the message passed, was blocked, or was rewritten.
    message
        The (possibly rewritten) message text. `None` when blocked.
    reason
        Human-readable explanation of why the guard acted.
    guard_name
        Name of the guard that produced this result.
    """

    action: GuardAction
    message: Optional[str] = None
    reason: Optional[str] = None
    guard_name: Optional[str] = None

    @classmethod
    def passed(cls, *, reason: str | None = None) -> GuardResult:
        """Create a passing result (message is unchanged).

        Parameters
        ----------
        reason
            Optional explanation (e.g., for logging/telemetry).
        """
        return cls(action=GuardAction.PASSED, reason=reason)

    @classmethod
    def blocked(cls, reason: str) -> GuardResult:
        """Create a blocked result (message is rejected).

        Parameters
        ----------
        reason
            Explanation of why the message was blocked.
        """
        return cls(action=GuardAction.BLOCKED, reason=reason)

    @classmethod
    def rewrite(cls, new_text: str, *, reason: str | None = None) -> GuardResult:
        """Create a rewrite result (message is modified and continues).

        Parameters
        ----------
        new_text
            The rewritten message text that replaces the original.
        reason
            Optional explanation of what was changed.
        """
        return cls(action=GuardAction.REWRITTEN, message=new_text, reason=reason)


class GuardPhase(Enum):
    """When the guard runs in the message lifecycle."""

    INPUT = "input"
    OUTPUT = "output"
    BOTH = "both"


# ---------------------------------------------------------------------------
# Guard function protocol and decorator
# ---------------------------------------------------------------------------


class GuardFunc(Protocol):
    """Protocol for guard functions."""

    def __call__(self, text: str) -> GuardResult: ...


@dataclass
class Guard:
    """A configured guardrail ready to be attached to a ChatBot.

    Attributes
    ----------
    name
        Human-readable guard name (used in telemetry and error messages).
    func
        The guard function: takes a string, returns a `GuardResult`.
    phase
        When this guard runs (input, output, or both).
    """

    name: str
    func: Callable[[str], GuardResult]
    phase: GuardPhase = GuardPhase.BOTH
    states: list[str] | None = None

    def applies_to_state(self, current_state: str | None) -> bool:
        """Check whether this guard should run in the given pathway state.

        Parameters
        ----------
        current_state
            The current pathway state name, or ``None`` if no pathway is active.

        Returns
        -------
        bool
            ``True`` if the guard should run.
        """
        if self.states is None:
            return True
        if current_state is None:
            return False
        return current_state in self.states

    def check(self, text: str) -> GuardResult:
        """Run the guard against the given text.

        Parameters
        ----------
        text
            Message text to validate.

        Returns
        -------
        GuardResult
            The outcome with action, optional rewrite, and reason.
        """
        result = self.func(text)
        # Attach guard name to result for telemetry
        if result.guard_name is None:
            result = GuardResult(
                action=result.action,
                message=result.message,
                reason=result.reason,
                guard_name=self.name,
            )
        return result


def guardrail(
    func: Callable[[str], GuardResult] | None = None,
    *,
    name: str | None = None,
    phase: GuardPhase = GuardPhase.BOTH,
    states: list[str] | None = None,
) -> Guard | Callable[[Callable[[str], GuardResult]], Guard]:
    """Decorator that turns a function into a composable guardrail.

    Can be used as a bare decorator or with keyword arguments.

    Parameters
    ----------
    func
        The guard function (when used as bare `@guardrail`).
    name
        Override the guard's display name (defaults to the function name).
    phase
        When to run: `"input"`, `"output"`, or `"both"` (default).
    states
        Optional list of pathway state names where this guard is active.
        When ``None`` (the default), the guard runs in all states.
        When set, the guard is skipped unless the current pathway state
        matches one of the listed names.

    Returns
    -------
    Guard
        A configured guard instance.

    Examples
    --------
    Bare decorator:

    ```python
    @tb.guardrail
    def no_profanity(text: str) -> tb.GuardResult:
        if any(w in text.lower() for w in BAD_WORDS):
            return tb.GuardResult.blocked("Profanity detected")
        return tb.GuardResult.passed()
    ```

    With options:

    ```python
    @tb.guardrail(name="PII Filter", phase=GuardPhase.INPUT)
    def strip_emails(text: str) -> tb.GuardResult:
        cleaned = re.sub(r'\\S+@\\S+', '[EMAIL]', text)
        if cleaned != text:
            return tb.GuardResult.rewrite(cleaned, reason="Stripped email addresses")
        return tb.GuardResult.passed()
    ```
    """
    if func is not None:
        # Bare @guardrail usage
        guard_name = name or func.__name__
        return Guard(name=guard_name, func=func, phase=phase, states=states)

    # @guardrail(...) with arguments
    def decorator(fn: Callable[[str], GuardResult]) -> Guard:
        guard_name = name or fn.__name__
        return Guard(name=guard_name, func=fn, phase=phase, states=states)

    return decorator


# ---------------------------------------------------------------------------
# Guard pipeline
# ---------------------------------------------------------------------------


@dataclass
class GuardActivation:
    """Record of a single guard activation (for telemetry).

    Attributes
    ----------
    guard_name
        Which guard fired.
    phase
        Whether it was checking input or output.
    action
        What the guard did (passed, blocked, rewritten).
    reason
        Human-readable reason (if any).
    original_text
        The text before the guard acted (truncated for privacy).
    """

    guard_name: str
    phase: GuardPhase
    action: GuardAction
    reason: Optional[str] = None
    original_text: Optional[str] = None


@dataclass
class GuardPipelineResult:
    """Result of running a message through the full guard pipeline.

    Attributes
    ----------
    text
        The final message text (original, rewritten, or None if blocked).
    blocked
        Whether the message was blocked by any guard.
    block_reason
        Reason for blocking (if blocked).
    activations
        Log of all guard activations in order.
    """

    text: Optional[str]
    blocked: bool = False
    block_reason: Optional[str] = None
    activations: list[GuardActivation] = field(default_factory=list)


class GuardPipeline:
    """Ordered pipeline of guards that processes messages sequentially.

    Guards run in the order they were added. If a guard blocks, the pipeline
    stops immediately. If a guard rewrites, subsequent guards see the rewritten
    text.
    """

    def __init__(self) -> None:
        self._guards: list[Guard] = []
        self._stats: dict[str, dict[str, int]] = {}

    @property
    def guards(self) -> list[Guard]:
        """The ordered list of guards in this pipeline."""
        return self._guards.copy()

    def add(self, guard: Guard) -> None:
        """Add a guard to the end of the pipeline.

        Parameters
        ----------
        guard
            A configured `Guard` instance.
        """
        self._guards.append(guard)
        if guard.name not in self._stats:
            self._stats[guard.name] = {"passed": 0, "blocked": 0, "rewritten": 0}

    def run(
        self,
        text: str,
        phase: GuardPhase,
        *,
        current_state: str | None = None,
    ) -> GuardPipelineResult:
        """Run all applicable guards against the text.

        Parameters
        ----------
        text
            The message text to validate.
        phase
            Current phase (`INPUT` or `OUTPUT`). Guards with matching
            phase or `BOTH` will run.
        current_state
            The current pathway state name.  Guards with a ``states``
            restriction are skipped when the state doesn't match.

        Returns
        -------
        GuardPipelineResult
            The final text (possibly rewritten) and activation log.
        """
        activations: list[GuardActivation] = []
        current_text = text

        for guard in self._guards:
            # Skip guards not applicable to this phase
            if guard.phase != GuardPhase.BOTH and guard.phase != phase:
                continue

            # Skip guards not applicable to the current pathway state
            if not guard.applies_to_state(current_state):
                continue

            result = guard.check(current_text)

            # Record activation
            activation = GuardActivation(
                guard_name=guard.name,
                phase=phase,
                action=result.action,
                reason=result.reason,
                original_text=current_text[:200] if result.action != GuardAction.PASSED else None,
            )
            activations.append(activation)

            # Update stats
            self._stats[guard.name][result.action.value] += 1

            if result.action == GuardAction.BLOCKED:
                return GuardPipelineResult(
                    text=None,
                    blocked=True,
                    block_reason=result.reason,
                    activations=activations,
                )
            elif result.action == GuardAction.REWRITTEN:
                current_text = result.message  # type: ignore[assignment]

        return GuardPipelineResult(text=current_text, activations=activations)

    def stats(self) -> dict[str, dict[str, int]]:
        """Get activation statistics for all guards.

        Returns
        -------
        dict[str, dict[str, int]]
            Mapping of guard name to counts of passed/blocked/rewritten.
        """
        return {k: v.copy() for k, v in self._stats.items()}

    def reset_stats(self) -> None:
        """Reset all activation counters to zero."""
        for counts in self._stats.values():
            counts["passed"] = 0
            counts["blocked"] = 0
            counts["rewritten"] = 0


# ---------------------------------------------------------------------------
# Built-in guards
# ---------------------------------------------------------------------------


def no_pii(
    *,
    patterns: list[str] | None = None,
    action: str = "rewrite",
) -> Guard:
    """Guard that detects and handles Personally Identifiable Information.

    Detects email addresses, phone numbers, SSNs, and credit card numbers
    by default. Can be extended with custom regex patterns.

    Parameters
    ----------
    patterns
        Additional regex patterns to detect. Each should match PII strings.
    action
        What to do when PII is found: `"rewrite"` (replace with
        placeholders) or `"block"` (reject the message entirely).

    Returns
    -------
    Guard
        A configured PII detection guard.

    Examples
    --------
    ```python
    bot = tb.ChatBot().guardrail(tb.no_pii())
    bot = tb.ChatBot().guardrail(tb.no_pii(action="block"))
    ```
    """
    # Standard PII patterns
    pii_patterns: list[tuple[str, str, str]] = [
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]", "email address"),
        (r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[PHONE]", "phone number"),
        (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]", "SSN"),
        (r"\b(?:\d{4}[-\s]?){3}\d{4}\b", "[CARD]", "credit card number"),
    ]

    # Add custom patterns
    if patterns:
        for pat in patterns:
            pii_patterns.append((pat, "[PII]", "custom PII pattern"))

    def _check(text: str) -> GuardResult:
        found_types: list[str] = []
        cleaned = text

        for pattern, replacement, pii_type in pii_patterns:
            if re.search(pattern, cleaned):
                found_types.append(pii_type)
                cleaned = re.sub(pattern, replacement, cleaned)

        if not found_types:
            return GuardResult.passed()

        reason = f"Detected PII: {', '.join(found_types)}"

        if action == "block":
            return GuardResult.blocked(reason)

        return GuardResult.rewrite(cleaned, reason=reason)

    return Guard(name="no_pii", func=_check, phase=GuardPhase.BOTH)


def max_response_length(max_chars: int) -> Guard:
    """Guard that enforces a maximum response length.

    Truncates responses that exceed the character limit, appending an
    ellipsis to indicate truncation.

    Parameters
    ----------
    max_chars
        Maximum allowed characters in the response.

    Returns
    -------
    Guard
        A configured length enforcement guard (output-only).

    Examples
    --------
    ```python
    bot = tb.ChatBot().guardrail(tb.max_response_length(500))
    ```
    """

    def _check(text: str) -> GuardResult:
        if len(text) <= max_chars:
            return GuardResult.passed()
        truncated = text[:max_chars].rsplit(" ", 1)[0] + "..."
        return GuardResult.rewrite(
            truncated,
            reason=f"Response exceeded {max_chars} chars (was {len(text)})",
        )

    return Guard(name="max_response_length", func=_check, phase=GuardPhase.OUTPUT)


def tone_check(expected_tone: str, *, indicators: dict[str, list[str]] | None = None) -> Guard:
    """Guard that flags responses not matching the expected tone.

    Uses keyword-based heuristics to detect tone mismatches. For production
    use with high accuracy, consider extending with an LLM-based judge.

    Parameters
    ----------
    expected_tone
        The desired tone (e.g., `"professional"`, `"casual"`, `"formal"`).
    indicators
        Optional mapping of tone names to indicator words/phrases. If not
        provided, uses built-in defaults for common tones.

    Returns
    -------
    Guard
        A configured tone-checking guard (output-only).

    Examples
    --------
    ```python
    bot = tb.ChatBot().guardrail(tb.tone_check("professional"))
    ```
    """
    default_indicators: dict[str, list[str]] = {
        "professional": [
            "lol",
            "lmao",
            "omg",
            "wtf",
            "bruh",
            "dude",
            "gonna",
            "wanna",
            "ain't",
            "nah",
            "yeah",
            "haha",
            "😂",
            "🤣",
            "💀",
        ],
        "formal": [
            "hey",
            "hi there",
            "what's up",
            "cool",
            "awesome",
            "gonna",
            "wanna",
            "gotta",
            "yeah",
            "nope",
            "yep",
        ],
        "casual": [
            "pursuant to",
            "hereinafter",
            "aforementioned",
            "notwithstanding",
            "whereby",
            "thereof",
            "heretofore",
        ],
    }

    violation_words = (indicators or default_indicators).get(expected_tone, [])

    def _check(text: str) -> GuardResult:
        if not violation_words:
            return GuardResult.passed()

        text_lower = text.lower()
        violations = [w for w in violation_words if w.lower() in text_lower]

        if not violations:
            return GuardResult.passed()

        return GuardResult.blocked(
            f"Tone mismatch: expected '{expected_tone}' but found "
            f"indicators: {', '.join(violations[:5])}"
        )

    return Guard(name="tone_check", func=_check, phase=GuardPhase.OUTPUT)


def disclaimer_required(disclaimer_text: str, *, position: str = "end") -> Guard:
    """Guard that ensures a required disclaimer appears in the response.

    If the disclaimer is missing, it is appended (or prepended) automatically.

    Parameters
    ----------
    disclaimer_text
        The exact disclaimer text that must appear.
    position
        Where to add the disclaimer if missing: `"end"` (default) or `"start"`.

    Returns
    -------
    Guard
        A configured disclaimer enforcement guard (output-only).

    Examples
    --------
    ```python
    bot = tb.ChatBot().guardrail(
        tb.disclaimer_required("This is not financial advice.")
    )
    ```
    """

    def _check(text: str) -> GuardResult:
        if disclaimer_text in text:
            return GuardResult.passed()

        if position == "start":
            new_text = f"{disclaimer_text}\n\n{text}"
        else:
            new_text = f"{text}\n\n{disclaimer_text}"

        return GuardResult.rewrite(
            new_text,
            reason="Added required disclaimer",
        )

    return Guard(name="disclaimer_required", func=_check, phase=GuardPhase.OUTPUT)


def must_cite_sources(*, min_citations: int = 1) -> Guard:
    """Guard that requires responses to contain source citations.

    Detects common citation patterns: URLs, bracketed references, footnotes,
    and `Source:` labels.

    Parameters
    ----------
    min_citations
        Minimum number of citation patterns required. Defaults to 1.

    Returns
    -------
    Guard
        A configured citation enforcement guard (output-only).

    Examples
    --------
    ```python
    bot = tb.ChatBot().guardrail(tb.must_cite_sources())
    ```
    """
    citation_patterns = [
        r"https?://\S+",  # URLs
        r"\[\d+\]",  # [1], [2], etc.
        r"\[\^.+?\]",  # [^footnote]
        r"Source:\s*\S+",  # Source: ...
        r"Reference:\s*\S+",  # Reference: ...
        r"\(.*?\d{4}\)",  # (Author, 2024) style
    ]

    def _check(text: str) -> GuardResult:
        citation_count = 0
        for pattern in citation_patterns:
            citation_count += len(re.findall(pattern, text))

        if citation_count >= min_citations:
            return GuardResult.passed()

        return GuardResult.blocked(
            f"Response must contain at least {min_citations} citation(s). Found {citation_count}."
        )

    return Guard(name="must_cite_sources", func=_check, phase=GuardPhase.OUTPUT)


def max_input_length(max_chars: int) -> Guard:
    """Guard that rejects user messages exceeding a character limit.

    Parameters
    ----------
    max_chars
        Maximum allowed characters in the input message.

    Returns
    -------
    Guard
        A configured input length guard (input-only).

    Examples
    --------
    ```python
    bot = tb.ChatBot().guardrail(tb.max_input_length(10000))
    ```
    """

    def _check(text: str) -> GuardResult:
        if len(text) <= max_chars:
            return GuardResult.passed()
        return GuardResult.blocked(f"Input too long: {len(text)} chars (max {max_chars})")

    return Guard(name="max_input_length", func=_check, phase=GuardPhase.INPUT)


def keyword_block(
    keywords: list[str],
    *,
    case_sensitive: bool = False,
    phase: GuardPhase = GuardPhase.BOTH,
) -> Guard:
    """Guard that blocks messages containing any of the specified keywords.

    Parameters
    ----------
    keywords
        List of words or phrases to block.
    case_sensitive
        Whether matching should be case-sensitive.
    phase
        When to apply: input, output, or both.

    Returns
    -------
    Guard
        A configured keyword blocking guard.

    Examples
    --------
    ```python
    bot = tb.ChatBot().guardrail(
        tb.keyword_block(["password", "secret key", "api_key"])
    )
    ```
    """

    def _check(text: str) -> GuardResult:
        check_text = text if case_sensitive else text.lower()
        for kw in keywords:
            check_kw = kw if case_sensitive else kw.lower()
            if check_kw in check_text:
                return GuardResult.blocked(f"Blocked keyword detected: '{kw}'")
        return GuardResult.passed()

    return Guard(name="keyword_block", func=_check, phase=phase)


# ---------------------------------------------------------------------------
# Guard resolution from declarative specs
# ---------------------------------------------------------------------------

# Registry of built-in guard factory functions
GUARD_FACTORIES: dict[str, Any] = {
    "no_pii": no_pii,
    "max_response_length": max_response_length,
    "tone_check": tone_check,
    "disclaimer_required": disclaimer_required,
    "must_cite_sources": must_cite_sources,
    "max_input_length": max_input_length,
    "keyword_block": keyword_block,
}


def resolve_guards(specs: list[str | dict[str, Any]]) -> list[Guard]:
    """Create Guard instances from declarative guard specifications.

    Guard specs are either a plain string (guard name, no arguments) or a
    dict with a single key (guard name) whose value is a dict of keyword
    arguments to pass to the factory function.

    Parameters
    ----------
    specs
        List of guard specifications. Each is either a string naming a
        built-in guard (e.g., `"no_pii"`) or a single-key dict mapping
        the guard name to its keyword arguments (e.g.,
        `{"disclaimer_required": {"disclaimer_text": "..."}}`).

    Returns
    -------
    list[Guard]
        Instantiated guards ready to add to a pipeline.

    Raises
    ------
    ValueError
        If a guard name is not found in the registry.
    TypeError
        If a spec has an unsupported type.
    """
    guards: list[Guard] = []

    for spec in specs:
        if isinstance(spec, str):
            factory = GUARD_FACTORIES.get(spec)
            if factory is None:
                raise ValueError(f"Unknown guard '{spec}'. Available: {sorted(GUARD_FACTORIES)}")
            guards.append(factory())
        elif isinstance(spec, dict):
            if len(spec) != 1:
                raise ValueError(
                    f"Guard spec dict must have exactly one key, got {len(spec)}: {spec}"
                )
            name, kwargs = next(iter(spec.items()))
            factory = GUARD_FACTORIES.get(name)
            if factory is None:
                raise ValueError(f"Unknown guard '{name}'. Available: {sorted(GUARD_FACTORIES)}")
            guards.append(factory(**kwargs))
        else:
            raise TypeError(f"Guard spec must be a str or dict, got {type(spec).__name__}")

    return guards
