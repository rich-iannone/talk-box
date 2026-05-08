from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from talk_box.conversation import Message
    from talk_box.models import ModelProfile
    from talk_box.prompt_builder import PromptBuilder, PromptSection


# ── Token estimation ───────────────────────────────────────────────────────


def estimate_tokens(text: str) -> int:
    """Estimate the token count for a string using a character-based heuristic.

    Uses the approximation of 1 token ≈ 4 characters for English text, which
    aligns with typical BPE tokenizers (GPT, Claude, Llama). For non-English
    or code-heavy text, this may undercount slightly.

    Parameters
    ----------
    text
        The text to estimate tokens for.

    Returns
    -------
    int
        Estimated token count (always at least 1 for non-empty text).

    Examples
    --------
    ```python
    import talk_box as tb

    tokens = tb.estimate_tokens("Hello, world!")
    print(tokens)  # ~4
    ```
    """
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


# ── Enums ──────────────────────────────────────────────────────────────────


class FitStrategy(Enum):
    """Strategy for fitting content into a token budget.

    Attributes
    ----------
    TRUNCATE_OLDEST
        Drop the oldest messages first, preserving the most recent context.
    TRUNCATE_MIDDLE
        Keep the first and last messages, dropping from the middle.
        Preserves the opening context and the most recent exchanges.
    TRIM_SECTIONS
        For prompts: drop lowest-priority PromptBuilder sections first.
    """

    TRUNCATE_OLDEST = "truncate_oldest"
    TRUNCATE_MIDDLE = "truncate_middle"
    TRIM_SECTIONS = "trim_sections"


# ── Result dataclasses ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class FitResult:
    """Result of fitting content into a context window budget.

    Parameters
    ----------
    messages
        The messages that fit within the budget.
    system_prompt
        The system prompt (possibly truncated).
    tokens_used
        Estimated total tokens consumed.
    token_budget
        The total token budget that was targeted.
    messages_dropped
        Number of messages that were dropped to fit.
    strategy_used
        The strategy that was applied.
    """

    messages: list[dict[str, str]]
    system_prompt: str
    tokens_used: int
    token_budget: int
    messages_dropped: int = 0
    strategy_used: FitStrategy = FitStrategy.TRUNCATE_OLDEST


@dataclass(frozen=True)
class PromptFitResult:
    """Result of fitting a PromptBuilder output into a token budget.

    Parameters
    ----------
    text
        The fitted prompt text.
    tokens_used
        Estimated tokens in the fitted text.
    token_budget
        The token budget that was targeted.
    sections_dropped
        Names/descriptions of sections that were dropped to fit.
    """

    text: str
    tokens_used: int
    token_budget: int
    sections_dropped: list[str] = field(default_factory=list)


# ── ContextWindow class ────────────────────────────────────────────────────


class ContextWindow:
    """Manages fitting content into a model's token budget.

    Combines token estimation with truncation strategies to ensure prompts
    and conversation messages stay within a model's context window.

    Parameters
    ----------
    max_tokens
        Explicit token budget. If provided, overrides any model profile lookup.
    model
        A model key (e.g., `"ollama:llama3.2:latest"`) or a `~talk_box.models.ModelProfile`
        instance. The profile's `context_window` is used as the budget.
    reserve_output
        Tokens to reserve for the model's response. Subtracted from the
        budget before fitting input content. Defaults to 1024.
    strategy
        The default strategy for fitting content. Can be overridden per call.
    token_counter
        Optional custom token counting function. Defaults to `estimate_tokens`.

    Examples
    --------
    ```python
    import talk_box as tb

    # From a model profile
    ctx = tb.ContextWindow(model="ollama:llama3.2:latest")

    # With explicit budget
    ctx = tb.ContextWindow(max_tokens=8192)

    # With custom settings
    ctx = tb.ContextWindow(
        max_tokens=32_768,
        reserve_output=4096,
        strategy="truncate_middle",
    )
    ```
    """

    def __init__(
        self,
        *,
        max_tokens: int | None = None,
        model: str | ModelProfile | None = None,
        reserve_output: int = 1024,
        strategy: str | FitStrategy = FitStrategy.TRUNCATE_OLDEST,
        token_counter: Callable[[str], int] | None = None,
    ) -> None:
        if max_tokens is None and model is None:
            raise ValueError("Provide either 'max_tokens' or 'model'.")

        self._token_counter = token_counter or estimate_tokens

        # Resolve budget from model profile if needed
        if max_tokens is not None:
            self._max_tokens = max_tokens
        else:
            self._max_tokens = self._resolve_model_budget(model)

        self._reserve_output = reserve_output
        self._input_budget = self._max_tokens - self._reserve_output

        if isinstance(strategy, str):
            strategy = FitStrategy(strategy)
        self._strategy = strategy

    @staticmethod
    def _resolve_model_budget(model: Any) -> int:
        """Resolve token budget from a model key or ModelProfile."""
        from talk_box.models import ModelProfile, get_model_profile

        if isinstance(model, ModelProfile):
            if model.context_window is None:
                raise ValueError(f"Model profile '{model.key}' has no context_window set.")
            return model.context_window

        # It's a string key like "ollama:llama3.2:latest"
        profile = get_model_profile(model)
        if profile is None:
            raise ValueError(
                f"No model profile found for '{model}'. "
                "Use max_tokens= or register the model first."
            )
        if profile.context_window is None:
            raise ValueError(f"Model profile '{model}' has no context_window set.")
        return profile.context_window

    @property
    def max_tokens(self) -> int:
        """Total context window budget in tokens."""
        return self._max_tokens

    @property
    def input_budget(self) -> int:
        """Tokens available for input (max_tokens - reserve_output)."""
        return self._input_budget

    @property
    def reserve_output(self) -> int:
        """Tokens reserved for the model's response."""
        return self._reserve_output

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the configured counter.

        Parameters
        ----------
        text
            The text to count tokens for.

        Returns
        -------
        int
            Token count.
        """
        return self._token_counter(text)

    def fits(self, text: str) -> bool:
        """Check whether text fits within the input budget.

        Parameters
        ----------
        text
            Text to check.

        Returns
        -------
        bool
            True if token count is within budget.
        """
        return self._token_counter(text) <= self._input_budget

    def overflow(self, text: str) -> int:
        """Calculate how many tokens over budget the text is.

        Parameters
        ----------
        text
            Text to check.

        Returns
        -------
        int
            Number of tokens over budget (0 if within budget).
        """
        return max(0, self._token_counter(text) - self._input_budget)

    # ── Message fitting ────────────────────────────────────────────────────

    def fit_messages(
        self,
        messages: list[dict[str, str] | Message],
        *,
        system_prompt: str = "",
        strategy: str | FitStrategy | None = None,
    ) -> FitResult:
        """Fit a list of conversation messages into the token budget.

        The system prompt is always preserved. Messages are dropped according
        to the selected strategy until they fit within the remaining budget.

        Parameters
        ----------
        messages
            List of message dicts (with `"role"` and `"content"` keys) or
            `~talk_box.conversation.Message` objects.
        system_prompt
            The system prompt text (always preserved in full).
        strategy
            Override the default strategy for this call.

        Returns
        -------
        FitResult
            The fitted messages plus metadata about what was dropped.

        Examples
        --------
        ```python
        import talk_box as tb

        ctx = tb.ContextWindow(max_tokens=4096, reserve_output=512)
        messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there! How can I help?"},
            {"role": "user", "content": "Tell me about Python."},
        ]
        result = ctx.fit_messages(messages, system_prompt="You are helpful.")
        print(f"Using {result.tokens_used}/{result.token_budget} tokens")
        print(f"Dropped {result.messages_dropped} messages")
        ```
        """
        if isinstance(strategy, str):
            strategy = FitStrategy(strategy)
        strat = strategy or self._strategy

        # Normalize messages to dicts
        normalized = self._normalize_messages(messages)

        # Budget after system prompt
        system_tokens = self._token_counter(system_prompt) if system_prompt else 0
        remaining_budget = self._input_budget - system_tokens

        if remaining_budget <= 0:
            # System prompt alone exceeds budget — return it truncated
            return FitResult(
                messages=[],
                system_prompt=system_prompt,
                tokens_used=system_tokens,
                token_budget=self._input_budget,
                messages_dropped=len(normalized),
                strategy_used=strat,
            )

        # Check if everything fits
        total_msg_tokens = sum(self._token_counter(m["content"]) for m in normalized)
        if total_msg_tokens <= remaining_budget:
            return FitResult(
                messages=normalized,
                system_prompt=system_prompt,
                tokens_used=system_tokens + total_msg_tokens,
                token_budget=self._input_budget,
                messages_dropped=0,
                strategy_used=strat,
            )

        # Apply strategy
        if strat == FitStrategy.TRUNCATE_OLDEST:
            fitted, dropped = self._truncate_oldest(normalized, remaining_budget)
        elif strat == FitStrategy.TRUNCATE_MIDDLE:
            fitted, dropped = self._truncate_middle(normalized, remaining_budget)
        else:
            # Default to truncate_oldest for message fitting
            fitted, dropped = self._truncate_oldest(normalized, remaining_budget)

        fitted_tokens = sum(self._token_counter(m["content"]) for m in fitted)
        return FitResult(
            messages=fitted,
            system_prompt=system_prompt,
            tokens_used=system_tokens + fitted_tokens,
            token_budget=self._input_budget,
            messages_dropped=dropped,
            strategy_used=strat,
        )

    def _truncate_oldest(
        self, messages: list[dict[str, str]], budget: int
    ) -> tuple[list[dict[str, str]], int]:
        """Drop oldest messages until remaining fit within budget."""
        # Work backwards from the newest message
        kept: list[dict[str, str]] = []
        tokens_used = 0

        for msg in reversed(messages):
            msg_tokens = self._token_counter(msg["content"])
            if tokens_used + msg_tokens <= budget:
                kept.append(msg)
                tokens_used += msg_tokens
            else:
                break

        kept.reverse()
        dropped = len(messages) - len(kept)
        return kept, dropped

    def _truncate_middle(
        self, messages: list[dict[str, str]], budget: int
    ) -> tuple[list[dict[str, str]], int]:
        """Keep first and last messages, dropping from the middle."""
        if len(messages) <= 2:
            return self._truncate_oldest(messages, budget)

        # Always try to keep the first message (sets context) and the last
        first_tokens = self._token_counter(messages[0]["content"])
        last_tokens = self._token_counter(messages[-1]["content"])

        # If even first + last don't fit, fall back to truncate_oldest
        if first_tokens + last_tokens > budget:
            return self._truncate_oldest(messages, budget)

        # Greedily add messages from the end (before the last) until budget
        remaining = budget - first_tokens - last_tokens
        tail_kept: list[dict[str, str]] = []

        for msg in reversed(messages[1:-1]):
            msg_tokens = self._token_counter(msg["content"])
            if remaining >= msg_tokens:
                tail_kept.append(msg)
                remaining -= msg_tokens
            else:
                break

        tail_kept.reverse()
        kept = [messages[0]] + tail_kept + [messages[-1]]
        dropped = len(messages) - len(kept)
        return kept, dropped

    # ── Prompt fitting ─────────────────────────────────────────────────────

    def fit_prompt(self, builder: PromptBuilder) -> PromptFitResult:
        """Fit a PromptBuilder's output into the token budget.

        If the full prompt fits, returns it unchanged. Otherwise, drops lowest-priority sections
        until it fits.

        Parameters
        ----------
        builder
            A `~talk_box.prompt_builder.PromptBuilder` instance.

        Returns
        -------
        PromptFitResult
            The fitted prompt text and metadata.

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
        ctx = tb.ContextWindow(max_tokens=2048)
        result = ctx.fit_prompt(builder)
        print(f"Prompt uses {result.tokens_used} tokens")
        ```
        """
        full_text = str(builder)
        full_tokens = self._token_counter(full_text)

        if full_tokens <= self._input_budget:
            return PromptFitResult(
                text=full_text,
                tokens_used=full_tokens,
                token_budget=self._input_budget,
                sections_dropped=[],
            )

        # Need to trim — rebuild with sections removed by priority (lowest first)
        return self._trim_prompt_sections(builder)

    def _trim_prompt_sections(self, builder: PromptBuilder) -> PromptFitResult:
        """Rebuild a prompt dropping lowest-priority sections until it fits."""
        # Access internal sections list (sorted by priority, lowest = most droppable)
        sections: list[PromptSection] = list(builder._sections)
        # Sort by priority descending (higher value = lower priority = drop first)
        sections_by_priority = sorted(sections, key=lambda s: s.priority.value, reverse=True)

        dropped_names: list[str] = []
        remaining_sections = set(id(s) for s in sections)

        for section in sections_by_priority:
            remaining_sections.discard(id(section))
            dropped_names.append(section.content[:50].strip().replace("\n", " "))

            # Rebuild prompt without dropped sections
            text = self._rebuild_prompt_without(builder, remaining_sections)
            tokens = self._token_counter(text)

            if tokens <= self._input_budget:
                return PromptFitResult(
                    text=text,
                    tokens_used=tokens,
                    token_budget=self._input_budget,
                    sections_dropped=dropped_names,
                )

        # Even after dropping all sections, still over budget — return what we have
        text = self._rebuild_prompt_without(builder, set())
        tokens = self._token_counter(text)
        return PromptFitResult(
            text=text,
            tokens_used=tokens,
            token_budget=self._input_budget,
            sections_dropped=dropped_names,
        )

    def _rebuild_prompt_without(self, builder: PromptBuilder, keep_section_ids: set[int]) -> str:
        """Rebuild a PromptBuilder's output excluding certain sections."""
        from talk_box._text_formatter import wrap_prompt_text

        prompt_parts: list[str] = []

        # 1. Persona (always keep)
        if builder._persona:
            prompt_parts.append(builder._persona)

        # 2. Critical constraints (always keep)
        if builder._constraints:
            prompt_parts.append("\nCRITICAL REQUIREMENTS:")
            prompt_parts.append(f"- {builder._constraints[0]}")

        # 3. Task context (always keep)
        if builder._task_context:
            prompt_parts.append(f"\nTASK: {builder._task_context}")

        # 4. Vocabulary (always keep — usually small)
        if builder._vocabulary:
            prompt_parts.append("\nDOMAIN VOCABULARY:")
            for term in builder._vocabulary:
                prompt_parts.append(f"- **{term.term}**: {term.definition}")

        # 5. Sections (filtered)
        sorted_sections = sorted(builder._sections, key=lambda s: (s.priority.value, s.order_hint))
        for section in sorted_sections:
            if id(section) in keep_section_ids:
                prompt_parts.append(f"\n{section.content}")

        # 6. Standard constraints (keep if space allows)
        if len(builder._constraints) > 1:
            prompt_parts.append("\nADDITIONAL CONSTRAINTS:")
            for constraint in builder._constraints[1:]:
                prompt_parts.append(f"- {constraint}")

        # 7. Output format
        if builder._output_format:
            prompt_parts.append("\nOUTPUT FORMAT:")
            for fmt in builder._output_format:
                prompt_parts.append(f"- {fmt}")

        # 8. Examples (droppable — skip when trimming aggressively)
        if keep_section_ids:  # Only include examples if we still have sections
            if builder._examples:
                prompt_parts.append("\nEXAMPLES:")
                for i, ex in enumerate(builder._examples, 1):
                    prompt_parts.append(f"\nExample {i}:")
                    prompt_parts.append(f"Input: {ex['input']}")
                    prompt_parts.append(f"Output: {ex['output']}")

        # 9. Final emphasis (always keep)
        if builder._final_emphasis:
            prompt_parts.append(f"\n{builder._final_emphasis}")

        raw_prompt = "\n".join(prompt_parts)
        return wrap_prompt_text(raw_prompt, width=100)

    # ── Utilities ──────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_messages(
        messages: list[dict[str, str] | Any],
    ) -> list[dict[str, str]]:
        """Normalize Message objects or dicts to a consistent dict format."""
        normalized: list[dict[str, str]] = []
        for msg in messages:
            if isinstance(msg, dict):
                normalized.append(msg)
            else:
                # Assume it's a Message dataclass with role and content
                normalized.append({"role": msg.role, "content": msg.content})
        return normalized
