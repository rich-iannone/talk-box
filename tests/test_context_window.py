import pytest

from talk_box.context_window import (
    ContextWindow,
    FitResult,
    FitStrategy,
    PromptFitResult,
    estimate_tokens,
)


# ── Token estimation ───────────────────────────────────────────────────────


class TestEstimateTokens:
    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_short_text(self):
        # "Hello" = 5 chars → ceil(5/4) = 2 tokens
        assert estimate_tokens("Hello") == 2

    def test_longer_text(self):
        # 100 chars → 25 tokens
        text = "a" * 100
        assert estimate_tokens(text) == 25

    def test_minimum_one_token(self):
        # Single char → ceil(1/4) = 1
        assert estimate_tokens("x") == 1

    def test_realistic_sentence(self):
        text = "The quick brown fox jumps over the lazy dog."
        tokens = estimate_tokens(text)
        # 44 chars → ceil(44/4) = 11
        assert tokens == 11


# ── FitStrategy enum ───────────────────────────────────────────────────────


class TestFitStrategy:
    def test_values(self):
        assert FitStrategy.TRUNCATE_OLDEST.value == "truncate_oldest"
        assert FitStrategy.TRUNCATE_MIDDLE.value == "truncate_middle"
        assert FitStrategy.TRIM_SECTIONS.value == "trim_sections"

    def test_from_string(self):
        assert FitStrategy("truncate_oldest") == FitStrategy.TRUNCATE_OLDEST
        assert FitStrategy("truncate_middle") == FitStrategy.TRUNCATE_MIDDLE


# ── ContextWindow initialization ──────────────────────────────────────────


class TestContextWindowInit:
    def test_explicit_max_tokens(self):
        ctx = ContextWindow(max_tokens=8192)
        assert ctx.max_tokens == 8192
        assert ctx.reserve_output == 1024
        assert ctx.input_budget == 8192 - 1024

    def test_custom_reserve(self):
        ctx = ContextWindow(max_tokens=4096, reserve_output=512)
        assert ctx.input_budget == 4096 - 512

    def test_no_args_raises(self):
        with pytest.raises(ValueError, match="Provide either"):
            ContextWindow()

    def test_from_model_profile(self):
        from talk_box.models import ModelProfile

        profile = ModelProfile(provider="test", model="test-model", context_window=16_384)
        ctx = ContextWindow(model=profile)
        assert ctx.max_tokens == 16_384

    def test_from_model_key(self):
        # Use a known built-in profile
        ctx = ContextWindow(model="anthropic:claude-sonnet-4-6")
        assert ctx.max_tokens == 200_000

    def test_unknown_model_key_raises(self):
        with pytest.raises(ValueError, match="No model profile found"):
            ContextWindow(model="nonexistent:model-xyz")

    def test_model_without_context_window_raises(self):
        from talk_box.models import ModelProfile

        profile = ModelProfile(provider="test", model="no-ctx", context_window=None)
        with pytest.raises(ValueError, match="no context_window"):
            ContextWindow(model=profile)

    def test_strategy_from_string(self):
        ctx = ContextWindow(max_tokens=8192, strategy="truncate_middle")
        assert ctx._strategy == FitStrategy.TRUNCATE_MIDDLE

    def test_custom_token_counter(self):
        # Custom counter: 1 token per word
        def counter(text):
            return len(text.split())

        ctx = ContextWindow(max_tokens=100, token_counter=counter)
        assert ctx.count_tokens("one two three") == 3


# ── Helper methods ─────────────────────────────────────────────────────────


class TestContextWindowHelpers:
    def test_count_tokens(self):
        ctx = ContextWindow(max_tokens=1000)
        assert ctx.count_tokens("Hello world") == 3  # 11 chars → ceil(11/4)

    def test_fits_within_budget(self):
        ctx = ContextWindow(max_tokens=100, reserve_output=0)
        short_text = "a" * 100  # 25 tokens
        assert ctx.fits(short_text) is True

    def test_fits_over_budget(self):
        ctx = ContextWindow(max_tokens=10, reserve_output=0)
        long_text = "a" * 100  # 25 tokens > 10 budget
        assert ctx.fits(long_text) is False

    def test_overflow_zero_when_fits(self):
        ctx = ContextWindow(max_tokens=1000, reserve_output=0)
        assert ctx.overflow("hello") == 0

    def test_overflow_positive_when_exceeds(self):
        ctx = ContextWindow(max_tokens=10, reserve_output=0)
        text = "a" * 100  # 25 tokens
        assert ctx.overflow(text) == 15  # 25 - 10


# ── Message fitting: truncate_oldest ──────────────────────────────────────


class TestFitMessagesTruncateOldest:
    def _make_messages(self, n: int, content_len: int = 40) -> list[dict[str, str]]:
        """Create n messages with content_len chars each (→ content_len/4 tokens)."""
        return [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"{'x' * content_len}"}
            for i in range(n)
        ]

    def test_all_fit(self):
        ctx = ContextWindow(max_tokens=10000, reserve_output=0)
        messages = self._make_messages(5, content_len=40)  # 5 * 10 = 50 tokens
        result = ctx.fit_messages(messages)

        assert result.messages_dropped == 0
        assert len(result.messages) == 5
        assert result.tokens_used == 50

    def test_drops_oldest(self):
        # Budget: 30 tokens. Each message = 10 tokens. 5 messages = 50 tokens.
        # Should keep last 3 messages.
        ctx = ContextWindow(max_tokens=30, reserve_output=0)
        messages = [
            {"role": "user", "content": f"msg{i}" + "x" * 36}  # ~10 tokens each
            for i in range(5)
        ]
        result = ctx.fit_messages(messages)

        assert result.messages_dropped == 2
        assert len(result.messages) == 3
        # Kept messages should be the last 3
        assert "msg2" in result.messages[0]["content"]
        assert "msg3" in result.messages[1]["content"]
        assert "msg4" in result.messages[2]["content"]

    def test_system_prompt_reserves_space(self):
        # Budget: 50 tokens total input. System prompt = 20 tokens (80 chars).
        # Remaining: 30 tokens for messages.
        ctx = ContextWindow(max_tokens=50, reserve_output=0)
        messages = self._make_messages(5, content_len=40)  # 5 * 10 = 50 tokens
        system = "s" * 80  # 20 tokens

        result = ctx.fit_messages(messages, system_prompt=system)
        assert result.messages_dropped == 2
        assert len(result.messages) == 3

    def test_system_prompt_exceeds_budget(self):
        ctx = ContextWindow(max_tokens=10, reserve_output=0)
        messages = self._make_messages(3)
        system = "s" * 200  # 50 tokens > 10 budget

        result = ctx.fit_messages(messages, system_prompt=system)
        assert result.messages == []
        assert result.messages_dropped == 3

    def test_empty_messages(self):
        ctx = ContextWindow(max_tokens=1000, reserve_output=0)
        result = ctx.fit_messages([], system_prompt="Hello")
        assert result.messages == []
        assert result.messages_dropped == 0

    def test_returns_fit_result_type(self):
        ctx = ContextWindow(max_tokens=1000)
        result = ctx.fit_messages([{"role": "user", "content": "hi"}])
        assert isinstance(result, FitResult)
        assert result.strategy_used == FitStrategy.TRUNCATE_OLDEST


# ── Message fitting: truncate_middle ──────────────────────────────────────


class TestFitMessagesTruncateMiddle:
    def test_keeps_first_and_last(self):
        # Budget allows 3 messages (30 tokens). Have 5 messages.
        # Should keep first, last, and fill from the end.
        ctx = ContextWindow(max_tokens=30, reserve_output=0)
        messages = [
            {"role": "user", "content": f"msg{i}" + "x" * 36}  # ~10 tokens each
            for i in range(5)
        ]
        result = ctx.fit_messages(messages, strategy="truncate_middle")

        assert result.messages_dropped == 2
        # First message preserved
        assert "msg0" in result.messages[0]["content"]
        # Last message preserved
        assert "msg4" in result.messages[-1]["content"]

    def test_two_messages_kept(self):
        # Budget allows only 2 messages. Keep first and last.
        ctx = ContextWindow(max_tokens=20, reserve_output=0)
        messages = [{"role": "user", "content": f"msg{i}" + "x" * 36} for i in range(5)]
        result = ctx.fit_messages(messages, strategy="truncate_middle")

        assert "msg0" in result.messages[0]["content"]
        assert "msg4" in result.messages[-1]["content"]
        assert result.messages_dropped == 3

    def test_all_fit_no_drop(self):
        ctx = ContextWindow(max_tokens=10000, reserve_output=0)
        messages = [{"role": "user", "content": "short msg"} for _ in range(3)]
        result = ctx.fit_messages(messages, strategy="truncate_middle")
        assert result.messages_dropped == 0
        assert len(result.messages) == 3

    def test_fallback_when_first_last_exceed(self):
        # Each message is 10 tokens but budget is only 5
        ctx = ContextWindow(max_tokens=5, reserve_output=0)
        messages = [
            {"role": "user", "content": "x" * 40},  # 10 tokens
            {"role": "user", "content": "y" * 40},  # 10 tokens
            {"role": "user", "content": "z" * 40},  # 10 tokens
        ]
        result = ctx.fit_messages(messages, strategy="truncate_middle")
        # Falls back to truncate_oldest, can't keep any
        assert result.messages_dropped == 3


# ── Prompt fitting ─────────────────────────────────────────────────────────


class TestFitPrompt:
    def test_prompt_fits_unchanged(self):
        from talk_box.prompt_builder import PromptBuilder

        builder = PromptBuilder().persona("analyst").task_context("Analyze data")
        ctx = ContextWindow(max_tokens=10000, reserve_output=0)

        result = ctx.fit_prompt(builder)
        assert isinstance(result, PromptFitResult)
        assert result.sections_dropped == []
        assert result.text == str(builder)

    def test_drops_sections_when_over_budget(self):
        from talk_box.prompt_builder import PromptBuilder, Priority

        builder = (
            PromptBuilder()
            .persona("analyst")
            .task_context("Analyze data")
            .structured_section("BACKGROUND", "A" * 400, priority=Priority.LOW)
            .structured_section("DETAILS", "B" * 400, priority=Priority.MEDIUM)
        )

        # Full prompt would be ~250+ tokens. Set tight budget.
        full_tokens = estimate_tokens(str(builder))
        tight_budget = full_tokens - 50  # Force dropping something

        ctx = ContextWindow(max_tokens=tight_budget, reserve_output=0)
        result = ctx.fit_prompt(builder)

        assert len(result.sections_dropped) >= 1
        assert result.tokens_used <= tight_budget

    def test_preserves_persona_and_task(self):
        from talk_box.prompt_builder import PromptBuilder, Priority

        builder = (
            PromptBuilder()
            .persona("helpful assistant")
            .task_context("Answer questions")
            .structured_section("EXTRA", "Z" * 1000, priority=Priority.LOW)
        )

        ctx = ContextWindow(max_tokens=100, reserve_output=0)
        result = ctx.fit_prompt(builder)

        assert "helpful assistant" in result.text
        assert "Answer questions" in result.text


# ── Strategy override per call ─────────────────────────────────────────────


class TestStrategyOverride:
    def test_override_strategy(self):
        ctx = ContextWindow(max_tokens=30, reserve_output=0, strategy="truncate_oldest")
        messages = [{"role": "user", "content": f"msg{i}" + "x" * 36} for i in range(5)]
        result = ctx.fit_messages(messages, strategy="truncate_middle")
        assert result.strategy_used == FitStrategy.TRUNCATE_MIDDLE
        # Should keep first and last
        assert "msg0" in result.messages[0]["content"]
        assert "msg4" in result.messages[-1]["content"]


# ── Message normalization ──────────────────────────────────────────────────


class TestMessageNormalization:
    def test_dict_messages(self):
        ctx = ContextWindow(max_tokens=1000, reserve_output=0)
        messages = [{"role": "user", "content": "hello"}]
        result = ctx.fit_messages(messages)
        assert result.messages == [{"role": "user", "content": "hello"}]

    def test_message_objects(self):
        from talk_box.conversation import Message

        ctx = ContextWindow(max_tokens=1000, reserve_output=0)
        messages = [Message(role="user", content="hello")]
        result = ctx.fit_messages(messages)
        assert result.messages == [{"role": "user", "content": "hello"}]


# ── Custom token counter ──────────────────────────────────────────────────


class TestCustomTokenCounter:
    def test_word_based_counter(self):
        def counter(text):
            return len(text.split())

        ctx = ContextWindow(max_tokens=10, reserve_output=0, token_counter=counter)

        messages = [
            {"role": "user", "content": "one two three four five"},  # 5 tokens
            {"role": "user", "content": "six seven eight nine ten"},  # 5 tokens
            {"role": "user", "content": "eleven twelve thirteen fourteen fifteen"},  # 5 tokens
        ]
        result = ctx.fit_messages(messages)
        # Budget = 10 words, can fit last 2 messages
        assert result.messages_dropped == 1
        assert len(result.messages) == 2
