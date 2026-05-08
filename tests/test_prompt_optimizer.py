"""Tests for talk_box.prompt_optimizer module."""

from talk_box.prompt_optimizer import (
    OptimizationLevel,
    OptimizeResult,
    _collapse_blank_lines,
    _compact_constraints,
    _compact_examples,
    _compact_persona,
    _compact_section_headers,
    _compact_vocabulary,
    _normalize_whitespace,
    _strip_markdown,
    _truncate_long_lines,
    optimize_prompt,
)


# ── OptimizationLevel enum ────────────────────────────────────────────────


class TestOptimizationLevel:
    def test_values(self):
        assert OptimizationLevel.LIGHT.value == "light"
        assert OptimizationLevel.MODERATE.value == "moderate"
        assert OptimizationLevel.AGGRESSIVE.value == "aggressive"

    def test_from_string(self):
        assert OptimizationLevel("moderate") == OptimizationLevel.MODERATE


# ── Strip markdown ─────────────────────────────────────────────────────────


class TestStripMarkdown:
    def test_bold(self):
        assert _strip_markdown("**important**") == "important"

    def test_bold_underscores(self):
        assert _strip_markdown("__important__") == "important"

    def test_italic(self):
        assert _strip_markdown("*emphasis*") == "emphasis"

    def test_inline_code(self):
        assert _strip_markdown("`code`") == "code"

    def test_preserves_plain_text(self):
        text = "No formatting here"
        assert _strip_markdown(text) == text

    def test_mixed_formatting(self):
        text = "Use **bold** and `code` and *italic*"
        assert _strip_markdown(text) == "Use bold and code and italic"

    def test_preserves_snake_case(self):
        text = "my_variable_name"
        assert _strip_markdown(text) == "my_variable_name"


# ── Compact persona ───────────────────────────────────────────────────────


class TestCompactPersona:
    def test_with_expertise(self):
        text = "You are a data analyst with expertise in statistics."
        assert _compact_persona(text) == "Role: data analyst. Expertise: statistics."

    def test_with_expertise_an(self):
        text = "You are an expert reviewer with expertise in code quality."
        assert _compact_persona(text) == "Role: expert reviewer. Expertise: code quality."

    def test_without_expertise(self):
        text = "You are a helpful assistant.\nDo your best."
        result = _compact_persona(text)
        assert "Role: helpful assistant." in result

    def test_no_persona_unchanged(self):
        text = "TASK: Analyze data"
        assert _compact_persona(text) == text


# ── Compact section headers ───────────────────────────────────────────────


class TestCompactSectionHeaders:
    def test_removes_blank_before_header(self):
        text = "Some text\n\nCRITICAL REQUIREMENTS:\n- Do this"
        result = _compact_section_headers(text)
        assert "\n\nCRITICAL" not in result
        assert "\nCRITICAL REQUIREMENTS:" in result

    def test_multiple_headers(self):
        text = "Intro\n\nTASK CONTEXT:\nDo X\n\nADDITIONAL CONSTRAINTS:\n- Y"
        result = _compact_section_headers(text)
        assert result.count("\n\n") == 0


# ── Compact examples ──────────────────────────────────────────────────────


class TestCompactExamples:
    def test_condenses_example(self):
        text = "Example 1:\nInput: What is 2+2?\nOutput: 4"
        result = _compact_examples(text)
        assert "Ex1: What is 2+2? → 4" in result

    def test_multiple_examples(self):
        text = (
            "EXAMPLES:\n\n"
            "Example 1:\nInput: Hello\nOutput: Hi\n\n"
            "Example 2:\nInput: Bye\nOutput: Goodbye"
        )
        result = _compact_examples(text)
        assert "Ex1: Hello → Hi" in result
        assert "Ex2: Bye → Goodbye" in result
        assert "EXAMPLES:" not in result

    def test_no_examples_unchanged(self):
        text = "No examples here"
        assert _compact_examples(text) == text


# ── Compact constraints ───────────────────────────────────────────────────


class TestCompactConstraints:
    def test_merges_short_constraints(self):
        text = "CRITICAL REQUIREMENTS:\n- Be concise\n- Use examples\n"
        result = _compact_constraints(text)
        assert "CRITICAL REQUIREMENTS: Be concise; Use examples" in result

    def test_preserves_long_constraints(self):
        long = "x" * 70
        text = f"CRITICAL REQUIREMENTS:\n- {long}\n- Also this\n"
        result = _compact_constraints(text)
        # Should keep list format since one constraint > 60 chars
        assert f"- {long}" in result

    def test_additional_constraints(self):
        text = "ADDITIONAL CONSTRAINTS:\n- Short one\n- Another\n"
        result = _compact_constraints(text)
        assert "ADDITIONAL CONSTRAINTS: Short one; Another" in result


# ── Compact vocabulary ─────────────────────────────────────────────────────


class TestCompactVocabulary:
    def test_truncates_long_definition(self):
        text = "- Revenue: Total income. This includes all sources of revenue from operations and investments."
        result = _compact_vocabulary(text)
        assert "- Revenue: Total income." in result
        assert "investments" not in result

    def test_preserves_short_definition(self):
        text = "- API: Application programming interface."
        result = _compact_vocabulary(text)
        assert result == text


# ── Truncate long lines ───────────────────────────────────────────────────


class TestTruncateLongLines:
    def test_truncates_long_line(self):
        text = "a " * 150  # 300 chars
        result = _truncate_long_lines(text, max_chars=100)
        assert len(result) < 150
        assert result.endswith("...")

    def test_preserves_short_lines(self):
        text = "Short line"
        assert _truncate_long_lines(text) == text

    def test_preserves_task_header(self):
        text = "TASK: " + "x" * 250
        result = _truncate_long_lines(text)
        assert result == text


# ── Collapse blank lines ──────────────────────────────────────────────────


class TestCollapseBlankLines:
    def test_collapses_triple(self):
        text = "A\n\n\nB"
        assert _collapse_blank_lines(text) == "A\n\nB"

    def test_collapses_many(self):
        text = "A\n\n\n\n\nB"
        assert _collapse_blank_lines(text) == "A\n\nB"

    def test_preserves_double(self):
        text = "A\n\nB"
        assert _collapse_blank_lines(text) == "A\n\nB"


# ── Normalize whitespace ──────────────────────────────────────────────────


class TestNormalizeWhitespace:
    def test_strips_trailing(self):
        text = "Hello   \nWorld  "
        result = _normalize_whitespace(text)
        assert result == "Hello\nWorld"

    def test_strips_leading_blanks(self):
        text = "\n\nHello"
        result = _normalize_whitespace(text)
        assert result == "Hello"

    def test_strips_trailing_blanks(self):
        text = "Hello\n\n"
        result = _normalize_whitespace(text)
        assert result == "Hello"


# ── optimize_prompt integration ────────────────────────────────────────────


class TestOptimizePrompt:
    def test_returns_optimize_result(self):
        result = optimize_prompt("Hello world")
        assert isinstance(result, OptimizeResult)
        assert result.optimized_tokens > 0
        assert result.level == OptimizationLevel.MODERATE

    def test_level_from_string(self):
        result = optimize_prompt("Hello", level="light")
        assert result.level == OptimizationLevel.LIGHT

    def test_light_strips_markdown(self):
        text = "Use **bold** text"
        result = optimize_prompt(text, level="light")
        assert "**" not in result.text
        assert "bold" in result.text

    def test_moderate_compacts_persona(self):
        text = "You are a data analyst with expertise in statistics.\n\nTASK: analyze"
        result = optimize_prompt(text, level="moderate")
        assert "Role: data analyst" in result.text

    def test_moderate_compacts_examples(self):
        text = "Example 1:\nInput: Hello\nOutput: Hi"
        result = optimize_prompt(text, level="moderate")
        assert "Ex1:" in result.text

    def test_aggressive_truncates(self):
        long_line = "word " * 60  # 300 chars
        text = f"- term: {long_line}"
        result = optimize_prompt(text, level="aggressive")
        assert result.optimized_tokens <= result.original_tokens

    def test_reduction_pct_calculated(self):
        text = (
            "You are a data analyst with expertise in statistics.\n\n"
            "CRITICAL REQUIREMENTS:\n- Be concise\n- Use data\n\n"
            "Example 1:\nInput: Revenue?\nOutput: $1M\n\n"
            "Example 2:\nInput: Costs?\nOutput: $500K"
        )
        result = optimize_prompt(text, level="moderate")
        assert result.reduction_pct > 0
        assert result.original_tokens > result.optimized_tokens

    def test_with_prompt_builder(self):
        from talk_box.prompt_builder import PromptBuilder

        builder = (
            PromptBuilder()
            .persona("analyst", "data science")
            .task_context("Analyze sales data")
            .constraint("Be concise")
            .example("Q: Revenue?", "A: $1.2M")
        )
        result = optimize_prompt(builder, level="moderate")
        assert isinstance(result, OptimizeResult)
        assert "analyst" in result.text
        assert result.original_tokens > 0

    def test_empty_string(self):
        result = optimize_prompt("")
        assert result.text == ""
        assert result.reduction_pct == 0.0

    def test_aggressive_on_large_prompt(self):
        from talk_box.prompt_builder import Priority, PromptBuilder

        builder = (
            PromptBuilder()
            .persona("senior data analyst", "financial analytics")
            .task_context("Perform comprehensive quarterly revenue analysis")
            .critical_constraint("All figures must be auditable")
            .constraint("Include year-over-year comparisons")
            .constraint("Use standard financial terms")
            .structured_section(
                "BACKGROUND",
                "The company has seen significant growth in Q3 with revenue "
                "increasing across all product lines. Market conditions remain "
                "favorable despite ongoing economic uncertainties.",
                priority=Priority.MEDIUM,
            )
            .example("Q: What was Q3 revenue?", "A: $4.2M, up 15% YoY")
            .example("Q: Top product?", "A: Widget Pro, 42% of revenue")
            .output_format("Use bullet points for key findings")
            .focus_on("Revenue trends and profitability")
        )

        original = str(builder)
        result = optimize_prompt(builder, level="aggressive")
        assert result.optimized_tokens < result.original_tokens
        assert result.reduction_pct > 0
        # Key content preserved
        assert "analyst" in result.text.lower()
        assert "revenue" in result.text.lower()
