import pytest
from talk_box import (
    PromptBuilder,
    Priority,
    PromptSection,
    architectural_analysis_prompt,
    code_review_prompt,
    debugging_prompt,
    ChatBot,
)


class TestPromptBuilder:
    """Test the PromptBuilder class and its methods."""

    def test_builder_creation(self):
        """Test that the builder can be created and initialized."""
        builder = PromptBuilder()
        assert builder is not None
        assert len(builder._sections) == 0
        assert builder._persona is None
        assert builder._task_context is None
        assert len(builder._constraints) == 0

    def test_persona_method(self):
        """Test the persona method."""
        builder = PromptBuilder()
        result = builder.persona("senior engineer", "security analysis")

        # Test method chaining
        assert result is builder
        assert "senior engineer" in builder._persona
        assert "security analysis" in builder._persona

    def test_task_context_method(self):
        """Test the task_context method."""
        builder = PromptBuilder()
        builder.task_context("Analyze code for security issues")

        assert builder._task_context == "Analyze code for security issues"

    def test_critical_constraint_method(self):
        """Test the critical_constraint method."""
        builder = PromptBuilder()
        builder.critical_constraint("Focus on critical security issues")

        assert len(builder._constraints) == 1
        assert builder._constraints[0] == "Focus on critical security issues"

    def test_core_analysis_method(self):
        """Test the core_analysis method."""
        builder = PromptBuilder()
        analysis_points = ["Security", "Performance", "Maintainability"]
        builder.core_analysis(analysis_points)

        # Should create a structured section
        assert len(builder._sections) == 1
        section = builder._sections[0]
        assert "CORE ANALYSIS" in section.content
        assert section.priority == Priority.HIGH

    def test_structured_section_method(self):
        """Test the structured_section method."""
        builder = PromptBuilder()
        builder.structured_section(
            "Test Section", ["Item 1", "Item 2"], priority=Priority.CRITICAL, required=True
        )

        assert len(builder._sections) == 1
        section = builder._sections[0]
        assert "TEST SECTION (Required)" in section.content
        assert section.priority == Priority.CRITICAL

    def test_output_format_method(self):
        """Test the output_format method."""
        builder = PromptBuilder()
        format_specs = ["Use bullet points", "Include examples"]
        builder.output_format(format_specs)

        assert builder._output_format == format_specs

    def test_avoid_topics_method(self):
        """Test the avoid_topics method."""
        builder = PromptBuilder()
        topics = ["personal criticism", "style issues"]
        builder.avoid_topics(topics)

        assert len(builder._constraints) == 1
        assert "personal criticism" in builder._constraints[0]
        assert "style issues" in builder._constraints[0]

    def test_focus_on_method(self):
        """Test the focus_on method that leverages both primacy and recency bias."""
        builder = PromptBuilder()
        builder.focus_on("identifying security vulnerabilities")

        # Should add both critical constraint and final emphasis
        assert len(builder._constraints) == 1
        assert "identifying security vulnerabilities" in builder._constraints[0]
        assert builder._final_emphasis is not None
        assert "identifying security vulnerabilities" in builder._final_emphasis

    def test_method_chaining(self):
        """Test that all methods support method chaining."""
        builder = (
            PromptBuilder()
            .persona("test engineer")
            .task_context("test task")
            .critical_constraint("test constraint")
            .core_analysis(["test analysis"])
            .output_format(["test format"])
            .focus_on("test focus")
        )

        assert builder._persona is not None
        assert builder._task_context is not None
        assert len(builder._constraints) == 2  # critical + focus
        assert len(builder._sections) == 1
        assert len(builder._output_format) == 1
        assert builder._final_emphasis is not None

    def test_build_method(self):
        """Test the build method creates a properly structured prompt."""
        prompt = str(
            PromptBuilder()
            .persona("senior engineer")
            .critical_constraint("Focus on critical issues")
            .task_context("Review code")
            .core_analysis(["Security", "Performance"])
            .output_format(["Use bullet points"])
            .final_emphasis("Prioritize by impact")
        )

        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "senior engineer" in prompt
        assert "Focus on critical issues" in prompt
        assert "Review code" in prompt
        assert "Security" in prompt
        assert "Prioritize by impact" in prompt

    def test_structure_inspection(self):
        """Test the builder's internal structure can be inspected."""
        builder = (
            PromptBuilder()
            .persona("test engineer")
            .critical_constraint("test constraint")
            .core_analysis(["test"])
            .output_format(["test format"])
        )

        # Test internal structure access
        assert builder._persona is not None
        assert "test engineer" in builder._persona
        assert len(builder._constraints) == 1
        assert "test constraint" in builder._constraints[0]
        assert len(builder._sections) == 1
        assert len(builder._output_format) == 1
        assert "test format" in builder._output_format[0]

        # Test string conversion works
        prompt_text = str(builder)
        assert isinstance(prompt_text, str)
        assert len(prompt_text) > 0
        assert "test engineer" in prompt_text


class TestPreConfiguredBuilders:
    """Test the pre-configured prompt builders."""

    def test_architectural_analysis_prompt(self):
        """Test the architectural analysis pre-configured builder."""
        builder = architectural_analysis_prompt()

        assert isinstance(builder, PromptBuilder)

        # Test that the builder has expected components
        assert "architect" in builder._persona.lower()
        assert len(builder._sections) >= 2

        prompt = str(builder)
        assert "architect" in prompt.lower()
        assert "architectural" in prompt.lower()

    def test_code_review_prompt(self):
        """Test the code review pre-configured builder."""
        builder = code_review_prompt()

        assert isinstance(builder, PromptBuilder)

        # Test that the builder has expected components
        assert "engineer" in builder._persona.lower()
        assert len(builder._output_format) > 0

        prompt = str(builder)
        assert "engineer" in prompt.lower()
        assert "critical issues" in prompt.lower() or "improvements" in prompt.lower()

    def test_debugging_prompt(self):
        """Test the debugging pre-configured builder."""
        builder = debugging_prompt()

        assert isinstance(builder, PromptBuilder)

        # Test that the builder has expected components
        assert "debugger" in builder._persona.lower()
        assert "root cause" in builder._final_emphasis.lower()

        prompt = str(builder)
        assert "debugger" in prompt.lower()
        assert "root cause" in prompt.lower()


class TestChatBotIntegration:
    """Test integration with the ChatBot class."""

    def test_prompt_builder_method(self):
        """Test the ChatBot.prompt_builder method."""
        bot = ChatBot()

        # Test general builder
        builder = bot.prompt_builder()
        assert isinstance(builder, PromptBuilder)

        # Test pre-configured builders
        arch_builder = bot.prompt_builder("architectural")
        assert isinstance(arch_builder, PromptBuilder)

        review_builder = bot.prompt_builder("code_review")
        assert isinstance(review_builder, PromptBuilder)

        debug_builder = bot.prompt_builder("debugging")
        assert isinstance(debug_builder, PromptBuilder)

    def test_structured_prompt_method(self):
        """Test the ChatBot.structured_prompt method."""
        bot = ChatBot()

        result = bot.structured_prompt(
            persona="test engineer",
            task="test task",
            constraints=["test constraint"],
            format=["test format"],
            focus="test focus",
        )

        # Should return the bot for chaining
        assert result is bot

        # Should set system prompt
        assert bot._config.get("system_prompt") is not None
        system_prompt = bot._config["system_prompt"]
        assert "test engineer" in system_prompt
        assert "test task" in system_prompt

    def test_chain_prompts_method(self):
        """Test the ChatBot.chain_prompts method."""
        bot = ChatBot()

        prompt1 = "You are an expert engineer."
        prompt2 = str(PromptBuilder().core_analysis(["Security"]))
        prompt3 = "Focus on critical issues."

        result = bot.chain_prompts(prompt1, prompt2, prompt3)

        # Should return the bot for chaining
        assert result is bot

        # Should set system prompt
        assert bot._config.get("system_prompt") is not None
        system_prompt = bot._config["system_prompt"]
        assert "expert engineer" in system_prompt
        assert "Security" in system_prompt
        assert "critical issues" in system_prompt


class TestPromptSection:
    """Test the PromptSection dataclass."""

    def test_prompt_section_creation(self):
        """Test creating a PromptSection."""
        section = PromptSection(
            content="Test content", priority=Priority.HIGH, section_type="test", order_hint=1
        )

        assert section.content == "Test content"
        assert section.priority == Priority.HIGH
        assert section.section_type == "test"
        assert section.order_hint == 1

    def test_prompt_section_defaults(self):
        """Test PromptSection with default values."""
        section = PromptSection("Test content")

        assert section.content == "Test content"
        assert section.priority == Priority.MEDIUM
        assert section.section_type == "general"
        assert section.order_hint == 0


class TestPriority:
    """Test the Priority enum."""

    def test_priority_values(self):
        """Test that Priority enum has expected values."""
        assert Priority.CRITICAL.value == "critical"
        assert Priority.HIGH.value == "high"
        assert Priority.MEDIUM.value == "medium"
        assert Priority.LOW.value == "low"

    def test_priority_ordering(self):
        """Test that priorities can be compared for sorting."""
        priorities = [Priority.LOW, Priority.CRITICAL, Priority.MEDIUM, Priority.HIGH]

        # Should be able to sort by value
        sorted_priorities = sorted(priorities, key=lambda p: p.value)
        assert sorted_priorities[0] == Priority.CRITICAL
        assert sorted_priorities[1] == Priority.HIGH
        assert sorted_priorities[2] == Priority.LOW  # "low" comes before "medium" alphabetically
        assert sorted_priorities[3] == Priority.MEDIUM
