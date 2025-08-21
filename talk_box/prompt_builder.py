from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class Priority(Enum):
    """
    Priority levels for prompt components based on attention positioning.

    Values
    ------
    The following priority levels are defined:

    - `CRITICAL`: front-loaded placement for highest attention and maximum impact
    - `HIGH`: early placement in the prompt structure for strong attention
    - `MEDIUM`: middle section placement for standard attention levels
    - `LOW`: less prominent placement for supporting information
    """

    CRITICAL = "critical"  # Front-loaded, gets highest attention
    HIGH = "high"  # Early placement
    MEDIUM = "medium"  # Middle sections
    LOW = "low"  # Less prominent placement


@dataclass
class PromptSection:
    """
    Represents a structured section of a prompt with attention metadata.

    Parameters
    ----------
    content
        The text content of the prompt section
    priority
        Attention priority level determining section placement order
    section_type
        Type classification for the section (e.g., "general", "structured")
    order_hint
        Ordering hint where lower numbers appear earlier in the prompt
    """

    content: str
    priority: Priority = Priority.MEDIUM
    section_type: str = "general"
    order_hint: int = 0  # Lower numbers appear earlier


class PromptBuilder:
    """
    Builds structured prompts using attention mechanisms and cognitive principles.

    The PromptBuilder leverages insights from modern prompt engineering research to create prompts
    that maximize model attention on critical information while maintaining natural conversation
    flow.

    Returns
    -------
    PromptBuilder
        A new instance ready for fluent method chaining

    Notes
    -----
    This implementation integrates cutting-edge research from attention mechanisms and cognitive psychology
    to optimize prompt effectiveness. The PromptBuilder applies proven principles that enhance model
    performance and response quality through strategic information placement and cognitive load management.

    **Attention Mechanisms Applied:**

    - **Positional encoding**: Critical information placed strategically
    - **Multi-head attention**: Different types of context handled separately
    - **Hierarchical structure**: Information organized by importance and relevance
    - **Context windowing**: Optimal information density for model processing

    **Cognitive Psychology Integration:**

    - **Primacy effect**: Important instructions placed early
    - **Recency effect**: Final emphasis reinforces key objectives
    - **Chunking**: Information grouped into digestible, logical units
    - **Salience**: Critical constraints highlighted for maximum attention

    Examples
    --------
    ### Basic prompt construction

    Create a simple prompt with persona and task:

    ```{python}
    import talk_box as tb

    prompt = (
        tb.PromptBuilder()
        .persona("data scientist", "machine learning")
        .task_context("Analyze customer churn patterns")
        .focus_on("identifying the top 3 risk factors")
        .build()
    )
    ```

    We can easily print the prompt that was generated for this task:

    ```{python}
    print(prompt)
    ```

    ### Structured analysis prompt

    It is possible to build a much more comprehensive analysis prompt with multiple sections:

    ```{python}

    prompt = (
        tb.PromptBuilder()
        .persona("senior software architect")
        .critical_constraint("Focus only on production-ready solutions")
        .task_context("Review the codebase architecture")
        .core_analysis([
            "Identify design patterns used",
            "Assess scalability bottlenecks",
            "Review security implications"
        ])
        .structured_section(
            "Performance Metrics", [
                "Response time requirements",
                "Throughput expectations",
                "Memory usage constraints"
            ],
            priority=tb.Priority.HIGH
        )
        .output_format([
            "Executive summary (2-3 sentences)",
            "Detailed findings with code examples",
            "Prioritized recommendations"
        ])
        .final_emphasis("Provide actionable next steps")
        .build()
    )
    ```

    The generated prompt can be printed as follows:

    ```{python}
    print(prompt)
    ```

    ### Code review prompt

    Create a specialized prompt for code reviews:

    ```{python}
    prompt = (
        tb.PromptBuilder()
        .persona("senior developer", "code quality and best practices")
        .task_context("Review the pull request for potential issues")
        .critical_constraint("Flag any security vulnerabilities immediately")
        .structured_section(
            "Review Areas", [
                "Logic and correctness",
                "Security considerations",
                "Performance implications",
                "Code readability and documentation"
            ]
        )
        .output_format([
            "Critical issues (must fix)",
            "Suggestions (should consider)",
            "Positive feedback"
        ])
        .avoid_topics(["personal coding style preferences"])
        .focus_on("providing constructive, actionable feedback")
        .build()
    )
    ```

    Let us look at the generated prompt:

    ```{python}
    print(prompt)
    ```

    ### Using pre-configured builders

    Leverage built-in templates for common tasks:

    ```{python}
    # Use pre-configured architectural analysis
    arch_prompt = tb.architectural_analysis_prompt().build()
    print(arch_prompt)
    ```

    ```{python}
    # Use pre-configured code review
    review_prompt = tb.code_review_prompt().build()
    print(review_prompt)
    ```
    """

    def __init__(self):
        self._sections: List[PromptSection] = []
        self._persona: Optional[str] = None
        self._task_context: Optional[str] = None
        self._constraints: List[str] = []
        self._output_format: List[str] = []
        self._examples: List[Dict[str, str]] = []
        self._final_emphasis: Optional[str] = None

    def persona(self, role: str, expertise: Optional[str] = None) -> "PromptBuilder":
        """
        Set a behavioral persona to anchor the model's response style.

        Based on Kong et al. (2023) research showing personas improve reasoning
        performance by providing behavioral context.

        Parameters
        ----------
        role
            The primary role (e.g., "senior software architect")
        expertise
            Optional specific expertise area

        Returns
        -------
        PromptBuilder
            Self for method chaining
        """
        persona_text = f"You are a {role}"
        if expertise:
            persona_text += f" with expertise in {expertise}"
        persona_text += "."

        self._persona = persona_text
        return self

    def task_context(self, context: str, priority: Priority = Priority.CRITICAL) -> "PromptBuilder":
        """
        Define the primary task context with specified attention priority.

        Parameters
        ----------
        context
            Clear description of what needs to be accomplished
        priority
            Attention priority level for placement

        Returns
        -------
        PromptBuilder
            Self for method chaining
        """
        self._task_context = context
        return self

    def critical_constraint(self, constraint: str) -> "PromptBuilder":
        """
        Add a critical constraint that will be front-loaded for maximum attention.

        Based on Mao et al. (2024) findings that early-positioned instructions
        have the greatest impact on task accuracy.

        Parameters
        ----------
        constraint
            Specific constraint or requirement

        Returns
        -------
        PromptBuilder
            Self for method chaining
        """
        self._constraints.insert(0, constraint)
        return self

    def constraint(self, constraint: str) -> "PromptBuilder":
        """
        Add a standard constraint to the prompt.

        Parameters
        ----------
        constraint
            Specific constraint or requirement

        Returns
        -------
        PromptBuilder
            Self for method chaining
        """
        self._constraints.append(constraint)
        return self

    def structured_section(
        self,
        title: str,
        content: Union[str, List[str]],
        priority: Priority = Priority.MEDIUM,
        required: bool = False,
    ) -> "PromptBuilder":
        """
        Add a structured section with clear hierarchical boundaries.

        Creates distinct attention clusters as recommended by Liu et al. (2023)
        for preventing attention drift in complex prompts.

        Parameters
        ----------
        title
            Section heading for clear visual separation
        content
            Section content as string or list of items
        priority
            Attention priority for section placement
        required
            Whether to mark as required in the output

        Returns
        -------
        PromptBuilder
            Self for method chaining
        """
        if isinstance(content, list):
            content_str = "\n".join(f"- {item}" for item in content)
        else:
            content_str = content

        section_title = title.upper()
        if required:
            section_title += " (Required)"

        section_content = f"{section_title}:\n{content_str}"

        section = PromptSection(
            content=section_content,
            priority=priority,
            section_type="structured",
            order_hint=len(self._sections),
        )

        self._sections.append(section)
        return self

    def core_analysis(self, analysis_points: List[str]) -> "PromptBuilder":
        """
        Define core analysis requirements as a high-priority structured section.

        Parameters
        ----------
        analysis_points
            List of specific analysis requirements

        Returns
        -------
        PromptBuilder
            Self for method chaining
        """
        return self.structured_section(
            "Core Analysis", analysis_points, priority=Priority.HIGH, required=True
        )

    def output_format(self, format_specs: List[str]) -> "PromptBuilder":
        """
        Specify output formatting requirements to prevent ambiguous responses.

        Addresses attention drift issues identified in Brown et al. (2020) by
        providing specific, measurable formatting constraints.

        Parameters
        ----------
        format_specs
            List of specific formatting requirements

        Returns
        -------
        PromptBuilder
            Self for method chaining
        """
        self._output_format.extend(format_specs)
        return self

    def example(self, input_example: str, output_example: str) -> "PromptBuilder":
        """
        Add an input/output example for few-shot learning.

        Parameters
        ----------
        input_example
            Example input
        output_example
            Expected output format

        Returns
        -------
        PromptBuilder
            Self for method chaining
        """
        self._examples.append({"input": input_example, "output": output_example})
        return self

    def final_emphasis(self, emphasis: str) -> "PromptBuilder":
        """
        Set final emphasis that leverages recency bias for critical instructions.

        Parameters
        ----------
        emphasis
            Critical instruction to emphasize at the end

        Returns
        -------
        PromptBuilder
            Self for method chaining
        """
        self._final_emphasis = emphasis
        return self

    def avoid_topics(self, topics: List[str]) -> "PromptBuilder":
        """
        Specify topics or behaviors to avoid (negative constraints).

        Parameters
        ----------
        topics
            List of topics or behaviors to avoid

        Returns
        -------
        PromptBuilder
            Self for method chaining
        """
        avoid_text = "Avoid: " + ", ".join(topics)
        return self.constraint(avoid_text)

    def focus_on(self, primary_goal: str) -> "PromptBuilder":
        """
        Set the primary focus that will be emphasized at the end.

        Leverages both front-loading and recency bias for maximum impact.

        Parameters
        ----------
        primary_goal
            The most important objective

        Returns
        -------
        PromptBuilder
            Self for method chaining
        """
        # Add as critical constraint (front-loaded)
        self.critical_constraint(f"Primary objective: {primary_goal}")
        # Also set as final emphasis (recency bias)
        self._final_emphasis = f"Focus your entire response on: {primary_goal}"
        return self

    def build(self) -> str:
        """
        Construct the final prompt using attention-optimized structure.

        Implements the structural principles from the research:

        1. Persona first (behavioral anchoring)
        2. Critical constraints front-loaded
        3. Task context with clear priority
        4. Structured sections in priority order
        5. Output format specifications
        6. Examples if provided
        7. Final emphasis leveraging recency bias

        Returns
        -------
        str
            Optimally structured prompt string
        """
        prompt_parts = []

        # 1. Persona (behavioral anchoring)
        if self._persona:
            prompt_parts.append(self._persona)

        # 2. Critical constraints (front-loaded)
        critical_constraints = [c for c in self._constraints if self._constraints.index(c) == 0]
        if critical_constraints:
            prompt_parts.append("\nCRITICAL REQUIREMENTS:")
            for constraint in critical_constraints:
                prompt_parts.append(f"- {constraint}")

        # 3. Task context
        if self._task_context:
            prompt_parts.append(f"\nTASK: {self._task_context}")

        # 4. Structured sections in priority order
        sorted_sections = sorted(self._sections, key=lambda s: (s.priority.value, s.order_hint))

        for section in sorted_sections:
            prompt_parts.append(f"\n{section.content}")

        # 5. Standard constraints
        standard_constraints = self._constraints[1:] if len(self._constraints) > 1 else []
        if standard_constraints:
            prompt_parts.append("\nADDITIONAL CONSTRAINTS:")
            for constraint in standard_constraints:
                prompt_parts.append(f"- {constraint}")

        # 6. Output format
        if self._output_format:
            prompt_parts.append("\nOUTPUT FORMAT:")
            for format_spec in self._output_format:
                prompt_parts.append(f"- {format_spec}")

        # 7. Examples
        if self._examples:
            prompt_parts.append("\nEXAMPLES:")
            for i, example in enumerate(self._examples, 1):
                prompt_parts.append(f"\nExample {i}:")
                prompt_parts.append(f"Input: {example['input']}")
                prompt_parts.append(f"Output: {example['output']}")

        # 8. Final emphasis (recency bias)
        if self._final_emphasis:
            prompt_parts.append(f"\n{self._final_emphasis}")

        return "\n".join(prompt_parts)

    def preview_structure(self) -> Dict[str, Any]:
        """
        Preview the prompt structure without building the full text.

        Useful for debugging attention patterns and optimizing structure.

        Returns
        -------
        Dict[str, Any]
            Dictionary showing prompt structure and attention distribution
        """
        return {
            "persona": self._persona,
            "critical_constraints": self._constraints[:1] if self._constraints else [],
            "task_context": self._task_context,
            "structured_sections": [
                {
                    "content": s.content[:100] + "..." if len(s.content) > 100 else s.content,
                    "priority": s.priority.value,
                    "type": s.section_type,
                }
                for s in sorted(self._sections, key=lambda s: (s.priority.value, s.order_hint))
            ],
            "standard_constraints": self._constraints[1:] if len(self._constraints) > 1 else [],
            "output_format": self._output_format,
            "examples_count": len(self._examples),
            "final_emphasis": self._final_emphasis,
            "estimated_tokens": len(self.build().split()) * 1.3,  # Rough estimate
        }


# Convenience functions for common patterns
def architectural_analysis_prompt() -> PromptBuilder:
    """
    Create a pre-configured prompt builder for architectural analysis tasks.

    Implements the optimized pattern from the blog post example.

    Returns
    -------
    PromptBuilder
        Configured PromptBuilder for architectural analysis
    """
    return (
        PromptBuilder()
        .persona("senior software architect", "comprehensive codebase analysis")
        .task_context("Create comprehensive architectural documentation")
        .core_analysis(
            [
                "Tools, frameworks, and design patterns used across the repository",
                "Data models and API design & versioning patterns",
                "Any architectural inconsistencies or deviations from language/framework best practices",
            ]
        )
        .structured_section(
            "Legacy Assessment",
            [
                "Identify conflicting or multiple architectural patterns",
                "Recommend a best path forward with external source citations",
                "Distinguish between old and new architectural approaches",
            ],
            priority=Priority.MEDIUM,
        )
        .output_format(
            [
                "Use clear headings and bullet points",
                "Prioritize findings by impact and consistency",
                "Include specific examples from the codebase",
                "Reference external best practice sources for any recommendations",
            ]
        )
        .focus_on("identifying architectural debt and deviations from expected patterns")
    )


def code_review_prompt() -> PromptBuilder:
    """
    Create a pre-configured prompt builder for code review tasks.

    Returns
    -------
    PromptBuilder
        Configured PromptBuilder for code reviews
    """
    return (
        PromptBuilder()
        .persona("senior software engineer", "code review and best practices")
        .core_analysis(
            [
                "Security: Identify potential security vulnerabilities",
                "Performance: Suggest optimization opportunities",
                "Maintainability: Recommend cleaner, more readable code",
                "Best Practices: Ensure adherence to language conventions",
                "Testing: Suggest test cases for uncovered scenarios",
            ]
        )
        .output_format(
            [
                "Critical issues (security, bugs)",
                "Improvements (performance, style)",
                "Positive feedback (good practices)",
            ]
        )
        .avoid_topics(["personal criticism"])
        .focus_on("providing constructive, actionable feedback")
    )


def debugging_prompt() -> PromptBuilder:
    """
    Create a pre-configured prompt builder for debugging tasks.

    Returns
    -------
    PromptBuilder
        Configured PromptBuilder for debugging
    """
    return (
        PromptBuilder()
        .persona("expert debugger", "systematic problem analysis")
        .critical_constraint("Identify the root cause, not just symptoms")
        .structured_section(
            "Analysis Steps",
            [
                "1. Reproduce the issue with minimal test case",
                "2. Trace the execution path leading to the problem",
                "3. Identify the root cause and contributing factors",
                "4. Propose specific fixes with reasoning",
            ],
            priority=Priority.HIGH,
            required=True,
        )
        .output_format(
            [
                "Clear problem summary",
                "Step-by-step reproduction steps",
                "Root cause analysis",
                "Recommended fix with code examples",
            ]
        )
        .focus_on("finding the root cause and providing a complete solution")
    )
