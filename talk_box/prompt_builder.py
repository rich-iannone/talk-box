from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union


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
    Represents a structured section of an attention-optimized prompt with priority and ordering metadata.

    The `PromptSection` class is a fundamental building block used by `PromptBuilder` to create
    sophisticated, attention-optimized prompts. Each section encapsulates content along with
    metadata that controls how the section is positioned and prioritized within the final prompt.
    This enables precise control over attention flow and information hierarchy.

    **Integration with PromptBuilder**:

    While users can create `PromptSection` objects directly, they are typically created automatically
    by `PromptBuilder` methods. The sections are then assembled according to attention principles
    to create optimized final prompts. This design provides both high-level convenience through
    `PromptBuilder` and fine-grained control through direct `PromptSection` manipulation.

    **Attention Optimization**:

    Each section contributes to the overall attention strategy:
    - **Priority**: Determines relative importance and influences final ordering
    - **Section Type**: Enables grouping and specialized handling of content types
    - **Order Hint**: Provides fine-grained control over section positioning
    - **Content**: The actual prompt text optimized for the section's role

    The combination of these attributes allows the prompt building system to create prompts
    that leverage attention mechanisms effectively, ensuring critical information receives
    appropriate model focus while maintaining natural conversation flow.

    Parameters
    ----------
    content
        The text content of the prompt section. This is the actual text that will appear
        in the final prompt. Content should be crafted to serve the section's specific
        purpose within the overall prompt strategy.
    priority
        Attention priority level determining section placement order and emphasis.
        Higher priority sections are typically placed in more prominent positions.
        Defaults to `Priority.MEDIUM`.
    section_type
        Type classification for the section enabling specialized handling and grouping.
        This allows the prompt builder to apply type-specific optimization strategies.
        Defaults to `"general"`.
    order_hint
        Fine-grained ordering hint where lower numbers appear earlier in the prompt.
        This provides precise control over section positioning beyond priority levels.
        Sections with the same priority are ordered by this value. Defaults to `0`.

    Returns
    -------
    PromptSection
        A new prompt section with the specified content and metadata.

    Priority Levels
    ---------------
    The available priority levels are:

    - `Priority.CRITICAL`: highest importance, placed prominently
    - `Priority.HIGH`: important content requiring strong attention
    - `Priority.MEDIUM`: standard priority for general content
    - `Priority.LOW`: supporting information, de-emphasized placement
    - `Priority.MINIMAL`: background context, least prominent placement

    Section Types
    -------------
    Common section types include:

    - `"persona"`: role and behavioral context
    - `"constraint"`: requirements and limitations
    - `"analysis"`: core analysis tasks and objectives
    - `"format"`: output formatting requirements
    - `"example"`: input/output examples and demonstrations
    - `"emphasis"`: final reinforcement and focus directives
    - `"general"`: general-purpose content

    Section Lifecycle
    -----------------
    Prompt sections typically follow this lifecycle within the prompt building process:

    1. **Creation**: sections are created with content and metadata
    2. **Collection**: multiple sections are gathered by the PromptBuilder
    3. **Sorting**: sections are ordered by priority and order_hint values
    4. **Grouping**: sections are grouped by type for specialized handling
    5. **Assembly**: final prompt is constructed from ordered sections
    6. **Optimization**: content is refined for attention and coherence

    Design Principles
    -----------------
    **Attention Optimization**: sections are designed to work together to guide model
    attention effectively, with priority and positioning controlling information hierarchy.

    **Modularity**: each section encapsulates a specific aspect of the prompt, enabling
    reusable components and systematic prompt construction.

    **Flexibility**: the section system supports both structured workflows through
    standard section types and custom applications through extensible metadata.

    **Composability**: sections can be combined, reordered, and manipulated to create
    sophisticated prompt strategies for different use cases.

    **Cognitive Alignment**: section design aligns with cognitive psychology principles
    like primacy/recency effects and information chunking for optimal comprehension.

    Integration Notes
    -----------------
    - **Automatic Ordering**: when used with `PromptBuilder`, sections are automatically ordered by
    priority and order_hint for optimal attention flow
    - **Type-Based Processing**: section types enable specialized handling and validation within the
    prompt building pipeline
    - **Content Optimization**: section content should be crafted for clarity and specificity to
    maximize prompt effectiveness
    - **Memory Efficiency**: sections are lightweight dataclasses suitable for large-scale prompt
    construction workflows

    The `PromptSection` class provides the foundation for systematic, attention-optimized prompt
    engineering, enabling both simple prompt construction and sophisticated multi-component prompt
    strategies.

    Examples
    --------
    ### Creating basic prompt sections

    Create sections for different types of prompt content:

    ```python
    import talk_box as tb

    # High-priority persona section
    persona_section = tb.PromptSection(
        content="You are a senior software architect with expertise in distributed systems.",
        priority=tb.Priority.CRITICAL,
        section_type="persona",
        order_hint=1
    )

    # Critical constraint section
    constraint_section = tb.PromptSection(
        content="Focus only on scalability issues that impact performance.",
        priority=tb.Priority.CRITICAL,
        section_type="constraint",
        order_hint=2
    )

    # Medium-priority analysis section
    analysis_section = tb.PromptSection(
        content="Analyze the system architecture for bottlenecks and optimization opportunities.",
        priority=tb.Priority.MEDIUM,
        section_type="analysis",
        order_hint=10
    )

    print(f"Persona: {persona_section.content}")
    print(f"Priority: {persona_section.priority}")
    print(f"Type: {persona_section.section_type}")
    ```

    ### Working with section priorities

    Use priorities to control attention hierarchy:

    ```python
    # Create sections with different priorities
    sections = [
        tb.PromptSection(
            content="Secondary consideration: Check for code style consistency.",
            priority=tb.Priority.LOW,
            section_type="analysis"
        ),
        tb.PromptSection(
            content="CRITICAL: Identify security vulnerabilities immediately.",
            priority=tb.Priority.CRITICAL,
            section_type="constraint"
        ),
        tb.PromptSection(
            content="Important: Focus on performance bottlenecks.",
            priority=tb.Priority.HIGH,
            section_type="analysis"
        ),
        tb.PromptSection(
            content="Background context: This is a financial application.",
            priority=tb.Priority.MINIMAL,
            section_type="general"
        )
    ]
    ```

    ### Using section types for specialized handling

    Organize content by type for targeted optimization:

    ```python
    # Create sections representing different prompt components
    prompt_sections = [
        tb.PromptSection(
            content="You are an expert code reviewer.",
            priority=tb.Priority.CRITICAL,
            section_type="persona"
        ),
        tb.PromptSection(
            content="Focus on security issues and performance problems.",
            priority=tb.Priority.HIGH,
            section_type="constraint"
        ),
        tb.PromptSection(
            content="Analyze the code for bugs, security flaws, and inefficiencies.",
            priority=tb.Priority.MEDIUM,
            section_type="analysis"
        ),
        tb.PromptSection(
            content="Format: List critical issues first, then suggestions.",
            priority=tb.Priority.MEDIUM,
            section_type="format"
        ),
        tb.PromptSection(
            content="Example: 'CRITICAL: SQL injection vulnerability on line 42'",
            priority=tb.Priority.LOW,
            section_type="example"
        )
    ]
    ```

    ### Fine-grained ordering with order_hint

    Use order_hint for precise section positioning:

    ```python
    # Create sections with same priority but different order hints
    setup_sections = [
        tb.PromptSection(
            content="You are a helpful assistant.",
            priority=tb.Priority.HIGH,
            section_type="persona",
            order_hint=1  # First
        ),
        tb.PromptSection(
            content="You specialize in Python programming.",
            priority=tb.Priority.HIGH,
            section_type="persona",
            order_hint=2  # Second
        ),
        tb.PromptSection(
            content="You focus on writing clean, efficient code.",
            priority=Priority.HIGH,
            section_type="persona",
            order_hint=3  # Third
        )
    ]
    ```

    ### Building sections for different prompt strategies

    Create sections optimized for specific attention patterns:

    ```python
    # Front-loading critical information (primacy bias)
    critical_first = tb.PromptSection(
        content="IMMEDIATE PRIORITY: Check for buffer overflow vulnerabilities.",
        priority=tb.Priority.CRITICAL,
        section_type="constraint",
        order_hint=1
    )

    # Core task definition
    main_task = tb.PromptSection(
        content="Review this C++ code for security issues and memory management problems.",
        priority=tb.Priority.HIGH,
        section_type="analysis",
        order_hint=10
    )

    # Final emphasis (recency bias)
    final_emphasis = tb.PromptSection(
        content="Remember: Security vulnerabilities are the highest priority.",
        priority=Priority.HIGH,
        section_type="emphasis",
        order_hint=100
    )
    ```

    ### Integration with PromptBuilder workflow

    See how sections work within the larger prompt building process:

    ```python
    # Create a prompt builder
    builder = tb.PromptBuilder()

    # Builder methods create PromptSection objects internally
    builder.persona("senior developer", "code review")
    builder.critical_constraint("Focus on security vulnerabilities")
    builder.core_analysis(["Memory management", "Input validation", "Error handling"])

    # You can also add custom sections directly
    custom_section = tb.PromptSection(
        content="Pay special attention to authentication mechanisms.",
        priority=tb.Priority.HIGH,
        section_type="constraint",
        order_hint=5
    )

    # Add custom section to builder's internal collection
    builder.sections.append(custom_section)

    # Build final prompt (sections are automatically ordered and assembled)
    final_prompt = builder
    print("Final assembled prompt:")
    print(final_prompt)
    ```

    ### Advanced section manipulation

    Perform sophisticated operations on section collections:

    ```python
    # Create a collection of mixed sections
    mixed_sections = [
        tb.PromptSection("Core task", tb.Priority.HIGH, "analysis", 1),
        tb.PromptSection("Important constraint", tb.Priority.HIGH, "constraint", 2),
        tb.PromptSection("Background info", tb.Priority.LOW, "general", 3),
        tb.PromptSection("Critical requirement", tb.Priority.CRITICAL, "constraint", 0),
        tb.PromptSection("Output format", tb.Priority.MEDIUM, "format", 4)
    ]

    # Filter high-priority sections
    high_priority = [s for s in mixed_sections if s.priority.value >= tb.Priority.HIGH.value]

    # Find all constraint sections
    constraints = [s for s in mixed_sections if s.section_type == "constraint"]

    # Get earliest section by order_hint
    earliest = min(mixed_sections, key=lambda s: s.order_hint)

    # Calculate total content length
    total_length = sum(len(s.content) for s in mixed_sections)

    print(f"High priority sections: {len(high_priority)}")
    print(f"Constraint sections: {len(constraints)}")
    print(f"Earliest section: {earliest.content}")
    print(f"Total content length: {total_length} characters")
    ```

    ### Custom section types for specialized workflows

    Define custom section types for specific applications:

    ```python
    # Custom section types for code review workflow
    code_review_sections = [
        tb.PromptSection(
            content="You are a senior code reviewer with 10+ years experience.",
            priority=tb.Priority.CRITICAL,
            section_type="reviewer_persona"
        ),
        tb.PromptSection(
            content="This code will be deployed to production systems.",
            priority=tb.Priority.HIGH,
            section_type="deployment_context"
        ),
        tb.PromptSection(
            content="Check: Security, Performance, Maintainability, Testing",
            priority=tb.Priority.HIGH,
            section_type="review_checklist"
        ),
        tb.PromptSection(
            content="Format: Critical issues first, then improvements, then praise",
            priority=tb.Priority.MEDIUM,
            section_type="response_structure"
        ),
        tb.PromptSection(
            content="Remember: Constructive feedback builds better developers.",
            priority=tb.Priority.MEDIUM,
            section_type="review_philosophy"
        )
    ]

    # Process sections by custom type
    workflow_map = {
        "reviewer_persona": "Sets reviewer identity and expertise",
        "deployment_context": "Provides operational context",
        "review_checklist": "Defines evaluation criteria",
        "response_structure": "Controls output organization",
        "review_philosophy": "Guides feedback tone and approach"
    }

    print("Code review workflow sections:")
    for section in code_review_sections:
        purpose = workflow_map.get(section.section_type, "General purpose")
        print(f"• {section.section_type}: {purpose}")
        print(f"  Content: {section.content}")
        print()
    ```
    """

    content: str
    priority: Priority = Priority.MEDIUM
    section_type: str = "general"
    order_hint: int = 0  # Lower numbers appear earlier


class PromptBuilder:
    """
    Builds structured prompts using attention mechanisms and cognitive principles.

    The `PromptBuilder` leverages insights from modern prompt engineering research to create prompts
    that maximize model attention on critical information while maintaining natural conversation
    flow.

    Returns
    -------
    PromptBuilder
        A new instance ready for fluent method chaining

    Notes
    -----
    This implementation integrates recent research from attention mechanisms and cognitive psychology
    to optimize prompt effectiveness. The `PromptBuilder` applies proven principles that enhance model
    performance and response quality through strategic information placement and cognitive load management.

    **Attention Mechanisms Applied:**

    - **Positional encoding**: critical information placed strategically
    - **Multi-head attention**: different types of context handled separately
    - **Hierarchical structure**: information organized by importance and relevance
    - **Context windowing**: optimal information density for model processing

    **Cognitive Psychology Integration:**

    - **Primacy effect**: important instructions placed early
    - **Recency effect**: final emphasis reinforces key objectives
    - **Chunking**: information grouped into digestible, logical units
    - **Salience**: critical constraints highlighted for maximum attention

    **Prompt Building Methods**

    The `PromptBuilder` provides a comprehensive set of methods for creating structured, attention-optimized prompts.
    All methods support fluent chaining for natural prompt construction:

    **Core Foundation Methods:**

    - `persona(role, expertise=None)`: set the AI's identity and behavioral framework
    - `task_context(context, priority=CRITICAL)`: define the primary objective and scope
    - `critical_constraint(constraint)`: add front-loaded, non-negotiable requirements
    - `constraint(constraint)`: add important but secondary requirements

    **Structure and Analysis Methods:**

    - `structured_section(title, content, priority=MEDIUM, required=False)`: create organized content sections
    - `core_analysis(analysis_points)`: define required analytical focus areas
    - `output_format(format_specs)`: specify response structure and formatting requirements
    - `example(input_example, output_example)`: provide concrete input/output demonstrations

    **Focus and Guidance Methods:**

    - `focus_on(primary_goal)`: emphasize the most important objective
    - `avoid_topics(topics)`: explicitly exclude irrelevant or problematic areas
    - `final_emphasis(emphasis)`: add closing reinforcement using recency bias

    **Output Methods:**

    - `build()`: generate the final structured prompt string
    - `preview_structure()`: preview the prompt organization and metadata

    Each method is designed to work together in the attention-optimized prompt structure,
    with positioning and formatting automatically handled to maximize model performance.

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
    arch_prompt = tb.architectural_analysis_prompt()
    print(arch_prompt)
    ```

    ```{python}
    # Use pre-configured code review
    review_prompt = tb.code_review_prompt()
    print(review_prompt)
    ```
    """

    def __init__(self):
        self._sections: List[PromptSection] = []
        self._persona: Optional[str] = None
        self._task_context: Optional[str] = None
        self._task_priority: Priority = Priority.CRITICAL
        self._constraints: List[str] = []
        self._output_format: List[str] = []
        self._examples: List[Dict[str, str]] = []
        self._final_emphasis: Optional[str] = None

    def persona(self, role: str, expertise: Optional[str] = None) -> "PromptBuilder":
        """
        Set a behavioral persona to anchor the model's response style and establish expertise
        context.

        The persona method establishes the AI's identity and behavioral framework, which serves as
        the foundation for all subsequent interactions. This method leverages behavioral psychology
        principles to create consistent, expert-level responses aligned with the specified role and
        domain expertise.

        **Research Foundation**: Based on research demonstrating that personas significantly improve
        reasoning performance by providing behavioral context and role-specific cognitive
        frameworks. The persona acts as a cognitive anchor that guides response generation, tone,
        and the depth of domain-specific knowledge applied.

        **Prompt Positioning**: The persona is always placed at the beginning of the final prompt
        structure to establish behavioral context before any task-specific instructions. This
        follows the principle of behavioral anchoring, where early identity establishment influences
        all subsequent reasoning patterns.

        **Best Practices**:

        - use specific, professional role titles rather than generic descriptions
        - include relevant experience levels when appropriate (`"senior"`, `"expert"`, `"lead"`,
        etc.)
        - match expertise areas to the expected task complexity
        - consider domain-specific terminology and communication styles

        Parameters
        ----------
        role
            The primary professional role or identity the AI should adopt. This should be
            specific and professional (e.g., "senior software architect", "data scientist",
            "technical writer"). The role influences response style, terminology, and the
            level of technical depth provided.
        expertise
            Specific area of expertise or specialization within the role. This narrows
            the focus and enhances domain-specific knowledge application (e.g.,
            "distributed systems", "machine learning", "API documentation").
            If not provided, the persona will be general within the specified role.

        Returns
        -------
        PromptBuilder
            Self for method chaining, allowing combination with other prompt building
            methods to create comprehensive, structured prompts.

        Examples
        --------
        ### Basic role assignment

        Set a clear professional identity for the AI:

        ```python
        import talk_box as tb

        # Simple role without specific expertise
        builder = (
            tb.PromptBuilder()
            .persona("data analyst")
            .task_context("Analyze customer satisfaction survey results")
        )

        print(builder)
        ```

        ### Role with domain expertise

        Combine role with specific area of expertise:

        ```python
        # Specialized expertise within role
        builder = (
            tb.PromptBuilder()
            .persona("software engineer", "backend API development")
            .task_context("Review the authentication service architecture")
            .core_analysis([
                "Security implementation patterns",
                "Scalability considerations",
                "Error handling strategies"
            ])
        )

        print(builder)
        ```

        ### Senior-level expertise

        Use seniority indicators for complex tasks:

        ```python
        # Senior-level role for complex analysis
        builder = (
            tb.PromptBuilder()
            .persona("senior software architect", "distributed systems")
            .critical_constraint("Focus on production-scale considerations")
            .task_context("Design a microservices architecture for high-traffic e-commerce")
        )
        ```

        ### Domain-specific personas

        Create personas tailored to specific industries or domains:

        ```python
        # Healthcare domain expertise
        healthcare_builder = (
            tb.PromptBuilder()
            .persona("healthcare data analyst", "clinical research")
            .task_context("Analyze patient outcome data for treatment effectiveness")
        )

        # Financial services expertise
        finance_builder = (
            tb.PromptBuilder()
            .persona("quantitative analyst", "risk management")
            .task_context("Evaluate portfolio risk exposure across asset classes")
        )

        # Educational technology expertise
        edtech_builder = (
            tb.PromptBuilder()
            .persona("educational technologist", "learning analytics")
            .task_context("Design metrics for measuring student engagement")
        )
        ```

        ### Combining personas with other prompt elements

        Build comprehensive prompts with persona as the foundation:

        ```python
        # Complete code review prompt with expert persona
        review_prompt = (
            tb.PromptBuilder()
            .persona("senior code reviewer", "security and performance")
            .critical_constraint("Prioritize security vulnerabilities over style issues")
            .task_context("Review this Python Flask application for production readiness")
            .core_analysis([
                "Authentication and authorization implementation",
                "Input validation and sanitization",
                "Database query optimization",
                "Error handling and logging"
            ])
            .output_format([
                "Critical security issues (immediate attention)",
                "Performance bottlenecks (optimization opportunities)",
                "Code quality improvements (maintainability)",
                "Positive patterns (reinforcement)"
            ])
            .final_emphasis("Focus on issues that could impact production security or performance")
        )
        ```

        ### Persona influence on response style

        See how different personas affect response characteristics:

        ```python
        # Technical depth variation
        beginner_persona = (
            tb.PromptBuilder()
            .persona("junior developer")
            .task_context("Explain RESTful API design principles")
        )

        expert_persona = (
            tb.PromptBuilder()
            .persona("principal engineer", "API architecture")
            .task_context("Explain RESTful API design principles")
        )

        # The expert persona will provide more sophisticated insights,
        # advanced patterns, and industry best practices compared to
        # the junior developer persona's more fundamental explanations
        ```

        ### Multiple expertise areas

        Handle roles with multiple specializations:

        ```python
        # Broad expertise combining multiple areas
        fullstack_persona = (
            tb.PromptBuilder()
            .persona("full-stack architect", "web applications and cloud infrastructure")
            .task_context("Design end-to-end solution for real-time collaboration platform")
        )

        # Research-focused persona with interdisciplinary expertise
        research_persona = (
            tb.PromptBuilder()
            .persona("research scientist", "machine learning and cognitive psychology")
            .task_context("Evaluate AI model bias in human-computer interaction contexts")
        )
        ```

        ### Persona consistency across conversations

        Maintain consistent persona behavior in extended interactions:

        ```python
        # Establish consistent technical writing persona
        technical_writer = (
            tb.PromptBuilder()
            .persona("technical documentation specialist", "developer tools")
            .task_context("Create user guide for API integration")
        )

        # The persona will consistently use:
        # - clear, user-focused language
        # - structured, step-by-step explanations
        # - practical examples and code snippets
        # - troubleshooting and best practice guidance
        ```

        Integration Notes
        -----------------
        - **Behavioral Anchoring**: the persona establishes cognitive framework before task instructions
        - **Response Consistency**: maintains consistent voice and expertise level throughout interaction
        - **Domain Knowledge**: activates relevant knowledge domains and professional terminology
        - **Communication Style**: influences formality, technical depth, and explanatory approach
        - **Quality Indicators**: expert personas tend to provide more nuanced, comprehensive responses

        The persona method provides the foundational identity that guides all subsequent AI
        behavior, ensuring responses align with professional expectations and domain expertise
        requirements.
        """
        persona_text = f"You are a {role}"
        if expertise:
            persona_text += f" with expertise in {expertise}"
        persona_text += "."

        self._persona = persona_text
        return self

    def task_context(self, context: str, priority: Priority = Priority.CRITICAL) -> "PromptBuilder":
        """
        Define the primary task context that establishes what needs to be accomplished.

        The task context serves as the central objective that guides the entire prompt. It appears
        prominently in the final prompt structure and provides clear direction for the AI model.
        This method is essential for creating focused, goal-oriented prompts that produce
        relevant and actionable responses.

        **Positioning and Attention**: Task context is typically placed early in the prompt
        structure (after persona and critical constraints) to establish clear expectations.
        The default CRITICAL priority ensures the task receives prominent attention placement.

        **Best Practices**:

        - use clear, specific language that defines measurable outcomes
        - focus on action-oriented descriptions ("analyze", "review", "create")
        - avoid vague or ambiguous task descriptions
        - include scope boundaries when appropriate

        Parameters
        ----------
        context
            Clear, specific description of what needs to be accomplished. Should be
            action-oriented and provide sufficient detail for the AI to understand
            the expected scope and deliverables.
        priority
            Attention priority level for task placement in the final prompt.
            Defaults to `Priority.CRITICAL` to ensure the main task receives
            prominent positioning and maximum attention.

        Returns
        -------
        PromptBuilder
            Self for method chaining, allowing combination with other prompt
            building methods.

        Examples
        --------
        ### Basic task definition

        Set a clear, focused task for the prompt:

        ```python
        import talk_box as tb

        # Simple task context
        builder = (
            tb.PromptBuilder()
            .persona("data analyst")
            .task_context("Analyze the customer churn data to identify key patterns")
        )

        print(builder)
        ```

        ### Task with custom priority

        Use different priority levels for task positioning:

        ```python
        # High priority task (but not critical)
        builder = (
            tb.PromptBuilder()
            .persona("software architect")
            .critical_constraint("Focus only on security vulnerabilities")
            .task_context(
                "Review the authentication system architecture",
                priority=tb.Priority.HIGH
            )
        )
        ```

        ### Detailed task with scope boundaries

        Create comprehensive task descriptions with clear boundaries:

        ```python
        # Detailed task with specific scope
        builder = (
            tb.PromptBuilder()
            .persona("technical writer", "API documentation")
            .task_context(
                "Create comprehensive API documentation for the user management endpoints, "
                "including authentication requirements, request/response examples, "
                "and error handling procedures"
            )
            .core_analysis([
                "Document each endpoint's purpose and functionality",
                "Provide complete request/response schemas",
                "Include practical usage examples"
            ])
        )
        ```
        """
        self._task_context = context
        self._task_priority = priority
        return self

    def critical_constraint(self, constraint: str) -> "PromptBuilder":
        """
        Add a critical constraint that will be front-loaded for maximum attention and impact.

        Critical constraints are the highest-priority requirements that must be prominently
        positioned in the final prompt to ensure maximum model attention and compliance.
        These constraints are automatically placed in the "CRITICAL REQUIREMENTS" section
        immediately after the persona and before the main task, leveraging the primacy effect
        to maximize their influence on response generation.

        **Research Foundation**: based on findings demonstrating that early-positioned instructions
        have the greatest impact on task accuracy and model compliance. The front-loading strategy
        ensures critical requirements receive maximum attention allocation during the model's
        processing phase.

        **Attention Positioning**: critical constraints are placed at the very beginning of the
        constraint hierarchy, appearing before any task context or analysis requirements. This
        strategic positioning leverages cognitive psychology principles where information presented
        early has disproportionate influence on decision-making and response generation.

        **Use Cases**: Critical constraints are ideal for:

        - security and safety requirements that cannot be compromised
        - output format restrictions that must be strictly followed
        - behavioral boundaries that define acceptable response patterns
        - quality thresholds that determine response adequacy
        - time-sensitive or high-stakes operational requirements

        **Constraint Hierarchy**: Multiple critical constraints are ordered by insertion, with the
        first added appearing first in the final prompt. This allows for fine-grained control over
        the relative importance of multiple critical requirements.

        Parameters
        ----------
        constraint
            Specific constraint or requirement that must receive maximum attention. Should be clear,
            actionable, and measurable when possible. Use imperative language for direct instruction
            (e.g., `"Focus only on security vulnerabilities"`, `"Provide exactly 3
            recommendations"`, `"Avoid discussing implementation details"`).

        Returns
        -------
        PromptBuilder
            Self for method chaining, allowing combination with other prompt building methods to
            create comprehensive, structured prompts.

        Examples
        --------
        ### Security-focused critical constraint

        Prioritize security considerations above all else:

        ```python
        import talk_box as tb

        # Security-first code review
        builder = (
            tb.PromptBuilder()
            .persona("senior security engineer", "application security")
            .critical_constraint("Flag any security vulnerabilities immediately")
            .task_context("Review this authentication implementation")
            .core_analysis([
                "Input validation and sanitization",
                "Authentication mechanisms",
                "Authorization controls"
            ])
        )

        print(builder)
        ```

        ### Output format critical constraint

        Enforce strict output formatting requirements:

        ```python
        # Structured response requirement
        builder = (
            tb.PromptBuilder()
            .persona("data analyst", "business intelligence")
            .critical_constraint("Provide exactly 3 key findings with supporting data")
            .task_context("Analyze quarterly sales performance")
            .output_format([
                "Finding 1: [Insight] - [Supporting metric]",
                "Finding 2: [Insight] - [Supporting metric]",
                "Finding 3: [Insight] - [Supporting metric]"
            ])
        )
        ```

        ### Behavioral boundary critical constraint

        Set clear behavioral boundaries for sensitive topics:

        ```python
        # Medical advice boundary
        builder = (
            tb.PromptBuilder()
            .persona("health information specialist")
            .critical_constraint(
                "Do not provide specific medical diagnoses or treatment recommendations"
            )
            .task_context(
                "Explain general wellness concepts and direct to healthcare professionals"
            )
        )
        ```

        ### Quality threshold critical constraint

        Define minimum quality standards for responses:

        ```python
        # Production-ready focus
        builder = (
            tb.PromptBuilder()
            .persona("senior software architect", "enterprise systems")
            .critical_constraint("Focus only on production-ready, scalable solutions")
            .task_context("Design microservices architecture for high-traffic application")
            .core_analysis([
                "Scalability patterns",
                "Fault tolerance mechanisms",
                "Performance optimization strategies"
            ])
        )
        ```

        ### Multiple critical constraints with hierarchy

        Layer multiple critical requirements in order of importance:

        ```python
        # Hierarchical critical constraints
        builder = (
            tb.PromptBuilder()
            .persona("principal engineer", "financial systems")

            # First priority -- Regulatory compliance
            .critical_constraint("Ensure all recommendations comply with financial regulations")

            # Second priority -- Proven solutions
            .critical_constraint("Focus on solutions with proven track records in banking")

            # Third priority -- Security prioritization
            .critical_constraint("Prioritize security over performance optimizations")

            .task_context("Architect payment processing system for online banking")
        )
        ```

        ### Time-sensitive critical constraint

        Handle urgent or time-critical requirements:

        ```python
        # Emergency response scenario
        builder = (
            tb.PromptBuilder()
            .persona("incident response specialist", "system outages")
            .critical_constraint("Provide immediate actionable steps for system recovery")
            .task_context("Diagnose and resolve database connection failures")
            .output_format([
                "Immediate actions (next 5 minutes)",
                "Short-term fixes (next hour)",
                "Long-term prevention (next sprint)"
            ])
        )
        ```

        ### Domain-specific critical constraint

        Apply domain-specific requirements that cannot be compromised:

        ```python
        # Healthcare data processing
        healthcare_builder = (
            tb.PromptBuilder()
            .persona("healthcare data engineer", "HIPAA compliance")
            .critical_constraint("Ensure all recommendations maintain patient data privacy")
            .task_context("Design data pipeline for clinical research")
        )

        # Educational content creation
        education_builder = (
            tb.PromptBuilder()
            .persona("curriculum designer", "K-12 education")
            .critical_constraint("Ensure content is age-appropriate for target grade level")
            .task_context("Create interactive science lesson plan")
        )

        # Financial analysis
        finance_builder = (
            tb.PromptBuilder()
            .persona("risk analyst", "portfolio management")
            .critical_constraint("Include risk disclaimers for all investment recommendations")
            .task_context("Analyze emerging market investment opportunities")
        )
        ```

        ### Combining with other constraint types

        Use critical constraints alongside standard constraints:

        ```python
        # Comprehensive constraint strategy
        builder = (
            tb.PromptBuilder()
            .persona("technical lead", "code quality")
            .critical_constraint("Identify blocking issues that prevent deployment") # Critical
            .task_context("Review pull request for production release")
            .constraint("Consider coding style consistency")                         # Standard
            .constraint("Suggest performance improvements")                          # Standard
            .core_analysis([
                "Security vulnerabilities",
                "Logic errors and edge cases",
                "Integration and compatibility issues"
            ])
        )
        ```

        ### Constraint measurement and validation

        Create measurable constraints for objective evaluation:

        ```python
        # Measurable performance constraint
        builder = (
            tb.PromptBuilder()
            .persona("performance engineer", "web optimization")
            .critical_constraint("All recommendations must target sub-100ms response times")
            .task_context("Optimize API endpoint performance")
        )

        # Quantitative analysis constraint
        analysis_builder = (
            tb.PromptBuilder()
            .persona("data scientist", "statistical analysis")
            .critical_constraint("Include confidence intervals and statistical significance for all findings")
            .task_context("Analyze A/B test results for conversion optimization")
        )
        ```

        Integration Notes
        -----------------
        - **Primacy Effect**: Critical constraints appear early in the prompt for maximum impact
        - **Attention Allocation**: Front-loading ensures these requirements receive priority processing
        - **Constraint Ordering**: Multiple critical constraints maintain insertion order for hierarchical importance
        - **Quality Assurance**: Critical constraints serve as quality gates for response evaluation
        - **Behavioral Anchoring**: Works with persona to establish both identity and non-negotiable requirements

        The critical_constraint method ensures that the most important requirements are positioned
        for maximum attention and compliance, creating a foundation of non-negotiable standards
        that guide all subsequent reasoning and response generation.
        """
        self._constraints.insert(0, constraint)
        return self

    def constraint(self, constraint: str) -> "PromptBuilder":
        """
        Add a standard constraint to the prompt that will appear in the additional constraints
        section.

        Standard constraints are important requirements and guidelines that shape the AI's response
        but are not as critical as front-loaded constraints. These constraints appear in the
        `ADDITIONAL CONSTRAINTS` section after the main task context and structured sections,
        providing important guidance while maintaining the attention hierarchy of the prompt.

        **Positioning Strategy**: standard constraints are positioned after critical constraints
        and core content to maintain optimal attention flow. This positioning ensures that
        essential task information receives primary focus while still communicating important
        requirements and preferences to the model.

        **Use Cases**: standard constraints are ideal for:

        - quality preferences and style guidelines
        - secondary requirements that enhance output quality
        - behavioral preferences that improve response tone
        - technical preferences for implementation approaches
        - context-specific guidelines that refine the response scope

        **Constraint Hierarchy**: standard constraints appear in the order they are added,
        after any critical constraints. This allows for logical grouping of related
        requirements and systematic constraint organization.

        **Relationship to Critical Constraints**: while `critical_constraint()` is used for
        non-negotiable requirements that must be front-loaded, `constraint()` is used for
        important but secondary requirements that guide response quality and style.

        Parameters
        ----------
        constraint
            Specific constraint, requirement, or guideline that should influence the AI's response.
            Should be clear and actionable, using directive language when appropriate (e.g., `"Use
            clear, concise language"`, `"Include practical examples"`, `"Avoid overly technical
            jargon"`).

        Returns
        -------
        PromptBuilder
            Self for method chaining, allowing combination with other prompt building methods to
            create comprehensive, structured prompts.

        Examples
        --------
        ### Quality and style constraints

        Add constraints that improve response quality and consistency:

        ```python
        import talk_box as tb

        # Documentation quality constraints
        builder = (
            tb.PromptBuilder()
            .persona("technical writer", "API documentation")
            .task_context("Create user guide for authentication API")
            .constraint("Use clear, concise language appropriate for developers")
            .constraint("Include practical code examples for each endpoint")
            .constraint("Provide troubleshooting guidance for common issues")
            .core_analysis([
                "Authentication flow and requirements",
                "Error handling and status codes",
                "Rate limiting and best practices"
            ])
        )

        print(builder)
        ```

        ### Technical preference constraints

        Guide implementation approaches and technical choices:

        ```python
        # Architecture review with technical preferences
        builder = (
            tb.PromptBuilder()
            .persona("senior software architect", "microservices")
            .critical_constraint("Focus only on production-ready patterns")
            .task_context("Review microservices architecture design")
            .constraint("Prefer established patterns over novel approaches")
            .constraint("Consider scalability implications for each recommendation")
            .constraint("Include performance trade-offs in analysis")
            .core_analysis([
                "Service decomposition strategy",
                "Inter-service communication patterns",
                "Data consistency approaches"
            ])
        )
        ```

        ### Behavioral and tone constraints

        Shape the AI's communication style and approach:

        ```python
        # Code review with specific behavioral guidance
        builder = (
            tb.PromptBuilder()
            .persona("senior developer", "code quality")
            .task_context("Review pull request for junior developer")
            .constraint("Provide constructive, encouraging feedback")
            .constraint("Explain the reasoning behind each suggestion")
            .constraint("Include positive reinforcement for good practices")
            .constraint("Suggest learning resources for improvement areas")
            .core_analysis([
                "Code correctness and logic",
                "Security considerations",
                "Maintainability and readability"
            ])
        )
        ```

        ### Context-specific constraints

        Add domain or situation-specific requirements:

        ```python
        # Healthcare application constraints
        healthcare_builder = (
            tb.PromptBuilder()
            .persona("healthcare software architect", "HIPAA compliance")
            .critical_constraint("All recommendations must maintain patient privacy")
            .task_context("Design patient data management system")
            .constraint("Consider healthcare industry regulations")
            .constraint("Prioritize data security over performance optimizations")
            .constraint("Include audit trail requirements in recommendations")
        )

        # Educational content constraints
        education_builder = (
            tb.PromptBuilder()
            .persona("curriculum designer", "computer science education")
            .task_context("Create programming exercises for beginners")
            .constraint("Use relatable, real-world examples")
            .constraint("Progress from simple to complex concepts gradually")
            .constraint("Include common mistake explanations")
            .constraint("Provide both guided and independent practice opportunities")
        )
        ```

        ### Multiple related constraints

        Group related constraints for comprehensive guidance:

        ```python
        # Data analysis with multiple quality constraints
        builder = (
            tb.PromptBuilder()
            .persona("data scientist", "business analytics")
            .task_context("Analyze customer behavior patterns")
            .constraint("Support findings with statistical evidence")
            .constraint("Use clear visualizations to illustrate trends")
            .constraint("Explain methodology and assumptions clearly")
            .constraint("Provide actionable business recommendations")
            .constraint("Include confidence levels for predictions")
            .core_analysis([
                "Customer segmentation patterns",
                "Behavioral trend analysis",
                "Predictive modeling opportunities"
            ])
        )
        ```

        ### Combining with critical constraints

        Use standard constraints to complement critical requirements:

        ```python
        # Security analysis with layered constraints
        builder = (
            tb.PromptBuilder()
            .persona("security engineer", "application security")
            .critical_constraint("Identify blocking security vulnerabilities immediately")
            .task_context("Security audit of web application")
            .constraint("Consider OWASP Top 10 guidelines")                    # Standard
            .constraint("Evaluate both code and infrastructure security")     # Standard
            .constraint("Provide remediation priority levels")                # Standard
            .constraint("Include compliance implications where relevant")      # Standard
            .core_analysis([
                "Authentication and authorization",
                "Input validation and sanitization",
                "Data protection and encryption"
            ])
        )
        ```

        ### Output enhancement constraints

        Improve the structure and usability of responses:

        ```python
        # Technical documentation with output quality constraints
        builder = (
            tb.PromptBuilder()
            .persona("technical documentation specialist")
            .task_context("Create troubleshooting guide for deployment issues")
            .constraint("Organize information from most common to least common issues")
            .constraint("Include step-by-step resolution procedures")
            .constraint("Provide prevention strategies for each issue type")
            .constraint("Use consistent formatting and terminology throughout")
            .output_format([
                "Issue description and symptoms",
                "Root cause analysis",
                "Step-by-step resolution",
                "Prevention recommendations"
            ])
        )
        ```

        ### Domain expertise constraints

        Leverage domain-specific knowledge and practices:

        ```python
        # Financial modeling with industry constraints
        finance_builder = (
            tb.PromptBuilder()
            .persona("quantitative analyst", "risk modeling")
            .task_context("Build portfolio risk assessment model")
            .constraint("Follow industry standard risk metrics (VaR, CVaR)")
            .constraint("Include stress testing scenarios")
            .constraint("Provide model validation approaches")
            .constraint("Consider regulatory compliance requirements")
        )

        # Machine learning with best practice constraints
        ml_builder = (
            tb.PromptBuilder()
            .persona("machine learning engineer", "model deployment")
            .task_context("Design ML pipeline for production deployment")
            .constraint("Include data drift monitoring strategies")
            .constraint("Address model explainability requirements")
            .constraint("Consider computational efficiency constraints")
            .constraint("Plan for model versioning and rollback capabilities")
        )
        ```

        ### Constraint measurement and evaluation

        Create constraints that enable objective assessment:

        ```python
        # Performance optimization with measurable constraints
        builder = (
            tb.PromptBuilder()
            .persona("performance engineer", "web applications")
            .task_context("Optimize application response times")
            .constraint("Target specific performance metrics (load time, throughput)")
            .constraint("Include before/after measurement strategies")
            .constraint("Consider mobile and desktop performance separately")
            .constraint("Provide implementation effort estimates")
            .core_analysis([
                "Frontend optimization opportunities",
                "Backend performance bottlenecks",
                "Database query optimization"
            ])
        )
        ```

        ### Collaborative and communication constraints

        Enhance team collaboration and knowledge sharing:

        ```python
        # Code review for team collaboration
        builder = (
            tb.PromptBuilder()
            .persona("tech lead", "team mentorship")
            .task_context("Review code changes for team learning")
            .constraint("Explain best practices for team knowledge sharing")
            .constraint("Suggest pair programming opportunities")
            .constraint("Identify patterns that could be standardized")
            .constraint("Recommend documentation improvements")
            .core_analysis([
                "Code quality and maintainability",
                "Team collaboration opportunities",
                "Knowledge transfer potential"
            ])
        )
        ```

        Integration Notes
        -----------------

        - **Attention Hierarchy**: Standard constraints appear after critical content to maintain focus
        - **Quality Enhancement**: These constraints refine and improve response quality without overriding priorities
        - **Flexibility**: Supports diverse requirement types from technical to behavioral to domain-specific
        - **Systematic Organization**: Constraints are grouped logically in the final prompt structure
        - **Complementary Function**: Works alongside critical constraints to create comprehensive requirement sets

        The constraint method provides flexible, systematic way to communicate important
        requirements and preferences that enhance response quality while respecting the overall
        attention optimization strategy of the prompt building system.
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
        Add a structured section with clear hierarchical boundaries and visual organization.

        Structured sections create distinct attention clusters that prevent attention drift in
        complex prompts by providing clear visual and cognitive boundaries around related content.
        Each section is formatted with an uppercase title and organized content, enabling the AI
        model to process information in logical, digestible chunks while maintaining focus on
        specific aspects of the task.

        **Research Foundation**: creates distinct attention clusters as recommended by Liu et al.
        (2023) for preventing attention drift in complex prompts. The structured approach leverages
        cognitive psychology principles of chunking and visual hierarchy to improve information
        processing and comprehension.

        **Attention Clustering**: structured sections group related information together, creating
        focused attention zones that help the model process complex requirements systematically.
        This prevents attention from being scattered across disconnected information and maintains
        cognitive coherence throughout the prompt.

        **Visual Hierarchy**: each section uses uppercase titles and consistent formatting to
        create clear visual boundaries. This visual organization helps both human readers and
        AI models navigate complex prompts more effectively.

        **Priority-Based Ordering**: sections are automatically ordered by priority and insertion
        order in the final prompt, ensuring that higher-priority content receives appropriate
        attention placement while maintaining logical information flow.

        Parameters
        ----------
        title
            Section heading that will be converted to uppercase for clear visual separation.
            Should be descriptive and specific to the content type (e.g., `"Review Areas"`,
            `"Performance Metrics"`, `"Security Requirements"`). The title helps create mental
            models for information organization.
        content
            Section content provided as either a single string or a list of items. When
            provided as a list, each item is automatically formatted with bullet points
            for clear visual organization. Content should be specific, actionable, and
            relevant to the section's purpose.
        priority
            Attention priority level for section placement in the final prompt structure.
            Higher priority sections appear earlier in the prompt to leverage primacy
            effects. Defaults to `Priority.MEDIUM` for balanced attention allocation.
        required
            Whether to mark the section as required in the output by appending `"(Required)"`
            to the section title. This visual indicator emphasizes critical sections that
            must be addressed in the response. Defaults to `False`.

        Returns
        -------
        PromptBuilder
            Self for method chaining, allowing combination with other prompt building
            methods to create comprehensive, structured prompts.

        Examples
        --------
        ### Basic structured section

        Create a simple section with clear organization:

        ```python
        import talk_box as tb

        # Single-item structured section
        builder = (
            tb.PromptBuilder()
            .persona("software architect", "system design")
            .task_context("Review microservices architecture")
            .structured_section(
                "Architecture Principles",
                "Focus on scalability, maintainability, and fault tolerance"
            )
        )

        print(builder)
        ```

        ### List-based structured section

        Use list format for multiple related items:

        ```python
        # Multi-item structured section
        builder = (
            tb.PromptBuilder()
            .persona("security engineer", "application security")
            .task_context("Conduct security audit of web application")
            .structured_section(
                "Security Focus Areas", [
                    "Authentication and authorization mechanisms",
                    "Input validation and sanitization",
                    "Data encryption and protection",
                    "API security and rate limiting"
                ]
            )
        )
        ```

        ### High-priority required section

        Create critical sections that must be addressed:

        ```python
        # High-priority required section
        builder = (
            tb.PromptBuilder()
            .persona("data scientist", "machine learning")
            .task_context("Evaluate model performance and bias")
            .structured_section(
                "Model Validation", [
                    "Accuracy metrics across demographic groups",
                    "Bias detection and mitigation strategies",
                    "Cross-validation and generalization testing",
                    "Ethical considerations and fairness metrics"
                ],
                priority=tb.Priority.HIGH,
                required=True
            )
        )
        ```

        ### Multiple sections with different priorities

        Build comprehensive prompts with multiple organized sections:

        ```python
        # Complex prompt with multiple structured sections
        builder = (
            tb.PromptBuilder()
            .persona("technical lead", "code review and mentorship")
            .critical_constraint("Focus on production readiness and team learning")
            .task_context("Review pull request for junior developer")
            .structured_section(
                "Code Quality Assessment", [
                    "Logic correctness and edge case handling",
                    "Security vulnerabilities and best practices",
                    "Performance implications and optimizations",
                    "Code readability and maintainability"
                ],
                priority=tb.Priority.HIGH,
                required=True
            )
            .structured_section(
                "Learning Opportunities", [
                    "Design patterns that could be applied",
                    "Best practices worth highlighting",
                    "Areas for skill development",
                    "Recommended learning resources"
                ],
                priority=tb.Priority.MEDIUM
            )
            .structured_section(
                "Team Knowledge Sharing", [
                    "Patterns that could be standardized",
                    "Documentation improvements needed",
                    "Opportunities for pair programming",
                    "Code that exemplifies good practices"
                ],
                priority=tb.Priority.LOW
            )
        )
        ```

        ### Domain-specific structured sections

        Create sections tailored to specific industries or contexts:

        ```python
        # Healthcare application review
        healthcare_builder = (
            tb.PromptBuilder()
            .persona("healthcare software architect", "HIPAA compliance")
            .task_context("Review patient data management system")
            .structured_section(
                "HIPAA Compliance Requirements", [
                    "Patient data encryption and access controls",
                    "Audit trail and logging mechanisms",
                    "Data minimization and retention policies",
                    "Breach detection and notification procedures"
                ],
                priority=tb.Priority.CRITICAL,
                required=True
            )
        )

        # Financial systems analysis
        finance_builder = (
            tb.PromptBuilder()
            .persona("financial systems architect", "regulatory compliance")
            .task_context("Design trading system architecture")
            .structured_section(
                "Regulatory Considerations", [
                    "Market data handling and latency requirements",
                    "Trade reporting and compliance monitoring",
                    "Risk management and circuit breakers",
                    "Audit trails and regulatory reporting"
                ],
                priority=tb.Priority.HIGH,
                required=True
            )
        )
        ```

        ### Performance and optimization sections

        Structure performance-related requirements clearly:

        ```python
        # Performance optimization prompt
        builder = (
            tb.PromptBuilder()
            .persona("performance engineer", "web application optimization")
            .task_context("Optimize application performance for high traffic")
            .structured_section(
                "Performance Targets", [
                    "Page load times under 2 seconds",
                    "API response times under 100ms",
                    "Support for 10,000 concurrent users",
                    "99.9% uptime availability"
                ],
                priority=tb.Priority.HIGH,
                required=True
            )
            .structured_section(
                "Optimization Areas", [
                    "Frontend asset optimization and caching",
                    "Database query performance and indexing",
                    "CDN implementation and edge caching",
                    "Server-side rendering and lazy loading"
                ],
                priority=tb.Priority.MEDIUM
            )
        )
        ```

        ### Educational content sections

        Organize learning objectives and pedagogical structure:

        ```python
        # Educational content design
        builder = (
            tb.PromptBuilder()
            .persona("curriculum designer", "computer science education")
            .task_context("Create comprehensive Python programming course")
            .structured_section(
                "Learning Objectives", [
                    "Understand fundamental programming concepts",
                    "Master Python syntax and data structures",
                    "Apply object-oriented programming principles",
                    "Build practical projects and applications"
                ],
                priority=tb.Priority.HIGH,
                required=True
            )
            .structured_section(
                "Pedagogical Approach", [
                    "Start with hands-on coding exercises",
                    "Progress from simple to complex concepts",
                    "Include real-world project examples",
                    "Provide immediate feedback and correction"
                ],
                priority=tb.Priority.MEDIUM
            )
        )
        ```

        ### Research and analysis sections

        Structure analytical requirements and methodologies:

        ```python
        # Research analysis prompt
        builder = (
            tb.PromptBuilder()
            .persona("research analyst", "market intelligence")
            .task_context("Analyze emerging technology adoption trends")
            .structured_section(
                "Research Methodology", [
                    "Quantitative data analysis and statistical testing",
                    "Qualitative interviews and survey analysis",
                    "Competitive landscape and market mapping",
                    "Trend analysis and future projections"
                ],
                priority=tb.Priority.HIGH,
                required=True
            )
            .structured_section(
                "Deliverable Requirements", [
                    "Executive summary with key findings",
                    "Detailed methodology and data sources",
                    "Visual charts and trend illustrations",
                    "Actionable recommendations and next steps"
                ],
                priority=tb.Priority.MEDIUM,
                required=True
            )
        )
        ```

        ### Quality assurance sections

        Structure testing and validation requirements:

        ```python
        # Quality assurance prompt
        builder = (
            tb.PromptBuilder()
            .persona("QA engineer", "test automation and quality assurance")
            .task_context("Develop comprehensive testing strategy")
            .structured_section(
                "Testing Scope", [
                    "Unit testing for individual components",
                    "Integration testing for system interactions",
                    "End-to-end testing for user workflows",
                    "Performance testing under load conditions"
                ],
                priority=tb.Priority.HIGH,
                required=True
            )
            .structured_section(
                "Quality Metrics", [
                    "Code coverage targets (minimum 80%)",
                    "Test execution time optimization",
                    "Defect detection and resolution rates",
                    "Automation coverage and maintenance"
                ],
                priority=tb.Priority.MEDIUM
            )
        )
        ```

        ### Mixed content types in sections

        Combine different content formats within sections:

        ```python
        # Mixed format content
        builder = (
            tb.PromptBuilder()
            .persona("technical writer", "API documentation")
            .task_context("Create comprehensive API documentation")
            .structured_section(
                "Documentation Standards",
                "Follow OpenAPI 3.0 specification for consistency and completeness"
            )
            .structured_section(
                "Required Documentation Elements", [
                    "Endpoint descriptions with purpose and usage",
                    "Request/response schemas with examples",
                    "Authentication and authorization details",
                    "Error codes and troubleshooting guidance"
                ],
                required=True
            )
        )
        ```

        Integration Notes
        -----------------
        - **Attention Clustering**: Creates focused information zones that prevent cognitive overload
        - **Visual Organization**: Consistent formatting improves prompt readability and navigation
        - **Priority-Based Ordering**: Sections are automatically sorted by priority for optimal attention flow
        - **Flexible Content**: Supports both single-string and list-based content organization
        - **Requirement Emphasis**: Required sections receive visual emphasis to ensure coverage
        - **Cognitive Chunking**: Information is organized in digestible units that align with human processing limits

        The structured_section method provides a powerful tool for organizing complex information
        in attention-optimized ways, enabling the creation of sophisticated prompts that maintain
        clarity and focus while addressing multiple aspects of complex tasks.
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
        Define core analysis requirements as a high-priority, required structured section.

        The core analysis method creates the central analytical framework that defines what specific
        aspects must be examined and addressed in the AI's response. This method automatically
        creates a "CORE ANALYSIS (Required)" section with high priority placement, ensuring that
        the fundamental analytical requirements receive prominent attention and are treated as
        non-negotiable deliverables.

        **Strategic Purpose**: core analysis requirements serve as the analytical backbone of the
        prompt, defining the specific dimensions of investigation that must be covered. Unlike
        general constraints or suggestions, core analysis points are treated as mandatory
        analytical objectives that structure the AI's systematic examination of the subject matter.

        **Attention Priority**: this method automatically assigns `Priority.HIGH` and marks the
        section as required, ensuring that core analysis requirements are prominently positioned
        after critical constraints and task context but before standard constraints and formatting
        requirements. This placement leverages attention optimization principles to ensure
        analytical objectives receive appropriate focus.

        **Analytical Framework**: each analysis point should represent a distinct analytical
        dimension or investigative angle that contributes to comprehensive coverage of the task.
        The points work together to create a systematic analytical framework that guides the AI's
        examination process and ensures thorough, structured analysis.

        **Quality Assurance**: by marking core analysis as required, this method establishes
        analytical accountability and the AI must address each specified analysis point to provide
        a complete response. This prevents superficial analysis and ensures comprehensive coverage
        of critical analytical dimensions.

        Parameters
        ----------
        analysis_points
            List of specific analysis requirements that define the mandatory analytical
            dimensions. Each point should be clear, actionable, and represent a distinct
            aspect of the analysis. Points should be formulated as analytical objectives
            rather than general suggestions (e.g., `"Evaluate security implementation patterns"`
            rather than `"Look at security"`).

        Returns
        -------
        PromptBuilder
            Self for method chaining, allowing combination with other prompt building
            methods to create comprehensive, structured prompts.

        Examples
        --------
        ### Software architecture analysis

        Define core analytical requirements for architectural review:

        ```python
        import talk_box as tb

        # Comprehensive architecture analysis
        builder = (
            tb.PromptBuilder()
            .persona("senior software architect", "enterprise systems")
            .critical_constraint("Focus on production-ready, scalable solutions")
            .task_context("Review microservices architecture for e-commerce platform")
            .core_analysis([
                "Evaluate service decomposition strategy and boundaries",
                "Assess inter-service communication patterns and protocols",
                "Analyze data consistency and transaction management approaches",
                "Review scalability patterns and load distribution mechanisms",
                "Examine security implementation across service boundaries"
            ])
        )

        print(builder)
        ```

        ### Security audit analysis

        Structure mandatory security analysis dimensions:

        ```python
        # Security-focused core analysis
        builder = (
            tb.PromptBuilder()
            .persona("security engineer", "application security")
            .critical_constraint("Prioritize critical vulnerabilities that block deployment")
            .task_context("Conduct comprehensive security audit of web application")
            .core_analysis([
                "Analyze authentication and authorization mechanisms",
                "Evaluate input validation and sanitization practices",
                "Assess data protection and encryption implementations",
                "Review API security and rate limiting strategies",
                "Examine logging, monitoring, and incident response capabilities"
            ])
        )
        ```

        ### Code quality review analysis

        Define analytical framework for code review:

        ```python
        # Code review with systematic analysis
        builder = (
            tb.PromptBuilder()
            .persona("senior developer", "code quality and best practices")
            .task_context("Review pull request for production deployment")
            .core_analysis([
                "Evaluate logic correctness and edge case handling",
                "Assess performance implications and optimization opportunities",
                "Review maintainability and code organization patterns",
                "Analyze test coverage and quality assurance approaches",
                "Examine security considerations and vulnerability patterns"
            ])
            .constraint("Provide constructive feedback with learning opportunities")
            .constraint("Include positive reinforcement for good practices")
        )
        ```

        ### Data science model analysis

        Structure analytical requirements for ML model evaluation:

        ```python
        # Machine learning model analysis
        builder = (
            tb.PromptBuilder()
            .persona("data scientist", "machine learning and model evaluation")
            .critical_constraint("Include bias detection and fairness assessment")
            .task_context("Evaluate machine learning model for production deployment")
            .core_analysis([
                "Assess model accuracy across different demographic groups",
                "Evaluate feature importance and model interpretability",
                "Analyze training data quality and representation",
                "Review model generalization and overfitting indicators",
                "Examine deployment considerations and monitoring requirements"
            ])
        )
        ```

        ### Business process analysis

        Define analytical framework for process improvement:

        ```python
        # Business process optimization analysis
        builder = (
            tb.PromptBuilder()
            .persona("business analyst", "process optimization")
            .task_context("Analyze customer onboarding process for efficiency improvements")
            .core_analysis([
                "Map current process flow and identify bottlenecks",
                "Evaluate customer experience and friction points",
                "Assess resource utilization and cost implications",
                "Analyze compliance and risk management considerations",
                "Identify automation opportunities and technology solutions"
            ])
            .constraint("Support recommendations with quantitative analysis")
            .constraint("Consider both short-term wins and long-term strategy")
        )
        ```

        ### Financial analysis framework

        Structure comprehensive financial evaluation:

        ```python
        # Financial performance analysis
        builder = (
            tb.PromptBuilder()
            .persona("financial analyst", "portfolio and risk management")
            .critical_constraint("Include regulatory compliance considerations")
            .task_context("Analyze investment portfolio performance and risk exposure")
            .core_analysis([
                "Evaluate return performance across asset classes and time periods",
                "Assess risk metrics including VaR, correlation, and concentration",
                "Analyze portfolio diversification and asset allocation effectiveness",
                "Review stress testing results and scenario analysis",
                "Examine liquidity management and cash flow projections"
            ])
        )
        ```

        ### Educational content analysis

        Define analytical framework for curriculum evaluation:

        ```python
        # Educational program analysis
        builder = (
            tb.PromptBuilder()
            .persona("education specialist", "curriculum design and assessment")
            .task_context("Evaluate computer science curriculum for effectiveness")
            .core_analysis([
                "Assess learning objective alignment with industry needs",
                "Evaluate pedagogical approaches and student engagement methods",
                "Analyze student performance data and learning outcomes",
                "Review practical application and project-based learning integration",
                "Examine accessibility and inclusivity considerations"
            ])
            .constraint("Include evidence-based recommendations for improvement")
            .constraint("Consider diverse learning styles and backgrounds")
        )
        ```

        ### Healthcare system analysis

        Structure analytical requirements for healthcare evaluation:

        ```python
        # Healthcare system analysis
        builder = (
            tb.PromptBuilder()
            .persona("healthcare systems analyst", "quality improvement")
            .critical_constraint("Ensure all recommendations maintain patient safety")
            .task_context("Analyze patient care delivery system for quality improvements")
            .core_analysis([
                "Evaluate patient safety indicators and adverse event patterns",
                "Assess care coordination and communication effectiveness",
                "Analyze resource utilization and operational efficiency",
                "Review patient satisfaction and experience metrics",
                "Examine technology integration and workflow optimization"
            ])
        )
        ```

        ### Research methodology analysis

        Define analytical framework for research evaluation:

        ```python
        # Research methodology analysis
        builder = (
            tb.PromptBuilder()
            .persona("research methodologist", "quantitative and qualitative analysis")
            .task_context("Evaluate research study design and methodology")
            .core_analysis([
                "Assess research design appropriateness for stated objectives",
                "Evaluate sampling methodology and representativeness",
                "Analyze data collection methods and measurement validity",
                "Review statistical analysis approaches and assumptions",
                "Examine ethical considerations and bias mitigation strategies"
            ])
            .constraint("Support evaluation with methodological best practices")
            .constraint("Include recommendations for study improvement")
        )
        ```

        ### Product development analysis

        Structure analytical requirements for product evaluation:

        ```python
        # Product development analysis
        builder = (
            tb.PromptBuilder()
            .persona("product manager", "user experience and market analysis")
            .task_context("Analyze new product feature for market readiness")
            .core_analysis([
                "Evaluate user needs alignment and problem-solution fit",
                "Assess market opportunity and competitive landscape",
                "Analyze technical feasibility and implementation complexity",
                "Review user experience design and usability considerations",
                "Examine business model viability and revenue potential"
            ])
            .constraint("Include data-driven insights and user feedback")
            .constraint("Consider both MVP and long-term roadmap implications")
        )
        ```

        ### Infrastructure analysis

        Define analytical framework for infrastructure evaluation:

        ```python
        # Infrastructure analysis
        builder = (
            tb.PromptBuilder()
            .persona("infrastructure architect", "cloud and DevOps")
            .critical_constraint("Prioritize reliability and cost optimization")
            .task_context("Analyze cloud infrastructure for scalability and efficiency")
            .core_analysis([
                "Evaluate current resource utilization and capacity planning",
                "Assess security posture and compliance requirements",
                "Analyze cost optimization opportunities and spending patterns",
                "Review disaster recovery and business continuity preparations",
                "Examine monitoring, alerting, and observability capabilities"
            ])
        )
        ```

        Integration Notes
        -----------------
        - **Analytical Structure**: Creates systematic framework for comprehensive analysis
        - **High Priority Placement**: Automatically positioned prominently in the prompt hierarchy
        - **Required Coverage**: Marked as required to ensure all analytical dimensions are addressed
        - **Quality Assurance**: Establishes analytical accountability and prevents superficial responses
        - **Systematic Investigation**: Guides AI through structured, thorough examination process
        - **Comprehensive Coverage**: Ensures critical analytical aspects are not overlooked

        The core_analysis method provides the analytical backbone for sophisticated prompts,
        ensuring that complex tasks receive systematic, thorough examination across all critical
        dimensions while maintaining focus on the most important analytical objectives.
        """
        return self.structured_section(
            "Core Analysis", analysis_points, priority=Priority.HIGH, required=True
        )

    def output_format(self, format_specs: List[str]) -> "PromptBuilder":
        """
        Specify output formatting requirements to prevent ambiguous responses and ensure structured deliverables.

        Output formatting requirements define the structural and organizational expectations for the AI's
        response, providing clear specifications that prevent ambiguous or inconsistently formatted outputs.
        These requirements appear in the `"OUTPUT FORMAT"` section near the end of the prompt, ensuring
        that formatting guidance influences response generation while maintaining the attention hierarchy
        for more critical content.

        **Research Foundation**: addresses attention drift issues by
        providing specific, measurable formatting constraints that anchor response structure. Clear
        formatting requirements help maintain cognitive coherence and ensure that complex responses
        remain organized and accessible to human readers.

        **Structural Guidance**: output format specifications serve as response templates that guide
        the AI's information organization and presentation. Unlike content-focused constraints, these
        requirements focus on how information should be structured, ordered, and presented to maximize
        clarity and usability.

        **Response Quality**: well-defined formatting requirements significantly improve response
        quality by preventing stream-of-consciousness outputs and ensuring systematic information
        organization. This is particularly important for complex analytical tasks where information
        hierarchy and clear structure are essential for comprehension.

        **Professional Standards**: formatting specifications enable alignment with professional
        documentation standards, report formats, and organizational communication preferences,
        ensuring that AI-generated content meets workplace and industry expectations.

        Parameters
        ----------
        format_specs
            List of specific formatting requirements that define how the response should be
            structured and organized. Each specification should be clear, actionable, and
            measurable when possible. Specifications can address organization, headings,
            lists, examples, priorities, or any structural aspects of the response
            (e.g., `"Start with executive summary"`, `"Use bullet points for key findings"`,
            `"Include code examples for each recommendation"`).

        Returns
        -------
        PromptBuilder
            Self for method chaining, allowing combination with other prompt building
            methods to create comprehensive, structured prompts.

        Examples
        --------
        ### Basic formatting requirements

        Define clear structure for analytical responses:

        ```python
        import talk_box as tb

        # Simple formatting for code review
        builder = (
            tb.PromptBuilder()
            .persona("senior developer", "code review")
            .task_context("Review pull request for production deployment")
            .core_analysis([
                "Security vulnerabilities and risks",
                "Performance implications and optimizations",
                "Code quality and maintainability issues"
            ])
            .output_format([
                "Start with overall assessment (approve/request changes)",
                "List critical issues that must be fixed",
                "Provide suggestions for improvements",
                "Include positive feedback on good practices"
            ])
        )

        print(builder)
        ```

        ### Executive reporting format

        Structure responses for business stakeholders:

        ```python
        # Executive report formatting
        builder = (
            tb.PromptBuilder()
            .persona("business analyst", "strategic planning")
            .task_context("Analyze market expansion opportunity")
            .core_analysis([
                "Market size and growth potential",
                "Competitive landscape analysis",
                "Risk assessment and mitigation strategies",
                "Resource requirements and timeline"
            ])
            .output_format([
                "Executive summary (2-3 key sentences)",
                "Detailed findings with supporting data",
                "Risk assessment with mitigation strategies",
                "Recommended action items with priorities",
                "Timeline and resource requirements"
            ])
        )
        ```

        ### Technical documentation format

        Structure comprehensive technical documentation:

        ```python
        # Technical documentation formatting
        builder = (
            tb.PromptBuilder()
            .persona("technical writer", "API documentation")
            .task_context("Create comprehensive API reference documentation")
            .core_analysis([
                "Endpoint functionality and purpose",
                "Request/response schemas and examples",
                "Authentication and authorization requirements",
                "Error handling and status codes"
            ])
            .output_format([
                "Overview section with API purpose and scope",
                "Authentication section with setup instructions",
                "Endpoint documentation with examples",
                "Error codes reference with troubleshooting",
                "SDK and integration examples"
            ])
        )
        ```

        ### Research and analysis format

        Structure academic or research-style outputs:

        ```python
        # Research analysis formatting
        builder = (
            tb.PromptBuilder()
            .persona("research analyst", "data science")
            .task_context("Analyze customer behavior patterns from survey data")
            .core_analysis([
                "Demographic segmentation and trends",
                "Behavioral pattern identification",
                "Statistical significance of findings",
                "Predictive modeling opportunities"
            ])
            .output_format([
                "Methodology section with data sources and approach",
                "Key findings with statistical evidence",
                "Visual descriptions for charts and graphs",
                "Limitations and confidence intervals",
                "Recommendations with supporting rationale"
            ])
        )
        ```

        ### Prioritized troubleshooting format

        Structure systematic problem-solving responses:

        ```python
        # Troubleshooting guide formatting
        builder = (
            tb.PromptBuilder()
            .persona("systems engineer", "infrastructure troubleshooting")
            .task_context("Diagnose and resolve application performance issues")
            .core_analysis([
                "Performance bottleneck identification",
                "Resource utilization analysis",
                "Configuration and optimization opportunities",
                "Monitoring and alerting improvements"
            ])
            .output_format([
                "Problem summary and impact assessment",
                "Immediate actions (next 15 minutes)",
                "Short-term fixes (next 2 hours)",
                "Long-term optimizations (next sprint)",
                "Prevention strategies and monitoring setup"
            ])
        )
        ```

        ### Educational content format

        Structure learning-focused outputs:

        ```python
        # Educational content formatting
        builder = (
            tb.PromptBuilder()
            .persona("programming instructor", "Python education")
            .task_context("Create comprehensive lesson on object-oriented programming")
            .core_analysis([
                "Core OOP concepts and principles",
                "Practical examples and use cases",
                "Common mistakes and misconceptions",
                "Progressive skill building exercises"
            ])
            .output_format([
                "Learning objectives and prerequisites",
                "Concept explanations with code examples",
                "Hands-on exercises with solutions",
                "Common pitfalls and debugging tips",
                "Additional resources for further learning"
            ])
        )
        ```

        ### Security assessment format

        Structure security analysis outputs:

        ```python
        # Security assessment formatting
        builder = (
            tb.PromptBuilder()
            .persona("security engineer", "application security")
            .critical_constraint("Prioritize critical vulnerabilities that block deployment")
            .task_context("Conduct security audit of web application")
            .core_analysis([
                "Critical security vulnerabilities",
                "Authentication and authorization flaws",
                "Data protection and encryption issues",
                "Infrastructure and configuration weaknesses"
            ])
            .output_format([
                "Executive summary with risk level",
                "Critical issues requiring immediate attention",
                "Medium priority security improvements",
                "Best practice recommendations",
                "Remediation timeline and effort estimates"
            ])
        )
        ```

        ### Comparative analysis format

        Structure side-by-side comparisons:

        ```python
        # Comparative analysis formatting
        builder = (
            tb.PromptBuilder()
            .persona("technology consultant", "solution architecture")
            .task_context("Compare cloud database solutions for enterprise application")
            .core_analysis([
                "Performance characteristics and benchmarks",
                "Cost structure and pricing models",
                "Scalability and availability features",
                "Integration and migration considerations"
            ])
            .output_format([
                "Summary comparison table with key metrics",
                "Detailed analysis for each solution",
                "Pros and cons for specific use cases",
                "Recommendation with rationale",
                "Implementation considerations and timeline"
            ])
        )
        ```

        ### Code review format with examples

        Structure technical review with code samples:

        ```python
        # Code review with examples formatting
        builder = (
            tb.PromptBuilder()
            .persona("senior software engineer", "code quality")
            .task_context("Review Python code for performance and best practices")
            .core_analysis([
                "Code correctness and logic issues",
                "Performance optimization opportunities",
                "Security considerations and vulnerabilities",
                "Maintainability and documentation quality"
            ])
            .output_format([
                "Overall code quality assessment",
                "Critical issues with code examples and fixes",
                "Performance improvements with before/after examples",
                "Style and convention recommendations",
                "Positive patterns worth highlighting"
            ])
        )
        ```

        ### Financial analysis format

        Structure financial reporting outputs:

        ```python
        # Financial analysis formatting
        builder = (
            tb.PromptBuilder()
            .persona("financial analyst", "investment research")
            .task_context("Analyze quarterly performance and investment recommendations")
            .core_analysis([
                "Revenue and profitability trends",
                "Key performance indicator analysis",
                "Risk factors and market conditions",
                "Future outlook and projections"
            ])
            .output_format([
                "Investment recommendation (buy/hold/sell)",
                "Key financial metrics and trends",
                "Risk assessment with quantified impacts",
                "Scenario analysis and sensitivity testing",
                "Target price and timeline projections"
            ])
        )
        ```

        ### Project planning format

        Structure project management outputs:

        ```python
        # Project planning formatting
        builder = (
            tb.PromptBuilder()
            .persona("project manager", "software development")
            .task_context("Create implementation plan for new feature development")
            .core_analysis([
                "Technical requirements and dependencies",
                "Resource allocation and team capacity",
                "Risk factors and mitigation strategies",
                "Timeline and milestone planning"
            ])
            .output_format([
                "Project scope and objectives summary",
                "Phase breakdown with deliverables",
                "Resource requirements and team assignments",
                "Timeline with key milestones and dependencies",
                "Risk register with mitigation plans"
            ])
        )
        ```

        Integration Notes
        -----------------
        - **Response Structure**: Provides clear templates for organized, professional outputs
        - **Cognitive Clarity**: Prevents stream-of-consciousness responses through structured guidance
        - **Quality Assurance**: Ensures consistent formatting that meets professional standards
        - **Information Hierarchy**: Guides appropriate organization of complex information
        - **Accessibility**: Improves readability and navigability of AI-generated content
        - **Professional Alignment**: Enables compliance with organizational communication standards

        The output_format method ensures that AI responses are well-structured, professionally
        formatted, and organized in ways that maximize clarity, usability, and impact for human
        readers across diverse professional contexts.
        """
        self._output_format.extend(format_specs)
        return self

    def example(self, input_example: str, output_example: str) -> "PromptBuilder":
        # fmt: off
        """
Add an input/output example for few-shot learning and response format demonstration.

Examples are a powerful few-shot learning technique that provides concrete demonstrations
of expected input/output patterns, helping the AI understand the desired response format,
style, and level of detail. Examples appear in the `"EXAMPLES"` section near the end of
the prompt, allowing the model to learn from specific demonstrations while maintaining
the attention hierarchy for core content.

**Few-Shot Learning**: examples leverage the AI model's ability to learn from demonstrations
without explicit training. By providing concrete input/output pairs, examples enable the
model to infer patterns, styles, and expected behaviors that might be difficult to specify
through constraints alone.

**Response Calibration**: examples serve as calibration tools that help establish the
appropriate level of detail, technical depth, formatting style, and analytical approach
for responses. This is particularly valuable for complex tasks where abstract descriptions
of requirements might be ambiguous.

**Pattern Recognition**: multiple examples can demonstrate variations in approach, showing
how different types of inputs should be handled while maintaining consistent output quality
and format. This helps the AI generalize appropriately across different scenarios.

**Quality Anchoring**: examples set quality expectations by demonstrating high-quality
responses that serve as benchmarks for the AI's own outputs. This helps maintain
consistency and professionalism across different prompt executions.

Parameters
----------
input_example
    Example input that represents a typical or representative case that the AI might
    encounter. Should be realistic, relevant to the task context, and demonstrate
    the type and complexity of input the AI will be processing. The input should
    be specific enough to provide clear guidance while being generalizable to
    similar scenarios.
output_example
    Expected output format and content that demonstrates the desired response style,
    level of detail, structure, and quality. Should exemplify the formatting
    requirements, analytical depth, and professional standards expected in the
    actual response. The output should be comprehensive enough to serve as a
    template while being specific to the input provided.

Returns
-------
PromptBuilder
    Self for method chaining, allowing combination with other prompt building
    methods to create comprehensive, structured prompts.

Examples
--------
### Code review example

Demonstrate expected code review format and depth:

````python
import talk_box as tb

# Code review with example
builder = (
    tb.PromptBuilder()
    .persona("senior developer", "code quality and security")
    .task_context("Review Python code for security and best practices")
    .core_analysis([
        "Security vulnerabilities and risks",
        "Code quality and maintainability",
        "Performance optimization opportunities",
        "Best practice adherence"
    ])
    .example(
        input_example='''
def authenticate_user(username, password):
    query = "SELECT * FROM users WHERE username = '" + username + "' AND password = '" + password + "'"
    result = db.execute(query)
    return len(result) > 0
        ''',
        output_example='''
**CRITICAL SECURITY ISSUE**: SQL Injection Vulnerability
- **Problem**: Direct string concatenation in SQL query allows SQL injection attacks
- **Risk Level**: High - Could lead to data breach or unauthorized access
- **Fix**: Use parameterized queries or ORM methods
- **Example Fix**:
  ```python
  query = "SELECT * FROM users WHERE username = %s AND password = %s"
  result = db.execute(query, (username, password))
  ```

**SECURITY ISSUE**: Plain Text Password Storage
- **Problem**: Passwords should never be stored or compared in plain text
- **Fix**: Implement password hashing with salt (e.g., bcrypt, scrypt)

**CODE QUALITY**: Function should return user object, not boolean
**PERFORMANCE**: Consider adding database indexes on username field
        '''
    )
    .output_format([
        "Start with critical security issues",
        "Include specific code examples and fixes",
        "Provide risk assessment for each issue",
        "End with positive feedback where applicable"
    ])
)

print(builder)
````

### API documentation example

Show expected documentation format and completeness:

````python
# API documentation with example
builder = (
    tb.PromptBuilder()
    .persona("technical writer", "API documentation")
    .task_context("Create comprehensive API endpoint documentation")
    .core_analysis([
        "Endpoint purpose and functionality",
        "Request/response schemas with examples",
        "Authentication requirements",
        "Error handling and status codes"
    ])
    .example(
        input_example="POST /api/v1/users - Create new user endpoint",
        output_example='''
## POST /api/v1/users

**Purpose**: Create a new user account in the system

**Authentication**: Bearer token required

**Request Schema**:
```json
{
  "username": "string (required, 3-50 chars)",
  "email": "string (required, valid email format)",
  "password": "string (required, min 8 chars)",
  "profile": {
    "firstName": "string (optional)",
    "lastName": "string (optional)"
  }
}
```

**Success Response (201 Created)**:
```json
{
  "id": "12345",
  "username": "johndoe",
  "email": "john@example.com",
  "profile": {
    "firstName": "John",
    "lastName": "Doe"
  },
  "createdAt": "2023-01-15T10:30:00Z"
}
```

**Error Responses**:
- `400 Bad Request`: Invalid input data
- `409 Conflict`: Username or email already exists
- `401 Unauthorized`: Invalid or missing authentication token
        '''
    )
)
````

### Data analysis example

Demonstrate analytical depth and presentation style:

````python
# Data analysis with example
builder = (
    tb.PromptBuilder()
    .persona("data scientist", "business analytics")
    .task_context("Analyze customer behavior data and provide insights")
    .core_analysis([
        "Customer segmentation patterns",
        "Behavioral trends and anomalies",
        "Statistical significance of findings",
        "Business recommendations"
    ])
    .example(
        input_example="Customer purchase data showing 15% increase in mobile transactions but 8% decrease in desktop purchases over Q3",
        output_example='''
**KEY FINDING**: Mobile-First Customer Behavior Shift

**Statistical Analysis**:
- Mobile transactions: +15% (p < 0.01, statistically significant)
- Desktop transactions: -8% (p < 0.05, statistically significant)
- Net transaction volume: +3.2% overall growth

**Customer Segmentation**:
- Millennials (25-40): 67% mobile preference, driving the increase
- Gen X (41-55): 45% mobile adoption, slower transition
- Baby Boomers (55+): 23% mobile usage, prefer desktop

**Business Implications**:
1. **Mobile Optimization Priority**: Invest in mobile UX improvements
2. **Desktop Support**: Maintain desktop experience for older demographics
3. **Cross-Platform Strategy**: Develop unified experience across devices

**Recommendations**:
- Allocate 70% of UX budget to mobile optimization
- Implement responsive design improvements
- Create mobile-specific promotional campaigns
        '''
    )
)
````

### Security audit example

Show comprehensive security assessment format:

````python
# Security audit with example
builder = (
    tb.PromptBuilder()
    .persona("security engineer", "application security")
    .critical_constraint("Prioritize vulnerabilities by CVSS score and exploitability")
    .task_context("Conduct security assessment of web application")
    .core_analysis([
        "Authentication and authorization flaws",
        "Input validation vulnerabilities",
        "Data protection issues",
        "Infrastructure security gaps"
    ])
    .example(
        input_example="Web application with user authentication, file upload, and database integration",
        output_example='''
**CRITICAL (CVSS 9.1)**: Unrestricted File Upload
- **Issue**: Application allows upload of executable files (.php, .jsp)
- **Impact**: Remote code execution, full system compromise
- **Evidence**: Uploaded test.php successfully executed
- **Remediation**: Implement file type whitelist, scan uploads, store outside web root

**HIGH (CVSS 7.8)**: SQL Injection in Search Function
- **Issue**: User input directly concatenated into SQL queries
- **Impact**: Data exfiltration, privilege escalation
- **Evidence**: Union-based injection successful in /search endpoint
- **Remediation**: Use parameterized queries, input validation

**MEDIUM (CVSS 5.4)**: Missing Security Headers
- **Issue**: No HSTS, CSP, or X-Frame-Options headers
- **Impact**: Clickjacking, man-in-the-middle attacks
- **Remediation**: Implement comprehensive security header policy

**Compliance Status**:
- OWASP Top 10: 3 critical violations identified
- Estimated remediation time: 2-3 weeks
        '''
    )
)
````

### Business analysis example

Demonstrate strategic analysis format:

````python
# Business strategy with example
builder = (
    tb.PromptBuilder()
    .persona("business consultant", "strategic planning")
    .task_context("Analyze market expansion opportunity")
    .core_analysis([
        "Market size and growth potential",
        "Competitive landscape assessment",
        "Resource requirements and ROI",
        "Risk factors and mitigation"
    ])
    .example(
        input_example="SaaS company considering expansion into European markets, currently serving 50k US customers",
        output_example='''
**EXECUTIVE SUMMARY**: European expansion presents significant opportunity with estimated 3-year ROI of 180%

**MARKET ANALYSIS**:
- Total Addressable Market: €2.3B (growing 12% annually)
- Immediate Serviceable Market: €180M across UK, Germany, France
- Competitive Gap: Limited players in mid-market segment

**FINANCIAL PROJECTION**:
- Year 1: €2M investment, €800K revenue
- Year 2: €1.2M additional investment, €3.2M revenue
- Year 3: €5.8M revenue, 18% profit margin
- Break-even: Month 14

**KEY SUCCESS FACTORS**:
1. GDPR compliance and data localization
2. Local partnerships for market entry
3. Multi-language product localization
4. European customer support team

**RISK MITIGATION**:
- Regulatory: Early GDPR consultation (€50K budget)
- Competition: Differentiated pricing strategy
- Execution: Phased rollout starting with UK market

**RECOMMENDATION**: Proceed with expansion, starting Q2 2024
        '''
    )
)
````

### Educational content example

Show instructional design and learning format:

````python
# Educational content with example
builder = (
    tb.PromptBuilder()
    .persona("programming instructor", "Python education")
    .task_context("Create lesson on Python data structures")
    .core_analysis([
        "Concept explanation with clear examples",
        "Progressive skill building exercises",
        "Common mistakes and debugging",
        "Real-world applications"
    ])
    .example(
        input_example="Teach Python dictionaries to intermediate beginners",
        output_example='''
**LEARNING OBJECTIVE**: Master Python dictionaries for data organization and lookup

**CONCEPT EXPLANATION**:
Dictionaries store key-value pairs, like a real-world dictionary where you look up words (keys) to find definitions (values).

```python
# Creating a dictionary
student = {
    "name": "Alice",
    "age": 20,
    "major": "Computer Science"
}
```

**HANDS-ON EXERCISE**:
```python
# Task: Create an inventory system
inventory = {"apples": 50, "bananas": 30, "oranges": 25}

# Add new item
inventory["grapes"] = 40

# Update quantity
inventory["apples"] += 10

# Check if item exists
if "mangoes" in inventory:
    print(f"Mangoes: {inventory['mangoes']}")
else:
    print("Mangoes not in stock")
```

**COMMON MISTAKES**:
1. Using mutable objects as keys (lists, other dictionaries)
2. Forgetting that dictionaries are unordered (Python < 3.7)
3. KeyError when accessing non-existent keys

**REAL-WORLD APPLICATION**: User authentication, configuration settings, caching data
        '''
    )
)
````

### Multiple examples for variation

Use multiple examples to show different scenarios:

````python
# Multiple examples for code review
builder = (
    tb.PromptBuilder()
    .persona("senior developer", "code mentorship")
    .task_context("Provide educational code review for junior developers")
    .core_analysis([
        "Code correctness and logic",
        "Best practices and patterns",
        "Performance considerations",
        "Learning opportunities"
    ])
    .example(
        input_example="Simple function with basic logic error",
        output_example="Focus on explaining the logic error clearly with corrected version and learning points"
    )
    .example(
        input_example="Complex function with performance issues",
        output_example="Analyze algorithmic complexity, suggest optimizations, explain trade-offs between readability and performance"
    )
    .example(
        input_example="Well-written code with minor style issues",
        output_example="Acknowledge good practices, suggest minor improvements, reinforce positive patterns"
    )
)
````

Integration Notes
-----------------
- **Few-Shot Learning**: Leverages AI's pattern recognition for improved response quality
- **Format Demonstration**: Shows concrete examples of expected output structure and style
- **Quality Calibration**: Establishes benchmarks for response depth and professionalism
- **Variation Handling**: Multiple examples can demonstrate different scenarios and approaches
- **Learning Reinforcement**: Examples reinforce other prompt elements like constraints and formatting
- **Prompt Positioning**: Examples appear late in prompt to provide final guidance before response generation

The example method provides powerful demonstration-based learning that significantly improves
response quality, consistency, and alignment with expectations through concrete input/output
pattern recognition rather than abstract instruction following.
        """
        # fmt: on
        self._examples.append({"input": input_example, "output": output_example})
        return self

    def final_emphasis(self, emphasis: str) -> "PromptBuilder":
        """
        Set final emphasis that leverages recency bias to ensure critical instructions receive maximum attention.

        Final emphasis strategically positions the most important instruction at the very end of the prompt,
        leveraging the psychological principle of recency bias to ensure that critical guidance remains
        fresh in the AI's attention during response generation. This method provides a powerful way to
        reinforce the most essential requirement or constraint that must not be overlooked.

        **Recency Bias**: research in cognitive psychology demonstrates that information presented at the
        end of a sequence receives heightened attention and retention. By placing critical instructions
        at the prompt's conclusion, final emphasis ensures that the most important guidance influences
        the AI's response generation process when attention is most focused on producing output.

        **Attention Anchoring**: final emphasis serves as an attention anchor that prevents drift from
        core objectives during complex prompt processing. When prompts contain extensive context,
        constraints, and examples, the final emphasis acts as a cognitive reset that refocuses attention
        on the primary objective before response generation begins.

        **Override Mechanism**: final emphasis can serve as an override mechanism for complex prompts
        where multiple competing priorities might create confusion. By explicitly stating the most
        critical requirement at the end, this method ensures that primary objectives take precedence
        over secondary considerations when trade-offs must be made.

        **Quality Assurance**: the strategic placement of final emphasis helps prevent AI responses
        that technically satisfy prompt requirements but miss the primary intent. This is particularly
        valuable for complex analytical tasks where technical completeness might overshadow the
        core objective.

        Parameters
        ----------
        emphasis
            The most critical instruction or objective that must receive primary attention
            during response generation. Should be formulated as a clear, actionable directive
            that captures the essential requirement (e.g., `"Focus your entire response on
            practical implementation steps"`, `"Prioritize security considerations above all else"`,
            `"Ensure all recommendations are cost-effective and implementable"`).

        Returns
        -------
        PromptBuilder
            Self for method chaining, allowing combination with other prompt building
            methods to create comprehensive, structured prompts.

        Examples
        --------
        ### Security-focused analysis

        Ensure security remains the primary consideration despite other requirements:

        ```python
        import talk_box as tb

        # Security audit with final emphasis
        builder = (
            tb.PromptBuilder()
            .persona("security engineer", "application security")
            .task_context("Review web application for deployment readiness")
            .core_analysis([
                "Authentication and authorization mechanisms",
                "Input validation and data sanitization",
                "Infrastructure security configuration",
                "Compliance with security standards"
            ])
            .constraint("Include performance optimization suggestions")
            .constraint("Consider user experience implications")
            .output_format([
                "Executive summary with risk assessment",
                "Critical security issues requiring immediate attention",
                "Performance and UX recommendations where applicable"
            ])
            .final_emphasis("Security vulnerabilities must be identified and addressed before any performance or UX considerations")
        )

        print(builder)
        ```

        ### Cost-conscious recommendations

        Emphasize budget constraints in business analysis:

        ```python
        # Business strategy with cost emphasis
        builder = (
            tb.PromptBuilder()
            .persona("business consultant", "strategic planning")
            .task_context("Develop growth strategy for startup with limited funding")
            .core_analysis([
                "Market opportunity assessment",
                "Competitive landscape analysis",
                "Resource requirements and scaling plan",
                "Revenue generation strategies"
            ])
            .constraint("Include innovative growth tactics")
            .constraint("Consider partnership opportunities")
            .output_format([
                "Executive summary with growth potential",
                "Detailed strategy with implementation phases",
                "Resource allocation and timeline"
            ])
            .final_emphasis("All recommendations must be implementable with minimal upfront investment and show clear ROI within 6 months")
        )
        ```

        ### Learning-focused code review

        Prioritize educational value in technical feedback:

        ```python
        # Code review with learning emphasis
        builder = (
            tb.PromptBuilder()
            .persona("senior developer", "mentorship and code quality")
            .task_context("Review junior developer's code for learning and improvement")
            .core_analysis([
                "Code correctness and functionality",
                "Best practices and design patterns",
                "Performance optimization opportunities",
                "Security considerations"
            ])
            .constraint("Identify areas for improvement")
            .constraint("Provide specific examples and fixes")
            .output_format([
                "Overall assessment with learning objectives",
                "Technical issues with explanations and solutions",
                "Positive reinforcement for good practices"
            ])
            .final_emphasis("Frame all feedback as learning opportunities with clear explanations of why changes improve the code")
        )
        ```

        ### User experience priority

        Ensure UX considerations override technical preferences:

        ```python
        # Product feature analysis with UX emphasis
        builder = (
            tb.PromptBuilder()
            .persona("product manager", "user experience and design")
            .task_context("Evaluate new feature proposal for mobile application")
            .core_analysis([
                "User needs and problem-solution fit",
                "Technical implementation complexity",
                "Performance and scalability impact",
                "Business value and metrics"
            ])
            .constraint("Consider technical feasibility constraints")
            .constraint("Include development effort estimates")
            .output_format([
                "Feature assessment with user impact analysis",
                "Implementation recommendations",
                "Success metrics and validation plan"
            ])
            .final_emphasis("User experience and accessibility must be prioritized over technical convenience or development speed")
        )
        ```

        ### Compliance-focused audit

        Ensure regulatory requirements take precedence:

        ```python
        # Financial audit with compliance emphasis
        builder = (
            tb.PromptBuilder()
            .persona("compliance officer", "financial regulations")
            .task_context("Audit trading platform for regulatory compliance")
            .core_analysis([
                "Know Your Customer (KYC) procedures",
                "Anti-Money Laundering (AML) controls",
                "Transaction monitoring and reporting",
                "Data protection and privacy measures"
            ])
            .constraint("Include efficiency improvement suggestions")
            .constraint("Consider user experience impact")
            .output_format([
                "Compliance status summary",
                "Critical violations requiring immediate attention",
                "Process improvement recommendations"
            ])
            .final_emphasis("Regulatory compliance violations must be flagged as blocking issues regardless of operational impact or user friction")
        )
        ```

        ### Academic rigor emphasis

        Prioritize methodological soundness in research analysis:

        ```python
        # Research methodology review with academic emphasis
        builder = (
            tb.PromptBuilder()
            .persona("research methodologist", "quantitative analysis")
            .task_context("Evaluate research study design and statistical approach")
            .core_analysis([
                "Sample size and power analysis",
                "Statistical methodology appropriateness",
                "Bias mitigation and control variables",
                "Validity and reliability measures"
            ])
            .constraint("Consider practical implementation constraints")
            .constraint("Include suggestions for improvement")
            .output_format([
                "Methodological assessment summary",
                "Statistical validity evaluation",
                "Recommendations for strengthening the study"
            ])
            .final_emphasis("Academic rigor and statistical validity must not be compromised for practical convenience or timeline pressures")
        )
        ```

        ### Scalability priority

        Emphasize long-term architectural considerations:

        ```python
        # Architecture review with scalability emphasis
        builder = (
            tb.PromptBuilder()
            .persona("solution architect", "enterprise systems")
            .task_context("Review microservices architecture for e-commerce platform")
            .core_analysis([
                "Service decomposition and boundaries",
                "Data consistency and transaction management",
                "Performance and load distribution",
                "Security and monitoring capabilities"
            ])
            .constraint("Consider current team capabilities")
            .constraint("Include migration strategy from current system")
            .output_format([
                "Architecture assessment with scalability analysis",
                "Implementation roadmap with phases",
                "Risk assessment and mitigation strategies"
            ])
            .final_emphasis("All architectural decisions must prioritize long-term scalability and maintainability over short-term development convenience")
        )
        ```

        ### Practical implementation focus

        Ensure recommendations are actionable and realistic:

        ```python
        # Strategic planning with implementation emphasis
        builder = (
            tb.PromptBuilder()
            .persona("operations manager", "process improvement")
            .task_context("Develop operational efficiency improvement plan")
            .core_analysis([
                "Current process bottlenecks and inefficiencies",
                "Technology automation opportunities",
                "Staff training and capability requirements",
                "Cost-benefit analysis of improvements"
            ])
            .constraint("Include theoretical best practices")
            .constraint("Consider industry benchmarks")
            .output_format([
                "Current state assessment",
                "Improvement recommendations with priorities",
                "Implementation timeline and resource requirements"
            ])
            .final_emphasis("Every recommendation must include specific, actionable steps that can be implemented with existing resources within 90 days")
        )
        ```

        ### Quality over quantity emphasis

        Prioritize depth and thoroughness over breadth:

        ```python
        # Content analysis with quality emphasis
        builder = (
            tb.PromptBuilder()
            .persona("content strategist", "editorial quality")
            .task_context("Evaluate content library for quality and effectiveness")
            .core_analysis([
                "Content accuracy and factual verification",
                "Engagement metrics and user feedback",
                "SEO optimization and discoverability",
                "Brand consistency and messaging alignment"
            ])
            .constraint("Include competitive analysis")
            .constraint("Consider content volume requirements")
            .output_format([
                "Content quality assessment",
                "Priority improvement areas",
                "Content strategy recommendations"
            ])
            .final_emphasis("Focus on identifying and improving the highest-impact content pieces rather than addressing all content issues superficially")
        )
        ```

        ### Innovation balance

        Emphasize balanced approach between innovation and stability:

        ```python
        # Technology evaluation with balance emphasis
        builder = (
            tb.PromptBuilder()
            .persona("technology director", "innovation and stability")
            .task_context("Evaluate emerging technologies for enterprise adoption")
            .core_analysis([
                "Technology maturity and stability assessment",
                "Integration complexity and risks",
                "Competitive advantages and differentiation",
                "Team learning curve and adoption timeline"
            ])
            .constraint("Include cutting-edge technology options")
            .constraint("Consider conservative enterprise requirements")
            .output_format([
                "Technology assessment matrix",
                "Adoption recommendations by priority",
                "Risk mitigation and implementation strategy"
            ])
            .final_emphasis("Balance innovation opportunities with enterprise stability requirements, prioritizing technologies that provide significant value with manageable risk")
        )
        ```

        Integration Notes
        -----------------
        - **Recency Bias Leverage**: Strategically positions critical guidance at prompt conclusion for maximum impact
        - **Attention Anchoring**: Prevents objective drift during complex prompt processing
        - **Priority Override**: Ensures primary objectives take precedence when trade-offs are required
        - **Quality Assurance**: Prevents technically complete but intent-missing responses
        - **Cognitive Reset**: Refocuses attention on core objectives before response generation
        - **Strategic Positioning**: Complements front-loaded critical constraints with end-positioned emphasis

        The final_emphasis method provides a powerful attention management tool that ensures the most
        critical requirements maintain prominence throughout the AI's response generation process,
        leveraging psychological principles to maximize adherence to primary objectives.
        """
        self._final_emphasis = emphasis
        return self

    def avoid_topics(self, topics: List[str]) -> "PromptBuilder":
        """
        Specify topics or behaviors to avoid through negative constraints that guide AI responses
        away from unwanted content.

        Negative constraints provide explicit guidance about what the AI should not include or discuss in
        its response, creating clear boundaries that prevent unwanted content, inappropriate suggestions,
        or off-topic discussions. This method adds an "Avoid:" constraint that appears in the standard
        constraints section, providing clear guidance about prohibited topics or approaches.

        **Negative Guidance**: research in cognitive psychology shows that explicit negative instructions
        can be effective when combined with positive guidance. By clearly stating what to avoid, this
        method helps the AI navigate complex topics while staying within appropriate boundaries and
        maintaining focus on desired outcomes.

        **Boundary Setting**: avoid topics serves as a content filter and boundary-setting mechanism
        that prevents responses from venturing into sensitive, irrelevant, or counterproductive areas.
        This is particularly valuable for professional contexts where certain topics or approaches
        could be inappropriate or harmful.

        **Risk Mitigation**: negative constraints help mitigate risks associated with AI-generated
        content by explicitly excluding potentially problematic topics, biased perspectives, or
        approaches that could lead to harmful or inappropriate recommendations.

        **Focus Enhancement**: by eliminating distracting or irrelevant topics, `avoid_topics()`
        helps maintain laser focus on the core objectives and prevents the AI from exploring
        tangential areas that might dilute the quality or relevance of the response.

        Parameters
        ----------
        topics
            List of specific topics, behaviors, approaches, or content areas that should be
            explicitly avoided in the response. Each item should be clearly defined and
            specific enough to provide clear guidance (e.g., "controversial political opinions",
            "deprecated technologies", "cost-cutting through layoffs", "quick fixes without
            testing").

        Returns
        -------
        PromptBuilder
            Self for method chaining, allowing combination with other prompt building
            methods to create comprehensive, structured prompts.

        Examples
        --------
        ### Technical architecture review

        Avoid outdated or problematic technologies:

        ```python
        import talk_box as tb

        # Architecture review avoiding deprecated approaches
        builder = (
            tb.PromptBuilder()
            .persona("solution architect", "modern enterprise systems")
            .task_context("Design scalable microservices architecture for e-commerce platform")
            .core_analysis([
                "Service decomposition strategy",
                "Inter-service communication patterns",
                "Data consistency approaches",
                "Scalability and performance optimization"
            ])
            .avoid_topics([
                "Monolithic architecture patterns",
                "Deprecated Java EE technologies",
                "Synchronous blocking communication",
                "Database shared between services",
                "Manual deployment processes"
            ])
            .output_format([
                "Architecture overview with service boundaries",
                "Technology stack recommendations",
                "Implementation roadmap with phases"
            ])
        )

        print(builder)
        ```

        ### Business strategy consultation

        Avoid ethically questionable or short-term approaches:

        ```python
        # Business strategy avoiding problematic tactics
        builder = (
            tb.PromptBuilder()
            .persona("business consultant", "sustainable growth strategies")
            .task_context("Develop growth strategy for struggling retail company")
            .core_analysis([
                "Market positioning and competitive advantages",
                "Operational efficiency improvements",
                "Customer experience enhancements",
                "Revenue diversification opportunities"
            ])
            .avoid_topics([
                "Mass layoffs as primary cost reduction",
                "Exploiting regulatory loopholes",
                "Aggressive customer data monetization",
                "Environmental impact trade-offs for profit",
                "Anti-competitive pricing strategies"
            ])
            .constraint("Focus on sustainable, long-term solutions")
            .output_format([
                "Strategic assessment with market analysis",
                "Growth initiatives with ethical considerations",
                "Implementation timeline with stakeholder impact"
            ])
        )
        ```

        ### Security audit guidance

        Avoid security through obscurity and weak practices:

        ```python
        # Security audit avoiding poor practices
        builder = (
            tb.PromptBuilder()
            .persona("security engineer", "application security best practices")
            .task_context("Audit web application security for financial services company")
            .core_analysis([
                "Authentication and authorization mechanisms",
                "Data protection and encryption standards",
                "Input validation and sanitization",
                "Infrastructure security configuration"
            ])
            .avoid_topics([
                "Security through obscurity approaches",
                "Custom cryptographic implementations",
                "Storing passwords in plain text or weak hashing",
                "Disabling security features for convenience",
                "Ignoring OWASP recommendations"
            ])
            .critical_constraint("All recommendations must follow industry security standards")
            .output_format([
                "Security assessment with risk levels",
                "Critical vulnerabilities requiring immediate attention",
                "Best practice implementation roadmap"
            ])
        )
        ```

        ### Educational content development

        Avoid outdated or confusing learning approaches:

        ```python
        # Educational content avoiding poor pedagogical practices
        builder = (
            tb.PromptBuilder()
            .persona("instructional designer", "modern programming education")
            .task_context("Create comprehensive Python programming curriculum for beginners")
            .core_analysis([
                "Progressive skill building sequence",
                "Hands-on practice opportunities",
                "Real-world application examples",
                "Common mistake prevention strategies"
            ])
            .avoid_topics([
                "Memorization-based learning without understanding",
                "Outdated Python 2.x syntax and practices",
                "Complex theoretical concepts before practical foundation",
                "Overwhelming students with too many options",
                "Abstract examples without real-world relevance"
            ])
            .constraint("Include diverse learning styles and accessibility considerations")
            .output_format([
                "Curriculum structure with learning objectives",
                "Module breakdown with practical exercises",
                "Assessment strategies and progress tracking"
            ])
        )
        ```

        ### Product management guidance

        Avoid feature creep and user-hostile practices:

        ```python
        # Product strategy avoiding common pitfalls
        builder = (
            tb.PromptBuilder()
            .persona("product manager", "user-centered design")
            .task_context("Prioritize feature roadmap for mobile productivity application")
            .core_analysis([
                "User needs analysis and pain points",
                "Competitive landscape assessment",
                "Technical feasibility and resource requirements",
                "Business impact and revenue potential"
            ])
            .avoid_topics([
                "Feature additions without user validation",
                "Dark patterns to increase engagement artificially",
                "Copying competitor features without strategic purpose",
                "Technical debt accumulation for speed",
                "Ignoring accessibility requirements"
            ])
            .constraint("Prioritize user value and experience quality")
            .output_format([
                "Feature prioritization matrix with user impact",
                "Roadmap timeline with development phases",
                "Success metrics and validation plans"
            ])
        )
        ```

        ### Financial investment analysis

        Avoid high-risk or unethical investment approaches:

        ```python
        # Investment analysis avoiding risky strategies
        builder = (
            tb.PromptBuilder()
            .persona("financial advisor", "responsible investment strategies")
            .task_context("Develop investment portfolio strategy for retirement planning")
            .core_analysis([
                "Risk tolerance assessment and diversification",
                "Long-term growth potential analysis",
                "Market volatility considerations",
                "Tax efficiency optimization"
            ])
            .avoid_topics([
                "High-risk speculative investments as primary strategy",
                "Market timing strategies without evidence",
                "Single-sector concentration for growth",
                "Ignoring client risk tolerance for higher returns",
                "Investments in companies with poor ESG practices"
            ])
            .constraint("Focus on evidence-based investment principles")
            .output_format([
                "Portfolio allocation with risk assessment",
                "Investment timeline with rebalancing strategy",
                "Performance expectations and scenario analysis"
            ])
        )
        ```

        ### Code review guidance

        Avoid problematic coding practices and shortcuts:

        ```python
        # Code review avoiding poor development practices
        builder = (
            tb.PromptBuilder()
            .persona("senior developer", "code quality and best practices")
            .task_context("Review pull request for production deployment")
            .core_analysis([
                "Code correctness and functionality",
                "Security vulnerability assessment",
                "Performance implications and optimization",
                "Maintainability and documentation quality"
            ])
            .avoid_topics([
                "Quick fixes that introduce technical debt",
                "Skipping unit tests for faster delivery",
                "Hard-coding configuration values",
                "Ignoring error handling for edge cases",
                "Copy-pasting code without understanding"
            ])
            .constraint("Provide constructive feedback with learning opportunities")
            .output_format([
                "Code quality assessment with specific examples",
                "Security and performance concerns",
                "Improvement recommendations with rationale"
            ])
        )
        ```

        Integration Notes
        -----------------
        - **Boundary Setting**: establishes clear content and approach boundaries for AI responses
        - **Risk Mitigation**: prevents problematic or inappropriate content through explicit exclusion
        - **Focus Enhancement**: eliminates distracting topics to maintain response relevance
        - **Professional Standards**: ensures responses align with ethical and professional guidelines
        - **Quality Assurance**: prevents low-quality approaches through negative guidance
        - **Complementary Constraints**: works alongside positive constraints to create comprehensive guidance

        The `avoid_topics()` method provides essential boundary-setting capabilities that ensure AI
        responses remain appropriate, focused, and aligned with professional standards while
        explicitly excluding problematic approaches or content areas that could compromise response
        quality or appropriateness.
        """
        # Create strong refusal language instead of weak "avoid" guidance
        if len(topics) == 1:
            refusal_text = (
                f"IMPORTANT CONSTRAINT: You MUST NOT provide any information, advice, or discussion about {topics[0]}. "
                f"If asked about {topics[0]}, politely decline and say: "
                f"'I'm not able to help with {topics[0]}. Is there something else I can assist you with instead?'"
            )
        else:
            topics_list = ", ".join(topics[:-1]) + f", or {topics[-1]}"
            refusal_text = (
                f"IMPORTANT CONSTRAINT: You MUST NOT provide any information, advice, or discussion about {topics_list}. "
                f"If asked about any of these topics, politely decline and say: "
                f"'I'm not able to help with that topic. Is there something else I can assist you with instead?'"
            )

        return self.constraint(refusal_text)

    def pathways(self, pathway_spec) -> "PromptBuilder":
        """
        Add conversational pathway guidance to structure and guide conversation flow.

        Pathways provide flexible conversation flow guidance that helps AI assistants navigate
        complex interactions while maintaining natural conversation patterns. Unlike rigid state
        machines, pathways serve as intelligent guardrails that adapt to user behavior while
        ensuring important steps and information gathering requirements are addressed.

        **Flexible Structure**: Pathways provide conversation guidance without enforcing rigid
        adherence, allowing the AI to adapt to natural conversation patterns while ensuring key
        objectives are met. This balances structure with conversational flexibility.

        **Attention Optimization**: Pathway specifications are integrated into the prompt structure
        at an optimal position for AI attention, providing clear guidance without overwhelming
        other prompt components.

        **Information Management**: Pathways help ensure systematic information gathering and
        step completion while maintaining user-friendly interactions. This is particularly
        valuable for complex processes like bookings, troubleshooting, or guided workflows.

        Parameters
        ----------
        pathway_spec
            A Pathways object created using the chainable Pathways API, or a dictionary
            specification containing pathway definition. The specification includes states,
            transitions, information requirements, and flow control logic.

        Returns
        -------
        PromptBuilder
            Self for method chaining, allowing combination with other prompt building
            methods to create comprehensive, structured prompts.

        Examples
        --------
        ### Customer support pathway

        Create a structured support flow:

        ```python
        import talk_box as tb

        # Define support pathway
        support_pathway = (
            tb.Pathways("Technical Support")
            .description("Systematic technical problem resolution")
            .activation_conditions([
                "User reports technical issues",
                "User needs troubleshooting help"
            ])
            .start_with("problem_identification")
                .chat_state("problem_identification")
                .description("Understand the technical problem")
                .collect(["issue_description", "error_messages", "recent_changes"])
                .next_state("basic_diagnostics")
            .then("basic_diagnostics")
                .decision_state("basic_diagnostics")
                .description("Determine if basic fixes might work")
                .branch_on("Simple configuration issue", "quick_fix")
                .branch_on("Complex system problem", "advanced_diagnostics")
            .then("quick_fix")
                .chat_state("quick_fix")
                .description("Provide immediate solution steps")
                .success_condition("Problem is resolved")
                .fallback("Problem persists", "advanced_diagnostics")

        )

        # Use in prompt
        prompt = (
            tb.PromptBuilder()
            .persona("technical support specialist", "troubleshooting")
            .pathways(support_pathway)
            .final_emphasis("Follow pathway while adapting to user needs")

        )
        ```

        ### Booking flow pathway

        Guide users through complex booking process:

        ```python
        # Flight booking pathway
        booking_pathway = (
            tb.Pathways("Flight Booking")
            .description("Guide users through booking a flight")
            .activation_conditions([
                "User wants to book a flight",
                "User asks about flight reservations"
            ])
            .start_with("greeting")
                .chat_state("greeting")
                .description("Welcome user and understand their travel needs")
                .collect(["departure city", "destination", "travel dates"])
                .next_state("search_flights")
            .then("search_flights")
                .tool_state("search_flights")
                .description("Search available flights")
                .tools(["flight_search_api"])
                .next_state("present_options")
            .then("present_options")
                .chat_state("present_options")
                .description("Show flight options to user")
                .success_condition("User selects a flight option")
                .next_state("booking_confirmation")

        )

        # Integration with bot
        bot = (
            tb.ChatBot()
            .provider_model("openai:gpt-4-turbo")
            .system_prompt(
                tb.PromptBuilder()
                .persona("travel agent", "flight booking specialist")
                .pathways(booking_pathway)
                .output_format([
                    "Be clear about next steps",
                    "Confirm information before proceeding",
                    "Provide helpful alternatives when needed"
                ])

            )
        )
        ```

        Integration Notes
        -----------------
        - **Flexible Guidance**: Pathways provide structure without rigidity, allowing natural conversation flow
        - **Information Gathering**: Systematic collection of required information while maintaining user experience
        - **Adaptive Branching**: Support for conditional flows based on user responses and circumstances
        - **Tool Integration**: Clear guidance on when and how to use external tools within the conversation flow
        - **Completion Tracking**: Built-in success conditions and completion criteria for complex processes

        The pathways method enables sophisticated conversation flow management while preserving the
        natural, adaptive qualities that make AI conversations engaging and user-friendly.
        """
        # Handle both Pathways objects and dictionary specifications
        if hasattr(pathway_spec, "_to_prompt_text"):
            pathway_text = pathway_spec._to_prompt_text()
        elif hasattr(pathway_spec, "_build"):
            # If it has a build method but no _to_prompt_text, it might be a built spec
            built_spec = pathway_spec._build()
            pathway_text = self._format_pathway_spec(built_spec)
        elif isinstance(pathway_spec, dict):
            pathway_text = self._format_pathway_spec(pathway_spec)
        else:
            raise ValueError("pathway_spec must be a Pathways object or dictionary specification")

        # Add as a high-priority structured section
        return self.structured_section(
            title="Conversational Pathway",
            content=pathway_text,
            priority=Priority.HIGH,
            required=True,
        )

    def _format_pathway_spec(self, spec: dict) -> str:
        """Format a pathway specification dictionary into prompt text."""
        lines = []

        # Title and description
        lines.append(f"**{spec.get('title', 'Conversation Flow')}**")
        if spec.get("description"):
            lines.append(f"Purpose: {spec['description']}")

        # Activation conditions
        if spec.get("activation_conditions"):
            lines.append("Activate when:")
            for condition in spec["activation_conditions"]:
                lines.append(f"- {condition}")

        # States and flow
        if spec.get("states"):
            lines.append("Flow guidance:")
            for state_name, state in spec["states"].items():
                lines.append(
                    f"- {state_name.upper()} ({state.get('type', 'chat')}): {state.get('description', '')}"
                )

                if state.get("required_info"):
                    lines.append(f"  Required: {', '.join(state['required_info'])}")

                if state.get("tools"):
                    lines.append(f"  Tools: {', '.join(state['tools'])}")

        # Completion and guidance
        if spec.get("completion_criteria"):
            lines.append(f"Complete when: {'; '.join(spec['completion_criteria'])}")

        lines.append(
            "Follow as flexible guidance, adapting to user conversation patterns while ensuring key objectives are addressed."
        )

        return "\n".join(lines)

    def focus_on(self, primary_goal: str) -> "PromptBuilder":
        """
        Set the primary focus that leverages both front-loading and recency bias for maximum attention impact.

        Focus_on is a powerful dual-positioning method that ensures the most critical objective receives
        maximum attention throughout the prompt by strategically placing it both at the beginning (as a
        critical constraint) and at the end (as final emphasis). This dual-anchor approach leverages
        both primacy and recency effects to create the strongest possible attention focus on the primary
        objective.

        **Dual Attention Strategy**: research in cognitive psychology demonstrates that information
        positioned at both the beginning and end of a sequence receives the highest attention and
        retention. By anchoring the primary goal at both positions, focus_on ensures that the most
        critical objective maintains prominence throughout the entire prompt processing sequence.

        **Primacy and Recency Effects**: the method capitalizes on both primacy bias (heightened
        attention to early information) and recency bias (heightened attention to final information)
        to create a reinforcing attention pattern that keeps the primary objective at the forefront
        of the AI's processing throughout response generation.

        **Objective Reinforcement**: unlike single-position emphasis methods, focus_on creates a
        reinforcing loop where the primary goal is established early as a critical requirement and
        then reinforced at the end as the ultimate focus. This dual reinforcement significantly
        reduces the risk of objective drift in complex prompts.

        **Attention Hierarchy Management**: by explicitly establishing one primary objective above
        all others, this method helps manage attention hierarchy in complex prompts with multiple
        competing requirements, ensuring that when trade-offs must be made, the primary goal takes
        clear precedence.

        Parameters
        ----------
        primary_goal
            The single most important objective that must receive maximum attention and priority
            throughout the AI's response. Should be formulated as a clear, specific, and measurable
            objective that captures the essential purpose of the prompt (e.g., "Provide actionable
            security recommendations", "Create implementable cost reduction strategies", "Generate
            learning-focused technical explanations").

        Returns
        -------
        PromptBuilder
            Self for method chaining, allowing combination with other prompt building
            methods to create comprehensive, structured prompts.

        Examples
        --------
        ### Security-first system analysis

        Ensure security remains the absolute priority across all considerations:

        ```python
        import talk_box as tb

        # System analysis with security focus
        builder = (
            tb.PromptBuilder()
            .persona("security architect", "enterprise security design")
            .focus_on("Identify and eliminate all security vulnerabilities before considering any other improvements")
            .task_context("Analyze enterprise application architecture for production deployment")
            .core_analysis([
                "Authentication and authorization mechanisms",
                "Data protection and encryption standards",
                "Network security and access controls",
                "Infrastructure security configuration"
            ])
            .constraint("Include performance optimization suggestions where security-compatible")
            .constraint("Consider user experience implications of security measures")
            .output_format([
                "Security assessment with risk severity levels",
                "Critical vulnerabilities requiring immediate attention",
                "Security-first recommendations with implementation priorities"
            ])
        )

        print(builder)
        ```

        ### Cost-effectiveness priority

        Prioritize cost-effective solutions above all other considerations:

        ```python
        # Business optimization with cost focus
        builder = (
            tb.PromptBuilder()
            .persona("operations consultant", "cost optimization and efficiency")
            .focus_on("Maximize cost savings while maintaining operational quality")
            .task_context("Develop operational efficiency improvement plan for manufacturing company")
            .core_analysis([
                "Current cost structure and inefficiencies",
                "Automation and technology opportunities",
                "Process optimization potential",
                "Resource allocation improvements"
            ])
            .constraint("Include innovation opportunities where cost-effective")
            .constraint("Consider employee impact and change management")
            .output_format([
                "Cost analysis with savings potential",
                "Implementation priorities by ROI and payback period",
                "Budget-conscious recommendations with measurable outcomes"
            ])
        )
        ```

        ### User experience excellence

        Make user experience the paramount consideration in all decisions:

        ```python
        # Product development with UX focus
        builder = (
            tb.PromptBuilder()
            .persona("UX designer", "user-centered product design")
            .focus_on("Optimize every aspect of the user experience above technical or business convenience")
            .task_context("Redesign mobile banking application interface for improved usability")
            .core_analysis([
                "User journey mapping and pain points",
                "Accessibility and inclusive design requirements",
                "Interface clarity and intuitive navigation",
                "Performance impact on user experience"
            ])
            .constraint("Consider technical implementation constraints")
            .constraint("Include business stakeholder requirements")
            .output_format([
                "UX assessment with user impact analysis",
                "Design recommendations prioritized by user value",
                "Implementation plan with user testing validation"
            ])
        )
        ```

        ### Learning-centered education

        Prioritize learning effectiveness over all other educational considerations:

        ```python
        # Educational design with learning focus
        builder = (
            tb.PromptBuilder()
            .persona("instructional designer", "evidence-based learning design")
            .focus_on("Maximize student learning outcomes and knowledge retention")
            .task_context("Design comprehensive data science curriculum for career changers")
            .core_analysis([
                "Learning objective alignment and progression",
                "Skill building sequence and scaffolding",
                "Practice opportunities and feedback mechanisms",
                "Real-world application and project integration"
            ])
            .constraint("Consider time constraints and resource limitations")
            .constraint("Include diverse learning styles and accessibility")
            .output_format([
                "Curriculum structure with learning outcome mapping",
                "Module design with skill progression tracking",
                "Assessment strategy focused on competency development"
            ])
        )
        ```

        ### Performance optimization priority

        Make system performance the primary consideration across all decisions:

        ```python
        # Technical optimization with performance focus
        builder = (
            tb.PromptBuilder()
            .persona("performance engineer", "high-scale system optimization")
            .focus_on("Achieve maximum system performance and scalability")
            .task_context("Optimize distributed microservices architecture for peak traffic handling")
            .core_analysis([
                "Current performance bottlenecks and limitations",
                "Scalability patterns and load distribution",
                "Caching strategies and data optimization",
                "Infrastructure scaling and resource management"
            ])
            .constraint("Maintain code maintainability where possible")
            .constraint("Consider development team capabilities")
            .output_format([
                "Performance analysis with benchmark comparisons",
                "Optimization recommendations by impact and effort",
                "Implementation roadmap with performance milestones"
            ])
        )
        ```

        ### Compliance-first approach

        Ensure regulatory compliance takes absolute precedence:

        ```python
        # Compliance analysis with regulatory focus
        builder = (
            tb.PromptBuilder()
            .persona("compliance officer", "financial services regulation")
            .focus_on("Ensure 100% regulatory compliance before any operational considerations")
            .task_context("Audit investment management platform for regulatory adherence")
            .core_analysis([
                "Regulatory requirement mapping and gaps",
                "Risk assessment and mitigation strategies",
                "Documentation and audit trail completeness",
                "Process compliance and control effectiveness"
            ])
            .constraint("Include operational efficiency opportunities where compliant")
            .constraint("Consider user experience impact of compliance measures")
            .output_format([
                "Compliance status with regulatory requirement tracking",
                "Critical violations requiring immediate remediation",
                "Compliance-first recommendations with implementation priorities"
            ])
        )
        ```

        ### Innovation-driven development

        Prioritize innovative solutions that provide competitive advantage:

        ```python
        # Technology strategy with innovation focus
        builder = (
            tb.PromptBuilder()
            .persona("innovation strategist", "emerging technology adoption")
            .focus_on("Identify and implement innovative solutions that create significant competitive advantage")
            .task_context("Develop technology roadmap for digital transformation initiative")
            .core_analysis([
                "Emerging technology opportunities and applications",
                "Competitive differentiation potential",
                "Implementation feasibility and risk assessment",
                "ROI and business impact projections"
            ])
            .constraint("Consider enterprise stability and risk tolerance")
            .constraint("Include team capability development requirements")
            .output_format([
                "Innovation assessment with competitive impact analysis",
                "Technology recommendations prioritized by advantage potential",
                "Implementation strategy with innovation milestones"
            ])
        )
        ```

        ### Quality-first manufacturing

        Make product quality the overriding priority in all manufacturing decisions:

        ```python
        # Manufacturing optimization with quality focus
        builder = (
            tb.PromptBuilder()
            .persona("quality engineer", "manufacturing excellence")
            .focus_on("Achieve superior product quality that exceeds customer expectations")
            .task_context("Optimize manufacturing process for automotive component production")
            .core_analysis([
                "Current quality metrics and defect analysis",
                "Process control and variability reduction",
                "Quality assurance and testing protocols",
                "Continuous improvement opportunities"
            ])
            .constraint("Consider production efficiency where quality-compatible")
            .constraint("Include cost implications of quality improvements")
            .output_format([
                "Quality assessment with defect root cause analysis",
                "Process improvements prioritized by quality impact",
                "Implementation plan with quality validation metrics"
            ])
        )
        ```

        ### Customer satisfaction focus

        Prioritize customer satisfaction and loyalty above all business metrics:

        ```python
        # Customer experience with satisfaction focus
        builder = (
            tb.PromptBuilder()
            .persona("customer success manager", "customer loyalty and satisfaction")
            .focus_on("Maximize customer satisfaction and long-term loyalty")
            .task_context("Develop customer service improvement strategy for SaaS platform")
            .core_analysis([
                "Current customer satisfaction metrics and feedback",
                "Service delivery gaps and pain points",
                "Customer success journey optimization",
                "Retention and loyalty enhancement opportunities"
            ])
            .constraint("Consider operational resource constraints")
            .constraint("Include revenue impact where customer-positive")
            .output_format([
                "Customer satisfaction analysis with improvement priorities",
                "Service enhancement recommendations by customer impact",
                "Implementation timeline with satisfaction measurement"
            ])
        )
        ```

        ### Sustainability leadership

        Make environmental sustainability the primary driver of all decisions:

        ```python
        # Business strategy with sustainability focus
        builder = (
            tb.PromptBuilder()
            .persona("sustainability director", "environmental leadership")
            .focus_on("Minimize environmental impact while maintaining business viability")
            .task_context("Develop sustainable operations strategy for retail supply chain")
            .core_analysis([
                "Current environmental impact assessment",
                "Sustainable technology and process opportunities",
                "Supply chain optimization for sustainability",
                "Carbon footprint reduction strategies"
            ])
            .constraint("Consider financial viability and business continuity")
            .constraint("Include stakeholder impact and change management")
            .output_format([
                "Sustainability assessment with environmental impact metrics",
                "Green initiatives prioritized by environmental benefit",
                "Implementation roadmap with sustainability milestones"
            ])
        )
        ```

        Integration Notes
        -----------------
        - **Dual Positioning**: Leverages both primacy and recency effects for maximum attention impact
        - **Objective Reinforcement**: Creates reinforcing attention pattern that prevents goal drift
        - **Attention Hierarchy**: Establishes clear priority structure for complex prompts
        - **Trade-off Guidance**: Provides clear decision criteria when competing objectives conflict
        - **Quality Assurance**: Ensures responses align with the most critical objective throughout
        - **Strategic Emphasis**: Combines front-loaded critical constraints with end-positioned final emphasis

        The focus_on method provides the strongest possible attention management by establishing the
        primary objective as both the opening critical requirement and closing final emphasis,
        creating a dual-anchor system that maintains unwavering focus on the most important goal
        throughout the entire AI response generation process.
        """
        # Add as critical constraint (front-loaded)
        self.critical_constraint(f"Primary objective: {primary_goal}")
        # Also set as final emphasis (recency bias)
        self._final_emphasis = f"Focus your entire response on: {primary_goal}"
        return self

    def _build(self) -> str:
        """
        Internal method to construct the final prompt using attention-optimized structure.

        This method is used internally by ChatBot to create the system prompt while preserving
        the structured data for testing and analysis.
        """
        # fmt: off
        """
        Construct the final prompt using attention-optimized structure based on cognitive psychology principles.

        The build method transforms the accumulated prompt components into a strategically structured prompt
        that maximizes AI attention and response quality through evidence-based sequencing. This method
        implements a comprehensive attention management system that leverages primacy effects, recency bias,
        and cognitive load optimization to ensure that the most critical information receives maximum focus
        during AI processing.

        **Attention Architecture**: the prompt structure follows a carefully researched sequence that aligns
        with how large language models process and prioritize information. Each section is positioned to
        optimize attention allocation, with critical elements placed at psychologically proven high-attention
        positions (beginning and end) while supporting information is organized to minimize cognitive load.

        **Cognitive Load Management**: the structured approach prevents cognitive overload by presenting
        information in digestible, hierarchically organized sections. This allows the AI to process complex
        requirements systematically while maintaining focus on the most important objectives throughout
        response generation.

        **Priority-Based Organization**: all prompt sections are automatically sorted by priority level,
        ensuring that high-priority information receives prominent placement and attention. This systematic
        prioritization prevents important requirements from being overshadowed by less critical details.

        **Behavioral Anchoring**: the persona-first structure establishes behavioral context before presenting
        tasks or constraints, allowing the AI to adopt the appropriate role and mindset before processing
        specific requirements. This behavioral anchoring significantly improves response quality and
        consistency with desired expertise levels.

        **Structural Sequence**: the method implements an 8-section attention-optimized structure:

        1. **Persona**: establishes behavioral context and expertise level
        2. **Critical Constraints**: front-loads most important requirements (primacy effect)
        3. **Task Context**: provides clear objective with priority level
        4. **Structured Sections**: presents analysis requirements in priority order
        5. **Standard Constraints**: adds additional requirements without overwhelming
        6. **Output Format**: specifies structure and presentation requirements
        7. **Examples**: provides concrete demonstrations when available
        8. **Final Emphasis**: leverages recency bias for ultimate priority

        Returns
        -------
        str
            Complete prompt string with attention-optimized structure, ready for submission to LLMs.
            The prompt follows evidence-based sequencing principles to maximize response quality and
            adherence to requirements.

        Examples
        --------
        ### Basic prompt construction

        Simple prompt with essential components:

        ```python
        import talk_box as tb

        # Build a basic technical analysis prompt
        builder = (
            tb.PromptBuilder()
            .persona("software architect", "system design")
            .task_context("Design API architecture for e-commerce platform")
            .core_analysis([
                "Service boundaries and responsibilities",
                "Data flow and state management",
                "Security and authentication approach"
            ])
            .output_format([
                "Architecture overview diagram",
                "API specification with endpoints",
                "Implementation recommendations"
            ])
        )

        prompt = builder
        print(prompt)
        ```

        Output structure:

        ```
        You are a software architect specializing in system design.

        TASK: Design API architecture for e-commerce platform

        Core Analysis Requirements:
        - Service boundaries and responsibilities
        - Data flow and state management
        - Security and authentication approach

        OUTPUT FORMAT:
        - Architecture overview diagram
        - API specification with endpoints
        - Implementation recommendations
        ```

        ### Complex prompt with full feature set

        Comprehensive prompt utilizing all available features:

        ```python
        # Build a complex security audit prompt
        builder = (
            tb.PromptBuilder()
            .persona("security engineer", "application security")
            .focus_on("Identify critical security vulnerabilities that pose immediate risk")
            .task_context(
                "Audit web application for production deployment",
                priority=tb.Priority.HIGH
            )
            .core_analysis([
                "Authentication and authorization mechanisms",
                "Input validation and data sanitization",
                "Infrastructure security configuration"
                ],
                priority=tb.Priority.HIGH
            )
            .structured_section(
                "Risk Assessment Framework",
                "Evaluate each finding using CVSS scoring methodology",
                priority=tb.Priority.MEDIUM
            )
            .constraint("Include remediation effort estimates")
            .avoid_topics([
                "Security through obscurity approaches",
                "Custom cryptographic implementations"
            ])
            .output_format([
                "Executive summary with critical issues",
                "Detailed findings with CVSS scores",
                "Prioritized remediation roadmap"
            ])
            .example(
                "SQL injection vulnerability in login form",
                "**Critical**: SQL injection in authentication endpoint (CVSS 9.8). Immediate fix required: implement parameterized queries and input validation."
            )
        )

        prompt = builder
        print(prompt)
        ```

        Output structure demonstrates full attention optimization:
        ```
        You are a security engineer specializing in application security.

        CRITICAL REQUIREMENTS:
        - Primary objective: Identify critical security vulnerabilities that pose immediate risk

        TASK: Audit web application for production deployment

        Core Analysis Requirements:
        - Authentication and authorization mechanisms
        - Input validation and data sanitization
        - Infrastructure security configuration

        Risk Assessment Framework:
        Evaluate each finding using CVSS scoring methodology

        ADDITIONAL CONSTRAINTS:
        - Include remediation effort estimates
        - Avoid: Security through obscurity approaches, Custom cryptographic implementations

        OUTPUT FORMAT:
        - Executive summary with critical issues
        - Detailed findings with CVSS scores
        - Prioritized remediation roadmap

        EXAMPLES:

        Example 1:
        Input: SQL injection vulnerability in login form
        Output: **Critical**: SQL injection in authentication endpoint (CVSS 9.8). Immediate fix required - implement parameterized queries and input validation.

        Focus your entire response on: Identify critical security vulnerabilities that pose immediate risk
        ```

        ### Business strategy prompt with constraints

        Professional business analysis with comprehensive requirements:

        ```python
        # Build a strategic planning prompt
        builder = (
            tb.PromptBuilder()
            .persona("business consultant", "strategic planning and market analysis")
            .task_context("Develop market entry strategy for SaaS startup")
            .critical_constraint("All recommendations must be implementable with $500K budget")
            .core_analysis([
                "Target market size and segmentation",
                "Competitive landscape and positioning",
                "Go-to-market strategy and channels",
                "Revenue projections and unit economics"
            ])
            .constraint("Focus on B2B market opportunities")
            .constraint("Include 12-month timeline with milestones")
            .avoid_topics([
                "Aggressive customer acquisition without unit economics validation",
                "Market strategies requiring significant upfront capital"
            ])
            .output_format([
                "Market analysis with addressable market sizing",
                "Competitive positioning and differentiation strategy",
                "Implementation roadmap with budget allocation"
            ])
            .final_emphasis("Prioritize strategies with fastest path to sustainable revenue")
        )

        prompt = builder
        ```

        ### Educational content development

        Learning-focused prompt with pedagogical considerations:

        ```python
        # Build an instructional design prompt
        builder = (
            tb.PromptBuilder()
            .persona("instructional designer", "adult learning and skill development")
            .task_context("Create Python programming curriculum for career changers")
            .core_analysis([
                "Learning objective progression and scaffolding",
                "Hands-on practice and project integration",
                "Assessment strategies and competency validation"
            ])
            .structured_section(
                "Cognitive Load Management",
                "Design lessons that prevent information overload while building complexity",
                priority=tb.Priority.HIGH
            )
            .constraint("Include diverse learning styles and accessibility considerations")
            .constraint("Provide clear success metrics and progress tracking")
            .avoid_topics([
                "Memorization-based learning without practical application",
                "Complex theoretical concepts before foundational skills"
            ])
            .output_format([
                "Curriculum structure with weekly learning outcomes",
                "Module breakdown with practice exercises",
                "Assessment rubrics and competency milestones"
            ])
            .example(
                "Week 3: Functions and code organization",
                "Learning Outcome: Students write reusable functions with clear parameters and return values. Practice: Build a calculator with separate functions for each operation. Assessment: Code review focusing on function design principles."
            )
        )

        prompt = builder
        ```

        ### Technical code review prompt

        Development-focused prompt with quality emphasis:

        ```python
        # Build a code review prompt
        builder = (
            tb.PromptBuilder()
            .persona("senior developer", "code quality and mentorship")
            .task_context("Review pull request for production deployment readiness")
            .critical_constraint("Code must meet production quality standards before approval")
            .core_analysis([
                "Code correctness and functionality",
                "Security vulnerability assessment",
                "Performance implications and optimization opportunities",
                "Maintainability and documentation quality"
            ])
            .constraint("Provide constructive feedback with learning opportunities")
            .constraint("Include specific examples and improvement suggestions")
            .avoid_topics([
                "Quick fixes that introduce technical debt",
                "Skipping unit tests for faster delivery"
            ])
            .output_format([
                "Overall assessment with approval recommendation",
                "Specific issues categorized by severity",
                "Improvement suggestions with code examples"
            ])
            .focus_on("Ensure code quality meets production standards while providing educational feedback")
        )

        prompt = builder
        ```

        ### Medical analysis prompt with safety emphasis

        Healthcare-focused prompt with patient safety priority:

        ```python
        # Build a medical analysis prompt
        builder = (
            tb.PromptBuilder()
            .persona("healthcare systems analyst", "patient safety and quality improvement")
            .task_context("Analyze hospital workflow for efficiency improvements")
            .critical_constraint("Patient safety must never be compromised for efficiency gains")
            .core_analysis([
                "Patient flow and care coordination effectiveness",
                "Resource utilization and bottleneck identification",
                "Technology integration opportunities",
                "Staff workflow and communication patterns"
            ])
            .structured_section(
                "Safety Impact Assessment",
                "Evaluate how each proposed change affects patient safety outcomes",
                priority=tb.Priority.CRITICAL
            )
            .constraint("Include staff training and change management requirements")
            .avoid_topics([
                "Efficiency improvements that reduce patient care time",
                "Cost-cutting measures that compromise safety protocols"
            ])
            .output_format([
                "Workflow analysis with safety impact assessment",
                "Improvement recommendations prioritized by patient benefit",
                "Implementation plan with safety validation protocols"
            ])
            .final_emphasis("Every recommendation must enhance or maintain patient safety as the primary objective")
        )

        prompt = builder
        ```

        ### Research methodology prompt

        Academic research with methodological rigor:

        ```python
        # Build a research design prompt
        builder = (
            tb.PromptBuilder()
            .persona("research methodologist", "quantitative analysis and study design")
            .task_context("Design study to evaluate educational intervention effectiveness")
            .core_analysis([
                "Research design appropriateness and validity",
                "Sample size calculations and statistical power",
                "Bias mitigation and control strategies",
                "Measurement instruments and reliability"
            ])
            .structured_section(
                "Ethical Considerations",
                "Address participant protection and informed consent requirements",
                priority=tb.Priority.HIGH
            )
            .constraint("Maintain rigorous scientific standards throughout")
            .constraint("Include practical implementation considerations")
            .avoid_topics([
                "P-hacking or selective result reporting",
                "Inadequate sample sizes for meaningful conclusions"
            ])
            .output_format([
                "Study design with methodological justification",
                "Statistical analysis plan with power calculations",
                "Ethical review requirements and procedures"
            ])
            .example(
                "Randomized controlled trial with 200 participants",
                "Design: RCT with treatment/control groups (n=100 each). Power analysis indicates 80% power to detect medium effect size (d=0.5) at α=0.05. Block randomization ensures balanced groups."
            )
        )

        prompt = builder
        ```

        ### Financial analysis prompt with risk management

        Investment analysis with comprehensive risk assessment:

        ```python
        # Build a financial analysis prompt
        builder = (
            tb.PromptBuilder()
            .persona("financial analyst", "investment research and risk assessment")
            .task_context("Evaluate portfolio diversification strategy for institutional investor")
            .core_analysis([
                "Asset allocation optimization and correlation analysis",
                "Risk-adjusted return projections",
                "Market volatility and stress testing",
                "Liquidity requirements and constraints"
            ])
            .structured_section(
                "ESG Integration",
                "Incorporate environmental, social, and governance factors into analysis",
                priority=tb.Priority.MEDIUM
            )
            .constraint("Include regulatory compliance requirements")
            .constraint("Consider tax efficiency optimization opportunities")
            .avoid_topics([
                "High-risk speculative investments without proper analysis",
                "Ignoring correlation risks during market stress"
            ])
            .output_format([
                "Portfolio optimization with efficient frontier analysis",
                "Risk assessment with scenario modeling",
                "Implementation roadmap with rebalancing strategy"
            ])
            .focus_on("Optimize risk-adjusted returns while maintaining portfolio stability")
        )

        prompt = builder
        ```

        ### Content strategy prompt with audience focus

        Marketing content development with user-centric approach:

        ```python
        # Build a content strategy prompt
        builder = (
            tb.PromptBuilder()
            .persona("content strategist", "audience engagement and value creation")
            .task_context("Develop content marketing strategy for B2B software company")
            .core_analysis([
                "Audience persona development and pain point analysis",
                "Content format effectiveness and engagement metrics",
                "Distribution channel optimization and reach",
                "Competitive content landscape and differentiation"
            ])
            .constraint("Focus on providing genuine value to target audience")
            .constraint("Include SEO optimization without compromising quality")
            .avoid_topics([
                "Clickbait tactics without delivering promised value",
                "Content created solely for volume metrics"
            ])
            .output_format([
                "Content calendar with audience value mapping",
                "Content format recommendations with engagement projections",
                "Distribution strategy with channel optimization"
            ])
            .example(
                "Technical decision-maker pain point: evaluating security solutions",
                "Content: 'Security Framework Comparison Guide' - detailed analysis of popular frameworks with implementation complexity, cost analysis, and real-world case studies."
            )
            .final_emphasis("Every piece of content must solve a specific problem for the target audience")
        )

        prompt = builder
        ```

        Integration Notes
        -----------------
        - **Attention Architecture**: implements evidence-based sequencing for maximum AI focus
        - **Cognitive Load Management**: organizes information to prevent processing overload
        - **Priority-Based Organization**: automatically sorts sections by importance level
        - **Behavioral Anchoring**: establishes role context before presenting requirements
        - **Structural Optimization**: uses 8-section framework for comprehensive coverage
        - **Psychological Principles**: leverages primacy, recency, and attention management research

        The build method represents the culmination of the attention-optimized prompt engineering
        approach, transforming accumulated components into a strategically structured prompt that
        maximizes AI performance through evidence-based design principles and cognitive psychology
        research.
        """
        # fmt: on
        prompt_parts = []

        # 1. Persona
        if self._persona:
            prompt_parts.append(self._persona)

        # 2. Critical constraints
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

        # 8. Final emphasis
        if self._final_emphasis:
            prompt_parts.append(f"\n{self._final_emphasis}")

        return "\n".join(prompt_parts)

    def __str__(self) -> str:
        """Return the complete built prompt text."""
        return self._build()

    def print(self) -> None:
        """Print the complete built prompt text."""
        print(self._build())

    def __repr__(self) -> str:
        """Return a developer-friendly representation of the PromptBuilder configuration."""
        components = []

        # Add persona if present
        if self._persona:
            persona_short = self._persona.replace("You are a ", "").replace("You are an ", "")
            if len(persona_short) > 50:
                persona_short = persona_short[:47] + "..."
            components.append(f"persona='{persona_short}'")

        # Add task context if present
        if self._task_context:
            context_short = self._task_context
            if len(context_short) > 40:
                context_short = context_short[:37] + "..."
            components.append(f"task='{context_short}'")

        # Add constraints count
        if self._constraints:
            components.append(f"constraints={len(self._constraints)}")

        # Add sections count
        if self._sections:
            components.append(f"sections={len(self._sections)}")

        # Add output format count
        if self._output_format:
            components.append(f"output_format={len(self._output_format)}")

        # Add final emphasis indicator
        if self._final_emphasis:
            components.append("final_emphasis=True")

        # Build the representation
        if components:
            return f"PromptBuilder({', '.join(components)})"
        else:
            return "PromptBuilder(empty)"


# Convenience functions for common patterns
def architectural_analysis_prompt() -> PromptBuilder:
    """
    Create a pre-configured prompt builder for architectural analysis tasks.

    Implements the optimized pattern from the blog post example.

    Returns
    -------
    PromptBuilder
        Configured PromptBuilder for architectural analysis.
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
        Configured PromptBuilder for code reviews.
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
        Configured PromptBuilder for debugging tasks.
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
