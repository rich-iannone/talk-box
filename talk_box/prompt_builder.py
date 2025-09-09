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
    Represents a structured section of an attention-optimized prompt with priority and ordering
    metadata.

    The `PromptSection` class is a fundamental building block used by `PromptBuilder` to create
    sophisticated, attention-optimized prompts. Each section encapsulates content along with
    metadata that controls how the section is positioned and prioritized within the final prompt.
    This enables precise control over attention flow and information hierarchy.

    **Integration with PromptBuilder**:

    While users can create `PromptSection` objects directly, they are typically created
    automatically by `PromptBuilder` methods. The sections are then assembled according to attention
    principles to create optimized final prompts. This design provides both high-level convenience
    through `PromptBuilder` and fine-grained control through direct `PromptSection` manipulation.

    **Attention Optimization**:

    Each section contributes to the overall attention strategy:

    - **Priority**: determines relative importance and influences final ordering
    - **Section Type**: enables grouping and specialized handling of content types
    - **Order Hint**: provides fine-grained control over section positioning
    - **Content**: the actual prompt text optimized for the section's role

    The combination of these attributes allows the prompt building system to create prompts that
    leverage attention mechanisms effectively, ensuring critical information receives appropriate
    model focus while maintaining natural conversation flow.

    Parameters
    ----------
    content
        The text content of the prompt section. This is the actual text that will appear in the
        final prompt. Content should be crafted to serve the section's specific purpose within the
        overall prompt strategy.
    priority
        Attention priority level determining section placement order and emphasis. Higher priority
        sections are typically placed in more prominent positions. Defaults to `Priority.MEDIUM`.
    section_type
        Type classification for the section enabling specialized handling and grouping. This allows
        the prompt builder to apply type-specific optimization strategies. Defaults to `"general"`.
    order_hint
        Fine-grained ordering hint where lower numbers appear earlier in the prompt. This provides
        precise control over section positioning beyond priority levels. Sections with the same
        priority are ordered by this value. Defaults to `0`.

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
    **Attention Optimization**: sections are designed to work together to guide model attention
    effectively, with priority and positioning controlling information hierarchy.

    **Modularity**: each section encapsulates a specific aspect of the prompt, enabling reusable
    components and systematic prompt construction.

    **Flexibility**: the section system supports both structured workflows through standard section
    types and custom applications through extensible metadata.

    **Composability**: sections can be combined, reordered, and manipulated to create sophisticated
    prompt strategies for different use cases.

    **Cognitive Alignment**: section design aligns with cognitive psychology principles like
    primacy/recency effects and information chunking for optimal comprehension.

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
    The `PromptBuilder` applies proven principles that enhance model performance and response
    quality through strategic information placement and cognitive load management.

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

    The `PromptBuilder` provides a comprehensive set of methods for creating structured,
    attention-optimized prompts. All methods support fluent chaining for natural prompt
    construction.

    The core foundation methods:

    - `persona(role, expertise=None)`: set the AI's identity and behavioral framework
    - `task_context(context, priority=CRITICAL)`: define the primary objective and scope
    - `critical_constraint(constraint)`: add front-loaded, non-negotiable requirements
    - `constraint(constraint)`: add important but secondary requirements

    The structure and analysis methods:

    - `structured_section(title, content, priority=MEDIUM, required=False)`: create organized
    content sections
    - `core_analysis(analysis_points)`: define required analytical focus areas
    - `output_format(format_specs)`: specify response structure and formatting requirements
    - `example(input_example, output_example)`: provide concrete input/output demonstrations

    The focus and guidance methods:

    - `focus_on(primary_goal)`: emphasize the most important objective
    - `avoid_topics(topics)`: explicitly exclude irrelevant or problematic areas
    - `final_emphasis(emphasis)`: add closing reinforcement using recency bias

    Output methods:

    - `build()`: generate the final structured prompt string
    - `preview_structure()`: preview the prompt organization and metadata

    Each method is designed to work together in the attention-optimized prompt structure, with
    positioning and formatting automatically handled to maximize model performance.

    Examples
    --------
    ### Basic prompt construction

    Create a simple prompt with persona and task:

    ```{python}
    import talk_box as tb

    prompt = (
        tb.PromptBuilder()
        .persona("data scientist", "machine learning")
        .task_context("analyze customer churn patterns")
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
        .critical_constraint("focus only on production-ready solutions")
        .task_context("review the codebase architecture")
        .core_analysis([
            "identify design patterns used",
            "assess scalability bottlenecks",
            "review security implications"
        ])
        .structured_section(
            "Performance Metrics", [
                "response time requirements",
                "throughput expectations",
                "memory usage constraints"
            ],
            priority=tb.Priority.HIGH
        )
        .output_format([
            "executive summary (2-3 sentences)",
            "detailed findings with code examples",
            "prioritized recommendations"
        ])
        .final_emphasis("provide actionable next steps")

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
        .task_context("review the pull request for potential issues")
        .critical_constraint("flag any security vulnerabilities immediately")
        .structured_section(
            "Review Areas", [
                "logic and correctness",
                "security considerations",
                "performance implications",
                "code readability and documentation"
            ]
        )
        .output_format([
            "critical issues (must fix)",
            "suggestions (should consider)",
            "positive feedback"
        ])
        .avoid_topics(["personal coding style preferences"])
        .focus_on("providing constructive, actionable feedback")

    )
    ```

    We can look at the generated prompt:

    ```{python}
    print(prompt)
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

        Parameters
        ----------
        role
            The primary professional role or identity the AI should adopt. This should be specific
            and professional (e.g., `"senior software architect"`, `"data scientist"`,
            `"technical writer"`, etc.). The role influences response style, terminology, and the
            level of technical depth provided.
        expertise
            Specific area of expertise or specialization within the role. This narrows the focus and
            enhances domain-specific knowledge application (e.g., `"distributed systems"`,
            `"machine learning"`, `"API documentation"`, etc.). If not provided, the persona will be
            general within the specified role.

        Returns
        -------
        PromptBuilder
            Self for method chaining, allowing combination with other prompt building methods to
            create comprehensive, structured prompts.

        Research Foundation
        -------------------
        **Behavioral Psychology.** Persona establishment leverages behavioral psychology principles
        to create consistent response patterns aligned with professional roles and expertise domains.

        **Identity Anchoring.** Setting a clear professional identity serves as a cognitive anchor
        that influences all subsequent AI reasoning and response generation processes.

        **Domain Expertise Activation.** Specifying expertise areas activates relevant knowledge
        domains and professional terminology appropriate to the specified field.

        Prompt Positioning
        ------------------
        Persona statements are positioned at the very beginning of prompts to establish the
        behavioral framework before any task instructions or constraints are provided.

        Best Practices
        --------------
        Follow these guidelines for effective persona establishment:

        - use specific, professional role titles rather than generic descriptions
        - include relevant expertise areas to enhance domain-specific knowledge application
        - ensure persona aligns with the complexity and scope of the intended task
        - maintain consistency with persona throughout all prompt elements

        Integration Notes
        -----------------
        - **Behavioral Anchoring**: the persona establishes cognitive framework before task
        instructions
        - **Response Consistency**: maintains consistent voice and expertise level throughout
        interaction
        - **Domain Knowledge**: activates relevant knowledge domains and professional terminology
        - **Communication Style**: influences formality, technical depth, and explanatory approach
        - **Quality Indicators**: expert personas tend to provide more nuanced, comprehensive
        responses

        The `.persona()` method provides the foundational identity that guides all subsequent AI
        behavior, ensuring responses align with professional expectations and domain expertise
        requirements.

        Examples
        --------
        ### Basic role assignment

        Set a clear professional identity for the AI:

        ```{python}
        import talk_box as tb

        # Simple role without specific expertise
        builder = (
            tb.PromptBuilder()
            .persona("data analyst")
            .task_context("analyze customer satisfaction survey results")
        )

        print(builder)
        ```

        ### Role with domain expertise

        Combine role with specific area of expertise:

        ```{python}
        # Specialized expertise within role
        builder = (
            tb.PromptBuilder()
            .persona("software engineer", "backend API development")
            .task_context("review the authentication service architecture")
            .core_analysis([
                "security implementation patterns",
                "scalability considerations",
                "error handling strategies"
            ])
        )

        print(builder)
        ```

        ### Senior-level expertise

        Use seniority indicators for complex tasks:

        ```{python}
        # Senior-level role for complex analysis
        builder = (
            tb.PromptBuilder()
            .persona("senior software architect", "distributed systems")
            .critical_constraint("focus on production-scale considerations")
            .task_context("design a microservices architecture for high-traffic e-commerce")
        )
        ```

        ### Domain-specific personas

        We can create personas tailored to specific industries or domains. Here is one that is
        focused on healthcare domain expertise:

        ```{python}
        healthcare_builder = (
            tb.PromptBuilder()
            .persona("healthcare data analyst", "clinical research")
            .task_context("analyze patient outcome data for treatment effectiveness")
        )

        print(healthcare_builder)
        ```

        This is a specialized persona for the financial services industry:

        ```{python}
        finance_builder = (
            tb.PromptBuilder()
            .persona("quantitative analyst", "risk management")
            .task_context("evaluate portfolio risk exposure across asset classes")
        )

        print(finance_builder)
        ```

        This is a persona with educational technology expertise:

        ```{python}
        edtech_builder = (
            tb.PromptBuilder()
            .persona("educational technologist", "learning analytics")
            .task_context("design metrics for measuring student engagement")
        )

        print(edtech_builder)
        ```

        ### Combining personas with other prompt elements

        Build comprehensive prompts with persona as the foundation:

        ```{python}
        # Complete code review prompt with expert persona
        review_prompt = (
            tb.PromptBuilder()
            .persona("senior code reviewer", "security and performance")
            .critical_constraint("prioritize security vulnerabilities over style issues")
            .task_context("review this Python Flask application for production readiness")
            .core_analysis([
                "authentication and authorization implementation",
                "input validation and sanitization",
                "database query optimization",
                "error handling and logging"
            ])
            .output_format([
                "critical security issues (immediate attention)",
                "performance bottlenecks (optimization opportunities)",
                "code quality improvements (maintainability)",
                "positive patterns (reinforcement)"
            ])
            .final_emphasis("focus on issues that could impact production security or performance")
        )

        print(review_prompt)
        ```

        ### Persona influence on response style

        Subtle differences in personas can affect response characteristics:

        ```{python}
        # Technical depth variation
        beginner_persona = (
            tb.PromptBuilder()
            .persona("junior developer")
            .task_context("explain RESTful API design principles")
        )

        print(beginner_persona)
        ```

        ```{python}
        expert_persona = (
            tb.PromptBuilder()
            .persona("principal engineer", "API architecture")
            .task_context("explain RESTful API design principles")
        )

        print(expert_persona)
        ```

        The expert persona will provide more sophisticated insights, advanced patterns, and industry
        best practices compared to the junior developer persona's more fundamental explanations.

        ### Multiple expertise areas

        We can handle roles with multiple specializations. This persona has broad expertise
        combining multiple areas.

        ```{python}
        fullstack_persona = (
            tb.PromptBuilder()
            .persona("full-stack architect", "web applications and cloud infrastructure")
            .task_context("design end-to-end solution for real-time collaboration platform")
        )

        print(fullstack_persona)
        ```

        This is a research-focused persona with interdisciplinary expertise.

        ```{python}
        research_persona = (
            tb.PromptBuilder()
            .persona("research scientist", "machine learning and cognitive psychology")
            .task_context("evaluate AI model bias in human-computer interaction contexts")
        )

        print(research_persona)
        ```

        ### Persona consistency across conversations

        Maintain consistent persona behavior in extended interactions:

        ```{python}
        # Establish consistent technical writing persona
        technical_writer = (
            tb.PromptBuilder()
            .persona("technical documentation specialist", "developer tools")
            .task_context("create user guide for API integration")
        )

        print(technical_writer)
        ```
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
        This method is essential for creating focused, goal-oriented prompts that produce relevant
        and actionable responses.

        Parameters
        ----------
        context
            Clear, specific description of what needs to be accomplished. Should be
            action-oriented and provide sufficient detail for the AI to understand
            the expected scope and deliverables.
        priority
            Attention priority level for task placement in the final prompt. Defaults to
            `Priority.CRITICAL` to ensure the main task receives prominent positioning and maximum
            attention.

        Returns
        -------
        PromptBuilder
            Self for method chaining, allowing combination with other prompt building methods.

        Prompt Positioning
        ------------------
        Task context is typically placed early in the prompt structure (after persona and critical
        constraints) to establish clear expectations. The default `CRITICAL` priority ensures the
        task receives prominent attention placement.

        Best Practices
        --------------
        Follow these guidelines for effective task definition:

        - use clear, specific language that defines measurable outcomes
        - focus on action-oriented descriptions ("analyze", "review", "create")
        - avoid vague or ambiguous task descriptions
        - include scope boundaries when appropriate

        Examples
        --------
        ### Basic task definition

        Set a clear, focused task for the prompt:

        ```{python}
        import talk_box as tb

        # Simple task context
        builder = (
            tb.PromptBuilder()
            .persona("data analyst")
            .task_context("analyze the customer churn data to identify key patterns")
        )

        print(builder)
        ```

        ### Task with custom priority

        Use different priority levels for task positioning. Here is an example of a high priority
        task that is important but not critical:

        ```{python}
        builder = (
            tb.PromptBuilder()
            .persona("software architect")
            .critical_constraint("focus only on security vulnerabilities")
            .task_context(
                "review the authentication system architecture",
                priority=tb.Priority.HIGH
            )
        )

        print(builder)
        ```

        ### Detailed task with scope boundaries

        Create comprehensive task descriptions with clear boundaries:

        ```{python}
        # Detailed task with specific scope
        builder = (
            tb.PromptBuilder()
            .persona("technical writer", "API documentation")
            .task_context(
                "create comprehensive API documentation for the user management endpoints, "
                "including authentication requirements, request/response examples, "
                "and error handling procedures"
            )
            .core_analysis([
                "document each endpoint's purpose and functionality",
                "provide complete request/response schemas",
                "include practical usage examples"
            ])
        )

        print(builder)
        ```
        """
        self._task_context = context
        self._task_priority = priority
        return self

    def focus_on(self, primary_goal: str) -> "PromptBuilder":
        """
        Set the primary focus that leverages both front-loading and recency bias for maximum
        attention impact.

        `.focus_on()` provides a powerful dual-positioning method that ensures the most critical
        objective receives maximum attention throughout the prompt by strategically placing it both
        at the beginning (as a critical constraint) and at the end (as final emphasis). This
        dual-anchor approach leverages both primacy and recency effects to create the strongest
        possible attention focus on the primary objective.

        Parameters
        ----------
        primary_goal
            The single most important objective that must receive maximum attention and priority
            throughout the AI's response. Should be formulated as a clear, specific, and measurable
            objective that captures the essential purpose of the prompt (e.g.,
            `"provide actionable security recommendations"`,
            `"create implementable cost reduction strategies"`,
            `"generate learning-focused technical explanations"`, etc.).

        Returns
        -------
        PromptBuilder
            Self for method chaining, allowing combination with other prompt building methods to
            create comprehensive, structured prompts.

        Research Foundation
        ------------------
        **Dual Attention Strategy.** Information positioned at both the beginning and end of a
        sequence receives the highest attention and retention. By anchoring the primary goal at both
        positions, `.focus_on()` ensures that the most critical objective maintains prominence
        throughout the entire prompt processing sequence.

        **Primacy and Recency Effects.** The method capitalizes on both primacy bias (heightened
        attention to early information) and recency bias (heightened attention to final information)
        to create a reinforcing attention pattern that keeps the primary objective at the forefront
        of the AI's processing throughout response generation.

        **Objective Reinforcement.** Unlike single-position emphasis methods, `.focus_on()` creates
        a reinforcing loop where the primary goal is established early as a critical requirement and
        then reinforced at the end as the ultimate focus. This dual reinforcement significantly
        reduces the risk of objective drift in complex prompts.

        **Attention Hierarchy Management.** By explicitly establishing one primary objective above
        all others, this method helps manage attention hierarchy in complex prompts with multiple
        competing requirements, ensuring that when trade-offs must be made, the primary goal takes
        clear precedence.

        Integration Notes
        -----------------
        - **Dual Positioning**: leverages both primacy and recency effects for maximum attention
        impact
        - **Objective Reinforcement**: creates reinforcing attention pattern that prevents goal
        drift
        - **Attention Hierarchy**: establishes clear priority structure for complex prompts
        - **Trade-off Guidance**: provides clear decision criteria when competing objectives
        conflict
        - **Quality Assurance**: ensures responses align with the most critical objective throughout
        - **Strategic Emphasis**: combines front-loaded critical constraints with end-positioned
        final emphasis

        The `.focus_on()` method provides the strongest possible attention management by
        establishing the primary objective as both the opening critical requirement and closing
        final emphasis, creating a dual-anchor system that maintains unwavering focus on the most
        important goal throughout the entire AI response generation process.

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
            .focus_on("identify and eliminate all security vulnerabilities before considering any other improvements")
            .task_context("analyze enterprise application architecture for production deployment")
            .core_analysis([
                "authentication and authorization mechanisms",
                "data protection and encryption standards",
                "network security and access controls",
                "infrastructure security configuration"
            ])
            .constraint("include performance optimization suggestions where security-compatible")
            .constraint("consider user experience implications of security measures")
            .output_format([
                "security assessment with risk severity levels",
                "critical vulnerabilities requiring immediate attention",
                "security-first recommendations with implementation priorities"
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
            .focus_on("maximize cost savings while maintaining operational quality")
            .task_context("develop operational efficiency improvement plan for manufacturing company")
            .core_analysis([
                "current cost structure and inefficiencies",
                "automation and technology opportunities",
                "process optimization potential",
                "resource allocation improvements"
            ])
            .constraint("include innovation opportunities where cost-effective")
            .constraint("consider employee impact and change management")
            .output_format([
                "cost analysis with savings potential",
                "implementation priorities by ROI and payback period",
                "budget-conscious recommendations with measurable outcomes"
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
            .focus_on("optimize every aspect of the user experience above technical or business convenience")
            .task_context("redesign mobile banking application interface for improved usability")
            .core_analysis([
                "user journey mapping and pain points",
                "accessibility and inclusive design requirements",
                "interface clarity and intuitive navigation",
                "performance impact on user experience"
            ])
            .constraint("consider technical implementation constraints")
            .constraint("include business stakeholder requirements")
            .output_format([
                "UX assessment with user impact analysis",
                "design recommendations prioritized by user value",
                "implementation plan with user testing validation"
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
            .focus_on("maximize student learning outcomes and knowledge retention")
            .task_context("design comprehensive data science curriculum for career changers")
            .core_analysis([
                "learning objective alignment and progression",
                "skill building sequence and scaffolding",
                "practice opportunities and feedback mechanisms",
                "real-world application and project integration"
            ])
            .constraint("consider time constraints and resource limitations")
            .constraint("include diverse learning styles and accessibility")
            .output_format([
                "curriculum structure with learning outcome mapping",
                "module design with skill progression tracking",
                "assessment strategy focused on competency development"
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
            .focus_on("ensure 100% regulatory compliance before any operational considerations")
            .task_context("audit investment management platform for regulatory adherence")
            .core_analysis([
                "regulatory requirement mapping and gaps",
                "risk assessment and mitigation strategies",
                "documentation and audit trail completeness",
                "process compliance and control effectiveness"
            ])
            .constraint("include operational efficiency opportunities where compliant")
            .constraint("consider user experience impact of compliance measures")
            .output_format([
                "compliance status with regulatory requirement tracking",
                "critical violations requiring immediate remediation",
                "compliance-first recommendations with implementation priorities"
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
            .focus_on("identify and implement innovative solutions that create significant competitive advantage")
            .task_context("develop technology roadmap for digital transformation initiative")
            .core_analysis([
                "emerging technology opportunities and applications",
                "competitive differentiation potential",
                "implementation feasibility and risk assessment",
                "ROI and business impact projections"
            ])
            .constraint("consider enterprise stability and risk tolerance")
            .constraint("include team capability development requirements")
            .output_format([
                "innovation assessment with competitive impact analysis",
                "technology recommendations prioritized by advantage potential",
                "implementation strategy with innovation milestones"
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
            .focus_on("achieve superior product quality that exceeds customer expectations")
            .task_context("optimize manufacturing process for automotive component production")
            .core_analysis([
                "current quality metrics and defect analysis",
                "process control and variability reduction",
                "quality assurance and testing protocols",
                "continuous improvement opportunities"
            ])
            .constraint("consider production efficiency where quality-compatible")
            .constraint("include cost implications of quality improvements")
            .output_format([
                "quality assessment with defect root cause analysis",
                "process improvements prioritized by quality impact",
                "implementation plan with quality validation metrics"
            ])
        )
        ```
        """
        # Add as critical constraint (front-loaded)
        self.critical_constraint(f"Primary objective: {primary_goal}")
        # Also set as final emphasis (recency bias)
        self._final_emphasis = f"Focus your entire response on: {primary_goal}"
        return self

    def critical_constraint(self, constraint: str) -> "PromptBuilder":
        """
        Add a critical constraint that will be front-loaded for maximum attention and impact.

        Critical constraints are the highest-priority requirements that must be prominently
        positioned in the final prompt to ensure maximum model attention and compliance. These
        constraints are automatically placed in the `"CRITICAL REQUIREMENTS"` section immediately
        after the persona and before the main task, leveraging the primacy effect to maximize their
        influence on response generation.

        Parameters
        ----------
        constraint
            Specific constraint or requirement that must receive maximum attention. Should be clear,
            actionable, and measurable when possible. Use imperative language for direct instruction
            (e.g., `"Focus only on security vulnerabilities"`,
            `"Provide exactly 3 recommendations"`,
            `"Avoid discussing implementation details"`, etc.).

        Returns
        -------
        PromptBuilder
            Self for method chaining, allowing combination with other prompt building methods to
            create comprehensive, structured prompts.

        Research Foundation
        -------------------
        **Primacy Effect Research.** Based on findings demonstrating that early-positioned
        instructions have the greatest impact on task accuracy and model compliance. The
        front-loading strategy ensures critical requirements receive maximum attention allocation
        during the model's processing phase.

        **Attention Positioning Theory.** Critical constraints are placed at the very beginning of
        the constraint hierarchy, appearing before any task context or analysis requirements. This
        strategic positioning leverages cognitive psychology principles where information presented
        early has disproportionate influence on decision-making and response generation.

        **Constraint Hierarchy Management.** Multiple critical constraints are ordered by insertion,
        with the first added appearing first in the final prompt. This allows for fine-grained
        control over the relative importance of multiple critical requirements, creating a clear
        precedence structure for complex prompts with competing priorities.

        **Use Case Classification.** Critical constraints are ideal for security and safety
        requirements that cannot be compromised, output format restrictions that must be strictly
        followed, behavioral boundaries that define acceptable response patterns, quality thresholds
        that determine response adequacy, and time-sensitive or high-stakes operational requirements.

        Integration Notes
        -----------------
        - **Primacy Effect**: critical constraints appear early in the prompt for maximum impact
        - **Attention Allocation**: front-loading ensures these requirements receive priority
        processing
        - **Constraint Ordering**: multiple critical constraints maintain insertion order for
        hierarchical importance
        - **Quality Assurance**: critical constraints serve as quality gates for response evaluation
        - **Behavioral Anchoring**: works with persona to establish both identity and non-negotiable
        requirements

        The `.critical_constraint()` method ensures that the most important requirements are
        positioned for maximum attention and compliance, creating a foundation of non-negotiable
        standards that guide all subsequent reasoning and response generation.

        Examples
        --------
        ### Security-focused critical constraint

        Prioritize security considerations above all else:

        ```{python}
        import talk_box as tb

        # Security-first code review
        builder = (
            tb.PromptBuilder()
            .persona("senior security engineer", "application security")
            .critical_constraint("flag any security vulnerabilities immediately")
            .task_context("review this authentication implementation")
            .core_analysis([
                "input validation and sanitization",
                "authentication mechanisms",
                "authorization controls"
            ])
        )

        print(builder)
        ```

        ### Output format critical constraint

        Enforce strict output formatting requirements:

        ```{python}
        # Structured response requirement
        builder = (
            tb.PromptBuilder()
            .persona("data analyst", "business intelligence")
            .critical_constraint("provide exactly 3 key findings with supporting data")
            .task_context("analyze quarterly sales performance")
            .output_format([
                "Finding 1: [Insight] - [Supporting metric]",
                "Finding 2: [Insight] - [Supporting metric]",
                "Finding 3: [Insight] - [Supporting metric]"
            ])
        )

        print(builder)
        ```

        ### Behavioral boundary critical constraint

        Set clear behavioral boundaries for sensitive topics:

        ```{python}
        # Medical advice boundary
        builder = (
            tb.PromptBuilder()
            .persona("health information specialist")
            .critical_constraint(
                "do not provide specific medical diagnoses or treatment recommendations"
            )
            .task_context(
                "explain general wellness concepts and direct to healthcare professionals"
            )
        )

        print(builder)
        ```

        ### Quality threshold critical constraint

        Define minimum quality standards for responses:

        ```{python}
        # Production-ready focus
        builder = (
            tb.PromptBuilder()
            .persona("senior software architect", "enterprise systems")
            .critical_constraint("focus only on production-ready, scalable solutions")
            .task_context("design microservices architecture for high-traffic application")
            .core_analysis([
                "scalability patterns",
                "fault tolerance mechanisms",
                "performance optimization strategies"
            ])
        )

        print(builder)
        ```

        ### Multiple critical constraints with hierarchy

        Layer multiple critical requirements in order of importance:

        ```{python}
        # Hierarchical critical constraints
        builder = (
            tb.PromptBuilder()
            .persona("principal engineer", "financial systems")

            # First priority -- Regulatory compliance
            .critical_constraint("ensure all recommendations comply with financial regulations")

            # Second priority -- Proven solutions
            .critical_constraint("focus on solutions with proven track records in banking")

            # Third priority -- Security prioritization
            .critical_constraint("prioritize security over performance optimizations")

            .task_context("architect payment processing system for online banking")
        )

        print(builder)
        ```

        ### Time-sensitive critical constraint

        Handle urgent or time-critical requirements:

        ```{python}
        # Emergency response scenario
        builder = (
            tb.PromptBuilder()
            .persona("incident response specialist", "system outages")
            .critical_constraint("provide immediate actionable steps for system recovery")
            .task_context("diagnose and resolve database connection failures")
            .output_format([
                "immediate actions (next 5 minutes)",
                "short-term fixes (next hour)",
                "long-term prevention (next sprint)"
            ])
        )

        print(builder)
        ```

        ### Domain-specific critical constraint

        Apply domain-specific requirements that cannot be compromised. In this example, we focus on
        healthcare data privacy:

        ```{python}
        builder = (
            tb.PromptBuilder()
            .persona("healthcare data engineer", "HIPAA compliance")
            .critical_constraint("ensure all recommendations maintain patient data privacy")
            .task_context("design data pipeline for clinical research")
        )

        print(builder)
        ```

        ### Combining with other constraint types

        You can use critical constraints alongside standard constraints. Here, we combine two
        `.constraint()` calls with a front-loaded critical constraint:

        ```{python}
        # Comprehensive constraint strategy
        builder = (
            tb.PromptBuilder()
            .persona("technical lead", "code quality")
            .task_context("review pull request for production release")
            .critical_constraint("identify blocking issues that prevent deployment") # Critical
            .constraint("consider coding style consistency")                         # Standard
            .constraint("suggest performance improvements")                          # Standard
            .core_analysis([
                "security vulnerabilities",
                "logic errors and edge cases",
                "integration and compatibility issues"
            ])
        )

        print(builder)
        ```
        """
        self._constraints.insert(0, constraint)
        return self

    def core_analysis(self, analysis_points: List[str]) -> "PromptBuilder":
        """
        Define core analysis requirements as a high-priority, required structured section.

        The core analysis method creates the central analytical framework that defines what specific
        aspects must be examined and addressed in the AI's response. This method automatically
        creates a `"CORE ANALYSIS (Required)"` section with high priority placement, ensuring that
        the fundamental analytical requirements receive prominent attention and are treated as
        non-negotiable deliverables.

        Parameters
        ----------
        analysis_points
            List of specific analysis requirements that define the mandatory analytical dimensions.
            Each point should be clear, actionable, and represent a distinct aspect of the analysis.
            Points should be formulated as analytical objectives rather than general suggestions
            (e.g., `"evaluate security implementation patterns"` rather than `"look at security"`).

        Returns
        -------
        PromptBuilder
            Self for method chaining, allowing combination with other prompt building methods to
            create comprehensive, structured prompts.

        Research Foundation
        -------------------
        **Strategic Purpose Framework.** Core analysis requirements serve as the analytical backbone
        of the prompt, defining the specific dimensions of investigation that must be covered.
        Unlike general constraints or suggestions, core analysis points are treated as mandatory
        analytical objectives that structure the AI's systematic examination of the subject matter.

        **Attention Priority Theory.** This method automatically assigns `Priority.HIGH` and marks
        the section as required, ensuring that core analysis requirements are prominently positioned
        after critical constraints and task context but before standard constraints and formatting
        requirements. This placement leverages attention optimization principles to ensure
        analytical objectives receive appropriate focus.

        **Analytical Framework Design.** Each analysis point should represent a distinct analytical
        dimension or investigative angle that contributes to comprehensive coverage of the task. The
        points work together to create a systematic analytical framework that guides the AI's
        examination process and ensures thorough, structured analysis.

        **Quality Assurance Mechanism.** By marking core analysis as required, this method
        establishes analytical accountability and the AI must address each specified analysis point
        to provide a complete response. This prevents superficial analysis and ensures comprehensive
        coverage of critical analytical dimensions.

        Integration Notes
        -----------------
        - **Analytical Structure**: creates systematic framework for comprehensive analysis
        - **High Priority Placement**: automatically positioned prominently in the prompt hierarchy
        - **Required Coverage**: marked as required to ensure all analytical dimensions are
        addressed
        - **Quality Assurance**: establishes analytical accountability and prevents superficial
        responses
        - **Systematic Investigation**: guides AI through structured, thorough examination process
        - **Comprehensive Coverage**: ensures critical analytical aspects are not overlooked

        The `.core_analysis()` method provides the analytical backbone for sophisticated prompts,
        ensuring that complex tasks receive systematic, thorough examination across all critical
        dimensions while maintaining focus on the most important analytical objectives.

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
            .critical_constraint("focus on production-ready, scalable solutions")
            .task_context("review microservices architecture for e-commerce platform")
            .core_analysis([
                "evaluate service decomposition strategy and boundaries",
                "assess inter-service communication patterns and protocols",
                "analyze data consistency and transaction management approaches",
                "review scalability patterns and load distribution mechanisms",
                "examine security implementation across service boundaries"
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
            .critical_constraint("prioritize critical vulnerabilities that block deployment")
            .task_context("conduct comprehensive security audit of web application")
            .core_analysis([
                "analyze authentication and authorization mechanisms",
                "evaluate input validation and sanitization practices",
                "assess data protection and encryption implementations",
                "review API security and rate limiting strategies",
                "examine logging, monitoring, and incident response capabilities"
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
            .task_context("review pull request for production deployment")
            .core_analysis([
                "evaluate logic correctness and edge case handling",
                "assess performance implications and optimization opportunities",
                "review maintainability and code organization patterns",
                "analyze test coverage and quality assurance approaches",
                "examine security considerations and vulnerability patterns"
            ])
            .constraint("provide constructive feedback with learning opportunities")
            .constraint("include positive reinforcement for good practices")
        )
        ```

        ### Data science model analysis

        Structure analytical requirements for ML model evaluation:

        ```python
        # Machine learning model analysis
        builder = (
            tb.PromptBuilder()
            .persona("data scientist", "machine learning and model evaluation")
            .critical_constraint("include bias detection and fairness assessment")
            .task_context("evaluate machine learning model for production deployment")
            .core_analysis([
                "assess model accuracy across different demographic groups",
                "evaluate feature importance and model interpretability",
                "analyze training data quality and representation",
                "review model generalization and overfitting indicators",
                "examine deployment considerations and monitoring requirements"
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
            .task_context("analyze customer onboarding process for efficiency improvements")
            .core_analysis([
                "map current process flow and identify bottlenecks",
                "evaluate customer experience and friction points",
                "assess resource utilization and cost implications",
                "analyze compliance and risk management considerations",
                "identify automation opportunities and technology solutions"
            ])
            .constraint("support recommendations with quantitative analysis")
            .constraint("consider both short-term wins and long-term strategy")
        )
        ```

        ### Financial analysis framework

        Structure comprehensive financial evaluation:

        ```python
        # Financial performance analysis
        builder = (
            tb.PromptBuilder()
            .persona("financial analyst", "portfolio and risk management")
            .critical_constraint("include regulatory compliance considerations")
            .task_context("analyze investment portfolio performance and risk exposure")
            .core_analysis([
                "evaluate return performance across asset classes and time periods",
                "assess risk metrics including VaR, correlation, and concentration",
                "analyze portfolio diversification and asset allocation effectiveness",
                "review stress testing results and scenario analysis",
                "examine liquidity management and cash flow projections"
            ])
        )
        ```
        """
        return self.structured_section(
            "Core Analysis", analysis_points, priority=Priority.HIGH, required=True
        )

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

        Parameters
        ----------
        title
            Section heading that will be converted to uppercase for clear visual separation. Should
            be descriptive and specific to the content type (e.g., `"Review Areas"`,
            `"Performance Metrics"`, `"Security Requirements"`, etc.). The title helps create mental
            models for information organization.
        content
            Section content provided as either a single string or a list of items. When provided as
            a list, each item is automatically formatted with bullet points for clear visual
            organization. Content should be specific, actionable, and relevant to the section's
            purpose.
        priority
            Attention priority level for section placement in the final prompt structure. Higher
            priority sections appear earlier in the prompt to leverage primacy effects. Defaults to
            `Priority.MEDIUM` for balanced attention allocation.
        required
            Whether to mark the section as required in the output by appending `"(Required)"` to the
            section title. This visual indicator emphasizes critical sections that must be addressed
            in the response. Defaults to `False`.

        Returns
        -------
        PromptBuilder
            Self for method chaining, allowing combination with other prompt building methods to
            create comprehensive, structured prompts.

        Research Foundation
        -------------------
        **Attention Clustering Theory.** Creates distinct attention clusters for preventing
        attention drift in complex prompts. The structured approach leverages cognitive psychology
        principles of chunking and visual hierarchy to improve information processing and
        comprehension.

        **Cognitive Boundary Management.** Structured sections group related information together,
        creating focused attention zones that help the model process complex requirements
        systematically. This prevents attention from being scattered across disconnected information
        and maintains cognitive coherence throughout the prompt.

        **Visual Hierarchy Psychology.** Each section uses uppercase titles and consistent
        formatting to create clear visual boundaries. This visual organization helps both human
        readers and AI models navigate complex prompts more effectively by leveraging established
        patterns of visual information processing.

        **Priority-Based Information Architecture.** Sections are automatically ordered by priority
        and insertion order in the final prompt, ensuring that higher-priority content receives
        appropriate attention placement while maintaining logical information flow that aligns with
        cognitive processing patterns.

        Integration Notes
        -----------------
        - **Attention Clustering**: creates focused information zones that prevent cognitive
        overload
        - **Visual Organization**: consistent formatting improves prompt readability and navigation
        - **Priority-Based Ordering**: sections are automatically sorted by priority for optimal
        attention flow
        - **Flexible Content**: supports both single-string and list-based content organization
        - **Requirement Emphasis**: required sections receive visual emphasis to ensure coverage
        - **Cognitive Chunking**: information is organized in digestible units that align with human
        processing limits

        The `.structured_section()` method provides a powerful tool for organizing complex
        information in attention-optimized ways, enabling the creation of sophisticated prompts that
        maintain clarity and focus while addressing multiple aspects of complex tasks.

        Examples
        --------
        ### Basic structured section

        Create a simple section with clear organization:

        ```{python}
        import talk_box as tb

        # Single-item structured section
        builder = (
            tb.PromptBuilder()
            .persona("software architect", "system design")
            .task_context("review microservices architecture")
            .structured_section(
                "Architecture Principles",
                "focus on scalability, maintainability, and fault tolerance"
            )
        )

        print(builder)
        ```

        ### List-based structured section

        Use list format for multiple related items:

        ```{python}
        # Multi-item structured section
        builder = (
            tb.PromptBuilder()
            .persona("security engineer", "application security")
            .task_context("conduct security audit of web application")
            .structured_section(
                "Security Focus Areas", [
                    "authentication and authorization mechanisms",
                    "input validation and sanitization",
                    "data encryption and protection",
                    "API security and rate limiting"
                ]
            )
        )

        print(builder)
        ```

        ### High-priority required section

        Create critical sections that must be addressed:

        ```{python}
        # High-priority required section
        builder = (
            tb.PromptBuilder()
            .persona("data scientist", "machine learning")
            .task_context("evaluate model performance and bias")
            .structured_section(
                "Model Validation", [
                    "accuracy metrics across demographic groups",
                    "bias detection and mitigation strategies",
                    "cross-validation and generalization testing",
                    "ethical considerations and fairness metrics"
                ],
                priority=tb.Priority.HIGH,
                required=True
            )
        )

        print(builder)
        ```

        ### Multiple sections with different priorities

        Build comprehensive prompts with multiple organized sections:

        ```{python}
        # Complex prompt with multiple structured sections
        builder = (
            tb.PromptBuilder()
            .persona("technical lead", "code review and mentorship")
            .critical_constraint("focus on production readiness and team learning")
            .task_context("review pull request for junior developer")
            .structured_section(
                "Code Quality Assessment", [
                    "logic correctness and edge case handling",
                    "security vulnerabilities and best practices",
                    "performance implications and optimizations",
                    "code readability and maintainability"
                ],
                priority=tb.Priority.HIGH,
                required=True
            )
            .structured_section(
                "Learning Opportunities", [
                    "design patterns that could be applied",
                    "best practices worth highlighting",
                    "areas for skill development",
                    "recommended learning resources"
                ],
                priority=tb.Priority.MEDIUM
            )
            .structured_section(
                "Team Knowledge Sharing", [
                    "patterns that could be standardized",
                    "documentation improvements needed",
                    "opportunities for pair programming",
                    "code that exemplifies good practices"
                ],
                priority=tb.Priority.LOW
            )
        )

        print(builder)
        ```

        ### Domain-specific structured sections

        Create sections tailored to specific industries or contexts. Here's an example for
        healthcare:

        ```{python}
        builder = (
            tb.PromptBuilder()
            .persona("healthcare software architect", "HIPAA compliance")
            .task_context("review patient data management system")
            .structured_section(
                "HIPAA Compliance Requirements", [
                    "patient data encryption and access controls",
                    "audit trail and logging mechanisms",
                    "data minimization and retention policies",
                    "breach detection and notification procedures"
                ],
                priority=tb.Priority.CRITICAL,
                required=True
            )
        )

        print(builder)
        ```

        And here's one for finance:

        ```{python}
        builder = (
            tb.PromptBuilder()
            .persona("financial systems architect", "regulatory compliance")
            .task_context("design trading system architecture")
            .structured_section(
                "Regulatory Considerations", [
                    "market data handling and latency requirements",
                    "trade reporting and compliance monitoring",
                    "risk management and circuit breakers",
                    "audit trails and regulatory reporting"
                ],
                priority=tb.Priority.HIGH,
                required=True
            )
        )

        print(builder)
        ```

        ### Performance and optimization sections

        Structure performance-related requirements clearly:

        ```{python}
        builder = (
            tb.PromptBuilder()
            .persona("performance engineer", "web application optimization")
            .task_context("optimize application performance for high traffic")
            .structured_section(
                "Performance Targets", [
                    "page load times under 2 seconds",
                    "API response times under 100ms",
                    "support for 10,000 concurrent users",
                    "99.9% uptime availability"
                ],
                priority=tb.Priority.HIGH,
                required=True
            )
            .structured_section(
                "Optimization Areas", [
                    "frontend asset optimization and caching",
                    "database query performance and indexing",
                    "CDN implementation and edge caching",
                    "server-side rendering and lazy loading"
                ],
                priority=tb.Priority.MEDIUM
            )
        )

        print(builder)
        ```

        ### Research and analysis sections

        Structure analytical requirements and methodologies:

        ```{python}
        # Research analysis prompt
        builder = (
            tb.PromptBuilder()
            .persona("research analyst", "market intelligence")
            .task_context("analyze emerging technology adoption trends")
            .structured_section(
                "Research Methodology", [
                    "quantitative data analysis and statistical testing",
                    "qualitative interviews and survey analysis",
                    "competitive landscape and market mapping",
                    "trend analysis and future projections"
                ],
                priority=tb.Priority.HIGH,
                required=True
            )
            .structured_section(
                "Deliverable Requirements", [
                    "executive summary with key findings",
                    "detailed methodology and data sources",
                    "visual charts and trend illustrations",
                    "actionable recommendations and next steps"
                ],
                priority=tb.Priority.MEDIUM,
                required=True
            )
        )

        print(builder)
        ```

        ### Mixed content types in sections

        Combine different content formats within sections:

        ```{python}
        builder = (
            tb.PromptBuilder()
            .persona("technical writer", "API documentation")
            .task_context("create comprehensive API documentation")
            .structured_section(
                "documentation Standards",
                "follow OpenAPI 3.0 specification for consistency and completeness"
            )
            .structured_section(
                "Required Documentation Elements", [
                    "endpoint descriptions with purpose and usage",
                    "request/response schemas with examples",
                    "authentication and authorization details",
                    "error codes and troubleshooting guidance"
                ],
                required=True
            )
        )

        print(builder)
        ```
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

    def constraint(self, constraint: str) -> "PromptBuilder":
        """
        Add a standard constraint to the prompt that will appear in the additional constraints
        section.

        Standard constraints are important requirements and guidelines that shape the AI's response
        but are not as critical as front-loaded constraints. These constraints appear in the
        `ADDITIONAL CONSTRAINTS` section after the main task context and structured sections,
        providing important guidance while maintaining the attention hierarchy of the prompt.

        Parameters
        ----------
        constraint
            Specific constraint, requirement, or guideline that should influence the AI's response.
            Should be clear and actionable, using directive language when appropriate (e.g.,
            `"Use clear, concise language"`, `"Include practical examples"`,
            `"Avoid overly technical jargon"`, etc.).

        Returns
        -------
        PromptBuilder
            Self for method chaining, allowing combination with other prompt building methods to
            create comprehensive, structured prompts.

        Research Foundation
        -------------------
        **Positioning Strategy Theory.** Standard constraints are positioned after critical
        constraints and core content to maintain optimal attention flow. This positioning ensures
        that essential task information receives primary focus while still communicating important
        requirements and preferences to the model.

        **Constraint Hierarchy Management.** Standard constraints appear in the order they are
        added, after any critical constraints. This allows for logical grouping of related
        requirements and systematic constraint organization that respects cognitive processing
        priorities.

        **Use Case Classification.** Standard constraints are ideal for quality preferences and
        style guidelines, secondary requirements that enhance output quality, behavioral preferences
        that improve response tone, technical preferences for implementation approaches, and
        context-specific guidelines that refine the response scope.

        **Differentiated Constraint Strategy.** While `critical_constraint()` is used for
        non-negotiable requirements that must be front-loaded, `constraint()` is used for important
        but secondary requirements that guide response quality and style without overriding primary
        attention allocation.

        Integration Notes
        -----------------
        - **Attention Hierarchy**: standard constraints appear after critical content to maintain
        focus
        - **Quality Enhancement**: these constraints refine and improve response quality without
        overriding priorities
        - **Flexibility**: supports diverse requirement types from technical to behavioral to
        domain specific
        - **Systematic Organization**: constraints are grouped logically in the final prompt
        structure
        - **Complementary Function**: works alongside critical constraints to create comprehensive
        requirement sets

        The `.constraint()` method provides flexible, systematic way to communicate important
        requirements and preferences that enhance response quality while respecting the overall
        attention optimization strategy of the prompt building system.

        Examples
        --------
        ### Quality and style constraints

        Add constraints that improve response quality and consistency:

        ```{python}
        import talk_box as tb

        # Documentation quality constraints
        builder = (
            tb.PromptBuilder()
            .persona("technical writer", "API documentation")
            .task_context("create user guide for authentication API")
            .constraint("use clear, concise language appropriate for developers")
            .constraint("include practical code examples for each endpoint")
            .constraint("provide troubleshooting guidance for common issues")
            .core_analysis([
                "authentication flow and requirements",
                "error handling and status codes",
                "rate limiting and best practices"
            ])
        )

        print(builder)
        ```

        ### Technical preference constraints

        Guide implementation approaches and technical choices:

        ```{python}
        # Architecture review with technical preferences
        builder = (
            tb.PromptBuilder()
            .persona("senior software architect", "microservices")
            .critical_constraint("focus only on production-ready patterns")
            .task_context("review microservices architecture design")
            .constraint("prefer established patterns over novel approaches")
            .constraint("consider scalability implications for each recommendation")
            .constraint("include performance trade-offs in analysis")
            .core_analysis([
                "service decomposition strategy",
                "inter-service communication patterns",
                "data consistency approaches"
            ])
        )
        ```

        ### Behavioral and tone constraints

        Shape the AI's communication style and approach:

        ```{python}
        # Code review with specific behavioral guidance
        builder = (
            tb.PromptBuilder()
            .persona("senior developer", "code quality")
            .task_context("review pull request for junior developer")
            .constraint("provide constructive, encouraging feedback")
            .constraint("explain the reasoning behind each suggestion")
            .constraint("include positive reinforcement for good practices")
            .constraint("suggest learning resources for improvement areas")
            .core_analysis([
                "code correctness and logic",
                "security considerations",
                "maintainability and readability"
            ])
        )
        ```

        ### Context-specific constraints

        Add domain or situation-specific requirements. In this example for a healthcare application,
        we focus on HIPAA compliance and patient privacy.

        ```{python}
        builder = (
            tb.PromptBuilder()
            .persona("healthcare software architect", "HIPAA compliance")
            .critical_constraint("all recommendations must maintain patient privacy")
            .task_context("design patient data management system")
            .constraint("consider healthcare industry regulations")
            .constraint("prioritize data security over performance optimizations")
            .constraint("include audit trail requirements in recommendations")
        )

        print(builder)
        ```

        ### Multiple related constraints

        Group related constraints for comprehensive guidance. This example focuses on data analysis
        with multiple quality constraints:

        ```{python}
        builder = (
            tb.PromptBuilder()
            .persona("data scientist", "business analytics")
            .task_context("analyze customer behavior patterns")
            .constraint("support findings with statistical evidence")
            .constraint("use clear visualizations to illustrate trends")
            .constraint("explain methodology and assumptions clearly")
            .constraint("provide actionable business recommendations")
            .constraint("include confidence levels for predictions")
            .core_analysis([
                "customer segmentation patterns",
                "behavioral trend analysis",
                "predictive modeling opportunities"
            ])
        )

        print(builder)
        ```

        ### Combining with critical constraints

        Use standard constraints to complement critical requirements:

        ```{python}
        builder = (
            tb.PromptBuilder()
            .persona("security engineer", "application security")
            .critical_constraint("Identify blocking security vulnerabilities immediately")
            .task_context("security audit of web application")
            .constraint("consider OWASP Top 10 guidelines")                    # Standard
            .constraint("evaluate both code and infrastructure security")      # Standard
            .constraint("provide remediation priority levels")                 # Standard
            .constraint("include compliance implications where relevant")      # Standard
            .core_analysis([
                "authentication and authorization",
                "input validation and sanitization",
                "data protection and encryption"
            ])
        )

        print(builder)
        ```

        ### Output enhancement constraints

        We can improve the structure and usability of responses by adding quality-focused
        constraints:

        ```{python}
        builder = (
            tb.PromptBuilder()
            .persona("technical documentation specialist")
            .task_context("create troubleshooting guide for deployment issues")
            .constraint("organize information from most common to least common issues")
            .constraint("include step-by-step resolution procedures")
            .constraint("provide prevention strategies for each issue type")
            .constraint("use consistent formatting and terminology throughout")
            .output_format([
                "issue description and symptoms",
                "root cause analysis",
                "step-by-step resolution",
                "prevention recommendations"
            ])
        )

        print(builder)
        ```
        """
        self._constraints.append(constraint)
        return self

    def avoid_topics(self, topics: List[str]) -> "PromptBuilder":
        """
        Specify topics or behaviors to avoid through negative constraints that guide AI responses
        away from unwanted content.

        Negative constraints provide explicit guidance about what the AI should not include or
        discuss in its response, creating clear boundaries that prevent unwanted content,
        inappropriate suggestions, or off-topic discussions. This method adds an "Avoid:" constraint
        that appears in the standard constraints section, providing clear guidance about prohibited
        topics or approaches.

        Parameters
        ----------
        topics
            List of specific topics, behaviors, approaches, or content areas that should be
            explicitly avoided in the response. Each item should be clearly defined and specific
            enough to provide clear guidance (e.g., `"controversial political opinions"`,
            `"deprecated technologies"`, `"cost-cutting through layoffs"`,
            `"quick fixes without testing"`, etc.).

        Returns
        -------
        PromptBuilder
            Self for method chaining, allowing combination with other prompt building methods to
            create comprehensive, structured prompts.

        Research Foundation
        -------------------
        **Negative Guidance Psychology.** Research in cognitive psychology shows that explicit
        negative instructions can be effective when combined with positive guidance. By clearly
        stating what to avoid, this method helps the AI navigate complex topics while staying within
        appropriate boundaries and maintaining focus on desired outcomes.

        **Boundary Setting Mechanism.** Avoid topics serves as a content filter and
        boundary-setting mechanism that prevents responses from venturing into sensitive,
        irrelevant, or counterproductive areas. This is particularly valuable for professional
        contexts where certain topics or approaches could be inappropriate or harmful.

        **Risk Mitigation Strategy.** Negative constraints help mitigate risks associated with
        AI-generated content by explicitly excluding potentially problematic topics, biased
        perspectives, or approaches that could lead to harmful or inappropriate recommendations.

        **Focus Enhancement Theory.** By eliminating distracting or irrelevant topics,
        `.avoid_topics()` helps maintain laser focus on the core objectives and prevents the AI from
        exploring tangential areas that might dilute the quality or relevance of the response.

        Integration Notes
        -----------------
        - **Boundary Setting**: establishes clear content and approach boundaries for AI responses
        - **Risk Mitigation**: prevents problematic or inappropriate content through explicit
        exclusion
        - **Focus Enhancement**: eliminates distracting topics to maintain response relevance
        - **Professional Standards**: ensures responses align with ethical and professional
        guidelines
        - **Quality Assurance**: prevents low-quality approaches through negative guidance
        - **Complementary Constraints**: works alongside positive constraints to create
        comprehensive guidance

        The `.avoid_topics()` method provides essential boundary-setting capabilities that ensure AI
        responses remain appropriate, focused, and aligned with professional standards while
        explicitly excluding problematic approaches or content areas that could compromise response
        quality or appropriateness.

        Examples
        --------
        ### Technical architecture review

        Avoid outdated or problematic technologies and approaches for a bot focused on modern
        software architecture:

        ```{python}
        import talk_box as tb

        builder = (
            tb.PromptBuilder()
            .persona("solution architect", "modern enterprise systems")
            .task_context("design scalable microservices architecture for e-commerce platform")
            .core_analysis([
                "service decomposition strategy",
                "inter-service communication patterns",
                "data consistency approaches",
                "scalability and performance optimization"
            ])
            .avoid_topics([
                "monolithic architecture patterns",
                "deprecated Java EE technologies",
                "synchronous blocking communication",
                "database shared between services",
                "manual deployment processes"
            ])
            .output_format([
                "architecture overview with service boundaries",
                "technology stack recommendations",
                "implementation roadmap with phases"
            ])
        )

        print(builder)
        ```

        ### Business strategy consultation

        Avoid ethically questionable or short-term approaches when advising on business strategy:

        ```{python}
        builder = (
            tb.PromptBuilder()
            .persona("business consultant", "sustainable growth strategies")
            .task_context("develop growth strategy for struggling retail company")
            .core_analysis([
                "market positioning and competitive advantages",
                "operational efficiency improvements",
                "customer experience enhancements",
                "revenue diversification opportunities"
            ])
            .avoid_topics([
                "mass layoffs as primary cost reduction",
                "exploiting regulatory loopholes",
                "aggressive customer data monetization",
                "environmental impact trade-offs for profit",
                "anti-competitive pricing strategies"
            ])
            .constraint("focus on sustainable, long-term solutions")
            .output_format([
                "strategic assessment with market analysis",
                "growth initiatives with ethical considerations",
                "implementation timeline with stakeholder impact"
            ])
        )

        print(builder)
        ```

        ### Security audit guidance

        Avoid security through obscurity and weak practices:

        ```{python}
        builder = (
            tb.PromptBuilder()
            .persona("security engineer", "application security best practices")
            .task_context("audit web application security for financial services company")
            .core_analysis([
                "authentication and authorization mechanisms",
                "data protection and encryption standards",
                "input validation and sanitization",
                "infrastructure security configuration"
            ])
            .avoid_topics([
                "security through obscurity approaches",
                "custom cryptographic implementations",
                "storing passwords in plain text or weak hashing",
                "disabling security features for convenience",
                "ignoring OWASP recommendations"
            ])
            .critical_constraint("all recommendations must follow industry security standards")
            .output_format([
                "security assessment with risk levels",
                "critical vulnerabilities requiring immediate attention",
                "best practice implementation roadmap"
            ])
        )

        print(builder)
        ```

        ### Educational content development

        Avoid outdated or confusing learning approaches by specifying poor pedagogical practices:

        ```{python}
        builder = (
            tb.PromptBuilder()
            .persona("instructional designer", "modern programming education")
            .task_context("create comprehensive Python programming curriculum for beginners")
            .core_analysis([
                "progressive skill building sequence",
                "hands-on practice opportunities",
                "real-world application examples",
                "common mistake prevention strategies"
            ])
            .avoid_topics([
                "memorization-based learning without understanding",
                "outdated Python 2.x syntax and practices",
                "complex theoretical concepts before practical foundation",
                "overwhelming students with too many options",
                "abstract examples without real-world relevance"
            ])
            .constraint("include diverse learning styles and accessibility considerations")
            .output_format([
                "curriculum structure with learning objectives",
                "module breakdown with practical exercises",
                "assessment strategies and progress tracking"
            ])
        )

        print(builder)
        ```

        ### Code review guidance

        Avoid problematic coding practices and shortcuts. For this example, we create a code review
        prompt that emphasizes constructive feedback while avoiding poor development practices:

        ```{python}
        builder = (
            tb.PromptBuilder()
            .persona("senior developer", "code quality and best practices")
            .task_context("review pull request for production deployment")
            .core_analysis([
                "code correctness and functionality",
                "security vulnerability assessment",
                "performance implications and optimization",
                "maintainability and documentation quality"
            ])
            .avoid_topics([
                "quick fixes that introduce technical debt",
                "skipping unit tests for faster delivery",
                "hard-coding configuration values",
                "ignoring error handling for edge cases",
                "copy-pasting code without understanding"
            ])
            .constraint("provide constructive feedback with learning opportunities")
            .output_format([
                "code quality assessment with specific examples",
                "security and performance concerns",
                "improvement recommendations with rationale"
            ])
        )

        print(builder)
        ```
        """
        # Create strong refusal language instead of weak "avoid" guidance
        if len(topics) == 1:
            refusal_text = (
                f"IMPORTANT CONSTRAINT: You MUST NOT provide any information, advice, or discussion about {topics[0]}. "
                f"If asked about {topics[0]}, politely decline and redirect by saying something to the effect of "
                f"'I'm not able to help with {topics[0]}. Is there something else I can assist you with instead?' "
                f"(adapt the language and phrasing to match the conversation's language and tone)."
            )
        else:
            topics_list = ", ".join(topics[:-1]) + f", or {topics[-1]}"
            refusal_text = (
                f"IMPORTANT CONSTRAINT: You MUST NOT provide any information, advice, or discussion about {topics_list}. "
                f"If asked about any of these topics, politely decline and redirect by saying something to the effect of "
                f"'I'm not able to help with that topic. Is there something else I can assist you with instead?' "
                f"(adapt the language and phrasing to match the conversation's language and tone)."
            )

        return self.constraint(refusal_text)

    def output_format(self, format_specs: List[str]) -> "PromptBuilder":
        """
        Specify output formatting requirements to prevent ambiguous responses and ensure structured
        deliverables.

        Output formatting requirements define the structural and organizational expectations for the
        AI's response, providing clear specifications that prevent ambiguous or inconsistently
        formatted outputs. These requirements appear in the `"OUTPUT FORMAT"` section near the end
        of the prompt, ensuring that formatting guidance influences response generation while
        maintaining the attention hierarchy for more critical content.

        Parameters
        ----------
        format_specs
            List of specific formatting requirements that define how the response should be
            structured and organized. Each specification should be clear, actionable, and measurable
            when possible. Specifications can address organization, headings, lists, examples,
            priorities, or any structural aspects of the response (e.g.,
            `"Start with executive summary"`, `"Use bullet points for key findings"`,
            `"Include code examples for each recommendation"`).

        Returns
        -------
        PromptBuilder
            Self for method chaining, allowing combination with other prompt building methods to
            create comprehensive, structured prompts.

        Research Foundation
        -------------------
        **Attention Drift Mitigation.** Addresses attention drift issues by providing specific,
        measurable formatting constraints that anchor response structure. Clear formatting
        requirements help maintain cognitive coherence and ensure that complex responses remain
        organized and accessible to human readers.

        **Structural Guidance Framework.** Output format specifications serve as response templates
        that guide the AI's information organization and presentation. Unlike content-focused
        constraints, these requirements focus on how information should be structured, ordered, and
        presented to maximize clarity and usability.

        **Response Quality Enhancement.** Well-defined formatting requirements significantly improve
        response quality by preventing stream-of-consciousness outputs and ensuring systematic
        information organization. This is particularly important for complex analytical tasks where
        information hierarchy and clear structure are essential for comprehension.

        **Professional Standards Alignment.** Formatting specifications enable alignment with
        professional documentation standards, report formats, and organizational communication
        preferences, ensuring that AI-generated content meets workplace and industry expectations.

        Integration Notes
        -----------------
        - **Response Structure**: provides clear templates for organized, professional outputs
        - **Cognitive Clarity**: prevents stream-of-consciousness responses through structured
        guidance
        - **Quality Assurance**: ensures consistent formatting that meets professional standards
        - **Information Hierarchy**: guides appropriate organization of complex information
        - **Accessibility**: improves readability and navigability of AI-generated content
        - **Professional Alignment**: enables compliance with organizational communication standards

        The `.output_format()` method ensures that AI responses are well-structured, professionally
        formatted, and organized in ways that maximize clarity, usability, and impact for human
        readers across diverse professional contexts.

        Examples
        --------
        ### Basic formatting requirements

        Define clear structure for analytical responses in code reviews with bullet points and
        summaries within the `.output_format()` method:

        ```{python}
        import talk_box as tb

        builder = (
            tb.PromptBuilder()
            .persona("senior developer", "code review")
            .task_context("review pull request for production deployment")
            .core_analysis([
                "security vulnerabilities and risks",
                "performance implications and optimizations",
                "code quality and maintainability issues"
            ])
            .output_format([
                "start with overall assessment (approve/request changes)",
                "list critical issues that must be fixed",
                "provide suggestions for improvements",
                "include positive feedback on good practices"
            ])
        )

        print(builder)
        ```

        ### Executive reporting format

        Structure responses for business stakeholders with clear sections and prioritized findings:

        ```{python}
        builder = (
            tb.PromptBuilder()
            .persona("business analyst", "strategic planning")
            .task_context("analyze market expansion opportunity")
            .core_analysis([
                "market size and growth potential",
                "competitive landscape analysis",
                "risk assessment and mitigation strategies",
                "resource requirements and timeline"
            ])
            .output_format([
                "executive summary (2-3 key sentences)",
                "detailed findings with supporting data",
                "risk assessment with mitigation strategies",
                "recommended action items with priorities",
                "timeline and resource requirements"
            ])
        )

        print(builder)
        ```

        ### Technical documentation format

        Structure comprehensive technical documentation with clear sections and examples:

        ```{python}
        # Technical documentation formatting
        builder = (
            tb.PromptBuilder()
            .persona("technical writer", "API documentation")
            .task_context("create comprehensive API reference documentation")
            .core_analysis([
                "endpoint functionality and purpose",
                "request/response schemas and examples",
                "authentication and authorization requirements",
                "error handling and status codes"
            ])
            .output_format([
                "overview section with API purpose and scope",
                "authentication section with setup instructions",
                "endpoint documentation with examples",
                "error codes reference with troubleshooting",
                "SDK and integration examples"
            ])
        )

        print(builder)
        ```

        ### Research and analysis format

        Structure academic or research-style outputs with an output format that includes
        methodology, findings, and recommendations:

        ```{python}
        # Research analysis formatting
        builder = (
            tb.PromptBuilder()
            .persona("research analyst", "data science")
            .task_context("analyze customer behavior patterns from survey data")
            .core_analysis([
                "demographic segmentation and trends",
                "behavioral pattern identification",
                "statistical significance of findings",
                "predictive modeling opportunities"
            ])
            .output_format([
                "methodology section with data sources and approach",
                "key findings with statistical evidence",
                "visual descriptions for charts and graphs",
                "limitations and confidence intervals",
                "recommendations with supporting rationale"
            ])
        )

        print(builder)
        ```
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

Parameters
----------
input_example
    Example input that represents a typical or representative case that the AI might
    encounter. Should be realistic, relevant to the task context, and demonstrate the type
    and complexity of input the AI will be processing. The input should be specific enough
    to provide clear guidance while being generalizable to similar scenarios.
output_example
    Expected output format and content that demonstrates the desired response style, level
    of detail, structure, and quality. Should exemplify the formatting requirements,
    analytical depth, and professional standards expected in the actual response. The
    output should be comprehensive enough to serve as a template while being specific to
    the input provided.

Returns
-------
PromptBuilder
    Self for method chaining, allowing combination with other prompt building methods to
    create comprehensive, structured prompts.

Research Foundation
-------------------
**Few-Shot Learning Theory.** Examples leverage the AI model's ability to learn from
demonstrations without explicit training. By providing concrete input/output pairs,
examples enable the model to infer patterns, styles, and expected behaviors that might be
difficult to specify through constraints alone.

**Response Calibration Mechanism.** Examples serve as calibration tools that help establish
the appropriate level of detail, technical depth, formatting style, and analytical approach
for responses. This is particularly valuable for complex tasks where abstract descriptions
of requirements might be ambiguous.

**Pattern Recognition Enhancement.** Multiple examples can demonstrate variations in
approach, showing how different types of inputs should be handled while maintaining
consistent output quality and format. This helps the AI generalize appropriately across
different scenarios.

**Quality Anchoring Psychology.** Examples set quality expectations by demonstrating
high-quality responses that serve as benchmarks for the AI's own outputs. This helps
maintain consistency and professionalism across different prompt executions.

Integration Notes
-----------------
- **Few-Shot Learning**: leverages AI's pattern recognition for improved response quality
- **Format Demonstration**: shows concrete examples of expected output structure and style
- **Quality Calibration**: establishes benchmarks for response depth and professionalism
- **Variation Handling**: multiple examples can demonstrate different scenarios and approaches
- **Learning Reinforcement**: examples reinforce other prompt elements like constraints and
formatting
- **Prompt Positioning**: examples appear late in prompt to provide final guidance before response
generation

The `.example()` method provides powerful demonstration-based learning that significantly
improves response quality, consistency, and alignment with expectations through concrete
input/output pattern recognition rather than abstract instruction following.

Examples
--------
### Code review example

Demonstrate expected code review format and depth while providing constructive feedback:

````{python}
import talk_box as tb

builder = (
    tb.PromptBuilder()
    .persona("senior developer", "code quality and security")
    .task_context("review Python code for security and best practices")
    .core_analysis([
        "security vulnerabilities and risks",
        "code quality and maintainability",
        "performance optimization opportunities",
        "best practice adherence"
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
- **Problem**: direct string concatenation in SQL query allows SQL injection attacks
- **Risk Level**: high, and could lead to a data breach or unauthorized access
- **Fix**: use parameterized queries or ORM methods
- **Example Fix**:
  ```python
  query = "SELECT * FROM users WHERE username = %s AND password = %s"
  result = db.execute(query, (username, password))
  ```

**SECURITY ISSUE**: Plain Text Password Storage
- **Problem**: passwords should never be stored or compared in plain text
- **Fix**: implement password hashing with salt (e.g., bcrypt, scrypt)

**CODE QUALITY**: function should return user object, not boolean
**PERFORMANCE**: consider adding database indexes on username field
        '''
    )
    .output_format([
        "start with critical security issues",
        "include specific code examples and fixes",
        "provide risk assessment for each issue",
        "end with positive feedback where applicable"
    ])
)

print(builder)
````

### Data analysis example

Demonstrate analytical depth and presentation style using structured findings and recommendations:

````{python}
# Data analysis with example
builder = (
    tb.PromptBuilder()
    .persona("data scientist", "business analytics")
    .task_context("analyze customer behavior data and provide insights")
    .core_analysis([
        "customer segmentation patterns",
        "behavioral trends and anomalies",
        "statistical significance of findings",
        "business recommendations"
    ])
    .example(
        input_example="customer purchase data showing 15% increase in mobile transactions but 8% decrease in desktop purchases over Q3",
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

print(builder)
````

### Business analysis example

Demonstrate strategic analysis format in business contexts with clear sections and actionable
recommendations:

````{python}
builder = (
    tb.PromptBuilder()
    .persona("business consultant", "strategic planning")
    .task_context("analyze market expansion opportunity")
    .core_analysis([
        "market size and growth potential",
        "competitive landscape assessment",
        "resource requirements and ROI",
        "risk factors and mitigation"
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

print(builder)
````

### Educational content example

Show educational content format with clear learning objectives and hands-on exercises:

````{python}
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

print(builder)
````

### Multiple examples for variation

Use multiple examples to show different scenarios and approaches within the same prompt:

````{python}
builder = (
    tb.PromptBuilder()
    .persona("senior developer", "code mentorship")
    .task_context("provide educational code review for junior developers")
    .core_analysis([
        "code correctness and logic",
        "best practices and patterns",
        "performance considerations",
        "learning opportunities"
    ])
    .example(
        input_example="simple function with basic logic error",
        output_example="focus on explaining the logic error clearly with corrected version and learning points"
    )
    .example(
        input_example="complex function with performance issues",
        output_example="analyze algorithmic complexity, suggest optimizations, explain trade-offs between readability and performance"
    )
    .example(
        input_example="well-written code with minor style issues",
        output_example="acknowledge good practices, suggest minor improvements, reinforce positive patterns"
    )
)

print(builder)
````
        """
        # fmt: on
        self._examples.append({"input": input_example, "output": output_example})
        return self

    def final_emphasis(self, emphasis: str) -> "PromptBuilder":
        """
        Set final emphasis that leverages recency bias to ensure critical instructions receive
        maximum attention.

        Final emphasis strategically positions the most important instruction at the very end of the
        system prompt, leveraging the psychological principle of recency bias to ensure that
        critical guidance remains fresh in the AI's attention during response generation. This
        method provides a powerful way to reinforce the most essential requirement or constraint
        that must not be overlooked.

        Parameters
        ----------
        emphasis
            The most critical instruction or objective that must receive primary attention during
            response generation. Should be formulated as a clear, actionable directive that captures
            the essential requirement (e.g.,
            `"focus your entire response on practical implementation steps"`,
            `"prioritize security considerations above all else"`,
            `"ensure all recommendations are cost-effective and implementable"`, etc.).

        Returns
        -------
        PromptBuilder
            Self for method chaining, allowing combination with other prompt building methods to
            create comprehensive, structured prompts.

        Research Foundation
        -------------------
        **Recency Bias Psychology.** Research in cognitive psychology demonstrates that information
        presented at the end of a sequence receives heightened attention and retention. By placing
        critical instructions at the prompt's conclusion, final emphasis ensures that the most
        important guidance influences the AI's response generation process when attention is most
        focused on producing output.

        **Attention Anchoring Mechanism.** Final emphasis serves as an attention anchor that
        prevents drift from core objectives during complex prompt processing. When prompts contain
        extensive context, constraints, and examples, the final emphasis acts as a cognitive reset
        that refocuses attention on the primary objective before response generation begins.

        **Override Mechanism Theory.** Final emphasis can serve as an override mechanism for complex
        prompts where multiple competing priorities might create confusion. By explicitly stating
        the most critical requirement at the end, this method ensures that primary objectives take
        precedence over secondary considerations when trade-offs must be made.

        **Quality Assurance Strategy.** The strategic placement of final emphasis helps prevent AI
        responses that technically satisfy prompt requirements but miss the primary intent. This is
        particularly valuable for complex analytical tasks where technical completeness might
        overshadow the core objective.

        Integration Notes
        -----------------
        - **Recency Bias Leverage**: strategically positions critical guidance at prompt conclusion
        for maximum impact
        - **Attention Anchoring**: prevents objective drift during complex prompt processing
        - **Priority Override**: ensures primary objectives take precedence when trade-offs are
        required
        - **Quality Assurance**: prevents technically complete but intent-missing responses
        - **Cognitive Reset**: refocuses attention on core objectives before response generation
        - **Strategic Positioning**: complements front-loaded critical constraints with
        end-positioned emphasis

        The `.final_emphasis()` method provides a powerful attention management tool that ensures
        the most critical requirements maintain prominence throughout the AI's response generation
        process, leveraging psychological principles to maximize adherence to primary objectives.

        Examples
        --------
        ### Security-focused analysis

        Ensure security remains the primary consideration despite other requirements using the
        `.final_emphasis()` method:

        ```{python}
        import talk_box as tb

        builder = (
            tb.PromptBuilder()
            .persona("security engineer", "application security")
            .task_context("review web application for deployment readiness")
            .core_analysis([
                "authentication and authorization mechanisms",
                "input validation and data sanitization",
                "infrastructure security configuration",
                "compliance with security standards"
            ])
            .constraint("include performance optimization suggestions")
            .constraint("consider user experience implications")
            .output_format([
                "executive summary with risk assessment",
                "critical security issues requiring immediate attention",
                "performance and UX recommendations where applicable"
            ])
            .final_emphasis(
                "security vulnerabilities must be identified and addressed before "
                "any performance or UX considerations"
            )
        )

        print(builder)
        ```

        ### Cost-conscious recommendations

        Emphasize budget constraints in business analysis by placing cost considerations at the end:

        ```{python}
        builder = (
            tb.PromptBuilder()
            .persona("business consultant", "strategic planning")
            .task_context("develop growth strategy for startup with limited funding")
            .core_analysis([
                "market opportunity assessment",
                "competitive landscape analysis",
                "resource requirements and scaling plan",
                "revenue generation strategies"
            ])
            .constraint("include innovative growth tactics")
            .constraint("consider partnership opportunities")
            .output_format([
                "executive summary with growth potential",
                "detailed strategy with implementation phases",
                "resource allocation and timeline"
            ])
            .final_emphasis(
                "all recommendations must be implementable with minimal upfront investment "
                "and show clear ROI within 6 months"
            )
        )

        print(builder)
        ```

        ### Learning-focused code review

        Prioritize educational value in technical feedback through the use of final emphasis:

        ```{python}
        builder = (
            tb.PromptBuilder()
            .persona("senior developer", "mentorship and code quality")
            .task_context("review junior developer's code for learning and improvement")
            .core_analysis([
                "code correctness and functionality",
                "best practices and design patterns",
                "performance optimization opportunities",
                "security considerations"
            ])
            .constraint("identify areas for improvement")
            .constraint("provide specific examples and fixes")
            .output_format([
                "overall assessment with learning objectives",
                "technical issues with explanations and solutions",
                "positive reinforcement for good practices"
            ])
            .final_emphasis(
                "frame all feedback as learning opportunities with clear explanations of why "
                "changes improve the code"
            )
        )

        print(builder)
        ```

        ### User experience priority

        Ensure UX considerations override technical preferences. This is particularly important in
        product management and design contexts:

        ```{python}
        builder = (
            tb.PromptBuilder()
            .persona("product manager", "user experience and design")
            .task_context("evaluate new feature proposal for mobile application")
            .core_analysis([
                "user needs and problem-solution fit",
                "technical implementation complexity",
                "performance and scalability impact",
                "business value and metrics"
            ])
            .constraint("consider technical feasibility constraints")
            .constraint("include development effort estimates")
            .output_format([
                "feature assessment with user impact analysis",
                "implementation recommendations",
                "success metrics and validation plan"
            ])
            .final_emphasis(
                "user experience and accessibility must be prioritized over technical "
                "convenience or development speed"
            )
        )

        print(builder)
        ```

        ### Quality over quantity emphasis

        Prioritize depth and thoroughness over breadth. With the `.final_emphasis()` method, ensure
        that the AI focuses on high-impact content rather than trying to address every possible
        issue.

        ```{python}
        builder = (
            tb.PromptBuilder()
            .persona("content strategist", "editorial quality")
            .task_context("evaluate content library for quality and effectiveness")
            .core_analysis([
                "content accuracy and factual verification",
                "engagement metrics and user feedback",
                "SEO optimization and discoverability",
                "brand consistency and messaging alignment"
            ])
            .constraint("include competitive analysis")
            .constraint("consider content volume requirements")
            .output_format([
                "content quality assessment",
                "priority improvement areas",
                "content strategy recommendations"
            ])
            .final_emphasis(
                "focus on identifying and improving the highest-impact content pieces rather "
                "than addressing all content issues superficially"
            )
        )

        print(builder)
        ```
        """
        self._final_emphasis = emphasis
        return self

    def pathways(self, pathway_spec) -> "PromptBuilder":
        """
        Add conversational pathway guidance to structure and guide conversation flow.

        Pathways provide flexible conversation flow guidance that helps AI assistants navigate
        complex interactions while maintaining natural conversation patterns. This method requires
        a `Pathways` object created using the chainable Pathways API, which defines states,
        transitions, and flow control logic for structured conversations. The method enables
        sophisticated conversation flow management while preserving the natural, adaptive qualities
        that make AI conversations engaging and user-friendly.

        Parameters
        ----------
        pathway_spec
            A `Pathways` object created using the chainable Pathways API, or a dictionary
            specification containing pathway definition. The specification includes states,
            transitions, information requirements, and flow control logic.

        Returns
        -------
        PromptBuilder
            Self for method chaining, allowing combination with other prompt building methods to
            create comprehensive, structured prompts.

        Research Foundation
        -------------------
        **Conversational State Management Theory.** Unlike rigid state machines, pathways serve as
        intelligent guardrails that adapt to user behavior while ensuring important steps and
        information gathering requirements are addressed. This approach balances structure with
        conversational flexibility, allowing natural dialogue patterns while maintaining systematic
        progress toward objectives.

        **Adaptive Flow Psychology.** Pathways provide conversation guidance without enforcing rigid
        adherence, allowing the AI to adapt to natural conversation patterns while ensuring key
        objectives are met. This balances structure with conversational flexibility and helps
        ensure systematic information gathering and step completion while maintaining user-friendly
        interactions.

        **Attention Optimization Integration.** Pathway specifications are integrated into the
        prompt structure at an optimal position for AI attention, providing clear guidance without
        overwhelming other prompt components. This strategic positioning ensures that conversation
        flow guidance receives appropriate attention while maintaining the overall prompt's
        cognitive load balance.

        Integration Notes
        -----------------
        - **Flexible Guidance**: pathways provide structure without rigidity, allowing natural
        conversation flow
        - **Information Gathering**: systematic collection of required information while maintaining
        user experience
        - **Adaptive Branching**: support for conditional flows based on user responses and
        circumstances
        - **Tool Integration**: clear guidance on when and how to use external tools within the
        conversation flow
        - **Completion Tracking**: built-in success conditions and completion criteria for complex
        processes

        The `.pathways()` method enables sophisticated conversation flow management while preserving
        the natural, adaptive qualities that make AI conversations engaging and user-friendly.

        Examples
        --------
        ### Customer support pathway

        Create a structured support flow:

        ```python
        import talk_box as tb

        # Define support pathway
        support_pathway = (
            tb.Pathways(
                title="Technical Support",
                desc="systematic technical problem resolution",
                activation=["user reports technical issues", "user needs troubleshooting help"]
            )
            # === STATE: problem_identification ===
            .state("understand the technical problem", id="problem_identification")
            .required(["issue description", "error messages", "recent changes"])
            .next_state("basic_diagnostics")
            # === STATE: basic_diagnostics ===
            .state("determine if basic fixes might work", id="basic_diagnostics")
            .branch_on("simple configuration issue", id="quick_fix")
            .branch_on("complex system problem", id="advanced_diagnostics")
            # === STATE: quick_fix ===
            .state("provide immediate solution steps", id="quick_fix")
            .success_condition("problem is resolved")
            # === STATE: advanced_diagnostics ===
            .state("perform detailed system analysis", id="advanced_diagnostics")
            .success_condition("root cause identified and resolved")
        )

        # Use in prompt
        prompt = (
            tb.PromptBuilder()
            .persona("technical support specialist", "troubleshooting")
            .pathways(support_pathway)
            .final_emphasis("Follow pathway while adapting to user needs")
        )
        ```
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

    def _build(self) -> str:
        """
        Internal method to construct the final prompt using attention-optimized structure.

        This method is used internally by ChatBot to create the system prompt while preserving the
        structured data for testing and analysis.
        """
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
