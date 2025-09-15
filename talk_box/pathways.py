from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from ._text_formatter import wrap_prompt_text


class StateType(Enum):
    """Types of states in a pathway."""

    CHAT = "chat"
    TOOL = "tool"
    DECISION = "decision"
    COLLECT = "collect"
    SUMMARY = "summary"


@dataclass
class PathwayState:
    """Represents a state in a conversational pathway."""

    name: str
    state_type: StateType
    description: str
    required_info: List[str] = None
    optional_info: List[str] = None
    tools: List[str] = None
    success_conditions: List[str] = None
    fallback_actions: List[str] = None
    next_states: List[str] = None
    priority: int = 1

    def __post_init__(self):
        """Initialize empty lists for None values."""
        if self.required_info is None:
            self.required_info = []
        if self.optional_info is None:
            self.optional_info = []
        if self.tools is None:
            self.tools = []
        if self.success_conditions is None:
            self.success_conditions = []
        if self.fallback_actions is None:
            self.fallback_actions = []
        if self.next_states is None:
            self.next_states = []


@dataclass
class PathwayTransition:
    """Represents a transition between pathway states."""

    from_state: str
    to_state: str
    condition: Optional[str] = None
    priority: int = 1


class Pathways:
    """
    Chainable builder for defining structured conversational pathways.

    The `Pathways` class provides intelligent conversation flow guidance while maintaining
    flexibility to adapt to natural conversation patterns. They serve as guardrails rather than
    rigid state machines, helping LLMs provide consistent and thorough assistance while remaining
    responsive to user needs and conversational context.

    Parameters
    ----------
    title
        A short, descriptive name for the pathway.
    desc
        Clear, concise explanation of the pathway's purpose and scope.
    activation
        Specific situations or user intents that trigger pathway activation. Can be a single string
        or a list of strings.
    completion_criteria
        High-level conditions that indicate the pathway's objectives have been fully achieved. Can
        be a single string or a list of strings. Optional.
    fallback_strategy
        General approach for handling situations where the pathway doesn't apply or users need
        different support. Optional.

    Returns
    -------
    Pathways
        The configured `Pathways` object for further chaining with `.state()` and other methods.

    Building Pathways
    -----------------
    Building a pathway follows a specific sequence that ensures proper configuration and flow logic.
    Each step builds upon the previous one to create a coherent conversation structure.

    ### 1. Pathway Setup (call once, in order)

    ```python
    pathway = (
        tb.Pathways(
            title="Title",
            desc="Purpose and scope",               # What this pathway does
            activation=[...],                       # When to use this pathway
            completion_criteria=[...],              # What makes pathway successful
            fallback_strategy="..."                 # Handle unexpected situations
        )
        # First .state() call automatically becomes the starting state
    ```

    ### 2. State Definition

    ```python
    pathway = (
        tb.Pathways(
            title="Support Flow",
            desc="Customer support pathway",
            activation="User needs help"
        )
        # === STATE: intake ===
        .state("intake: gather customer information")
        .required(["issue description", "contact info"])
        .next_state("triage")
        # === STATE: triage ===
        .state("triage: route to appropriate support")
        .branch_on("Technical issue", id="tech_support")
        .branch_on("Billing question", id="billing")
        # === STATE: tech_support ===
        .state("tech support: resolve technical problems")
        .success_condition("Issue resolved")
    )
    ```

    This approach provides:

    - visual state boundaries with `# === STATE: description ===` comments
    - natural state definition with shorthand syntax: `.state("id: what happens here")`
    - smart type inference where `.tools()` → `"tool"`, `.branch_on()` → `"decision"`,
    `.required()` → `"collect"`
    - automatic start state where the first `.state()` becomes the starting state

    ### 3. State Configuration Pattern (repeat for each state):

    ```python
        # Define the state with description first
        .state("What happens in this state", id="state_name")
        .state("info: collect required information")            # inferred as "collect", id="info"
        .state("make decision: evaluate options and choose")    # "make decision" → id="make_decision"
        .state("use tools: apply specific capabilities")        # type inferred as "tool" from .tools()
        .state("final summary: provide conclusion", type="summary")  # explicit type with shorthand syntax
        .state("Wrap up")                                       # Linear states don't really need IDs

        # Configure the state
        .required([...])                               # What must be accomplished
        .optional([...])                               # What would be nice to have
        .tools([...])                                  # Available tools (infers type="tool")
        .success_condition("When state succeeds")      # How to know it's complete

        # Define state transitions (choose one)
        .next_state("next_state")                      # Linear progression
        .branch_on("condition", id="target_state")     # Conditional (infers type="decision")
        .next_state("common_state")                    # Reconverge after branching
        .fallback("error_condition", "backup_state")   # Error handling
    ```

    State Types and Their Purpose
    -----------------------------
    Each state type serves a specific role in the conversation flow:

    - `type="chat"`: open conversation, explanations, guidance (default)
    - `type="decision"`: branching logic, must use `branch_on()` not `next_state()`
    - `type="collect"`: structured information gathering
    - `type="tool"`: using specific tools or APIs, requires `tools()`
    - `type="summary"`: conclusions, confirmations, completion actions

    Key Rules
    ---------
    - description is required and provided in `.state()` method
    - if you use `type="decision"`, you must use `branch_on()` and never `next_state()`
    - `type="tool"` must include a `tools()` specification
    - state names must be unique and use `"lowercase_with_underscores"`
    - target states in transitions must be defined later with another `.state()`

    Examples
    --------
    The following examples demonstrate common pathway patterns that address different conversation
    needs. The first shows a simple linear flow where states progress sequentially—ideal for
    straightforward processes. The second illustrates branching logic that routes users down
    different paths before converging to a common endpoint—perfect for triage and support scenarios.

    ### Simple Linear Flow

    This password reset pathway demonstrates the basic pattern: setup the pathway, define states
    sequentially, and specify what information each state needs to collect. Notice how each state
    builds naturally toward the goal of helping the user regain access to their account.

    ```{python}
    import talk_box as tb

    simple_pathway = (
        tb.Pathways(
            title="Password Reset",
            desc="Help users reset their forgotten passwords",
            activation=["User can't log in", "User forgot password"],
            completion_criteria="User successfully logs in with new password",
            fallback_strategy="If user lacks access to recovery methods, escalate to manual verification"
        )
        # === STATE: verification ===
        .state("verification: verify user identity")
        .required(["email address", "account verification"])
        .next_state("password_update")
        # === STATE: password_update ===
        .state("password update: guide user through creating new password")
        .required(["new password is created", "password requirements are met"])
        .success_condition("User successfully logs in with new password")
    )
    ```

    This linear flow moves step-by-step from identity verification to password creation. Each state
    has clear requirements and success conditions, making the pathway easy to follow and validate.

    ### Branching Flow with Decision Points

    This customer support pathway demonstrates decision state branching using the unified `.state()`
    method. Notice how different support paths merge back to a common completion state, ensuring
    consistent wrap-up regardless of the support type provided.

    ```{python}
    support_pathway = (
        tb.Pathways(
            title="Customer Support",
            desc="route and resolve customer inquiries",
            activation=["user needs help", "user reports problem"],
            completion_criteria=["customer issue fully resolved", "customer satisfied"],
            fallback_strategy="if issue is complex, escalate to human support"
        )
        # === STATE: triage ===
        .state("triage: determine the type of support needed")
        .branch_on("Technical problem reported", id="technical_support")
        .branch_on("Billing question asked", id="billing_support")
        .branch_on("General inquiry made", id="general_help")
        # === STATE: technical_support ===
        .state("technical support: diagnose and resolve technical issues")
        .tools(["system_diagnostics", "troubleshooting_guide"])
        .success_condition("Technical issue is resolved")
        .next_state("completion")
        # === STATE: billing_support ===
        .state("billing support: address billing and account questions")
        .required(["billing issue is understood", "solution is provided"])
        .next_state("completion")
        # === STATE: completion ===
        .state("completion: ensure customer satisfaction and wrap up", type="summary")
        .required(["issue resolved confirmation", "follow up if needed"])
        .success_condition("Customer satisfaction confirmed")
    )
    ```

    This branching example shows how `.state()` creates clear decision points that route
    conversations appropriately, then merge back together for consistent completion.

    Inspecting Pathways
    -------------------
    Once you've built a pathway, you can inspect it using different string representations:

    ```{python}
    import talk_box as tb

    # Create a simple pathway
    pathway = (
        tb.Pathways(
            title="Quick Help",
            desc="Provide rapid assistance",
            activation="User needs help",
            completion_criteria="User's problem is resolved",
            fallback_strategy="If problem is complex, escalate to specialized support"
        )
        # === STATE: problem_intake ===
        .state("problem intake: understand the issue details")
        .required(["issue description"])
        .next_state("provide_solution")
        # === STATE: provide_solution ===
        .state("provide solution: offer targeted assistance")
        .success_condition("User's problem is resolved")
    )
    ```

    We can view the pathway in two ways, either as a brief summary by examining the object itself:

    ```{python}
    pathway
    ```

    Or with `print()` for a more detailed view:

    ```{python}
    print(pathway)
    ```

    The summary view gives you a quick overview, while the detailed view shows the state types,
    description, and other configuration details. This is especially useful when debugging complex
    pathways or understanding existing pathway configurations.
    """

    def __init__(
        self,
        title: str,
        desc: str = "",
        activation: Union[str, List[str], None] = None,
        completion_criteria: Union[str, List[str], None] = None,
        fallback_strategy: str = None,
    ):
        self.title = title
        self._description: str = desc

        # Convert activation to list if it's a single string
        if activation is None:
            self._activation_conditions: List[str] = []
        elif isinstance(activation, str):
            self._activation_conditions: List[str] = [activation]
        else:
            self._activation_conditions: List[str] = activation

        # Convert completion_criteria to list if needed
        if completion_criteria is None:
            self._completion_criteria: List[str] = []
        elif isinstance(completion_criteria, str):
            self._completion_criteria: List[str] = [completion_criteria]
        else:
            self._completion_criteria: List[str] = completion_criteria

        self._fallback_strategy: Optional[str] = fallback_strategy

        self._states: Dict[str, PathwayState] = {}
        self._transitions: List[PathwayTransition] = []
        self._current_state_name: Optional[str] = None
        self._start_state: Optional[str] = None

    def state(self, desc: str, id: str = None, type: str = None) -> "Pathways":
        """
        Define a state with natural language description as the primary identifier.

        The first state you define becomes the starting state automatically. State type is inferred
        from subsequent method calls, making the API more intuitive and reducing the need to specify
        types upfront.

        Parameters
        ----------
        desc
            Clear description of the state's purpose and what should happen. This is the primary
            identifier and should be specific about the expected interaction or outcome. Supports
            shorthand syntax: use `"id: description"` format to specify both ID and description in
            one parameter (e.g., `"completion: ensure customer satisfaction and wrap up"`). The ID
            part will be automatically normalized (spaces converted to underscores, etc.).
        id
            Optional unique identifier for the state. Required only when other states need to
            reference this state (via `.branch_on()`, `.next_state()`). If not
            provided, an ID will be extracted from `desc` using `"id: description"` format if
            present, otherwise auto-generated. If explicitly provided, shorthand parsing
            is bypassed.
        type
            Optional explicit state type. If not provided, the type will be inferred from subsequent
            method calls

        Returns
        -------
        Pathways
            Self for method chaining, allowing combination with other pathway building methods to
            create comprehensive conversation flows.

        Research Foundation
        -------------------
        **State Type Inference System.** Based on method usage, the state type is automatically
        inferred to reduce cognitive load and API complexity. The inference follows this hierarchy:
        `.tools()` → `"tool"`, `.branch_on()` → `"decision"`, `.required()` → `"collect"`, with
        `"chat"` as the default. If multiple methods suggest different types, the first inference
        takes precedence to maintain consistency.

        **Automatic Start State Management.** The first state defined automatically becomes the
        starting state, eliminating the need for explicit start state configuration. This
        simplifies pathway creation while ensuring every pathway has a clear entry point for
        conversation flow.

        **ID Generation and Reference System.** When no explicit ID is provided, the system
        auto-generates snake_case identifiers from state descriptions, ensuring uniqueness through
        numeric suffixes when conflicts occur. IDs are only required when other states need to
        reference the state for transitions or branching.

        Integration Notes
        -----------------
        - **Description Priority**: description always comes first as the primary identifier
        - **Reference Requirements**: ID only needed when other states need to reference this state
        - **Shorthand Syntax**: use `"id: description"` format to combine ID and description in one
        parameter for cleaner syntax; ID part is automatically normalized (spaces → underscores)
        - **Explicit ID Priority**: when `id=` parameter is provided, shorthand parsing is bypassed
        - **Type Inference**: inferred from usage patterns to reduce explicit configuration
        - **Conflict Resolution**: first method call determines type, conflicts generate warnings
        - **Auto-generation**: IDs use `snake_case` from description with uniqueness guarantees
        - **Visual Organization**: use `# === STATE: name ===` comments for visual state separation
        in complex pathways
        - **Explicit Type Usage**: consider explicit types for complex workflows, documentation
        clarity, team development, mixed functionality, or error prevention

        The `.state()` method creates clear conversation boundaries and progression, with each state
        having a specific purpose that builds toward the final goal. When to use explicit types:
        in complex workflows where type inference might be ambiguous, for documentation clarity
        when the state's purpose isn't obvious from methods, for team development to make intentions
        explicit, for mixed functionality when a state serves multiple purposes, or as error
        prevention for avoiding unintended type inference conflicts.

        Examples
        --------
        Complete pathway showing `.state()` method with both traditional and shorthand syntax:

        ```{python}
        import talk_box as tb

        # Creating a complete product recommendation pathway
        pathway = (
            tb.Pathways(
                title="Product Recommendation",
                desc="Help customers find the right product for their needs",
                activation="Customer needs product guidance"
            )

            # Traditional syntax: separate id parameter ---
            # === STATE: welcome ===
            .state("welcome: welcome customer and understand their situation")
            .required(["the customer's goal", "a budget range"])
            .next_state("needs_analysis")

            # Shorthand syntax: "id: description" format ---
            # === STATE: needs_analysis ===
            .state("needs analysis: analyze customer requirements and preferences")
            .required(["specific requirements", "priorities"])
            .success_condition("customer needs are clearly understood")
            .next_state("final_recommendation")

            # Spaces in ID automatically become underscores ---
            # === STATE: final_recommendation ===
            .state("final recommendation: present tailored product matches")
            .required(["product matches", "rationale"])
            .success_condition("customer has clear next steps")
        )

        # See how the pathway materializes
        print(pathway)
        ```
        """
        import re

        # Parse shorthand syntax: "id: description" format
        if id is None and ":" in desc:
            # Check if desc follows "id: description" pattern
            parts = desc.split(":", 1)
            if len(parts) == 2:
                potential_id = parts[0].strip()
                # Convert spaces and other non-identifier chars to underscores for valid ID
                normalized_id = re.sub(r"[^a-zA-Z0-9_]", "_", potential_id)
                normalized_id = re.sub(r"_+", "_", normalized_id)  # Collapse multiple underscores
                normalized_id = normalized_id.strip("_")  # Remove leading/trailing underscores

                # Ensure it starts with a letter or underscore
                if normalized_id and not normalized_id[0].isalpha() and normalized_id[0] != "_":
                    normalized_id = "_" + normalized_id

                # Use normalized ID if it results in a valid identifier
                if normalized_id and re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", normalized_id):
                    id = normalized_id
                    # Keep the full description including the "id: " prefix
                    # desc remains unchanged

        # Generate ID from description if still not provided
        if id is None:
            # Create snake_case ID from description
            id = re.sub(r"[^\w\s]", "", desc.lower())
            id = re.sub(r"\s+", "_", id.strip())
            # Ensure uniqueness
            base_id = id
            counter = 1
            while id in self._states:
                id = f"{base_id}_{counter}"
                counter += 1

        # Set as start state if this is the first state defined
        if not self._start_state:
            self._start_state = id

        # Set current state for subsequent configuration
        self._current_state_name = id

        # Create the state with initial type (will be refined by inference)
        initial_type = type if type else "chat"  # Default to chat until inferred

        state_type_map = {
            "chat": StateType.CHAT,
            "collect": StateType.COLLECT,
            "decision": StateType.DECISION,
            "tool": StateType.TOOL,
            "summary": StateType.SUMMARY,
        }

        if initial_type not in state_type_map:
            raise ValueError(
                f"Invalid state type '{initial_type}'. Must be one of: {list(state_type_map.keys())}"
            )

        # Create the PathwayState
        self._states[id] = PathwayState(
            name=id,
            state_type=state_type_map[initial_type],
            description=desc,
            required_info=[],
            optional_info=[],
            tools=[],
            success_conditions=[],
            next_states=[],
        )

        # Store whether type was explicitly set (for inference logic)
        if not hasattr(self, "_explicit_types"):
            self._explicit_types = {}
        self._explicit_types[id] = type is not None

        return self

    def _infer_state_type(self, new_type: str, strength: str = "weak") -> None:
        """
        Infer and update state type based on method usage.

        Parameters
        ----------
        new_type
            The type being inferred from method usage.
        strength
            How strong the inference is: "weak" (preserve chat unless strong evidence),
            "strong" (always infer unless explicitly set).
        """
        if not self._current_state_name or self._current_state_name not in self._states:
            return

        current_state = self._states[self._current_state_name]
        was_explicit = self._explicit_types.get(self._current_state_name, False)

        # If type was explicitly set, warn about conflicts but don't change
        if was_explicit and current_state.state_type.value != new_type:
            import warnings

            warnings.warn(
                f"State '{self._current_state_name}' was explicitly set to "
                f"'{current_state.state_type.value}' but method suggests '{new_type}'. "
                f"Keeping explicit type '{current_state.state_type.value}'.",
                UserWarning,
            )
            return

        # If already inferred a different type, warn but keep first inference
        if (
            not was_explicit
            and current_state.state_type.value != "chat"
            and current_state.state_type.value != new_type
        ):
            import warnings

            warnings.warn(
                f"State '{self._current_state_name}' was inferred as "
                f"'{current_state.state_type.value}' but method suggests '{new_type}'. "
                f"Keeping first inference '{current_state.state_type.value}'.",
                UserWarning,
            )
            return

        # Conservative inference: only change from "chat" if we have strong evidence
        # or if the current type already matches
        should_update = False

        if current_state.state_type.value == new_type:
            # Already the suggested type
            should_update = True
        elif current_state.state_type.value == "chat":
            # Change from chat based on evidence strength
            if strength in ["strong", "moderate"]:
                should_update = True
            # For weak evidence, preserve chat (conversational flows are common)

        if should_update:
            state_type_map = {
                "chat": StateType.CHAT,
                "collect": StateType.COLLECT,
                "decision": StateType.DECISION,
                "tool": StateType.TOOL,
                "summary": StateType.SUMMARY,
            }
            current_state.state_type = state_type_map[new_type]

    def required(self, info_types: Union[str, List[str]]) -> "Pathways":
        """
        Specify required information for the current state to be considered complete.

        Use to define what must be obtained before the state can transition to the next step. The
        LLM will focus on gathering this information before proceeding. Can be used in any order
        within the state configuration.

        Parameters
        ----------
        info_types
            Essential information that must be collected or established. Can be a single string or a
            list of strings. Be specific and measurable.

        Returns
        -------
        Pathways
            Self for method chaining, allowing combination with other pathway building methods to
            create comprehensive conversation flows.

        Integration Notes
        -----------------
        - **State Progression**: state cannot progress until required items are addressed
        - **Type Inference**: infers state type as `"collect"` if not explicitly set
        - **Specificity**: be specific and concrete for clear guidance
        - **Complementary Use**: pair with `.optional()` for nice-to-have information
        - **Completion Criteria**: use `.success_condition()` to define when requirements are truly
        met
        - **Systematic Collection**: ensures thorough data gathering before progression

        The `.required()` method ensures the LLM won't proceed until essential information is
        collected, preventing incomplete processes and ensuring thorough data gathering.

        Examples
        --------
        Complete pathway showing `.required()` defining essential information:

        ```{python}
        import talk_box as tb

        # Creating a loan application pathway
        pathway = (
            tb.Pathways(
                title="Loan Application Process",
                desc="guide customers through loan application requirements",
                activation="customer wants to apply for a loan"
            )
            # === STATE: personal_info ===
            .state("personal info: gather basic applicant information")

            # .required() ensures critical data is collected ---
            .required(["applicant's full name", "current employment status"])

            .next_state("financial_details")
            # === STATE: financial_details ===
            .state("financial details: collect financial information")

            # .required() can specify multiple essential items ---

            .required([
                "verified annual income amount",
                "detailed monthly expenses breakdown",
                "complete existing debt information",
                "authorization to check credit score"
            ])

            .success_condition("All financial data verified")
            .next_state("review")
            # === STATE: review ===
            .state("review: review application completeness")

            # .required() works with single items too ---
            .required("applicant's legal signature and consent")

            .success_condition("application ready for processing")
        )

        # See the pathway with required information highlighted
        print(pathway)
        ```
        """
        # Moderate inference: required() suggests collect but allows explicit override
        self._infer_state_type("collect", strength="moderate")

        # Convert string to list if needed
        if isinstance(info_types, str):
            info_types = [info_types]

        if self._current_state_name in self._states:
            self._states[self._current_state_name].required_info.extend(info_types)
        return self

    def optional(self, info_types: Union[str, List[str]]) -> "Pathways":
        """
        Specify optional information that would be helpful but not required.

        Use to define nice-to-have information that can improve the outcome but isn't essential
        for state completion. The LLM will attempt to gather this if the conversation allows.
        Often used alongside `.required()` to create comprehensive information gathering states.

        Parameters
        ----------
        info_types
            Additional information that would be beneficial but not essential. Can be a single
            string or a list of strings.

        Returns
        -------
        Pathways
            Self for method chaining, allowing combination with other pathway building methods to
            create comprehensive conversation flows.

        Integration Notes
        -----------------
        - **Flexible Progression**: state can progress without optional items
        - **Enhanced Outcomes**: helps create more comprehensive outcomes when available
        - **Balanced Flow**: use sparingly as too many optionals can slow the flow
        - **State Compatibility**: best used in states with `type="collect"` or structured chat
        states
        - **Complementary Use**: often used alongside `.required()` to create comprehensive
        information gathering
        - **Conversation Adaptation**: allows gathering helpful information when conversation
        naturally allows

        The `.optional()` method allows conversations to gather helpful information when available,
        but doesn't block progress if users want to move forward quickly.

        Examples
        --------
        Complete pathway showing `.optional()` enhancing outcomes without blocking progress:

        ```{python}
        import talk_box as tb

        # Creating a travel booking pathway
        pathway = (
            tb.Pathways(
                title="Flight Booking Assistant",
                desc="help customers find and book flights",
                activation="customer wants to book a flight"
            )
            # === STATE: travel_basics ===
            .state("travel basics: gather essential travel details")
            .required(["departure city", "destination city", "preferred travel date"])

            # .optional() adds helpful details without slowing the process -----
            .optional([
                "return date if roundtrip",
                "preferred departure time window",
                "airline preference or loyalty program"
            ])

            .next_state("search_flights")
            # === STATE: search_flights ===
            .state("search flights: find matching flights")
            .required("available flight options found and presented")

            # .optional() can improve personalization ---
            .optional("preferred seating section or specific seat requests")

            .success_condition("customer has reviewed flight options")
            .next_state("booking")
            # === STATE: booking ===
            .state("booking: complete the booking")
            .required(["valid payment information", "complete traveler details for all passengers"])

            # .optional() for enhanced services ---
            .optional([
                "travel insurance coverage options",
                "special meal requests or dietary needs",
                "frequent flyer number for miles credit"
            ])

            .success_condition("booking confirmed")
        )

        # See how optional items enhance the pathway
        print(pathway)
        ```
        """
        # Convert string to list if needed
        if isinstance(info_types, str):
            info_types = [info_types]

        if self._current_state_name in self._states:
            self._states[self._current_state_name].optional_info.extend(info_types)

            # Smart inference: if state has both required and optional, it's likely collect
            current_state = self._states[self._current_state_name]
            required_count = len(current_state.required_info)
            optional_count = len(current_state.optional_info)

            # Strong signal: both required and optional items suggest structured collection
            if required_count > 0 and optional_count > 0:
                self._infer_state_type("collect", strength="strong")

        return self

    def tools(self, tool_names: Union[str, List[str]]) -> "Pathways":
        """
        Specify tools available for use in the current state.

        Essential for `type="tool"` states, but can also be used in other states where specific
        capabilities are needed. Typically combined with `.success_condition()` to define when tool
        usage is complete.

        Parameters
        ----------
        tool_names
            Names of specific tools or capabilities the LLM should use. Can be a single string or a
            list of strings. These should match actual available tools.

        Returns
        -------
        Pathways
            Self for method chaining, allowing combination with other pathway building methods to
            create comprehensive conversation flows.

        Integration Notes
        -----------------
        - **Type Inference**: infers state type as `"tool"` if not explicitly set
        - **Tool Matching**: tool names should match actual available capabilities
        - **Completion Criteria**: use `.success_condition()` to define completion criteria
        - **Error Handling**: consider `.fallback()` for when tools fail
        - **State Focus**: essential for `type="tool"` states but can be used in other states where
        specific capabilities are needed
        - **Capability Specification**: tells the LLM what specific capabilities are available at
        each step

        The `.tools()` method tells the LLM what specific capabilities are available at each step,
        automatically inferring the state type as "tool" when tools are the primary focus.

        Examples
        --------
        Complete pathway showing `.tools()` enabling specific capabilities:

        ```{python}
        import talk_box as tb

        # Creating a technical diagnosis pathway
        pathway = (
            tb.Pathways(
                title="System Diagnostics",
                desc="diagnose and resolve technical issues",
                activation="user reports technical problems"
            )
            # === STATE: problem_intake ===
            .state("problem intake: understand the reported issue")
            .required(["problem description", "system details", "error messages"])
            .next_state("initial_diagnosis")
            # === STATE: initial_diagnosis ===
            .state("initial diagnosis: run initial diagnostic checks")

            # .tools() specifies what capabilities are available ---
            .tools([
                "system_health_checker",
                "log_analyzer",
                "performance_monitor"
            ])

            .success_condition("initial diagnosis completed")
            .next_state("detailed_analysis")
            # === STATE: detailed_analysis ===
            .state("detailed analysis: perform detailed system analysis")

            # .tools() can specify advanced diagnostic tools ---
            .tools([
                "network_diagnostics",
                "database_integrity_check",
                "security_scan"
            ])

            .required(["the root cause is identified"])
            .next_state("solution")
            # === STATE: solution ===
            .state("solution: implement solution")

            # .tools() for implementation capabilities ---
            .tools("automated_repair_tool")

            .success_condition("issue resolved and system stable")
        )

        # See how tools are integrated into the pathway
        print(pathway)
        ```
        """
        # Strong inference: tools() strongly suggests tool state
        self._infer_state_type("tool", strength="strong")

        # Convert string to list if needed
        if isinstance(tool_names, str):
            tool_names = [tool_names]

        if self._current_state_name in self._states:
            self._states[self._current_state_name].tools.extend(tool_names)
        return self

    def success_condition(self, condition: str) -> "Pathways":
        """
        Define what indicates successful completion of the current state.

        Use to specify when the state's objectives are met and it's ready to transition. More
        specific than just completing `.required()` items. Can be used in any order within the state
        configuration.

        Parameters
        ----------
        condition
            Specific, observable condition indicating the state succeeded. Use action-oriented
            language that the LLM can recognize.

        Returns
        -------
        Pathways
            Self for method chaining, allowing combination with other pathway building methods to
            create comprehensive conversation flows.

        Integration Notes
        -----------------
        - **Completion Clarity**: more specific than just completing `.required()` items
        - **Observable Criteria**: should be observable/confirmable in conversation
        - **Active Voice**: use active voice like `"user confirms..."` not
        `"user understanding confirmed"`
        - **Multiple Conditions**: can have multiple success conditions for complex states
        - **Progression Control**: prevents premature progression and ensures thorough coverage
        - **Objective Definition**: specifies when the state's objectives are met and ready to
        transition

        The `.success_condition()` method ensures the LLM knows exactly when each step is truly
        complete, preventing premature progression and ensuring thorough coverage.

        Examples
        --------
        Complete pathway showing `.success_condition()` defining clear completion criteria:

        ```{python}
        import talk_box as tb

        # Creating a learning assessment pathway
        pathway = (
            tb.Pathways(
                title="Skill Assessment",
                desc="evaluate student understanding and provide targeted feedback",
                activation="student completes a learning module"
            )
            # === STATE: practice ===
            .state("practice: present practice problems")
            .required(["problems are attempted", "student provided responses"])

            # .success_condition() defines when understanding is demonstrated ---
            .success_condition("student correctly solves at least 3 out of 5 problems")

            .next_state("feedback")
            # === STATE: feedback ===
            .state("feedback: provide personalized feedback")
            .required(["specific feedback", "improvement areas"])

            # .success_condition() ensures feedback is constructive ---
            .success_condition("student understands their mistakes and next steps")

            .next_state("advanced_practice")
            # === STATE: advanced_practice ===
            .state("advanced practice: offer advanced challenges")
            .required("challenging problems are presented")
            .optional("hints if needed")

            # .success_condition() confirms mastery ---
            .success_condition("student demonstrates confident problem-solving ability")
        )

        # See how success conditions guide the learning process
        print(pathway)
        ```
        """
        if self._current_state_name in self._states:
            self._states[self._current_state_name].success_conditions.append(condition)
        return self

    def next_state(self, state_name: str) -> "Pathways":
        """
        Define direct transition to the next state.

        Use for linear progression after state completion. Do not use with `type="decision"`
        states (use `.branch_on()` instead). This creates unconditional forward movement in the
        pathway.

        Parameters
        ----------
        state_name
            Name of the state to transition to next. The target state must be defined later in the
            pathway using `.state()`.

        Returns
        -------
        Pathways
            Self for method chaining, allowing combination with other pathway building methods to
            create comprehensive conversation flows.

        Integration Notes
        -----------------
        - **Linear Progression**: creates unconditional transition after state completion
        - **Decision State Restriction**: cannot be used with `type="decision"` states; use
        `.branch_on()` instead
        - **Forward Declaration**: target state must be defined later with `.state()`
        - **Sequential Flow**: perfect for processes with a clear order
        - **Unconditional Movement**: for conditional logic, use `.branch_on()`
        - **Straightforward Routing**: creates sequential flows where each step naturally follows
        the previous

        The `.next_state()` method creates straightforward, sequential flows where each step
        naturally follows the previous one, perfect for processes with a clear order.

        Examples
        --------
        Complete pathway showing `.next_state()` creating linear progression:

        ```{python}
        import talk_box as tb

        # Creating a customer onboarding pathway
        pathway = (
            tb.Pathways(
                title="Customer Onboarding",
                desc="welcome new customers and set up their accounts",
                activation="new customer signs up"
            )
            # === STATE: welcome ===
            .state("welcome: welcome and collect basic information")
            .required(["full name", "email", "company name"])

            # .next_state() creates smooth linear progression ---
            .next_state("account_setup")

            # === STATE: account_setup ===
            .state("account setup: set up account preferences")
            .required(["password is created", "preferences are selected"])
            .success_condition("account is fully configured")

            # .next_state() continues the sequential flow ---
            .next_state("feature_tour")

            # === STATE: feature_tour ===
            .state("feature tour: provide guided feature tour")
            .required("key features are demonstrated")
            .success_condition("customer understands main functionality")

            # .next_state() leads to final step ---
            .next_state("completion")

            # === STATE: completion ===
            .state("completion: complete onboarding process")
            .required(["welcome resources are provided", "next steps are explained"])
            .success_condition("customer is ready to use the platform")
        )

        # See the clear linear progression
        print(pathway)
        ```
        """
        if self._current_state_name:
            self._transitions.append(
                PathwayTransition(from_state=self._current_state_name, to_state=state_name)
            )
        return self

    def branch_on(self, condition: str, id: str) -> "Pathways":
        """
        Define conditional branch to another state based on specific conditions.

        Use with decision states to create multiple possible transitions based on user responses,
        detected conditions, or conversation context. Each branch should represent a distinct path
        through the workflow.

        Parameters
        ----------
        condition
            Specific, recognizable condition that triggers this branch. Be concrete and observable
            in conversation.
        id
            Target state ID for this branch condition. The target state must be defined later with
            `.state()`.

        Returns
        -------
        Pathways
            Self for method chaining, allowing combination with other pathway building methods to
            create comprehensive conversation flows.

        Integration Notes
        -----------------
        - **Type Inference**: infers current state type as `"decision"` if not explicitly set
        - **Mutual Exclusivity**: conditions should be mutually exclusive when possible
        - **Forward Declaration**: each branch must lead to a state defined later with `.state()`
        - **Concrete Conditions**: be specific like `"user mentions password issues"` not
        `"user has problems"`
        - **Smart Routing**: enables routing based on conditions with automatic decision state
        inference
        - **Reconvergence**: allows multiple pathways that can reconverge later

        The `.branch_on()` method enables smart routing based on conditions, automatically inferring
        the `"decision"` state type and allowing multiple pathways that can reconverge later.

        Examples
        --------
        Complete pathway showing `.branch_on()` creating conditional routing:

        ```{python}
        import talk_box as tb

        # Creating a healthcare triage pathway
        pathway = (
            tb.Pathways(
                title="Medical Triage",
                desc="route patients to appropriate care based on symptoms",
                activation="patient seeks medical assistance"
            )
            # === STATE: initial_assessment ===
            .state("initial assessment: assess patient symptoms and urgency")
            .required(["symptoms are described", "pain level", "duration"])
            .success_condition("Symptoms are clearly documented")
            .next_state("triage_decision")
            # === STATE: triage_decision ===
            .state("triage decision: determine appropriate care level")
            .required("urgency is evaluated")

            # .branch_on() routes based on severity -----
            .branch_on("severe or life-threatening symptoms", id="emergency_care")
            .branch_on("moderate symptoms requiring prompt attention", id="urgent_care")
            .branch_on("mild symptoms manageable with routine care", id="standard_care")

            # The first branch leads to emergency care -----
            # === STATE: emergency_care ===
            .state("emergency care: initiate emergency protocol")
            .required(["911 is called", "immediate first aid is provided"])
            .success_condition("emergency services are contacted")
            .next_state("follow_up")

            # The second branch leads to urgent care -----
            # === STATE: urgent_care ===
            .state("urgent care: schedule urgent care appointment")
            .required(["same day appointment", "preparation instructions"])
            .success_condition("urgent care is arranged")
            .next_state("follow_up")

            # The third branch leads to standard care -----
            # === STATE: standard_care ===
            .state("standard care: provide self-care guidance")
            .required(["home care instructions", "symptom monitoring"])
            .success_condition("patient understands self-care plan")
            .next_state("follow_up")
            # === STATE: follow_up ===
            .state("follow up: arrange follow-up care")
            .required(["follow up is scheduled"])
            .success_condition("continuity of care is ensured")
        )

        # See how branching creates appropriate care pathways
        print(pathway)
        ```
        """
        # Infer state type as "decision"
        # Strong inference: branch_on() strongly suggests decision state
        self._infer_state_type("decision", strength="strong")

        if self._current_state_name:
            self._transitions.append(
                PathwayTransition(
                    from_state=self._current_state_name, to_state=id, condition=condition
                )
            )
        return self

    def fallback(self, condition: str, state_name: str) -> "Pathways":
        """
        Define fallback transition when normal state progression fails.

        Use when you need to handle error conditions, user confusion, or when expected outcomes
        don't occur. Provides graceful recovery paths instead of getting stuck in a state.

        Parameters
        ----------
        condition
            Specific condition that triggers the fallback. Usually describes a failure or unexpected
            situation.
        state_name
            State to transition to when fallback condition occurs.

        Returns
        -------
        Pathways
            Self for method chaining, allowing combination with other pathway building methods to
            create comprehensive conversation flows.

        Integration Notes
        -----------------
        - **Error Handling**: use for error handling and recovery from unexpected situations
        - **Graceful Degradation**: provides graceful degradation instead of getting stuck
        - **Failure Scenarios**: condition should describe failure scenarios clearly
        - **Combined Usage**: can be used alongside `.next_state()` or `.branch_on()`
        - **Stuck Prevention**: ensures conversations don't get stuck when expected outcomes don't
        occur
        - **Recovery Paths**: provides alternative paths for complex scenarios and edge cases

        The `.fallback()` method ensures conversations don't get stuck when expected outcomes don't
        occur, providing alternative paths for complex scenarios and edge cases.

        Examples
        --------
        Complete pathway showing `.fallback()` providing graceful error recovery:

        ```{python}
        import talk_box as tb

        # Creating a complex problem-solving pathway with fallbacks
        pathway = (
            tb.Pathways(
                title="Technical Problem Resolution",
                desc="systematic approach to solving technical issues",
                activation="user encounters a technical problem"
            )
            # === STATE: problem_analysis ===
            .state("problem analysis: understand the problem details")
            .required(["problem description", "system context", "error details"])
            .success_condition("problem is clearly defined")
            .next_state("solution_attempt")
            # === STATE: solution_attempt ===
            .state("solution attempt: apply standard solution")
            .required(["solution is implemented", "results are verified"])
            .success_condition("problem is resolved")

            # .fallback() handles situations where standard solutions don't work ---
            .fallback("solution doesn't resolve the issue", "advanced_troubleshooting")

            .next_state("completion")
            # === STATE: advanced_troubleshooting ===
            .state("advanced troubleshooting: advanced diagnostic procedures")
            .tools(["system_diagnostics", "log_analyzer", "network_tracer"])
            .required("root cause is identified")
            .success_condition("advanced solution is implemented")

            # .fallback() provides escalation when even advanced methods fail -----
            .fallback("issue remains unresolved after advanced diagnostics", "expert_escalation")

            .next_state("completion")

            # === STATE: expert_escalation ===
            .state("expert escalation: escalate to specialist support")
            .required(["detailed case summary", "expert is contacted"])
            .success_condition("case is transferred to appropriate specialist")
            .next_state("completion")
            # === STATE: completion ===
            .state("completion: confirm resolution and document")
            .required(["resolution is confirmed", "case is documented"])
            .success_condition("issue fully resolved and documented")
        )

        # See how fallbacks provide multiple recovery paths
        print(pathway)
        ```
        """
        if self._current_state_name in self._states:
            self._states[self._current_state_name].fallback_actions.append(
                f"If {condition}, go to {state_name}"
            )
        return self

    def _apply_final_state_inference(self):
        """Apply automatic type inference for final states (states with no outgoing transitions)."""
        # Find states with no outgoing transitions
        states_with_outgoing = set()
        for transition in self._transitions:
            states_with_outgoing.add(transition.from_state)

        final_states = []
        for state_name in self._states:
            if state_name not in states_with_outgoing:
                final_states.append(state_name)

        # Apply summary inference to final states
        for state_name in final_states:
            state = self._states[state_name]
            # Only apply if the state is still CHAT (default) - this means no explicit type was set
            # and no other inference has been applied yet
            if state.state_type == StateType.CHAT:
                state.state_type = StateType.SUMMARY

    def _build(self) -> Dict[str, Any]:
        """
        Internal method to build the complete pathway specification.

        This method is used internally by ChatBot and PromptBuilder to create the pathway
        specification while preserving the structured data for testing and analysis.

        Returns
        -------
        Dict[str, Any]
            Complete pathway specification ready for prompt integration
        """
        # Apply final state inference before building
        self._apply_final_state_inference()

        return {
            "title": self.title,
            "description": self._description,
            "activation_conditions": self._activation_conditions,
            "start_state": self._start_state,
            "states": {name: self._state_to_dict(state) for name, state in self._states.items()},
            "transitions": [self._transition_to_dict(t) for t in self._transitions],
            "completion_criteria": self._completion_criteria,
            "fallback_strategy": self._fallback_strategy,
        }

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access to pathway data."""
        return self._build().get(key)

    def __str__(self) -> str:
        """Return detailed pathway specification for print() display."""
        return wrap_prompt_text(self._to_prompt_text())

    def __repr__(self) -> str:
        """Return a developer-friendly representation of the Pathways configuration."""
        data = self._build()
        components = [f"'{data['title']}'"]

        # Add description if present
        if data.get("description"):
            desc = data["description"]
            if len(desc) > 40:
                desc = desc[:37] + "..."
            components.append(f"description='{desc}'")

        # Add states info
        if data["states"]:
            state_types = set(state.get("type", "unknown") for state in data["states"].values())
            components.append(f"states={len(data['states'])} ({', '.join(sorted(state_types))})")

        # Add activation conditions
        conditions = data.get("activation_conditions", [])
        if conditions:
            components.append(f"conditions={len(conditions)}")

        # Add completion criteria
        criteria = data.get("completion_criteria", [])
        if criteria:
            components.append(f"completion_criteria={len(criteria)}")

        # Add fallback strategy if present
        if data.get("fallback_strategy"):
            components.append("fallback_strategy=True")

        return f"Pathways({', '.join(components)})"

    def _state_to_dict(self, state: PathwayState) -> Dict[str, Any]:
        """Convert PathwayState to dictionary."""
        return {
            "name": state.name,
            "type": state.state_type.value,
            "description": state.description,
            "required_info": state.required_info,
            "optional_info": state.optional_info,
            "tools": state.tools,
            "success_conditions": state.success_conditions,
            "fallback_actions": state.fallback_actions,
            "priority": state.priority,
        }

    def _transition_to_dict(self, transition: PathwayTransition) -> Dict[str, Any]:
        """Convert PathwayTransition to dictionary."""
        return {
            "from": transition.from_state,
            "to": transition.to_state,
            "condition": transition.condition,
            "priority": transition.priority,
        }

    def _to_prompt_text(self) -> str:
        """
        Internal method to generate text specification for inclusion in system prompts.

        Returns
        -------
        str
            Formatted pathway specification for LLM consumption
        """
        spec = self._build()
        lines = []

        # Title and description
        lines.append(f"**{spec['title']}**")
        if spec.get("description"):
            lines.append(f"Purpose: {spec['description']}")

        # Activation conditions
        if spec.get("activation_conditions"):
            lines.append("Activate when:")
            for condition in spec["activation_conditions"]:
                lines.append(f"- {condition}")

        # Build transitions map for easier lookup
        transitions_from = {}
        transitions_to = {}
        for transition in spec.get("transitions", []):
            from_state = transition["from"]
            to_state = transition["to"]
            if from_state not in transitions_from:
                transitions_from[from_state] = []
            if to_state not in transitions_to:
                transitions_to[to_state] = []
            transitions_from[from_state].append(transition)
            transitions_to[to_state].append(transition)

        # Flow guidance showing states with clear branching structure
        lines.append("Flow guidance:")

        # Helper function to format a single state
        def format_single_state(state_name: str, indent: str = "") -> List[str]:
            if state_name not in spec.get("states", {}):
                return []

            state = spec["states"][state_name]
            state_lines = []

            # State header with type
            state_lines.append(
                f"{indent}- {state_name.upper()} ({state['type']}): {state.get('description', '')}"
            )

            # Helper function to format lists with numbering if multiple items
            def format_list(items, label):
                if not items:
                    return
                if len(items) == 1:
                    state_lines.append(f"{indent}  {label}: {items[0]}")
                else:
                    formatted_items = ", ".join(
                        [f"({i + 1}) {item}" for i, item in enumerate(items)]
                    )
                    state_lines.append(f"{indent}  {label}: {formatted_items}")

            # Required information
            if state.get("required_info"):
                format_list(state["required_info"], "Required")

            # Optional information
            if state.get("optional_info"):
                format_list(state["optional_info"], "Optional")

            # Tools
            if state.get("tools"):
                format_list(state["tools"], "Tools")

            # Success conditions
            if state.get("success_conditions"):
                state_lines.append(f"{indent}  Success: {'; '.join(state['success_conditions'])}")

            return state_lines

        # Start with the start state
        start_state = spec.get("start_state")
        if start_state:
            lines.extend(format_single_state(start_state))

            # Follow the flow
            current_states = [start_state]
            processed = {start_state}

            while current_states:
                next_states = []

                for current_state in current_states:
                    if current_state not in transitions_from:
                        continue

                    transitions = transitions_from[current_state]

                    # Check for branching (conditional transitions)
                    conditional_transitions = [t for t in transitions if t.get("condition")]
                    direct_transitions = [t for t in transitions if not t.get("condition")]

                    if conditional_transitions:
                        # Show branching options
                        for i, transition in enumerate(conditional_transitions, 1):
                            lines.append(
                                f"  Branch {i}: {transition['condition']} → {transition['to'].upper()}"
                            )

                        # Add branch target states with indentation
                        for transition in conditional_transitions:
                            target_state = transition["to"]
                            if target_state not in processed:
                                lines.extend(format_single_state(target_state, "  "))
                                processed.add(target_state)
                                next_states.append(target_state)

                    # Handle direct transitions
                    for transition in direct_transitions:
                        target_state = transition["to"]
                        if target_state not in processed:
                            lines.extend(format_single_state(target_state))
                            processed.add(target_state)
                            next_states.append(target_state)

                current_states = next_states

        # Completion criteria
        if spec.get("completion_criteria"):
            criteria = spec["completion_criteria"]
            if len(criteria) == 1:
                lines.append(f"Complete when: {criteria[0]}")
            else:
                formatted_criteria = ", ".join(
                    [f"({i + 1}) {criterion}" for i, criterion in enumerate(criteria)]
                )
                lines.append(f"Complete when: {formatted_criteria}")

        # Fallback strategy
        if spec.get("fallback_strategy"):
            lines.append(f"Fallback: {spec['fallback_strategy']}")

        lines.append(
            "Follow as flexible guidance, adapting to user conversation patterns while ensuring key objectives are addressed."
        )

        return "\n".join(lines)

    def visualize(self, title: str = None, filename: str = None, auto_open: bool = True) -> str:
        """
        Create an HTML visualization of this pathway and save to file.

        This method generates a flowchart diagram showing all states, transitions, and
        branching logic using pure HTML/CSS. The visualization includes:

        - Color-coded boxes based on state type (collect, tool, decision, summary)
        - Clear flow arrows showing progression
        - Reconvergence indicators for states with multiple parents
        - Professional styling with hover effects

        Parameters
        ----------
        title : str, optional
            Title for the visualization page. If None, uses the pathway title.
        filename : str, optional
            Name for the HTML file (without extension). If None, uses "pathway_visualization".
        auto_open : bool, default True
            Whether to automatically open the visualization in the default browser.

        Returns
        -------
        str
            Path to the generated HTML file

        Examples
        --------
        Create and display a pathway visualization:

        ```python
        import talk_box as tb

        # Create a pathway
        pathway = (
            tb.Pathways(
                title="Customer Support",
                desc="Handle customer inquiries efficiently"
            )
            .state("intake: gather customer information")
            .next_state("triage")
            .state("triage: determine support type")
            .branch_on("Technical issue", id="tech_support")
            .branch_on("Billing question", id="billing")
            .state("tech_support: resolve technical problems")
            .tools(["diagnostic_tool"])
            .next_state("completion")
            .state("billing: handle billing inquiries")
            .next_state("completion")
            .state("completion: wrap up and follow up", type="summary")
        )

        # Generate and open visualization
        pathway.visualize()  # Opens in browser automatically

        # Save to specific file without opening
        pathway.visualize(filename="my_pathway", auto_open=False)
        ```
        """
        import os
        import webbrowser
        from pathlib import Path

        if title is None:
            title = f"Pathway Visualization: {self.title}"

        if filename is None:
            filename = "pathway_visualization"

        # Create output directory
        output_dir = Path("pathway_visualizations")
        output_dir.mkdir(exist_ok=True)

        # Generate HTML content using the same visualization as _repr_html_
        # but wrapped in a full HTML page
        flowchart_content = self._repr_html_()

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            padding: 30px;
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
            font-weight: 700;
        }}
        .description {{
            text-align: center;
            font-size: 1.2em;
            color: #666;
            margin-bottom: 40px;
            font-style: italic;
        }}
        .visualization {{
            margin: 20px 0;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #e9ecef;
            color: #6c757d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <div class="description">{self._description}</div>
        <div class="visualization">
            {flowchart_content}
        </div>
        <div class="footer">
            Generated by Talk Box Pathways • Flowchart Visualization
        </div>
    </div>
</body>
</html>"""

        # Save to file
        output_path = output_dir / f"{filename}.html"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        # Open in browser if requested
        if auto_open:
            webbrowser.open(f"file://{os.path.abspath(output_path)}")

        return str(output_path)

    def _extract_pathway_data(self):
        """Extract pathway data for visualization (self-contained)."""
        nodes = []
        edges = []
        processed_states = set()

        # Get all states from the pathway
        all_states = set(self._states.keys())

        # Add states referenced in transitions
        for transition in self._transitions:
            all_states.add(transition.from_state)
            all_states.add(transition.to_state)

        # Create nodes
        for state_name in all_states:
            if state_name in processed_states:
                continue

            processed_states.add(state_name)

            # Get state object if it exists
            state_obj = self._states.get(state_name)

            # Determine state type
            state_type = "unknown"
            if state_obj:
                if hasattr(state_obj, "state_type"):
                    # Convert enum to string
                    if hasattr(state_obj.state_type, "value"):
                        state_type = state_obj.state_type.value.lower()
                    elif hasattr(state_obj.state_type, "name"):
                        state_type = state_obj.state_type.name.lower()
                    else:
                        state_type = str(state_obj.state_type).lower()

                    # Preserve state types for accurate visualization
                    if state_type == "chat":
                        state_type = "chat"
                    elif state_type == "collect":
                        state_type = "collect"
                    elif state_type == "tool":
                        state_type = "tool"
                    elif state_type == "decision":
                        state_type = "decision"
                    elif state_type == "summary":
                        state_type = "summary"
                elif hasattr(state_obj, "type"):
                    state_type = str(state_obj.type).lower()

            # Detect special state characteristics
            is_undefined = state_obj is None  # Referenced but never defined
            is_final = False

            # Check if this is a final state (no outgoing transitions)
            outgoing_transitions = [t for t in self._transitions if t.from_state == state_name]
            if len(outgoing_transitions) == 0 and state_name != self._start_state:
                is_final = True
                # Auto-infer final states as summary if not explicitly typed and not undefined
                if not is_undefined and state_type == "unknown" and state_obj:
                    state_type = "summary"

            # Create node
            node = {
                "id": state_name,
                "label": state_name.replace("_", " ").title(),
                "state_type": state_type,
                "is_start": state_name == self._start_state,
                "is_final": is_final,
                "is_undefined": is_undefined,
            }

            # Add description if available
            if state_obj and hasattr(state_obj, "description"):
                node["description"] = state_obj.description

            nodes.append(node)

        # Create edges from transitions
        for transition in self._transitions:
            edge = {"from": transition.from_state, "to": transition.to_state}

            # Add transition label if available
            if hasattr(transition, "condition") and transition.condition:
                edge["label"] = str(transition.condition)

            edges.append(edge)

        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "start_state": self._start_state,
                "total_states": len(nodes),
                "total_transitions": len(edges),
            },
        }

    def _repr_html_(self) -> str:
        """
        Generate HTML representation for notebook display.

        This method is automatically called when a Pathways object is displayed in a notebook,
        providing an inline visualization.

        Returns
        -------
        str
            HTML content with embedded pathway visualization
        """
        import uuid

        # Extract pathway data
        pathway_data = self._extract_pathway_data()

        # Create unique ID for this diagram
        diagram_id = f"pathway-{uuid.uuid4().hex[:8]}"

        # Build a visual flowchart representation instead of a strict tree
        def build_flowchart_html(pathway_data, diagram_id):
            """Build a flowchart-style visualization that handles reconvergence clearly."""
            nodes = {n["id"]: n for n in pathway_data["nodes"]}
            edges = pathway_data["edges"]

            # Find start node
            start_node_id = pathway_data["metadata"]["start_state"]

            # Build adjacency lists
            adjacency = {}  # node -> children
            parents = {}  # node -> parents

            for edge in edges:
                # Forward edges (node -> children)
                if edge["from"] not in adjacency:
                    adjacency[edge["from"]] = []
                adjacency[edge["from"]].append(edge["to"])

                # Backward edges (node -> parents)
                if edge["to"] not in parents:
                    parents[edge["to"]] = []
                parents[edge["to"]].append(edge["from"])

            # Color scheme for different state types
            colors = {
                "chat": {
                    "bg": "#E0F7FA",
                    "border": "#00ACC1",
                    "text": "#006064",
                },  # Teal - communication
                "collect": {
                    "bg": "#E8F5E8",
                    "border": "#4CAF50",
                    "text": "#2E7D32",
                },  # Green - gathering
                "tool": {
                    "bg": "#FFF3E0",
                    "border": "#FF9800",
                    "text": "#E65100",
                },  # Orange - work/action
                "decision": {
                    "bg": "#FFEBEE",
                    "border": "#F44336",
                    "text": "#C62828",
                },  # Red - critical thinking
                "summary": {
                    "bg": "#F3E5F5",
                    "border": "#9C27B0",
                    "text": "#4A148C",
                },  # Purple - conclusion
                "unknown": {
                    "bg": "#F5F5F5",
                    "border": "#9E9E9E",
                    "text": "#424242",
                },  # Gray - unknown
            }

            def create_node_box(node, is_reconvergence=False, node_id=None):
                """Create a styled box for a pathway node."""
                state_type = node.get("state_type", "unknown")
                color = colors.get(state_type, colors["unknown"])
                is_start = node.get("is_start", False)
                is_final = node.get("is_final", False)
                is_undefined = node.get("is_undefined", False)

                # Enhanced border styling for special states
                if is_start:
                    border_width = "3px"
                    border_style = "solid"
                elif is_undefined:
                    border_width = "3px"  # Make undefined states more prominent
                    border_style = "dashed"  # Clear visual indicator of missing definition
                elif is_final:
                    border_width = "2px"
                    border_style = "double"  # Double border for final states
                else:
                    border_width = "2px"
                    border_style = "solid"

                # Add a unique ID for connecting lines
                unique_id = (
                    f"node-{diagram_id}-{node_id or 'unknown'}-{hash(node['label']) % 10000}"
                )

                # Enhanced styling with better visual indicators
                additional_styles = ""
                if is_reconvergence:
                    additional_styles += f"background: linear-gradient(45deg, {color['bg']} 25%, transparent 25%, transparent 50%, {color['bg']} 50%, {color['bg']} 75%, transparent 75%, transparent); background-size: 8px 8px;"

                if is_undefined:
                    # Make undefined states more visually distinct
                    additional_styles += (
                        "box-shadow: 0 0 8px rgba(255, 0, 0, 0.3), 0 2px 6px rgba(0,0,0,0.1);"
                    )
                elif is_final:
                    # Subtle glow for final states
                    additional_styles += (
                        f"box-shadow: 0 0 6px {color['border']}30, 0 2px 6px rgba(0,0,0,0.1);"
                    )

                return f"""<div id="{unique_id}" class="pathway-node" style="
                    background: {color["bg"]};
                    border: {border_width} {border_style} {color["border"]};
                    border-radius: 8px;
                    padding: 12px 16px;
                    margin: 8px;
                    color: {color["text"]};
                    font-weight: 500;
                    text-align: center;
                    min-width: 140px;
                    position: relative;
                    {additional_styles}
                ">{node["label"]}</div>"""

            # Calculate proper levels for nodes, handling reconvergence
            def calculate_node_levels():
                """Calculate the optimal level for each node, ensuring reconvergence points are placed correctly."""
                node_levels = {}

                def get_max_parent_level(node_id):
                    """Get the maximum level of all parent nodes."""
                    if node_id == start_node_id:
                        return 0

                    parent_nodes = parents.get(node_id, [])
                    if not parent_nodes:
                        return 0

                    max_level = 0
                    for parent_id in parent_nodes:
                        if parent_id in node_levels:
                            max_level = max(max_level, node_levels[parent_id] + 1)
                        else:
                            # Recursively calculate parent level
                            parent_level = get_max_parent_level(parent_id)
                            node_levels[parent_id] = parent_level
                            max_level = max(max_level, parent_level + 1)

                    return max_level

                # Calculate levels for all nodes
                for node_id in nodes:
                    if node_id not in node_levels:
                        node_levels[node_id] = get_max_parent_level(node_id)

                return node_levels

            # Get optimal levels for all nodes
            level_nodes = calculate_node_levels()

            # Group nodes by their calculated levels
            levels_dict = {}
            max_level = 0
            for node_id, level in level_nodes.items():
                if level not in levels_dict:
                    levels_dict[level] = []
                levels_dict[level].append(node_id)
                max_level = max(max_level, level)

            # Build the flowchart level by level with connecting lines
            result_html = ""
            all_connections = []  # Store connections for drawing lines

            for level_counter in range(max_level + 1):
                if level_counter not in levels_dict:
                    continue

                level_html = f'<div class="pathway-level" data-level="{level_counter}" style="display: flex; justify-content: center; align-items: center; margin: 25px 0; flex-wrap: nowrap; position: relative; min-width: max-content;">'

                for node_id in levels_dict[level_counter]:
                    if node_id not in nodes:
                        continue

                    node = nodes[node_id]

                    # Check if this is a reconvergence point
                    is_reconvergence = len(parents.get(node_id, [])) > 1

                    # Check if this is a final state (no outgoing connections)
                    is_final_state = len(adjacency.get(node_id, [])) == 0

                    # Apply reconvergence styling to all reconvergence points (including final states)
                    use_reconvergence_style = is_reconvergence

                    # Add the node box with ID
                    level_html += create_node_box(node, use_reconvergence_style, node_id)

                    # Store connections to children for drawing lines later
                    children = adjacency.get(node_id, [])
                    for child_id in children:
                        all_connections.append((node_id, child_id))

                level_html += "</div>"
                result_html += level_html

            # Create SVG overlay for connection lines
            svg_connections = ""
            for parent_id, child_id in all_connections:
                parent_node_id = (
                    f"node-{diagram_id}-{parent_id}-{hash(nodes[parent_id]['label']) % 10000}"
                )
                child_node_id = (
                    f"node-{diagram_id}-{child_id}-{hash(nodes[child_id]['label']) % 10000}"
                )

                svg_connections += f"""
                    <path class="connection-line"
                          data-from="{parent_node_id}"
                          data-to="{child_node_id}"
                          stroke="#667eea"
                          stroke-width="2"
                          fill="none"
                          marker-end="url(#{diagram_id}-arrowhead)"
                          style="opacity: 0.7;">
                    </path>"""

            # Add connection lines with SVG
            connections_svg = f"""
            <svg class="pathway-connections" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; z-index: 1;">
                <defs>
                    <marker id="{diagram_id}-arrowhead" markerWidth="10" markerHeight="7"
                            refX="0" refY="3.5" orient="auto">
                        <polygon points="0 0, 10 3.5, 0 7" fill="#667eea" />
                    </marker>
                </defs>
                {svg_connections}
            </svg>

            <script>
                // Initialize pathway diagram manager if not already exists
                if (!window.pathwayDiagramManager) {{
                    window.pathwayDiagramManager = {{
                        diagrams: new Map(),
                        initialized: false,

                        // Register a diagram with its update function
                        register: function(diagramId, updateFunction) {{
                            this.diagrams.set(diagramId, updateFunction);
                            // If the manager is already initialized, immediately update this diagram
                            if (this.initialized) {{
                                setTimeout(updateFunction, 10);
                            }}
                        }},

                        // Initialize all diagrams
                        init: function() {{
                            if (this.initialized) return;
                            this.initialized = true;

                            // Update all registered diagrams
                            this.updateAll();

                            // Set up single global event listeners
                            window.addEventListener('resize', () => this.updateAll());
                        }},

                        // Update all registered diagrams
                        updateAll: function() {{
                            this.diagrams.forEach((updateFunction, diagramId) => {{
                                try {{
                                    updateFunction();
                                }} catch (error) {{
                                    console.warn(`Failed to update diagram ${{diagramId}}:`, error);
                                }}
                            }});
                        }}
                    }};

                    // Initialize when DOM is ready
                    if (document.readyState === 'loading') {{
                        document.addEventListener('DOMContentLoaded', () => {{
                            window.pathwayDiagramManager.init();
                        }});
                    }} else {{
                        // DOM already loaded
                        window.pathwayDiagramManager.init();
                    }}
                }}

                // Function to update connection lines for this specific diagram
                function updateConnections_{diagram_id.replace("-", "_")}() {{
                    const diagram = document.getElementById('{diagram_id}');
                    if (!diagram) return;

                    const connections = diagram.querySelectorAll('.connection-line');
                    connections.forEach(line => {{
                        const fromId = line.getAttribute('data-from');
                        const toId = line.getAttribute('data-to');

                        // Use scoped selection to ensure we get elements from the right diagram
                        const fromElement = diagram.querySelector(`#${{fromId}}`);
                        const toElement = diagram.querySelector(`#${{toId}}`);

                        if (fromElement && toElement) {{
                            const diagramRect = diagram.getBoundingClientRect();
                            const fromRect = fromElement.getBoundingClientRect();
                            const toRect = toElement.getBoundingClientRect();
                            const svgRect = line.closest('svg').getBoundingClientRect();

                            const fromX = fromRect.left + fromRect.width / 2 - svgRect.left;
                            const fromY = fromRect.bottom - svgRect.top;
                            const toX = toRect.left + toRect.width / 2 - svgRect.left;
                            // Stop the line 10 pixels before the box to leave space for the arrowhead
                            const toY = toRect.top - svgRect.top - 10;

                            // Create smooth curved connection
                            const midY = fromY + (toY - fromY) / 2;
                            const controlOffset = Math.abs(toY - fromY) * 0.3;

                            const path = `M${{fromX}},${{fromY}} C${{fromX}},${{fromY + controlOffset}} ${{toX}},${{toY - controlOffset}} ${{toX}},${{toY}}`;
                            line.setAttribute('d', path);
                        }}
                    }});
                }}

                // Register this diagram with the manager
                window.pathwayDiagramManager.register('{diagram_id}', updateConnections_{diagram_id.replace("-", "_")});
            </script>"""

            # Wrap result in a positioned container with unique ID
            result_html = f"""
            <div id="{diagram_id}" style="position: relative; margin: 20px 0;">
                {result_html}
                {connections_svg}
            </div>"""

            return (
                result_html if result_html else "<p>No pathway structure found</p>"
            )  # Create the flowchart HTML

        flowchart_structure = build_flowchart_html(pathway_data, diagram_id)

        # Create inline HTML with tree-based visualization
        html_content = f"""
        <div style="
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            border: 1px solid #e1e5e9;
            border-radius: 12px;
            padding: 24px;
            margin: 15px 0;
            background: white;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            max-width: 100%;
        ">
            <!-- Embedded CSS for tree visualization -->
            <style>
                #{diagram_id} .tree ul {{
                    position: relative;
                    padding: 1em 0;
                    white-space: nowrap;
                    margin: 0 auto;
                    text-align: center;
                }}

                #{diagram_id} .tree ul::after {{
                    content: '';
                    display: table;
                    clear: both;
                }}

                #{diagram_id} .tree li {{
                    display: inline-block;
                    vertical-align: top;
                    text-align: center;
                    list-style-type: none;
                    position: relative;
                    padding: 1em .5em 0 .5em;
                }}

                #{diagram_id} .tree li::before,
                #{diagram_id} .tree li::after {{
                    content: '';
                    position: absolute;
                    top: 0;
                    right: 50%;
                    border-top: 2px solid #667eea;
                    width: 50%;
                    height: 1em;
                }}

                #{diagram_id} .tree li::after {{
                    right: auto;
                    left: 50%;
                    border-left: 2px solid #667eea;
                }}

                #{diagram_id} .tree li:only-child::after,
                #{diagram_id} .tree li:only-child::before {{
                    display: none;
                }}

                #{diagram_id} .tree li:only-child {{
                    padding-top: 0;
                }}

                #{diagram_id} .tree li:first-child::before,
                #{diagram_id} .tree li:last-child::after {{
                    border: 0 none;
                }}

                #{diagram_id} .tree li:last-child::before {{
                    border-right: 2px solid #667eea;
                    border-radius: 0 5px 0 0;
                }}

                #{diagram_id} .tree li:first-child::after {{
                    border-radius: 5px 0 0 0;
                }}

                #{diagram_id} .tree ul ul::before {{
                    content: '';
                    position: absolute;
                    top: 0;
                    left: 50%;
                    border-left: 2px solid #667eea;
                    width: 0;
                    height: 1em;
                }}

                #{diagram_id} .tree li a:hover + ul li::after,
                #{diagram_id} .tree li a:hover + ul li::before,
                #{diagram_id} .tree li a:hover + ul::before,
                #{diagram_id} .tree li a:hover + ul ul::before {{
                    border-color: #e9453f;
                }}
            </style>            <!-- Header -->
            <div style="
                display: flex;
                align-items: center;
                margin-bottom: 18px;
                padding-bottom: 15px;
                border-bottom: 2px solid #e1e5e9;
            ">
                <div style="
                    width: 12px;
                    height: 12px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 50%;
                    margin-right: 15px;
                    flex-shrink: 0;
                "></div>
                <h3 style="
                    margin: 0;
                    color: #2c3e50;
                    font-size: 1.6em;
                    font-weight: 700;
                    line-height: 1.2;
                ">{self.title}</h3>
            </div>

            <!-- Description -->
            <div style="margin-bottom: 20px;">
                <p style="
                    margin: 0;
                    color: #555;
                    font-size: 1em;
                    line-height: 1.6;
                    font-style: italic;
                ">{self._description}</p>
            </div>

            <!-- Tree-based Flow Visualization -->
            <div id="{diagram_id}" style="
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                border-radius: 10px;
                padding: 30px;
                margin: 20px 0;
                border: 1px solid #dee2e6;
                overflow-x: auto;
                overflow-y: visible;
            ">
                <div class="flowchart" style="min-width: max-content;">
                    {flowchart_structure}
                </div>
            </div>

            <!-- Legend -->
            <div style="
                display: flex;
                gap: 12px;
                margin-top: 20px;
                font-size: 0.85em;
                flex-wrap: nowrap;
                justify-content: center;
                overflow-x: auto;
            ">
                <div style="
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    padding: 8px 12px;
                    background: #E0F7FA;
                    color: #006064;
                    border-radius: 8px;
                    font-weight: 600;
                    border: 1px solid #00ACC1;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                ">
                    <span style="width: 12px; height: 12px; background: #00ACC1; border-radius: 50%; flex-shrink: 0;"></span>
                    Chat
                </div>
                <div style="
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    padding: 8px 12px;
                    background: #E8F5E8;
                    color: #2E7D32;
                    border-radius: 8px;
                    font-weight: 600;
                    border: 1px solid #4CAF50;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                ">
                    <span style="width: 12px; height: 12px; background: #4CAF50; border-radius: 50%; flex-shrink: 0;"></span>
                    Collect
                </div>
                <div style="
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    padding: 8px 12px;
                    background: #FFF3E0;
                    color: #E65100;
                    border-radius: 8px;
                    font-weight: 600;
                    border: 1px solid #FF9800;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                ">
                    <span style="width: 12px; height: 12px; background: #FF9800; border-radius: 50%; flex-shrink: 0;"></span>
                    Tool
                </div>
                <div style="
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    padding: 8px 12px;
                    background: #FFEBEE;
                    color: #C62828;
                    border-radius: 8px;
                    font-weight: 600;
                    border: 1px solid #F44336;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                ">
                    <span style="width: 12px; height: 12px; background: #F44336; border-radius: 50%; flex-shrink: 0;"></span>
                    Decision
                </div>
                <div style="
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    padding: 8px 12px;
                    background: #F3E5F5;
                    color: #4A148C;
                    border-radius: 8px;
                    font-weight: 600;
                    border: 1px solid #9C27B0;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                ">
                    <span style="width: 12px; height: 12px; background: #9C27B0; border-radius: 50%; flex-shrink: 0;"></span>
                    Summary
                </div>
            </div>

            <!-- Footer Note -->
            <div style="
                text-align: center;
                margin-top: 20px;
                padding-top: 15px;
                border-top: 1px solid #e9ecef;
                color: #6c757d;
                font-size: 0.8em;
                font-style: italic;
            ">
                💡 Use <code>.visualize()</code> to save detailed pathway diagrams
            </div>
        </div>
        """

        return html_content
