from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union


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

    ### 2. State Definition (using unified `.state()`method)

    ```python
    pathway = (
        tb.Pathways(
            title="Support Flow",
            desc="Customer support pathway",
            activation="User needs help"
        )
        # === STATE: intake ===
        .state("Gather customer information", id="intake")
        .required(["issue description", "contact info"])
        .next_state("triage")
        # === STATE: triage ===
        .state("Route to appropriate support", id="triage")
        .branch_on("Technical issue", id="tech_support")
        .branch_on("Billing question", id="billing")
        # === STATE: tech_support ===
        .state("Resolve technical problems", id="tech_support")
        .success_condition("Issue resolved")
    )
    ```

    This approach provides:

    - **Visual state boundaries** with `# === STATE: description ===` comments
    - **Natural state definition** with description first: `.state("What happens here", id="name")`
    - **Smart type inference** where `.tools()` → tool, `.branch_on()` → decision, `.required()` → collect
    - **Automatic start state** where the first `.state()` becomes the starting state

    ### 3. State Configuration Pattern (repeat for each state):

    ```python
        # Define the state with description first
        .state("What happens in this state", id="state_name")
        .state("Gather information", id="state_name")  # type inferred as "collect" from .required()
        .state("Make decisions", id="state_name")      # type inferred as "decision" from .branch_on()
        .state("Use tools/APIs", id="state_name")      # type inferred as "tool" from .tools()
        .state("Wrap up")                              # Linear states don't need IDs

        # Configure the state
        .required([...])                            # What must be accomplished
        .optional([...])                            # What would be nice to have
        .tools([...])                               # Available tools (infers type="tool")
        .success_condition("When state succeeds")   # How to know it's complete

        # Define state transitions (choose one)
        .next_state("next_state")                   # Linear progression
        .branch_on("condition", id="target_state")  # Conditional (infers type="decision")
        .merge_to("common_state")                   # Reconverge after branching
        .fallback("error_condition", "backup_state") # Error handling
    ```

    ### 4. Pathway Completion (call once at end):

    ```python
                ### 4. Pathway Completion (now in constructor):

        ```python
        tb.Pathways(
            title="Support Flow",
            desc="Customer support pathway",
            activation="User needs help",
            completion_criteria=[...],                 # What makes pathway successful
            fallback_strategy="..."                    # Handle unexpected situations
        )
    ```

    State Types and Their Purpose
    -----------------------------
    Each state type serves a specific role in the conversation flow:

    - **`type="chat"`**: Open conversation, explanations, guidance (default)
    - **`type="decision"`**: Branching logic, must use `branch_on()` not `next_state()`
    - **`type="collect"`**: Structured information gathering
    - **`type="tool"`**: Using specific tools or APIs, requires `tools()`
    - **`type="summary"`**: Conclusions, confirmations, completion actions

    Key Rules
    ---------
    - escription is required and provided in `.state()` method
    - `type="decision"` must use `branch_on()`, never `next_state()`
    - `type="tool"` must include `tools()` specification
    - State names must be unique and use `lowercase_with_underscores`
    - Target states in transitions must be defined later with another `.state()`

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
        .state("Verify user identity", id="verification")
        .required(["email_address", "account_verification"])
        .next_state("password_update")
        # === STATE: password_update ===
        .state("Guide user through creating new password", id="password_update")
        .required(["new_password_created", "password_requirements_met"])
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
            desc="Route and resolve customer inquiries",
            activation=["User needs help", "User reports problem"],
            completion_criteria=["Customer issue fully resolved", "Customer satisfied"],
            fallback_strategy="If issue is complex, escalate to human support"
        )
        # === STATE: triage ===
        .state("Determine the type of support needed", id="triage")
        .branch_on("Technical problem reported", id="technical_support")
        .branch_on("Billing question asked", id="billing_support")
        .branch_on("General inquiry made", id="general_help")
        # === STATE: technical_support ===
        .state("Diagnose and resolve technical issues", id="technical_support")
        .tools(["system_diagnostics", "troubleshooting_guide"])
        .success_condition("Technical issue is resolved")
        .merge_to("completion")
        # === STATE: billing_support ===
        .state("Address billing and account questions", id="billing_support")
        .required(["billing_issue_understood", "solution_provided"])
        .merge_to("completion")
        # === STATE: completion ===
        .state("Ensure customer satisfaction and wrap up", id="completion", type="summary")
        .required(["issue_resolved_confirmation", "follow_up_if_needed"])
        .completion_actions(["log_interaction", "send_summary_email"])
    )
    ```

    This branching example shows how `.state()` creates clear decision points that route conversations
    appropriately, then merge back together for consistent completion.

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
        # === STATE: intake ===
        .state("Understand the problem", id="intake")
        .required(["issue_description"])
        .next_state("solution")
        # === STATE: solution ===
        .state("Provide solution", id="solution")
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
            identifier and should be specific about the expected interaction or outcome.
        id
            Optional unique identifier for the state. Required only when other states need to
            reference this state (via `.branch_on()`, `.next_state()`, `.merge_to()`). If not
            provided, an ID will be auto-generated from the description.
        type
            Optional explicit state type. If not provided, the type will be inferred from subsequent
            method calls

        Conflict Resolution
        -------------------
        Based on method usage, the state type is inferred as follows:

        - `.tools()` → `"tool"`
        - `.branch_on()` → `"decision"`
        - `.required()` → `"collect"`
        - Default → `"chat"`

        If multiple methods suggest different types, the first inference takes precedence.

        Examples
        --------
        Natural API with type inference:

        ```python
        pathway = (
            tb.Pathways(
                title="Customer Support",
                desc="Help resolve customer issues"
            )
            .state("Welcome customer and understand their needs")
            .required(["problem_description", "urgency_level"])
            .state("Determine support approach needed", id="triage")
            .branch_on("Technical issue", id="troubleshooting")
            .branch_on("Billing question", id="billing_help")
            .state("Provide technical troubleshooting", id="troubleshooting")
            .tools(["diagnostic_tool", "knowledge_base"])
        )
        ```

        Explicit type specification (useful for complex workflows):

        ```python
        pathway = (
            tb.Pathways(
                title="Data Analysis Pipeline",
                desc="Guide user through structured data analysis"
            )
            .state("Gather analysis requirements", id="requirements", type="collect")
            .required(["dataset_info", "analysis_goals"])
            .state("Choose analysis approach", id="method_selection", type="decision")
            .branch_on("Statistical analysis needed", id="stats")
            .branch_on("Machine learning required", id="ml_pipeline")
            .state("Run statistical tests", id="stats", type="tool")
            .tools(["scipy_stats", "hypothesis_testing"])
            .state("Summarize findings", id="summary", type="summary")
        )
        ```

        When to use explicit types:

        - **Complex workflows** where type inference might be ambiguous
        - **Documentation clarity** when the state's purpose isn't obvious from methods
        - **Team development** to make intentions explicit for other developers
        - **Mixed functionality** when a state serves multiple purposes (e.g., both collecting info
        and using tools)
        - **Error prevention** to avoid unintended type inference conflicts

        Notes
        -----
        - description always comes first
        - ID only needed when other states need to reference this state
        - type inferred from usage: `.tools()` → `"tool"`, `.branch_on()` → `"decision"`
        - first method call determines type, and conflicts generate warnings
        - auto-generated IDs use `snake_case` from description
        - use `# === STATE: name ===` comments for visual state separation in complex pathways
        """
        # Generate ID from description if not provided
        if id is None:
            # Create snake_case ID from description
            import re

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

    def _infer_state_type(self, new_type: str) -> None:
        """
        Infer and update state type based on method usage.

        Parameters
        ----------
        new_type
            The type being inferred from method usage.
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

        # Update state type if it's still default "chat" or matches
        if current_state.state_type.value in ["chat", new_type]:
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

        Examples
        --------
        ```python
        .state("booking_details", desc="Get essential travel information")
            .required(["departure_city", "destination", "travel_date"])
            .optional(["return_date", "time_preference"])
            .next_state("search_flights")

        # Or for a single item
        .state("get_email", desc="Collect user email")
            .required("email_address")
        ```

        Notes
        -----
        - state cannot progress until required items are addressed
        - be specific and concrete
        - pair with `.optional()` for nice-to-have information
        - use `.success_condition()` to define when requirements are truly met
        - infers state type as `"collect"` if not explicitly set
        """
        # Infer state type as "collect"
        self._infer_state_type("collect")

        # Convert string to list if needed
        if isinstance(info_types, str):
            info_types = [info_types]

        if self._current_state_name in self._states:
            self._states[self._current_state_name].required_info.extend(info_types)
        return self

    def optional(self, info_types: Union[str, List[str]]) -> "Pathways":
        """
        Specify optional information that would be helpful but not required.

        Use with required() to define nice-to-have information that can improve the outcome but
        isn't essential for state completion. The LLM will attempt to gather this if the
        conversation allows.

        Parameters
        ----------
        info_types
            Additional information that would be beneficial but not essential. Can be a single
            string or a list of strings.

        Examples
        --------
        ```python
        .required(["departure_city", "destination", "travel_date"])
        .optional(["airline_preference", "seat_preference", "meal_requirements"])

        # Or for a single optional item
        .required("user_email")
        .optional("phone_number")
        ```

        Notes
        -----
        - state can progress without optional items
        - helps create more comprehensive outcomes when available
        - use sparingly as too many optionals can slow the flow
        - best used in states with `type="collect"` or structured chat states
        """
        # Convert string to list if needed
        if isinstance(info_types, str):
            info_types = [info_types]

        if self._current_state_name in self._states:
            self._states[self._current_state_name].optional_info.extend(info_types)
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

        Examples
        --------
        ```python
        .state("Find flights matching user criteria")
            .tools(["flight_search_api", "price_comparison_tool"])
            .success_condition("Found at least 3 flight options")
            .next_state("present_options")

        # Or for a single tool
        .state("Search database")
            .tools("database_search")
        ```

        Notes
        -----
        - infers state type as `"tool"` if not explicitly set
        - tool names should match actual available capabilities
        - Use `.success_condition()` to define completion criteria
        - Consider `.fallback()` for when tools fail
        """
        # Infer state type as "tool"
        self._infer_state_type("tool")

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
        specific than just completing required() items. Can be used in any order within the state
        configuration.

        Parameters
        ----------
        condition : str
            Specific, observable condition indicating the state succeeded. Use action-oriented
            language that the LLM can recognize.

        Examples
        --------
        ```python
        .state("explanation", desc="Explain the solution to the user")
            .required(["solution_steps", "expected_outcome"])
            .success_condition("User confirms understanding and agrees to proceed")
            .next_state("implementation")
        ```

        Notes
        -----
        - more specific than just completing required() items
        - should be observable/confirmable in conversation
        - use active voice: `"User confirms..."` not `"User understanding confirmed"`
        - can have multiple success conditions for complex states
        """
        if self._current_state_name in self._states:
            self._states[self._current_state_name].success_conditions.append(condition)
        return self

    def next_state(self, state_name: str) -> "Pathways":
        """
        Define direct transition to the next state.

        Use for linear progression after state completion. Do not use with
        type="decision" states - use branch_on() instead. This creates
        unconditional forward movement in the pathway.

        Parameters
        ----------
        state_name
            Name of the state to transition to next. The target state must be defined later in the
            pathway using `.state()`.

        Examples
        --------
        ```python
        .state("greeting", desc="Welcome user and understand their needs")
            .required(["user_goal", "urgency"])
            .success_condition("User has explained their situation")
            .next_state("assessment")  # Linear progression
        ```

        Notes
        -----
        - creates unconditional transition after state completion
        - cannot be used with `type="decision"` states; use `.branch_on()` instead
        - target state must be defined later with `.state()`
        - for conditional logic, use `.branch_on()`
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

        Examples
        --------
        ```python
        .state("Determine support type needed", id="triage")
            .branch_on("User reports technical error", id="technical_support")
            .branch_on("User has billing question", id="billing_support")
            .branch_on("User needs general help", id="general_assistance")
        ```

        Notes
        -----
        - infers current state type as `"decision"` if not explicitly set
        - conditions should be mutually exclusive when possible
        - each branch must lead to a state defined later with `.state()`
        - be specific: `"User mentions password issues"` not `"User has problems"`
        """
        # Infer state type as "decision"
        self._infer_state_type("decision")

        if self._current_state_name:
            self._transitions.append(
                PathwayTransition(
                    from_state=self._current_state_name, to_state=id, condition=condition
                )
            )
        return self

    def merge_to(self, state_name: str) -> "Pathways":
        """
        Merge current state path back to a common state.

        Use when multiple branches need to converge back to a single workflow state. This is
        essentially an alias for `.next_state()` but makes the intent clearer in branching
        scenarios.

        Parameters
        ----------
        state_name
            Common state where multiple paths converge.

        Examples
        --------
        ```python
        .state("Address billing-specific questions", id="billing_help")
            .required(["billing_issue_understood", "solution_provided"])
            .merge_to("completion")  # Merge back to common end
        .state("Provide technical assistance", id="technical_help")
            .tools(["diagnostic_tool", "troubleshooting_guide"])
            .merge_to("completion")  # Same convergence point
        ```

        Notes
        -----
        - functionally identical to `.next_state()` but clearer for branching
        - use after branch paths to show convergence
        - target state should typically be defined later
        """
        return self.next_state(state_name)

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

        Examples
        --------
        ```python
        .state("Explain the recommended solution", id="solution_explanation")
            .success_condition("User understands and accepts solution")
            .fallback("User expresses confusion or disagreement", "clarification")
            .next_state("implementation")
        ```

        Notes
        -----
        - use for error handling and recovery
        - condition should describe failure scenarios
        - provides graceful degradation instead of getting stuck
        - can be used alongside `.next_state()` or `.branch_on()`
        """
        if self._current_state_name in self._states:
            self._states[self._current_state_name].fallback_actions.append(
                f"If {condition}, go to {state_name}"
            )
        return self

    def completion_actions(self, actions: Union[str, List[str]]) -> "Pathways":
        """
        Define actions to take when the state completes successfully.

        Use in summary states or final states to specify follow-up actions that should occur after
        successful completion. These might be system actions, notifications, or next steps.

        Parameters
        ----------
        actions
            Specific actions to perform upon successful state completion. Can be a single string or
            a list of strings. Use action-oriented language.

        Examples
        --------
        ```python
        .state("Finalize and confirm the booking", id="completion", type="summary")
            .required(["booking_details_confirmed", "payment_processed"])
            .completion_actions([
                "send_confirmation_email",
                "update_booking_system",
                "schedule_reminder_notifications"
            ])

        # Or for a single action
        .state("Complete order processing", type="summary")
            .completion_actions("send_confirmation_email")
        ```

        Notes
        -----
        - best used in states with `type="summary"` or final states
        - describes what happens after successful completion
        - use specific action verbs: `"send"`, `"update"`, `"schedule"`
        - can represent both system actions and user guidance
        """
        # Convert string to list if needed
        if isinstance(actions, str):
            actions = [actions]

        if self._current_state_name in self._states:
            self._states[self._current_state_name].fallback_actions.extend(actions)
        return self

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
        return self._to_prompt_text()

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

        # Flow guidance - show states with clear branching structure
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

            # Required information
            if state.get("required_info"):
                state_lines.append(f"{indent}  Required: {', '.join(state['required_info'])}")

            # Optional information
            if state.get("optional_info"):
                state_lines.append(f"{indent}  Optional: {', '.join(state['optional_info'])}")

            # Tools
            if state.get("tools"):
                state_lines.append(f"{indent}  Tools: {', '.join(state['tools'])}")

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
            lines.append(f"Complete when: {'; '.join(spec['completion_criteria'])}")

        # Fallback strategy
        if spec.get("fallback_strategy"):
            lines.append(f"Fallback: {spec['fallback_strategy']}")

        lines.append(
            "Follow as flexible guidance, adapting to user conversation patterns while ensuring key objectives are addressed."
        )

        return "\n".join(lines)
