from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


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

    The `Pathways` class provides intelligent conversation flow guidance while maintaining flexibility
    to adapt to natural conversation patterns. They serve as guardrails rather than rigid state
    machines, helping LLMs provide consistent, thorough assistance while remaining responsive to user
    needs and conversational context.

    Building a Pathway
    ------------------

    Building a pathway follows a specific sequence that ensures proper configuration and flow logic.
    Each step builds upon the previous one to create a coherent conversation structure.

    ### 1. Pathway Setup (call once, in order)

    ```python
    pathway = (
        tb.Pathways("Title")
        .description("Purpose and scope")           # Required: What this pathway does
        .activation_conditions([...])               # Required: When to use this pathway
        # First .state() call automatically becomes the starting state
    ```

    ### 2. State Definition (using unified .state() method)

    ```python
    pathway = (
        tb.Pathways("Support Flow")
        .description("Customer support pathway")
        .activation_conditions(["User needs help"])
        # === STATE: intake ===
        .state(id="intake", type="collect")        # First state is automatically the start
        .description("Gather customer information")
        .required(["issue description", "contact info"])
        .next_state("triage")
        # === STATE: triage ===
        .state(id="triage", type="decision")       # Simplified state definition
        .description("Route to appropriate support")
        .branch_on("Technical issue", "tech_support")
        .branch_on("Billing question", "billing")
        # === STATE: tech_support ===
        .state(id="tech_support")                  # Defaults to "chat" type
        .description("Resolve technical problems")
        .success_condition("Issue resolved")
    )
    ```

    This approach provides:
    - **Visual state boundaries** with `# === STATE: name ===` comments
    - **Unified state definition** with `.state(id="name", type="type")`
    - **Smart defaults** where `type="chat"` is assumed if not specified
    - **Automatic start state** where the first `.state()` becomes the starting state

    ### 3. State Configuration Pattern (repeat for each state):

    ```python
        # Define the state
        .state(id="state_name")                     # Defaults to type="chat"
        .state(id="state_name", type="collect")     # For gathering information
        .state(id="state_name", type="decision")    # For branching logic
        .state(id="state_name", type="tool")        # For using tools/APIs
        .state(id="state_name", type="summary")     # For wrapping up

        # Configure the state
        .description("What happens in this state")  # Required for all states
        .required([...])                            # What must be accomplished
        .optional([...])                            # What would be nice to have
        .tools([...])                               # Available tools (type="tool" only)
        .success_condition("When state succeeds")   # How to know it's complete

        # Define state transitions (choose one)
        .next_state("next_state")                   # Linear progression
        .branch_on("condition", "target_state")     # Conditional (type="decision" only)
        .merge_to("common_state")                   # Reconverge after branching
        .fallback("error_condition", "backup_state") # Error handling
    ```

    ### 4. Pathway Completion (call once at end):

    ```python
        .completion_criteria([...])                 # What makes pathway successful
        .fallback_strategy("...")                   # Handle unexpected situations
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
    - Always call `description()` immediately after `.state()`
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

    ```python
    import talk_box as tb

    simple_pathway = (
        tb.Pathways("Password Reset")
        .description("Help users reset their forgotten passwords")
        .activation_conditions(["User can't log in", "User forgot password"])
        # === STATE: verification ===
        .state(id="verification", type="collect")
        .description("Verify user identity")
        .required(["email_address", "account_verification"])
        .next_state("password_update")
        # === STATE: password_update ===
        .state(id="password_update")  # defaults to "chat"
        .description("Guide user through creating new password")
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

    ```python
    support_pathway = (
        tb.Pathways("Customer Support")
        .description("Route and resolve customer inquiries")
        .activation_conditions(["User needs help", "User reports problem"])
        # === STATE: triage ===
        .state(id="triage", type="decision")
        .description("Determine the type of support needed")
        .branch_on("Technical problem reported", "technical_support")
        .branch_on("Billing question asked", "billing_support")
        .branch_on("General inquiry made", "general_help")
        # === STATE: technical_support ===
        .state(id="technical_support", type="tool")
        .description("Diagnose and resolve technical issues")
        .tools(["system_diagnostics", "troubleshooting_guide"])
        .success_condition("Technical issue is resolved")
        .merge_to("completion")
        # === STATE: billing_support ===
        .state(id="billing_support")  # defaults to "chat"
        .description("Address billing and account questions")
        .required(["billing_issue_understood", "solution_provided"])
        .merge_to("completion")
        # === STATE: completion ===
        .state(id="completion", type="summary")
        .description("Ensure customer satisfaction and wrap up")
        .required(["issue_resolved_confirmation", "follow_up_if_needed"])
        .completion_actions(["log_interaction", "send_summary_email"])
        .completion_criteria(["Customer issue fully resolved", "Customer satisfied"])
        .fallback_strategy("If issue is complex, escalate to human support")
    )
    ```

    This branching example shows how `.state()` creates clear decision points that route conversations
    appropriately, then merge back together for consistent completion.
    """

    def __init__(self, title: str):
        """
        Initialize a new pathway.

        Parameters
        ----------
        title : str
            Short, descriptive name for the pathway
        """
        self.title = title
        self._description: str = ""
        self._activation_conditions: List[str] = []
        self._states: Dict[str, PathwayState] = {}
        self._transitions: List[PathwayTransition] = []
        self._current_state_name: Optional[str] = None
        self._start_state: Optional[str] = None
        self._completion_criteria: List[str] = []
        self._fallback_strategy: Optional[str] = None

    def description(self, desc: str) -> "Pathways":
        """
        Set the pathway description.

        Call this early in your pathway definition, typically right after initialization. This
        provides essential context for when and why this pathway should be used, helping both
        developers and LLMs understand the pathway's purpose and scope.

        Parameters
        ----------
        desc : str
            Clear, concise explanation of the pathway's purpose and scope. Focus on what problems
            this pathway solves and what outcomes it achieves.

        Examples
        --------
        Begin every pathway with a clear description that explains its purpose:

        ```python
        pathway = (
            tb.Pathways("Customer Support")
            .description("Handle customer inquiries with systematic troubleshooting")
            # Continue with activation_conditions() and start_with()...
        )
        ```

        Notes
        -----
        Must be called at the pathway level, before defining any states. This description becomes
        part of the system prompt to help the LLM understand when and how to use the pathway.
        """
        self._description = desc
        return self

    def activation_conditions(self, conditions: List[str]) -> "Pathways":
        """
        Define when this pathway should be activated in conversation.

        Call this after `description()` and before `start_with()`. These conditions help the LLM
        determine when to follow this structured flow versus free conversation. Well-defined
        activation conditions ensure the pathway triggers at appropriate times without being too
        restrictive or too broad.

        Parameters
        ----------
        conditions
            Specific situations or user intents that trigger pathway activation. Use concrete,
            observable conditions rather than vague descriptions. Each condition should represent
            a clear scenario where this pathway would be helpful.

        Examples
        --------
        Be specific about when the pathway should activate. These conditions guide the LLM's
        decision-making:

        ```python
        .activation_conditions([
            "User reports a technical problem",
            "User asks for help troubleshooting",
            "User mentions error messages or system failures"
        ])
        ```

        Good conditions are observable and specific, while poor conditions are vague or too broad:

        ```python
        # Good (specific and actionable)
        "User wants to book a flight for specific dates"

        # Poor (too vague)
        "User needs help"
        ```

        Notes
        -----
        - Be specific: "User wants to book a flight" not "User needs help"
        - Use present tense and active voice for clarity
        - 3-5 conditions typically provide good coverage without complexity
        - Conditions should be mutually exclusive with other pathways when possible
        """
        self._activation_conditions = conditions
        return self

    def state(self, id: str, type: str = "chat") -> "Pathways":
        """
        Define a state with the specified type in a single method call.

        The first state you define becomes the starting state automatically. The `type`
        parameter defaults to "chat" since it's the most common state type.

        Parameters
        ----------
        id
            Unique identifier for the state. Use descriptive names that clearly indicate
            the state's purpose, like `greeting`, `problem_assessment`, or `verification`.
        type
            The type of state to create. Options are:
            - "chat" (default): Open conversation, explanations, guidance
            - "collect": Structured information gathering
            - "decision": Branching logic based on conditions
            - "tool": Using specific tools or APIs
            - "summary": Conclusions, confirmations, wrap-up        Examples
        --------
        Create states directly without separate type methods:

        ```python
        pathway = (
            tb.Pathways("Customer Support")
            .description("Help resolve customer issues")
            .state(id="greeting")  # defaults to chat
            .description("Welcome customer and understand their needs")
            .collect(["problem_description", "urgency_level"])
            .next_state("triage")
            .state(id="triage", type="decision")
            .description("Determine support approach needed")
            .branch_on("Technical issue", "troubleshooting")
            .branch_on("Billing question", "billing_help")
        )
        ```

        Notes
        -----
        - First `.state()` call automatically becomes the starting state
        - Must be followed immediately by `.description()`
        - State IDs must be unique within the pathway
        - Use lowercase with underscores for state IDs
        """
        # Set as start state if this is the first state defined
        if not self._start_state:
            self._start_state = id

        # Set current state for subsequent configuration
        self._current_state_name = id

        # Create the state based on type
        state_type_map = {
            "chat": StateType.CHAT,
            "collect": StateType.COLLECT,
            "decision": StateType.DECISION,
            "tool": StateType.TOOL,
            "summary": StateType.SUMMARY,
        }

        if type not in state_type_map:
            raise ValueError(
                f"Invalid state type '{type}'. Must be one of: {list(state_type_map.keys())}"
            )

        # Create the PathwayState
        self._states[id] = PathwayState(
            name=id,
            state_type=state_type_map[type],
            description="",  # Will be set by subsequent .description() call
            required_info=[],
            optional_info=[],
            tools=[],
            success_conditions=[],
            next_states=[],
        )

        return self

    def description(self, desc: str) -> "Pathways":
        """
        Add description to the current state.

        This method behaves differently depending on context:
        - At pathway level (before any state definition): Sets pathway description
        - At state level (after state() method): Sets state description

        Always call immediately after .state() method.
        Provides essential context for what happens in this state.

        Parameters
        ----------
        desc : str
            Clear description of the state's purpose and what should happen.
            Be specific about the expected interaction or outcome.

        Examples
        --------
        ```python
        .state("assessment")
            .description("Ask questions to understand the user's specific problem")
            # Continue with collect(), required(), etc.
        ```

        Notes
        -----
        - State descriptions guide LLM behavior within that state
        - Be specific about what should happen, not just what the state is
        - Use action-oriented language: "Ask...", "Explain...", "Gather..."
        """
        if self._current_state_name and self._current_state_name in self._states:
            self._states[self._current_state_name].description = desc
        else:
            self._description = desc
        return self

    def collect(self, info_types: List[str]) -> "Pathways":
        """
        Specify information to collect in the current state.

        Use in any state to define what information the LLM should gather from
        the user. This is an alias for required() - use whichever reads better
        in your context.

        Parameters
        ----------
        info_types : List[str]
            Specific types of information to gather. Be concrete and actionable.

        Examples
        --------
        ```python
        .state("user_registration", type="collect")
            .description("Gather user registration information")
            .collect(["full_name", "email_address", "preferred_contact_method"])
        ```

        Notes
        -----
        - Same as required() - use whichever reads better
        - Be specific: "email address" not "contact info"
        - Works with any state type but most natural with type="collect"
        """
        if self._current_state_name in self._states:
            self._states[self._current_state_name].required_info.extend(info_types)
        return self

    def required(self, info_types: List[str]) -> "Pathways":
        """
        Specify required information for the current state to be considered complete.

        Use after description() to define what must be obtained before the state
        can transition to the next step. The LLM will focus on gathering this
        information before proceeding.

        Parameters
        ----------
        info_types : List[str]
            Essential information that must be collected or established.
            Be specific and measurable.

        Examples
        --------
        ```python
        .state("booking_details")
            .description("Get essential travel information")
            .required(["departure_city", "destination", "travel_date"])
            .optional(["return_date", "time_preference"])
            .next_state("search_flights")
        ```

        Notes
        -----
        - State cannot progress until required items are addressed
        - Be specific and concrete
        - Pair with optional() for nice-to-have information
        - Use success_condition() to define when requirements are truly met
        """
        if self._current_state_name in self._states:
            self._states[self._current_state_name].required_info.extend(info_types)
        return self

    def optional(self, info_types: List[str]) -> "Pathways":
        """
        Specify optional information that would be helpful but not required.

        Use with required() to define nice-to-have information that can improve
        the outcome but isn't essential for state completion. The LLM will
        attempt to gather this if the conversation allows.

        Parameters
        ----------
        info_types : List[str]
            Additional information that would be beneficial but not essential.

        Examples
        --------
        ```python
        .required(["departure_city", "destination", "travel_date"])
        .optional(["airline_preference", "seat_preference", "meal_requirements"])
        ```

        Notes
        -----
        - State can progress without optional items
        - Helps create more comprehensive outcomes when available
        - Use sparingly - too many optionals can slow the flow
        - Best used in collect_state() or structured chat_state()
        """
        if self._current_state_name in self._states:
            self._states[self._current_state_name].optional_info.extend(info_types)
        return self

    def tools(self, tool_names: List[str]) -> "Pathways":
        """
        Specify tools available for use in the current state.

        Essential for type="tool" states, but can also be used in other states
        where specific capabilities are needed. Follow this with success_condition()
        to define when tool usage is complete.

        Parameters
        ----------
        tool_names : List[str]
            Names of specific tools or capabilities the LLM should use.
            These should match actual available tools.

        Examples
        --------
        ```python
        .state("flight_search", type="tool")
            .description("Find flights matching user criteria")
            .tools(["flight_search_api", "price_comparison_tool"])
            .success_condition("Found at least 3 flight options")
            .next_state("present_options")
        ```

        Notes
        -----
        - Required for type="tool" states, optional for others
        - Tool names should match actual available capabilities
        - Use success_condition() to define completion criteria
        - Consider fallback() for when tools fail
        """
        if self._current_state_name in self._states:
            self._states[self._current_state_name].tools.extend(tool_names)
        return self

    def success_condition(self, condition: str) -> "Pathways":
        """
        Define what indicates successful completion of the current state.

        Use after configuring state requirements to specify when the state's
        objectives are met and it's ready to transition. More specific than
        just completing required() items.

        Parameters
        ----------
        condition : str
            Specific, observable condition indicating the state succeeded.
            Use action-oriented language that the LLM can recognize.

        Examples
        --------
        ```python
        .state("explanation")
            .description("Explain the solution to the user")
            .required(["solution_steps", "expected_outcome"])
            .success_condition("User confirms understanding and agrees to proceed")
            .next_state("implementation")
        ```

        Notes
        -----
        - More specific than just completing required() items
        - Should be observable/confirmable in conversation
        - Use active voice: "User confirms..." not "User understanding confirmed"
        - Can have multiple success conditions for complex states
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
        state_name : str
            Name of the state to transition to next. The target state must be
            defined later in the pathway using state().

        Examples
        --------
        ```python
        .state("greeting")
            .description("Welcome user and understand their needs")
            .required(["user_goal", "urgency"])
            .success_condition("User has explained their situation")
            .next_state("assessment")  # Linear progression
        ```

        Notes
        -----
        - Creates unconditional transition after state completion
        - Cannot be used with type="decision" states - use branch_on() instead
        - Target state must be defined later with state()
        - For conditional logic, use branch_on()
        """
        if self._current_state_name:
            self._transitions.append(
                PathwayTransition(from_state=self._current_state_name, to_state=state_name)
            )
        return self

    def branch_on(self, condition: str, state_name: str) -> "Pathways":
        """
        Define conditional branch to another state based on specific conditions.

        Use with decision_state() to create multiple possible transitions based
        on user responses, detected conditions, or conversation context. Each
        branch should represent a distinct path through the workflow.

        Parameters
        ----------
        condition : str
            Specific, recognizable condition that triggers this branch.
            Be concrete and observable in conversation.
        state_name : str
            Target state name for this branch condition.

        Examples
        --------
        ```python
        .decision_state("triage")
            .description("Determine support type needed")
            .branch_on("User reports technical error", "technical_support")
            .branch_on("User has billing question", "billing_support")
            .branch_on("User needs general help", "general_assistance")
        ```

        Notes
        -----
        - Use only with decision_state(), not with other state types
        - Conditions should be mutually exclusive when possible
        - Each branch must lead to a state defined with then()
        - Be specific: "User mentions password issues" not "User has problems"
        """
        if self._current_state_name:
            self._transitions.append(
                PathwayTransition(
                    from_state=self._current_state_name, to_state=state_name, condition=condition
                )
            )
        return self

    def merge_to(self, state_name: str) -> "Pathways":
        """
        Merge current state path back to a common state.

        Use when multiple branches need to converge back to a single workflow
        state. This is essentially an alias for next_state() but makes the
        intent clearer in branching scenarios.

        Parameters
        ----------
        state_name : str
            Common state where multiple paths converge.

        Examples
        --------
        ```python
        .then("billing_help")
            .chat_state("billing_help")
            .description("Address billing-specific questions")
            .merge_to("completion")  # Merge back to common end
        .then("technical_help")
            .chat_state("technical_help")
            .description("Provide technical assistance")
            .merge_to("completion")  # Same convergence point
        ```

        Notes
        -----
        - Functionally identical to next_state() but clearer for branching
        - Use after branch paths to show convergence
        - Target state should typically be defined later
        """
        return self.next_state(state_name)

    def fallback(self, condition: str, state_name: str) -> "Pathways":
        """
        Define fallback transition when normal state progression fails.

        Use when you need to handle error conditions, user confusion, or when
        expected outcomes don't occur. Provides graceful recovery paths instead
        of getting stuck in a state.

        Parameters
        ----------
        condition : str
            Specific condition that triggers the fallback. Usually describes
            a failure or unexpected situation.
        state_name : str
            State to transition to when fallback condition occurs.

        Examples
        --------
        ```python
        .chat_state("solution_explanation")
            .description("Explain the recommended solution")
            .success_condition("User understands and accepts solution")
            .fallback("User expresses confusion or disagreement", "clarification")
            .next_state("implementation")
        ```

        Notes
        -----
        - Use for error handling and recovery
        - Condition should describe failure scenarios
        - Provides graceful degradation instead of getting stuck
        - Can be used alongside next_state() or branch_on()
        """
        if self._current_state_name in self._states:
            self._states[self._current_state_name].fallback_actions.append(
                f"If {condition}, go to {state_name}"
            )
        return self

    def completion_actions(self, actions: List[str]) -> "Pathways":
        """
        Define actions to take when the state completes successfully.

        Use in summary_state() or final states to specify follow-up actions
        that should occur after successful completion. These might be system
        actions, notifications, or next steps.

        Parameters
        ----------
        actions : List[str]
            Specific actions to perform upon successful state completion.
            Use action-oriented language.

        Examples
        --------
        ```python
        .summary_state()
            .description("Finalize and confirm the booking")
            .required(["booking_details_confirmed", "payment_processed"])
            .completion_actions([
                "send_confirmation_email",
                "update_booking_system",
                "schedule_reminder_notifications"
            ])
        ```

        Notes
        -----
        - Best used in summary_state() or final states
        - Describes what happens after successful completion
        - Use specific action verbs: "send", "update", "schedule"
        - Can represent both system actions and user guidance
        """
        if self._current_state_name in self._states:
            self._states[self._current_state_name].fallback_actions.extend(actions)
        return self

    def completion_criteria(self, criteria: List[str]) -> "Pathways":
        """
        Define overall criteria for considering the entire pathway complete.

        Use at the pathway level (after all states defined) to specify what
        constitutes successful completion of the entire conversation flow.
        These are higher-level than individual state success conditions.

        Parameters
        ----------
        criteria : List[str]
            High-level conditions that indicate the pathway's objectives
            have been fully achieved.

        Examples
        --------
        ```python
        .completion_criteria([
            "User's technical issue has been resolved",
            "User knows how to prevent similar problems",
            "User is satisfied with the support experience"
        ])
        ```

        Notes
        -----
        - Use at pathway level, not state level
        - Higher-level than individual state success conditions
        - Describes overall pathway success
        - Usually called near the end of pathway definition
        """
        self._completion_criteria.extend(criteria)
        return self

    def fallback_strategy(self, strategy: str) -> "Pathways":
        """
        Define the overall strategy for handling unexpected situations.

        Use at the pathway level to provide general guidance for when the
        structured flow doesn't fit the conversation or when users go off-script.
        This is the meta-level fallback for the entire pathway.

        Parameters
        ----------
        strategy : str
            General approach for handling situations where the pathway
            doesn't apply or users need different support.

        Examples
        --------
        ```python
        .fallback_strategy(
            "If user's needs don't fit this pathway, acknowledge their specific "
            "situation and provide direct assistance while noting pathway limitations"
        )
        ```

        Notes
        -----
        - Use at pathway level, not state level
        - Provides meta-guidance for when pathway doesn't apply
        - Should encourage helpful, adaptive responses
        - Usually the last method called in pathway definition
        """
        self._fallback_strategy = strategy
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
        """Return a string representation of the pathway."""
        data = self._build()
        return f"Pathways('{data['title']}', states={len(data['states'])}, conditions={len(data.get('activation_conditions', []))})"

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

    def to_prompt_text(self) -> str:
        """
        Generate text specification for inclusion in system prompts.

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

    def generate_prompt(self) -> str:
        """
        Generate text specification for inclusion in system prompts.

        This is an alias for to_prompt_text() for backwards compatibility.

        Returns
        -------
        str
            Formatted pathway specification for LLM consumption
        """
        return self.to_prompt_text()
