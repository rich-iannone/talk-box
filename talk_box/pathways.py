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
    Chainable builder for defining conversational pathways.

    Pathways provide structured conversation flow guidance while maintaining the flexibility
    to adapt to natural conversation patterns. They serve as intelligent guardrails rather
    than rigid state machines.

    Examples
    --------
    ### Flight booking pathway

    ```python
    import talk_box as tb

    flight_pathway = (
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
            .next_state("destination_check")
        .then("destination_check")
            .decision_state("destination_check")
            .description("Verify if user has destination in mind")
            .branch_on("User has specific destination", "date_selection")
            .branch_on("User needs destination help", "destination_help")
        .then("destination_help")
            .chat_state("destination_help")
            .description("Help user choose a destination")
            .tools(["search_destinations", "get_travel_recommendations"])
            .merge_to("date_selection")
        .then("date_selection")
            .collect_state("date_selection")
            .description("Get preferred travel dates")
            .required(["departure_date", "return_date"])
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
        .then("booking_confirmation")
            .summary_state("booking_confirmation")
            .description("Confirm booking details and complete reservation")
            .required(["flight_selection", "passenger_details", "payment_info"])
            .completion_actions(["book_flight", "send_confirmation"])
        .build()
    )

    # Use in a ChatBot
    bot = (
        tb.ChatBot()
        .provider_model("openai:gpt-4-turbo")
        .system_prompt(
            tb.PromptBuilder()
            .persona("travel agent", "flight booking specialist")
            .pathways(flight_pathway)
            .final_emphasis("Follow the pathway while adapting to user needs")
            .build()
        )
    )
    ```

    ### Troubleshooting pathway

    ```python
    troubleshooting_pathway = (
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
        .then("advanced_diagnostics")
            .tool_state("advanced_diagnostics")
            .description("Run comprehensive system analysis")
            .tools(["system_scan", "log_analysis", "performance_check"])
            .next_state("solution_implementation")
        .then("solution_implementation")
            .chat_state("solution_implementation")
            .description("Guide user through solution steps")
            .next_state("verification")
        .then("verification")
            .summary_state("verification")
            .description("Confirm issue resolution")
            .success_condition("User confirms problem is solved")
            .fallback("Issue remains", "escalation_options")
        .build()
    )
    ```
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
        self._description: Optional[str] = None
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

        Parameters
        ----------
        desc : str
            Description explaining the purpose and context of the pathway
        """
        self._description = desc
        return self

    def activation_conditions(self, conditions: List[str]) -> "Pathways":
        """
        Define when this pathway should be activated.

        Parameters
        ----------
        conditions : List[str]
            Contextual queries that determine when pathway should be active
        """
        self._activation_conditions = conditions
        return self

    def start_with(self, state_name: str) -> "Pathways":
        """
        Define the initial state of the pathway.

        Parameters
        ----------
        state_name : str
            Name of the starting state
        """
        self._start_state = state_name
        self._current_state_name = state_name
        return self

    def then(self, state_name: str) -> "Pathways":
        """
        Move to defining the next state.

        Parameters
        ----------
        state_name : str
            Name of the next state to define
        """
        self._current_state_name = state_name
        return self

    def chat_state(self, name: Optional[str] = None) -> "Pathways":
        """
        Define a conversational state.

        Parameters
        ----------
        name : str, optional
            State name. If not provided, uses current state name.
        """
        state_name = name or self._current_state_name
        self._states[state_name] = PathwayState(
            name=state_name, state_type=StateType.CHAT, description=""
        )
        return self

    def tool_state(self, name: Optional[str] = None) -> "Pathways":
        """
        Define a tool execution state.

        Parameters
        ----------
        name : str, optional
            State name. If not provided, uses current state name.
        """
        state_name = name or self._current_state_name
        self._states[state_name] = PathwayState(
            name=state_name, state_type=StateType.TOOL, description=""
        )
        return self

    def decision_state(self, name: Optional[str] = None) -> "Pathways":
        """
        Define a decision/branching state.

        Parameters
        ----------
        name : str, optional
            State name. If not provided, uses current state name.
        """
        state_name = name or self._current_state_name
        self._states[state_name] = PathwayState(
            name=state_name, state_type=StateType.DECISION, description=""
        )
        return self

    def collect_state(self, name: Optional[str] = None) -> "Pathways":
        """
        Define an information collection state.

        Parameters
        ----------
        name : str, optional
            State name. If not provided, uses current state name.
        """
        state_name = name or self._current_state_name
        self._states[state_name] = PathwayState(
            name=state_name, state_type=StateType.COLLECT, description=""
        )
        return self

    def summary_state(self, name: Optional[str] = None) -> "Pathways":
        """
        Define a summary/completion state.

        Parameters
        ----------
        name : str, optional
            State name. If not provided, uses current state name.
        """
        state_name = name or self._current_state_name
        self._states[state_name] = PathwayState(
            name=state_name, state_type=StateType.SUMMARY, description=""
        )
        return self

    def description(self, desc: str) -> "Pathways":
        """
        Add description to current state or pathway.

        Parameters
        ----------
        desc : str
            Description text
        """
        if self._current_state_name and self._current_state_name in self._states:
            self._states[self._current_state_name].description = desc
        else:
            self._description = desc
        return self

    def collect(self, info_types: List[str]) -> "Pathways":
        """
        Specify information to collect in current state.

        Parameters
        ----------
        info_types : List[str]
            Types of information to collect
        """
        if self._current_state_name in self._states:
            self._states[self._current_state_name].required_info.extend(info_types)
        return self

    def required(self, info_types: List[str]) -> "Pathways":
        """
        Specify required information for current state.

        Parameters
        ----------
        info_types : List[str]
            Required information types
        """
        if self._current_state_name in self._states:
            self._states[self._current_state_name].required_info.extend(info_types)
        return self

    def optional(self, info_types: List[str]) -> "Pathways":
        """
        Specify optional information for current state.

        Parameters
        ----------
        info_types : List[str]
            Optional information types
        """
        if self._current_state_name in self._states:
            self._states[self._current_state_name].optional_info.extend(info_types)
        return self

    def tools(self, tool_names: List[str]) -> "Pathways":
        """
        Specify tools available in current state.

        Parameters
        ----------
        tool_names : List[str]
            Names of tools that can be used
        """
        if self._current_state_name in self._states:
            self._states[self._current_state_name].tools.extend(tool_names)
        return self

    def success_condition(self, condition: str) -> "Pathways":
        """
        Define success condition for current state.

        Parameters
        ----------
        condition : str
            Condition that indicates successful completion
        """
        if self._current_state_name in self._states:
            self._states[self._current_state_name].success_conditions.append(condition)
        return self

    def next_state(self, state_name: str) -> "Pathways":
        """
        Define the next state for direct transition.

        Parameters
        ----------
        state_name : str
            Name of the next state
        """
        if self._current_state_name:
            self._transitions.append(
                PathwayTransition(from_state=self._current_state_name, to_state=state_name)
            )
        return self

    def branch_on(self, condition: str, state_name: str) -> "Pathways":
        """
        Define conditional branch to another state.

        Parameters
        ----------
        condition : str
            Condition for taking this branch
        state_name : str
            Target state name
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
        Merge current state path to another state.

        Parameters
        ----------
        state_name : str
            State to merge into
        """
        return self.next_state(state_name)

    def fallback(self, condition: str, state_name: str) -> "Pathways":
        """
        Define fallback transition.

        Parameters
        ----------
        condition : str
            Condition triggering fallback
        state_name : str
            Fallback target state
        """
        if self._current_state_name in self._states:
            self._states[self._current_state_name].fallback_actions.append(
                f"If {condition}, go to {state_name}"
            )
        return self

    def completion_actions(self, actions: List[str]) -> "Pathways":
        """
        Define actions to take on pathway completion.

        Parameters
        ----------
        actions : List[str]
            Actions to perform when pathway completes
        """
        if self._current_state_name in self._states:
            self._states[self._current_state_name].fallback_actions.extend(actions)
        return self

    def completion_criteria(self, criteria: List[str]) -> "Pathways":
        """
        Define criteria for pathway completion.

        Parameters
        ----------
        criteria : List[str]
            Conditions indicating pathway completion
        """
        self._completion_criteria.extend(criteria)
        return self

    def fallback_strategy(self, strategy: str) -> "Pathways":
        """
        Define overall fallback strategy.

        Parameters
        ----------
        strategy : str
            Strategy to use when pathway cannot progress normally
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

    def preview(self) -> Dict[str, Any]:
        """
        Build and return the complete pathway specification for preview purposes.

        This method is intended for development, debugging, and standalone pathway creation.
        When using with ChatBot or PromptBuilder, prefer passing the Pathways object directly
        rather than calling .preview() first, as this preserves the structured data for testing
        and analysis.

        Returns
        -------
        Dict[str, Any]
            The complete pathway specification.

        Note
        ----
        If you're using this with ChatBot, consider passing the Pathways object
        directly instead:

        # Preferred - preserves structure for testing
        bot.system_prompt(PromptBuilder().pathways(pathway))

        # Works but loses structured data benefits
        bot.system_prompt(PromptBuilder().pathways(pathway.preview()))
        """
        return self._build()

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
        lines.append(f"## CONVERSATIONAL PATHWAY: {spec['title']}")
        if spec.get("description"):
            lines.append(f"\n**Purpose**: {spec['description']}")

        # Activation conditions
        if spec.get("activation_conditions"):
            lines.append("\n**Activate When**:")
            for condition in spec["activation_conditions"]:
                lines.append(f"- {condition}")

        # Flow overview
        lines.append("\n**Conversation Flow**:")
        if spec.get("start_state"):
            lines.append(f"- START: {spec['start_state']}")

        # State definitions
        for state_name, state in spec.get("states", {}).items():
            lines.append(f"\n**{state_name.upper()}** ({state['type']})")
            if state.get("description"):
                lines.append(f"  Purpose: {state['description']}")

            if state.get("required_info"):
                lines.append(f"  Required: {', '.join(state['required_info'])}")

            if state.get("optional_info"):
                lines.append(f"  Optional: {', '.join(state['optional_info'])}")

            if state.get("tools"):
                lines.append(f"  Tools: {', '.join(state['tools'])}")

            if state.get("success_conditions"):
                lines.append(f"  Success: {'; '.join(state['success_conditions'])}")

        # Transitions
        transitions_by_from = {}
        for transition in spec.get("transitions", []):
            from_state = transition["from"]
            if from_state not in transitions_by_from:
                transitions_by_from[from_state] = []
            transitions_by_from[from_state].append(transition)

        if transitions_by_from:
            lines.append("\n**Flow Control**:")
            for from_state, transitions in transitions_by_from.items():
                for transition in transitions:
                    if transition.get("condition"):
                        lines.append(
                            f"- {from_state} → {transition['to']} (if {transition['condition']})"
                        )
                    else:
                        lines.append(f"- {from_state} → {transition['to']}")

        # Completion and fallback
        if spec.get("completion_criteria"):
            lines.append(f"\n**Completion**: {'; '.join(spec['completion_criteria'])}")

        if spec.get("fallback_strategy"):
            lines.append(f"\n**Fallback Strategy**: {spec['fallback_strategy']}")

        lines.append(
            "\n**Adaptive Guidance**: Follow this pathway as a flexible guide, not rigid rules. Adapt to user conversation patterns while ensuring key information is gathered and important steps are addressed."
        )

        return "\n".join(lines)
