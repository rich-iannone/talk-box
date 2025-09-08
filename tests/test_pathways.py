import pytest
import talk_box as tb
from talk_box.pathways import StateType


# Basic instantiation tests
def test_state_method_invalid_type():
    """Test .state() method with invalid type raises error."""
    with pytest.raises(ValueError, match="Invalid state type"):
        tb.Pathways("Test").state("Test state", id="test", type="invalid")


def test_pathways_creation():
    """Test basic Pathways object creation."""
    pathway = tb.Pathways("Test Pathway")
    assert pathway.title == "Test Pathway"
    assert pathway._description == ""
    assert len(pathway._states) == 0
    assert len(pathway._transitions) == 0
    assert pathway._start_state is None
    assert pathway._current_state_name is None


def test_pathways_with_description():
    """Test Pathways creation with description."""
    pathway = tb.Pathways("Test", desc="A test pathway")
    assert pathway.title == "Test"
    assert pathway._description == "A test pathway"


def test_pathways_with_activation_conditions():
    """Test Pathways with activation conditions."""
    pathway = tb.Pathways("Test", activation=["condition1", "condition2"])
    assert pathway._activation_conditions == ["condition1", "condition2"]


def test_pathways_with_single_string_activation():
    """Test Pathways with single string activation (promoted to list)."""
    pathway = tb.Pathways("Test", activation="single condition")
    assert pathway._activation_conditions == ["single condition"]


def test_pathways_with_none_activation():
    """Test Pathways with no activation conditions."""
    pathway = tb.Pathways("Test", activation=None)
    assert pathway._activation_conditions == []


def test_pathways_prompt_generation():
    """Test pathway generates a prompt."""
    pathway = tb.Pathways("Test", desc="A test pathway").state("A test state", id="test_state")
    prompt = pathway._to_prompt_text()
    assert "Test" in prompt
    assert "A test pathway" in prompt
    assert "TEST_STATE" in prompt  # State names are uppercase in the prompt


# New unified .state() method tests
def test_state_method_basic():
    """Test the new unified .state() method with default type."""
    pathway = tb.Pathways("Test").state("A test state", id="test_state")

    assert "test_state" in pathway._states
    state = pathway._states["test_state"]
    assert state.state_type == StateType.CHAT  # Default type
    assert state.description == "A test state"
    assert pathway._start_state == "test_state"  # First state becomes start state


def test_state_method_with_type():
    """Test .state() method with explicit type."""
    pathway = tb.Pathways("Test").state("A collect state", id="collect_test", type="collect")

    assert "collect_test" in pathway._states
    state = pathway._states["collect_test"]
    assert state.state_type == StateType.COLLECT
    assert state.description == "A collect state"
    assert pathway._start_state == "collect_test"


def test_state_method_all_types():
    """Test .state() method with all state types."""
    pathway = (
        tb.Pathways("Test")
        .state("Chat state", id="chat_state", type="chat")
        .next_state("collect_state")
        .state("Collect state", id="collect_state", type="collect")
        .next_state("decision_state")
        .state("Decision state", id="decision_state", type="decision")
        .branch_on("Option A", id="tool_state")
        .state("Tool state", id="tool_state", type="tool")
        .next_state("summary_state")
        .state("Summary state", id="summary_state", type="summary")
    )

    assert len(pathway._states) == 5
    assert pathway._states["chat_state"].state_type == StateType.CHAT
    assert pathway._states["collect_state"].state_type == StateType.COLLECT
    assert pathway._states["decision_state"].state_type == StateType.DECISION
    assert pathway._states["tool_state"].state_type == StateType.TOOL
    assert pathway._states["summary_state"].state_type == StateType.SUMMARY
    assert pathway._start_state == "chat_state"  # First one becomes start


def test_state_method_invalid_type():
    """Test .state() method with invalid type raises error."""
    with pytest.raises(ValueError, match="Invalid state type 'invalid'"):
        tb.Pathways("Test").state("test state", id="test", type="invalid")


def test_state_method_auto_start_state():
    """Test that first .state() call automatically sets start state."""
    pathway = (
        tb.Pathways("Test")
        .state("First state", id="first")
        .next_state("second")
        .state("Second state", id="second")
    )

    assert pathway._start_state == "first"
    assert pathway._current_state_name == "second"  # Should be the last one defined
    assert len(pathway._states) == 2


def test_state_method_chaining():
    """Test .state() method with full configuration chaining."""
    pathway = (
        tb.Pathways("Test")
        .state("Complex state with all features", id="complex_state", type="collect")
        .required(["name", "email"])
        .optional(["phone"])
        .success_condition("User provided information")
        .next_state("next_state")
    )

    state = pathway._states["complex_state"]
    assert state.state_type == StateType.COLLECT
    assert state.description == "Complex state with all features"
    assert state.required_info == ["name", "email"]
    assert state.optional_info == ["phone"]
    assert "User provided information" in state.success_conditions


def test_unified_api_example():
    """Test a complete pathway using the new unified API."""
    pathway = (
        tb.Pathways(
            title="Unified API Test",
            desc="Testing the simplified API",
            activation=["User needs unified help"],
        )
        # === STATE: greeting ===
        .state("Welcome the user", id="greeting")  # defaults to chat
        .required(["user_welcomed"])
        .next_state("assessment")
        # === STATE: assessment ===
        .state("Gather user information", id="assessment", type="collect")
        .required(["user_name", "user_goal"])
        .optional(["user_background"])
        .next_state("routing")
        # === STATE: routing ===
        .state("Route to appropriate assistance", id="routing", type="decision")
        .branch_on("Technical help needed", id="tech_support")
        .branch_on("General information", id="info_sharing")
        # === STATE: tech_support ===
        .state("Provide technical assistance", id="tech_support", type="tool")
        .tools(["diagnostics", "troubleshooting"])
        .next_state("completion")
        # === STATE: info_sharing ===
        .state("Share relevant information", id="info_sharing")  # defaults to chat
        .required(["information_provided"])
        .next_state("completion")
        # === STATE: completion ===
        .state("Wrap up the interaction", id="completion", type="summary")
        .required(["satisfaction_confirmed"])
        .success_condition("User's needs fully addressed")
    )

    # Verify structure
    assert len(pathway._states) == 6
    assert pathway._start_state == "greeting"
    assert pathway.title == "Unified API Test"
    assert pathway._description == "Testing the simplified API"

    # Verify state types
    assert (
        pathway._states["greeting"].state_type == StateType.COLLECT
    )  # Now inferred from .required()
    assert pathway._states["assessment"].state_type == StateType.COLLECT
    assert pathway._states["routing"].state_type == StateType.DECISION
    assert pathway._states["tech_support"].state_type == StateType.TOOL
    assert (
        pathway._states["info_sharing"].state_type == StateType.COLLECT
    )  # Now inferred from .required()
    assert pathway._states["completion"].state_type == StateType.SUMMARY


def test_state_configuration_methods():
    """Test state configuration methods work with unified API."""
    pathway = (
        tb.Pathways("Test")
        .state("Test state", id="test_state")
        .required(["req1", "req2"])
        .optional(["opt1", "opt2"])
        .success_condition("First condition")
        .success_condition("Second condition")
    )

    state = pathway._states["test_state"]
    assert state.description == "Test state"
    assert state.required_info == ["req1", "req2"]
    assert state.optional_info == ["opt1", "opt2"]
    assert len(state.success_conditions) == 2
    assert "First condition" in state.success_conditions
    assert "Second condition" in state.success_conditions


def test_tool_state_configuration():
    """Test tool state configuration with unified API."""
    pathway = (
        tb.Pathways("Test")
        .state("A tool state", id="tool_test", type="tool")
        .tools(["tool1", "tool2"])
        .success_condition("Tools used successfully")
    )

    state = pathway._states["tool_test"]
    assert state.state_type == StateType.TOOL
    assert state.tools == ["tool1", "tool2"]
    assert "Tools used successfully" in state.success_conditions


def test_decision_state_branching():
    """Test decision state branching with unified API."""
    pathway = (
        tb.Pathways("Test")
        .state("decision_state state", id="decision_state", type="decision")
        .branch_on("Condition A", id="state_a")
        .branch_on("Condition B", id="state_b")
    )

    # Check transitions were created
    assert len(pathway._transitions) == 2
    transition_targets = [t.to_state for t in pathway._transitions]
    assert "state_a" in transition_targets
    assert "state_b" in transition_targets

    # Check conditions
    conditions = [t.condition for t in pathway._transitions]
    assert "Condition A" in conditions
    assert "Condition B" in conditions


def test_linear_progression():
    """Test linear state progression with unified API."""
    pathway = (
        tb.Pathways("Test")
        .state("first state", id="first")
        .next_state("second")
        .state("second state", id="second")
        .next_state("third")
        .state("third state", id="third", type="summary")
    )

    # Check all states exist
    assert len(pathway._states) == 3
    assert pathway._start_state == "first"

    # Check transitions
    assert len(pathway._transitions) == 2
    transitions = {t.from_state: t.to_state for t in pathway._transitions}
    assert transitions["first"] == "second"
    assert transitions["second"] == "third"


def test_complex_branching_pathway():
    """Test complex pathway with branching and merging using unified API."""
    pathway = (
        tb.Pathways(
            title="Complex Test",
            desc="Complex branching pathway",
            activation=["User needs complex help"],
        )
        # === STATE: intake ===
        .state("intake state", id="intake", type="collect")
        .required(["user_info", "problem_type"])
        .next_state("triage")
        # === STATE: triage ===
        .state("triage state", id="triage", type="decision")
        .branch_on("Simple problem", id="simple_resolution")
        .branch_on("Complex problem", id="detailed_analysis")
        .branch_on("Urgent issue", id="escalation")
        # === STATE: simple_resolution ===
        .state("simple_resolution state", id="simple_resolution")
        .required(["quick_solution_provided"])
        .next_state("completion")
        # === STATE: detailed_analysis ===
        .state("detailed_analysis state", id="detailed_analysis", type="tool")
        .tools(["analysis_tools", "diagnostic_suite"])
        .success_condition("Root cause identified")
        .next_state("complex_resolution")
        # === STATE: complex_resolution ===
        .state("complex_resolution state", id="complex_resolution")
        .required(["comprehensive_solution", "implementation_plan"])
        .next_state("completion")
        # === STATE: escalation ===
        .state("escalation state", id="escalation", type="tool")
        .tools(["escalation_system", "priority_queue"])
        .success_condition("Issue escalated successfully")
        .next_state("completion")
        # === STATE: completion ===
        .state("completion state", id="completion", type="summary")
        .required(["issue_resolved", "customer_satisfied"])
        .success_condition("Customer issue fully addressed")
    )

    # Verify pathway structure
    assert len(pathway._states) == 7
    assert pathway._start_state == "intake"
    assert pathway.title == "Complex Test"

    # Verify state types
    expected_types = {
        "intake": StateType.COLLECT,
        "triage": StateType.DECISION,
        "simple_resolution": StateType.COLLECT,  # Now inferred from .required()
        "detailed_analysis": StateType.TOOL,
        "complex_resolution": StateType.COLLECT,  # Now inferred from .required()
        "escalation": StateType.TOOL,
        "completion": StateType.SUMMARY,
    }

    for state_name, expected_type in expected_types.items():
        assert pathway._states[state_name].state_type == expected_type

    # Verify branching from triage
    triage_transitions = [t for t in pathway._transitions if t.from_state == "triage"]
    assert len(triage_transitions) == 3

    branch_targets = [t.to_state for t in triage_transitions]
    assert "simple_resolution" in branch_targets
    assert "detailed_analysis" in branch_targets
    assert "escalation" in branch_targets

    # Verify all paths lead to completion
    completion_transitions = [t for t in pathway._transitions if t.to_state == "completion"]
    assert len(completion_transitions) == 3  # simple_resolution, complex_resolution, escalation


def test_pathway_prompt_generation():
    """Test that pathways generate proper prompts with unified API."""
    pathway = (
        tb.Pathways(
            title="Support Flow",
            desc="Customer support pathway",
            activation=["Customer needs help"],
        )
        .state("Greet the customer warmly", id="greeting")
        .required(["customer_welcomed"])
        .next_state("assessment")
        .state("Assess customer needs thoroughly", id="assessment", type="collect")
        .required(["issue_type", "urgency_level"])
        .optional(["customer_history"])
        .success_condition("Customer needs clearly understood")
    )

    prompt = pathway._to_prompt_text()

    # Basic pathway info should be present
    assert "Support Flow" in prompt
    assert "Customer support pathway" in prompt
    assert "Customer needs help" in prompt

    # State information should be included
    assert "GREETING" in prompt  # State names are uppercase in prompts
    assert "ASSESSMENT" in prompt
    assert "Greet the customer warmly" in prompt
    assert "Assess customer needs thoroughly" in prompt
    assert "customer_welcomed" in prompt
    assert "issue_type" in prompt
    assert "urgency_level" in prompt
