import pytest
import talk_box as tb
from talk_box.pathways import StateType


# Basic instantiation tests
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
    pathway = tb.Pathways("Test").description("A test pathway")
    assert pathway.title == "Test"
    assert pathway._description == "A test pathway"


def test_pathways_with_activation_conditions():
    """Test Pathways with activation conditions."""
    pathway = tb.Pathways("Test").activation_conditions(["condition1", "condition2"])
    assert pathway._activation_conditions == ["condition1", "condition2"]


def test_pathways_prompt_generation():
    """Test pathway generates a prompt."""
    pathway = (
        tb.Pathways("Test")
        .description("A test pathway")
        .state(id="test_state")
        .description("A test state")
    )
    prompt = pathway.generate_prompt()
    assert "Test" in prompt
    assert "A test pathway" in prompt
    assert "TEST_STATE" in prompt  # State names are uppercase in the prompt


# New unified .state() method tests
def test_state_method_basic():
    """Test the new unified .state() method with default type."""
    pathway = tb.Pathways("Test").state(id="test_state").description("A test state")

    assert "test_state" in pathway._states
    state = pathway._states["test_state"]
    assert state.state_type == StateType.CHAT  # Default type
    assert state.description == "A test state"
    assert pathway._start_state == "test_state"  # First state becomes start state


def test_state_method_with_type():
    """Test .state() method with explicit type."""
    pathway = (
        tb.Pathways("Test").state(id="collect_test", type="collect").description("A collect state")
    )

    assert "collect_test" in pathway._states
    state = pathway._states["collect_test"]
    assert state.state_type == StateType.COLLECT
    assert state.description == "A collect state"
    assert pathway._start_state == "collect_test"


def test_state_method_all_types():
    """Test .state() method with all state types."""
    pathway = (
        tb.Pathways("Test")
        .state(id="chat_state", type="chat")
        .description("Chat state")
        .state(id="collect_state", type="collect")
        .description("Collect state")
        .state(id="decision_state", type="decision")
        .description("Decision state")
        .state(id="tool_state", type="tool")
        .description("Tool state")
        .state(id="summary_state", type="summary")
        .description("Summary state")
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
        tb.Pathways("Test").state(id="test", type="invalid")


def test_state_method_auto_start_state():
    """Test that first .state() call automatically sets start state."""
    pathway = (
        tb.Pathways("Test")
        .state(id="first")
        .description("First state")
        .state(id="second")
        .description("Second state")
    )

    assert pathway._start_state == "first"
    assert pathway._current_state_name == "second"  # Should be the last one defined
    assert len(pathway._states) == 2


def test_state_method_chaining():
    """Test .state() method with full configuration chaining."""
    pathway = (
        tb.Pathways("Test")
        .state(id="complex_state", type="collect")
        .description("Complex state with all features")
        .required(["req1", "req2"])
        .optional(["opt1"])
        .success_condition("Success achieved")
        .next_state("next_state")
    )

    state = pathway._states["complex_state"]
    assert state.state_type == StateType.COLLECT
    assert state.description == "Complex state with all features"
    assert state.required_info == ["req1", "req2"]
    assert state.optional_info == ["opt1"]
    assert "Success achieved" in state.success_conditions


def test_unified_api_example():
    """Test a complete pathway using the new unified API."""
    pathway = (
        tb.Pathways("Unified API Test")
        .description("Testing the simplified API")
        .activation_conditions(["User needs unified help"])
        # === STATE: greeting ===
        .state(id="greeting")  # defaults to chat
        .description("Welcome the user")
        .required(["user_welcomed"])
        .next_state("assessment")
        # === STATE: assessment ===
        .state(id="assessment", type="collect")
        .description("Gather user information")
        .required(["user_name", "user_goal"])
        .optional(["user_background"])
        .next_state("routing")
        # === STATE: routing ===
        .state(id="routing", type="decision")
        .description("Route to appropriate assistance")
        .branch_on("Technical help needed", "tech_support")
        .branch_on("General information", "info_sharing")
        # === STATE: tech_support ===
        .state(id="tech_support", type="tool")
        .description("Provide technical assistance")
        .tools(["diagnostics", "troubleshooting"])
        .next_state("completion")
        # === STATE: info_sharing ===
        .state(id="info_sharing")  # defaults to chat
        .description("Share relevant information")
        .required(["information_provided"])
        .next_state("completion")
        # === STATE: completion ===
        .state(id="completion", type="summary")
        .description("Wrap up the interaction")
        .required(["satisfaction_confirmed"])
        .success_condition("User's needs fully addressed")
    )

    # Verify structure
    assert len(pathway._states) == 6
    assert pathway._start_state == "greeting"
    assert pathway.title == "Unified API Test"
    assert pathway._description == "Testing the simplified API"

    # Verify state types
    assert pathway._states["greeting"].state_type == StateType.CHAT
    assert pathway._states["assessment"].state_type == StateType.COLLECT
    assert pathway._states["routing"].state_type == StateType.DECISION
    assert pathway._states["tech_support"].state_type == StateType.TOOL
    assert pathway._states["info_sharing"].state_type == StateType.CHAT
    assert pathway._states["completion"].state_type == StateType.SUMMARY


def test_state_configuration_methods():
    """Test state configuration methods work with unified API."""
    pathway = (
        tb.Pathways("Test")
        .state(id="test_state")
        .description("Test state")
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
        .state(id="tool_test", type="tool")
        .description("A tool state")
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
        .state(id="decision_state", type="decision")
        .description("Decision point")
        .branch_on("Condition A", "state_a")
        .branch_on("Condition B", "state_b")
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
        .state(id="first")
        .description("First state")
        .next_state("second")
        .state(id="second")
        .description("Second state")
        .next_state("third")
        .state(id="third", type="summary")
        .description("Final state")
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
        tb.Pathways("Complex Test")
        .description("Complex branching pathway")
        .activation_conditions(["User needs complex help"])
        # === STATE: intake ===
        .state(id="intake", type="collect")
        .description("Gather initial information")
        .required(["user_info", "problem_type"])
        .next_state("triage")
        # === STATE: triage ===
        .state(id="triage", type="decision")
        .description("Route based on problem complexity")
        .branch_on("Simple problem", "simple_resolution")
        .branch_on("Complex problem", "detailed_analysis")
        .branch_on("Urgent issue", "escalation")
        # === STATE: simple_resolution ===
        .state(id="simple_resolution")
        .description("Handle simple problems quickly")
        .required(["quick_solution_provided"])
        .next_state("completion")
        # === STATE: detailed_analysis ===
        .state(id="detailed_analysis", type="tool")
        .description("Analyze complex problems thoroughly")
        .tools(["analysis_tools", "diagnostic_suite"])
        .success_condition("Root cause identified")
        .next_state("complex_resolution")
        # === STATE: complex_resolution ===
        .state(id="complex_resolution")
        .description("Provide detailed solution")
        .required(["comprehensive_solution", "implementation_plan"])
        .next_state("completion")
        # === STATE: escalation ===
        .state(id="escalation", type="tool")
        .description("Escalate urgent issues")
        .tools(["escalation_system", "priority_queue"])
        .success_condition("Issue escalated successfully")
        .next_state("completion")
        # === STATE: completion ===
        .state(id="completion", type="summary")
        .description("Ensure customer satisfaction")
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
        "simple_resolution": StateType.CHAT,
        "detailed_analysis": StateType.TOOL,
        "complex_resolution": StateType.CHAT,
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
        tb.Pathways("Support Flow")
        .description("Customer support pathway")
        .activation_conditions(["Customer needs help"])
        .state(id="greeting")
        .description("Greet the customer warmly")
        .required(["customer_welcomed"])
        .next_state("assessment")
        .state(id="assessment", type="collect")
        .description("Understand the customer's needs")
        .required(["issue_type", "urgency_level"])
        .optional(["customer_history"])
        .success_condition("Customer needs clearly understood")
    )

    prompt = pathway.generate_prompt()

    # Basic pathway info should be present
    assert "Support Flow" in prompt
    assert "Customer support pathway" in prompt
    assert "Customer needs help" in prompt

    # State information should be included
    assert "GREETING" in prompt  # State names are uppercase in prompts
    assert "ASSESSMENT" in prompt
    assert "Greet the customer warmly" in prompt
    assert "Understand the customer's needs" in prompt
    assert "customer_welcomed" in prompt
    assert "issue_type" in prompt
    assert "urgency_level" in prompt
