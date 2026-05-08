from talk_box.agent import Agent
from talk_box.pathways import Pathways
from talk_box.personas import create_persona


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(name="test_agent", role="helpful assistant"):
    """Create a minimal agent for testing."""
    persona = create_persona(name, persona_role=role)
    return Agent(name=name, persona=persona)


# ---------------------------------------------------------------------------
# Agent assignment via .state(agent=...)
# ---------------------------------------------------------------------------


class TestStateAgentParam:
    def test_agent_in_state_call(self):
        pathway = (
            Pathways(title="Test", desc="test pathway")
            .state("triage: classify the issue", agent="triage_bot")
            .next_state("resolve")
            .state("resolve: resolve the issue", agent="resolver")
        )
        spec = pathway._build()
        assert spec["states"]["triage"]["agent"] == "triage_bot"
        assert spec["states"]["resolve"]["agent"] == "resolver"

    def test_state_without_agent(self):
        pathway = Pathways(title="Test", desc="test pathway").state("step1: do something")
        spec = pathway._build()
        assert "agent" not in spec["states"]["step1"]

    def test_mixed_agent_and_no_agent_states(self):
        pathway = (
            Pathways(title="Test", desc="test pathway")
            .state("triage: classify", agent="triage_bot")
            .next_state("manual")
            .state("manual: manual step")
            .next_state("resolve")
            .state("resolve: fix it", agent="tech_bot")
        )
        spec = pathway._build()
        assert spec["states"]["triage"]["agent"] == "triage_bot"
        assert "agent" not in spec["states"]["manual"]
        assert spec["states"]["resolve"]["agent"] == "tech_bot"


# ---------------------------------------------------------------------------
# Agent assignment via .agent() method
# ---------------------------------------------------------------------------


class TestAgentMethod:
    def test_agent_method_sets_agent(self):
        pathway = (
            Pathways(title="Test", desc="test pathway")
            .state("triage: classify the issue")
            .agent("triage_bot")
        )
        spec = pathway._build()
        assert spec["states"]["triage"]["agent"] == "triage_bot"

    def test_agent_method_chainable(self):
        pathway = (
            Pathways(title="Test", desc="test pathway")
            .state("triage: classify the issue")
            .agent("triage_bot")
            .next_state("resolve")
            .state("resolve: fix it")
            .agent("resolver")
            .success_condition("Issue resolved")
        )
        spec = pathway._build()
        assert spec["states"]["triage"]["agent"] == "triage_bot"
        assert spec["states"]["resolve"]["agent"] == "resolver"

    def test_agent_method_overrides_state_param(self):
        pathway = (
            Pathways(title="Test", desc="test pathway")
            .state("triage: classify", agent="original")
            .agent("override")
        )
        spec = pathway._build()
        assert spec["states"]["triage"]["agent"] == "override"


# ---------------------------------------------------------------------------
# Agent registry
# ---------------------------------------------------------------------------


class TestAgentRegistry:
    def test_register_and_get_agent(self):
        triage = _make_agent("triage", "triage specialist")
        pathway = (
            Pathways(title="Test", desc="test pathway")
            .register_agent("triage", triage)
            .state("triage: classify issue", agent="triage")
        )
        retrieved = pathway.get_agent("triage")
        assert retrieved is triage

    def test_get_agent_returns_none_for_unregistered(self):
        pathway = Pathways(title="Test", desc="test pathway").state(
            "triage: classify issue", agent="missing_bot"
        )
        assert pathway.get_agent("triage") is None

    def test_get_agent_returns_none_for_no_agent_state(self):
        pathway = Pathways(title="Test", desc="test pathway").state("step: do something")
        assert pathway.get_agent("step") is None

    def test_get_agent_returns_none_for_unknown_state(self):
        pathway = Pathways(title="Test", desc="test pathway")
        assert pathway.get_agent("nonexistent") is None

    def test_register_multiple_agents(self):
        triage = _make_agent("triage", "triage specialist")
        tech = _make_agent("tech", "tech expert")
        pathway = (
            Pathways(title="Test", desc="test pathway")
            .register_agent("triage", triage)
            .register_agent("tech", tech)
            .state("triage: classify issue", agent="triage")
            .branch_on("technical", id="tech_support")
            .state("tech_support: fix it", agent="tech")
        )
        assert pathway.get_agent("triage") is triage
        assert pathway.get_agent("tech_support") is tech

    def test_register_agent_chainable(self):
        agent = _make_agent()
        result = Pathways(title="Test", desc="test pathway").register_agent("bot", agent)
        assert isinstance(result, Pathways)


# ---------------------------------------------------------------------------
# Build output
# ---------------------------------------------------------------------------


class TestBuildOutput:
    def test_build_includes_agents_list(self):
        triage = _make_agent("triage", "triage specialist")
        pathway = (
            Pathways(title="Test", desc="test pathway")
            .register_agent("triage", triage)
            .state("triage: classify", agent="triage")
        )
        spec = pathway._build()
        assert "agents" in spec
        assert "triage" in spec["agents"]

    def test_build_empty_agents_list(self):
        pathway = Pathways(title="Test", desc="test pathway").state("step: do something")
        spec = pathway._build()
        assert spec["agents"] == []


# ---------------------------------------------------------------------------
# Prompt text includes agent info
# ---------------------------------------------------------------------------


class TestPromptText:
    def test_agent_appears_in_prompt_text(self):
        pathway = (
            Pathways(title="Test", desc="test pathway")
            .state("triage: classify the issue", agent="triage_bot")
            .next_state("resolve")
            .state("resolve: fix it", agent="tech_expert")
        )
        text = pathway._to_prompt_text()
        assert "Agent: triage_bot" in text
        assert "Agent: tech_expert" in text

    def test_no_agent_line_when_not_set(self):
        pathway = Pathways(title="Test", desc="test pathway").state("step: do something")
        text = pathway._to_prompt_text()
        assert "Agent:" not in text


# ---------------------------------------------------------------------------
# Multi-agent pathway end-to-end
# ---------------------------------------------------------------------------


class TestMultiAgentEndToEnd:
    def test_full_multi_agent_pathway(self):
        """End-to-end test matching the PLAN.md example pattern."""
        triage = _make_agent("triage_bot", "customer support triage specialist")
        tech = _make_agent("tech_expert", "technical support engineer")
        billing = _make_agent("billing_expert", "billing support specialist")

        pathway = (
            Pathways(
                title="Customer Support",
                desc="Multi-agent customer support with escalation",
                activation="Customer needs help",
            )
            .register_agent("triage_bot", triage)
            .register_agent("tech_expert", tech)
            .register_agent("billing_expert", billing)
            .state("triage: classify the issue", agent="triage_bot")
            .branch_on("technical problem", id="tech_support")
            .branch_on("billing question", id="billing_support")
            .state("tech_support: resolve technical issues", agent="tech_expert")
            .success_condition("Issue resolved")
            .next_state("completion")
            .state("billing_support: address billing questions", agent="billing_expert")
            .required(["account_id", "issue_type"])
            .next_state("completion")
            .state("completion: wrap up", type="summary")
            .success_condition("Customer satisfied")
        )

        spec = pathway._build()

        # Verify structure
        assert len(spec["states"]) == 4
        assert spec["states"]["triage"]["agent"] == "triage_bot"
        assert spec["states"]["tech_support"]["agent"] == "tech_expert"
        assert spec["states"]["billing_support"]["agent"] == "billing_expert"
        assert "agent" not in spec["states"]["completion"]

        # Verify agent retrieval
        assert pathway.get_agent("triage") is triage
        assert pathway.get_agent("tech_support") is tech
        assert pathway.get_agent("billing_support") is billing
        assert pathway.get_agent("completion") is None

        # Verify agents in build output
        assert set(spec["agents"]) == {"triage_bot", "tech_expert", "billing_expert"}

        # Verify prompt text
        text = pathway._to_prompt_text()
        assert "Agent: triage_bot" in text
        assert "Agent: tech_expert" in text
        assert "Agent: billing_expert" in text
