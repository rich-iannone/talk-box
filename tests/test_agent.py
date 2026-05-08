from talk_box.agent import Agent
from talk_box.capture import EventType
from talk_box.memory import MemoryStore, MemoryTier
from talk_box.personas import create_persona
from talk_box.retention import RetentionPolicy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_persona(**kwargs):
    """Create a minimal persona for testing."""
    defaults = {
        "name": "test_agent",
        "persona_role": "helpful test assistant",
        "category": "custom",
        "description": "A test persona.",
    }
    defaults.update(kwargs)
    return create_persona(**defaults)


def _make_agent(persona=None, **kwargs):
    """Create an agent with a mock-ready chatbot."""
    if persona is None:
        persona = _make_persona()
    agent = Agent(
        name=kwargs.pop("name", persona.name),
        persona=persona,
        memory=kwargs.pop("memory", MemoryStore(long_term_path=":memory:")),
        **kwargs,
    )
    return agent


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestAgentConstruction:
    def test_basic_creation(self):
        agent = _make_agent()
        assert agent.name == "test_agent"
        assert agent.persona.persona_role == "helpful test assistant"
        assert agent.conversation is None

    def test_chatbot_is_configured_from_persona(self):
        persona = _make_persona(temperature=0.2, max_tokens=500)
        agent = _make_agent(persona=persona)
        assert agent.chatbot._config["temperature"] == 0.2
        assert agent.chatbot._config["max_tokens"] == 500

    def test_chatbot_gets_system_prompt(self):
        agent = _make_agent()
        # The system prompt should be set (PromptBuilder object)
        assert agent.chatbot._config.get("system_prompt_builder") is not None

    def test_custom_instructions(self):
        agent = _make_agent(instructions="Always respond in French.")
        # The system prompt builder should contain the instruction
        builder = agent.chatbot._config.get("system_prompt_builder")
        assert builder is not None
        prompt_text = str(builder)
        assert "Always respond in French" in prompt_text

    def test_metadata(self):
        agent = _make_agent(metadata={"team": "support", "version": "1.0"})
        assert agent.metadata["team"] == "support"
        assert agent.metadata["version"] == "1.0"

    def test_avoid_topics_applied(self):
        persona = _make_persona(avoid_topics=["politics", "religion"])
        agent = _make_agent(persona=persona)
        avoid = agent.chatbot._config.get("avoid_topics", [])
        assert "politics" in avoid
        assert "religion" in avoid

    def test_tools_applied(self):
        persona = _make_persona(tools=["text_stats"])
        agent = _make_agent(persona=persona)
        tools = agent.chatbot._config.get("tools", [])
        assert "text_stats" in tools


# ---------------------------------------------------------------------------
# from_persona factory
# ---------------------------------------------------------------------------


class TestFromPersona:
    def test_from_builtin_persona(self):
        agent = Agent.from_persona("code_reviewer")
        assert agent.name == "code_reviewer"
        assert agent.persona.display_name == "Code Reviewer"
        assert agent.persona.category == "technical"

    def test_from_persona_custom_name(self):
        agent = Agent.from_persona("code_reviewer", name="my_reviewer")
        assert agent.name == "my_reviewer"
        assert agent.persona.name == "code_reviewer"

    def test_from_persona_with_memory(self):
        mem = MemoryStore(long_term_path=":memory:")
        agent = Agent.from_persona("code_reviewer", memory=mem)
        assert agent.memory is mem

    def test_from_persona_with_instructions(self):
        agent = Agent.from_persona(
            "code_reviewer",
            instructions="Focus on security issues only.",
        )
        prompt_text = str(agent.chatbot._config.get("system_prompt_builder", ""))
        assert "Focus on security issues only" in prompt_text


# ---------------------------------------------------------------------------
# respond
# ---------------------------------------------------------------------------


class TestRespond:
    def test_respond_returns_conversation(self):
        agent = _make_agent()
        agent.chatbot.mock_responses(["Hello! I can help with that."])
        convo = agent.respond("Hi there")
        assert convo is not None
        assert convo.get_message_count() >= 2  # user + assistant

    def test_respond_continues_conversation(self):
        agent = _make_agent()
        agent.chatbot.mock_responses(["First reply.", "Second reply."])
        convo1 = agent.respond("First message")
        count_after_first = convo1.get_message_count()
        convo2 = agent.respond("Second message")
        assert convo2.get_message_count() > count_after_first

    def test_respond_with_explicit_conversation(self):
        agent = _make_agent()
        agent.chatbot.mock_responses(["Reply 1.", "Reply 2."])
        convo = agent.respond("Hello")
        count_after_first = convo.get_message_count()
        # Pass conversation explicitly
        convo2 = agent.respond("Follow up", conversation=convo)
        assert convo2.get_message_count() > count_after_first

    def test_respond_records_capture_events(self):
        agent = _make_agent()
        agent.chatbot.mock_responses(["Test response."])
        agent.respond("Test prompt")

        prompts = agent.capture.prompts()
        responses = agent.capture.responses()
        assert len(prompts) == 1
        assert prompts[0].content == "Test prompt"
        assert len(responses) == 1
        assert responses[0].content == "Test response."

    def test_respond_applies_retention(self):
        policy = RetentionPolicy(
            remember_tags=["identity"],
            forget_tags=["scratch"],
        )
        persona = _make_persona(retention=policy)
        agent = _make_agent(persona=persona)

        # Add some memories
        agent.remember("user_name", "Alice", tags=("identity",))
        agent.remember("temp", "junk", tags=("scratch",))

        agent.chatbot.mock_responses(["Got it."])
        agent.respond("Hello")

        # Retention should have removed scratch-tagged entry
        assert agent.recall("user_name") == "Alice"
        assert agent.recall("temp") is None

    def test_respond_no_retention_when_not_set(self):
        agent = _make_agent()
        agent.remember("temp", "data", tags=("scratch",))
        agent.chatbot.mock_responses(["OK."])
        agent.respond("Hi")
        # Without retention policy, nothing is removed
        assert agent.recall("temp") == "data"


# ---------------------------------------------------------------------------
# Memory convenience methods
# ---------------------------------------------------------------------------


class TestMemory:
    def test_remember_and_recall(self):
        agent = _make_agent()
        agent.remember("key1", "value1")
        assert agent.recall("key1") == "value1"

    def test_recall_default(self):
        agent = _make_agent()
        assert agent.recall("missing") is None
        assert agent.recall("missing", "fallback") == "fallback"

    def test_remember_with_tier(self):
        agent = _make_agent()
        agent.remember("st_key", "st_val", tier=MemoryTier.SHORT_TERM)
        assert agent.recall("st_key") == "st_val"

    def test_remember_with_tags(self):
        agent = _make_agent()
        agent.remember("tagged", "val", tags=("important",))
        entries = agent.memory.search(tags=["important"])
        assert len(entries) == 1
        assert entries[0].key == "tagged"


# ---------------------------------------------------------------------------
# Properties and reset
# ---------------------------------------------------------------------------


class TestProperties:
    def test_retention_property(self):
        policy = RetentionPolicy(remember_tags=["identity"])
        persona = _make_persona(retention=policy)
        agent = _make_agent(persona=persona)
        assert agent.retention is policy

    def test_retention_none_when_not_set(self):
        agent = _make_agent()
        assert agent.retention is None

    def test_conversation_starts_none(self):
        agent = _make_agent()
        assert agent.conversation is None

    def test_conversation_set_after_respond(self):
        agent = _make_agent()
        agent.chatbot.mock_responses(["Hi!"])
        agent.respond("Hello")
        assert agent.conversation is not None

    def test_reset_conversation(self):
        agent = _make_agent()
        agent.chatbot.mock_responses(["Hi!"])
        agent.respond("Hello")
        assert agent.conversation is not None
        agent.reset_conversation()
        assert agent.conversation is None


# ---------------------------------------------------------------------------
# Top-level imports
# ---------------------------------------------------------------------------


class TestTopLevelImport:
    def test_import_agent(self):
        import talk_box as tb

        assert hasattr(tb, "Agent")

    def test_in_all(self):
        import talk_box

        assert "Agent" in talk_box.__all__
