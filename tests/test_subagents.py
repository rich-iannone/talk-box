from talk_box.agent import Agent
from talk_box.personas import create_persona
from talk_box.shared_state import SharedState
from talk_box.subagents import (
    SubagentResult,
    children,
    delegate,
    parent_name,
    spawn,
)


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
    """Create an agent with defaults."""
    if persona is None:
        persona = _make_persona()
    return Agent(
        name=kwargs.pop("name", persona.name),
        persona=persona,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# SubagentResult
# ---------------------------------------------------------------------------


class TestSubagentResult:
    def test_frozen(self):
        from talk_box.conversation import Conversation

        result = SubagentResult(
            response="ok",
            agent="child",
            parent="parent",
            conversation=Conversation(),
            duration=0.5,
        )
        assert result.response == "ok"
        assert result.agent == "child"
        assert result.parent == "parent"
        assert result.duration == 0.5

    def test_fields(self):
        from talk_box.conversation import Conversation

        result = SubagentResult(
            response="done",
            agent="a",
            parent="p",
            conversation=Conversation(),
            duration=1.23,
        )
        assert result.response == "done"
        assert result.agent == "a"
        assert result.parent == "p"
        assert result.duration == 1.23


# ---------------------------------------------------------------------------
# spawn
# ---------------------------------------------------------------------------


class TestSpawn:
    def test_spawn_inherits_persona(self):
        parent = _make_agent()
        child = spawn(parent, "helper")
        assert child.name == "helper"
        assert child.persona is parent.persona

    def test_spawn_with_persona_definition(self):
        parent = _make_agent()
        other_persona = _make_persona(
            name="other", persona_role="data analyst", description="Analyzes data."
        )
        child = spawn(parent, "analyst", persona=other_persona)
        assert child.name == "analyst"
        assert child.persona.persona_role == "data analyst"

    def test_spawn_with_persona_name(self):
        parent = _make_agent()
        child = spawn(parent, "reviewer", persona="code_reviewer")
        assert child.name == "reviewer"
        assert child.persona.name == "code_reviewer"

    def test_spawn_with_instructions(self):
        parent = _make_agent()
        child = spawn(parent, "helper", instructions="Focus on security.")
        prompt_text = str(child.chatbot._config.get("system_prompt_builder", ""))
        assert "Focus on security" in prompt_text

    def test_spawn_with_metadata(self):
        parent = _make_agent()
        child = spawn(parent, "helper", metadata={"priority": "high"})
        assert child.metadata["priority"] == "high"

    def test_spawn_tracks_parent(self):
        parent = _make_agent()
        child = spawn(parent, "helper")
        assert parent_name(child) == "test_agent"

    def test_spawn_tracks_children(self):
        parent = _make_agent()
        spawn(parent, "a")
        spawn(parent, "b")
        assert children(parent) == ["a", "b"]

    def test_spawn_links_shared_state(self):
        parent = _make_agent()
        state = SharedState()
        child = spawn(parent, "helper", shared_state=state)
        assert parent.metadata["_shared_state"] is state
        assert child.metadata["_shared_state"] is state

    def test_spawn_without_shared_state(self):
        parent = _make_agent()
        child = spawn(parent, "helper")
        assert "_shared_state" not in parent.metadata
        assert "_shared_state" not in child.metadata

    def test_spawn_invalid_persona_type(self):
        parent = _make_agent()
        import pytest

        with pytest.raises(TypeError, match="str, PersonaDefinition, or None"):
            spawn(parent, "bad", persona=42)  # type: ignore[arg-type]

    def test_spawn_multiple_children(self):
        parent = _make_agent()
        c1 = spawn(parent, "first")
        c2 = spawn(parent, "second")
        c3 = spawn(parent, "third")
        assert children(parent) == ["first", "second", "third"]
        assert parent_name(c1) == parent.name
        assert parent_name(c2) == parent.name
        assert parent_name(c3) == parent.name

    def test_child_has_own_memory(self):
        parent = _make_agent()
        child = spawn(parent, "helper")
        parent.remember("key", "parent_val")
        assert child.recall("key") is None

    def test_child_has_own_conversation(self):
        parent = _make_agent()
        child = spawn(parent, "helper")
        assert child.conversation is None
        assert parent.conversation is None


# ---------------------------------------------------------------------------
# delegate
# ---------------------------------------------------------------------------


class TestDelegate:
    def test_delegate_returns_result(self):
        parent = _make_agent()
        child = spawn(parent, "helper")
        child.chatbot.mock_responses(["Task completed."])
        result = delegate(parent, child, "Do the thing")
        assert isinstance(result, SubagentResult)
        assert result.response == "Task completed."
        assert result.agent == "helper"
        assert result.parent == "test_agent"
        assert result.duration >= 0

    def test_delegate_records_in_shared_state(self):
        parent = _make_agent()
        state = SharedState()
        child = spawn(parent, "helper", shared_state=state)
        child.chatbot.mock_responses(["Done."])
        delegate(parent, child, "Analyze this")
        assert state.get("delegated_task", namespace="helper") == "Analyze this"
        assert state.get("delegated_result", namespace="helper") == "Done."

    def test_delegate_with_explicit_shared_state(self):
        parent = _make_agent()
        child = spawn(parent, "helper")  # No shared state from spawn
        state = SharedState()
        child.chatbot.mock_responses(["Result."])
        delegate(parent, child, "Check this", shared_state=state)
        assert state.get("delegated_task", namespace="helper") == "Check this"
        assert state.get("delegated_result", namespace="helper") == "Result."

    def test_delegate_without_shared_state(self):
        parent = _make_agent()
        child = spawn(parent, "helper")
        child.chatbot.mock_responses(["OK."])
        result = delegate(parent, child, "Simple task")
        assert result.response == "OK."
        # No shared state — no error

    def test_delegate_auto_discovers_shared_state(self):
        parent = _make_agent()
        state = SharedState()
        child = spawn(parent, "helper", shared_state=state)
        child.chatbot.mock_responses(["Found it."])
        # Don't pass shared_state explicitly — should auto-discover
        delegate(parent, child, "Look for it")
        assert state.get("delegated_task", namespace="helper") == "Look for it"

    def test_delegate_updates_subagent_conversation(self):
        parent = _make_agent()
        child = spawn(parent, "helper")
        child.chatbot.mock_responses(["Answer."])
        result = delegate(parent, child, "Question?")
        assert result.conversation is not None
        assert result.conversation.get_message_count() >= 2

    def test_delegate_records_capture_events(self):
        parent = _make_agent()
        child = spawn(parent, "helper")
        child.chatbot.mock_responses(["Response."])
        delegate(parent, child, "Do work")
        events = child.capture.events
        assert len(events) >= 2  # prompt + response

    def test_delegate_multiple_tasks(self):
        parent = _make_agent()
        state = SharedState()
        child = spawn(parent, "helper", shared_state=state)
        child.chatbot.mock_responses(["First result.", "Second result."])
        r1 = delegate(parent, child, "Task 1")
        r2 = delegate(parent, child, "Task 2")
        assert r1.response == "First result."
        assert r2.response == "Second result."
        # Shared state has the latest task/result
        assert state.get("delegated_task", namespace="helper") == "Task 2"
        assert state.get("delegated_result", namespace="helper") == "Second result."

    def test_delegate_tracks_history(self):
        parent = _make_agent()
        state = SharedState()
        child = spawn(parent, "helper", shared_state=state)
        child.chatbot.mock_responses(["Done."])
        delegate(parent, child, "Work")
        history = state.history
        # At least 2 entries: delegated_task + delegated_result
        assert len(history) >= 2
        task_change = history[0]
        assert task_change.key == "delegated_task"
        assert task_change.agent == "test_agent"  # parent set the task
        result_change = history[1]
        assert result_change.key == "delegated_result"
        assert result_change.agent == "helper"  # child set the result


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------


class TestIntrospection:
    def test_children_empty(self):
        agent = _make_agent()
        assert children(agent) == []

    def test_parent_name_none(self):
        agent = _make_agent()
        assert parent_name(agent) is None

    def test_parent_child_relationship(self):
        parent = _make_agent()
        child = spawn(parent, "sub")
        assert children(parent) == ["sub"]
        assert parent_name(child) == parent.name

    def test_children_returns_copy(self):
        parent = _make_agent()
        spawn(parent, "a")
        result = children(parent)
        result.append("fake")
        assert children(parent) == ["a"]

    def test_grandchild(self):
        grandparent = _make_agent(name="gp")
        parent_agent = spawn(grandparent, "parent")
        child = spawn(parent_agent, "child")
        assert children(grandparent) == ["parent"]
        assert children(parent_agent) == ["child"]
        assert parent_name(parent_agent) == "gp"
        assert parent_name(child) == "parent"


# ---------------------------------------------------------------------------
# Top-level imports
# ---------------------------------------------------------------------------


class TestImport:
    def test_import_from_submodule(self):
        from talk_box.subagents import (
            spawn,
            delegate,
            children,
            parent_name,
            SubagentResult,
        )

        assert spawn is not None
        assert delegate is not None
        assert children is not None
        assert parent_name is not None
        assert SubagentResult is not None
