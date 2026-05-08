"""Tests for talk_box.capture module."""

from talk_box.capture import (
    CaptureEvent,
    ConversationCapture,
    EventType,
)


# ---------------------------------------------------------------------------
# EventType enum
# ---------------------------------------------------------------------------


class TestEventType:
    def test_values(self):
        assert EventType.PROMPT.value == "prompt"
        assert EventType.RESPONSE.value == "response"
        assert EventType.TOOL_CALL.value == "tool_call"
        assert EventType.TOOL_RESULT.value == "tool_result"
        assert EventType.GUARD_CHECK.value == "guard_check"
        assert EventType.PATHWAY_TRANSITION.value == "pathway_transition"
        assert EventType.ERROR.value == "error"
        assert EventType.METADATA.value == "metadata"

    def test_all_members(self):
        assert len(EventType) == 8


# ---------------------------------------------------------------------------
# CaptureEvent
# ---------------------------------------------------------------------------


class TestCaptureEvent:
    def test_basic(self):
        e = CaptureEvent(event_type=EventType.PROMPT, content="Hello")
        assert e.event_type == EventType.PROMPT
        assert e.content == "Hello"
        assert e.model == ""
        assert e.role == ""
        assert e.duration_ms is None
        assert e.metadata == {}

    def test_frozen(self):
        e = CaptureEvent(event_type=EventType.PROMPT)
        try:
            e.content = "new"  # type: ignore[misc]
            assert False, "Should be frozen"
        except Exception:
            pass

    def test_to_dict(self):
        e = CaptureEvent(
            event_type=EventType.RESPONSE,
            content="Hello world",
            timestamp=1000.0,
            event_id="abc",
            model="openai:gpt-4o",
            role="assistant",
            duration_ms=150.5,
            metadata={"tokens": 10},
        )
        d = e.to_dict()
        assert d["event_type"] == "response"
        assert d["content"] == "Hello world"
        assert d["model"] == "openai:gpt-4o"
        assert d["duration_ms"] == 150.5
        assert d["metadata"] == {"tokens": 10}

    def test_from_dict(self):
        data = {
            "event_type": "prompt",
            "content": "What is Python?",
            "timestamp": 1000.0,
            "event_id": "xyz",
            "model": "",
            "role": "user",
            "duration_ms": None,
            "metadata": {},
        }
        e = CaptureEvent.from_dict(data)
        assert e.event_type == EventType.PROMPT
        assert e.content == "What is Python?"
        assert e.role == "user"

    def test_roundtrip(self):
        original = CaptureEvent(
            event_type=EventType.TOOL_CALL,
            content="search",
            timestamp=123.456,
            event_id="id1",
            model="test:m1",
            metadata={"args": {"q": "hello"}},
        )
        restored = CaptureEvent.from_dict(original.to_dict())
        assert restored.event_type == original.event_type
        assert restored.content == original.content
        assert restored.timestamp == original.timestamp
        assert restored.metadata == original.metadata


# ---------------------------------------------------------------------------
# ConversationCapture — recording
# ---------------------------------------------------------------------------


class TestCaptureRecording:
    def test_record_prompt(self):
        cap = ConversationCapture()
        event = cap.record_prompt("Hello")
        assert event.event_type == EventType.PROMPT
        assert event.content == "Hello"
        assert event.role == "user"
        assert len(cap) == 1

    def test_record_prompt_system_role(self):
        cap = ConversationCapture()
        event = cap.record_prompt("You are helpful", role="system")
        assert event.role == "system"

    def test_record_response(self):
        cap = ConversationCapture()
        event = cap.record_response("Hi there!", model="openai:gpt-4o", duration_ms=200)
        assert event.event_type == EventType.RESPONSE
        assert event.content == "Hi there!"
        assert event.model == "openai:gpt-4o"
        assert event.role == "assistant"
        assert event.duration_ms == 200

    def test_record_tool_call(self):
        cap = ConversationCapture()
        event = cap.record_tool_call("search", arguments={"query": "Python"}, model="test:m1")
        assert event.event_type == EventType.TOOL_CALL
        assert event.content == "search"
        assert event.metadata["tool_name"] == "search"
        assert event.metadata["arguments"] == {"query": "Python"}

    def test_record_tool_call_no_arguments(self):
        cap = ConversationCapture()
        event = cap.record_tool_call("get_time")
        assert "arguments" not in event.metadata

    def test_record_tool_result(self):
        cap = ConversationCapture()
        event = cap.record_tool_result("search", "Found 5 results", duration_ms=50)
        assert event.event_type == EventType.TOOL_RESULT
        assert event.content == "Found 5 results"
        assert event.metadata["tool_name"] == "search"
        assert event.metadata["success"] is True

    def test_record_tool_result_failure(self):
        cap = ConversationCapture()
        event = cap.record_tool_result("search", "Connection error", success=False)
        assert event.metadata["success"] is False

    def test_record_guard_check_passed(self):
        cap = ConversationCapture()
        event = cap.record_guard_check("no_pii", True)
        assert event.event_type == EventType.GUARD_CHECK
        assert "passed" in event.content
        assert event.metadata["guard_name"] == "no_pii"
        assert event.metadata["passed"] is True

    def test_record_guard_check_failed(self):
        cap = ConversationCapture()
        event = cap.record_guard_check("max_length", False, message="Too long")
        assert "failed" in event.content
        assert "Too long" in event.content
        assert event.metadata["passed"] is False

    def test_record_pathway_transition(self):
        cap = ConversationCapture()
        event = cap.record_pathway_transition("greeting", "main", trigger="user_input")
        assert event.event_type == EventType.PATHWAY_TRANSITION
        assert "greeting -> main" in event.content
        assert "(user_input)" in event.content
        assert event.metadata["from_state"] == "greeting"
        assert event.metadata["to_state"] == "main"
        assert event.metadata["trigger"] == "user_input"

    def test_record_pathway_transition_no_trigger(self):
        cap = ConversationCapture()
        event = cap.record_pathway_transition("a", "b")
        assert event.content == "a -> b"
        assert "trigger" not in event.metadata

    def test_record_error(self):
        cap = ConversationCapture()
        event = cap.record_error("Connection timed out", error_type="TimeoutError")
        assert event.event_type == EventType.ERROR
        assert event.content == "Connection timed out"
        assert event.metadata["error_type"] == "TimeoutError"

    def test_record_error_no_type(self):
        cap = ConversationCapture()
        event = cap.record_error("Something went wrong")
        assert "error_type" not in event.metadata

    def test_record_generic(self):
        cap = ConversationCapture()
        event = cap.record(EventType.METADATA, "session_start", metadata={"version": "1.0"})
        assert event.event_type == EventType.METADATA
        assert event.metadata["version"] == "1.0"

    def test_events_have_unique_ids(self):
        cap = ConversationCapture()
        e1 = cap.record_prompt("Hello")
        e2 = cap.record_response("Hi")
        assert e1.event_id != e2.event_id
        assert e1.event_id != ""
        assert e2.event_id != ""

    def test_events_have_timestamps(self):
        cap = ConversationCapture()
        e = cap.record_prompt("Hello")
        assert e.timestamp > 0

    def test_metadata_not_mutated(self):
        cap = ConversationCapture()
        original_meta = {"key": "value"}
        cap.record_prompt("Hello", metadata=original_meta)
        original_meta["new_key"] = "new_value"
        events = cap.events
        assert "new_key" not in events[0].metadata


# ---------------------------------------------------------------------------
# ConversationCapture — properties
# ---------------------------------------------------------------------------


class TestCaptureProperties:
    def test_session_id_auto_generated(self):
        cap = ConversationCapture()
        assert len(cap.session_id) > 0

    def test_session_id_custom(self):
        cap = ConversationCapture(session_id="my-session")
        assert cap.session_id == "my-session"

    def test_metadata(self):
        cap = ConversationCapture(metadata={"model": "gpt-4o"})
        assert cap.metadata == {"model": "gpt-4o"}

    def test_metadata_default(self):
        cap = ConversationCapture()
        assert cap.metadata == {}

    def test_metadata_returns_copy(self):
        cap = ConversationCapture(metadata={"key": "value"})
        meta = cap.metadata
        meta["new"] = "data"
        assert "new" not in cap.metadata

    def test_events_returns_copy(self):
        cap = ConversationCapture()
        cap.record_prompt("Hello")
        events = cap.events
        events.clear()
        assert len(cap) == 1

    def test_start_time(self):
        cap = ConversationCapture()
        assert cap.start_time > 0

    def test_duration_ms(self):
        cap = ConversationCapture()
        assert cap.duration_ms >= 0

    def test_len(self):
        cap = ConversationCapture()
        assert len(cap) == 0
        cap.record_prompt("a")
        cap.record_response("b")
        assert len(cap) == 2


# ---------------------------------------------------------------------------
# ConversationCapture — query methods
# ---------------------------------------------------------------------------


class TestCaptureQueries:
    def _make_capture(self):
        cap = ConversationCapture()
        cap.record_prompt("Q1")
        cap.record_response("A1", model="m1")
        cap.record_tool_call("search", arguments={"q": "test"})
        cap.record_tool_result("search", "result")
        cap.record_prompt("Q2")
        cap.record_response("A2", model="m1")
        cap.record_guard_check("no_pii", True)
        cap.record_error("timeout")
        return cap

    def test_filter(self):
        cap = self._make_capture()
        assert len(cap.filter(EventType.PROMPT)) == 2
        assert len(cap.filter(EventType.RESPONSE)) == 2
        assert len(cap.filter(EventType.TOOL_CALL)) == 1

    def test_prompts(self):
        cap = self._make_capture()
        prompts = cap.prompts()
        assert len(prompts) == 2
        assert prompts[0].content == "Q1"
        assert prompts[1].content == "Q2"

    def test_responses(self):
        cap = self._make_capture()
        resps = cap.responses()
        assert len(resps) == 2
        assert resps[0].content == "A1"

    def test_tool_calls(self):
        cap = self._make_capture()
        calls = cap.tool_calls()
        assert len(calls) == 1
        assert calls[0].metadata["tool_name"] == "search"

    def test_tool_results(self):
        cap = self._make_capture()
        results = cap.tool_results()
        assert len(results) == 1
        assert results[0].content == "result"

    def test_errors(self):
        cap = self._make_capture()
        errs = cap.errors()
        assert len(errs) == 1
        assert errs[0].content == "timeout"

    def test_turns(self):
        cap = self._make_capture()
        t = cap.turns()
        assert len(t) == 2
        assert t[0][0].content == "Q1"
        assert t[0][1] is not None
        assert t[0][1].content == "A1"
        assert t[1][0].content == "Q2"
        assert t[1][1] is not None

    def test_turns_unmatched_prompt(self):
        cap = ConversationCapture()
        cap.record_prompt("Q1")
        t = cap.turns()
        assert len(t) == 1
        assert t[0][0].content == "Q1"
        assert t[0][1] is None


# ---------------------------------------------------------------------------
# ConversationCapture — serialization
# ---------------------------------------------------------------------------


class TestCaptureSerialization:
    def test_to_dict(self):
        cap = ConversationCapture(session_id="test-session", metadata={"v": 1})
        cap.record_prompt("Hello")
        cap.record_response("Hi")
        d = cap.to_dict()
        assert d["session_id"] == "test-session"
        assert d["metadata"] == {"v": 1}
        assert len(d["events"]) == 2
        assert d["events"][0]["event_type"] == "prompt"
        assert d["events"][1]["event_type"] == "response"

    def test_from_dict(self):
        original = ConversationCapture(session_id="s1", metadata={"m": "test"})
        original.record_prompt("Q")
        original.record_response("A")
        d = original.to_dict()

        restored = ConversationCapture.from_dict(d)
        assert restored.session_id == "s1"
        assert restored.metadata == {"m": "test"}
        assert len(restored) == 2
        assert restored.events[0].content == "Q"
        assert restored.events[1].content == "A"

    def test_roundtrip_dict(self):
        cap = ConversationCapture(session_id="round")
        cap.record_prompt("Hello", metadata={"lang": "en"})
        cap.record_response("Hi", model="m1", duration_ms=100)
        cap.record_tool_call("search", arguments={"q": "test"})
        cap.record_guard_check("no_pii", True)
        cap.record_error("oops", error_type="RuntimeError")

        restored = ConversationCapture.from_dict(cap.to_dict())
        assert len(restored) == len(cap)
        for orig, rest in zip(cap.events, restored.events):
            assert orig.event_type == rest.event_type
            assert orig.content == rest.content
            assert orig.metadata == rest.metadata

    def test_to_json(self, tmp_path):
        cap = ConversationCapture(session_id="json-test")
        cap.record_prompt("Hello")
        cap.record_response("Hi")
        path = tmp_path / "capture.json"
        cap.to_json(path)
        assert path.exists()
        # Verify valid JSON
        import json

        data = json.loads(path.read_text())
        assert data["session_id"] == "json-test"
        assert len(data["events"]) == 2

    def test_from_json(self, tmp_path):
        cap = ConversationCapture(session_id="load-test")
        cap.record_prompt("Question")
        cap.record_response("Answer", model="m1")
        path = tmp_path / "capture.json"
        cap.to_json(path)

        loaded = ConversationCapture.from_json(path)
        assert loaded.session_id == "load-test"
        assert len(loaded) == 2
        assert loaded.events[0].content == "Question"
        assert loaded.events[1].model == "m1"

    def test_to_json_creates_directory(self, tmp_path):
        cap = ConversationCapture()
        cap.record_prompt("test")
        path = tmp_path / "subdir" / "nested" / "capture.json"
        cap.to_json(path)
        assert path.exists()

    def test_from_json_not_found(self, tmp_path):
        try:
            ConversationCapture.from_json(tmp_path / "nonexistent.json")
            assert False, "Should raise FileNotFoundError"
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestCaptureIntegration:
    def test_full_conversation_flow(self):
        cap = ConversationCapture(
            session_id="integration-test",
            metadata={"model": "gpt-4o", "persona": "code_reviewer"},
        )

        cap.record_prompt("You are a code reviewer.", role="system")
        cap.record_prompt("Review this function: def add(a, b): return a + b")
        cap.record_guard_check("max_input_length", True)
        cap.record_response(
            "The function is clean and simple.",
            model="openai:gpt-4o",
            duration_ms=350,
        )
        cap.record_tool_call("lint_check", arguments={"code": "def add(a, b): return a + b"})
        cap.record_tool_result("lint_check", "No issues found", duration_ms=20)
        cap.record_pathway_transition("review", "summary", trigger="review_complete")

        assert len(cap) == 7
        assert len(cap.prompts()) == 2
        assert len(cap.responses()) == 1
        assert len(cap.tool_calls()) == 1
        assert len(cap.tool_results()) == 1
        assert cap.session_id == "integration-test"

    def test_event_ordering_preserved(self):
        cap = ConversationCapture()
        cap.record_prompt("First")
        cap.record_response("Second")
        cap.record_prompt("Third")
        events = cap.events
        assert events[0].content == "First"
        assert events[1].content == "Second"
        assert events[2].content == "Third"
