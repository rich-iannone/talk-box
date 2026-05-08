"""Tests for talk_box.replay module."""

from talk_box.capture import ConversationCapture
from talk_box.replay import ReplayResult, ReplayTurn, replay


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _echo_responder(model_key: str, prompt: str) -> str:
    """Simple responder that echoes the prompt with the model key."""
    return f"[{model_key}] {prompt}"


def _make_source(
    *, system_prompt: str = "", prompts_responses: list[tuple[str, str]] | None = None
):
    """Create a ConversationCapture with some recorded turns."""
    cap = ConversationCapture(session_id="original-session", metadata={"model": "original-model"})
    if system_prompt:
        cap.record_prompt(system_prompt, role="system")
    for prompt, response in prompts_responses or []:
        cap.record_prompt(prompt)
        cap.record_response(response, model="original-model")
    return cap


# ---------------------------------------------------------------------------
# ReplayTurn
# ---------------------------------------------------------------------------


class TestReplayTurn:
    def test_basic(self):
        turn = ReplayTurn(
            turn_number=1,
            prompt="Hello",
            original_response="Hi",
            replayed_response="Hey",
        )
        assert turn.turn_number == 1
        assert turn.prompt == "Hello"
        assert turn.original_response == "Hi"
        assert turn.replayed_response == "Hey"

    def test_with_models(self):
        turn = ReplayTurn(
            turn_number=1,
            prompt="Hello",
            original_response="Hi",
            replayed_response="Hey",
            original_model="gpt-4o",
            replay_model="gpt-4o-mini",
            duration_ms=100.0,
        )
        assert turn.original_model == "gpt-4o"
        assert turn.replay_model == "gpt-4o-mini"
        assert turn.duration_ms == 100.0

    def test_frozen(self):
        turn = ReplayTurn(turn_number=1, prompt="a", original_response="b", replayed_response="c")
        try:
            turn.prompt = "new"  # type: ignore[misc]
            assert False, "Should be frozen"
        except Exception:
            pass


# ---------------------------------------------------------------------------
# ReplayResult
# ---------------------------------------------------------------------------


class TestReplayResult:
    def test_turn_count(self):
        result = ReplayResult(
            original_session_id="s1",
            replay_session_id="s2",
            replay_model="m1",
            turns=[
                ReplayTurn(1, "Q1", "A1", "R1"),
                ReplayTurn(2, "Q2", "A2", "R2"),
            ],
            capture=ConversationCapture(),
        )
        assert result.turn_count == 2

    def test_empty_turns(self):
        result = ReplayResult(
            original_session_id="s1",
            replay_session_id="s2",
            replay_model="m1",
            turns=[],
            capture=ConversationCapture(),
        )
        assert result.turn_count == 0


# ---------------------------------------------------------------------------
# replay() — basic usage
# ---------------------------------------------------------------------------


class TestReplayBasic:
    def test_single_turn(self):
        source = _make_source(prompts_responses=[("What is Python?", "A programming language.")])
        result = replay(source, _echo_responder, model="test-model")

        assert result.original_session_id == "original-session"
        assert result.replay_model == "test-model"
        assert result.turn_count == 1

        turn = result.turns[0]
        assert turn.turn_number == 1
        assert turn.prompt == "What is Python?"
        assert turn.original_response == "A programming language."
        assert turn.replayed_response == "[test-model] What is Python?"
        assert turn.original_model == "original-model"
        assert turn.replay_model == "test-model"
        assert turn.duration_ms is not None and turn.duration_ms >= 0

    def test_multi_turn(self):
        source = _make_source(
            prompts_responses=[
                ("Q1", "A1"),
                ("Q2", "A2"),
                ("Q3", "A3"),
            ]
        )
        result = replay(source, _echo_responder, model="m2")

        assert result.turn_count == 3
        assert result.turns[0].prompt == "Q1"
        assert result.turns[1].prompt == "Q2"
        assert result.turns[2].prompt == "Q3"
        assert result.turns[2].replayed_response == "[m2] Q3"

    def test_no_model(self):
        source = _make_source(prompts_responses=[("Hello", "Hi")])
        result = replay(source, _echo_responder)

        assert result.replay_model == ""
        assert result.turns[0].replayed_response == "[] Hello"

    def test_empty_source_raises(self):
        source = ConversationCapture()
        try:
            replay(source, _echo_responder)
            assert False, "Should raise ValueError"
        except ValueError as e:
            assert "no user prompts" in str(e).lower()

    def test_system_only_raises(self):
        source = ConversationCapture()
        source.record_prompt("You are helpful", role="system")
        try:
            replay(source, _echo_responder)
            assert False, "Should raise ValueError"
        except ValueError as e:
            assert "no user prompts" in str(e).lower()


# ---------------------------------------------------------------------------
# replay() — system prompt handling
# ---------------------------------------------------------------------------


class TestReplaySystemPrompt:
    def test_original_system_prompt_reused(self):
        source = _make_source(
            system_prompt="You are a helpful assistant.",
            prompts_responses=[("Hello", "Hi")],
        )
        result = replay(source, _echo_responder, model="m1")

        assert result.system_prompt == "You are a helpful assistant."
        # System prompt should be in the replay capture
        sys_prompts = [e for e in result.capture.prompts() if e.role == "system"]
        assert len(sys_prompts) == 1
        assert sys_prompts[0].content == "You are a helpful assistant."

    def test_custom_system_prompt_overrides(self):
        source = _make_source(
            system_prompt="Original system prompt",
            prompts_responses=[("Hello", "Hi")],
        )
        result = replay(
            source,
            _echo_responder,
            model="m1",
            system_prompt="New system prompt",
        )

        assert result.system_prompt == "New system prompt"
        sys_prompts = [e for e in result.capture.prompts() if e.role == "system"]
        assert len(sys_prompts) == 1
        assert sys_prompts[0].content == "New system prompt"

    def test_explicit_empty_system_prompt(self):
        source = _make_source(
            system_prompt="Original system prompt",
            prompts_responses=[("Hello", "Hi")],
        )
        result = replay(source, _echo_responder, model="m1", system_prompt="")

        assert result.system_prompt == ""
        sys_prompts = [e for e in result.capture.prompts() if e.role == "system"]
        assert len(sys_prompts) == 0

    def test_no_system_prompt(self):
        source = _make_source(prompts_responses=[("Hello", "Hi")])
        result = replay(source, _echo_responder, model="m1")

        assert result.system_prompt == ""
        sys_prompts = [e for e in result.capture.prompts() if e.role == "system"]
        assert len(sys_prompts) == 0


# ---------------------------------------------------------------------------
# replay() — capture recording
# ---------------------------------------------------------------------------


class TestReplayCapture:
    def test_capture_has_events(self):
        source = _make_source(prompts_responses=[("Q1", "A1"), ("Q2", "A2")])
        result = replay(source, _echo_responder, model="m1")

        cap = result.capture
        assert len(cap.prompts()) == 2
        assert len(cap.responses()) == 2
        assert cap.responses()[0].model == "m1"

    def test_capture_metadata(self):
        source = _make_source(prompts_responses=[("Q", "A")])
        result = replay(source, _echo_responder, model="m1", metadata={"env": "test"})

        meta = result.capture.metadata
        assert meta["replay_of"] == "original-session"
        assert meta["replay_model"] == "m1"
        assert meta["env"] == "test"

    def test_capture_serializable(self, tmp_path):
        source = _make_source(prompts_responses=[("Q1", "A1")])
        result = replay(source, _echo_responder, model="m1")

        path = tmp_path / "replay.json"
        result.capture.to_json(path)
        loaded = ConversationCapture.from_json(path)
        assert len(loaded) == len(result.capture)

    def test_replay_session_id_unique(self):
        source = _make_source(prompts_responses=[("Q", "A")])
        r1 = replay(source, _echo_responder, model="m1")
        r2 = replay(source, _echo_responder, model="m1")
        assert r1.replay_session_id != r2.replay_session_id


# ---------------------------------------------------------------------------
# replay() — error handling
# ---------------------------------------------------------------------------


class TestReplayErrors:
    def test_responder_error_recorded(self):
        def failing_responder(model: str, prompt: str) -> str:
            raise RuntimeError("Model unavailable")

        source = _make_source(prompts_responses=[("Q1", "A1")])
        result = replay(source, failing_responder, model="broken-model")

        assert result.turn_count == 1
        assert result.turns[0].replayed_response == ""
        errors = result.capture.errors()
        assert len(errors) == 1
        assert "Model unavailable" in errors[0].content

    def test_partial_failure(self):
        call_count = 0

        def sometimes_fails(model: str, prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise ValueError("Intermittent failure")
            return f"OK: {prompt}"

        source = _make_source(prompts_responses=[("Q1", "A1"), ("Q2", "A2"), ("Q3", "A3")])
        result = replay(source, sometimes_fails, model="m1")

        assert result.turn_count == 3
        assert result.turns[0].replayed_response == "OK: Q1"
        assert result.turns[1].replayed_response == ""  # Failed
        assert result.turns[2].replayed_response == "OK: Q3"
        assert len(result.capture.errors()) == 1

    def test_error_type_recorded(self):
        def type_error_responder(model: str, prompt: str) -> str:
            raise TypeError("bad type")

        source = _make_source(prompts_responses=[("Q", "A")])
        result = replay(source, type_error_responder, model="m1")

        error = result.capture.errors()[0]
        assert error.metadata["error_type"] == "TypeError"


# ---------------------------------------------------------------------------
# replay() — edge cases
# ---------------------------------------------------------------------------


class TestReplayEdgeCases:
    def test_source_with_extra_events(self):
        """Source has tool calls, guards, etc. — only user prompts are replayed."""
        source = ConversationCapture(session_id="complex")
        source.record_prompt("You are a bot", role="system")
        source.record_prompt("Hello")
        source.record_guard_check("no_pii", True)
        source.record_response("Hi there", model="m1")
        source.record_tool_call("search", arguments={"q": "test"})
        source.record_tool_result("search", "Found it")
        source.record_prompt("Thanks")
        source.record_response("You're welcome", model="m1")

        result = replay(source, _echo_responder, model="m2")

        assert result.turn_count == 2
        assert result.turns[0].prompt == "Hello"
        assert result.turns[1].prompt == "Thanks"

    def test_unmatched_prompt(self):
        """Source has a prompt without a response."""
        source = ConversationCapture(session_id="unmatched")
        source.record_prompt("Q1")
        source.record_response("A1", model="m1")
        source.record_prompt("Q2")  # No response

        result = replay(source, _echo_responder, model="m2")

        assert result.turn_count == 2
        assert result.turns[0].original_response == "A1"
        assert result.turns[1].original_response == ""  # No original response

    def test_replay_of_replay(self):
        """Replay a previously replayed capture."""
        source = _make_source(prompts_responses=[("Hello", "Hi")])
        r1 = replay(source, _echo_responder, model="m1")
        r2 = replay(r1.capture, _echo_responder, model="m2")

        assert r2.turn_count == 1
        assert r2.turns[0].original_response == "[m1] Hello"
        assert r2.turns[0].replayed_response == "[m2] Hello"
