from talk_box.capture import ConversationCapture
from talk_box.diff import DiffResult, TurnDiff, TurnStatus, diff


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_capture(
    session_id: str = "session",
    *,
    system_prompt: str = "",
    prompts_responses: list[tuple[str, str]] | None = None,
    model: str = "model-a",
):
    cap = ConversationCapture(session_id=session_id)
    if system_prompt:
        cap.record_prompt(system_prompt, role="system")
    for prompt, response in prompts_responses or []:
        cap.record_prompt(prompt)
        cap.record_response(response, model=model)
    return cap


# ---------------------------------------------------------------------------
# TurnDiff
# ---------------------------------------------------------------------------


class TestTurnDiff:
    def test_basic(self):
        td = TurnDiff(
            turn_number=1,
            prompt="Hello",
            left_response="Hi",
            right_response="Hey",
            status=TurnStatus.CHANGED,
            similarity=0.0,
        )
        assert td.turn_number == 1
        assert td.status == TurnStatus.CHANGED

    def test_frozen(self):
        td = TurnDiff(1, "a", "b", "c", TurnStatus.IDENTICAL, 1.0)
        try:
            td.prompt = "new"  # type: ignore[misc]
            assert False, "Should be frozen"
        except Exception:
            pass


# ---------------------------------------------------------------------------
# TurnStatus
# ---------------------------------------------------------------------------


class TestTurnStatus:
    def test_values(self):
        assert TurnStatus.IDENTICAL.value == "identical"
        assert TurnStatus.SIMILAR.value == "similar"
        assert TurnStatus.CHANGED.value == "changed"
        assert TurnStatus.ADDED.value == "added"
        assert TurnStatus.REMOVED.value == "removed"

    def test_all_members(self):
        assert len(TurnStatus) == 5


# ---------------------------------------------------------------------------
# DiffResult — properties
# ---------------------------------------------------------------------------


class TestDiffResultProperties:
    def test_counts(self):
        turns = [
            TurnDiff(1, "Q1", "A", "A", TurnStatus.IDENTICAL, 1.0),
            TurnDiff(2, "Q2", "A", "B similar words overlap here", TurnStatus.SIMILAR, 0.8),
            TurnDiff(3, "Q3", "A", "completely different", TurnStatus.CHANGED, 0.1),
            TurnDiff(4, "Q4", "A", "", TurnStatus.REMOVED, 0.0),
            TurnDiff(5, "Q5", "", "B", TurnStatus.ADDED, 0.0),
        ]
        result = DiffResult("s1", "s2", turns)
        assert result.turn_count == 5
        assert result.identical_count == 1
        assert result.similar_count == 1
        assert result.changed_count == 1
        assert result.removed_count == 1
        assert result.added_count == 1

    def test_similarity_score(self):
        turns = [
            TurnDiff(1, "Q1", "A", "A", TurnStatus.IDENTICAL, 1.0),
            TurnDiff(2, "Q2", "A", "B", TurnStatus.CHANGED, 0.0),
        ]
        result = DiffResult("s1", "s2", turns)
        assert result.similarity_score == 0.5

    def test_similarity_score_empty(self):
        result = DiffResult("s1", "s2", [])
        assert result.similarity_score == 1.0

    def test_summary(self):
        turns = [
            TurnDiff(1, "Q1", "A", "A", TurnStatus.IDENTICAL, 1.0),
            TurnDiff(2, "Q2", "A", "B", TurnStatus.CHANGED, 0.2),
        ]
        result = DiffResult("left", "right", turns)
        s = result.summary()
        assert s["left_session_id"] == "left"
        assert s["right_session_id"] == "right"
        assert s["turn_count"] == 2
        assert s["identical"] == 1
        assert s["changed"] == 1
        assert s["similarity_score"] == 0.6


# ---------------------------------------------------------------------------
# diff() — identical captures
# ---------------------------------------------------------------------------


class TestDiffIdentical:
    def test_same_capture(self):
        cap = _make_capture(prompts_responses=[("Hello", "Hi there")])
        result = diff(cap, cap)

        assert result.turn_count == 1
        assert result.turns[0].status == TurnStatus.IDENTICAL
        assert result.turns[0].similarity == 1.0
        assert result.identical_count == 1

    def test_identical_content(self):
        left = _make_capture("s1", prompts_responses=[("Q1", "Response A"), ("Q2", "Response B")])
        right = _make_capture("s2", prompts_responses=[("Q1", "Response A"), ("Q2", "Response B")])
        result = diff(left, right)

        assert result.turn_count == 2
        assert all(t.status == TurnStatus.IDENTICAL for t in result.turns)
        assert result.similarity_score == 1.0


# ---------------------------------------------------------------------------
# diff() — changed responses
# ---------------------------------------------------------------------------


class TestDiffChanged:
    def test_completely_different(self):
        left = _make_capture("s1", prompts_responses=[("Hello", "Alpha bravo charlie")])
        right = _make_capture("s2", prompts_responses=[("Hello", "Delta echo foxtrot")])
        result = diff(left, right)

        assert result.turn_count == 1
        assert result.turns[0].status == TurnStatus.CHANGED
        assert result.turns[0].similarity == 0.0

    def test_similar_responses(self):
        left = _make_capture(
            "s1",
            prompts_responses=[
                (
                    "Hello",
                    "Python is a great programming language for data science web apps and backend development",
                )
            ],
        )
        right = _make_capture(
            "s2",
            prompts_responses=[
                (
                    "Hello",
                    "Python is a wonderful programming language for data science web apps and backend systems",
                )
            ],
        )
        result = diff(left, right)

        assert result.turn_count == 1
        assert result.turns[0].status == TurnStatus.SIMILAR
        assert 0.7 <= result.turns[0].similarity < 1.0

    def test_mixed_turns(self):
        left = _make_capture(
            "s1",
            prompts_responses=[
                ("Q1", "Same response here"),
                ("Q2", "Totally alpha bravo charlie delta echo"),
            ],
        )
        right = _make_capture(
            "s2",
            prompts_responses=[
                ("Q1", "Same response here"),
                ("Q2", "Completely foxtrot golf hotel india juliet"),
            ],
        )
        result = diff(left, right)

        assert result.turn_count == 2
        assert result.turns[0].status == TurnStatus.IDENTICAL
        assert result.turns[1].status == TurnStatus.CHANGED


# ---------------------------------------------------------------------------
# diff() — added and removed turns
# ---------------------------------------------------------------------------


class TestDiffAddedRemoved:
    def test_left_has_more_turns(self):
        left = _make_capture("s1", prompts_responses=[("Q1", "A1"), ("Q2", "A2"), ("Q3", "A3")])
        right = _make_capture("s2", prompts_responses=[("Q1", "A1")])
        result = diff(left, right)

        assert result.turn_count == 3
        assert result.turns[0].status == TurnStatus.IDENTICAL
        assert result.turns[1].status == TurnStatus.REMOVED
        assert result.turns[2].status == TurnStatus.REMOVED
        assert result.removed_count == 2

    def test_right_has_more_turns(self):
        left = _make_capture("s1", prompts_responses=[("Q1", "A1")])
        right = _make_capture("s2", prompts_responses=[("Q1", "A1"), ("Q2", "A2")])
        result = diff(left, right)

        assert result.turn_count == 2
        assert result.turns[0].status == TurnStatus.IDENTICAL
        assert result.turns[1].status == TurnStatus.ADDED
        assert result.added_count == 1

    def test_removed_turn_has_zero_similarity(self):
        left = _make_capture("s1", prompts_responses=[("Q1", "A1"), ("Q2", "A2")])
        right = _make_capture("s2", prompts_responses=[("Q1", "A1")])
        result = diff(left, right)

        assert result.turns[1].similarity == 0.0
        assert result.turns[1].right_response == ""

    def test_added_turn_has_zero_similarity(self):
        left = _make_capture("s1", prompts_responses=[("Q1", "A1")])
        right = _make_capture("s2", prompts_responses=[("Q1", "A1"), ("Q2", "A2")])
        result = diff(left, right)

        assert result.turns[1].similarity == 0.0
        assert result.turns[1].left_response == ""


# ---------------------------------------------------------------------------
# diff() — system prompt handling
# ---------------------------------------------------------------------------


class TestDiffSystemPrompt:
    def test_system_prompts_excluded(self):
        left = _make_capture(
            "s1", system_prompt="You are helpful", prompts_responses=[("Hello", "Hi")]
        )
        right = _make_capture(
            "s2", system_prompt="You are concise", prompts_responses=[("Hello", "Hi")]
        )
        result = diff(left, right)

        # Should only compare user prompt turns, not system prompts
        assert result.turn_count == 1
        assert result.turns[0].status == TurnStatus.IDENTICAL


# ---------------------------------------------------------------------------
# diff() — model tracking
# ---------------------------------------------------------------------------


class TestDiffModelTracking:
    def test_models_recorded(self):
        left = _make_capture("s1", prompts_responses=[("Q", "A")], model="gpt-4o")
        right = _make_capture("s2", prompts_responses=[("Q", "A")], model="gpt-4o-mini")
        result = diff(left, right)

        assert result.turns[0].left_model == "gpt-4o"
        assert result.turns[0].right_model == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# diff() — threshold
# ---------------------------------------------------------------------------


class TestDiffThreshold:
    def test_custom_threshold(self):
        # With words that share some overlap
        left = _make_capture(
            "s1",
            prompts_responses=[("Q", "alpha bravo charlie delta echo foxtrot golf hotel")],
        )
        right = _make_capture(
            "s2",
            prompts_responses=[("Q", "alpha bravo charlie delta india juliet kilo lima")],
        )
        # Jaccard: 4 shared / 12 total = 0.333...
        # With default threshold 0.7, this is CHANGED
        result_default = diff(left, right)
        assert result_default.turns[0].status == TurnStatus.CHANGED

        # With threshold 0.3, this should be SIMILAR
        result_low = diff(left, right, similarity_threshold=0.3)
        assert result_low.turns[0].status == TurnStatus.SIMILAR
        assert result_low.similarity_threshold == 0.3


# ---------------------------------------------------------------------------
# diff() — edge cases
# ---------------------------------------------------------------------------


class TestDiffEdgeCases:
    def test_both_empty(self):
        left = ConversationCapture(session_id="s1")
        right = ConversationCapture(session_id="s2")
        result = diff(left, right)

        assert result.turn_count == 0
        assert result.similarity_score == 1.0

    def test_empty_responses(self):
        left = _make_capture("s1", prompts_responses=[("Q", "")])
        right = _make_capture("s2", prompts_responses=[("Q", "")])
        result = diff(left, right)

        assert result.turns[0].status == TurnStatus.IDENTICAL
        assert result.turns[0].similarity == 1.0

    def test_one_empty_response(self):
        left = _make_capture("s1", prompts_responses=[("Q", "Some answer here")])
        right = _make_capture("s2", prompts_responses=[("Q", "")])
        result = diff(left, right)

        assert result.turns[0].similarity == 0.0
        assert result.turns[0].status == TurnStatus.CHANGED

    def test_metadata_passed_through(self):
        left = _make_capture("s1", prompts_responses=[("Q", "A")])
        right = _make_capture("s2", prompts_responses=[("Q", "A")])
        result = diff(left, right, metadata={"env": "test"})

        assert result.metadata == {"env": "test"}

    def test_session_ids(self):
        left = _make_capture("left-session", prompts_responses=[("Q", "A")])
        right = _make_capture("right-session", prompts_responses=[("Q", "A")])
        result = diff(left, right)

        assert result.left_session_id == "left-session"
        assert result.right_session_id == "right-session"

    def test_extra_events_ignored(self):
        """Tool calls, guards, etc. don't affect the diff — only prompts and responses."""
        left = ConversationCapture(session_id="s1")
        left.record_prompt("Hello")
        left.record_guard_check("no_pii", True)
        left.record_response("Hi there", model="m1")
        left.record_tool_call("search")

        right = ConversationCapture(session_id="s2")
        right.record_prompt("Hello")
        right.record_response("Hi there", model="m2")

        result = diff(left, right)
        assert result.turn_count == 1
        assert result.turns[0].status == TurnStatus.IDENTICAL

    def test_prompt_without_response(self):
        left = ConversationCapture(session_id="s1")
        left.record_prompt("Q1")
        left.record_response("A1", model="m1")
        left.record_prompt("Q2")  # No response

        right = ConversationCapture(session_id="s2")
        right.record_prompt("Q1")
        right.record_response("A1", model="m2")
        right.record_prompt("Q2")
        right.record_response("A2", model="m2")

        result = diff(left, right)
        assert result.turn_count == 2
        assert result.turns[0].status == TurnStatus.IDENTICAL
        # Left has no response for Q2, right does
        assert result.turns[1].left_response == ""
        assert result.turns[1].right_response == "A2"
