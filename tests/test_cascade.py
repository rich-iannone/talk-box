"""Tests for talk_box.cascade module."""

from talk_box.cascade import (
    CascadeResult,
    CascadeRound,
    cascade,
    estimate_confidence,
)
from talk_box.consensus import ConsensusStrategy
from talk_box.routing import RoutingStrategy, TaskComplexity


# ---------------------------------------------------------------------------
# CascadeRound
# ---------------------------------------------------------------------------


class TestCascadeRound:
    def test_basic(self):
        r = CascadeRound(round_number=1, model="test:m1", response="Hello", confidence=0.8)
        assert r.round_number == 1
        assert r.model == "test:m1"
        assert r.response == "Hello"
        assert r.confidence == 0.8

    def test_frozen(self):
        r = CascadeRound(round_number=1, model="a", response="b", confidence=0.5)
        try:
            r.confidence = 0.9  # type: ignore[misc]
            assert False, "Should be frozen"
        except Exception:
            pass


# ---------------------------------------------------------------------------
# CascadeResult
# ---------------------------------------------------------------------------


class TestCascadeResult:
    def test_defaults(self):
        result = CascadeResult(
            winner="Hello",
            winner_model="test:m1",
            confidence=0.8,
            fanned_out=False,
        )
        assert result.winner == "Hello"
        assert result.winner_model == "test:m1"
        assert result.confidence == 0.8
        assert result.fanned_out is False
        assert result.rounds == []
        assert result.consensus_strategy is None
        assert result.agreement_score is None
        assert result.initial_model == ""
        assert result.initial_confidence == 0.0
        assert result.complexity == TaskComplexity.SIMPLE

    def test_with_fan_out(self):
        result = CascadeResult(
            winner="Answer",
            winner_model="a:b",
            confidence=0.7,
            fanned_out=True,
            consensus_strategy=ConsensusStrategy.MAJORITY,
            agreement_score=0.85,
            initial_model="a:b",
            initial_confidence=0.4,
            complexity=TaskComplexity.COMPLEX,
        )
        assert result.fanned_out is True
        assert result.consensus_strategy == ConsensusStrategy.MAJORITY
        assert result.agreement_score == 0.85


# ---------------------------------------------------------------------------
# estimate_confidence
# ---------------------------------------------------------------------------


class TestEstimateConfidence:
    def test_empty_string(self):
        assert estimate_confidence("") == 0.0

    def test_whitespace_only(self):
        assert estimate_confidence("   ") == 0.0

    def test_assertive_response(self):
        score = estimate_confidence("The answer is 42. This is because the question is clear.")
        assert score > 0.5

    def test_hedging_response(self):
        score = estimate_confidence("I'm not sure, but maybe it could be 42? I think so, perhaps.")
        assert score < 0.5

    def test_refusal_response(self):
        score = estimate_confidence("I cannot answer that question. As an AI, I'm unable to help.")
        assert score < 0.3

    def test_code_block_increases_confidence(self):
        plain = estimate_confidence("Use a for loop to iterate.")
        with_code = estimate_confidence(
            "Use a for loop:\n```python\nfor i in range(10):\n    print(i)\n```"
        )
        assert with_code > plain

    def test_detailed_response_higher_than_short(self):
        short = estimate_confidence("Yes.")
        detailed = estimate_confidence(
            "Yes, Python is a programming language. It was created by Guido van Rossum "
            "in 1991. It is known for its clean syntax and extensive standard library. "
            "Python supports multiple programming paradigms including object-oriented, "
            "functional, and procedural styles."
        )
        assert detailed > short

    def test_structured_list_increases_confidence(self):
        plain = estimate_confidence("There are several benefits of Python.")
        with_list = estimate_confidence(
            "There are several benefits of Python:\n"
            "- Clean syntax\n"
            "- Rich ecosystem\n"
            "- Strong community\n"
            "- Great documentation\n"
        )
        assert with_list >= plain

    def test_confidence_clamped_to_range(self):
        # Very hedging response
        score = estimate_confidence(
            "I'm not sure, maybe, perhaps, possibly, I think, I believe, "
            "it depends, it's unclear, hard to say, I don't know"
        )
        assert 0.0 <= score <= 1.0

    def test_confidence_upper_bound(self):
        score = estimate_confidence(
            "The answer is definitely correct. The result is clearly valid. "
            "This is because the solution is specifically designed. "
            "Certainly this is obvious.\n```python\nx = 42\n```\n"
            "Here is a detailed explanation with many words to push the length up "
            "and demonstrate a thorough, well-considered response that covers "
            "all the relevant aspects of the question in great detail."
        )
        assert 0.0 <= score <= 1.0

    def test_numbered_list(self):
        score = estimate_confidence(
            "Follow these steps:\n"
            "1. Install Python\n"
            "2. Create a virtual environment\n"
            "3. Install dependencies\n"
            "4. Run the tests\n"
        )
        assert score > 0.5


# ---------------------------------------------------------------------------
# cascade — no fan-out (confident initial response)
# ---------------------------------------------------------------------------


class TestCascadeNoFanOut:
    def _confident_responder(self, model_key: str, prompt: str) -> str:
        return "The answer is 42. This is because the question has a clear, definitive answer."

    def test_returns_immediately_when_confident(self):
        result = cascade("What is 6 * 7?", self._confident_responder)
        assert result.fanned_out is False
        assert result.confidence >= 0.6
        assert "42" in result.winner
        assert len(result.rounds) == 1
        assert result.rounds[0].round_number == 1

    def test_initial_model_recorded(self):
        result = cascade("What is Python?", self._confident_responder)
        assert result.initial_model != ""
        assert result.winner_model == result.initial_model

    def test_no_consensus_fields(self):
        result = cascade("Simple question", self._confident_responder)
        assert result.consensus_strategy is None
        assert result.agreement_score is None

    def test_complexity_recorded(self):
        result = cascade("What is Python?", self._confident_responder)
        assert isinstance(result.complexity, TaskComplexity)


# ---------------------------------------------------------------------------
# cascade — fan-out (low confidence initial response)
# ---------------------------------------------------------------------------


class TestCascadeFanOut:
    def _hedging_responder(self, model_key: str, prompt: str) -> str:
        """Responder that always gives hedging answers."""
        return f"I'm not sure, but maybe the answer is related to {model_key}. Perhaps it depends."

    def _mixed_responder(self, model_key: str, prompt: str) -> str:
        """First call hedges, subsequent calls are confident."""
        if not hasattr(self, "_call_count"):
            self._call_count = 0
        self._call_count += 1
        if self._call_count == 1:
            return "I think maybe it's something. I'm not sure, perhaps it depends."
        return "The answer is definitely Python. This is because Python is a programming language."

    def test_fans_out_when_confidence_low(self):
        result = cascade(
            "Explain quantum computing",
            self._hedging_responder,
            confidence_threshold=0.8,
        )
        assert result.fanned_out is True
        assert len(result.rounds) > 1

    def test_consensus_fields_populated(self):
        result = cascade(
            "Complex question",
            self._hedging_responder,
            confidence_threshold=0.8,
        )
        assert result.consensus_strategy is not None
        assert result.agreement_score is not None

    def test_max_fan_out_respected(self):
        result = cascade(
            "Something",
            self._hedging_responder,
            confidence_threshold=0.8,
            max_fan_out=2,
        )
        # Initial + max_fan_out = at most 3 rounds
        assert len(result.rounds) <= 3

    def test_fan_out_strategy_used(self):
        result = cascade(
            "Test question",
            self._hedging_responder,
            confidence_threshold=0.8,
            fan_out_strategy=ConsensusStrategy.MOST_COMMON,
        )
        assert result.consensus_strategy == ConsensusStrategy.MOST_COMMON

    def test_initial_confidence_recorded(self):
        result = cascade(
            "Something uncertain",
            self._hedging_responder,
            confidence_threshold=0.8,
        )
        assert result.initial_confidence < 0.8
        assert result.initial_confidence > 0.0

    def test_all_rounds_have_responses(self):
        result = cascade(
            "Multi-round test",
            self._hedging_responder,
            confidence_threshold=0.8,
            max_fan_out=2,
        )
        for r in result.rounds:
            assert r.response != ""
            assert r.model != ""
            assert 0.0 <= r.confidence <= 1.0

    def test_mixed_confidence_triggers_fan_out(self):
        self._call_count = 0  # Reset counter
        result = cascade(
            "Something",
            self._mixed_responder,
            confidence_threshold=0.8,
        )
        assert result.fanned_out is True
        assert len(result.rounds) > 1


# ---------------------------------------------------------------------------
# cascade — configuration options
# ---------------------------------------------------------------------------


class TestCascadeConfig:
    def _confident_responder(self, model_key: str, prompt: str) -> str:
        return "The answer is clear and definitive. Certainly the result is correct."

    def _hedging_responder(self, model_key: str, prompt: str) -> str:
        return "I think maybe perhaps it could be something."

    def test_high_threshold_forces_fan_out(self):
        result = cascade(
            "Simple question",
            self._confident_responder,
            confidence_threshold=0.99,
        )
        # Even confident responses won't meet 0.99 threshold
        assert result.fanned_out is True

    def test_low_threshold_prevents_fan_out(self):
        result = cascade(
            "Question",
            self._hedging_responder,
            confidence_threshold=0.05,
        )
        # Even hedging responses will meet 0.05 threshold
        assert result.fanned_out is False

    def test_routing_strategy_parameter(self):
        result = cascade(
            "Define polymorphism",
            self._confident_responder,
            routing_strategy=RoutingStrategy.COST_OPTIMIZED,
        )
        assert isinstance(result, CascadeResult)

    def test_max_fan_out_one(self):
        result = cascade(
            "Test",
            self._hedging_responder,
            confidence_threshold=0.99,
            max_fan_out=1,
        )
        assert len(result.rounds) <= 2


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestCascadeIntegration:
    def test_result_has_all_fields(self):
        def responder(model_key: str, prompt: str) -> str:
            return "The answer is Python. This is because Python is widely used."

        result = cascade("What language?", responder)
        assert isinstance(result, CascadeResult)
        assert isinstance(result.winner, str)
        assert isinstance(result.winner_model, str)
        assert isinstance(result.confidence, float)
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.fanned_out, bool)
        assert isinstance(result.rounds, list)
        assert isinstance(result.complexity, TaskComplexity)

    def test_confidence_in_valid_range(self):
        def responder(model_key: str, prompt: str) -> str:
            return "Maybe I think perhaps it depends."

        result = cascade("Test", responder, confidence_threshold=0.99)
        assert 0.0 <= result.confidence <= 1.0

    def test_round_numbers_sequential(self):
        def responder(model_key: str, prompt: str) -> str:
            return "I'm not sure, maybe, perhaps. I think it depends on context."

        result = cascade("Test", responder, confidence_threshold=0.99, max_fan_out=3)
        for i, r in enumerate(result.rounds):
            assert r.round_number == i + 1
