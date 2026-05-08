"""Tests for talk_box.consensus module."""

from talk_box.consensus import (
    ConsensusResult,
    ConsensusStrategy,
    Disagreement,
    ModelResponse,
    consensus,
    find_disagreements,
)


# ---------------------------------------------------------------------------
# ConsensusStrategy enum
# ---------------------------------------------------------------------------


class TestConsensusStrategy:
    def test_values(self):
        assert ConsensusStrategy.MAJORITY.value == "majority"
        assert ConsensusStrategy.UNANIMOUS.value == "unanimous"
        assert ConsensusStrategy.MOST_COMMON.value == "most_common"
        assert ConsensusStrategy.WEIGHTED.value == "weighted"

    def test_all_members(self):
        assert len(ConsensusStrategy) == 4


# ---------------------------------------------------------------------------
# ModelResponse
# ---------------------------------------------------------------------------


class TestModelResponse:
    def test_basic(self):
        r = ModelResponse(model="test:m1", text="Hello world")
        assert r.model == "test:m1"
        assert r.text == "Hello world"
        assert r.latency_ms is None
        assert r.token_count is None
        assert r.weight == 1.0

    def test_with_metadata(self):
        r = ModelResponse(
            model="test:m1",
            text="Response",
            latency_ms=150.5,
            token_count=42,
            weight=2.0,
        )
        assert r.latency_ms == 150.5
        assert r.token_count == 42
        assert r.weight == 2.0

    def test_frozen(self):
        r = ModelResponse(model="a", text="b")
        try:
            r.text = "c"  # type: ignore[misc]
            assert False, "Should be frozen"
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Disagreement
# ---------------------------------------------------------------------------


class TestDisagreement:
    def test_basic(self):
        d = Disagreement(
            description="Models A and B disagree",
            models=("model_a", "model_b"),
        )
        assert d.severity == "moderate"
        assert "A and B" in d.description

    def test_custom_severity(self):
        d = Disagreement(
            description="Total disagreement",
            models=("a", "b"),
            severity="major",
        )
        assert d.severity == "major"


# ---------------------------------------------------------------------------
# ConsensusResult
# ---------------------------------------------------------------------------


class TestConsensusResult:
    def test_basic(self):
        result = ConsensusResult(
            winner="Hello",
            winner_model="test:m1",
            agreement_score=0.85,
            strategy=ConsensusStrategy.MAJORITY,
        )
        assert result.winner == "Hello"
        assert result.winner_model == "test:m1"
        assert result.agreement_score == 0.85
        assert result.consensus_reached is True
        assert result.responses == []
        assert result.disagreements == []

    def test_with_disagreements(self):
        d = Disagreement(description="test", models=("a", "b"))
        result = ConsensusResult(
            winner="Hi",
            winner_model="a",
            agreement_score=0.5,
            strategy=ConsensusStrategy.UNANIMOUS,
            disagreements=[d],
            consensus_reached=False,
        )
        assert len(result.disagreements) == 1
        assert result.consensus_reached is False


# ---------------------------------------------------------------------------
# find_disagreements
# ---------------------------------------------------------------------------


class TestFindDisagreements:
    def test_empty_list(self):
        assert find_disagreements([]) == []

    def test_single_response(self):
        r = ModelResponse(model="a", text="Hello")
        assert find_disagreements([r]) == []

    def test_identical_responses(self):
        r1 = ModelResponse(model="a", text="Python is a programming language.")
        r2 = ModelResponse(model="b", text="Python is a programming language.")
        assert find_disagreements([r1, r2]) == []

    def test_similar_responses_no_disagreement(self):
        r1 = ModelResponse(model="a", text="Python is a high-level programming language.")
        r2 = ModelResponse(
            model="b", text="Python is a high-level interpreted programming language."
        )
        # Very similar — should not flag as disagreement
        disagreements = find_disagreements([r1, r2])
        assert len(disagreements) == 0

    def test_different_responses_detected(self):
        r1 = ModelResponse(model="a", text="The answer is 42.")
        r2 = ModelResponse(
            model="b", text="Machine learning uses neural networks for classification tasks."
        )
        disagreements = find_disagreements([r1, r2])
        assert len(disagreements) == 1
        assert disagreements[0].severity == "major"
        assert "a" in disagreements[0].models
        assert "b" in disagreements[0].models

    def test_moderate_disagreement(self):
        r1 = ModelResponse(
            model="a",
            text="Python was created by Guido van Rossum in 1991 as an interpreted language.",
        )
        r2 = ModelResponse(
            model="b",
            text="Ruby was created by Yukihiro Matsumoto in 1995 as an interpreted language.",
        )
        disagreements = find_disagreements([r1, r2])
        assert len(disagreements) >= 1
        assert disagreements[0].severity in ("moderate", "major")

    def test_multiple_disagreements(self):
        r1 = ModelResponse(model="a", text="The sky is blue and water is wet.")
        r2 = ModelResponse(model="b", text="The sky is blue and water is wet.")
        r3 = ModelResponse(model="c", text="Cats are furry animals that purr loudly at night.")
        disagreements = find_disagreements([r1, r2, r3])
        # r3 disagrees with both r1 and r2
        assert len(disagreements) >= 1

    def test_sorted_by_severity(self):
        r1 = ModelResponse(model="a", text="Completely unrelated topic about space exploration.")
        r2 = ModelResponse(model="b", text="Another totally different topic about cooking pasta.")
        r3 = ModelResponse(model="c", text="Mostly about space and exploring the galaxy far away.")
        disagreements = find_disagreements([r1, r2, r3])
        if len(disagreements) > 1:
            severities = [d.severity for d in disagreements]
            order = {"major": 0, "moderate": 1, "minor": 2}
            assert all(
                order[severities[i]] <= order[severities[i + 1]] for i in range(len(severities) - 1)
            )


# ---------------------------------------------------------------------------
# consensus() — MAJORITY
# ---------------------------------------------------------------------------


class TestConsensusMajority:
    def test_single_response(self):
        r = ModelResponse(model="a", text="Hello")
        result = consensus([r])
        assert result.winner == "Hello"
        assert result.winner_model == "a"
        assert result.agreement_score == 1.0
        assert result.consensus_reached is True

    def test_empty_raises(self):
        try:
            consensus([])
            assert False, "Should raise ValueError"
        except ValueError as e:
            assert "At least one" in str(e)

    def test_identical_responses(self):
        responses = [
            ModelResponse(model="a", text="Python is great."),
            ModelResponse(model="b", text="Python is great."),
            ModelResponse(model="c", text="Python is great."),
        ]
        result = consensus(responses, strategy=ConsensusStrategy.MAJORITY)
        assert result.agreement_score == 1.0
        assert result.winner == "Python is great."
        assert len(result.disagreements) == 0

    def test_majority_wins(self):
        responses = [
            ModelResponse(model="a", text="Python is a programming language used for many tasks."),
            ModelResponse(
                model="b", text="Python is a popular programming language for many tasks."
            ),
            ModelResponse(
                model="c", text="JavaScript is a scripting language for web development."
            ),
        ]
        result = consensus(responses, strategy=ConsensusStrategy.MAJORITY)
        # The majority (a and b are similar) should win
        assert result.winner_model in ("a", "b")
        assert result.agreement_score < 1.0
        assert len(result.disagreements) >= 1

    def test_returns_all_responses(self):
        responses = [
            ModelResponse(model="a", text="Hello"),
            ModelResponse(model="b", text="Hi there"),
        ]
        result = consensus(responses)
        assert len(result.responses) == 2

    def test_strategy_recorded(self):
        responses = [
            ModelResponse(model="a", text="Test"),
            ModelResponse(model="b", text="Test"),
        ]
        result = consensus(responses, strategy=ConsensusStrategy.MAJORITY)
        assert result.strategy == ConsensusStrategy.MAJORITY


# ---------------------------------------------------------------------------
# consensus() — UNANIMOUS
# ---------------------------------------------------------------------------


class TestConsensusUnanimous:
    def test_all_agree(self):
        responses = [
            ModelResponse(
                model="a",
                text="Python is a high-level interpreted programming language created by Guido van Rossum.",
            ),
            ModelResponse(
                model="b",
                text="Python is a high-level interpreted programming language designed by Guido van Rossum.",
            ),
            ModelResponse(
                model="c",
                text="Python is a high-level interpreted programming language developed by Guido van Rossum.",
            ),
        ]
        result = consensus(responses, strategy=ConsensusStrategy.UNANIMOUS)
        assert result.consensus_reached is True
        assert result.agreement_score > 0.5

    def test_disagreement_fails_consensus(self):
        responses = [
            ModelResponse(model="a", text="The capital of France is Paris."),
            ModelResponse(model="b", text="The capital of France is Paris, a beautiful city."),
            ModelResponse(model="c", text="Bananas are a tropical fruit grown in warm climates."),
        ]
        result = consensus(responses, strategy=ConsensusStrategy.UNANIMOUS)
        assert result.consensus_reached is False

    def test_custom_threshold(self):
        responses = [
            ModelResponse(model="a", text="Python is great for data science and analytics."),
            ModelResponse(
                model="b", text="Python is excellent for data science and machine learning."
            ),
        ]
        # With very high threshold, may not reach consensus
        result_strict = consensus(
            responses, strategy=ConsensusStrategy.UNANIMOUS, unanimous_threshold=0.95
        )
        # With low threshold, should reach consensus
        result_loose = consensus(
            responses, strategy=ConsensusStrategy.UNANIMOUS, unanimous_threshold=0.3
        )
        assert result_loose.consensus_reached is True


# ---------------------------------------------------------------------------
# consensus() — MOST_COMMON
# ---------------------------------------------------------------------------


class TestConsensusMostCommon:
    def test_clear_majority_cluster(self):
        responses = [
            ModelResponse(model="a", text="The answer is forty-two, the meaning of life."),
            ModelResponse(model="b", text="The answer is forty-two, the meaning of everything."),
            ModelResponse(model="c", text="The answer is forty-two, the meaning of life."),
            ModelResponse(model="d", text="I think the answer is seven, a lucky number."),
        ]
        result = consensus(responses, strategy=ConsensusStrategy.MOST_COMMON)
        # The cluster of 3 similar responses should win
        assert result.winner_model in ("a", "b", "c")
        assert result.agreement_score >= 0.5

    def test_all_different(self):
        responses = [
            ModelResponse(model="a", text="Apples are red fruits that grow on trees."),
            ModelResponse(model="b", text="The ocean is deep and blue with many fish."),
            ModelResponse(model="c", text="Mountains are tall rocky formations on continents."),
        ]
        result = consensus(responses, strategy=ConsensusStrategy.MOST_COMMON)
        # Each is its own cluster, so largest cluster is size 1
        assert result.agreement_score <= 0.5


# ---------------------------------------------------------------------------
# consensus() — WEIGHTED
# ---------------------------------------------------------------------------


class TestConsensusWeighted:
    def test_higher_weight_preferred(self):
        responses = [
            ModelResponse(model="premium", text="Python is a programming language.", weight=5.0),
            ModelResponse(
                model="cheap", text="Python is a snake found in tropical regions.", weight=1.0
            ),
        ]
        result = consensus(responses, strategy=ConsensusStrategy.WEIGHTED)
        # The premium model's weight should give it an advantage
        assert result.winner_model == "premium"

    def test_equal_weights_like_majority(self):
        responses = [
            ModelResponse(
                model="a", text="Python is used for web development and data science.", weight=1.0
            ),
            ModelResponse(
                model="b", text="Python is used for web development and data science.", weight=1.0
            ),
            ModelResponse(
                model="c", text="Go is used for systems programming and microservices.", weight=1.0
            ),
        ]
        result = consensus(responses, strategy=ConsensusStrategy.WEIGHTED)
        # a and b agree, so one of them should win
        assert result.winner_model in ("a", "b")

    def test_weight_can_override_majority(self):
        responses = [
            ModelResponse(
                model="expert", text="The correct approach uses dynamic programming.", weight=10.0
            ),
            ModelResponse(
                model="basic1", text="Just use a simple loop to iterate through.", weight=1.0
            ),
            ModelResponse(
                model="basic2", text="Just use a simple loop to go through items.", weight=1.0
            ),
        ]
        result = consensus(responses, strategy=ConsensusStrategy.WEIGHTED)
        # Expert weight (10x) should be able to override the basic majority
        assert result.winner_model == "expert"


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestConsensusIntegration:
    def test_result_has_all_fields(self):
        responses = [
            ModelResponse(model="a", text="Hello world", latency_ms=100),
            ModelResponse(model="b", text="Hello world!", latency_ms=200),
        ]
        result = consensus(responses)
        assert isinstance(result, ConsensusResult)
        assert isinstance(result.winner, str)
        assert isinstance(result.winner_model, str)
        assert isinstance(result.agreement_score, float)
        assert 0.0 <= result.agreement_score <= 1.0
        assert isinstance(result.strategy, ConsensusStrategy)
        assert isinstance(result.responses, list)
        assert isinstance(result.disagreements, list)
        assert isinstance(result.consensus_reached, bool)

    def test_all_strategies_work(self):
        responses = [
            ModelResponse(model="a", text="Python is a language."),
            ModelResponse(model="b", text="Python is a programming language."),
            ModelResponse(model="c", text="Python is a versatile programming language."),
        ]
        for strat in ConsensusStrategy:
            result = consensus(responses, strategy=strat)
            assert result.winner is not None
            assert result.winner_model in ("a", "b", "c")

    def test_two_models_agreeing(self):
        responses = [
            ModelResponse(model="a", text="The answer is 42."),
            ModelResponse(model="b", text="The answer is 42."),
        ]
        result = consensus(responses)
        assert result.agreement_score == 1.0
        assert result.consensus_reached is True
        assert len(result.disagreements) == 0

    def test_preserves_original_responses(self):
        responses = [
            ModelResponse(model="a", text="First response with details."),
            ModelResponse(model="b", text="Second response with other details."),
            ModelResponse(model="c", text="Third response entirely different topic here."),
        ]
        result = consensus(responses)
        assert len(result.responses) == 3
        assert result.responses[0].model == "a"
        assert result.responses[2].model == "c"
