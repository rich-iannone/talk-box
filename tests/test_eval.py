import pytest

from talk_box.builder import ChatBot
from talk_box.eval import (
    DEFAULT_DIMENSIONS,
    EvalCase,
    EvalDimension,
    EvalResult,
    EvalResults,
    EvalScore,
    _build_judge_prompt,
    _get_persona_context,
    _parse_judge_response,
    _resolve_judge,
    _resolve_queries,
    eval,
    eval_regression,
    eval_suite,
    scorecard_table,
    sweep_table,
)


# ---------------------------------------------------------------------------
# EvalCase tests
# ---------------------------------------------------------------------------


class TestEvalCase:
    def test_basic_creation(self):
        case = EvalCase(query="How do I sort a list?")
        assert case.query == "How do I sort a list?"
        assert case.context == ""
        assert case.tags == ()

    def test_with_context_and_tags(self):
        case = EvalCase(
            query="Review this function",
            context="Should mention error handling",
            tags=("security", "python"),
        )
        assert case.context == "Should mention error handling"
        assert case.tags == ("security", "python")

    def test_frozen(self):
        case = EvalCase(query="test")
        with pytest.raises(AttributeError):
            case.query = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# EvalScore tests
# ---------------------------------------------------------------------------


class TestEvalScore:
    def test_creation(self):
        score = EvalScore(
            dimension=EvalDimension.RELEVANCE,
            score=0.9,
            explanation="Directly answers the question",
        )
        assert score.dimension == EvalDimension.RELEVANCE
        assert score.score == 0.9
        assert score.explanation == "Directly answers the question"

    def test_frozen(self):
        score = EvalScore(dimension=EvalDimension.SAFETY, score=1.0)
        with pytest.raises(AttributeError):
            score.score = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# EvalResult tests
# ---------------------------------------------------------------------------


class TestEvalResult:
    def test_avg_score_empty(self):
        result = EvalResult(variant="test", query="q", response="r")
        assert result.avg_score == 0.0

    def test_avg_score(self):
        result = EvalResult(
            variant="test",
            query="q",
            response="r",
            scores=[
                EvalScore(dimension=EvalDimension.RELEVANCE, score=0.8),
                EvalScore(dimension=EvalDimension.SAFETY, score=1.0),
                EvalScore(dimension=EvalDimension.TONE, score=0.6),
            ],
        )
        assert result.avg_score == pytest.approx(0.8)

    def test_score_for(self):
        result = EvalResult(
            variant="test",
            query="q",
            response="r",
            scores=[
                EvalScore(dimension=EvalDimension.RELEVANCE, score=0.9),
                EvalScore(dimension=EvalDimension.SAFETY, score=1.0),
            ],
        )
        assert result.score_for(EvalDimension.RELEVANCE) == 0.9
        assert result.score_for(EvalDimension.SAFETY) == 1.0
        assert result.score_for(EvalDimension.TONE) is None


# ---------------------------------------------------------------------------
# EvalResults tests
# ---------------------------------------------------------------------------


class TestEvalResults:
    @pytest.fixture
    def sample_results(self):
        return EvalResults(
            results=[
                EvalResult(
                    variant="baseline",
                    query="q1",
                    response="r1",
                    scores=[
                        EvalScore(dimension=EvalDimension.RELEVANCE, score=0.9),
                        EvalScore(dimension=EvalDimension.SAFETY, score=1.0),
                    ],
                ),
                EvalResult(
                    variant="baseline",
                    query="q2",
                    response="r2",
                    scores=[
                        EvalScore(dimension=EvalDimension.RELEVANCE, score=0.8),
                        EvalScore(dimension=EvalDimension.SAFETY, score=0.9),
                    ],
                ),
                EvalResult(
                    variant="stricter",
                    query="q1",
                    response="r3",
                    scores=[
                        EvalScore(dimension=EvalDimension.RELEVANCE, score=0.7),
                        EvalScore(dimension=EvalDimension.SAFETY, score=1.0),
                    ],
                ),
                EvalResult(
                    variant="stricter",
                    query="q2",
                    response="r4",
                    scores=[
                        EvalScore(dimension=EvalDimension.RELEVANCE, score=0.6),
                        EvalScore(dimension=EvalDimension.SAFETY, score=0.95),
                    ],
                ),
            ]
        )

    def test_variants(self, sample_results):
        assert sample_results.variants == ["baseline", "stricter"]

    def test_dimensions(self, sample_results):
        dims = sample_results.dimensions
        assert EvalDimension.RELEVANCE in dims
        assert EvalDimension.SAFETY in dims

    def test_scores_by_variant(self, sample_results):
        by_variant = sample_results.scores_by_variant()
        assert by_variant["baseline"]["relevance"] == pytest.approx(0.85)
        assert by_variant["baseline"]["safety"] == pytest.approx(0.95)
        assert by_variant["stricter"]["relevance"] == pytest.approx(0.65)
        assert by_variant["stricter"]["safety"] == pytest.approx(0.975)

    def test_summary(self, sample_results):
        summary = sample_results.summary()
        assert summary["total_queries"] == 2
        assert summary["total_results"] == 4
        assert summary["variants"] == ["baseline", "stricter"]
        assert "relevance" in summary["dimensions"]
        assert "safety" in summary["dimensions"]
        assert "baseline" in summary["overall_scores"]
        assert "stricter" in summary["overall_scores"]

    def test_passed_true(self, sample_results):
        assert sample_results.passed(threshold=0.5) is True

    def test_passed_false(self, sample_results):
        # Stricter variant has relevance=0.65, so overall is below 0.9
        assert sample_results.passed(threshold=0.9) is False

    def test_regressions(self, sample_results):
        regs = sample_results.regressions(baseline="baseline", threshold=0.05)
        assert "stricter" in regs
        assert "relevance" in regs["stricter"]
        # -0.2 drop in relevance
        assert regs["stricter"]["relevance"] == pytest.approx(-0.2)
        # Safety improved, so not in regressions
        assert "safety" not in regs["stricter"]

    def test_regressions_no_baseline(self, sample_results):
        # Uses first variant as baseline by default
        regs = sample_results.regressions()
        assert "stricter" in regs

    def test_regressions_empty(self):
        results = EvalResults(results=[])
        assert results.regressions() == {}

    def test_len(self, sample_results):
        assert len(sample_results) == 4

    def test_iter(self, sample_results):
        items = list(sample_results)
        assert len(items) == 4
        assert all(isinstance(r, EvalResult) for r in items)

    def test_getitem(self, sample_results):
        assert sample_results[0].variant == "baseline"
        assert sample_results[2].variant == "stricter"

    def test_to_dataframe(self, sample_results):
        pytest.importorskip("pandas")
        df = sample_results.to_dataframe()
        assert len(df) == 8  # 4 results × 2 dimensions each
        assert "variant" in df.columns
        assert "query" in df.columns
        assert "dimension" in df.columns
        assert "score" in df.columns

    def test_to_great_table(self, sample_results):
        pytest.importorskip("great_tables")
        pytest.importorskip("pandas")
        table = sample_results.to_great_table()
        # Just verify it returns a GT object without error
        assert table is not None

    def test_to_great_table_empty(self):
        pytest.importorskip("great_tables")
        pytest.importorskip("pandas")
        results = EvalResults(results=[])
        table = results.to_great_table()
        assert table is not None


# ---------------------------------------------------------------------------
# Judge prompt and parsing tests
# ---------------------------------------------------------------------------


class TestJudgePrompt:
    def test_build_judge_prompt(self):
        prompt = _build_judge_prompt(
            query="What is Python?",
            response="Python is a programming language.",
            dimensions=[EvalDimension.RELEVANCE, EvalDimension.SAFETY],
            persona_context="Persona: Python Tutor",
        )
        assert "What is Python?" in prompt
        assert "Python is a programming language." in prompt
        assert "relevance" in prompt
        assert "safety" in prompt
        assert "Python Tutor" in prompt

    def test_build_judge_prompt_no_context(self):
        prompt = _build_judge_prompt(
            query="hi",
            response="hello",
            dimensions=[EvalDimension.RELEVANCE],
        )
        assert "No persona context provided" in prompt


class TestParseJudgeResponse:
    def test_parse_standard_format(self):
        response = """relevance: 0.85 | Directly addresses the question
safety: 1.0 | No harmful content
instruction_adherence: 0.7 | Mostly follows instructions"""

        scores = _parse_judge_response(
            response,
            [
                EvalDimension.RELEVANCE,
                EvalDimension.SAFETY,
                EvalDimension.INSTRUCTION_ADHERENCE,
            ],
        )
        assert len(scores) == 3
        assert scores[0].dimension == EvalDimension.RELEVANCE
        assert scores[0].score == 0.85
        assert scores[0].explanation == "Directly addresses the question"
        assert scores[1].score == 1.0
        assert scores[2].score == 0.7

    def test_parse_without_explanation(self):
        response = "relevance: 0.9\nsafety: 1.0"
        scores = _parse_judge_response(response, [EvalDimension.RELEVANCE, EvalDimension.SAFETY])
        assert len(scores) == 2
        assert scores[0].score == 0.9
        assert scores[0].explanation == ""

    def test_parse_clamps_scores(self):
        response = "relevance: 1.5 | Too high\nsafety: -0.3 | Too low"
        scores = _parse_judge_response(response, [EvalDimension.RELEVANCE, EvalDimension.SAFETY])
        assert scores[0].score == 1.0
        assert scores[1].score == 0.0

    def test_parse_missing_dimensions_get_default(self):
        response = "relevance: 0.8 | Good"
        scores = _parse_judge_response(
            response,
            [EvalDimension.RELEVANCE, EvalDimension.SAFETY],
        )
        assert len(scores) == 2
        # Relevance parsed correctly
        relevance = next(s for s in scores if s.dimension == EvalDimension.RELEVANCE)
        assert relevance.score == 0.8
        # Safety gets default 0.5
        safety = next(s for s in scores if s.dimension == EvalDimension.SAFETY)
        assert safety.score == 0.5
        assert "Could not parse" in safety.explanation

    def test_parse_garbage_input(self):
        response = "This is not structured output at all."
        scores = _parse_judge_response(response, [EvalDimension.RELEVANCE])
        assert len(scores) == 1
        assert scores[0].score == 0.5

    def test_parse_invalid_score_value(self):
        response = "relevance: not_a_number | Bad format"
        scores = _parse_judge_response(response, [EvalDimension.RELEVANCE])
        assert scores[0].score == 0.5


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_resolve_judge_none(self):
        judge = _resolve_judge(None)
        assert isinstance(judge, ChatBot)
        assert judge._config["temperature"] == 0.1

    def test_resolve_judge_string(self):
        judge = _resolve_judge("openai:gpt-4o")
        assert isinstance(judge, ChatBot)
        assert judge._config["provider"] == "openai"
        assert judge._config["model"] == "gpt-4o"
        assert judge._config["temperature"] == 0.1

    def test_resolve_judge_chatbot(self):
        custom = ChatBot(name="My Judge").temperature(0.2)
        judge = _resolve_judge(custom)
        assert judge is custom

    def test_resolve_queries_explicit(self):
        queries = ["q1", "q2"]
        result = _resolve_queries(queries, {"v": ChatBot()})
        assert result == ["q1", "q2"]

    def test_resolve_queries_from_persona(self):
        from talk_box.personas import get_persona

        persona = get_persona("code_reviewer")
        bot = ChatBot().persona_pack("code_reviewer")
        result = _resolve_queries(None, {"default": bot})
        assert len(result) > 0
        assert result == persona.test_queries

    def test_resolve_queries_no_persona(self):
        bot = ChatBot()
        result = _resolve_queries(None, {"default": bot})
        assert result == []

    def test_get_persona_context_with_persona(self):
        bot = ChatBot().persona_pack("code_reviewer")
        context = _get_persona_context(bot)
        assert "Code Reviewer" in context
        assert "code review" in context.lower()

    def test_get_persona_context_no_persona(self):
        bot = ChatBot()
        context = _get_persona_context(bot)
        assert context == ""

    def test_get_persona_context_with_system_prompt(self):
        bot = ChatBot().system_prompt("You are a helpful assistant.")
        context = _get_persona_context(bot)
        assert "helpful assistant" in context


# ---------------------------------------------------------------------------
# Integration tests (with mock_responses)
# ---------------------------------------------------------------------------


class TestEvalIntegration:
    def test_eval_single_bot(self):
        """Eval a single bot with mock responses and mock judge."""
        bot = ChatBot().mock_responses(["Python is a programming language."])

        judge = ChatBot(name="Judge").mock_responses(
            [
                "relevance: 0.9 | Good\nsafety: 1.0 | Safe\n"
                "instruction_adherence: 0.8 | Mostly follows"
            ]
        )

        results = eval(
            bot,
            queries=["What is Python?"],
            judge=judge,
        )

        assert len(results) == 1
        assert results[0].variant == "default"
        assert results[0].query == "What is Python?"
        assert results[0].response == "Python is a programming language."
        assert len(results[0].scores) == 3
        assert results[0].score_for(EvalDimension.RELEVANCE) == 0.9

    def test_eval_multiple_variants(self):
        """Eval two variants against the same queries."""
        bot_a = ChatBot().mock_responses(["Response A1", "Response A2"])
        bot_b = ChatBot().mock_responses(["Response B1", "Response B2"])

        judge = ChatBot(name="Judge").mock_responses(
            [
                "relevance: 0.9 | Good\nsafety: 1.0 | Safe\n"
                "instruction_adherence: 0.85 | Follows well",
                "relevance: 0.8 | Decent\nsafety: 1.0 | Safe\ninstruction_adherence: 0.7 | Mostly",
                "relevance: 0.7 | Partial\nsafety: 0.95 | Mostly safe\n"
                "instruction_adherence: 0.6 | Some gaps",
                "relevance: 0.6 | Weak\nsafety: 1.0 | Safe\ninstruction_adherence: 0.5 | Half",
            ]
        )

        results = eval(
            variants={"alpha": bot_a, "beta": bot_b},
            queries=["q1", "q2"],
            judge=judge,
        )

        assert len(results) == 4
        assert results.variants == ["alpha", "beta"]
        by_variant = results.scores_by_variant()
        assert "alpha" in by_variant
        assert "beta" in by_variant

    def test_eval_uses_persona_test_queries(self):
        """Eval falls back to persona test_queries when no queries given."""
        bot = (
            ChatBot()
            .persona_pack("code_reviewer")
            .mock_responses(
                ["Review feedback"] * 4  # One per test_query
            )
        )

        judge = ChatBot(name="Judge").mock_responses(
            ["relevance: 0.8 | Ok\nsafety: 1.0 | Fine\ninstruction_adherence: 0.7 | Good"] * 4
        )

        results = eval(bot, judge=judge)
        # code_reviewer has 4 test_queries
        assert len(results) == 4

    def test_eval_custom_dimensions(self):
        """Custom dimensions are used for scoring."""
        bot = ChatBot().mock_responses(["Hello!"])
        judge = ChatBot(name="Judge").mock_responses(
            ["tone: 0.95 | Professional\nconciseness: 0.8 | Brief"]
        )

        results = eval(
            bot,
            queries=["Hi"],
            dimensions=[EvalDimension.TONE, EvalDimension.CONCISENESS],
            judge=judge,
        )

        assert len(results[0].scores) == 2
        assert results[0].score_for(EvalDimension.TONE) == 0.95
        assert results[0].score_for(EvalDimension.CONCISENESS) == 0.8

    def test_eval_errors_both_bot_and_variants(self):
        bot = ChatBot()
        with pytest.raises(ValueError, match="not both"):
            eval(bot, variants={"v": bot})

    def test_eval_errors_neither_bot_nor_variants(self):
        with pytest.raises(ValueError, match="Provide either"):
            eval()

    def test_eval_errors_no_queries(self):
        bot = ChatBot()  # No persona, no test_queries
        judge = ChatBot(name="Judge")
        with pytest.raises(ValueError, match="No queries"):
            eval(bot, judge=judge)


class TestEvalRegression:
    def test_regression_detection(self):
        """Detect when 'after' variant drops in quality."""
        before = ChatBot().mock_responses(["Good answer"])
        after = ChatBot().mock_responses(["Worse answer"])

        judge = ChatBot(name="Judge").mock_responses(
            [
                # Score for 'before'
                "relevance: 0.9 | Excellent\nsafety: 1.0 | Safe\n"
                "instruction_adherence: 0.85 | Good",
                # Score for 'after'
                "relevance: 0.5 | Weak\nsafety: 1.0 | Safe\ninstruction_adherence: 0.4 | Poor",
            ]
        )

        results = eval_regression(
            before=before,
            after=after,
            queries=["test query"],
            judge=judge,
            threshold=0.05,
        )

        regs = results.regressions(baseline="before", threshold=0.05)
        assert "after" in regs
        assert "relevance" in regs["after"]
        assert "instruction_adherence" in regs["after"]
        # Safety stayed the same (1.0 both), not a regression
        assert "safety" not in regs["after"]

    def test_no_regression(self):
        """No regression when 'after' is same or better."""
        before = ChatBot().mock_responses(["Answer"])
        after = ChatBot().mock_responses(["Better answer"])

        judge = ChatBot(name="Judge").mock_responses(
            [
                "relevance: 0.8 | Good\nsafety: 1.0 | Safe\ninstruction_adherence: 0.7 | Ok",
                "relevance: 0.9 | Better\nsafety: 1.0 | Safe\n"
                "instruction_adherence: 0.8 | Improved",
            ]
        )

        results = eval_regression(
            before=before,
            after=after,
            queries=["test query"],
            judge=judge,
        )

        regs = results.regressions(baseline="before")
        assert regs == {}


# ---------------------------------------------------------------------------
# EvalDimension enum tests
# ---------------------------------------------------------------------------


class TestEvalDimension:
    def test_all_dimensions(self):
        assert len(EvalDimension) == 6
        assert EvalDimension.RELEVANCE.value == "relevance"
        assert EvalDimension.SAFETY.value == "safety"
        assert EvalDimension.INSTRUCTION_ADHERENCE.value == "instruction_adherence"
        assert EvalDimension.TONE.value == "tone"
        assert EvalDimension.COMPLETENESS.value == "completeness"
        assert EvalDimension.CONCISENESS.value == "conciseness"

    def test_default_dimensions(self):
        assert len(DEFAULT_DIMENSIONS) == 3
        assert EvalDimension.RELEVANCE in DEFAULT_DIMENSIONS
        assert EvalDimension.SAFETY in DEFAULT_DIMENSIONS
        assert EvalDimension.INSTRUCTION_ADHERENCE in DEFAULT_DIMENSIONS


# ---------------------------------------------------------------------------
# to_scorecard tests
# ---------------------------------------------------------------------------


class TestScorecard:
    def test_scorecard_basic(self):
        results = EvalResults(
            results=[
                EvalResult(
                    variant="model_a",
                    query="test?",
                    response="yes",
                    scores=[
                        EvalScore(EvalDimension.RELEVANCE, 0.9, "good"),
                        EvalScore(EvalDimension.SAFETY, 1.0, "safe"),
                    ],
                    duration=0.5,
                ),
            ],
            config={"variants": ["model_a"], "dimensions": ["relevance", "safety"]},
        )

        card = results.to_scorecard()
        assert "generated_at" in card
        assert "model_a" in card["variants"]
        assert card["variants"]["model_a"]["dimensions"]["relevance"] == 0.9
        assert card["variants"]["model_a"]["overall"] == 0.95
        assert card["variants"]["model_a"]["num_queries"] == 1

    def test_scorecard_multiple_variants(self):
        results = EvalResults(
            results=[
                EvalResult(
                    variant="v1",
                    query="q",
                    response="r1",
                    scores=[EvalScore(EvalDimension.RELEVANCE, 0.8, "ok")],
                    duration=0.1,
                ),
                EvalResult(
                    variant="v2",
                    query="q",
                    response="r2",
                    scores=[EvalScore(EvalDimension.RELEVANCE, 0.9, "good")],
                    duration=0.2,
                ),
            ],
            config={},
        )

        card = results.to_scorecard()
        assert len(card["variants"]) == 2
        assert card["variants"]["v1"]["dimensions"]["relevance"] == 0.8
        assert card["variants"]["v2"]["dimensions"]["relevance"] == 0.9

    def test_scorecard_write_to_file(self, tmp_path):
        results = EvalResults(
            results=[
                EvalResult(
                    variant="v1",
                    query="q",
                    response="r",
                    scores=[EvalScore(EvalDimension.SAFETY, 1.0, "safe")],
                    duration=0.1,
                ),
            ],
            config={},
        )

        out = tmp_path / "sub" / "scorecard.json"
        card = results.to_scorecard(path=out)

        assert out.exists()
        import json

        loaded = json.loads(out.read_text())
        assert loaded["variants"]["v1"]["dimensions"]["safety"] == 1.0
        assert card == loaded

    def test_scorecard_empty_results(self):
        results = EvalResults(results=[], config={})
        card = results.to_scorecard()
        assert card["variants"] == {}


# ---------------------------------------------------------------------------
# eval_suite tests
# ---------------------------------------------------------------------------


class TestEvalSuite:
    def test_eval_suite_basic(self):
        judge = ChatBot(name="Judge").mock_responses(
            [
                "relevance: 0.85 | Good\nsafety: 1.0 | Safe\n"
                "instruction_adherence: 0.80 | Follows instructions",
                "relevance: 0.90 | Great\nsafety: 1.0 | Safe\n"
                "instruction_adherence: 0.88 | Very good",
            ]
        )

        results = eval_suite(
            "code_reviewer",
            models=["openai:gpt-4o", "anthropic:claude-sonnet-4-6"],
            queries=["Review this function"],
            judge=judge,
        )

        assert len(results.variants) == 2
        assert "openai:gpt-4o" in results.variants
        assert "anthropic:claude-sonnet-4-6" in results.variants
        assert results.config["persona"] == "code_reviewer"
        assert results.config["type"] == "suite"

    def test_eval_suite_uses_persona_test_queries(self):
        # code_reviewer has test_queries defined
        judge = ChatBot(name="Judge").mock_responses(
            [
                "relevance: 0.85 | ok\nsafety: 1.0 | safe\ninstruction_adherence: 0.80 | ok",
            ]
        )

        results = eval_suite(
            "code_reviewer",
            models=["openai:gpt-4o"],
            judge=judge,
        )

        assert len(results) >= 1

    def test_eval_suite_empty_models(self):
        with pytest.raises(ValueError, match="At least one model"):
            eval_suite("code_reviewer", models=[])

    def test_eval_suite_scorecard_output(self, tmp_path):
        judge = ChatBot(name="Judge").mock_responses(
            [
                "relevance: 0.85 | Good\nsafety: 1.0 | Safe\ninstruction_adherence: 0.80 | ok",
            ]
        )

        out = tmp_path / "scorecard.json"
        results = eval_suite(
            "code_reviewer",
            models=["openai:gpt-4o"],
            queries=["Review this"],
            judge=judge,
            scorecard_path=out,
        )

        assert out.exists()
        import json

        loaded = json.loads(out.read_text())
        assert "openai:gpt-4o" in loaded["variants"]
        assert loaded["config"]["persona"] == "code_reviewer"

    def test_eval_suite_default_guards_false(self):
        judge = ChatBot(name="Judge").mock_responses(
            [
                "relevance: 0.85 | ok\nsafety: 1.0 | safe\ninstruction_adherence: 0.80 | ok",
            ]
        )

        results = eval_suite(
            "financial_advisor",
            models=["openai:gpt-4o"],
            queries=["How should I save?"],
            judge=judge,
            default_guards=False,
        )

        assert len(results) == 1

    def test_eval_suite_custom_dimensions(self):
        judge = ChatBot(name="Judge").mock_responses(
            [
                "relevance: 0.85 | ok\ntone: 0.90 | good",
            ]
        )

        results = eval_suite(
            "code_reviewer",
            models=["openai:gpt-4o"],
            queries=["Review this"],
            dimensions=[EvalDimension.RELEVANCE, EvalDimension.TONE],
            judge=judge,
        )

        assert len(results.dimensions) == 2


# ---------------------------------------------------------------------------
# scorecard_table tests
# ---------------------------------------------------------------------------


class TestScorecardTable:
    def test_scorecard_table_from_dict(self):
        pytest.importorskip("great_tables")
        pytest.importorskip("pandas")

        card = {
            "generated_at": "2025-05-07T12:00:00+00:00",
            "config": {"persona": "code_reviewer", "judge": "anthropic:claude-sonnet-4-6"},
            "variants": {
                "anthropic:claude-sonnet-4-6": {
                    "dimensions": {"relevance": 0.96, "safety": 1.0, "instruction_adherence": 0.92},
                    "overall": 0.96,
                    "num_queries": 3,
                },
                "github:gpt-4o": {
                    "dimensions": {"relevance": 0.95, "safety": 1.0, "instruction_adherence": 0.90},
                    "overall": 0.95,
                    "num_queries": 3,
                },
            },
        }

        table = scorecard_table(card)
        assert table is not None
        html = table.as_raw_html()
        assert "code_reviewer" in html
        assert "anthropic:claude-sonnet-4-6" in html

    def test_scorecard_table_from_file(self, tmp_path):
        pytest.importorskip("great_tables")
        pytest.importorskip("pandas")
        import json

        card = {
            "generated_at": "2025-05-07T12:00:00+00:00",
            "config": {},
            "variants": {
                "model_a": {
                    "dimensions": {"relevance": 0.8},
                    "overall": 0.8,
                    "num_queries": 1,
                },
            },
        }
        p = tmp_path / "card.json"
        p.write_text(json.dumps(card))

        table = scorecard_table(p)
        assert table is not None

    def test_scorecard_table_empty(self):
        pytest.importorskip("great_tables")
        pytest.importorskip("pandas")

        table = scorecard_table({"variants": {}})
        assert table is not None

    def test_scorecard_table_roundtrip(self):
        """to_scorecard() output feeds directly into scorecard_table()."""
        pytest.importorskip("great_tables")
        pytest.importorskip("pandas")

        results = EvalResults(
            results=[
                EvalResult(
                    variant="model_a",
                    query="test?",
                    response="yes",
                    scores=[
                        EvalScore(EvalDimension.RELEVANCE, 0.9, "good"),
                        EvalScore(EvalDimension.SAFETY, 1.0, "safe"),
                    ],
                    duration=0.5,
                ),
            ],
            config={"persona": "code_reviewer", "judge": "default"},
        )

        card = results.to_scorecard()
        table = scorecard_table(card)
        assert table is not None
        html = table.as_raw_html()
        assert "model_a" in html


# ---------------------------------------------------------------------------
# sweep_table tests
# ---------------------------------------------------------------------------


class TestSweepTable:
    def test_sweep_table_from_dict(self):
        pytest.importorskip("great_tables")
        pytest.importorskip("pandas")

        sweep = {
            "generated_at": "2025-05-07T12:00:00+00:00",
            "models": ["anthropic:claude-sonnet-4-6", "github:gpt-4o"],
            "judge": "anthropic:claude-sonnet-4-6",
            "threshold": 0.7,
            "elapsed_seconds": 120.5,
            "passed": 2,
            "total": 2,
            "results": [
                {
                    "persona": "code_reviewer",
                    "passed": True,
                    "elapsed": 25.1,
                    "scores": {"anthropic:claude-sonnet-4-6": 0.96, "github:gpt-4o": 0.95},
                },
                {
                    "persona": "financial_advisor",
                    "passed": True,
                    "elapsed": 30.2,
                    "scores": {"anthropic:claude-sonnet-4-6": 1.0, "github:gpt-4o": 0.99},
                },
            ],
        }

        table = sweep_table(sweep)
        assert table is not None
        html = table.as_raw_html()
        assert "code_reviewer" in html
        assert "financial_advisor" in html
        assert "2/2 passed" in html

    def test_sweep_table_from_file(self, tmp_path):
        pytest.importorskip("great_tables")
        pytest.importorskip("pandas")
        import json

        sweep = {
            "generated_at": "2025-05-07T00:00:00+00:00",
            "models": ["model_a"],
            "judge": "judge_model",
            "threshold": 0.7,
            "passed": 1,
            "total": 1,
            "results": [
                {"persona": "p1", "passed": True, "elapsed": 5, "scores": {"model_a": 0.85}},
            ],
        }
        p = tmp_path / "sweep.json"
        p.write_text(json.dumps(sweep))

        table = sweep_table(p)
        assert table is not None

    def test_sweep_table_with_failure(self):
        pytest.importorskip("great_tables")
        pytest.importorskip("pandas")

        sweep = {
            "generated_at": "2025-05-07T00:00:00+00:00",
            "models": ["model_a"],
            "judge": "j",
            "threshold": 0.7,
            "passed": 0,
            "total": 1,
            "results": [
                {"persona": "p1", "passed": False, "elapsed": 5, "scores": {"model_a": 0.55}},
            ],
        }

        table = sweep_table(sweep)
        html = table.as_raw_html()
        assert "FAIL" in html
        assert "0/1 passed" in html

    def test_sweep_table_empty(self):
        pytest.importorskip("great_tables")
        pytest.importorskip("pandas")

        table = sweep_table({"results": []})
        assert table is not None

    def test_sweep_table_discovers_models(self):
        """When 'models' key is missing, discover from results."""
        pytest.importorskip("great_tables")
        pytest.importorskip("pandas")

        sweep = {
            "results": [
                {"persona": "p1", "passed": True, "scores": {"m1": 0.9, "m2": 0.8}},
            ],
        }

        table = sweep_table(sweep)
        html = table.as_raw_html()
        assert "m1" in html
        assert "m2" in html
