import pytest
from unittest.mock import Mock, patch
from talk_box.builder import ChatBot
from talk_box.pathways import Pathways
from talk_box.prompt_builder import PromptBuilder
from talk_box.conversation import Conversation
from talk_box.testing import (
    PathwayTestStrategy,
    PathwayTestResult,
    PathwayTestResults,
    PathwayTesterBot,
    PathwayJudgeBot,
    autotest_pathways,
    _extract_pathway_specs_from_bot,
    _extract_pathways_from_prompt_text,
    _parse_test_scenario,
)


class TestPathwayExtraction:
    """Test pathway extraction from bot configurations."""

    def test_extract_from_prompt_builder(self):
        """Test extracting pathways from PromptBuilder configuration."""
        # Create a pathway
        pathway = (
            Pathways(title="Test Pathway", desc="A test pathway", activation="User needs help")
            .state("initial: start conversation")
            .required(["user name"])
            .next_state("processing")
            .state("processing: handle request")
            .success_condition("request completed")
        )

        # Create bot with pathway
        prompt_builder = PromptBuilder().pathways(pathway)

        # Mock the _pathways attribute for testing
        if not hasattr(prompt_builder, "_pathways"):
            prompt_builder._pathways = [pathway]

        bot = ChatBot().system_prompt(prompt_builder)

        # Extract pathways
        specs = _extract_pathway_specs_from_bot(bot)

        assert len(specs) == 1
        assert specs[0]["title"] == "Test Pathway"
        assert specs[0]["description"] == "A test pathway"
        assert "initial" in specs[0]["states"]
        assert "processing" in specs[0]["states"]

    def test_extract_from_prompt_text(self):
        """Test extracting pathways from prompt text patterns."""
        prompt_text = """You are a helpful assistant.

**Customer Support**
Purpose: Help customers with their issues
Activate when:
- Customer has a problem
- User needs assistance

Flow guidance:
- INTAKE (collect): gather customer information
  Required: problem description, contact info
- RESOLUTION (chat): provide solution
"""

        specs = _extract_pathways_from_prompt_text(prompt_text)

        assert len(specs) >= 1
        found_spec = None
        for spec in specs:
            if spec["title"] == "Customer Support":
                found_spec = spec
                break

        assert found_spec is not None
        assert found_spec["description"] == "Help customers with their issues"
        assert "Customer has a problem" in found_spec["activation_conditions"]
        assert "User needs assistance" in found_spec["activation_conditions"]

    def test_extract_no_pathways(self):
        """Test extraction when no pathways are configured."""
        bot = ChatBot().system_prompt("You are a helpful assistant.")
        specs = _extract_pathway_specs_from_bot(bot)
        assert specs == []


class TestPathwayTesterBot:
    """Test the pathway tester bot functionality."""

    def test_initialization(self):
        """Test PathwayTesterBot initialization."""
        tester = PathwayTesterBot()
        assert tester.model == "openai:gpt-4-turbo"

        custom_tester = PathwayTesterBot(model="gpt-3.5-turbo")
        assert custom_tester.model == "gpt-3.5-turbo"

    def test_generate_test_scenarios(self):
        """Test scenario generation for different strategies."""
        # Mock the bot response
        mock_response = """
        SCENARIO 1:
        Initial: I need help with my account
        Follow-up 1: Can you help me reset my password?
        Follow-up 2: I forgot my email too
        Tests: Direct pathway following

        SCENARIO 2:
        Initial: My account is broken
        Follow-up 1: Just fix it now
        Tests: Incomplete information provision
        """

        tester = PathwayTesterBot()

        # Mock the bot property directly
        mock_bot = Mock()
        mock_bot.chat.return_value = mock_response
        tester._bot = mock_bot

        pathway_spec = {
            "title": "Account Support",
            "description": "Help with account issues",
            "states": {"intake": {}, "resolution": {}},
            "activation_conditions": ["Account problems"],
        }

        scenarios = tester.generate_test_scenarios(pathway_spec, PathwayTestStrategy.DIRECT_FLOW)

        assert len(scenarios) == 2
        assert "I need help with my account" in scenarios[0]
        assert "My account is broken" in scenarios[1]


class TestPathwayJudgeBot:
    """Test the pathway judge bot functionality."""

    def test_initialization(self):
        """Test PathwayJudgeBot initialization."""
        judge = PathwayJudgeBot()
        assert judge.model == "openai:gpt-4-turbo"

        custom_judge = PathwayJudgeBot(model="gpt-4")
        assert custom_judge.model == "gpt-4"

    def test_evaluate_conversation(self):
        """Test conversation evaluation for pathway adherence."""
        # Mock the judge bot response
        mock_response = """
        STATES_ACHIEVED: intake, resolution
        INFORMATION_GATHERED: user name, problem description, contact info
        PATHWAY_ADHERENCE: 0.85
        ISSUES: Missing validation step
        SUCCESS_CRITERIA: User problem resolved
        """

        judge = PathwayJudgeBot()

        # Mock the bot property directly
        mock_bot = Mock()
        mock_bot.chat.return_value = mock_response
        judge._bot = mock_bot

        # Create test conversation
        conversation = Conversation()
        conversation.add_user_message("I need help with my account")
        conversation.add_assistant_message("I can help you with that. What's your name?")
        conversation.add_user_message("John Doe")
        conversation.add_assistant_message("Thanks John. What account issue are you experiencing?")

        pathway_spec = {
            "title": "Account Support",
            "states": {
                "intake": {"description": "Gather user info"},
                "resolution": {"description": "Solve the problem"},
            },
        }

        evaluation = judge.evaluate_conversation(conversation, pathway_spec)

        assert evaluation["pathway_adherence_score"] == 0.85
        assert "intake" in evaluation["states_achieved"]
        assert "resolution" in evaluation["states_achieved"]
        assert "Missing validation step" in evaluation["issues"]


class TestPathwayScenarioParsing:
    """Test parsing of test scenarios."""

    def test_parse_scenario_multiple_messages(self):
        """Test parsing scenario with multiple messages."""
        scenario = """
        SCENARIO 1:
        Initial: I need help
        Follow-up 1: With my account
        Follow-up 2: It's not working
        Tests: Basic flow
        """

        messages = _parse_test_scenario(scenario)

        assert len(messages) == 3
        assert messages[0] == "I need help"
        assert messages[1] == "With my account"
        assert messages[2] == "It's not working"

    def test_parse_scenario_single_message(self):
        """Test parsing scenario with single message."""
        scenario = "Initial: Hello, I need assistance"
        messages = _parse_test_scenario(scenario)

        assert len(messages) == 1
        assert messages[0] == "Hello, I need assistance"

    def test_parse_scenario_fallback(self):
        """Test fallback parsing when format is unclear."""
        scenario = "This is just a plain message"
        messages = _parse_test_scenario(scenario)

        assert len(messages) == 1
        assert messages[0] == "This is just a plain message"


class TestPathwayTestResult:
    """Test PathwayTestResult data structure."""

    def test_initialization(self):
        """Test PathwayTestResult initialization."""
        conversation = Conversation()
        result = PathwayTestResult(
            pathway_title="Test Pathway",
            test_scenario="Test scenario",
            strategy=PathwayTestStrategy.DIRECT_FLOW,
            conversation=conversation,
            expected_states=["state1", "state2"],
            actual_progression=["state1"],
        )

        assert result.pathway_title == "Test Pathway"
        assert result.strategy == PathwayTestStrategy.DIRECT_FLOW
        assert result.expected_states == ["state1", "state2"]
        assert result.actual_progression == ["state1"]
        assert result.completed == False  # Default value
        assert result.pathway_adherence_score == 0.0  # Default value


class TestPathwayTestResults:
    """Test PathwayTestResults container and analysis."""

    def test_initialization_empty(self):
        """Test initialization with empty results."""
        results = PathwayTestResults([], {})
        assert len(results.results) == 0
        assert results.summary["total_tests"] == 0
        assert results.summary["avg_adherence_score"] == 0.0

    def test_initialization_with_results(self):
        """Test initialization with test results."""
        # Create mock results
        result1 = PathwayTestResult(
            pathway_title="Pathway A",
            test_scenario="Scenario 1",
            strategy=PathwayTestStrategy.DIRECT_FLOW,
            conversation=Conversation(),
            expected_states=["state1", "state2"],
            actual_progression=["state1", "state2"],
            completed=True,
        )
        result1.pathway_adherence_score = 0.9
        result1.states_achieved = ["state1", "state2"]

        result2 = PathwayTestResult(
            pathway_title="Pathway A",
            test_scenario="Scenario 2",
            strategy=PathwayTestStrategy.SKIP_STATES,
            conversation=Conversation(),
            expected_states=["state1", "state2"],
            actual_progression=["state1"],
            completed=True,
        )
        result2.pathway_adherence_score = 0.7
        result2.states_achieved = ["state1"]
        result2.issues = ["Skipped state2"]

        config = {"intensity": "medium"}
        results_container = PathwayTestResults([result1, result2], config)

        assert len(results_container) == 2
        assert results_container.summary["total_tests"] == 2
        assert results_container.summary["completed_tests"] == 2
        assert results_container.summary["avg_adherence_score"] == 0.8  # (0.9 + 0.7) / 2
        assert results_container.summary["issues_found"] == 1
        assert results_container.summary["pathways_tested"] == 1  # Only "Pathway A"

    def test_problem_summary(self):
        """Test problem summary generation."""
        result1 = PathwayTestResult(
            pathway_title="Pathway A",
            test_scenario="Scenario 1",
            strategy=PathwayTestStrategy.SKIP_STATES,
            conversation=Conversation(),
            expected_states=["state1", "state2"],
            actual_progression=["state1"],
            completed=True,
        )
        result1.issues = ["Skipped required state", "Missing information"]

        result2 = PathwayTestResult(
            pathway_title="Pathway B",
            test_scenario="Scenario 2",
            strategy=PathwayTestStrategy.RESISTANCE,
            conversation=Conversation(),
            expected_states=["state1"],
            actual_progression=[],
            completed=True,
        )
        result2.issues = ["Skipped required state"]

        results = PathwayTestResults([result1, result2], {})
        problems = results.get_problem_summary()

        assert len(problems) == 2
        # Should be sorted by frequency
        assert problems[0]["issue"] == "Skipped required state"
        assert problems[0]["frequency"] == 2
        assert problems[1]["issue"] == "Missing information"
        assert problems[1]["frequency"] == 1

    def test_adherence_distribution(self):
        """Test adherence score distribution calculation."""
        results_data = []

        # Create results with different adherence scores
        scores = [0.95, 0.85, 0.75, 0.65, 0.45]
        for i, score in enumerate(scores):
            result = PathwayTestResult(
                pathway_title=f"Pathway {i}",
                test_scenario=f"Scenario {i}",
                strategy=PathwayTestStrategy.DIRECT_FLOW,
                conversation=Conversation(),
                expected_states=["state1"],
                actual_progression=["state1"],
                completed=True,
            )
            result.pathway_adherence_score = score
            results_data.append(result)

        results = PathwayTestResults(results_data, {})
        distribution = results.get_adherence_distribution()

        assert distribution["excellent (0.9-1.0)"] == 1  # 0.95
        assert distribution["good (0.8-0.9)"] == 1  # 0.85
        assert distribution["fair (0.7-0.8)"] == 1  # 0.75
        assert distribution["poor (0.6-0.7)"] == 1  # 0.65
        assert distribution["failing (<0.6)"] == 1  # 0.45

    def test_html_representation(self):
        """Test HTML representation for Jupyter notebooks."""
        result = PathwayTestResult(
            pathway_title="Test Pathway",
            test_scenario="Test scenario",
            strategy=PathwayTestStrategy.DIRECT_FLOW,
            conversation=Conversation(),
            expected_states=["state1"],
            actual_progression=["state1"],
            completed=True,
        )
        result.pathway_adherence_score = 0.8
        result.test_duration = 45.0

        results = PathwayTestResults([result], {"intensity": "medium"})
        html = results._repr_html_()

        assert "Pathway Testing Results" in html
        assert "80.0%" in html  # Adherence score
        assert "DIRECT_FLOW" in html or "direct_flow" in html  # Strategy name
        assert "medium" in html  # Intensity
        assert "0.80" in html  # Individual test score in table


class TestAutotestPathways:
    """Test the main autotest_pathways function."""

    def test_no_pathways_configured(self):
        """Test error when bot has no pathways configured."""
        bot = ChatBot().system_prompt("You are a helpful assistant.")

        with pytest.raises(ValueError, match="no pathway specifications configured"):
            autotest_pathways(bot)

    @patch("talk_box.testing._extract_pathway_specs_from_bot")
    @patch("talk_box.testing.PathwayTesterBot")
    @patch("talk_box.testing.PathwayJudgeBot")
    @patch("talk_box.testing._run_pathway_tests")
    def test_basic_pathway_testing(
        self, mock_run_tests, mock_judge_class, mock_tester_class, mock_extract
    ):
        """Test basic pathway testing flow."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            # Mock pathway extraction
            mock_pathway_spec = {
                "title": "Test Pathway",
                "states": {"state1": {}, "state2": {}},
                "activation_conditions": ["test condition"],
            }
            mock_extract.return_value = [mock_pathway_spec]

            # Mock test results
            mock_result = PathwayTestResult(
                pathway_title="Test Pathway",
                test_scenario="Test scenario",
                strategy=PathwayTestStrategy.DIRECT_FLOW,
                conversation=Conversation(),
                expected_states=["state1", "state2"],
                actual_progression=["state1", "state2"],
                completed=True,
            )
            mock_result.pathway_adherence_score = 0.85
            mock_run_tests.return_value = [mock_result]

            # Mock bot components
            mock_tester_class.return_value = Mock()
            mock_judge_class.return_value = Mock()

            # Create bot and run test
            bot = ChatBot()
            bot._config = {"model": "gpt-4"}  # Mock config

            results = autotest_pathways(bot, test_intensity="light")

            assert isinstance(results, PathwayTestResults)
            assert len(results.results) == 1
            assert results.results[0].pathway_adherence_score == 0.85
            assert results.config["intensity"] == "light"

    def test_invalid_intensity(self):
        """Test error with invalid test intensity."""
        # Mock a bot with pathways
        with patch("talk_box.testing._extract_pathway_specs_from_bot") as mock_extract:
            mock_extract.return_value = [{"title": "Test", "states": {}}]
            bot = ChatBot()

            with pytest.raises(ValueError, match="test_intensity must be one of"):
                autotest_pathways(bot, test_intensity="invalid")

    def test_intensity_configurations(self):
        """Test different intensity level configurations."""
        with patch("talk_box.testing._extract_pathway_specs_from_bot") as mock_extract:
            with patch("talk_box.testing._run_pathway_tests") as mock_run_tests:
                with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                    mock_extract.return_value = [{"title": "Test", "states": {}}]
                    mock_run_tests.return_value = []

                    bot = ChatBot()
                    bot._config = {"model": "gpt-4"}

                    # Test light intensity
                    results = autotest_pathways(bot, test_intensity="light")
                    assert results.config["max_tests"] == 6
                    assert len(results.config["strategies"]) == 2

                    # Test medium intensity
                    results = autotest_pathways(bot, test_intensity="medium")
                    assert results.config["max_tests"] == 12
                    assert len(results.config["strategies"]) == 4

                    # Test thorough intensity
                    results = autotest_pathways(bot, test_intensity="thorough")
                    assert results.config["max_tests"] == 20
                    assert len(results.config["strategies"]) == 6

                    # Test exhaustive intensity
                    results = autotest_pathways(bot, test_intensity="exhaustive")
                    assert results.config["max_tests"] == 30
                    assert len(results.config["strategies"]) == 7  # All strategies

    def test_max_tests_override(self):
        """Test max_tests parameter override."""
        with patch("talk_box.testing._extract_pathway_specs_from_bot") as mock_extract:
            with patch("talk_box.testing._run_pathway_tests") as mock_run_tests:
                with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                    mock_extract.return_value = [{"title": "Test", "states": {}}]
                    mock_run_tests.return_value = []

                    bot = ChatBot()
                    bot._config = {"model": "gpt-4"}

                    results = autotest_pathways(bot, test_intensity="light", max_tests=15)
                    assert results.config["max_tests"] == 15  # Override should work


if __name__ == "__main__":
    pytest.main([__file__])
