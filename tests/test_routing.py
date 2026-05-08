"""Tests for talk_box.routing module."""

from talk_box.models import CostTier, ModelProfile
from talk_box.routing import (
    Router,
    RoutingResult,
    RoutingStrategy,
    TaskComplexity,
    classify_complexity,
    route,
)


# ---------------------------------------------------------------------------
# TaskComplexity enum
# ---------------------------------------------------------------------------


class TestTaskComplexity:
    def test_values(self):
        assert TaskComplexity.SIMPLE.value == "simple"
        assert TaskComplexity.MODERATE.value == "moderate"
        assert TaskComplexity.COMPLEX.value == "complex"
        assert TaskComplexity.EXPERT.value == "expert"

    def test_all_members(self):
        assert len(TaskComplexity) == 4


# ---------------------------------------------------------------------------
# RoutingStrategy enum
# ---------------------------------------------------------------------------


class TestRoutingStrategy:
    def test_values(self):
        assert RoutingStrategy.COST_OPTIMIZED.value == "cost_optimized"
        assert RoutingStrategy.QUALITY_OPTIMIZED.value == "quality_optimized"
        assert RoutingStrategy.LOCAL_FIRST.value == "local_first"
        assert RoutingStrategy.BALANCED.value == "balanced"

    def test_all_members(self):
        assert len(RoutingStrategy) == 4


# ---------------------------------------------------------------------------
# classify_complexity
# ---------------------------------------------------------------------------


class TestClassifyComplexity:
    def test_empty_string(self):
        assert classify_complexity("") == TaskComplexity.SIMPLE

    def test_whitespace_only(self):
        assert classify_complexity("   ") == TaskComplexity.SIMPLE

    def test_simple_question(self):
        assert classify_complexity("What is Python?") == TaskComplexity.SIMPLE

    def test_simple_list_request(self):
        assert classify_complexity("List the planets in our solar system") == TaskComplexity.SIMPLE

    def test_simple_translation(self):
        assert classify_complexity("Translate hello to French") == TaskComplexity.SIMPLE

    def test_moderate_implementation(self):
        result = classify_complexity("Implement a binary search tree in Python")
        assert result in (TaskComplexity.MODERATE, TaskComplexity.COMPLEX)

    def test_complex_refactoring(self):
        result = classify_complexity(
            "Refactor this module to use dependency injection. "
            "Here is the current code:\n```python\n" + "x = 1\n" * 50 + "```"
        )
        assert result in (TaskComplexity.COMPLEX, TaskComplexity.EXPERT)

    def test_expert_system_design(self):
        result = classify_complexity(
            "Architect a distributed system for real-time event processing "
            "with formal verification of consistency guarantees"
        )
        assert result == TaskComplexity.EXPERT

    def test_expert_security_audit(self):
        result = classify_complexity(
            "Perform a security audit looking for vulnerabilities and potential exploits "
            "in this authentication system with multi-step reasoning"
        )
        assert result == TaskComplexity.EXPERT

    def test_long_prompt_increases_complexity(self):
        short = classify_complexity("Fix this bug")
        long_prompt = "Fix this bug in the following code:\n" + "x = 1\n" * 300
        long_result = classify_complexity(long_prompt)
        # Long prompt should be at least as complex as the short one
        order = [
            TaskComplexity.SIMPLE,
            TaskComplexity.MODERATE,
            TaskComplexity.COMPLEX,
            TaskComplexity.EXPERT,
        ]
        assert order.index(long_result) >= order.index(short)

    def test_multiple_questions_increase_complexity(self):
        result = classify_complexity(
            "What causes this error? How do I fix it? Why does it only happen in production? "
            "Is there a better pattern?"
        )
        assert result != TaskComplexity.SIMPLE

    def test_code_blocks_increase_complexity(self):
        prompt = "Compare these:\n```python\nx=1\n```\n```python\ny=2\n```\n```python\nz=3\n```"
        result = classify_complexity(prompt)
        assert result != TaskComplexity.SIMPLE


# ---------------------------------------------------------------------------
# RoutingResult
# ---------------------------------------------------------------------------


class TestRoutingResult:
    def test_frozen(self):
        profile = ModelProfile(provider="test", model="m1")
        result = RoutingResult(
            model=profile,
            reason="test reason",
            complexity=TaskComplexity.SIMPLE,
        )
        assert result.model == profile
        assert result.reason == "test reason"
        assert result.complexity == TaskComplexity.SIMPLE
        assert result.alternatives == []

    def test_with_alternatives(self):
        p1 = ModelProfile(provider="test", model="m1")
        p2 = ModelProfile(provider="test", model="m2")
        result = RoutingResult(
            model=p1,
            reason="best fit",
            complexity=TaskComplexity.MODERATE,
            alternatives=[p2],
        )
        assert len(result.alternatives) == 1
        assert result.alternatives[0] == p2


# ---------------------------------------------------------------------------
# Router class
# ---------------------------------------------------------------------------


class TestRouter:
    def test_default_strategy(self):
        router = Router()
        assert router.strategy == RoutingStrategy.BALANCED

    def test_custom_strategy(self):
        router = Router(strategy=RoutingStrategy.LOCAL_FIRST)
        assert router.strategy == RoutingStrategy.LOCAL_FIRST

    def test_candidates_from_registry(self):
        router = Router()
        assert len(router.candidates) > 0

    def test_specific_candidates(self):
        router = Router(candidates=["anthropic:claude-sonnet-4-6", "openai:gpt-4o"])
        assert len(router.candidates) == 2

    def test_specific_candidates_invalid_key_skipped(self):
        router = Router(candidates=["anthropic:claude-sonnet-4-6", "nonexistent:fake-model"])
        assert len(router.candidates) == 1

    def test_route_returns_routing_result(self):
        router = Router()
        result = router.route("What is Python?")
        assert isinstance(result, RoutingResult)
        assert isinstance(result.model, ModelProfile)
        assert isinstance(result.complexity, TaskComplexity)
        assert isinstance(result.reason, str)
        assert len(result.reason) > 0

    def test_route_simple_task_cost_optimized(self):
        router = Router(strategy=RoutingStrategy.COST_OPTIMIZED)
        result = router.route("Define polymorphism")
        # Should prefer cheap models for simple tasks
        assert result.model.cost_tier in (CostTier.FREE, CostTier.LOW)

    def test_route_expert_task_quality_optimized(self):
        router = Router(strategy=RoutingStrategy.QUALITY_OPTIMIZED)
        result = router.route(
            "Architect a distributed system with formal verification of consistency"
        )
        # Should prefer premium/high models for expert tasks
        assert result.model.cost_tier in (CostTier.HIGH, CostTier.PREMIUM)

    def test_route_local_first_prefers_ollama(self):
        router = Router(strategy=RoutingStrategy.LOCAL_FIRST)
        result = router.route("Explain list comprehensions")
        assert result.model.provider == "ollama"

    def test_route_with_requires_tools(self):
        router = Router()
        result = router.route("Call the API and process results", requires=["tools"])
        assert result.model.supports_tools is True

    def test_route_with_requires_vision(self):
        router = Router()
        result = router.route("Describe this image", requires=["vision"])
        assert result.model.supports_vision is True

    def test_route_max_cost_tier(self):
        router = Router(max_cost_tier=CostTier.LOW)
        result = router.route("Write a function")
        assert result.model.cost_tier in (CostTier.FREE, CostTier.LOW)

    def test_route_max_cost_tier_override(self):
        router = Router(max_cost_tier=CostTier.PREMIUM)
        result = router.route("Write a function", max_cost_tier=CostTier.LOW)
        assert result.model.cost_tier in (CostTier.FREE, CostTier.LOW)

    def test_route_min_context_window(self):
        router = Router()
        result = router.route("Process this long document", min_context_window=200_000)
        assert result.model.context_window is not None
        assert result.model.context_window >= 200_000

    def test_route_no_candidates_raises(self):
        router = Router(candidates=["anthropic:claude-sonnet-4-6"])
        try:
            router.route(
                "Do something",
                requires=["vision", "tools"],
                max_cost_tier=CostTier.FREE,
            )
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "No candidate models" in str(e)

    def test_route_prefer_local_with_balanced(self):
        router = Router(prefer_local=True)
        result = router.route("Simple question about Python")
        # With prefer_local=True and balanced strategy, should behave like LOCAL_FIRST
        assert result.model.provider == "ollama"

    def test_route_alternatives_populated(self):
        router = Router()
        result = router.route("Build a web scraper")
        # Should have alternatives since many models qualify
        assert len(result.alternatives) > 0

    def test_route_complexity_in_result(self):
        router = Router()
        result = router.route("What is 2 + 2?")
        assert result.complexity == TaskComplexity.SIMPLE


# ---------------------------------------------------------------------------
# route() functional API
# ---------------------------------------------------------------------------


class TestRouteFunctional:
    def test_basic_routing(self):
        result = route("What is Python?")
        assert isinstance(result, RoutingResult)
        assert isinstance(result.model, ModelProfile)

    def test_strategy_parameter(self):
        result = route("Summarize this", strategy=RoutingStrategy.COST_OPTIMIZED)
        assert result.model.cost_tier in (CostTier.FREE, CostTier.LOW)

    def test_requires_parameter(self):
        result = route("Call an API", requires=["tools"])
        assert result.model.supports_tools is True

    def test_max_cost_tier_parameter(self):
        result = route("Write code", max_cost_tier=CostTier.MEDIUM)
        assert result.model.cost_tier in (CostTier.FREE, CostTier.LOW, CostTier.MEDIUM)

    def test_prefer_local_parameter(self):
        result = route("Explain this", prefer_local=True)
        assert result.model.provider == "ollama"

    def test_candidates_parameter(self):
        result = route(
            "Hello",
            candidates=["anthropic:claude-sonnet-4-6", "openai:gpt-4o"],
        )
        assert result.model.provider in ("anthropic", "openai")

    def test_min_context_window_parameter(self):
        result = route("Long document analysis", min_context_window=100_000)
        assert result.model.context_window is not None
        assert result.model.context_window >= 100_000

    def test_no_candidates_raises(self):
        try:
            route(
                "Do something",
                candidates=["nonexistent:model"],
            )
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_combined_parameters(self):
        result = route(
            "Analyze this code",
            strategy=RoutingStrategy.QUALITY_OPTIMIZED,
            requires=["tools"],
            max_cost_tier=CostTier.HIGH,
        )
        assert result.model.supports_tools is True
        assert result.model.cost_tier in (
            CostTier.FREE,
            CostTier.LOW,
            CostTier.MEDIUM,
            CostTier.HIGH,
        )
