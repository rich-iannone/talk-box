"""Hybrid routing: route tasks between local and cloud models based on complexity."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from talk_box.models import CostTier, ModelProfile, get_model_profile, list_models

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class TaskComplexity(Enum):
    """Estimated complexity of a task or prompt.

    Used by the router to match tasks to appropriately-capable models.
    """

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


class RoutingStrategy(Enum):
    """Strategy that governs how the router selects models.

    Attributes
    ----------
    COST_OPTIMIZED
        Prefer the cheapest model that meets requirements.
    QUALITY_OPTIMIZED
        Prefer the most capable model available.
    LOCAL_FIRST
        Prefer local models; fall back to cloud only when capabilities are missing.
    BALANCED
        Balance cost and quality, preferring mid-tier models.
    """

    COST_OPTIMIZED = "cost_optimized"
    QUALITY_OPTIMIZED = "quality_optimized"
    LOCAL_FIRST = "local_first"
    BALANCED = "balanced"


@dataclass(frozen=True)
class RoutingResult:
    """Result of routing a task to a model.

    Parameters
    ----------
    model
        The selected model profile.
    reason
        Human-readable explanation of why this model was chosen.
    complexity
        The estimated task complexity.
    alternatives
        Other candidate models that also met requirements, ranked by preference.
    """

    model: ModelProfile
    reason: str
    complexity: TaskComplexity
    alternatives: list[ModelProfile] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Complexity classification
# ---------------------------------------------------------------------------

# Patterns that suggest higher complexity
_EXPERT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(prove|theorem|formal verification|mathematical proof)\b", re.IGNORECASE),
    re.compile(r"\b(architect|system design|distributed system)\b", re.IGNORECASE),
    re.compile(r"\b(optimize|performance critical|latency sensitive)\b", re.IGNORECASE),
    re.compile(r"\b(security audit|vulnerability|exploit|CVE)\b", re.IGNORECASE),
    re.compile(r"\b(multi-step|chain of thought|reasoning)\b", re.IGNORECASE),
]

_COMPLEX_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(refactor|debug|troubleshoot|diagnose)\b", re.IGNORECASE),
    re.compile(r"\b(implement|build|create|design)\b", re.IGNORECASE),
    re.compile(r"\b(compare|analyze|evaluate|assess)\b", re.IGNORECASE),
    re.compile(r"\b(explain why|how does|what causes)\b", re.IGNORECASE),
    re.compile(r"```[\s\S]{200,}```", re.DOTALL),  # Large code blocks
]

_SIMPLE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(translate|summarize|rewrite|rephrase)\b", re.IGNORECASE),
    re.compile(r"\b(list|enumerate|name|define)\b", re.IGNORECASE),
    re.compile(r"\b(what is|who is|when was)\b", re.IGNORECASE),
    re.compile(r"\b(format|convert|fix typo)\b", re.IGNORECASE),
]


def classify_complexity(prompt: str) -> TaskComplexity:
    """Estimate the complexity of a task from its prompt text.

    Uses heuristics based on prompt length, vocabulary, and structural markers to
    classify the task into one of four complexity levels.

    Parameters
    ----------
    prompt
        The prompt text to classify.

    Returns
    -------
    TaskComplexity
        The estimated complexity level.

    Examples
    --------
    ```python
    import talk_box as tb

    tb.classify_complexity("What is Python?")
    # TaskComplexity.SIMPLE

    tb.classify_complexity("Design a distributed caching system with consistency guarantees")
    # TaskComplexity.EXPERT
    ```
    """
    if not prompt or not prompt.strip():
        return TaskComplexity.SIMPLE

    score = 0

    # Length-based scoring
    length = len(prompt)
    if length > 2000:
        score += 3
    elif length > 800:
        score += 2
    elif length > 300:
        score += 1

    # Pattern matching — accumulate expert matches
    expert_hits = sum(1 for p in _EXPERT_PATTERNS if p.search(prompt))
    score += expert_hits * 3

    if expert_hits == 0:
        for pattern in _COMPLEX_PATTERNS:
            if pattern.search(prompt):
                score += 2
                break

    simple_match = any(p.search(prompt) for p in _SIMPLE_PATTERNS)
    if simple_match and score == 0:
        return TaskComplexity.SIMPLE

    # Code presence increases complexity
    code_blocks = len(re.findall(r"```", prompt))
    if code_blocks >= 4:
        score += 2
    elif code_blocks >= 2:
        score += 1

    # Multiple questions increase complexity
    question_marks = prompt.count("?")
    if question_marks >= 3:
        score += 1

    # Classify based on accumulated score
    if score >= 6:
        return TaskComplexity.EXPERT
    elif score >= 4:
        return TaskComplexity.COMPLEX
    elif score >= 2:
        return TaskComplexity.MODERATE
    else:
        return TaskComplexity.SIMPLE


# ---------------------------------------------------------------------------
# Cost tier ordering (for scoring)
# ---------------------------------------------------------------------------

_COST_ORDER: dict[CostTier, int] = {
    CostTier.FREE: 0,
    CostTier.LOW: 1,
    CostTier.MEDIUM: 2,
    CostTier.HIGH: 3,
    CostTier.PREMIUM: 4,
}


def _cost_rank(tier: CostTier | None) -> int:
    """Numeric rank for a cost tier (0=free, 4=premium)."""
    if tier is None:
        return 2  # Assume medium when unknown
    return _COST_ORDER[tier]


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------


def _score_model(
    profile: ModelProfile,
    complexity: TaskComplexity,
    strategy: RoutingStrategy,
) -> float:
    """Score a candidate model for a given task and strategy.

    Higher scores are better. Range is approximately 0-100.
    """
    score = 50.0  # Baseline
    cost = _cost_rank(profile.cost_tier)

    if strategy == RoutingStrategy.COST_OPTIMIZED:
        # Strongly prefer cheaper models, penalize expensive ones
        score -= cost * 15
        # But still give some credit for capability match
        score += _complexity_fit_bonus(profile, complexity) * 0.5

    elif strategy == RoutingStrategy.QUALITY_OPTIMIZED:
        # Prefer expensive/capable models
        score += cost * 10
        score += _complexity_fit_bonus(profile, complexity)

    elif strategy == RoutingStrategy.LOCAL_FIRST:
        # Strong bonus for local models
        if profile.provider == "ollama":
            score += 30
        elif profile.cost_tier == CostTier.FREE:
            score += 15
        # Still consider capability
        score += _complexity_fit_bonus(profile, complexity) * 0.5

    elif strategy == RoutingStrategy.BALANCED:
        # Sweet spot: moderate cost, good capability match
        # Penalize extremes (both too cheap and too expensive)
        if cost == 2:  # MEDIUM
            score += 10
        elif cost in (1, 3):  # LOW or HIGH
            score += 5
        else:  # FREE or PREMIUM
            score -= 5
        score += _complexity_fit_bonus(profile, complexity)

    # Context window bonus for complex tasks
    if complexity in (TaskComplexity.COMPLEX, TaskComplexity.EXPERT):
        ctx = profile.context_window or 0
        if ctx >= 200_000:
            score += 5
        elif ctx >= 100_000:
            score += 3

    return score


def _complexity_fit_bonus(profile: ModelProfile, complexity: TaskComplexity) -> float:
    """Bonus for how well a model's tier matches the task complexity."""
    cost = _cost_rank(profile.cost_tier)

    complexity_to_ideal_cost: dict[TaskComplexity, int] = {
        TaskComplexity.SIMPLE: 0,  # FREE
        TaskComplexity.MODERATE: 1,  # LOW
        TaskComplexity.COMPLEX: 3,  # HIGH
        TaskComplexity.EXPERT: 4,  # PREMIUM
    }

    ideal = complexity_to_ideal_cost[complexity]
    distance = abs(cost - ideal)

    # Closer to ideal = higher bonus
    if distance == 0:
        return 15.0
    elif distance == 1:
        return 10.0
    elif distance == 2:
        return 5.0
    else:
        return 0.0


# ---------------------------------------------------------------------------
# Router class
# ---------------------------------------------------------------------------


class Router:
    """Configurable model router for hybrid local/cloud routing.

    The router maintains a pool of candidate models and selects the best one
    for a given task based on the configured strategy, required capabilities,
    and cost constraints.

    Parameters
    ----------
    strategy
        Default routing strategy. Can be overridden per-call.
    candidates
        List of model keys (e.g., ``"anthropic:claude-sonnet-4-6"``) to consider.
        If ``None``, uses all models in the registry.
    max_cost_tier
        Maximum cost tier to consider. Models above this tier are excluded.
    prefer_local
        Whether to give bonus scoring to local (Ollama) models.

    Examples
    --------
    ```python
    import talk_box as tb

    router = tb.Router(
        strategy=tb.RoutingStrategy.LOCAL_FIRST,
        max_cost_tier=tb.CostTier.MEDIUM,
    )

    result = router.route("What is a decorator in Python?")
    result.model.name       # e.g., "Llama 3.3 (Ollama)"
    result.complexity       # TaskComplexity.SIMPLE
    result.reason           # "Local model sufficient for simple task"
    ```
    """

    def __init__(
        self,
        *,
        strategy: RoutingStrategy = RoutingStrategy.BALANCED,
        candidates: list[str] | None = None,
        max_cost_tier: CostTier | None = None,
        prefer_local: bool = False,
    ) -> None:
        self._strategy = strategy
        self._max_cost_tier = max_cost_tier
        self._prefer_local = prefer_local

        if candidates is not None:
            self._candidates = [
                p for key in candidates if (p := get_model_profile(key)) is not None
            ]
        else:
            self._candidates = list_models()

    @property
    def strategy(self) -> RoutingStrategy:
        """The default routing strategy."""
        return self._strategy

    @property
    def candidates(self) -> list[ModelProfile]:
        """The pool of candidate models."""
        return list(self._candidates)

    def route(
        self,
        prompt: str,
        *,
        strategy: RoutingStrategy | None = None,
        requires: list[str] | None = None,
        max_cost_tier: CostTier | None = None,
        min_context_window: int | None = None,
    ) -> RoutingResult:
        """Route a task to the best available model.

        Parameters
        ----------
        prompt
            The task or prompt text to route.
        strategy
            Override the router's default strategy for this call.
        requires
            Required capabilities (e.g., ``["tools", "vision"]``).
        max_cost_tier
            Override the router's default max cost tier for this call.
        min_context_window
            Minimum context window size required.

        Returns
        -------
        RoutingResult
            The selected model, reason, complexity, and alternatives.

        Raises
        ------
        ValueError
            If no candidate models meet the requirements.

        Examples
        --------
        ```python
        import talk_box as tb

        router = tb.Router()

        # Route a simple question cheaply
        result = router.route("Define polymorphism")

        # Route a complex task requiring tools
        result = router.route(
            "Analyze this dataset and generate charts",
            requires=["tools", "vision"],
            strategy=tb.RoutingStrategy.QUALITY_OPTIMIZED,
        )
        ```
        """
        effective_strategy = strategy or self._strategy
        effective_max_cost = max_cost_tier or self._max_cost_tier

        # 1. Classify complexity
        complexity = classify_complexity(prompt)

        # 2. Filter candidates
        filtered = self._filter_candidates(
            requires=requires,
            max_cost_tier=effective_max_cost,
            min_context_window=min_context_window,
        )

        if not filtered:
            raise ValueError(
                "No candidate models meet the routing requirements. "
                "Relax constraints or add more candidates."
            )

        # 3. Apply local preference if set
        actual_strategy = effective_strategy
        if self._prefer_local and effective_strategy == RoutingStrategy.BALANCED:
            actual_strategy = RoutingStrategy.LOCAL_FIRST

        # 4. Score and rank
        scored = [(_score_model(p, complexity, actual_strategy), p) for p in filtered]
        scored.sort(key=lambda x: x[0], reverse=True)

        best_profile = scored[0][1]
        alternatives = [p for _, p in scored[1:5]]  # Top 4 alternatives

        # 5. Generate reason
        reason = _explain_choice(best_profile, complexity, actual_strategy)

        return RoutingResult(
            model=best_profile,
            reason=reason,
            complexity=complexity,
            alternatives=alternatives,
        )

    def _filter_candidates(
        self,
        *,
        requires: list[str] | None,
        max_cost_tier: CostTier | None,
        min_context_window: int | None,
    ) -> list[ModelProfile]:
        """Filter candidate models by requirements."""
        results = list(self._candidates)

        # Capability filter
        if requires:
            results = [p for p in results if all(p.supports(cap) is True for cap in requires)]

        # Cost filter
        if max_cost_tier is not None:
            max_rank = _cost_rank(max_cost_tier)
            results = [p for p in results if _cost_rank(p.cost_tier) <= max_rank]

        # Context window filter
        if min_context_window is not None:
            results = [
                p
                for p in results
                if p.context_window is not None and p.context_window >= min_context_window
            ]

        return results


# ---------------------------------------------------------------------------
# Explanation helper
# ---------------------------------------------------------------------------


def _explain_choice(
    profile: ModelProfile,
    complexity: TaskComplexity,
    strategy: RoutingStrategy,
) -> str:
    """Generate a human-readable reason for the routing decision."""
    parts: list[str] = []

    # Strategy context
    strategy_desc = {
        RoutingStrategy.COST_OPTIMIZED: "cost-optimized",
        RoutingStrategy.QUALITY_OPTIMIZED: "quality-optimized",
        RoutingStrategy.LOCAL_FIRST: "local-first",
        RoutingStrategy.BALANCED: "balanced",
    }
    parts.append(f"{strategy_desc[strategy]} routing")

    # Complexity context
    parts.append(f"{complexity.value} task")

    # Model characteristics
    if profile.provider == "ollama":
        parts.append("local model (free, no network)")
    elif profile.cost_tier == CostTier.FREE:
        parts.append("free tier")
    elif profile.cost_tier:
        parts.append(f"{profile.cost_tier.value} cost")

    return f"{profile.name}: {'; '.join(parts)}"


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------


def route(
    prompt: str,
    *,
    strategy: RoutingStrategy = RoutingStrategy.BALANCED,
    requires: list[str] | None = None,
    max_cost_tier: CostTier | None = None,
    prefer_local: bool = False,
    candidates: list[str] | None = None,
    min_context_window: int | None = None,
) -> RoutingResult:
    """Route a task to the best available model.

    This is a convenience function that creates a temporary `Router` instance.
    For repeated routing, create a `Router` directly to avoid re-filtering
    the candidate pool each time.

    Parameters
    ----------
    prompt
        The task or prompt text to route.
    strategy
        Routing strategy to use.
    requires
        Required capabilities (e.g., ``["tools", "vision"]``).
    max_cost_tier
        Maximum cost tier to consider.
    prefer_local
        Whether to prefer local (Ollama) models.
    candidates
        Specific model keys to consider. ``None`` uses all registered models.
    min_context_window
        Minimum context window size required.

    Returns
    -------
    RoutingResult
        The selected model, reason, complexity, and alternatives.

    Raises
    ------
    ValueError
        If no candidate models meet the requirements.

    Examples
    --------
    ```python
    import talk_box as tb

    # Simple routing with defaults
    result = tb.route("Summarize this paragraph")
    result.model.name  # e.g., "GPT-4o Mini"

    # Prefer local models, cap cost at MEDIUM
    result = tb.route(
        "Explain decorators in Python",
        strategy=tb.RoutingStrategy.LOCAL_FIRST,
        max_cost_tier=tb.CostTier.MEDIUM,
    )

    # Require specific capabilities
    result = tb.route(
        "Analyze this image and describe what you see",
        requires=["vision"],
        strategy=tb.RoutingStrategy.QUALITY_OPTIMIZED,
    )
    ```
    """
    router = Router(
        strategy=strategy,
        candidates=candidates,
        max_cost_tier=max_cost_tier,
        prefer_local=prefer_local,
    )
    return router.route(
        prompt,
        requires=requires,
        min_context_window=min_context_window,
    )
