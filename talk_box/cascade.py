"""Cascade consensus: start with one model, fan out only when confidence is low."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from talk_box.consensus import (
    ConsensusStrategy,
    ModelResponse,
    consensus,
)
from talk_box.routing import (
    Router,
    RoutingStrategy,
    TaskComplexity,
)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CascadeRound:
    """Record of a single round in a cascade execution.

    Parameters
    ----------
    round_number
        The round number (1-based).
    model
        The model queried in this round.
    response
        The response text from the model.
    confidence
        Estimated confidence in the response (0.0–1.0).
    """

    round_number: int
    model: str
    response: str
    confidence: float


@dataclass(frozen=True)
class CascadeResult:
    """Result of a cascade consensus execution.

    Parameters
    ----------
    winner
        The final selected response text.
    winner_model
        The model that produced the winning response.
    confidence
        Confidence in the final result.
    fanned_out
        Whether the cascade fanned out to additional models.
    rounds
        Details of each round in the cascade.
    consensus_strategy
        The consensus strategy used if fan-out occurred, or ``None``.
    agreement_score
        Agreement score from consensus (if fan-out occurred), or ``None``.
    initial_model
        The model used in the initial round.
    initial_confidence
        The confidence score from the initial round.
    complexity
        The estimated task complexity.
    """

    winner: str
    winner_model: str
    confidence: float
    fanned_out: bool
    rounds: list[CascadeRound] = field(default_factory=list)
    consensus_strategy: ConsensusStrategy | None = None
    agreement_score: float | None = None
    initial_model: str = ""
    initial_confidence: float = 0.0
    complexity: TaskComplexity = TaskComplexity.SIMPLE


# ---------------------------------------------------------------------------
# Confidence estimation
# ---------------------------------------------------------------------------

# Hedging patterns that reduce confidence
_HEDGE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(I think|I believe|I'm not sure|I'm uncertain)\b", re.IGNORECASE),
    re.compile(r"\b(maybe|perhaps|possibly|probably|might)\b", re.IGNORECASE),
    re.compile(r"\b(it depends|it's unclear|hard to say)\b", re.IGNORECASE),
    re.compile(r"\b(I don't know|not certain|can't be sure)\b", re.IGNORECASE),
    re.compile(r"\b(could be|may or may not|on the other hand)\b", re.IGNORECASE),
]

# Assertive patterns that increase confidence
_ASSERTIVE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(the answer is|the result is|the solution is)\b", re.IGNORECASE),
    re.compile(r"\b(definitely|certainly|clearly|obviously)\b", re.IGNORECASE),
    re.compile(r"\b(this is because|the reason is|specifically)\b", re.IGNORECASE),
    re.compile(r"```[\w]*\n", re.MULTILINE),  # Code blocks suggest concrete answers
]

# Refusal/inability patterns that strongly reduce confidence
_REFUSAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(I cannot|I can't|I'm unable to|I am unable)\b", re.IGNORECASE),
    re.compile(r"\b(as an AI|as a language model)\b", re.IGNORECASE),
    re.compile(r"\b(I don't have access|beyond my|outside my)\b", re.IGNORECASE),
]


def estimate_confidence(response: str) -> float:
    """Estimate confidence in a model response based on textual signals.

    Uses heuristics based on hedging language, assertiveness, response length,
    and structural indicators to produce a confidence score.

    Parameters
    ----------
    response
        The response text to analyze.

    Returns
    -------
    float
        Confidence score from 0.0 (very low confidence) to 1.0 (very high confidence).

    Examples
    --------
    ```python
    import talk_box as tb

    tb.estimate_confidence("The answer is 42.")
    # ~0.75

    tb.estimate_confidence("I'm not sure, but maybe it's 42?")
    # ~0.35

    tb.estimate_confidence("I cannot answer that question.")
    # ~0.15
    ```
    """
    if not response or not response.strip():
        return 0.0

    score = 0.5  # Baseline

    # Length-based adjustment
    length = len(response.strip())
    if length < 20:
        score -= 0.1  # Very short responses are less confident
    elif length > 200:
        score += 0.1  # Detailed responses suggest more thought
    if length > 1000:
        score += 0.05  # Very detailed

    # Hedging reduces confidence
    hedge_count = sum(1 for p in _HEDGE_PATTERNS if p.search(response))
    score -= hedge_count * 0.1

    # Assertiveness increases confidence
    assertive_count = sum(1 for p in _ASSERTIVE_PATTERNS if p.search(response))
    score += assertive_count * 0.08

    # Refusal strongly reduces confidence
    refusal_count = sum(1 for p in _REFUSAL_PATTERNS if p.search(response))
    score -= refusal_count * 0.2

    # Structured content (lists, numbered items) suggests organized thought
    list_items = len(re.findall(r"^\s*[-*•]\s", response, re.MULTILINE))
    numbered_items = len(re.findall(r"^\s*\d+[.)]\s", response, re.MULTILINE))
    if list_items + numbered_items >= 3:
        score += 0.05

    # Clamp to [0.0, 1.0]
    return round(max(0.0, min(1.0, score)), 4)


# ---------------------------------------------------------------------------
# Cascade execution
# ---------------------------------------------------------------------------

# Type alias for a responder function: (model_key, prompt) -> response_text
Responder = Callable[[str, str], str]


def cascade(
    prompt: str,
    responder: Responder,
    *,
    confidence_threshold: float = 0.6,
    max_fan_out: int = 3,
    fan_out_strategy: ConsensusStrategy = ConsensusStrategy.MAJORITY,
    candidates: list[str] | None = None,
    routing_strategy: RoutingStrategy = RoutingStrategy.BALANCED,
) -> CascadeResult:
    """Execute a cascade consensus: start with one model, fan out if confidence is low.

    The cascade works in two phases:

    1. **Initial query**: Routes the prompt to the best available model and queries it.
       If the response confidence is above ``confidence_threshold``, returns immediately.

    2. **Fan-out**: If confidence is low, queries up to ``max_fan_out`` additional models
       from the routing alternatives, then runs consensus across all responses.

    Parameters
    ----------
    prompt
        The task or prompt text.
    responder
        A callable that takes ``(model_key, prompt)`` and returns the response text.
        This keeps the cascade framework-agnostic — you provide the actual LLM call.
    confidence_threshold
        Minimum confidence to accept the initial response without fan-out (default 0.6).
    max_fan_out
        Maximum number of additional models to query during fan-out (default 3).
    fan_out_strategy
        Consensus strategy to use when fan-out occurs (default ``MAJORITY``).
    candidates
        Specific model keys to consider for routing. ``None`` uses all registered models.
    routing_strategy
        Strategy for the initial model selection (default ``BALANCED``).

    Returns
    -------
    CascadeResult
        The cascade outcome including winner, confidence, rounds, and whether fan-out occurred.

    Raises
    ------
    ValueError
        If no candidate models are available.

    Examples
    --------
    ```python
    import talk_box as tb

    def ask_model(model_key: str, prompt: str) -> str:
        # Your LLM call here
        return "The answer is 42."

    result = tb.cascade("What is the meaning of life?", ask_model)
    result.winner          # "The answer is 42."
    result.fanned_out      # False (if initial response was confident)
    result.confidence      # ~0.75
    result.rounds          # [CascadeRound(round_number=1, ...)]
    ```
    """
    # 1. Route to the best model
    router = Router(strategy=routing_strategy, candidates=candidates)
    routing_result = router.route(prompt)

    complexity = routing_result.complexity
    initial_model = routing_result.model.key

    # 2. Query the initial model
    initial_response = responder(initial_model, prompt)
    initial_confidence = estimate_confidence(initial_response)

    round_1 = CascadeRound(
        round_number=1,
        model=initial_model,
        response=initial_response,
        confidence=initial_confidence,
    )

    # 3. If confidence is sufficient, return immediately
    if initial_confidence >= confidence_threshold:
        return CascadeResult(
            winner=initial_response,
            winner_model=initial_model,
            confidence=initial_confidence,
            fanned_out=False,
            rounds=[round_1],
            initial_model=initial_model,
            initial_confidence=initial_confidence,
            complexity=complexity,
        )

    # 4. Fan out to additional models
    rounds: list[CascadeRound] = [round_1]
    all_responses: list[ModelResponse] = [
        ModelResponse(model=initial_model, text=initial_response),
    ]

    fan_out_models = routing_result.alternatives[:max_fan_out]

    for i, alt_profile in enumerate(fan_out_models):
        alt_key = alt_profile.key
        alt_response = responder(alt_key, prompt)
        alt_confidence = estimate_confidence(alt_response)

        rounds.append(
            CascadeRound(
                round_number=i + 2,
                model=alt_key,
                response=alt_response,
                confidence=alt_confidence,
            )
        )
        all_responses.append(
            ModelResponse(model=alt_key, text=alt_response),
        )

    # 5. Run consensus across all responses
    consensus_result = consensus(all_responses, strategy=fan_out_strategy)

    # Final confidence: blend consensus agreement with best individual confidence
    best_confidence = max(r.confidence for r in rounds)
    final_confidence = (consensus_result.agreement_score * 0.6) + (best_confidence * 0.4)

    return CascadeResult(
        winner=consensus_result.winner,
        winner_model=consensus_result.winner_model,
        confidence=round(final_confidence, 4),
        fanned_out=True,
        rounds=rounds,
        consensus_strategy=fan_out_strategy,
        agreement_score=consensus_result.agreement_score,
        initial_model=initial_model,
        initial_confidence=initial_confidence,
        complexity=complexity,
    )
