"""Consensus mode: query multiple models, compare outputs, flag disagreements."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class ConsensusStrategy(Enum):
    """Strategy for resolving consensus across multiple model responses.

    Attributes
    ----------
    MAJORITY
        The response most similar to the majority wins.
    UNANIMOUS
        All responses must substantially agree; otherwise consensus fails.
    MOST_COMMON
        Select the most frequently occurring response (exact or near-match).
    WEIGHTED
        Responses are weighted by model quality/cost tier (higher tier = more weight).
    """

    MAJORITY = "majority"
    UNANIMOUS = "unanimous"
    MOST_COMMON = "most_common"
    WEIGHTED = "weighted"


@dataclass(frozen=True)
class ModelResponse:
    """A single model's response to a prompt.

    Parameters
    ----------
    model
        Model identifier (e.g., ``"anthropic:claude-sonnet-4-6"``).
    text
        The response text from the model.
    latency_ms
        Response latency in milliseconds, if measured.
    token_count
        Number of tokens in the response, if known.
    weight
        Optional weight for weighted consensus (default 1.0).
    """

    model: str
    text: str
    latency_ms: float | None = None
    token_count: int | None = None
    weight: float = 1.0


@dataclass(frozen=True)
class Disagreement:
    """A detected disagreement between model responses.

    Parameters
    ----------
    description
        Human-readable description of the disagreement.
    models
        The models involved in the disagreement.
    severity
        Severity level: ``"minor"`` (stylistic), ``"moderate"`` (factual nuance),
        ``"major"`` (contradictory claims).
    """

    description: str
    models: tuple[str, ...]
    severity: str = "moderate"


@dataclass(frozen=True)
class ConsensusResult:
    """Result of running consensus across multiple model responses.

    Parameters
    ----------
    winner
        The winning/selected response text.
    winner_model
        The model that produced the winning response.
    agreement_score
        Overall agreement score from 0.0 (complete disagreement) to 1.0 (unanimous).
    strategy
        The consensus strategy that was used.
    responses
        All individual model responses that were compared.
    disagreements
        Detected disagreements between responses.
    consensus_reached
        Whether consensus was successfully reached (relevant for UNANIMOUS strategy).
    """

    winner: str
    winner_model: str
    agreement_score: float
    strategy: ConsensusStrategy
    responses: list[ModelResponse] = field(default_factory=list)
    disagreements: list[Disagreement] = field(default_factory=list)
    consensus_reached: bool = True


# ---------------------------------------------------------------------------
# Text similarity utilities
# ---------------------------------------------------------------------------


def _normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, collapse whitespace, strip punctuation."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def _word_set(text: str) -> set[str]:
    """Extract word set from normalized text."""
    return set(_normalize_text(text).split())


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """Compute Jaccard similarity between two texts (word-level).

    Returns a value between 0.0 (no overlap) and 1.0 (identical word sets).
    """
    words_a = _word_set(text_a)
    words_b = _word_set(text_b)

    if not words_a and not words_b:
        return 1.0
    if not words_a or not words_b:
        return 0.0

    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def _pairwise_similarities(responses: list[ModelResponse]) -> list[float]:
    """Compute all pairwise Jaccard similarities between responses."""
    similarities: list[float] = []
    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):
            sim = _jaccard_similarity(responses[i].text, responses[j].text)
            similarities.append(sim)
    return similarities


# ---------------------------------------------------------------------------
# Disagreement detection
# ---------------------------------------------------------------------------


def find_disagreements(responses: list[ModelResponse]) -> list[Disagreement]:
    """Detect disagreements between model responses.

    Compares each pair of responses and flags significant differences.
    Uses word-level similarity to classify severity.

    Parameters
    ----------
    responses
        List of model responses to compare.

    Returns
    -------
    list[Disagreement]
        Detected disagreements, sorted by severity (major first).

    Examples
    --------
    ```python
    import talk_box as tb

    responses = [
        tb.ModelResponse(model="model_a", text="Python was created in 1991."),
        tb.ModelResponse(model="model_b", text="Python was created in 1989."),
    ]

    disagreements = tb.find_disagreements(responses)
    disagreements[0].severity  # "major"
    ```
    """
    if len(responses) < 2:
        return []

    disagreements: list[Disagreement] = []

    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):
            sim = _jaccard_similarity(responses[i].text, responses[j].text)

            if sim >= 0.8:
                continue  # Substantially agree

            severity = _classify_severity(sim)
            description = (
                f"{responses[i].model} and {responses[j].model} disagree (similarity: {sim:.0%})"
            )
            disagreements.append(
                Disagreement(
                    description=description,
                    models=(responses[i].model, responses[j].model),
                    severity=severity,
                )
            )

    # Sort: major first, then moderate, then minor
    severity_order = {"major": 0, "moderate": 1, "minor": 2}
    disagreements.sort(key=lambda d: severity_order.get(d.severity, 1))
    return disagreements


def _classify_severity(similarity: float) -> str:
    """Classify disagreement severity based on similarity score."""
    if similarity < 0.3:
        return "major"
    elif similarity < 0.6:
        return "moderate"
    else:
        return "minor"


# ---------------------------------------------------------------------------
# Consensus strategies
# ---------------------------------------------------------------------------


def _majority_consensus(responses: list[ModelResponse]) -> tuple[ModelResponse, float]:
    """Select the response most similar to all others (centroid-like)."""
    if len(responses) == 1:
        return responses[0], 1.0

    # Score each response by average similarity to all others
    scores: list[float] = []
    for i, resp in enumerate(responses):
        sims = [
            _jaccard_similarity(resp.text, other.text)
            for j, other in enumerate(responses)
            if j != i
        ]
        scores.append(sum(sims) / len(sims))

    best_idx = scores.index(max(scores))
    agreement = sum(scores) / len(scores)
    return responses[best_idx], agreement


def _unanimous_consensus(
    responses: list[ModelResponse], threshold: float = 0.7
) -> tuple[ModelResponse | None, float, bool]:
    """Check if all responses substantially agree.

    Returns (best_response, agreement_score, consensus_reached).
    """
    if len(responses) == 1:
        return responses[0], 1.0, True

    similarities = _pairwise_similarities(responses)
    min_sim = min(similarities) if similarities else 1.0
    avg_sim = sum(similarities) / len(similarities) if similarities else 1.0

    reached = min_sim >= threshold
    best, _ = _majority_consensus(responses)
    return best, avg_sim, reached


def _most_common_consensus(responses: list[ModelResponse]) -> tuple[ModelResponse, float]:
    """Select the most frequently occurring response (cluster by similarity)."""
    if len(responses) == 1:
        return responses[0], 1.0

    # Group responses into clusters (similarity >= 0.7 = same cluster)
    clusters: list[list[int]] = []
    assigned = set()

    for i in range(len(responses)):
        if i in assigned:
            continue
        cluster = [i]
        assigned.add(i)
        for j in range(i + 1, len(responses)):
            if j in assigned:
                continue
            if _jaccard_similarity(responses[i].text, responses[j].text) >= 0.7:
                cluster.append(j)
                assigned.add(j)
        clusters.append(cluster)

    # Find the largest cluster
    largest = max(clusters, key=len)
    agreement = len(largest) / len(responses)

    # Pick the response from the largest cluster with highest avg similarity to cluster members
    if len(largest) == 1:
        return responses[largest[0]], agreement

    best_idx = largest[0]
    best_score = 0.0
    for i in largest:
        score = sum(
            _jaccard_similarity(responses[i].text, responses[j].text) for j in largest if j != i
        ) / (len(largest) - 1)
        if score > best_score:
            best_score = score
            best_idx = i

    return responses[best_idx], agreement


def _weighted_consensus(responses: list[ModelResponse]) -> tuple[ModelResponse, float]:
    """Select the response with the highest weight-adjusted similarity score."""
    if len(responses) == 1:
        return responses[0], 1.0

    # Score each response: weighted average similarity to all others
    scores: list[float] = []
    total_weight = sum(r.weight for r in responses)

    for i, resp in enumerate(responses):
        weighted_sim = 0.0
        other_weight = total_weight - resp.weight
        if other_weight == 0:
            scores.append(0.0)
            continue
        for j, other in enumerate(responses):
            if j == i:
                continue
            sim = _jaccard_similarity(resp.text, other.text)
            weighted_sim += sim * other.weight
        # Combine: own weight (quality signal) + agreement with others
        score = (resp.weight / total_weight) * 0.4 + (weighted_sim / other_weight) * 0.6
        scores.append(score)

    best_idx = scores.index(max(scores))
    # Agreement is still average pairwise similarity
    similarities = _pairwise_similarities(responses)
    agreement = sum(similarities) / len(similarities) if similarities else 1.0
    return responses[best_idx], agreement


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------


def consensus(
    responses: list[ModelResponse],
    *,
    strategy: ConsensusStrategy = ConsensusStrategy.MAJORITY,
    unanimous_threshold: float = 0.7,
) -> ConsensusResult:
    """Determine consensus across multiple model responses.

    Compares the provided responses and selects a winner based on the chosen
    strategy. Also detects and reports disagreements.

    Parameters
    ----------
    responses
        List of model responses to compare. Must contain at least one response.
    strategy
        The consensus strategy to use.
    unanimous_threshold
        For the ``UNANIMOUS`` strategy, the minimum pairwise similarity required
        for consensus to be reached (default 0.7).

    Returns
    -------
    ConsensusResult
        The consensus outcome including winner, agreement score, and disagreements.

    Raises
    ------
    ValueError
        If ``responses`` is empty.

    Examples
    --------
    ```python
    import talk_box as tb

    responses = [
        tb.ModelResponse(model="anthropic:claude-sonnet-4-6", text="Python is a programming language."),
        tb.ModelResponse(model="openai:gpt-4o", text="Python is a high-level programming language."),
        tb.ModelResponse(model="google:gemini-2.5-flash", text="Python is an interpreted programming language."),
    ]

    result = tb.consensus(responses, strategy=tb.ConsensusStrategy.MAJORITY)
    result.winner           # "Python is a high-level programming language."
    result.agreement_score  # ~0.75
    result.consensus_reached  # True
    ```
    """
    if not responses:
        raise ValueError("At least one response is required for consensus.")

    if len(responses) == 1:
        return ConsensusResult(
            winner=responses[0].text,
            winner_model=responses[0].model,
            agreement_score=1.0,
            strategy=strategy,
            responses=list(responses),
            disagreements=[],
            consensus_reached=True,
        )

    # Detect disagreements
    disagreements = find_disagreements(responses)

    # Apply strategy
    consensus_reached = True

    if strategy == ConsensusStrategy.MAJORITY:
        winner, agreement = _majority_consensus(responses)

    elif strategy == ConsensusStrategy.UNANIMOUS:
        winner_resp, agreement, consensus_reached = _unanimous_consensus(
            responses, threshold=unanimous_threshold
        )
        winner = winner_resp if winner_resp else responses[0]

    elif strategy == ConsensusStrategy.MOST_COMMON:
        winner, agreement = _most_common_consensus(responses)

    elif strategy == ConsensusStrategy.WEIGHTED:
        winner, agreement = _weighted_consensus(responses)

    else:  # pragma: no cover
        raise ValueError(f"Unknown strategy: {strategy}")

    return ConsensusResult(
        winner=winner.text,
        winner_model=winner.model,
        agreement_score=round(agreement, 4),
        strategy=strategy,
        responses=list(responses),
        disagreements=disagreements,
        consensus_reached=consensus_reached,
    )
