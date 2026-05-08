from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from talk_box.capture import ConversationCapture

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class TurnStatus(Enum):
    """Classification of how a turn differs between two captures.

    Attributes
    ----------
    IDENTICAL
        The responses are textually identical.
    SIMILAR
        The responses share significant word overlap (above the similarity
        threshold) but are not identical.
    CHANGED
        The responses differ significantly (below the similarity threshold).
    ADDED
        The turn exists only in the right capture.
    REMOVED
        The turn exists only in the left capture.
    """

    IDENTICAL = "identical"
    SIMILAR = "similar"
    CHANGED = "changed"
    ADDED = "added"
    REMOVED = "removed"


@dataclass(frozen=True)
class TurnDiff:
    """Comparison of a single turn between two captures.

    Parameters
    ----------
    turn_number
        The turn number (1-based).
    prompt
        The prompt text (from the left capture, or right if added).
    left_response
        The response from the left capture (empty if added).
    right_response
        The response from the right capture (empty if removed).
    status
        How this turn differs.
    similarity
        Word-level Jaccard similarity between the two responses (0.0-1.0).
        `1.0` for identical, `0.0` when one side is missing.
    left_model
        Model used in the left capture.
    right_model
        Model used in the right capture.
    """

    turn_number: int
    prompt: str
    left_response: str
    right_response: str
    status: TurnStatus
    similarity: float
    left_model: str = ""
    right_model: str = ""


@dataclass(frozen=True)
class DiffResult:
    """Result of comparing two conversation captures.

    Parameters
    ----------
    left_session_id
        Session ID of the left (baseline) capture.
    right_session_id
        Session ID of the right (comparison) capture.
    turns
        Turn-by-turn comparisons.
    similarity_threshold
        The threshold used to classify `SIMILAR` vs. `CHANGED`.
    metadata
        Additional metadata about the diff.
    """

    left_session_id: str
    right_session_id: str
    turns: list[TurnDiff]
    similarity_threshold: float = 0.7
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def turn_count(self) -> int:
        """Total number of turns compared."""
        return len(self.turns)

    @property
    def identical_count(self) -> int:
        """Number of turns with identical responses."""
        return sum(1 for t in self.turns if t.status == TurnStatus.IDENTICAL)

    @property
    def similar_count(self) -> int:
        """Number of turns with similar (but not identical) responses."""
        return sum(1 for t in self.turns if t.status == TurnStatus.SIMILAR)

    @property
    def changed_count(self) -> int:
        """Number of turns with significantly changed responses."""
        return sum(1 for t in self.turns if t.status == TurnStatus.CHANGED)

    @property
    def added_count(self) -> int:
        """Number of turns only in the right capture."""
        return sum(1 for t in self.turns if t.status == TurnStatus.ADDED)

    @property
    def removed_count(self) -> int:
        """Number of turns only in the left capture."""
        return sum(1 for t in self.turns if t.status == TurnStatus.REMOVED)

    @property
    def similarity_score(self) -> float:
        """Average similarity across all turns (0.0–1.0)."""
        if not self.turns:
            return 1.0
        return sum(t.similarity for t in self.turns) / len(self.turns)

    def summary(self) -> dict[str, Any]:
        """Return a summary dictionary of the diff.

        Returns
        -------
        dict[str, Any]
            Summary with counts and overall similarity.
        """
        return {
            "left_session_id": self.left_session_id,
            "right_session_id": self.right_session_id,
            "turn_count": self.turn_count,
            "identical": self.identical_count,
            "similar": self.similar_count,
            "changed": self.changed_count,
            "added": self.added_count,
            "removed": self.removed_count,
            "similarity_score": round(self.similarity_score, 4),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _word_set(text: str) -> set[str]:
    """Extract a set of lowercase words from text."""
    return set(text.lower().split())


def _jaccard(text_a: str, text_b: str) -> float:
    """Compute Jaccard similarity between two texts (word-level)."""
    words_a = _word_set(text_a)
    words_b = _word_set(text_b)
    if not words_a and not words_b:
        return 1.0
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def _classify(similarity: float, threshold: float) -> TurnStatus:
    """Classify a turn based on similarity score."""
    if similarity == 1.0:
        return TurnStatus.IDENTICAL
    if similarity >= threshold:
        return TurnStatus.SIMILAR
    return TurnStatus.CHANGED


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def diff(
    left: ConversationCapture,
    right: ConversationCapture,
    *,
    similarity_threshold: float = 0.7,
    metadata: dict[str, Any] | None = None,
) -> DiffResult:
    """Compare two conversation captures turn by turn.

    Extracts user prompts and their corresponding responses from each capture,
    then aligns them by position and computes word-level Jaccard similarity
    for each turn pair.

    Parameters
    ----------
    left
        The baseline (left) capture.
    right
        The comparison (right) capture.
    similarity_threshold
        Minimum Jaccard similarity to classify a turn as `SIMILAR` rather
        than `CHANGED`. Default `0.7`.
    metadata
        Additional metadata to attach to the result.

    Returns
    -------
    DiffResult
        Turn-by-turn comparison with status classifications and similarity
        scores.

    Examples
    --------
    ```python
    import talk_box as tb

    # Compare an original session with a replay
    original = tb.ConversationCapture.from_json("session.json")
    replayed = result.capture  # from tb.replay()

    result = tb.diff(original, replayed)

    print(f"Overall similarity: {result.similarity_score:.1%}")
    for turn in result.turns:
        print(f"Turn {turn.turn_number}: {turn.status.value} ({turn.similarity:.1%})")

    # Quick summary
    print(result.summary())
    ```
    """
    # Extract user prompts (exclude system prompts) and responses
    left_user_prompts = [e for e in left.prompts() if e.role != "system"]
    right_user_prompts = [e for e in right.prompts() if e.role != "system"]
    left_responses = left.responses()
    right_responses = right.responses()

    # Align turns by position
    max_turns = max(len(left_user_prompts), len(right_user_prompts))
    turn_diffs: list[TurnDiff] = []

    for i in range(max_turns):
        has_left = i < len(left_user_prompts)
        has_right = i < len(right_user_prompts)

        if has_left and has_right:
            # Both sides have this turn
            prompt = left_user_prompts[i].content
            left_resp = left_responses[i].content if i < len(left_responses) else ""
            right_resp = right_responses[i].content if i < len(right_responses) else ""
            left_model = left_responses[i].model if i < len(left_responses) else ""
            right_model = right_responses[i].model if i < len(right_responses) else ""

            sim = _jaccard(left_resp, right_resp)
            status = _classify(sim, similarity_threshold)

        elif has_left:
            # Turn only in left (removed)
            prompt = left_user_prompts[i].content
            left_resp = left_responses[i].content if i < len(left_responses) else ""
            right_resp = ""
            left_model = left_responses[i].model if i < len(left_responses) else ""
            right_model = ""
            sim = 0.0
            status = TurnStatus.REMOVED

        else:
            # Turn only in right (added)
            prompt = right_user_prompts[i].content
            left_resp = ""
            right_resp = right_responses[i].content if i < len(right_responses) else ""
            left_model = ""
            right_model = right_responses[i].model if i < len(right_responses) else ""
            sim = 0.0
            status = TurnStatus.ADDED

        turn_diffs.append(
            TurnDiff(
                turn_number=i + 1,
                prompt=prompt,
                left_response=left_resp,
                right_response=right_resp,
                status=status,
                similarity=sim,
                left_model=left_model,
                right_model=right_model,
            )
        )

    return DiffResult(
        left_session_id=left.session_id,
        right_session_id=right.session_id,
        turns=turn_diffs,
        similarity_threshold=similarity_threshold,
        metadata=dict(metadata or {}),
    )
