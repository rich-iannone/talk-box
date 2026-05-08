from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ReviewStatus(Enum):
    """Status of a human-in-the-loop review.

    Attributes
    ----------
    PENDING
        The review is awaiting a human decision.
    APPROVED
        The human approved the content.
    REJECTED
        The human rejected the content.
    REVISED
        The human revised the content before approving.
    """

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    REVISED = "revised"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReviewDecision:
    """A human's decision on a review request.

    Parameters
    ----------
    status
        The decision: approved, rejected, or revised.
    feedback
        Optional human feedback explaining the decision.
    revised_content
        Replacement content when the decision is ``REVISED``.
    decided_at
        Unix timestamp when the decision was made.
    decided_by
        Identifier for the human who made the decision.
    """

    status: ReviewStatus
    feedback: str = ""
    revised_content: str | None = None
    decided_at: float = 0.0
    decided_by: str = ""


@dataclass
class HumanReview:
    """A request for human review before proceeding.

    `HumanReview` captures content produced by an agent (or a pathway state) that requires human
    approval, rejection, or revision before the workflow continues.

    Parameters
    ----------
    content
        The content to be reviewed.
    agent
        Name of the agent that produced the content.
    context
        Additional context for the reviewer (e.g., task description, instructions).
    state
        The pathway state name where the review was requested, if applicable.
    metadata
        Arbitrary metadata attached to the review.
    created_at
        Unix timestamp when the review was created.

    Examples
    --------
    Create a review, then approve it:

    ```python
    import talk_box as tb

    review = tb.human_review("Deploy to production?", agent="deploy_bot")
    review.status          # ReviewStatus.PENDING
    review.is_pending      # True

    decision = tb.approve(review, feedback="Looks good")
    decision.status        # ReviewStatus.APPROVED
    review.is_resolved     # True
    ```
    """

    content: str
    agent: str = ""
    context: str = ""
    state: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    # Mutable decision state
    _decision: ReviewDecision | None = field(default=None, init=False, repr=False)

    @property
    def status(self) -> ReviewStatus:
        """Current status of this review."""
        if self._decision is None:
            return ReviewStatus.PENDING
        return self._decision.status

    @property
    def decision(self) -> ReviewDecision | None:
        """The human's decision, or ``None`` if still pending."""
        return self._decision

    @property
    def is_pending(self) -> bool:
        """Whether the review is still awaiting a decision."""
        return self._decision is None

    @property
    def is_resolved(self) -> bool:
        """Whether the review has been decided (approved, rejected, or revised)."""
        return self._decision is not None

    @property
    def is_approved(self) -> bool:
        """Whether the review was approved (including revised)."""
        if self._decision is None:
            return False
        return self._decision.status in (ReviewStatus.APPROVED, ReviewStatus.REVISED)

    @property
    def final_content(self) -> str:
        """The content to use going forward.

        Returns the revised content if the review was revised, otherwise the original content.
        """
        if self._decision is not None and self._decision.revised_content is not None:
            return self._decision.revised_content
        return self.content


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def human_review(
    content: str,
    *,
    agent: str = "",
    context: str = "",
    state: str = "",
    metadata: dict[str, Any] | None = None,
) -> HumanReview:
    """Create a human-in-the-loop review request.

    Parameters
    ----------
    content
        The content to be reviewed (e.g., an agent's response, a draft, a proposed action).
    agent
        Name of the agent that produced the content.
    context
        Additional context for the reviewer.
    state
        Pathway state name where the review was requested.
    metadata
        Arbitrary metadata.

    Returns
    -------
    HumanReview
        A pending review request.

    Examples
    --------
    ```python
    import talk_box as tb

    review = tb.human_review(
        "Refund $500 to customer account",
        agent="billing_agent",
        context="Customer requested refund for order #1234",
    )
    review.status  # ReviewStatus.PENDING
    ```
    """
    return HumanReview(
        content=content,
        agent=agent,
        context=context,
        state=state,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# Decision helpers
# ---------------------------------------------------------------------------


def approve(
    review: HumanReview,
    *,
    feedback: str = "",
    decided_by: str = "",
) -> ReviewDecision:
    """Approve a pending review.

    Parameters
    ----------
    review
        The review to approve.
    feedback
        Optional feedback from the reviewer.
    decided_by
        Identifier for the human making the decision.

    Returns
    -------
    ReviewDecision
        The approval decision.

    Raises
    ------
    ValueError
        If the review has already been resolved.

    Examples
    --------
    ```python
    import talk_box as tb

    review = tb.human_review("Send email to all users", agent="comms_bot")
    decision = tb.approve(review, feedback="Approved for send", decided_by="alice")
    review.is_approved  # True
    review.status       # ReviewStatus.APPROVED
    ```
    """
    if review.is_resolved:
        raise ValueError(f"Review is already resolved with status '{review.status.value}'")

    decision = ReviewDecision(
        status=ReviewStatus.APPROVED,
        feedback=feedback,
        decided_at=time.time(),
        decided_by=decided_by,
    )
    review._decision = decision
    return decision


def reject(
    review: HumanReview,
    *,
    feedback: str = "",
    decided_by: str = "",
) -> ReviewDecision:
    """Reject a pending review.

    Parameters
    ----------
    review
        The review to reject.
    feedback
        Feedback explaining the rejection.
    decided_by
        Identifier for the human making the decision.

    Returns
    -------
    ReviewDecision
        The rejection decision.

    Raises
    ------
    ValueError
        If the review has already been resolved.

    Examples
    --------
    ```python
    import talk_box as tb

    review = tb.human_review("Delete all user data", agent="cleanup_bot")
    decision = tb.reject(review, feedback="Too risky", decided_by="bob")
    review.is_approved  # False
    review.status       # ReviewStatus.REJECTED
    ```
    """
    if review.is_resolved:
        raise ValueError(f"Review is already resolved with status '{review.status.value}'")

    decision = ReviewDecision(
        status=ReviewStatus.REJECTED,
        feedback=feedback,
        decided_at=time.time(),
        decided_by=decided_by,
    )
    review._decision = decision
    return decision


def revise(
    review: HumanReview,
    revised_content: str,
    *,
    feedback: str = "",
    decided_by: str = "",
) -> ReviewDecision:
    """Revise the content and approve the review.

    Use when the human wants to modify the agent's output before proceeding.

    Parameters
    ----------
    review
        The review to revise.
    revised_content
        The corrected or modified content to use instead.
    feedback
        Feedback explaining the revision.
    decided_by
        Identifier for the human making the decision.

    Returns
    -------
    ReviewDecision
        The revision decision.

    Raises
    ------
    ValueError
        If the review has already been resolved.

    Examples
    --------
    ```python
    import talk_box as tb

    review = tb.human_review("Refund $500", agent="billing_bot")
    decision = tb.revise(review, "Refund $250", feedback="Partial refund only")
    review.final_content  # "Refund $250"
    review.is_approved    # True (revised counts as approved)
    ```
    """
    if review.is_resolved:
        raise ValueError(f"Review is already resolved with status '{review.status.value}'")

    decision = ReviewDecision(
        status=ReviewStatus.REVISED,
        feedback=feedback,
        revised_content=revised_content,
        decided_at=time.time(),
        decided_by=decided_by,
    )
    review._decision = decision
    return decision


# ---------------------------------------------------------------------------
# ReviewQueue
# ---------------------------------------------------------------------------


class ReviewQueue:
    """Collects and tracks multiple review requests.

    A `ReviewQueue` manages a list of `HumanReview` instances, providing filtering by status
    and bulk operations. Useful for workflows where multiple agent outputs need human approval.

    Examples
    --------
    ```python
    import talk_box as tb

    queue = tb.ReviewQueue()
    queue.add(tb.human_review("Draft email", agent="writer"))
    queue.add(tb.human_review("SQL query", agent="analyst"))
    queue.pending()  # 2 reviews
    tb.approve(queue.pending()[0])
    queue.pending()  # 1 review
    queue.resolved() # 1 review
    ```
    """

    def __init__(self) -> None:
        self._reviews: list[HumanReview] = []

    def add(self, review: HumanReview) -> HumanReview:
        """Add a review to the queue.

        Parameters
        ----------
        review
            The review to track.

        Returns
        -------
        HumanReview
            The same review, for chaining.
        """
        self._reviews.append(review)
        return review

    def pending(self) -> list[HumanReview]:
        """Return all pending reviews.

        Returns
        -------
        list[HumanReview]
            Reviews that have not yet been decided.
        """
        return [r for r in self._reviews if r.is_pending]

    def resolved(self) -> list[HumanReview]:
        """Return all resolved reviews.

        Returns
        -------
        list[HumanReview]
            Reviews that have been approved, rejected, or revised.
        """
        return [r for r in self._reviews if r.is_resolved]

    def by_agent(self, agent: str) -> list[HumanReview]:
        """Return reviews from a specific agent.

        Parameters
        ----------
        agent
            The agent name to filter by.

        Returns
        -------
        list[HumanReview]
            All reviews from the specified agent.
        """
        return [r for r in self._reviews if r.agent == agent]

    def by_status(self, status: ReviewStatus) -> list[HumanReview]:
        """Return reviews matching a specific status.

        Parameters
        ----------
        status
            The status to filter by.

        Returns
        -------
        list[HumanReview]
            All reviews with the given status.
        """
        return [r for r in self._reviews if r.status == status]

    @property
    def all(self) -> list[HumanReview]:
        """All reviews in the queue."""
        return list(self._reviews)

    def __len__(self) -> int:
        return len(self._reviews)
