import time

import pytest

from talk_box.hitl import (
    HumanReview,
    ReviewDecision,
    ReviewQueue,
    ReviewStatus,
    approve,
    human_review,
    reject,
    revise,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_review(**kwargs):
    """Create a review with sensible defaults."""
    defaults = {"content": "Draft output", "agent": "test_agent"}
    defaults.update(kwargs)
    return human_review(**defaults)


# ---------------------------------------------------------------------------
# ReviewStatus
# ---------------------------------------------------------------------------


class TestReviewStatus:
    def test_values(self):
        assert ReviewStatus.PENDING.value == "pending"
        assert ReviewStatus.APPROVED.value == "approved"
        assert ReviewStatus.REJECTED.value == "rejected"
        assert ReviewStatus.REVISED.value == "revised"

    def test_members(self):
        assert len(ReviewStatus) == 4


# ---------------------------------------------------------------------------
# ReviewDecision
# ---------------------------------------------------------------------------


class TestReviewDecision:
    def test_frozen(self):
        decision = ReviewDecision(status=ReviewStatus.APPROVED)
        with pytest.raises(AttributeError):
            decision.status = ReviewStatus.REJECTED  # type: ignore[misc]

    def test_fields(self):
        decision = ReviewDecision(
            status=ReviewStatus.REJECTED,
            feedback="Not good",
            revised_content=None,
            decided_at=1000.0,
            decided_by="alice",
        )
        assert decision.status == ReviewStatus.REJECTED
        assert decision.feedback == "Not good"
        assert decision.revised_content is None
        assert decision.decided_at == 1000.0
        assert decision.decided_by == "alice"

    def test_defaults(self):
        decision = ReviewDecision(status=ReviewStatus.APPROVED)
        assert decision.feedback == ""
        assert decision.revised_content is None
        assert decision.decided_at == 0.0
        assert decision.decided_by == ""


# ---------------------------------------------------------------------------
# HumanReview
# ---------------------------------------------------------------------------


class TestHumanReview:
    def test_creation(self):
        review = HumanReview(content="Check this", agent="bot")
        assert review.content == "Check this"
        assert review.agent == "bot"
        assert review.context == ""
        assert review.state == ""
        assert review.metadata == {}
        assert review.created_at > 0

    def test_status_pending_by_default(self):
        review = _make_review()
        assert review.status == ReviewStatus.PENDING

    def test_is_pending(self):
        review = _make_review()
        assert review.is_pending is True
        assert review.is_resolved is False

    def test_is_approved_false_when_pending(self):
        review = _make_review()
        assert review.is_approved is False

    def test_decision_none_when_pending(self):
        review = _make_review()
        assert review.decision is None

    def test_final_content_returns_original_when_pending(self):
        review = _make_review(content="original")
        assert review.final_content == "original"

    def test_final_content_returns_original_when_approved(self):
        review = _make_review(content="original")
        approve(review)
        assert review.final_content == "original"

    def test_final_content_returns_revised_when_revised(self):
        review = _make_review(content="original")
        revise(review, "modified")
        assert review.final_content == "modified"

    def test_metadata(self):
        review = _make_review(metadata={"priority": "high"})
        assert review.metadata["priority"] == "high"

    def test_state_field(self):
        review = _make_review(state="approval_gate")
        assert review.state == "approval_gate"


# ---------------------------------------------------------------------------
# human_review factory
# ---------------------------------------------------------------------------


class TestHumanReviewFactory:
    def test_creates_pending_review(self):
        review = human_review("content")
        assert review.content == "content"
        assert review.is_pending

    def test_with_all_params(self):
        review = human_review(
            "content",
            agent="bot",
            context="for testing",
            state="review_state",
            metadata={"key": "val"},
        )
        assert review.agent == "bot"
        assert review.context == "for testing"
        assert review.state == "review_state"
        assert review.metadata == {"key": "val"}

    def test_default_metadata(self):
        review = human_review("content")
        assert review.metadata == {}


# ---------------------------------------------------------------------------
# approve
# ---------------------------------------------------------------------------


class TestApprove:
    def test_approve_review(self):
        review = _make_review()
        decision = approve(review)
        assert decision.status == ReviewStatus.APPROVED
        assert review.status == ReviewStatus.APPROVED
        assert review.is_resolved is True
        assert review.is_approved is True

    def test_approve_with_feedback(self):
        review = _make_review()
        decision = approve(review, feedback="LGTM", decided_by="alice")
        assert decision.feedback == "LGTM"
        assert decision.decided_by == "alice"

    def test_approve_sets_timestamp(self):
        before = time.time()
        review = _make_review()
        decision = approve(review)
        after = time.time()
        assert before <= decision.decided_at <= after

    def test_approve_already_resolved_raises(self):
        review = _make_review()
        approve(review)
        with pytest.raises(ValueError, match="already resolved"):
            approve(review)


# ---------------------------------------------------------------------------
# reject
# ---------------------------------------------------------------------------


class TestReject:
    def test_reject_review(self):
        review = _make_review()
        decision = reject(review)
        assert decision.status == ReviewStatus.REJECTED
        assert review.status == ReviewStatus.REJECTED
        assert review.is_resolved is True
        assert review.is_approved is False

    def test_reject_with_feedback(self):
        review = _make_review()
        decision = reject(review, feedback="Too risky", decided_by="bob")
        assert decision.feedback == "Too risky"
        assert decision.decided_by == "bob"

    def test_reject_sets_timestamp(self):
        before = time.time()
        review = _make_review()
        decision = reject(review)
        after = time.time()
        assert before <= decision.decided_at <= after

    def test_reject_already_resolved_raises(self):
        review = _make_review()
        reject(review)
        with pytest.raises(ValueError, match="already resolved"):
            reject(review)


# ---------------------------------------------------------------------------
# revise
# ---------------------------------------------------------------------------


class TestRevise:
    def test_revise_review(self):
        review = _make_review(content="original")
        decision = revise(review, "modified")
        assert decision.status == ReviewStatus.REVISED
        assert review.status == ReviewStatus.REVISED
        assert review.is_resolved is True
        assert review.is_approved is True  # revised counts as approved

    def test_revise_sets_revised_content(self):
        review = _make_review(content="original")
        decision = revise(review, "new content")
        assert decision.revised_content == "new content"
        assert review.final_content == "new content"

    def test_revise_with_feedback(self):
        review = _make_review()
        decision = revise(review, "fixed", feedback="Adjusted amount", decided_by="carol")
        assert decision.feedback == "Adjusted amount"
        assert decision.decided_by == "carol"

    def test_revise_sets_timestamp(self):
        before = time.time()
        review = _make_review()
        decision = revise(review, "fixed")
        after = time.time()
        assert before <= decision.decided_at <= after

    def test_revise_already_resolved_raises(self):
        review = _make_review()
        revise(review, "fixed")
        with pytest.raises(ValueError, match="already resolved"):
            revise(review, "fixed again")

    def test_cannot_approve_after_reject(self):
        review = _make_review()
        reject(review)
        with pytest.raises(ValueError, match="already resolved"):
            approve(review)

    def test_cannot_reject_after_approve(self):
        review = _make_review()
        approve(review)
        with pytest.raises(ValueError, match="already resolved"):
            reject(review)


# ---------------------------------------------------------------------------
# ReviewQueue
# ---------------------------------------------------------------------------


class TestReviewQueue:
    def test_empty_queue(self):
        queue = ReviewQueue()
        assert len(queue) == 0
        assert queue.pending() == []
        assert queue.resolved() == []

    def test_add_review(self):
        queue = ReviewQueue()
        review = _make_review()
        result = queue.add(review)
        assert result is review
        assert len(queue) == 1

    def test_pending(self):
        queue = ReviewQueue()
        r1 = queue.add(_make_review())
        r2 = queue.add(_make_review())
        assert queue.pending() == [r1, r2]

    def test_resolved(self):
        queue = ReviewQueue()
        r1 = queue.add(_make_review())
        r2 = queue.add(_make_review())
        approve(r1)
        assert queue.resolved() == [r1]
        assert queue.pending() == [r2]

    def test_by_agent(self):
        queue = ReviewQueue()
        queue.add(_make_review(agent="a"))
        r2 = queue.add(_make_review(agent="b"))
        queue.add(_make_review(agent="a"))
        result = queue.by_agent("b")
        assert result == [r2]

    def test_by_status(self):
        queue = ReviewQueue()
        r1 = queue.add(_make_review())
        r2 = queue.add(_make_review())
        r3 = queue.add(_make_review())
        approve(r1)
        reject(r2)
        assert queue.by_status(ReviewStatus.APPROVED) == [r1]
        assert queue.by_status(ReviewStatus.REJECTED) == [r2]
        assert queue.by_status(ReviewStatus.PENDING) == [r3]

    def test_all_returns_copy(self):
        queue = ReviewQueue()
        queue.add(_make_review())
        result = queue.all
        result.append(_make_review())
        assert len(queue) == 1

    def test_mixed_workflow(self):
        queue = ReviewQueue()
        r1 = queue.add(_make_review(agent="writer", content="Draft email"))
        r2 = queue.add(_make_review(agent="analyst", content="SQL query"))
        r3 = queue.add(_make_review(agent="writer", content="Follow-up email"))

        approve(r1, feedback="Good to send")
        revise(r2, "SELECT * FROM users LIMIT 10", feedback="Add limit")
        # r3 still pending

        assert len(queue.pending()) == 1
        assert len(queue.resolved()) == 2
        assert queue.pending()[0] is r3
        assert r2.final_content == "SELECT * FROM users LIMIT 10"


# ---------------------------------------------------------------------------
# Top-level imports
# ---------------------------------------------------------------------------


class TestTopLevelImport:
    def test_import_from_package(self):
        import talk_box

        for name in [
            "HumanReview",
            "ReviewDecision",
            "ReviewStatus",
            "ReviewQueue",
            "human_review",
            "approve",
            "reject",
            "revise",
        ]:
            assert hasattr(talk_box, name)

    def test_all_contains_exports(self):
        import talk_box

        for name in [
            "HumanReview",
            "ReviewDecision",
            "ReviewStatus",
            "ReviewQueue",
            "human_review",
            "approve",
            "reject",
            "revise",
        ]:
            assert name in talk_box.__all__
