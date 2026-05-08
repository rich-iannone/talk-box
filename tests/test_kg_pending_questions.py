"""Tests for KnowledgeGraph.pending_questions / answer_question / dismiss_question."""

from __future__ import annotations

import pytest

from talk_box import Edge, KnowledgeGraph, Node, NodeType
from talk_box.enrichment_qa import QuestionStatus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def kg() -> KnowledgeGraph:
    """In-memory KG with ambiguous entities for Q&A testing."""
    g = KnowledgeGraph(":memory:")
    g.add_node(Node(id="e-alex-1", node_type=NodeType.ENTITY, name="Alex"))
    g.add_node(Node(id="e-alex-2", node_type=NodeType.ENTITY, name="Alex"))
    g.add_node(Node(id="e-sarah", node_type=NodeType.ENTITY, name="Sarah"))
    return g


@pytest.fixture()
def kg_with_weak_edge() -> KnowledgeGraph:
    """KG with a weak relationship edge."""
    g = KnowledgeGraph(":memory:")
    g.add_node(Node(id="e1", node_type=NodeType.ENTITY, name="Alice"))
    g.add_node(Node(id="e2", node_type=NodeType.ENTITY, name="Bob"))
    g.add_edge(Edge(source="e1", target="e2", relation="knows", weight=0.1))
    return g


# ---------------------------------------------------------------------------
# TestPendingQuestions
# ---------------------------------------------------------------------------


class TestPendingQuestions:
    """Tests for kg.pending_questions() convenience method."""

    def test_returns_questions(self, kg: KnowledgeGraph):
        questions = kg.pending_questions()
        assert len(questions) >= 1
        assert all(q.is_pending for q in questions)

    def test_sorted_by_impact(self, kg: KnowledgeGraph):
        questions = kg.pending_questions(sort_by="confusion_impact")
        if len(questions) > 1:
            assert questions[0].confusion_impact >= questions[1].confusion_impact

    def test_sorted_by_created(self, kg: KnowledgeGraph):
        questions = kg.pending_questions(sort_by="created_at")
        if len(questions) > 1:
            assert questions[0].created_at <= questions[1].created_at

    def test_limit(self, kg: KnowledgeGraph):
        questions = kg.pending_questions(limit=1)
        assert len(questions) <= 1

    def test_no_refresh(self):
        g = KnowledgeGraph(":memory:")
        g.add_node(Node(id="e1", node_type=NodeType.ENTITY, name="Alex"))
        g.add_node(Node(id="e2", node_type=NodeType.ENTITY, name="Alex"))
        # Without refresh, queue starts empty
        questions = g.pending_questions(refresh=False)
        assert len(questions) == 0

    def test_refresh_populates(self):
        g = KnowledgeGraph(":memory:")
        g.add_node(Node(id="e1", node_type=NodeType.ENTITY, name="Alex"))
        g.add_node(Node(id="e2", node_type=NodeType.ENTITY, name="Alex"))
        questions = g.pending_questions(refresh=True)
        assert len(questions) >= 1

    def test_empty_graph(self):
        g = KnowledgeGraph(":memory:")
        questions = g.pending_questions()
        assert len(questions) == 0

    def test_queue_persists_across_calls(self, kg: KnowledgeGraph):
        q1 = kg.pending_questions()
        q2 = kg.pending_questions(refresh=False)
        # Same queue, so same questions
        assert {q.id for q in q1} == {q.id for q in q2}

    def test_weak_edge_detection(self, kg_with_weak_edge: KnowledgeGraph):
        questions = kg_with_weak_edge.pending_questions()
        types = {q.question_type.value for q in questions}
        assert "relationship_uncertainty" in types


# ---------------------------------------------------------------------------
# TestAnswerQuestion
# ---------------------------------------------------------------------------


class TestAnswerQuestion:
    """Tests for kg.answer_question() convenience method."""

    def test_answer_with_choice(self, kg: KnowledgeGraph):
        questions = kg.pending_questions()
        q = questions[0]
        result = kg.answer_question(q.id, choice=0)
        assert result is not None
        assert result.status == QuestionStatus.ANSWERED
        assert result.answer_choice == 0

    def test_answer_with_freeform(self, kg: KnowledgeGraph):
        questions = kg.pending_questions()
        q = questions[0]
        result = kg.answer_question(q.id, freeform="It's Alex Torres")
        assert result is not None
        assert result.status == QuestionStatus.ANSWERED
        assert result.answer_freeform == "It's Alex Torres"

    def test_answer_with_both(self, kg: KnowledgeGraph):
        questions = kg.pending_questions()
        q = questions[0]
        result = kg.answer_question(q.id, choice=0, freeform="Confirmed")
        assert result is not None
        assert result.answer_choice == 0
        assert result.answer_freeform == "Confirmed"

    def test_answer_removes_from_pending(self, kg: KnowledgeGraph):
        questions = kg.pending_questions()
        q_id = questions[0].id
        kg.answer_question(q_id, choice=0)
        remaining = kg.pending_questions(refresh=False)
        assert all(q.id != q_id for q in remaining)

    def test_answer_not_found(self, kg: KnowledgeGraph):
        result = kg.answer_question("nonexistent", choice=0)
        assert result is None

    def test_answer_no_args_raises(self, kg: KnowledgeGraph):
        questions = kg.pending_questions()
        with pytest.raises(ValueError, match="Must provide"):
            kg.answer_question(questions[0].id)

    def test_answer_choice_out_of_range(self, kg: KnowledgeGraph):
        questions = kg.pending_questions()
        with pytest.raises(ValueError, match="out of range"):
            kg.answer_question(questions[0].id, choice=99)


# ---------------------------------------------------------------------------
# TestDismissQuestion
# ---------------------------------------------------------------------------


class TestDismissQuestion:
    """Tests for kg.dismiss_question() convenience method."""

    def test_dismiss(self, kg: KnowledgeGraph):
        questions = kg.pending_questions()
        q_id = questions[0].id
        result = kg.dismiss_question(q_id)
        assert result is not None
        assert result.status == QuestionStatus.DISMISSED

    def test_dismiss_removes_from_pending(self, kg: KnowledgeGraph):
        questions = kg.pending_questions()
        q_id = questions[0].id
        kg.dismiss_question(q_id)
        remaining = kg.pending_questions(refresh=False)
        assert all(q.id != q_id for q in remaining)

    def test_dismiss_not_found(self, kg: KnowledgeGraph):
        result = kg.dismiss_question("nonexistent")
        assert result is None


# ---------------------------------------------------------------------------
# TestQuestionStats
# ---------------------------------------------------------------------------


class TestQuestionStats:
    """Tests for kg.question_stats() convenience method."""

    def test_empty_graph(self):
        g = KnowledgeGraph(":memory:")
        stats = g.question_stats()
        assert stats["total"] == 0
        assert stats["pending"] == 0

    def test_after_refresh(self, kg: KnowledgeGraph):
        kg.pending_questions()  # triggers detection
        stats = kg.question_stats()
        assert stats["total"] >= 1
        assert stats["pending"] >= 1

    def test_after_answer(self, kg: KnowledgeGraph):
        questions = kg.pending_questions()
        kg.answer_question(questions[0].id, choice=0)
        stats = kg.question_stats()
        assert stats["answered"] >= 1

    def test_after_dismiss(self, kg: KnowledgeGraph):
        questions = kg.pending_questions()
        kg.dismiss_question(questions[0].id)
        stats = kg.question_stats()
        assert stats["dismissed"] >= 1

    def test_returns_dict(self, kg: KnowledgeGraph):
        stats = kg.question_stats()
        assert isinstance(stats, dict)
        assert set(stats.keys()) == {"total", "pending", "answered", "dismissed", "expired"}
