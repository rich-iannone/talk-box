"""Tests for talk_box.enrichment_qa module."""

from __future__ import annotations

import time

import pytest

from talk_box import KnowledgeGraph, Node, NodeType
from talk_box.enrichment_qa import (
    EnrichmentQuestion,
    QuestionOption,
    QuestionQueue,
    QuestionStatus,
    QuestionType,
    QueueStats,
    detect_factual_conflicts,
    detect_name_ambiguity,
    detect_weak_relationships,
    generate_question_id,
    pending_questions,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def empty_kg() -> KnowledgeGraph:
    """Empty in-memory knowledge graph."""
    return KnowledgeGraph(":memory:")


@pytest.fixture()
def ambiguous_kg(empty_kg: KnowledgeGraph) -> KnowledgeGraph:
    """KG with duplicate-named entities."""
    empty_kg.add_node(Node(id="e-alex-1", node_type=NodeType.ENTITY, name="Alex"))
    empty_kg.add_node(Node(id="e-alex-2", node_type=NodeType.ENTITY, name="Alex"))
    empty_kg.add_node(Node(id="e-sarah", node_type=NodeType.ENTITY, name="Sarah"))
    return empty_kg


@pytest.fixture()
def sample_question() -> EnrichmentQuestion:
    """A basic enrichment question."""
    return EnrichmentQuestion(
        id="q-test-001",
        question_type=QuestionType.ENTITY_AMBIGUITY,
        text='Who is "Alex"?',
        options=[
            QuestionOption(label="Alex Torres", node_ids=["e-alex-1"]),
            QuestionOption(label="Alex Kim", node_ids=["e-alex-2"]),
        ],
        node_ids=["e-alex-1", "e-alex-2"],
        confusion_impact=0.65,
    )


@pytest.fixture()
def queue_with_questions() -> QuestionQueue:
    """Queue pre-loaded with varied questions."""
    queue = QuestionQueue(max_per_session=5)
    for i in range(8):
        q = EnrichmentQuestion(
            id=f"q-{i:03d}",
            question_type=QuestionType.ENTITY_AMBIGUITY,
            text=f"Question {i}?",
            options=[QuestionOption(label="Yes"), QuestionOption(label="No")],
            node_ids=[f"e-{i}"],
            confusion_impact=i * 0.1,
        )
        queue.add(q)
    return queue


# ---------------------------------------------------------------------------
# TestQuestionType
# ---------------------------------------------------------------------------


class TestQuestionType:
    """Tests for QuestionType enum."""

    def test_values(self):
        assert QuestionType.ENTITY_AMBIGUITY.value == "entity_ambiguity"
        assert QuestionType.FACTUAL_CONFLICT.value == "factual_conflict"
        assert QuestionType.RELATIONSHIP_UNCERTAINTY.value == "relationship_uncertainty"
        assert QuestionType.TEMPORAL_CONFUSION.value == "temporal_confusion"


# ---------------------------------------------------------------------------
# TestQuestionStatus
# ---------------------------------------------------------------------------


class TestQuestionStatus:
    """Tests for QuestionStatus enum."""

    def test_values(self):
        assert QuestionStatus.PENDING.value == "pending"
        assert QuestionStatus.ANSWERED.value == "answered"
        assert QuestionStatus.DISMISSED.value == "dismissed"
        assert QuestionStatus.EXPIRED.value == "expired"


# ---------------------------------------------------------------------------
# TestQuestionOption
# ---------------------------------------------------------------------------


class TestQuestionOption:
    """Tests for QuestionOption frozen dataclass."""

    def test_creation(self):
        opt = QuestionOption(label="Option A", node_ids=["n1", "n2"])
        assert opt.label == "Option A"
        assert opt.node_ids == ["n1", "n2"]
        assert opt.metadata == {}

    def test_defaults(self):
        opt = QuestionOption(label="Simple")
        assert opt.node_ids == []
        assert opt.metadata == {}

    def test_with_metadata(self):
        opt = QuestionOption(label="With meta", metadata={"source": "doc-1"})
        assert opt.metadata["source"] == "doc-1"


# ---------------------------------------------------------------------------
# TestEnrichmentQuestion
# ---------------------------------------------------------------------------


class TestEnrichmentQuestion:
    """Tests for EnrichmentQuestion dataclass."""

    def test_creation(self, sample_question: EnrichmentQuestion):
        q = sample_question
        assert q.id == "q-test-001"
        assert q.question_type == QuestionType.ENTITY_AMBIGUITY
        assert q.text == 'Who is "Alex"?'
        assert len(q.options) == 2
        assert q.confusion_impact == 0.65
        assert q.status == QuestionStatus.PENDING

    def test_is_pending(self, sample_question: EnrichmentQuestion):
        assert sample_question.is_pending is True

    def test_is_expired_false(self, sample_question: EnrichmentQuestion):
        assert sample_question.is_expired is False

    def test_is_expired_true(self):
        q = EnrichmentQuestion(
            id="q-old",
            question_type=QuestionType.FACTUAL_CONFLICT,
            text="Old question?",
            created_at=time.time() - 700_000,  # > 7 days
            ttl_seconds=604_800.0,
        )
        assert q.is_expired is True

    def test_to_dict(self, sample_question: EnrichmentQuestion):
        d = sample_question.to_dict()
        assert d["id"] == "q-test-001"
        assert d["question_type"] == "entity_ambiguity"
        assert d["status"] == "pending"
        assert len(d["options"]) == 2
        assert d["options"][0]["label"] == "Alex Torres"
        assert d["confusion_impact"] == 0.65

    def test_default_ttl(self, sample_question: EnrichmentQuestion):
        assert sample_question.ttl_seconds == 604_800.0

    def test_custom_ttl(self):
        q = EnrichmentQuestion(
            id="q-short",
            question_type=QuestionType.TEMPORAL_CONFUSION,
            text="When?",
            ttl_seconds=3600.0,
        )
        assert q.ttl_seconds == 3600.0


# ---------------------------------------------------------------------------
# TestQuestionQueue
# ---------------------------------------------------------------------------


class TestQuestionQueue:
    """Tests for QuestionQueue management."""

    def test_creation(self):
        queue = QuestionQueue(max_per_session=5)
        assert queue.max_per_session == 5

    def test_default_max(self):
        queue = QuestionQueue()
        assert queue.max_per_session == 7

    def test_add(self, sample_question: EnrichmentQuestion):
        queue = QuestionQueue()
        queue.add(sample_question)
        assert queue.stats().total == 1

    def test_add_duplicate_ignored(self, sample_question: EnrichmentQuestion):
        queue = QuestionQueue()
        queue.add(sample_question)
        queue.add(sample_question)
        assert queue.stats().total == 1

    def test_add_many(self):
        queue = QuestionQueue()
        questions = [
            EnrichmentQuestion(
                id=f"q-{i}", question_type=QuestionType.ENTITY_AMBIGUITY, text=f"Q{i}?"
            )
            for i in range(5)
        ]
        added = queue.add_many(questions)
        assert added == 5
        assert queue.stats().total == 5

    def test_add_many_with_duplicates(self):
        queue = QuestionQueue()
        q1 = EnrichmentQuestion(id="q-1", question_type=QuestionType.ENTITY_AMBIGUITY, text="Q1?")
        queue.add(q1)
        questions = [
            q1,
            EnrichmentQuestion(id="q-2", question_type=QuestionType.ENTITY_AMBIGUITY, text="Q2?"),
        ]
        added = queue.add_many(questions)
        assert added == 1  # Only q-2 is new
        assert queue.stats().total == 2

    def test_pending_questions_sorted_by_impact(self, queue_with_questions: QuestionQueue):
        pending = queue_with_questions.pending_questions()
        # Sorted descending by confusion_impact, limited to max_per_session=5
        assert len(pending) == 5
        assert pending[0].confusion_impact >= pending[1].confusion_impact

    def test_pending_questions_sorted_by_created(self, queue_with_questions: QuestionQueue):
        pending = queue_with_questions.pending_questions(sort_by="created_at")
        assert len(pending) == 5
        assert pending[0].created_at <= pending[1].created_at

    def test_pending_questions_custom_limit(self, queue_with_questions: QuestionQueue):
        pending = queue_with_questions.pending_questions(limit=3)
        assert len(pending) == 3

    def test_answer_with_choice(self, sample_question: EnrichmentQuestion):
        queue = QuestionQueue()
        queue.add(sample_question)
        result = queue.answer("q-test-001", choice=0)
        assert result is not None
        assert result.status == QuestionStatus.ANSWERED
        assert result.answer_choice == 0
        assert result.answered_at is not None

    def test_answer_with_freeform(self, sample_question: EnrichmentQuestion):
        queue = QuestionQueue()
        queue.add(sample_question)
        result = queue.answer("q-test-001", freeform="It's Alex Torres from Eng")
        assert result is not None
        assert result.status == QuestionStatus.ANSWERED
        assert result.answer_freeform == "It's Alex Torres from Eng"

    def test_answer_with_both(self, sample_question: EnrichmentQuestion):
        queue = QuestionQueue()
        queue.add(sample_question)
        result = queue.answer("q-test-001", choice=1, freeform="Definitely Alex Kim")
        assert result is not None
        assert result.answer_choice == 1
        assert result.answer_freeform == "Definitely Alex Kim"

    def test_answer_no_args_raises(self, sample_question: EnrichmentQuestion):
        queue = QuestionQueue()
        queue.add(sample_question)
        with pytest.raises(ValueError, match="Must provide at least one"):
            queue.answer("q-test-001")

    def test_answer_choice_out_of_range(self, sample_question: EnrichmentQuestion):
        queue = QuestionQueue()
        queue.add(sample_question)
        with pytest.raises(ValueError, match="out of range"):
            queue.answer("q-test-001", choice=5)

    def test_answer_not_found(self):
        queue = QuestionQueue()
        result = queue.answer("nonexistent", choice=0)
        assert result is None

    def test_answer_already_answered(self, sample_question: EnrichmentQuestion):
        queue = QuestionQueue()
        queue.add(sample_question)
        queue.answer("q-test-001", choice=0)
        # Second answer attempt returns None (no longer pending)
        result = queue.answer("q-test-001", choice=1)
        assert result is None

    def test_dismiss(self, sample_question: EnrichmentQuestion):
        queue = QuestionQueue()
        queue.add(sample_question)
        result = queue.dismiss("q-test-001")
        assert result is not None
        assert result.status == QuestionStatus.DISMISSED

    def test_dismiss_not_found(self):
        queue = QuestionQueue()
        result = queue.dismiss("nonexistent")
        assert result is None

    def test_dismiss_all(self, queue_with_questions: QuestionQueue):
        count = queue_with_questions.dismiss_all()
        assert count == 8
        assert queue_with_questions.stats().pending == 0

    def test_remove_for_nodes(self, queue_with_questions: QuestionQueue):
        expired = queue_with_questions.remove_for_nodes(["e-0", "e-1"])
        assert expired == 2
        assert queue_with_questions.stats().expired == 2

    def test_get(self, sample_question: EnrichmentQuestion):
        queue = QuestionQueue()
        queue.add(sample_question)
        result = queue.get("q-test-001")
        assert result is not None
        assert result.id == "q-test-001"

    def test_get_not_found(self):
        queue = QuestionQueue()
        assert queue.get("nonexistent") is None

    def test_stats(self, queue_with_questions: QuestionQueue):
        # Answer one, dismiss one
        queue_with_questions.answer("q-000", choice=0)
        queue_with_questions.dismiss("q-001")
        stats = queue_with_questions.stats()
        assert stats.total == 8
        assert stats.answered == 1
        assert stats.dismissed == 1
        assert stats.pending == 6

    def test_stats_to_dict(self):
        stats = QueueStats(total=10, pending=5, answered=3, dismissed=1, expired=1)
        d = stats.to_dict()
        assert d == {"total": 10, "pending": 5, "answered": 3, "dismissed": 1, "expired": 1}

    def test_clear(self, queue_with_questions: QuestionQueue):
        queue_with_questions.clear()
        assert queue_with_questions.stats().total == 0

    def test_ttl_expiration(self):
        queue = QuestionQueue()
        old_q = EnrichmentQuestion(
            id="q-expired",
            question_type=QuestionType.ENTITY_AMBIGUITY,
            text="Old?",
            options=[QuestionOption(label="Yes")],
            created_at=time.time() - 700_000,  # > 7 days ago
            ttl_seconds=604_800.0,
        )
        queue.add(old_q)
        pending = queue.pending_questions()
        assert len(pending) == 0
        assert queue.stats().expired == 1


# ---------------------------------------------------------------------------
# TestDetectNameAmbiguity
# ---------------------------------------------------------------------------


class TestDetectNameAmbiguity:
    """Tests for detect_name_ambiguity detector."""

    def test_finds_duplicates(self, ambiguous_kg: KnowledgeGraph):
        questions = detect_name_ambiguity(ambiguous_kg)
        assert len(questions) == 1
        assert "Alex" in questions[0].text
        assert questions[0].question_type == QuestionType.ENTITY_AMBIGUITY

    def test_no_duplicates(self, empty_kg: KnowledgeGraph):
        empty_kg.add_node(Node(id="e1", node_type=NodeType.ENTITY, name="Alice"))
        empty_kg.add_node(Node(id="e2", node_type=NodeType.ENTITY, name="Bob"))
        questions = detect_name_ambiguity(empty_kg)
        assert len(questions) == 0

    def test_empty_graph(self, empty_kg: KnowledgeGraph):
        questions = detect_name_ambiguity(empty_kg)
        assert len(questions) == 0

    def test_case_insensitive(self, empty_kg: KnowledgeGraph):
        empty_kg.add_node(Node(id="e1", node_type=NodeType.ENTITY, name="alex"))
        empty_kg.add_node(Node(id="e2", node_type=NodeType.ENTITY, name="Alex"))
        questions = detect_name_ambiguity(empty_kg)
        assert len(questions) == 1

    def test_options_include_same_and_different(self, ambiguous_kg: KnowledgeGraph):
        questions = detect_name_ambiguity(ambiguous_kg)
        q = questions[0]
        labels = [o.label for o in q.options]
        assert any("same entity" in lbl.lower() for lbl in labels)
        assert any("different" in lbl.lower() for lbl in labels)

    def test_confusion_impact_scales(self, empty_kg: KnowledgeGraph):
        # 3 duplicates should have higher impact than 2
        for i in range(3):
            empty_kg.add_node(Node(id=f"e-alex-{i}", node_type=NodeType.ENTITY, name="Alex"))
        questions = detect_name_ambiguity(empty_kg)
        assert questions[0].confusion_impact > 0.3


# ---------------------------------------------------------------------------
# TestDetectFactualConflicts
# ---------------------------------------------------------------------------


class TestDetectFactualConflicts:
    """Tests for detect_factual_conflicts detector."""

    def test_finds_conflicts(self, empty_kg: KnowledgeGraph):
        empty_kg.add_node(
            Node(
                id="e-launch-1",
                node_type=NodeType.ENTITY,
                name="Launch",
                metadata={"date": "March 15"},
            )
        )
        empty_kg.add_node(
            Node(
                id="e-launch-2",
                node_type=NodeType.ENTITY,
                name="Launch",
                metadata={"date": "April 2"},
            )
        )
        questions = detect_factual_conflicts(empty_kg)
        assert len(questions) == 1
        assert "date" in questions[0].text.lower()

    def test_no_conflicts(self, empty_kg: KnowledgeGraph):
        empty_kg.add_node(Node(id="e1", node_type=NodeType.ENTITY, name="A", metadata={"x": "1"}))
        empty_kg.add_node(Node(id="e2", node_type=NodeType.ENTITY, name="B", metadata={"x": "2"}))
        # Different entities, no conflict
        questions = detect_factual_conflicts(empty_kg)
        assert len(questions) == 0

    def test_same_values_no_conflict(self, empty_kg: KnowledgeGraph):
        empty_kg.add_node(
            Node(id="e1", node_type=NodeType.ENTITY, name="Thing", metadata={"status": "active"})
        )
        empty_kg.add_node(
            Node(id="e2", node_type=NodeType.ENTITY, name="Thing", metadata={"status": "active"})
        )
        questions = detect_factual_conflicts(empty_kg)
        assert len(questions) == 0

    def test_empty_graph(self, empty_kg: KnowledgeGraph):
        questions = detect_factual_conflicts(empty_kg)
        assert len(questions) == 0


# ---------------------------------------------------------------------------
# TestDetectWeakRelationships
# ---------------------------------------------------------------------------


class TestDetectWeakRelationships:
    """Tests for detect_weak_relationships detector."""

    def test_finds_weak_edges(self, empty_kg: KnowledgeGraph):
        from talk_box import Edge

        empty_kg.add_node(Node(id="e1", node_type=NodeType.ENTITY, name="Alice"))
        empty_kg.add_node(Node(id="e2", node_type=NodeType.ENTITY, name="Bob"))
        empty_kg.add_edge(Edge(source="e1", target="e2", relation="knows", weight=0.1))
        questions = detect_weak_relationships(empty_kg)
        assert len(questions) == 1
        assert "Alice" in questions[0].text
        assert "Bob" in questions[0].text

    def test_strong_edges_ignored(self, empty_kg: KnowledgeGraph):
        from talk_box import Edge

        empty_kg.add_node(Node(id="e1", node_type=NodeType.ENTITY, name="Alice"))
        empty_kg.add_node(Node(id="e2", node_type=NodeType.ENTITY, name="Bob"))
        empty_kg.add_edge(Edge(source="e1", target="e2", relation="knows", weight=0.9))
        questions = detect_weak_relationships(empty_kg)
        assert len(questions) == 0

    def test_empty_graph(self, empty_kg: KnowledgeGraph):
        questions = detect_weak_relationships(empty_kg)
        assert len(questions) == 0


# ---------------------------------------------------------------------------
# TestGenerateQuestionId
# ---------------------------------------------------------------------------


class TestGenerateQuestionId:
    """Tests for generate_question_id helper."""

    def test_format(self):
        qid = generate_question_id()
        assert qid.startswith("eq-")
        assert len(qid) == 11  # "eq-" + 8 hex chars

    def test_unique(self):
        ids = {generate_question_id() for _ in range(100)}
        assert len(ids) == 100


# ---------------------------------------------------------------------------
# TestPendingQuestions
# ---------------------------------------------------------------------------


class TestPendingQuestions:
    """Tests for top-level pending_questions function."""

    def test_basic(self, ambiguous_kg: KnowledgeGraph):
        queue = QuestionQueue()
        questions = pending_questions(ambiguous_kg, queue, refresh=True)
        assert len(questions) >= 1
        assert all(q.is_pending for q in questions)

    def test_no_refresh(self, ambiguous_kg: KnowledgeGraph):
        queue = QuestionQueue()
        # Without refresh, queue is empty
        questions = pending_questions(ambiguous_kg, queue, refresh=False)
        assert len(questions) == 0

    def test_custom_generators(self, empty_kg: KnowledgeGraph):
        def always_one(kg: KnowledgeGraph) -> list[EnrichmentQuestion]:
            return [
                EnrichmentQuestion(
                    id="custom-1",
                    question_type=QuestionType.TEMPORAL_CONFUSION,
                    text="Custom question?",
                    confusion_impact=0.9,
                )
            ]

        queue = QuestionQueue()
        questions = pending_questions(empty_kg, queue, refresh=True, generators=[always_one])
        assert len(questions) == 1
        assert questions[0].text == "Custom question?"

    def test_sort_by(self, ambiguous_kg: KnowledgeGraph):
        queue = QuestionQueue()
        pending_questions(ambiguous_kg, queue, refresh=True)
        by_time = queue.pending_questions(sort_by="created_at")
        by_impact = queue.pending_questions(sort_by="confusion_impact")
        # Both return the same questions, possibly in different order
        assert {q.id for q in by_time} == {q.id for q in by_impact}

    def test_limit(self, ambiguous_kg: KnowledgeGraph):
        queue = QuestionQueue()
        # Add many questions manually
        for i in range(10):
            queue.add(
                EnrichmentQuestion(
                    id=f"q-{i}",
                    question_type=QuestionType.ENTITY_AMBIGUITY,
                    text=f"Q{i}?",
                    confusion_impact=i * 0.1,
                )
            )
        questions = pending_questions(ambiguous_kg, queue, refresh=False, limit=3)
        assert len(questions) == 3


# ---------------------------------------------------------------------------
# TestTopLevelImport
# ---------------------------------------------------------------------------


class TestTopLevelImport:
    """Tests that enrichment_qa exports are accessible from talk_box."""

    def test_import_question_type(self):
        from talk_box import QuestionType as QT

        assert QT.ENTITY_AMBIGUITY.value == "entity_ambiguity"

    def test_import_question_status(self):
        from talk_box import QuestionStatus as QS

        assert QS.PENDING.value == "pending"

    def test_import_enrichment_question(self):
        from talk_box import EnrichmentQuestion as EQ

        q = EQ(id="t", question_type=QuestionType.ENTITY_AMBIGUITY, text="x")
        assert q.id == "t"

    def test_import_question_queue(self):
        from talk_box import QuestionQueue as QQ

        queue = QQ()
        assert queue.max_per_session == 7

    def test_import_pending_questions(self):
        from talk_box import pending_questions as pq

        assert callable(pq)

    def test_import_detectors(self):
        from talk_box import (
            detect_factual_conflicts,
            detect_name_ambiguity,
            detect_weak_relationships,
        )

        assert callable(detect_name_ambiguity)
        assert callable(detect_factual_conflicts)
        assert callable(detect_weak_relationships)
