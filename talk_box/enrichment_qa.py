"""Enrichment Q&A: structured questions to resolve knowledge graph ambiguity."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from talk_box.knowledge_graph import KnowledgeGraph, NodeType

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class QuestionType(Enum):
    """Category of enrichment question.

    Values
    ------
    ENTITY_AMBIGUITY
        Multiple entities share a name or alias.
    FACTUAL_CONFLICT
        Contradictory facts across documents.
    RELATIONSHIP_UNCERTAINTY
        Unclear or ambiguous relationship between entities.
    TEMPORAL_CONFUSION
        Ambiguous timing or ordering of events.

    Examples
    --------
    ```python
    import talk_box as tb

    tb.QuestionType.ENTITY_AMBIGUITY
    tb.QuestionType.FACTUAL_CONFLICT
    ```
    """

    ENTITY_AMBIGUITY = "entity_ambiguity"
    FACTUAL_CONFLICT = "factual_conflict"
    RELATIONSHIP_UNCERTAINTY = "relationship_uncertainty"
    TEMPORAL_CONFUSION = "temporal_confusion"


class QuestionStatus(Enum):
    """Lifecycle state of an enrichment question.

    Values
    ------
    PENDING
        Awaiting user answer.
    ANSWERED
        User has provided an answer.
    DISMISSED
        User dismissed without answering.
    EXPIRED
        TTL elapsed or underlying data resolved the ambiguity.

    Examples
    --------
    ```python
    import talk_box as tb

    tb.QuestionStatus.PENDING
    tb.QuestionStatus.ANSWERED
    ```
    """

    PENDING = "pending"
    ANSWERED = "answered"
    DISMISSED = "dismissed"
    EXPIRED = "expired"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuestionOption:
    """A single answer option for an enrichment question.

    Parameters
    ----------
    label
        Human-readable option text.
    node_ids
        Node IDs that this option references (for graph updates).
    metadata
        Extra context about this option.

    Examples
    --------
    ```python
    import talk_box as tb

    option = tb.QuestionOption(
        label="Alex Torres (Engineering)",
        node_ids=["entity-alex-torres"],
    )
    option.label  # "Alex Torres (Engineering)"
    ```
    """

    label: str
    node_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnrichmentQuestion:
    """A structured question to resolve knowledge graph ambiguity.

    Parameters
    ----------
    id
        Unique question identifier.
    question_type
        Category of ambiguity this question resolves.
    text
        The question text displayed to the user.
    options
        Multiple-choice answer options.
    node_ids
        IDs of nodes involved in the ambiguity.
    confusion_impact
        How much resolving this question would reduce confusion (0.0–1.0).
    context
        Supporting context (e.g. document excerpts showing the conflict).
    status
        Current lifecycle state.
    created_at
        Unix timestamp of question creation.
    ttl_seconds
        Time-to-live in seconds; question expires after this duration.
    answer_choice
        Index of the chosen option (set after answering).
    answer_freeform
        Freeform text answer (set after answering).
    answered_at
        Unix timestamp when the question was answered.

    Examples
    --------
    ```python
    import talk_box as tb

    q = tb.EnrichmentQuestion(
        id="q-001",
        question_type=tb.QuestionType.ENTITY_AMBIGUITY,
        text='Who is "Alex" in your Project Alpha notes?',
        options=[
            tb.QuestionOption(label="Alex Torres (Engineering)"),
            tb.QuestionOption(label="Alex Kim (Design)"),
        ],
        node_ids=["entity-alex"],
        confusion_impact=0.65,
    )
    q.text  # 'Who is "Alex" in your Project Alpha notes?'
    ```
    """

    id: str
    question_type: QuestionType
    text: str
    options: list[QuestionOption] = field(default_factory=list)
    node_ids: list[str] = field(default_factory=list)
    confusion_impact: float = 0.0
    context: str = ""
    status: QuestionStatus = QuestionStatus.PENDING
    created_at: float = field(default_factory=time.time)
    ttl_seconds: float = 604_800.0  # 7 days
    answer_choice: int | None = None
    answer_freeform: str | None = None
    answered_at: float | None = None

    @property
    def is_pending(self) -> bool:
        """Whether the question is still awaiting an answer.

        Examples
        --------
        ```python
        q.is_pending  # True
        ```
        """
        return self.status == QuestionStatus.PENDING

    @property
    def is_expired(self) -> bool:
        """Whether the question has exceeded its TTL.

        Examples
        --------
        ```python
        q.is_expired  # False
        ```
        """
        return (time.time() - self.created_at) > self.ttl_seconds

    def to_dict(self) -> dict[str, Any]:
        """Serialize the question to a dictionary.

        Examples
        --------
        ```python
        d = q.to_dict()
        d["text"]  # 'Who is "Alex" in your ...'
        ```
        """
        return {
            "id": self.id,
            "question_type": self.question_type.value,
            "text": self.text,
            "options": [
                {"label": o.label, "node_ids": o.node_ids, "metadata": o.metadata}
                for o in self.options
            ],
            "node_ids": self.node_ids,
            "confusion_impact": self.confusion_impact,
            "context": self.context,
            "status": self.status.value,
            "created_at": self.created_at,
            "ttl_seconds": self.ttl_seconds,
            "answer_choice": self.answer_choice,
            "answer_freeform": self.answer_freeform,
            "answered_at": self.answered_at,
        }


# ---------------------------------------------------------------------------
# Question queue
# ---------------------------------------------------------------------------

# Type alias for question generator functions
QuestionGeneratorFn = Callable[[KnowledgeGraph], list[EnrichmentQuestion]]
"""A function that examines a KG and produces enrichment questions."""


@dataclass
class QueueStats:
    """Summary statistics for the question queue.

    Parameters
    ----------
    total
        Total questions ever created.
    pending
        Questions awaiting answers.
    answered
        Questions that have been answered.
    dismissed
        Questions dismissed by the user.
    expired
        Questions that expired due to TTL.

    Examples
    --------
    ```python
    import talk_box as tb

    queue = tb.QuestionQueue()
    stats = queue.stats()
    stats.pending  # 3
    ```
    """

    total: int = 0
    pending: int = 0
    answered: int = 0
    dismissed: int = 0
    expired: int = 0

    def to_dict(self) -> dict[str, int]:
        """Serialize stats to a dictionary.

        Examples
        --------
        ```python
        stats.to_dict()  # {"total": 5, "pending": 3, ...}
        ```
        """
        return {
            "total": self.total,
            "pending": self.pending,
            "answered": self.answered,
            "dismissed": self.dismissed,
            "expired": self.expired,
        }


class QuestionQueue:
    """A managed queue of enrichment questions for knowledge graph improvement.

    Holds questions sorted by confusion impact, enforces TTL expiration,
    and limits session batches to a configurable maximum.

    Parameters
    ----------
    max_per_session
        Maximum questions to present per session (default 7).

    Examples
    --------
    ```python
    import talk_box as tb

    queue = tb.QuestionQueue(max_per_session=5)
    queue.add(question)
    pending = queue.pending_questions()
    ```
    """

    def __init__(self, max_per_session: int = 7) -> None:
        self._questions: list[EnrichmentQuestion] = []
        self._max_per_session = max_per_session

    @property
    def max_per_session(self) -> int:
        """Maximum questions per session.

        Examples
        --------
        ```python
        queue.max_per_session  # 7
        ```
        """
        return self._max_per_session

    def add(self, question: EnrichmentQuestion) -> None:
        """Add a question to the queue.

        Duplicate IDs are ignored.

        Parameters
        ----------
        question
            The enrichment question to enqueue.

        Examples
        --------
        ```python
        queue.add(question)
        ```
        """
        if any(q.id == question.id for q in self._questions):
            return
        self._questions.append(question)

    def add_many(self, questions: list[EnrichmentQuestion]) -> int:
        """Add multiple questions to the queue.

        Parameters
        ----------
        questions
            Questions to enqueue.

        Returns
        -------
        int
            Number of questions actually added (excludes duplicates).

        Examples
        --------
        ```python
        added = queue.add_many(new_questions)
        added  # 3
        ```
        """
        before = len(self._questions)
        for q in questions:
            self.add(q)
        return len(self._questions) - before

    def pending_questions(
        self,
        *,
        sort_by: str = "confusion_impact",
        limit: int | None = None,
    ) -> list[EnrichmentQuestion]:
        """Get pending questions sorted by priority.

        Automatically expires questions past their TTL before returning.

        Parameters
        ----------
        sort_by
            Sort field: ``"confusion_impact"`` (default, descending) or
            ``"created_at"`` (ascending, oldest first).
        limit
            Maximum questions to return. Defaults to ``max_per_session``.

        Returns
        -------
        list[EnrichmentQuestion]
            Pending questions, sorted and limited.

        Examples
        --------
        ```python
        questions = queue.pending_questions()
        questions[0].confusion_impact  # highest impact first
        ```

        Limit to 3 questions:

        ```python
        top3 = queue.pending_questions(limit=3)
        ```
        """
        self._expire_stale()
        effective_limit = limit if limit is not None else self._max_per_session

        pending = [q for q in self._questions if q.status == QuestionStatus.PENDING]

        if sort_by == "confusion_impact":
            pending.sort(key=lambda q: q.confusion_impact, reverse=True)
        elif sort_by == "created_at":
            pending.sort(key=lambda q: q.created_at)

        return pending[:effective_limit]

    def answer(
        self,
        question_id: str,
        *,
        choice: int | None = None,
        freeform: str | None = None,
    ) -> EnrichmentQuestion | None:
        """Answer a question by ID.

        At least one of ``choice`` or ``freeform`` must be provided.

        Parameters
        ----------
        question_id
            The question to answer.
        choice
            Index of the selected option.
        freeform
            Freeform text answer.

        Returns
        -------
        EnrichmentQuestion | None
            The updated question, or ``None`` if not found or not pending.

        Raises
        ------
        ValueError
            If neither choice nor freeform is provided, or choice is
            out of range.

        Examples
        --------
        ```python
        q = queue.answer("q-001", choice=0)
        q.status  # QuestionStatus.ANSWERED

        q2 = queue.answer("q-002", freeform="It's Alex Torres")
        ```
        """
        if choice is None and freeform is None:
            msg = "Must provide at least one of 'choice' or 'freeform'."
            raise ValueError(msg)

        question = self._find_pending(question_id)
        if question is None:
            return None

        if choice is not None and (choice < 0 or choice >= len(question.options)):
            msg = f"Choice {choice} out of range for question with {len(question.options)} options."
            raise ValueError(msg)

        question.status = QuestionStatus.ANSWERED
        question.answer_choice = choice
        question.answer_freeform = freeform
        question.answered_at = time.time()
        return question

    def dismiss(self, question_id: str) -> EnrichmentQuestion | None:
        """Dismiss a pending question without answering.

        Parameters
        ----------
        question_id
            The question to dismiss.

        Returns
        -------
        EnrichmentQuestion | None
            The dismissed question, or ``None`` if not found or not pending.

        Examples
        --------
        ```python
        q = queue.dismiss("q-001")
        q.status  # QuestionStatus.DISMISSED
        ```
        """
        question = self._find_pending(question_id)
        if question is None:
            return None
        question.status = QuestionStatus.DISMISSED
        return question

    def dismiss_all(self) -> int:
        """Dismiss all pending questions.

        Returns
        -------
        int
            Number of questions dismissed.

        Examples
        --------
        ```python
        count = queue.dismiss_all()
        count  # 3
        ```
        """
        count = 0
        for q in self._questions:
            if q.status == QuestionStatus.PENDING:
                q.status = QuestionStatus.DISMISSED
                count += 1
        return count

    def remove_for_nodes(self, node_ids: list[str]) -> int:
        """Expire questions whose referenced nodes have been deleted.

        Call this after removing nodes to auto-dismiss related questions.

        Parameters
        ----------
        node_ids
            Node IDs that were deleted or changed.

        Returns
        -------
        int
            Number of questions expired.

        Examples
        --------
        ```python
        expired = queue.remove_for_nodes(["entity-alex"])
        expired  # 1
        ```
        """
        count = 0
        for q in self._questions:
            if q.status != QuestionStatus.PENDING:
                continue
            if any(nid in node_ids for nid in q.node_ids):
                q.status = QuestionStatus.EXPIRED
                count += 1
        return count

    def get(self, question_id: str) -> EnrichmentQuestion | None:
        """Get a question by ID regardless of status.

        Parameters
        ----------
        question_id
            The question identifier.

        Returns
        -------
        EnrichmentQuestion | None
            The question, or ``None`` if not found.

        Examples
        --------
        ```python
        q = queue.get("q-001")
        ```
        """
        for q in self._questions:
            if q.id == question_id:
                return q
        return None

    def stats(self) -> QueueStats:
        """Get summary statistics for the queue.

        Returns
        -------
        QueueStats
            Counts by status.

        Examples
        --------
        ```python
        s = queue.stats()
        s.pending  # 3
        s.answered  # 2
        ```
        """
        self._expire_stale()
        pending = sum(1 for q in self._questions if q.status == QuestionStatus.PENDING)
        answered = sum(1 for q in self._questions if q.status == QuestionStatus.ANSWERED)
        dismissed = sum(1 for q in self._questions if q.status == QuestionStatus.DISMISSED)
        expired = sum(1 for q in self._questions if q.status == QuestionStatus.EXPIRED)
        return QueueStats(
            total=len(self._questions),
            pending=pending,
            answered=answered,
            dismissed=dismissed,
            expired=expired,
        )

    def clear(self) -> None:
        """Remove all questions from the queue.

        Examples
        --------
        ```python
        queue.clear()
        queue.stats().total  # 0
        ```
        """
        self._questions.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_pending(self, question_id: str) -> EnrichmentQuestion | None:
        """Find a pending question by ID."""
        for q in self._questions:
            if q.id == question_id and q.status == QuestionStatus.PENDING:
                return q
        return None

    def _expire_stale(self) -> None:
        """Expire questions past their TTL."""
        now = time.time()
        for q in self._questions:
            if q.status == QuestionStatus.PENDING:
                if (now - q.created_at) > q.ttl_seconds:
                    q.status = QuestionStatus.EXPIRED


# ---------------------------------------------------------------------------
# Question generation helpers
# ---------------------------------------------------------------------------


def generate_question_id() -> str:
    """Generate a unique question ID.

    Returns
    -------
    str
        A UUID-based question identifier.

    Examples
    --------
    ```python
    import talk_box as tb

    qid = tb.generate_question_id()
    qid  # "eq-a1b2c3d4"
    ```
    """
    return f"eq-{uuid.uuid4().hex[:8]}"


def detect_name_ambiguity(kg: KnowledgeGraph) -> list[EnrichmentQuestion]:
    """Detect entities with duplicate or similar names.

    Generates questions for nodes whose names appear multiple times
    in the graph, suggesting the user clarify which entity is which.

    Parameters
    ----------
    kg
        The knowledge graph to examine.

    Returns
    -------
    list[EnrichmentQuestion]
        Questions about ambiguous entity names.

    Examples
    --------
    ```python
    import talk_box as tb

    kg = tb.KnowledgeGraph(":memory:")
    # Add two entities named "Alex"
    kg.add_node(tb.Node(id="e1", node_type=tb.NodeType.ENTITY, name="Alex"))
    kg.add_node(tb.Node(id="e2", node_type=tb.NodeType.ENTITY, name="Alex"))

    questions = tb.detect_name_ambiguity(kg)
    questions[0].text  # 'Multiple entities named "Alex" ...'
    ```
    """
    entities = kg.list_nodes(node_type=NodeType.ENTITY, limit=10_000)
    topics = kg.list_nodes(node_type=NodeType.TOPIC, limit=10_000)
    all_nodes = entities + topics

    # Group by lowercase name
    name_groups: dict[str, list[Any]] = {}
    for node in all_nodes:
        key = node.name.lower().strip()
        if key not in name_groups:
            name_groups[key] = []
        name_groups[key].append(node)

    questions: list[EnrichmentQuestion] = []
    for name, nodes in name_groups.items():
        if len(nodes) < 2:
            continue

        options = [
            QuestionOption(
                label=f"{n.name} ({n.node_type.value}, id={n.id})",
                node_ids=[n.id],
            )
            for n in nodes
        ]
        options.append(
            QuestionOption(label="They are the same entity", node_ids=[n.id for n in nodes])
        )
        options.append(QuestionOption(label="They are different entities"))

        # Impact scales with number of duplicates
        impact = min(1.0, 0.3 + 0.15 * (len(nodes) - 2))

        questions.append(
            EnrichmentQuestion(
                id=generate_question_id(),
                question_type=QuestionType.ENTITY_AMBIGUITY,
                text=f'Multiple entities named "{nodes[0].name}" exist in your knowledge graph. Are these the same?',
                options=options,
                node_ids=[n.id for n in nodes],
                confusion_impact=impact,
                context=f'Found {len(nodes)} nodes with name "{name}".',
            )
        )

    return questions


def detect_factual_conflicts(kg: KnowledgeGraph) -> list[EnrichmentQuestion]:
    """Detect nodes with contradictory metadata values.

    Looks for entity/topic pairs connected to the same document with
    conflicting metadata (e.g., different dates, statuses, or values
    for the same key).

    Parameters
    ----------
    kg
        The knowledge graph to examine.

    Returns
    -------
    list[EnrichmentQuestion]
        Questions about conflicting facts.

    Examples
    --------
    ```python
    import talk_box as tb

    questions = tb.detect_factual_conflicts(kg)
    ```
    """
    entities = kg.list_nodes(node_type=NodeType.ENTITY, limit=10_000)
    questions: list[EnrichmentQuestion] = []

    # Check entities that share metadata keys with different values
    meta_groups: dict[str, list[Any]] = {}
    for entity in entities:
        for key, value in entity.metadata.items():
            group_key = f"{entity.name.lower()}::{key}"
            if group_key not in meta_groups:
                meta_groups[group_key] = []
            meta_groups[group_key].append((entity, key, value))

    for group_key, entries in meta_groups.items():
        if len(entries) < 2:
            continue

        # Check if values actually conflict
        values = {str(v) for _, _, v in entries}
        if len(values) < 2:
            continue

        entity_name = entries[0][0].name
        meta_key = entries[0][1]
        options = [
            QuestionOption(
                label=f"{meta_key}={v} (from {e.id})",
                node_ids=[e.id],
            )
            for e, _, v in entries
        ]
        options.append(QuestionOption(label="None of these are correct"))

        impact = min(1.0, 0.4 + 0.1 * (len(entries) - 2))

        questions.append(
            EnrichmentQuestion(
                id=generate_question_id(),
                question_type=QuestionType.FACTUAL_CONFLICT,
                text=f'Conflicting values for "{meta_key}" on entity "{entity_name}". Which is correct?',
                options=options,
                node_ids=[e.id for e, _, _ in entries],
                confusion_impact=impact,
                context=f"Values found: {', '.join(values)}",
            )
        )

    return questions


def detect_weak_relationships(kg: KnowledgeGraph) -> list[EnrichmentQuestion]:
    """Detect edges with very low weight that may indicate uncertainty.

    Generates questions asking the user to confirm or deny weak
    relationships.

    Parameters
    ----------
    kg
        The knowledge graph to examine.

    Returns
    -------
    list[EnrichmentQuestion]
        Questions about uncertain relationships.

    Examples
    --------
    ```python
    import talk_box as tb

    questions = tb.detect_weak_relationships(kg)
    ```
    """
    entities = kg.list_nodes(node_type=NodeType.ENTITY, limit=10_000)
    questions: list[EnrichmentQuestion] = []
    seen_edges: set[tuple[str, str, str]] = set()

    for entity in entities:
        edges = kg.get_edges(entity.id)
        weak_edges = [e for e in edges if e.weight < 0.3]

        for edge in weak_edges:
            edge_key = (edge.source, edge.target, edge.relation)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            source_node = kg.get_node(edge.source)
            target_node = kg.get_node(edge.target)
            if source_node is None or target_node is None:
                continue

            options = [
                QuestionOption(
                    label=f"Yes, {source_node.name} {edge.relation} {target_node.name}",
                    node_ids=[edge.source, edge.target],
                ),
                QuestionOption(
                    label="No, this relationship is incorrect",
                    node_ids=[edge.source, edge.target],
                ),
                QuestionOption(label="I'm not sure"),
            ]

            questions.append(
                EnrichmentQuestion(
                    id=generate_question_id(),
                    question_type=QuestionType.RELATIONSHIP_UNCERTAINTY,
                    text=(
                        f'Is the relationship "{source_node.name}" '
                        f'→ "{edge.relation}" → "{target_node.name}" correct?'
                    ),
                    options=options,
                    node_ids=[edge.source, edge.target],
                    confusion_impact=0.3 + (0.3 - edge.weight),
                    context=f"Edge weight: {edge.weight:.2f} (low confidence).",
                )
            )

    return questions


# ---------------------------------------------------------------------------
# Top-level convenience function
# ---------------------------------------------------------------------------


def pending_questions(
    kg: KnowledgeGraph,
    queue: QuestionQueue,
    *,
    refresh: bool = False,
    generators: list[QuestionGeneratorFn] | None = None,
    sort_by: str = "confusion_impact",
    limit: int | None = None,
) -> list[EnrichmentQuestion]:
    """Get pending enrichment questions for a knowledge graph.

    Optionally refreshes the queue by running question generators to
    detect new ambiguities.

    Parameters
    ----------
    kg
        The knowledge graph to examine.
    queue
        The question queue to query (and optionally populate).
    refresh
        If ``True``, run generators to detect new questions before
        returning.
    generators
        Question generator functions to use when ``refresh=True``.
        Defaults to built-in detectors (name ambiguity, factual
        conflicts, weak relationships).
    sort_by
        Sort field: ``"confusion_impact"`` or ``"created_at"``.
    limit
        Maximum questions to return.

    Returns
    -------
    list[EnrichmentQuestion]
        Pending questions sorted by priority.

    Examples
    --------
    ```python
    import talk_box as tb

    kg = tb.KnowledgeGraph(":memory:")
    queue = tb.QuestionQueue()

    # Detect and return questions
    questions = tb.pending_questions(kg, queue, refresh=True)
    for q in questions:
        print(f"[{q.confusion_impact:.2f}] {q.text}")
    ```

    Use custom generators:

    ```python
    def my_detector(kg):
        return [...]

    questions = tb.pending_questions(
        kg, queue, refresh=True, generators=[my_detector]
    )
    ```
    """
    if refresh:
        if generators is None:
            generators = [
                detect_name_ambiguity,
                detect_factual_conflicts,
                detect_weak_relationships,
            ]
        for gen in generators:
            new_questions = gen(kg)
            queue.add_many(new_questions)

    return queue.pending_questions(sort_by=sort_by, limit=limit)
