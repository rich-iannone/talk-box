"""Tests for talk_box.confusion."""

from __future__ import annotations

import pytest

from talk_box.confusion import (
    ConfusionReport,
    ConfusionScore,
    confusion,
    node_confusion,
)
from talk_box.knowledge_graph import Edge, KnowledgeGraph, Node, NodeType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def kg() -> KnowledgeGraph:
    """A small graph with varying confusion levels."""
    g = KnowledgeGraph(":memory:")

    # Well-connected entity with full metadata
    g.add_node(
        Node(
            id="entity-python",
            node_type=NodeType.ENTITY,
            name="Python",
            content="A programming language.",
            metadata={"entity_type": "technology", "tags": ["programming"]},
        )
    )

    # Entity with no edges (orphan), no content
    g.add_node(
        Node(
            id="entity-orphan",
            node_type=NodeType.ENTITY,
            name="Mystery Entity",
            metadata={"entity_type": "unknown"},
        )
    )

    # Topic with edges
    g.add_node(
        Node(
            id="topic-ml",
            node_type=NodeType.TOPIC,
            name="machine-learning",
        )
    )

    # Document nodes (for edges)
    g.add_node(
        Node(
            id="doc-1",
            node_type=NodeType.DOCUMENT,
            name="ML Guide",
            content="Guide to machine learning with Python.",
            metadata={"_enriched": True, "tags": ["ml"]},
        )
    )
    g.add_node(
        Node(
            id="doc-2",
            node_type=NodeType.DOCUMENT,
            name="Coding Basics",
            content="Learn coding basics.",
            metadata={"_enriched": True, "tags": ["coding"]},
        )
    )
    g.add_node(
        Node(
            id="doc-3",
            node_type=NodeType.DOCUMENT,
            name="Data Science Intro",
            content="Getting started with data science.",
            metadata={"_enriched": True, "tags": ["data"]},
        )
    )

    # Edges: give Python good connectivity
    g.add_edge(Edge(source="doc-1", target="entity-python", relation="mentions"))
    g.add_edge(Edge(source="doc-2", target="entity-python", relation="mentions"))
    g.add_edge(Edge(source="doc-3", target="entity-python", relation="mentions"))
    g.add_edge(Edge(source="doc-1", target="topic-ml", relation="belongs_to"))

    return g


@pytest.fixture()
def empty_kg() -> KnowledgeGraph:
    return KnowledgeGraph(":memory:")


# ---------------------------------------------------------------------------
# ConfusionScore
# ---------------------------------------------------------------------------


class TestConfusionScore:
    def test_creation(self):
        s = ConfusionScore(
            node_id="x",
            node_name="Test",
            score=0.5,
            reasons=["orphan node (0 edges)"],
        )
        assert s.node_id == "x"
        assert s.node_name == "Test"
        assert s.score == 0.5
        assert len(s.reasons) == 1

    def test_defaults(self):
        s = ConfusionScore(node_id="a", node_name="A", score=0.0)
        assert s.reasons == []

    def test_frozen(self):
        s = ConfusionScore(node_id="a", node_name="A", score=0.0)
        with pytest.raises(AttributeError):
            s.score = 0.5  # type: ignore[misc]

    def test_repr(self):
        s = ConfusionScore(node_id="a", node_name="Test", score=0.42, reasons=["r1"])
        r = repr(s)
        assert "Test" in r
        assert "0.42" in r


# ---------------------------------------------------------------------------
# ConfusionReport
# ---------------------------------------------------------------------------


class TestConfusionReport:
    def test_empty_report(self):
        r = ConfusionReport()
        assert r.total == 0
        assert r.mean_score == 0.0
        assert r.max_score == 0.0
        assert r.confused_count == 0
        assert r.clear_count == 0

    def test_mean_score(self):
        r = ConfusionReport(
            scores=[
                ConfusionScore(node_id="a", node_name="A", score=0.2),
                ConfusionScore(node_id="b", node_name="B", score=0.8),
            ]
        )
        assert r.mean_score == pytest.approx(0.5)

    def test_max_score(self):
        r = ConfusionReport(
            scores=[
                ConfusionScore(node_id="a", node_name="A", score=0.2),
                ConfusionScore(node_id="b", node_name="B", score=0.8),
            ]
        )
        assert r.max_score == pytest.approx(0.8)

    def test_confused_nodes(self):
        r = ConfusionReport(
            scores=[
                ConfusionScore(node_id="a", node_name="A", score=0.1),
                ConfusionScore(node_id="b", node_name="B", score=0.5),
                ConfusionScore(node_id="c", node_name="C", score=0.9),
            ],
            threshold=0.3,
        )
        assert r.confused_count == 2
        assert r.clear_count == 1

    def test_custom_threshold(self):
        r = ConfusionReport(
            scores=[
                ConfusionScore(node_id="a", node_name="A", score=0.6),
            ],
            threshold=0.7,
        )
        assert r.confused_count == 0
        assert r.clear_count == 1

    def test_total(self):
        r = ConfusionReport(
            scores=[
                ConfusionScore(node_id="a", node_name="A", score=0.1),
                ConfusionScore(node_id="b", node_name="B", score=0.5),
            ]
        )
        assert r.total == 2

    def test_to_dict(self):
        r = ConfusionReport(
            scores=[
                ConfusionScore(node_id="a", node_name="A", score=0.5, reasons=["r1"]),
            ],
            threshold=0.3,
        )
        d = r.to_dict()
        assert d["total"] == 1
        assert d["confused_count"] == 1
        assert d["threshold"] == 0.3
        assert len(d["scores"]) == 1
        assert d["scores"][0]["node_id"] == "a"
        assert d["scores"][0]["reasons"] == ["r1"]

    def test_to_dict_rounding(self):
        r = ConfusionReport(
            scores=[
                ConfusionScore(node_id="a", node_name="A", score=0.33333),
            ]
        )
        d = r.to_dict()
        assert d["mean_score"] == 0.3333
        assert d["scores"][0]["score"] == 0.3333

    def test_repr(self):
        r = ConfusionReport(
            scores=[
                ConfusionScore(node_id="a", node_name="A", score=0.5),
            ]
        )
        assert "total=1" in repr(r)


# ---------------------------------------------------------------------------
# confusion() — main function
# ---------------------------------------------------------------------------


class TestConfusion:
    def test_empty_graph(self, empty_kg: KnowledgeGraph):
        report = confusion(empty_kg)
        assert report.total == 0
        assert report.mean_score == 0.0

    def test_scores_entities_and_topics(self, kg: KnowledgeGraph):
        report = confusion(kg)
        ids = [s.node_id for s in report.scores]
        assert "entity-python" in ids
        assert "entity-orphan" in ids
        assert "topic-ml" in ids
        # Documents should NOT be scored
        assert "doc-1" not in ids

    def test_orphan_has_high_score(self, kg: KnowledgeGraph):
        report = confusion(kg)
        orphan_score = next(s for s in report.scores if s.node_id == "entity-orphan")
        # Orphan with unknown type and no content should be highly confused
        assert orphan_score.score > 0.5

    def test_well_connected_has_low_score(self, kg: KnowledgeGraph):
        report = confusion(kg)
        python_score = next(s for s in report.scores if s.node_id == "entity-python")
        # Well-connected with good metadata
        assert python_score.score < 0.3

    def test_sorted_descending(self, kg: KnowledgeGraph):
        report = confusion(kg)
        scores = [s.score for s in report.scores]
        assert scores == sorted(scores, reverse=True)

    def test_filter_by_node_type(self, kg: KnowledgeGraph):
        report = confusion(kg, node_type=NodeType.ENTITY)
        for s in report.scores:
            node = kg.get_node(s.node_id)
            assert node is not None
            assert node.node_type == NodeType.ENTITY

    def test_custom_threshold(self, kg: KnowledgeGraph):
        report = confusion(kg, threshold=0.9)
        assert report.threshold == 0.9

    def test_documents_scored_when_requested(self, kg: KnowledgeGraph):
        report = confusion(kg, node_type=NodeType.DOCUMENT)
        ids = [s.node_id for s in report.scores]
        assert "doc-1" in ids
        assert "entity-python" not in ids


# ---------------------------------------------------------------------------
# node_confusion() — single node
# ---------------------------------------------------------------------------


class TestNodeConfusion:
    def test_by_node_object(self, kg: KnowledgeGraph):
        node = kg.get_node("entity-python")
        assert node is not None
        score = node_confusion(kg, node)
        assert 0.0 <= score.score <= 1.0
        assert score.node_id == "entity-python"

    def test_by_node_id(self, kg: KnowledgeGraph):
        score = node_confusion(kg, "entity-python")
        assert score.node_id == "entity-python"
        assert score.node_name == "Python"

    def test_missing_node_id(self, kg: KnowledgeGraph):
        score = node_confusion(kg, "nonexistent")
        assert score.score == 1.0
        assert "not found" in score.reasons[0]

    def test_orphan_node(self, kg: KnowledgeGraph):
        score = node_confusion(kg, "entity-orphan")
        assert score.score > 0.5
        reasons_str = " ".join(score.reasons)
        assert "orphan" in reasons_str.lower()

    def test_score_capped_at_one(self, kg: KnowledgeGraph):
        """Even with every factor maxed, score shouldn't exceed 1.0."""
        score = node_confusion(kg, "entity-orphan")
        assert score.score <= 1.0


# ---------------------------------------------------------------------------
# Factor: name ambiguity
# ---------------------------------------------------------------------------


class TestNameAmbiguity:
    def test_duplicate_names(self):
        """Nodes with identical names should have high name ambiguity."""
        kg = KnowledgeGraph(":memory:")
        kg.add_node(
            Node(
                id="e1",
                node_type=NodeType.ENTITY,
                name="Python",
                content="The language.",
                metadata={"entity_type": "technology"},
            )
        )
        kg.add_node(
            Node(
                id="e2",
                node_type=NodeType.ENTITY,
                name="Python",
                content="The snake.",
                metadata={"entity_type": "animal"},
            )
        )
        # Connect both so connectivity doesn't dominate
        kg.add_node(Node(id="d1", node_type=NodeType.DOCUMENT, name="Doc 1"))
        kg.add_edge(Edge(source="d1", target="e1", relation="mentions"))
        kg.add_edge(Edge(source="d1", target="e2", relation="mentions"))

        report = confusion(kg)
        e1_score = next(s for s in report.scores if s.node_id == "e1")
        reasons_str = " ".join(e1_score.reasons)
        assert "name collision" in reasons_str.lower()

    def test_substring_overlap(self):
        """Names that are substrings of each other should raise ambiguity."""
        kg = KnowledgeGraph(":memory:")
        kg.add_node(
            Node(
                id="e1",
                node_type=NodeType.ENTITY,
                name="Python",
                content="Programming language.",
                metadata={"entity_type": "technology"},
            )
        )
        kg.add_node(
            Node(
                id="e2",
                node_type=NodeType.ENTITY,
                name="Python Programming",
                content="A course.",
                metadata={"entity_type": "course"},
            )
        )
        # Connect both
        kg.add_node(Node(id="d1", node_type=NodeType.DOCUMENT, name="Doc 1"))
        kg.add_edge(Edge(source="d1", target="e1", relation="mentions"))
        kg.add_edge(Edge(source="d1", target="e2", relation="mentions"))

        report = confusion(kg)
        e1_score = next(s for s in report.scores if s.node_id == "e1")
        reasons_str = " ".join(e1_score.reasons)
        assert "overlap" in reasons_str.lower()

    def test_unique_name_no_ambiguity(self, kg: KnowledgeGraph):
        """A unique name should not trigger name ambiguity."""
        score = node_confusion(kg, "entity-python")
        reasons_str = " ".join(score.reasons)
        assert "name collision" not in reasons_str.lower()


# ---------------------------------------------------------------------------
# Factor: weak edges
# ---------------------------------------------------------------------------


class TestWeakEdges:
    def test_weak_edges_scored(self):
        kg = KnowledgeGraph(":memory:")
        kg.add_node(
            Node(
                id="e1",
                node_type=NodeType.ENTITY,
                name="Weak Entity",
                content="Has weak edges.",
                metadata={"entity_type": "thing"},
            )
        )
        kg.add_node(Node(id="d1", node_type=NodeType.DOCUMENT, name="Doc 1"))
        kg.add_edge(Edge(source="d1", target="e1", relation="mentions", weight=0.1))

        score = node_confusion(kg, "e1")
        reasons_str = " ".join(score.reasons)
        assert "weak" in reasons_str.lower()

    def test_strong_edges_no_penalty(self, kg: KnowledgeGraph):
        """Edges with weight >= 0.3 should not trigger weak edges factor."""
        score = node_confusion(kg, "entity-python")
        reasons_str = " ".join(score.reasons)
        assert "weak edge" not in reasons_str.lower()


# ---------------------------------------------------------------------------
# Factor: metadata
# ---------------------------------------------------------------------------


class TestMetadataFactor:
    def test_missing_content_and_tags(self):
        kg = KnowledgeGraph(":memory:")
        kg.add_node(
            Node(
                id="e1",
                node_type=NodeType.ENTITY,
                name="Bare Entity",
                metadata={"entity_type": "thing"},
            )
        )
        kg.add_node(Node(id="d1", node_type=NodeType.DOCUMENT, name="Doc"))
        kg.add_edge(Edge(source="d1", target="e1", relation="mentions"))

        score = node_confusion(kg, "e1")
        reasons_str = " ".join(score.reasons)
        assert "missing metadata" in reasons_str.lower()

    def test_full_metadata_no_penalty(self, kg: KnowledgeGraph):
        """entity-python has content, tags, entity_type — no metadata penalty."""
        score = node_confusion(kg, "entity-python")
        reasons_str = " ".join(score.reasons)
        assert "missing metadata" not in reasons_str.lower()


# ---------------------------------------------------------------------------
# Factor: type coverage
# ---------------------------------------------------------------------------


class TestTypeCoverage:
    def test_unknown_entity_type(self):
        kg = KnowledgeGraph(":memory:")
        kg.add_node(
            Node(
                id="e1",
                node_type=NodeType.ENTITY,
                name="Untyped",
                content="No type.",
                metadata={"entity_type": "unknown", "tags": ["x"]},
            )
        )
        kg.add_node(Node(id="d1", node_type=NodeType.DOCUMENT, name="Doc"))
        kg.add_edge(Edge(source="d1", target="e1", relation="mentions"))

        score = node_confusion(kg, "e1")
        reasons_str = " ".join(score.reasons)
        assert "entity type" in reasons_str.lower()

    def test_typed_entity_no_penalty(self, kg: KnowledgeGraph):
        score = node_confusion(kg, "entity-python")
        reasons_str = " ".join(score.reasons)
        assert "entity type" not in reasons_str.lower()

    def test_topic_nodes_skip_type_check(self, kg: KnowledgeGraph):
        """Type coverage factor only applies to entities, not topics."""
        score = node_confusion(kg, "topic-ml")
        reasons_str = " ".join(score.reasons)
        assert "entity type" not in reasons_str.lower()


# ---------------------------------------------------------------------------
# Top-level imports
# ---------------------------------------------------------------------------


class TestImport:
    def test_confusion_importable(self):
        from talk_box.confusion import confusion

        assert confusion is not None

    def test_confusion_score_importable(self):
        from talk_box.confusion import ConfusionScore

        assert ConfusionScore is not None

    def test_confusion_report_importable(self):
        from talk_box.confusion import ConfusionReport

        assert ConfusionReport is not None

    def test_node_confusion_importable(self):
        from talk_box.confusion import node_confusion

        assert node_confusion is not None
