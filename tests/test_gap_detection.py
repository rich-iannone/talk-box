"""Tests for talk_box.gap_detection."""

from __future__ import annotations

import time

import pytest

from talk_box.gap_detection import (
    Gap,
    GapReport,
    GapType,
    detect_gaps,
)
from talk_box.knowledge_graph import Edge, KnowledgeGraph, Node, NodeType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def kg() -> KnowledgeGraph:
    """A graph with various structural gaps."""
    g = KnowledgeGraph(":memory:")

    # Well-connected document
    g.add_node(
        Node(
            id="doc-1",
            node_type=NodeType.DOCUMENT,
            name="ML Guide",
            content="Guide to machine learning.",
        )
    )
    # Orphan document (no edges)
    g.add_node(
        Node(
            id="doc-orphan",
            node_type=NodeType.DOCUMENT,
            name="Forgotten Doc",
            content="Nobody links to me.",
        )
    )

    # Well-connected entity
    g.add_node(
        Node(
            id="entity-python",
            node_type=NodeType.ENTITY,
            name="Python",
            content="Programming language.",
        )
    )
    # Entity with only 1 edge (thin cluster candidate)
    g.add_node(
        Node(
            id="entity-rust",
            node_type=NodeType.ENTITY,
            name="Rust",
            content="Systems language.",
        )
    )
    # Orphan entity
    g.add_node(
        Node(
            id="entity-orphan",
            node_type=NodeType.ENTITY,
            name="Orphan Entity",
        )
    )

    # Co-occurring entities (for missing relationship detection)
    g.add_node(
        Node(
            id="entity-sklearn",
            node_type=NodeType.ENTITY,
            name="scikit-learn",
        )
    )

    # Topic with only 1 doc (weak topic)
    g.add_node(Node(id="topic-ml", node_type=NodeType.TOPIC, name="machine-learning"))
    # Topic with 0 docs (also weak)
    g.add_node(Node(id="topic-empty", node_type=NodeType.TOPIC, name="empty-topic"))

    # Edges
    g.add_edge(Edge(source="doc-1", target="entity-python", relation="mentions"))
    g.add_edge(Edge(source="doc-1", target="entity-sklearn", relation="mentions"))
    g.add_edge(Edge(source="doc-1", target="entity-rust", relation="mentions"))
    g.add_edge(Edge(source="doc-1", target="topic-ml", relation="belongs_to"))

    # Give Python another connection
    g.add_node(
        Node(
            id="doc-2",
            node_type=NodeType.DOCUMENT,
            name="Python Intro",
            content="Intro to Python.",
        )
    )
    g.add_edge(Edge(source="doc-2", target="entity-python", relation="mentions"))

    return g


@pytest.fixture()
def empty_kg() -> KnowledgeGraph:
    return KnowledgeGraph(":memory:")


@pytest.fixture()
def stale_kg() -> KnowledgeGraph:
    """A graph with old nodes."""
    g = KnowledgeGraph(":memory:")
    old_time = time.time() - (60 * 86_400)  # 60 days ago

    g.add_node(
        Node(
            id="doc-old",
            node_type=NodeType.DOCUMENT,
            name="Old Doc",
            content="Very old.",
            created_at=old_time,
            updated_at=old_time,
        )
    )
    g.add_node(
        Node(
            id="doc-new",
            node_type=NodeType.DOCUMENT,
            name="New Doc",
            content="Just updated.",
        )
    )
    g.add_edge(Edge(source="doc-old", target="doc-new", relation="related_to"))

    return g


# ---------------------------------------------------------------------------
# GapType enum
# ---------------------------------------------------------------------------


class TestGapType:
    def test_values(self):
        assert GapType.ORPHAN.value == "orphan"
        assert GapType.THIN_CLUSTER.value == "thin_cluster"
        assert GapType.STALE.value == "stale"
        assert GapType.MISSING_RELATIONSHIP.value == "missing_relationship"
        assert GapType.WEAK_TOPIC.value == "weak_topic"


# ---------------------------------------------------------------------------
# Gap dataclass
# ---------------------------------------------------------------------------


class TestGap:
    def test_creation(self):
        g = Gap(
            gap_type=GapType.ORPHAN,
            node_ids=["x"],
            description="Orphan node.",
            suggestion="Connect it.",
            severity=0.7,
        )
        assert g.gap_type == GapType.ORPHAN
        assert g.node_ids == ["x"]
        assert g.severity == 0.7

    def test_defaults(self):
        g = Gap(gap_type=GapType.STALE)
        assert g.node_ids == []
        assert g.description == ""
        assert g.suggestion == ""
        assert g.severity == 0.5

    def test_frozen(self):
        g = Gap(gap_type=GapType.ORPHAN)
        with pytest.raises(AttributeError):
            g.severity = 1.0  # type: ignore[misc]

    def test_repr(self):
        g = Gap(gap_type=GapType.ORPHAN, node_ids=["a", "b"], severity=0.7)
        r = repr(g)
        assert "orphan" in r
        assert "nodes=2" in r
        assert "0.70" in r


# ---------------------------------------------------------------------------
# GapReport
# ---------------------------------------------------------------------------


class TestGapReport:
    def test_empty_report(self):
        r = GapReport()
        assert r.total == 0
        assert r.severity_score == 0.0
        assert r.orphan_count == 0
        assert r.stale_count == 0
        assert r.suggestions == []

    def test_total(self):
        r = GapReport(
            gaps=[
                Gap(gap_type=GapType.ORPHAN),
                Gap(gap_type=GapType.STALE),
            ]
        )
        assert r.total == 2

    def test_severity_score(self):
        r = GapReport(
            gaps=[
                Gap(gap_type=GapType.ORPHAN, severity=0.8),
                Gap(gap_type=GapType.STALE, severity=0.4),
            ]
        )
        assert r.severity_score == pytest.approx(0.6)

    def test_orphan_count(self):
        r = GapReport(
            gaps=[
                Gap(gap_type=GapType.ORPHAN),
                Gap(gap_type=GapType.ORPHAN),
                Gap(gap_type=GapType.STALE),
            ]
        )
        assert r.orphan_count == 2

    def test_stale_count(self):
        r = GapReport(
            gaps=[
                Gap(gap_type=GapType.STALE),
                Gap(gap_type=GapType.STALE),
                Gap(gap_type=GapType.ORPHAN),
            ]
        )
        assert r.stale_count == 2

    def test_by_type(self):
        r = GapReport(
            gaps=[
                Gap(gap_type=GapType.ORPHAN, node_ids=["a"]),
                Gap(gap_type=GapType.STALE, node_ids=["b"]),
                Gap(gap_type=GapType.ORPHAN, node_ids=["c"]),
            ]
        )
        orphans = r.by_type(GapType.ORPHAN)
        assert len(orphans) == 2
        stale = r.by_type(GapType.STALE)
        assert len(stale) == 1

    def test_suggestions(self):
        r = GapReport(
            gaps=[
                Gap(gap_type=GapType.ORPHAN, suggestion="Fix A"),
                Gap(gap_type=GapType.STALE, suggestion=""),
                Gap(gap_type=GapType.WEAK_TOPIC, suggestion="Fix C"),
            ]
        )
        assert r.suggestions == ["Fix A", "Fix C"]

    def test_to_dict(self):
        r = GapReport(
            gaps=[
                Gap(
                    gap_type=GapType.ORPHAN,
                    node_ids=["x"],
                    description="Orphan.",
                    suggestion="Fix.",
                    severity=0.7,
                )
            ]
        )
        d = r.to_dict()
        assert d["total"] == 1
        assert d["orphan_count"] == 1
        assert len(d["gaps"]) == 1
        assert d["gaps"][0]["type"] == "orphan"
        assert d["gaps"][0]["node_ids"] == ["x"]

    def test_repr(self):
        r = GapReport(gaps=[Gap(gap_type=GapType.ORPHAN, severity=0.7)])
        assert "total=1" in repr(r)


# ---------------------------------------------------------------------------
# detect_gaps() — orphan detection
# ---------------------------------------------------------------------------


class TestDetectGapsOrphans:
    def test_finds_orphan_nodes(self, kg: KnowledgeGraph):
        report = detect_gaps(kg)
        orphan_ids = set()
        for g in report.by_type(GapType.ORPHAN):
            orphan_ids.update(g.node_ids)
        assert "doc-orphan" in orphan_ids
        assert "entity-orphan" in orphan_ids

    def test_connected_nodes_not_orphans(self, kg: KnowledgeGraph):
        report = detect_gaps(kg)
        orphan_ids = set()
        for g in report.by_type(GapType.ORPHAN):
            orphan_ids.update(g.node_ids)
        assert "entity-python" not in orphan_ids
        assert "doc-1" not in orphan_ids

    def test_orphan_severity(self, kg: KnowledgeGraph):
        report = detect_gaps(kg)
        orphans = report.by_type(GapType.ORPHAN)
        for gap in orphans:
            assert gap.severity == 0.7


# ---------------------------------------------------------------------------
# detect_gaps() — thin clusters
# ---------------------------------------------------------------------------


class TestDetectGapsThinClusters:
    def test_finds_thin_entity(self, kg: KnowledgeGraph):
        """entity-rust has only 1 edge — should be thin with threshold=2."""
        report = detect_gaps(kg, thin_cluster_threshold=2)
        thin_ids = set()
        for g in report.by_type(GapType.THIN_CLUSTER):
            thin_ids.update(g.node_ids)
        assert "entity-rust" in thin_ids

    def test_well_connected_not_thin(self, kg: KnowledgeGraph):
        """entity-python has 2+ edges — should not be thin."""
        report = detect_gaps(kg, thin_cluster_threshold=2)
        thin_ids = set()
        for g in report.by_type(GapType.THIN_CLUSTER):
            thin_ids.update(g.node_ids)
        assert "entity-python" not in thin_ids

    def test_documents_excluded_from_thin(self, kg: KnowledgeGraph):
        """Document nodes should not be flagged as thin clusters."""
        report = detect_gaps(kg, thin_cluster_threshold=100)
        thin_ids = set()
        for g in report.by_type(GapType.THIN_CLUSTER):
            thin_ids.update(g.node_ids)
        # doc-2 has only 1 edge but is a DOCUMENT
        assert "doc-2" not in thin_ids

    def test_custom_threshold(self, kg: KnowledgeGraph):
        """With threshold=5, even well-connected nodes become thin."""
        report = detect_gaps(kg, thin_cluster_threshold=5)
        thin_ids = set()
        for g in report.by_type(GapType.THIN_CLUSTER):
            thin_ids.update(g.node_ids)
        # sklearn has 1 edge, rust has 1 edge — both thin
        assert "entity-sklearn" in thin_ids
        assert "entity-rust" in thin_ids


# ---------------------------------------------------------------------------
# detect_gaps() — stale detection
# ---------------------------------------------------------------------------


class TestDetectGapsStale:
    def test_finds_stale_nodes(self, stale_kg: KnowledgeGraph):
        report = detect_gaps(stale_kg, stale_days=30)
        stale_ids = set()
        for g in report.by_type(GapType.STALE):
            stale_ids.update(g.node_ids)
        assert "doc-old" in stale_ids

    def test_fresh_nodes_not_stale(self, stale_kg: KnowledgeGraph):
        report = detect_gaps(stale_kg, stale_days=30)
        stale_ids = set()
        for g in report.by_type(GapType.STALE):
            stale_ids.update(g.node_ids)
        assert "doc-new" not in stale_ids

    def test_custom_stale_days(self, stale_kg: KnowledgeGraph):
        """With stale_days=90, the 60-day-old node is not stale."""
        report = detect_gaps(stale_kg, stale_days=90)
        stale_ids = set()
        for g in report.by_type(GapType.STALE):
            stale_ids.update(g.node_ids)
        assert "doc-old" not in stale_ids

    def test_no_stale_in_fresh_graph(self, kg: KnowledgeGraph):
        """Default graph has freshly-created nodes."""
        report = detect_gaps(kg, stale_days=30)
        assert report.stale_count == 0


# ---------------------------------------------------------------------------
# detect_gaps() — missing relationships
# ---------------------------------------------------------------------------


class TestDetectGapsMissingRelationships:
    def test_finds_co_occurring_entities_without_edge(self, kg: KnowledgeGraph):
        """sklearn and rust co-occur in doc-1 but have no direct edge."""
        report = detect_gaps(kg)
        missing = report.by_type(GapType.MISSING_RELATIONSHIP)
        pair_sets = [set(g.node_ids) for g in missing]
        # At minimum, sklearn-rust should be flagged
        assert {"entity-sklearn", "entity-rust"} in pair_sets

    def test_connected_entities_not_flagged(self, kg: KnowledgeGraph):
        """If we add a direct edge, the pair should not be flagged."""
        kg.add_edge(
            Edge(
                source="entity-sklearn",
                target="entity-rust",
                relation="related_to",
            )
        )
        report = detect_gaps(kg)
        missing = report.by_type(GapType.MISSING_RELATIONSHIP)
        pair_sets = [set(g.node_ids) for g in missing]
        assert {"entity-sklearn", "entity-rust"} not in pair_sets

    def test_no_missing_when_no_cooccurrence(self, empty_kg: KnowledgeGraph):
        report = detect_gaps(empty_kg)
        assert len(report.by_type(GapType.MISSING_RELATIONSHIP)) == 0


# ---------------------------------------------------------------------------
# detect_gaps() — weak topics
# ---------------------------------------------------------------------------


class TestDetectGapsWeakTopics:
    def test_finds_empty_topic(self, kg: KnowledgeGraph):
        report = detect_gaps(kg, min_topic_docs=2)
        weak = report.by_type(GapType.WEAK_TOPIC)
        weak_ids = set()
        for g in weak:
            weak_ids.update(g.node_ids)
        assert "topic-empty" in weak_ids

    def test_finds_topic_below_threshold(self, kg: KnowledgeGraph):
        """topic-ml has 1 doc; with min_topic_docs=2, it's weak."""
        report = detect_gaps(kg, min_topic_docs=2)
        weak_ids = set()
        for g in report.by_type(GapType.WEAK_TOPIC):
            weak_ids.update(g.node_ids)
        assert "topic-ml" in weak_ids

    def test_sufficient_topic_not_flagged(self, kg: KnowledgeGraph):
        """With min_topic_docs=1, topic-ml should be fine."""
        report = detect_gaps(kg, min_topic_docs=1)
        weak_ids = set()
        for g in report.by_type(GapType.WEAK_TOPIC):
            weak_ids.update(g.node_ids)
        assert "topic-ml" not in weak_ids


# ---------------------------------------------------------------------------
# detect_gaps() — overall behavior
# ---------------------------------------------------------------------------


class TestDetectGapsGeneral:
    def test_empty_graph(self, empty_kg: KnowledgeGraph):
        report = detect_gaps(empty_kg)
        assert report.total == 0
        assert report.severity_score == 0.0

    def test_sorted_by_severity(self, kg: KnowledgeGraph):
        report = detect_gaps(kg)
        severities = [g.severity for g in report.gaps]
        assert severities == sorted(severities, reverse=True)

    def test_suggestions_populated(self, kg: KnowledgeGraph):
        report = detect_gaps(kg)
        assert len(report.suggestions) > 0
        for s in report.suggestions:
            assert isinstance(s, str)
            assert len(s) > 0


# ---------------------------------------------------------------------------
# Top-level imports
# ---------------------------------------------------------------------------


class TestImport:
    def test_detect_gaps_importable(self):
        from talk_box.gap_detection import detect_gaps

        assert detect_gaps is not None

    def test_gap_importable(self):
        from talk_box.gap_detection import Gap

        assert Gap is not None

    def test_gap_report_importable(self):
        from talk_box.gap_detection import GapReport

        assert GapReport is not None

    def test_gap_type_importable(self):
        from talk_box.gap_detection import GapType

        assert GapType is not None
