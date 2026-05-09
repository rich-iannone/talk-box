"""Tests for talk_box.freshness."""

from __future__ import annotations

import time

import pytest

from talk_box.freshness import (
    FreshnessEntry,
    FreshnessReport,
    FreshnessStatus,
    freshness_report,
)
from talk_box.knowledge_graph import KnowledgeGraph, Node, NodeType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def kg() -> KnowledgeGraph:
    """A graph with nodes of varying ages."""
    g = KnowledgeGraph(":memory:")
    now = time.time()

    # Fresh node (1 day old)
    g.add_node(
        Node(
            id="doc-fresh",
            node_type=NodeType.DOCUMENT,
            name="Fresh Doc",
            content="Just updated.",
            created_at=now - 86_400,
            updated_at=now - 86_400,
        )
    )
    # Aging node (15 days old)
    g.add_node(
        Node(
            id="doc-aging",
            node_type=NodeType.DOCUMENT,
            name="Aging Doc",
            content="Getting older.",
            created_at=now - (15 * 86_400),
            updated_at=now - (15 * 86_400),
        )
    )
    # Stale node (60 days old)
    g.add_node(
        Node(
            id="doc-stale",
            node_type=NodeType.DOCUMENT,
            name="Stale Doc",
            content="Very old.",
            created_at=now - (60 * 86_400),
            updated_at=now - (60 * 86_400),
        )
    )
    # Entity (3 days old)
    g.add_node(
        Node(
            id="entity-new",
            node_type=NodeType.ENTITY,
            name="New Entity",
            created_at=now - (3 * 86_400),
            updated_at=now - (3 * 86_400),
        )
    )
    # Topic (45 days old)
    g.add_node(
        Node(
            id="topic-old",
            node_type=NodeType.TOPIC,
            name="Old Topic",
            created_at=now - (45 * 86_400),
            updated_at=now - (45 * 86_400),
        )
    )

    return g


@pytest.fixture()
def empty_kg() -> KnowledgeGraph:
    return KnowledgeGraph(":memory:")


# ---------------------------------------------------------------------------
# FreshnessStatus enum
# ---------------------------------------------------------------------------


class TestFreshnessStatus:
    def test_values(self):
        assert FreshnessStatus.FRESH.value == "fresh"
        assert FreshnessStatus.AGING.value == "aging"
        assert FreshnessStatus.STALE.value == "stale"
        assert FreshnessStatus.UNKNOWN.value == "unknown"


# ---------------------------------------------------------------------------
# FreshnessEntry
# ---------------------------------------------------------------------------


class TestFreshnessEntry:
    def test_creation(self):
        e = FreshnessEntry(
            node_id="x",
            node_name="Test",
            node_type=NodeType.DOCUMENT,
            status=FreshnessStatus.FRESH,
            age_days=3,
            updated_at=1000.0,
        )
        assert e.node_id == "x"
        assert e.node_name == "Test"
        assert e.status == FreshnessStatus.FRESH
        assert e.age_days == 3

    def test_defaults(self):
        e = FreshnessEntry(
            node_id="a",
            node_name="A",
            node_type=NodeType.ENTITY,
            status=FreshnessStatus.UNKNOWN,
        )
        assert e.age_days == -1
        assert e.updated_at == 0.0

    def test_frozen(self):
        e = FreshnessEntry(
            node_id="a",
            node_name="A",
            node_type=NodeType.ENTITY,
            status=FreshnessStatus.FRESH,
        )
        with pytest.raises(AttributeError):
            e.age_days = 10  # type: ignore[misc]

    def test_repr(self):
        e = FreshnessEntry(
            node_id="a",
            node_name="Test Node",
            node_type=NodeType.DOCUMENT,
            status=FreshnessStatus.STALE,
            age_days=45,
        )
        r = repr(e)
        assert "Test Node" in r
        assert "stale" in r
        assert "45" in r


# ---------------------------------------------------------------------------
# FreshnessReport — properties
# ---------------------------------------------------------------------------


class TestFreshnessReport:
    def test_empty_report(self):
        r = FreshnessReport()
        assert r.total == 0
        assert r.fresh_count == 0
        assert r.aging_count == 0
        assert r.stale_count == 0
        assert r.unknown_count == 0
        assert r.coverage == 0.0
        assert r.mean_age_days == 0.0
        assert r.max_age_days == 0

    def test_counts(self):
        r = FreshnessReport(
            entries=[
                FreshnessEntry("a", "A", NodeType.DOCUMENT, FreshnessStatus.FRESH, 2),
                FreshnessEntry("b", "B", NodeType.DOCUMENT, FreshnessStatus.FRESH, 5),
                FreshnessEntry("c", "C", NodeType.DOCUMENT, FreshnessStatus.AGING, 15),
                FreshnessEntry("d", "D", NodeType.DOCUMENT, FreshnessStatus.STALE, 60),
                FreshnessEntry("e", "E", NodeType.ENTITY, FreshnessStatus.UNKNOWN, -1),
            ]
        )
        assert r.total == 5
        assert r.fresh_count == 2
        assert r.aging_count == 1
        assert r.stale_count == 1
        assert r.unknown_count == 1

    def test_coverage(self):
        r = FreshnessReport(
            entries=[
                FreshnessEntry("a", "A", NodeType.DOCUMENT, FreshnessStatus.FRESH, 1),
                FreshnessEntry("b", "B", NodeType.DOCUMENT, FreshnessStatus.STALE, 50),
                FreshnessEntry("c", "C", NodeType.DOCUMENT, FreshnessStatus.FRESH, 3),
                FreshnessEntry("d", "D", NodeType.DOCUMENT, FreshnessStatus.FRESH, 5),
            ]
        )
        assert r.coverage == pytest.approx(0.75)

    def test_mean_age_days(self):
        r = FreshnessReport(
            entries=[
                FreshnessEntry("a", "A", NodeType.DOCUMENT, FreshnessStatus.FRESH, 10),
                FreshnessEntry("b", "B", NodeType.DOCUMENT, FreshnessStatus.STALE, 30),
                FreshnessEntry("c", "C", NodeType.ENTITY, FreshnessStatus.UNKNOWN, -1),
            ]
        )
        # Only nodes with age >= 0 counted: (10 + 30) / 2 = 20
        assert r.mean_age_days == pytest.approx(20.0)

    def test_max_age_days(self):
        r = FreshnessReport(
            entries=[
                FreshnessEntry("a", "A", NodeType.DOCUMENT, FreshnessStatus.FRESH, 5),
                FreshnessEntry("b", "B", NodeType.DOCUMENT, FreshnessStatus.STALE, 90),
            ]
        )
        assert r.max_age_days == 90

    def test_stale_entries(self):
        r = FreshnessReport(
            entries=[
                FreshnessEntry("a", "A", NodeType.DOCUMENT, FreshnessStatus.FRESH, 1),
                FreshnessEntry("b", "B", NodeType.DOCUMENT, FreshnessStatus.STALE, 40),
                FreshnessEntry("c", "C", NodeType.DOCUMENT, FreshnessStatus.STALE, 60),
            ]
        )
        stale = r.stale_entries
        assert len(stale) == 2
        assert all(e.status == FreshnessStatus.STALE for e in stale)

    def test_fresh_entries(self):
        r = FreshnessReport(
            entries=[
                FreshnessEntry("a", "A", NodeType.DOCUMENT, FreshnessStatus.FRESH, 2),
                FreshnessEntry("b", "B", NodeType.DOCUMENT, FreshnessStatus.STALE, 40),
            ]
        )
        assert len(r.fresh_entries) == 1

    def test_by_type(self):
        r = FreshnessReport(
            entries=[
                FreshnessEntry("a", "A", NodeType.DOCUMENT, FreshnessStatus.FRESH, 1),
                FreshnessEntry("b", "B", NodeType.ENTITY, FreshnessStatus.FRESH, 2),
                FreshnessEntry("c", "C", NodeType.DOCUMENT, FreshnessStatus.STALE, 50),
            ]
        )
        docs = r.by_type(NodeType.DOCUMENT)
        assert len(docs) == 2
        entities = r.by_type(NodeType.ENTITY)
        assert len(entities) == 1

    def test_by_status(self):
        r = FreshnessReport(
            entries=[
                FreshnessEntry("a", "A", NodeType.DOCUMENT, FreshnessStatus.FRESH, 1),
                FreshnessEntry("b", "B", NodeType.DOCUMENT, FreshnessStatus.AGING, 15),
            ]
        )
        aging = r.by_status(FreshnessStatus.AGING)
        assert len(aging) == 1
        assert aging[0].node_id == "b"

    def test_to_dict(self):
        r = FreshnessReport(
            entries=[
                FreshnessEntry("a", "A", NodeType.DOCUMENT, FreshnessStatus.FRESH, 3),
            ],
            fresh_days=7.0,
            stale_days=30.0,
        )
        d = r.to_dict()
        assert d["total"] == 1
        assert d["fresh_count"] == 1
        assert d["fresh_days"] == 7.0
        assert d["stale_days"] == 30.0
        assert len(d["entries"]) == 1
        assert d["entries"][0]["node_id"] == "a"
        assert d["entries"][0]["status"] == "fresh"

    def test_repr(self):
        r = FreshnessReport(
            entries=[
                FreshnessEntry("a", "A", NodeType.DOCUMENT, FreshnessStatus.FRESH, 1),
                FreshnessEntry("b", "B", NodeType.DOCUMENT, FreshnessStatus.STALE, 40),
            ]
        )
        assert "total=2" in repr(r)
        assert "fresh=1" in repr(r)
        assert "stale=1" in repr(r)


# ---------------------------------------------------------------------------
# freshness_report() — main function
# ---------------------------------------------------------------------------


class TestFreshnessReportFunction:
    def test_empty_graph(self, empty_kg: KnowledgeGraph):
        report = freshness_report(empty_kg)
        assert report.total == 0
        assert report.coverage == 0.0

    def test_classifies_fresh(self, kg: KnowledgeGraph):
        report = freshness_report(kg)
        fresh_ids = [e.node_id for e in report.fresh_entries]
        assert "doc-fresh" in fresh_ids
        assert "entity-new" in fresh_ids

    def test_classifies_aging(self, kg: KnowledgeGraph):
        report = freshness_report(kg)
        aging = report.by_status(FreshnessStatus.AGING)
        aging_ids = [e.node_id for e in aging]
        assert "doc-aging" in aging_ids

    def test_classifies_stale(self, kg: KnowledgeGraph):
        report = freshness_report(kg)
        stale_ids = [e.node_id for e in report.stale_entries]
        assert "doc-stale" in stale_ids
        assert "topic-old" in stale_ids

    def test_sorted_oldest_first(self, kg: KnowledgeGraph):
        report = freshness_report(kg)
        ages = [e.age_days for e in report.entries]
        assert ages == sorted(ages, reverse=True)

    def test_filter_by_node_type(self, kg: KnowledgeGraph):
        report = freshness_report(kg, node_type=NodeType.DOCUMENT)
        for entry in report.entries:
            assert entry.node_type == NodeType.DOCUMENT
        assert report.total == 3

    def test_custom_fresh_days(self, kg: KnowledgeGraph):
        """With fresh_days=0, only nodes updated today are fresh."""
        report = freshness_report(kg, fresh_days=0)
        assert report.fresh_count == 0

    def test_custom_stale_days(self, kg: KnowledgeGraph):
        """With stale_days=10, doc-aging (15 days) becomes stale."""
        report = freshness_report(kg, stale_days=10)
        stale_ids = [e.node_id for e in report.stale_entries]
        assert "doc-aging" in stale_ids

    def test_unknown_timestamp(self):
        """FreshnessEntry with UNKNOWN status has age_days=-1."""
        entry = FreshnessEntry(
            node_id="no-time",
            node_name="No Timestamp",
            node_type=NodeType.ENTITY,
            status=FreshnessStatus.UNKNOWN,
            age_days=-1,
            updated_at=0.0,
        )
        assert entry.status == FreshnessStatus.UNKNOWN
        assert entry.age_days == -1

    def test_boundary_fresh(self, empty_kg: KnowledgeGraph):
        """Node exactly at fresh_days boundary should be FRESH."""
        now = time.time()
        empty_kg.add_node(
            Node(
                id="boundary",
                node_type=NodeType.DOCUMENT,
                name="Boundary",
                created_at=now - (7 * 86_400),
                updated_at=now - (7 * 86_400),
            )
        )
        report = freshness_report(empty_kg, fresh_days=7)
        assert report.entries[0].status == FreshnessStatus.FRESH

    def test_boundary_stale(self, empty_kg: KnowledgeGraph):
        """Node exactly at stale_days boundary should be STALE."""
        now = time.time()
        empty_kg.add_node(
            Node(
                id="boundary",
                node_type=NodeType.DOCUMENT,
                name="Boundary",
                created_at=now - (30 * 86_400),
                updated_at=now - (30 * 86_400),
            )
        )
        report = freshness_report(empty_kg, stale_days=30)
        assert report.entries[0].status == FreshnessStatus.STALE


# ---------------------------------------------------------------------------
# Top-level imports
# ---------------------------------------------------------------------------


class TestImport:
    def test_freshness_report_importable(self):
        from talk_box.freshness import freshness_report

        assert freshness_report is not None

    def test_freshness_report_class_importable(self):
        from talk_box.freshness import FreshnessReport

        assert FreshnessReport is not None

    def test_freshness_entry_importable(self):
        from talk_box.freshness import FreshnessEntry

        assert FreshnessEntry is not None

    def test_freshness_status_importable(self):
        from talk_box.freshness import FreshnessStatus

        assert FreshnessStatus is not None
