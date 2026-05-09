"""Tests for talk_box.knowledge_filter."""

from __future__ import annotations

import pytest

from talk_box.enrichment import EnrichmentConfig
from talk_box.knowledge_filter import (
    FilterResult,
    KnowledgeFilter,
    filter_for_persona,
    retrieve_context,
)
from talk_box.knowledge_graph import Edge, KnowledgeGraph, Node, NodeType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def kg() -> KnowledgeGraph:
    """Build a small knowledge graph for testing."""
    g = KnowledgeGraph(":memory:")

    # Documents
    g.add_node(
        Node(
            id="doc-1",
            node_type=NodeType.DOCUMENT,
            name="ML Guide",
            content="Introduction to machine learning.",
            metadata={"tags": ["ml", "python"], "_topics": ["machine-learning"]},
        )
    )
    g.add_node(
        Node(
            id="doc-2",
            node_type=NodeType.DOCUMENT,
            name="Legal Brief",
            content="Contract terms and conditions.",
            metadata={"tags": ["legal"], "confidential": True},
        )
    )
    g.add_node(
        Node(
            id="doc-3",
            node_type=NodeType.DOCUMENT,
            name="Old Report",
            content="Quarterly results from last year.",
            metadata={"expires": "2024-01-01", "tags": ["finance"]},
        )
    )
    g.add_node(
        Node(
            id="doc-4",
            node_type=NodeType.DOCUMENT,
            name="Python Tutorial",
            content="Learn Python step by step.",
            metadata={"tags": ["python"], "_topics": ["python"]},
        )
    )
    g.add_node(
        Node(
            id="doc-5",
            node_type=NodeType.DOCUMENT,
            name="Untagged Doc",
            content="No tags or topics here.",
        )
    )

    # Topics
    g.add_node(Node(id="topic-ml", node_type=NodeType.TOPIC, name="machine-learning"))
    g.add_node(Node(id="topic-py", node_type=NodeType.TOPIC, name="python"))
    g.add_node(Node(id="topic-legal", node_type=NodeType.TOPIC, name="legal"))

    # Entities
    g.add_node(
        Node(
            id="entity-sklearn",
            node_type=NodeType.ENTITY,
            name="scikit-learn",
            metadata={"tags": ["ml"]},
        )
    )

    # Edges: doc -> topic
    g.add_edge(Edge(source="doc-1", target="topic-ml", relation="belongs_to"))
    g.add_edge(Edge(source="doc-1", target="topic-py", relation="belongs_to"))
    g.add_edge(Edge(source="doc-4", target="topic-py", relation="belongs_to"))
    g.add_edge(Edge(source="doc-2", target="topic-legal", relation="belongs_to"))

    # Edges: doc -> entity
    g.add_edge(Edge(source="doc-1", target="entity-sklearn", relation="mentions"))

    return g


# ---------------------------------------------------------------------------
# KnowledgeFilter – basic construction
# ---------------------------------------------------------------------------


class TestKnowledgeFilterConstruction:
    def test_defaults(self):
        f = KnowledgeFilter()
        assert f.include_topics == []
        assert f.exclude_topics == []
        assert f.include_tags == []
        assert f.exclude_tags == []
        assert f.include_node_types == []
        assert f.exclude_confidential is False
        assert f.exclude_expired is False
        assert f.max_results == 100
        assert f.enrichment_config is None

    def test_is_empty_default(self):
        assert KnowledgeFilter().is_empty is True

    def test_is_empty_with_rules(self):
        f = KnowledgeFilter(include_topics=["ml"])
        assert f.is_empty is False

    def test_rule_count_zero(self):
        assert KnowledgeFilter().rule_count == 0

    def test_rule_count_multiple(self):
        f = KnowledgeFilter(
            include_topics=["ml"],
            exclude_tags=["draft"],
            exclude_confidential=True,
        )
        assert f.rule_count == 3

    def test_repr_passthrough(self):
        assert "pass-through" in repr(KnowledgeFilter())

    def test_repr_with_rules(self):
        f = KnowledgeFilter(include_topics=["ml"], exclude_confidential=True)
        r = repr(f)
        assert "ml" in r
        assert "no_confidential" in r


# ---------------------------------------------------------------------------
# KnowledgeFilter.apply – topic filters
# ---------------------------------------------------------------------------


class TestApplyTopicFilters:
    def test_include_topic_via_edges(self, kg: KnowledgeGraph):
        f = KnowledgeFilter(include_topics=["machine-learning"])
        result = f.apply(kg)
        ids = result.node_ids
        assert "doc-1" in ids
        # doc-4 doesn't belong_to machine-learning (only python)
        assert "doc-4" not in ids
        # The topic node itself should match
        assert "topic-ml" in ids

    def test_include_topic_via_metadata(self, kg: KnowledgeGraph):
        """Nodes with _topics metadata should also match."""
        f = KnowledgeFilter(include_topics=["python"])
        result = f.apply(kg)
        ids = result.node_ids
        assert "doc-1" in ids  # has _topics: ["machine-learning"] but belongs_to python via edge
        assert "doc-4" in ids  # has _topics: ["python"]
        assert "topic-py" in ids  # topic node named "python"

    def test_exclude_topic(self, kg: KnowledgeGraph):
        f = KnowledgeFilter(exclude_topics=["legal"])
        result = f.apply(kg)
        ids = result.node_ids
        assert "doc-2" not in ids
        assert "topic-legal" not in ids
        assert "doc-1" in ids

    def test_include_and_exclude_topics(self, kg: KnowledgeGraph):
        """Exclude wins over include."""
        f = KnowledgeFilter(
            include_topics=["python", "machine-learning"],
            exclude_topics=["machine-learning"],
        )
        result = f.apply(kg)
        ids = result.node_ids
        # doc-1 is in both python and machine-learning; exclude wins
        assert "doc-1" not in ids
        # doc-4 is only python
        assert "doc-4" in ids


# ---------------------------------------------------------------------------
# KnowledgeFilter.apply – tag filters
# ---------------------------------------------------------------------------


class TestApplyTagFilters:
    def test_include_tags(self, kg: KnowledgeGraph):
        f = KnowledgeFilter(include_tags=["python"])
        result = f.apply(kg)
        ids = result.node_ids
        assert "doc-1" in ids  # tags: ["ml", "python"]
        assert "doc-4" in ids  # tags: ["python"]
        assert "doc-2" not in ids  # tags: ["legal"]

    def test_exclude_tags(self, kg: KnowledgeGraph):
        f = KnowledgeFilter(exclude_tags=["legal"])
        result = f.apply(kg)
        ids = result.node_ids
        assert "doc-2" not in ids
        assert "doc-1" in ids

    def test_tags_case_insensitive(self, kg: KnowledgeGraph):
        f = KnowledgeFilter(include_tags=["PYTHON"])
        result = f.apply(kg)
        assert "doc-1" in result.node_ids


# ---------------------------------------------------------------------------
# KnowledgeFilter.apply – node type filters
# ---------------------------------------------------------------------------


class TestApplyNodeTypeFilters:
    def test_include_documents_only(self, kg: KnowledgeGraph):
        f = KnowledgeFilter(include_node_types=[NodeType.DOCUMENT])
        result = f.apply(kg)
        for node in result.nodes:
            assert node.node_type == NodeType.DOCUMENT

    def test_include_topics_only(self, kg: KnowledgeGraph):
        f = KnowledgeFilter(include_node_types=[NodeType.TOPIC])
        result = f.apply(kg)
        assert result.total == 3
        for node in result.nodes:
            assert node.node_type == NodeType.TOPIC

    def test_include_multiple_types(self, kg: KnowledgeGraph):
        f = KnowledgeFilter(include_node_types=[NodeType.ENTITY, NodeType.TOPIC])
        result = f.apply(kg)
        types = {n.node_type for n in result.nodes}
        assert NodeType.DOCUMENT not in types


# ---------------------------------------------------------------------------
# KnowledgeFilter.apply – confidential filter
# ---------------------------------------------------------------------------


class TestApplyConfidentialFilter:
    def test_exclude_confidential(self, kg: KnowledgeGraph):
        f = KnowledgeFilter(exclude_confidential=True)
        result = f.apply(kg)
        assert "doc-2" not in result.node_ids

    def test_include_confidential_by_default(self, kg: KnowledgeGraph):
        f = KnowledgeFilter()
        result = f.apply(kg)
        assert "doc-2" in result.node_ids


# ---------------------------------------------------------------------------
# KnowledgeFilter.apply – expired filter
# ---------------------------------------------------------------------------


class TestApplyExpiredFilter:
    def test_exclude_expired(self, kg: KnowledgeGraph):
        f = KnowledgeFilter(exclude_expired=True)
        result = f.apply(kg)
        assert "doc-3" not in result.node_ids

    def test_include_expired_by_default(self, kg: KnowledgeGraph):
        f = KnowledgeFilter()
        result = f.apply(kg)
        assert "doc-3" in result.node_ids

    def test_future_expiry_not_excluded(self, kg: KnowledgeGraph):
        """A node with an expiry in the future should not be excluded."""
        kg.add_node(
            Node(
                id="doc-future",
                node_type=NodeType.DOCUMENT,
                name="Future Doc",
                content="Still valid.",
                metadata={"expires": "2099-12-31"},
            )
        )
        f = KnowledgeFilter(exclude_expired=True)
        result = f.apply(kg)
        assert "doc-future" in result.node_ids

    def test_invalid_date_not_excluded(self, kg: KnowledgeGraph):
        """An unparseable expires value should not be excluded."""
        kg.add_node(
            Node(
                id="doc-bad-date",
                node_type=NodeType.DOCUMENT,
                name="Bad Date",
                content="Has a bad date.",
                metadata={"expires": "not-a-date"},
            )
        )
        f = KnowledgeFilter(exclude_expired=True)
        result = f.apply(kg)
        assert "doc-bad-date" in result.node_ids


# ---------------------------------------------------------------------------
# KnowledgeFilter.apply – max_results
# ---------------------------------------------------------------------------


class TestApplyMaxResults:
    def test_max_results_limits_output(self, kg: KnowledgeGraph):
        f = KnowledgeFilter(max_results=2)
        result = f.apply(kg)
        assert result.total <= 2

    def test_max_results_default(self, kg: KnowledgeGraph):
        f = KnowledgeFilter()
        result = f.apply(kg)
        # Should get all 9 nodes (5 docs + 3 topics + 1 entity)
        assert result.total == 9


# ---------------------------------------------------------------------------
# KnowledgeFilter.apply – combined filters
# ---------------------------------------------------------------------------


class TestApplyCombinedFilters:
    def test_topic_and_confidential(self, kg: KnowledgeGraph):
        """Combine topic inclusion with confidential exclusion."""
        # Add a confidential ML doc
        kg.add_node(
            Node(
                id="doc-secret-ml",
                node_type=NodeType.DOCUMENT,
                name="Secret ML Research",
                content="Confidential ML findings.",
                metadata={
                    "confidential": True,
                    "_topics": ["machine-learning"],
                },
            )
        )
        f = KnowledgeFilter(
            include_topics=["machine-learning"],
            exclude_confidential=True,
        )
        result = f.apply(kg)
        assert "doc-secret-ml" not in result.node_ids
        assert "doc-1" in result.node_ids  # non-confidential ML doc

    def test_tag_and_type(self, kg: KnowledgeGraph):
        f = KnowledgeFilter(
            include_tags=["ml"],
            include_node_types=[NodeType.DOCUMENT],
        )
        result = f.apply(kg)
        # doc-1 has tags ["ml", "python"] and is a DOCUMENT
        assert "doc-1" in result.node_ids
        # entity-sklearn has tag "ml" but is an ENTITY
        assert "entity-sklearn" not in result.node_ids


# ---------------------------------------------------------------------------
# KnowledgeFilter.apply – FilterResult
# ---------------------------------------------------------------------------


class TestFilterResult:
    def test_total(self):
        r = FilterResult(
            nodes=[
                Node(id="a", node_type=NodeType.DOCUMENT, name="A"),
                Node(id="b", node_type=NodeType.DOCUMENT, name="B"),
            ],
            excluded_count=3,
        )
        assert r.total == 2
        assert r.excluded_count == 3

    def test_node_ids(self):
        r = FilterResult(
            nodes=[
                Node(id="x", node_type=NodeType.ENTITY, name="X"),
            ]
        )
        assert r.node_ids == ["x"]

    def test_node_names(self):
        r = FilterResult(
            nodes=[
                Node(id="x", node_type=NodeType.ENTITY, name="Alpha"),
            ]
        )
        assert r.node_names == ["Alpha"]

    def test_repr(self):
        r = FilterResult(excluded_count=5)
        assert "total=0" in repr(r)
        assert "excluded=5" in repr(r)

    def test_empty(self):
        r = FilterResult()
        assert r.total == 0
        assert r.node_ids == []
        assert r.node_names == []


# ---------------------------------------------------------------------------
# KnowledgeFilter.search
# ---------------------------------------------------------------------------


class TestFilterSearch:
    def test_search_with_topic_filter(self, kg: KnowledgeGraph):
        f = KnowledgeFilter(include_topics=["python"])
        result = f.search(kg, "Python")
        # doc-4 "Python Tutorial" matches search and topic
        assert "doc-4" in result.node_ids

    def test_search_excludes_confidential(self, kg: KnowledgeGraph):
        f = KnowledgeFilter(exclude_confidential=True)
        result = f.search(kg, "Contract")
        assert "doc-2" not in result.node_ids

    def test_search_respects_limit(self, kg: KnowledgeGraph):
        f = KnowledgeFilter()
        result = f.search(kg, "e", limit=2)
        assert result.total <= 2

    def test_search_no_match(self, kg: KnowledgeGraph):
        f = KnowledgeFilter()
        result = f.search(kg, "zzzznonexistent")
        assert result.total == 0


# ---------------------------------------------------------------------------
# filter_for_persona
# ---------------------------------------------------------------------------


class TestFilterForPersona:
    def test_topic_tags(self):
        """Tags with 'topic:' prefix become include_topics."""

        class FakePersona:
            tags = ["topic:ml", "topic:python", "!topic:legal"]
            name = "ml_eng"

        f = filter_for_persona(FakePersona())
        assert "ml" in f.include_topics
        assert "python" in f.include_topics
        assert "legal" in f.exclude_topics

    def test_plain_tags_become_include_tags(self):
        class FakePersona:
            tags = ["technical", "advanced"]
            name = "test"

        f = filter_for_persona(FakePersona())
        assert f.include_tags == ["technical", "advanced"]
        assert f.include_topics == []

    def test_mixed_tags(self):
        class FakePersona:
            tags = ["topic:ml", "technical", "!topic:legal"]
            name = "test"

        f = filter_for_persona(FakePersona())
        assert f.include_topics == ["ml"]
        assert f.exclude_topics == ["legal"]
        assert f.include_tags == ["technical"]

    def test_extra_topics(self):
        class FakePersona:
            tags = ["topic:ml"]
            name = "test"

        f = filter_for_persona(
            FakePersona(),
            extra_include_topics=["python"],
            extra_exclude_topics=["finance"],
        )
        assert "ml" in f.include_topics
        assert "python" in f.include_topics
        assert "finance" in f.exclude_topics

    def test_defaults(self):
        class FakePersona:
            tags = []
            name = "test"

        f = filter_for_persona(FakePersona())
        assert f.exclude_confidential is True
        assert f.exclude_expired is True

    def test_custom_options(self):
        class FakePersona:
            tags = []
            name = "test"

        config = EnrichmentConfig(add_related_docs=5)
        f = filter_for_persona(
            FakePersona(),
            exclude_confidential=False,
            exclude_expired=False,
            max_results=50,
            enrichment_config=config,
        )
        assert f.exclude_confidential is False
        assert f.exclude_expired is False
        assert f.max_results == 50
        assert f.enrichment_config is config

    def test_no_tags_attribute(self):
        """Object without tags should produce an empty filter."""

        class Minimal:
            pass

        f = filter_for_persona(Minimal())
        assert f.include_topics == []
        assert f.include_tags == []

    def test_none_tags(self):
        """Tags set to None should not raise."""

        class NullTags:
            tags = None
            name = "test"

        f = filter_for_persona(NullTags())
        assert f.include_topics == []

    def test_integration_with_kg(self, kg: KnowledgeGraph):
        """End-to-end: persona tags -> filter -> apply."""

        class MLPersona:
            tags = ["topic:python", "!topic:legal"]
            name = "ml_eng"

        f = filter_for_persona(MLPersona())
        result = f.apply(kg)
        assert "doc-4" in result.node_ids  # Python Tutorial
        assert "doc-2" not in result.node_ids  # Legal Brief (topic:legal)


# ---------------------------------------------------------------------------
# retrieve_context
# ---------------------------------------------------------------------------


class TestRetrieveContext:
    def test_without_filter(self, kg: KnowledgeGraph):
        nodes = retrieve_context(kg, "Python")
        names = [n.name for n in nodes]
        assert "Python Tutorial" in names

    def test_with_filter(self, kg: KnowledgeGraph):
        f = KnowledgeFilter(exclude_confidential=True)
        nodes = retrieve_context(kg, "terms", knowledge_filter=f)
        ids = [n.id for n in nodes]
        assert "doc-2" not in ids

    def test_limit(self, kg: KnowledgeGraph):
        nodes = retrieve_context(kg, "e", limit=1)
        assert len(nodes) <= 1

    def test_no_results(self, kg: KnowledgeGraph):
        nodes = retrieve_context(kg, "zzzznonexistent")
        assert nodes == []


# ---------------------------------------------------------------------------
# Context directive integration
# ---------------------------------------------------------------------------


class TestContextDirectiveIntegration:
    def test_contexts_metadata_used_as_topics(self, kg: KnowledgeGraph):
        """Nodes with directive 'contexts' in metadata should match topic filters."""
        kg.add_node(
            Node(
                id="doc-ctx",
                node_type=NodeType.DOCUMENT,
                name="Directive Doc",
                content="Has context directive.",
                metadata={"contexts": ["engineering"]},
            )
        )
        f = KnowledgeFilter(include_topics=["engineering"])
        result = f.apply(kg)
        assert "doc-ctx" in result.node_ids


# ---------------------------------------------------------------------------
# Top-level imports
# ---------------------------------------------------------------------------


class TestImport:
    def test_knowledge_filter_importable(self):
        from talk_box.knowledge_filter import KnowledgeFilter

        assert KnowledgeFilter is not None

    def test_filter_result_importable(self):
        from talk_box.knowledge_filter import FilterResult

        assert FilterResult is not None

    def test_filter_for_persona_importable(self):
        from talk_box.knowledge_filter import filter_for_persona

        assert filter_for_persona is not None

    def test_retrieve_context_importable(self):
        from talk_box.knowledge_filter import retrieve_context

        assert retrieve_context is not None
