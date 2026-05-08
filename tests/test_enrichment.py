"""Tests for talk_box.enrichment."""

from __future__ import annotations

import pytest

from talk_box.enrichment import (
    EnrichmentConfig,
    EnrichmentPipeline,
    EnrichmentResult,
    ExtractedEntity,
    ExtractedRelationship,
    PipelineResult,
    _entity_node_id,
    _first_sentence,
    _guess_entity_type,
    _topic_node_id,
    regex_enricher,
)
from talk_box.knowledge_graph import KnowledgeGraph, Node, NodeType


# ---------------------------------------------------------------------------
# ExtractedEntity
# ---------------------------------------------------------------------------


class TestExtractedEntity:
    def test_basic_creation(self):
        e = ExtractedEntity(name="Python")
        assert e.name == "Python"
        assert e.entity_type == "unknown"
        assert e.mentions == 1
        assert e.metadata == {}

    def test_with_all_fields(self):
        e = ExtractedEntity(
            name="Sarah Chen",
            entity_type="person",
            mentions=5,
            metadata={"role": "Tech Lead"},
        )
        assert e.entity_type == "person"
        assert e.mentions == 5
        assert e.metadata["role"] == "Tech Lead"


# ---------------------------------------------------------------------------
# ExtractedRelationship
# ---------------------------------------------------------------------------


class TestExtractedRelationship:
    def test_basic_creation(self):
        r = ExtractedRelationship(source="A", target="B")
        assert r.source == "A"
        assert r.target == "B"
        assert r.relation == "related_to"
        assert r.weight == 1.0

    def test_with_all_fields(self):
        r = ExtractedRelationship(
            source="Sarah",
            target="API Migration",
            relation="works_on",
            weight=0.8,
            metadata={"context": "team lead"},
        )
        assert r.relation == "works_on"
        assert r.weight == 0.8
        assert r.metadata["context"] == "team lead"


# ---------------------------------------------------------------------------
# EnrichmentResult
# ---------------------------------------------------------------------------


class TestEnrichmentResult:
    def test_empty_result(self):
        r = EnrichmentResult()
        assert r.entity_count == 0
        assert r.topic_count == 0
        assert r.relationship_count == 0
        assert r.summary == ""
        assert r.entity_names == []

    def test_with_data(self):
        r = EnrichmentResult(
            entities=[
                ExtractedEntity(name="Python"),
                ExtractedEntity(name="Rust"),
            ],
            topics=["programming", "open-source"],
            relationships=[
                ExtractedRelationship(source="Python", target="Rust"),
            ],
            summary="About languages.",
        )
        assert r.entity_count == 2
        assert r.topic_count == 2
        assert r.relationship_count == 1
        assert r.entity_names == ["Python", "Rust"]

    def test_repr(self):
        r = EnrichmentResult(
            entities=[ExtractedEntity(name="A")],
            topics=["t"],
        )
        s = repr(r)
        assert "entities=1" in s
        assert "topics=1" in s
        assert "relationships=0" in s


# ---------------------------------------------------------------------------
# PipelineResult
# ---------------------------------------------------------------------------


class TestPipelineResult:
    def test_creation(self):
        r = PipelineResult(enriched=3, skipped=2, entities_created=5)
        assert r.enriched == 3
        assert r.skipped == 2
        assert r.entities_created == 5
        assert r.total == 5

    def test_defaults(self):
        r = PipelineResult()
        assert r.enriched == 0
        assert r.total == 0

    def test_repr(self):
        r = PipelineResult(enriched=1, entities_created=2, topics_created=3)
        s = repr(r)
        assert "enriched=1" in s
        assert "entities=2" in s
        assert "topics=3" in s

    def test_frozen(self):
        r = PipelineResult(enriched=1)
        with pytest.raises(AttributeError):
            r.enriched = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# regex_enricher
# ---------------------------------------------------------------------------


class TestRegexEnricher:
    def test_extracts_proper_nouns(self):
        result = regex_enricher("Notes", "Talked to Sarah Chen about the project.")
        names = result.entity_names
        assert "Sarah Chen" in names

    def test_extracts_multi_word_entities(self):
        result = regex_enricher("Notes", "Met with United Nations representatives.")
        names = result.entity_names
        assert "United Nations" in names

    def test_extracts_hashtags_as_topics(self):
        result = regex_enricher("Notes", "Working on #machine-learning and #data-science.")
        assert "machine-learning" in result.topics
        assert "data-science" in result.topics

    def test_no_duplicate_entities(self):
        result = regex_enricher("Notes", "Sarah Chen met Sarah Chen again.")
        count = result.entity_names.count("Sarah Chen")
        assert count == 1

    def test_counts_mentions(self):
        result = regex_enricher("Notes", "Sarah Chen is great. Sarah Chen agrees.")
        sarah = [e for e in result.entities if e.name == "Sarah Chen"][0]
        assert sarah.mentions >= 2

    def test_no_duplicate_topics(self):
        result = regex_enricher("Notes", "#python #python #python")
        assert result.topics.count("python") == 1

    def test_empty_content(self):
        result = regex_enricher("", "")
        assert result.entity_count == 0
        assert result.topic_count == 0

    def test_summary_from_first_sentence(self):
        result = regex_enricher("Title", "This is the first sentence. And the second.")
        assert result.summary == "This is the first sentence."

    def test_summary_no_sentence_boundary(self):
        result = regex_enricher("Title", "Short text")
        assert result.summary == "Short text"

    def test_guesses_person_type(self):
        result = regex_enricher("Notes", "Met with John Smith yesterday.")
        john = [e for e in result.entities if e.name == "John Smith"][0]
        assert john.entity_type == "person"


# ---------------------------------------------------------------------------
# _guess_entity_type
# ---------------------------------------------------------------------------


class TestGuessEntityType:
    def test_person_two_words(self):
        assert _guess_entity_type("John Smith") == "person"

    def test_person_three_words(self):
        assert _guess_entity_type("Mary Jane Watson") == "person"

    def test_non_person(self):
        assert _guess_entity_type("API") == "entity"

    def test_mixed_case_not_person(self):
        assert _guess_entity_type("McDonalds Corp") == "entity"


# ---------------------------------------------------------------------------
# _first_sentence
# ---------------------------------------------------------------------------


class TestFirstSentence:
    def test_simple(self):
        assert _first_sentence("Hello world. More text.") == "Hello world."

    def test_exclamation(self):
        assert _first_sentence("Wow! Amazing.") == "Wow!"

    def test_question(self):
        assert _first_sentence("What? Really.") == "What?"

    def test_no_punctuation(self):
        assert _first_sentence("No ending punctuation") == "No ending punctuation"

    def test_empty(self):
        assert _first_sentence("") == ""

    def test_long_text_truncated(self):
        long_text = "A " * 200
        result = _first_sentence(long_text)
        assert len(result) <= 204  # 200 + "..."


# ---------------------------------------------------------------------------
# _entity_node_id / _topic_node_id
# ---------------------------------------------------------------------------


class TestNodeIdHelpers:
    def test_entity_id_deterministic(self):
        id1 = _entity_node_id("ent", "Python")
        id2 = _entity_node_id("ent", "Python")
        assert id1 == id2

    def test_entity_id_uses_prefix(self):
        nid = _entity_node_id("myprefix", "Test")
        assert nid.startswith("myprefix-")

    def test_entity_id_case_insensitive(self):
        id1 = _entity_node_id("ent", "Python")
        id2 = _entity_node_id("ent", "python")
        assert id1 == id2

    def test_topic_id_deterministic(self):
        id1 = _topic_node_id("topic", "ml")
        id2 = _topic_node_id("topic", "ml")
        assert id1 == id2

    def test_different_names_different_ids(self):
        assert _entity_node_id("e", "A") != _entity_node_id("e", "B")

    def test_topic_id_uses_prefix(self):
        nid = _topic_node_id("tp", "science")
        assert nid.startswith("tp-")


# ---------------------------------------------------------------------------
# EnrichmentConfig
# ---------------------------------------------------------------------------


class TestEnrichmentConfig:
    def test_defaults(self):
        c = EnrichmentConfig()
        assert c.add_related_docs == 3
        assert c.add_entity_context is True
        assert c.add_temporal_context is False

    def test_custom(self):
        c = EnrichmentConfig(add_related_docs=5, add_temporal_context=True)
        assert c.add_related_docs == 5
        assert c.add_temporal_context is True

    def test_repr(self):
        c = EnrichmentConfig()
        assert "related_docs=3" in repr(c)


# ---------------------------------------------------------------------------
# EnrichmentPipeline
# ---------------------------------------------------------------------------


def _simple_enricher(title: str, content: str) -> EnrichmentResult:
    """Test enricher that returns predictable results."""
    return EnrichmentResult(
        entities=[
            ExtractedEntity(name="Alpha", entity_type="project", mentions=2),
            ExtractedEntity(name="Bob Smith", entity_type="person", mentions=1),
        ],
        topics=["engineering", "testing"],
        relationships=[
            ExtractedRelationship(source="Bob Smith", target="Alpha", relation="works_on"),
        ],
        summary=f"Summary of {title}.",
    )


def _empty_enricher(title: str, content: str) -> EnrichmentResult:
    """Enricher that returns empty results."""
    return EnrichmentResult()


class TestEnrichmentPipeline:
    def _add_doc(self, kg: KnowledgeGraph, doc_id: str, name: str, content: str) -> None:
        """Helper to add a document node."""
        kg.add_node(
            Node(
                id=doc_id,
                node_type=NodeType.DOCUMENT,
                name=name,
                content=content,
            )
        )

    def test_enriches_documents(self):
        kg = KnowledgeGraph(":memory:")
        self._add_doc(kg, "doc-1", "README", "About the project.")

        pipeline = EnrichmentPipeline(enrich_fn=_simple_enricher)
        result = pipeline.run(kg)

        assert result.enriched == 1
        assert result.skipped == 0
        assert result.entities_created == 2
        assert result.topics_created == 2

    def test_creates_entity_nodes(self):
        kg = KnowledgeGraph(":memory:")
        self._add_doc(kg, "doc-1", "Notes", "Content here.")

        pipeline = EnrichmentPipeline(enrich_fn=_simple_enricher)
        pipeline.run(kg)

        entities = kg.list_nodes(node_type=NodeType.ENTITY)
        names = {e.name for e in entities}
        assert "Alpha" in names
        assert "Bob Smith" in names

    def test_creates_topic_nodes(self):
        kg = KnowledgeGraph(":memory:")
        self._add_doc(kg, "doc-1", "Notes", "Content here.")

        pipeline = EnrichmentPipeline(enrich_fn=_simple_enricher)
        pipeline.run(kg)

        topics = kg.list_nodes(node_type=NodeType.TOPIC)
        names = {t.name for t in topics}
        assert "engineering" in names
        assert "testing" in names

    def test_creates_mention_edges(self):
        kg = KnowledgeGraph(":memory:")
        self._add_doc(kg, "doc-1", "Notes", "Content.")

        pipeline = EnrichmentPipeline(enrich_fn=_simple_enricher)
        pipeline.run(kg)

        edges = kg.get_edges("doc-1", direction="outgoing", relation="mentions")
        assert len(edges) == 2

    def test_creates_topic_edges(self):
        kg = KnowledgeGraph(":memory:")
        self._add_doc(kg, "doc-1", "Notes", "Content.")

        pipeline = EnrichmentPipeline(enrich_fn=_simple_enricher)
        pipeline.run(kg)

        edges = kg.get_edges("doc-1", direction="outgoing", relation="belongs_to")
        assert len(edges) == 2

    def test_creates_relationship_edges(self):
        kg = KnowledgeGraph(":memory:")
        self._add_doc(kg, "doc-1", "Notes", "Content.")

        pipeline = EnrichmentPipeline(enrich_fn=_simple_enricher)
        pipeline.run(kg)

        # Bob Smith -> Alpha with "works_on"
        bob_id = _entity_node_id("entity", "Bob Smith")
        edges = kg.get_edges(bob_id, direction="outgoing", relation="works_on")
        assert len(edges) == 1

    def test_stores_summary_in_metadata(self):
        kg = KnowledgeGraph(":memory:")
        self._add_doc(kg, "doc-1", "README", "Content.")

        pipeline = EnrichmentPipeline(enrich_fn=_simple_enricher)
        pipeline.run(kg)

        doc = kg.get_node("doc-1")
        assert doc is not None
        assert doc.metadata["_summary"] == "Summary of README."

    def test_stores_topics_in_metadata(self):
        kg = KnowledgeGraph(":memory:")
        self._add_doc(kg, "doc-1", "Notes", "Content.")

        pipeline = EnrichmentPipeline(enrich_fn=_simple_enricher)
        pipeline.run(kg)

        doc = kg.get_node("doc-1")
        assert doc is not None
        assert doc.metadata["_topics"] == ["engineering", "testing"]

    def test_marks_enriched(self):
        kg = KnowledgeGraph(":memory:")
        self._add_doc(kg, "doc-1", "Notes", "Content.")

        pipeline = EnrichmentPipeline(enrich_fn=_simple_enricher)
        pipeline.run(kg)

        doc = kg.get_node("doc-1")
        assert doc is not None
        assert doc.metadata["_enriched"] is True
        assert "_enriched_at" in doc.metadata

    def test_skips_already_enriched(self):
        kg = KnowledgeGraph(":memory:")
        kg.add_node(
            Node(
                id="doc-1",
                node_type=NodeType.DOCUMENT,
                name="Notes",
                content="Content.",
                metadata={"_enriched": True, "_content_hash": "abc", "_enriched_hash": "abc"},
            )
        )

        call_count = 0

        def counting_enricher(title: str, content: str) -> EnrichmentResult:
            nonlocal call_count
            call_count += 1
            return EnrichmentResult()

        pipeline = EnrichmentPipeline(enrich_fn=counting_enricher)
        result = pipeline.run(kg)

        assert result.skipped == 1
        assert result.enriched == 0
        assert call_count == 0

    def test_re_enriches_changed_content(self):
        kg = KnowledgeGraph(":memory:")
        kg.add_node(
            Node(
                id="doc-1",
                node_type=NodeType.DOCUMENT,
                name="Notes",
                content="Updated content.",
                metadata={"_enriched": True, "_content_hash": "new", "_enriched_hash": "old"},
            )
        )

        pipeline = EnrichmentPipeline(enrich_fn=_empty_enricher)
        result = pipeline.run(kg)

        assert result.enriched == 1
        assert result.skipped == 0

    def test_force_re_enriches(self):
        kg = KnowledgeGraph(":memory:")
        kg.add_node(
            Node(
                id="doc-1",
                node_type=NodeType.DOCUMENT,
                name="Notes",
                content="Content.",
                metadata={"_enriched": True, "_content_hash": "x", "_enriched_hash": "x"},
            )
        )

        pipeline = EnrichmentPipeline(enrich_fn=_empty_enricher)
        result = pipeline.run(kg, force=True)

        assert result.enriched == 1
        assert result.skipped == 0

    def test_multiple_documents(self):
        kg = KnowledgeGraph(":memory:")
        self._add_doc(kg, "doc-1", "Doc 1", "First document.")
        self._add_doc(kg, "doc-2", "Doc 2", "Second document.")

        pipeline = EnrichmentPipeline(enrich_fn=_simple_enricher)
        result = pipeline.run(kg)

        assert result.enriched == 2
        # Entities are shared, so second doc doesn't recreate them
        assert result.entities_created == 2  # Created on first doc

    def test_limit(self):
        kg = KnowledgeGraph(":memory:")
        for i in range(5):
            self._add_doc(kg, f"doc-{i}", f"Doc {i}", f"Content {i}")

        pipeline = EnrichmentPipeline(enrich_fn=_empty_enricher)
        result = pipeline.run(kg, limit=3)

        assert result.enriched == 3

    def test_empty_enricher(self):
        kg = KnowledgeGraph(":memory:")
        self._add_doc(kg, "doc-1", "Notes", "Content.")

        pipeline = EnrichmentPipeline(enrich_fn=_empty_enricher)
        result = pipeline.run(kg)

        assert result.enriched == 1
        assert result.entities_created == 0
        assert result.topics_created == 0

    def test_custom_prefixes(self):
        kg = KnowledgeGraph(":memory:")
        self._add_doc(kg, "doc-1", "Notes", "Content.")

        pipeline = EnrichmentPipeline(
            enrich_fn=_simple_enricher,
            entity_prefix="ent",
            topic_prefix="top",
        )
        pipeline.run(kg)

        entities = kg.list_nodes(node_type=NodeType.ENTITY)
        topics = kg.list_nodes(node_type=NodeType.TOPIC)

        for e in entities:
            assert e.id.startswith("ent-")
        for t in topics:
            assert t.id.startswith("top-")

    def test_entity_metadata_preserved(self):
        kg = KnowledgeGraph(":memory:")
        self._add_doc(kg, "doc-1", "Notes", "Content.")

        pipeline = EnrichmentPipeline(enrich_fn=_simple_enricher)
        pipeline.run(kg)

        alpha_id = _entity_node_id("entity", "Alpha")
        alpha = kg.get_node(alpha_id)
        assert alpha is not None
        assert alpha.metadata["entity_type"] == "project"

    def test_mention_edge_weight(self):
        kg = KnowledgeGraph(":memory:")
        self._add_doc(kg, "doc-1", "Notes", "Content.")

        pipeline = EnrichmentPipeline(enrich_fn=_simple_enricher)
        pipeline.run(kg)

        # Alpha has 2 mentions → weight = 2/10 = 0.2
        alpha_id = _entity_node_id("entity", "Alpha")
        edges = kg.get_edges("doc-1", direction="outgoing", relation="mentions")
        alpha_edge = [e for e in edges if e.target == alpha_id][0]
        assert alpha_edge.weight == pytest.approx(0.2)

    def test_no_empty_graph(self):
        kg = KnowledgeGraph(":memory:")

        pipeline = EnrichmentPipeline(enrich_fn=_simple_enricher)
        result = pipeline.run(kg)

        assert result.enriched == 0
        assert result.total == 0


# ---------------------------------------------------------------------------
# Integration: regex_enricher + pipeline
# ---------------------------------------------------------------------------


class TestRegexEnricherIntegration:
    def test_enricher_with_pipeline(self):
        kg = KnowledgeGraph(":memory:")
        kg.add_node(
            Node(
                id="doc-1",
                node_type=NodeType.DOCUMENT,
                name="Meeting Notes",
                content="Talked to Sarah Chen about Python. Working on #machine-learning project.",
            )
        )

        pipeline = EnrichmentPipeline(enrich_fn=regex_enricher)
        result = pipeline.run(kg)

        assert result.enriched == 1
        assert result.entities_created >= 1  # At least Sarah Chen

        # Check entity was created
        entities = kg.list_nodes(node_type=NodeType.ENTITY)
        entity_names = {e.name for e in entities}
        assert "Sarah Chen" in entity_names

        # Check topic was created
        topics = kg.list_nodes(node_type=NodeType.TOPIC)
        topic_names = {t.name for t in topics}
        assert "machine-learning" in topic_names

        # Check summary was stored
        doc = kg.get_node("doc-1")
        assert doc is not None
        assert doc.metadata.get("_summary")


# ---------------------------------------------------------------------------
# Top-level imports
# ---------------------------------------------------------------------------


class TestTopLevelImport:
    def test_import_from_package(self):
        from talk_box import (
            EnrichmentConfig,
            EnrichmentPipeline,
            EnrichmentResult,
            ExtractedEntity,
            ExtractedRelationship,
            PipelineResult,
            regex_enricher,
        )

        assert EnrichmentConfig is not None
        assert EnrichmentPipeline is not None
        assert EnrichmentResult is not None
        assert ExtractedEntity is not None
        assert ExtractedRelationship is not None
        assert PipelineResult is not None
        assert regex_enricher is not None

    def test_all_contains_exports(self):
        import talk_box

        for name in [
            "EnrichmentConfig",
            "EnrichmentPipeline",
            "EnrichmentResult",
            "ExtractedEntity",
            "ExtractedRelationship",
            "PipelineResult",
            "regex_enricher",
        ]:
            assert name in talk_box.__all__, f"{name} missing from __all__"
