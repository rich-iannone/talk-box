"""Backend enrichment: extract entities, topics, relationships, and summaries from documents."""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Enrichment results
# ---------------------------------------------------------------------------


@dataclass
class ExtractedEntity:
    """An entity extracted from a document.

    Parameters
    ----------
    name
        The entity name (e.g. ``"Sarah Chen"``).
    entity_type
        Category such as ``"person"``, ``"organization"``, ``"technology"``,
        ``"project"``, ``"date"``.
    mentions
        Number of times the entity appears in the source document.
    metadata
        Extra information about the entity (role, description, etc.).

    Examples
    --------
    ```python
    import talk_box as tb

    entity = tb.ExtractedEntity(
        name="Sarah Chen",
        entity_type="person",
        mentions=3,
        metadata={"role": "Tech Lead"},
    )
    entity.name  # "Sarah Chen"
    ```
    """

    name: str
    entity_type: str = "unknown"
    mentions: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedRelationship:
    """A relationship extracted between two entities.

    Parameters
    ----------
    source
        Name of the source entity.
    target
        Name of the target entity.
    relation
        The relationship label (e.g. ``"works_on"``, ``"mentioned_with"``).
    weight
        Confidence or strength of the relationship (0.0–1.0).
    metadata
        Extra context about the relationship.

    Examples
    --------
    ```python
    import talk_box as tb

    rel = tb.ExtractedRelationship(
        source="Sarah Chen",
        target="API Migration",
        relation="concerned_about",
    )
    ```
    """

    source: str
    target: str
    relation: str = "related_to"
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnrichmentResult:
    """Complete enrichment output for a single document.

    Combines all extracted information: entities, topics, relationships,
    and an optional summary.

    Parameters
    ----------
    entities
        Entities found in the document.
    topics
        Topic labels assigned to the document.
    relationships
        Relationships between entities.
    summary
        A short summary of the document content.
    metadata
        Additional enrichment metadata (model used, timing, etc.).

    Examples
    --------
    ```python
    result = tb.EnrichmentResult(
        entities=[tb.ExtractedEntity(name="Python", entity_type="technology")],
        topics=["programming", "open-source"],
        summary="An overview of Python's key features.",
    )
    result.entity_names  # ["Python"]
    ```
    """

    entities: list[ExtractedEntity] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    relationships: list[ExtractedRelationship] = field(default_factory=list)
    summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def entity_names(self) -> list[str]:
        """Names of all extracted entities."""
        return [e.name for e in self.entities]

    @property
    def entity_count(self) -> int:
        """Number of extracted entities."""
        return len(self.entities)

    @property
    def topic_count(self) -> int:
        """Number of assigned topics."""
        return len(self.topics)

    @property
    def relationship_count(self) -> int:
        """Number of extracted relationships."""
        return len(self.relationships)

    def __repr__(self) -> str:
        return (
            f"EnrichmentResult(entities={self.entity_count}, "
            f"topics={self.topic_count}, "
            f"relationships={self.relationship_count})"
        )


# ---------------------------------------------------------------------------
# Enrichment pipeline
# ---------------------------------------------------------------------------

# Type alias for enrichment functions
EnrichmentFn = Callable[[str, str], EnrichmentResult]
"""A function that takes (title, content) and returns an EnrichmentResult."""


@dataclass
class EnrichmentPipeline:
    """Configurable pipeline for enriching documents in a knowledge graph.

    The pipeline applies an enrichment function to document nodes and
    creates entity nodes, topic nodes, and relationship edges in the
    graph.  Enrichment is incremental: documents are only re-enriched
    when their content changes.

    Parameters
    ----------
    enrich_fn
        A callable ``(title, content) -> EnrichmentResult`` that performs
        the actual extraction.  This is where LLM calls happen.
    entity_prefix
        Prefix for generated entity node IDs.
    topic_prefix
        Prefix for generated topic node IDs.

    Examples
    --------
    ```python
    import talk_box as tb

    def my_enricher(title: str, content: str) -> tb.EnrichmentResult:
        # Call your LLM here
        return tb.EnrichmentResult(
            entities=[tb.ExtractedEntity(name="Python", entity_type="technology")],
            topics=["programming"],
            summary="A document about Python.",
        )

    pipeline = tb.EnrichmentPipeline(enrich_fn=my_enricher)
    kg = tb.KnowledgeGraph(":memory:")
    # ... add document nodes via sync() ...
    result = pipeline.run(kg)
    result.enriched  # number of documents enriched
    ```
    """

    enrich_fn: EnrichmentFn
    entity_prefix: str = "entity"
    topic_prefix: str = "topic"

    def run(
        self,
        kg: Any,
        *,
        limit: int = 100,
        force: bool = False,
    ) -> PipelineResult:
        """Run enrichment on document nodes in the knowledge graph.

        Parameters
        ----------
        kg
            A :class:`~talk_box.knowledge_graph.KnowledgeGraph` instance.
        limit
            Maximum number of documents to enrich per run.
        force
            If ``True``, re-enrich all documents regardless of whether
            they've been enriched before.

        Returns
        -------
        PipelineResult
            Summary of enrichment activity.
        """
        from talk_box.knowledge_graph import Edge, GraphLayer, Node, NodeType

        docs = kg.list_nodes(node_type=NodeType.DOCUMENT, limit=limit)
        enriched = 0
        skipped = 0
        entities_created = 0
        topics_created = 0
        edges_created = 0

        for doc_node in docs:
            # Skip if already enriched (unless forced)
            if not force and doc_node.metadata.get("_enriched"):
                content_hash = doc_node.metadata.get("_content_hash", "")
                enriched_hash = doc_node.metadata.get("_enriched_hash", "")
                if content_hash == enriched_hash:
                    skipped += 1
                    continue

            # Run the enrichment function
            result = self.enrich_fn(doc_node.name, doc_node.content)

            # Store summary and enrichment metadata on the document node
            meta = dict(doc_node.metadata)
            meta["_enriched"] = True
            meta["_enriched_at"] = time.time()
            # Compute content hash if not already set (e.g., manually-added nodes)
            if not meta.get("_content_hash"):
                import hashlib

                meta["_content_hash"] = hashlib.sha256(doc_node.content.encode()).hexdigest()[:16]
            meta["_enriched_hash"] = meta["_content_hash"]
            if result.summary:
                meta["_summary"] = result.summary
            if result.topics:
                meta["_topics"] = result.topics

            updated_doc = Node(
                id=doc_node.id,
                node_type=NodeType.DOCUMENT,
                name=doc_node.name,
                content=doc_node.content,
                metadata=meta,
                embedding=doc_node.embedding,
                created_at=doc_node.created_at,
            )
            kg.add_node(updated_doc)

            # Create entity nodes and edges
            for entity in result.entities:
                entity_id = _entity_node_id(self.entity_prefix, entity.name)
                existing = kg.get_node(entity_id)

                if existing is None:
                    entity_node = Node(
                        id=entity_id,
                        node_type=NodeType.ENTITY,
                        name=entity.name,
                        metadata={
                            "entity_type": entity.entity_type,
                            **entity.metadata,
                        },
                        layer=GraphLayer.ENRICHMENT,
                    )
                    kg.add_node(entity_node)
                    entities_created += 1

                # Edge from document to entity
                edge = Edge(
                    source=doc_node.id,
                    target=entity_id,
                    relation="mentions",
                    weight=min(entity.mentions / 10.0, 1.0),
                    metadata={"entity_type": entity.entity_type},
                    layer=GraphLayer.ENRICHMENT,
                )
                kg.add_edge(edge)
                edges_created += 1

            # Create topic nodes and edges
            for topic_name in result.topics:
                topic_id = _topic_node_id(self.topic_prefix, topic_name)
                existing = kg.get_node(topic_id)

                if existing is None:
                    topic_node = Node(
                        id=topic_id,
                        node_type=NodeType.TOPIC,
                        name=topic_name,
                        layer=GraphLayer.ENRICHMENT,
                    )
                    kg.add_node(topic_node)
                    topics_created += 1

                # Edge from document to topic
                edge = Edge(
                    source=doc_node.id,
                    target=topic_id,
                    relation="belongs_to",
                    layer=GraphLayer.ENRICHMENT,
                )
                kg.add_edge(edge)
                edges_created += 1

            # Create relationship edges between entities
            for rel in result.relationships:
                src_id = _entity_node_id(self.entity_prefix, rel.source)
                tgt_id = _entity_node_id(self.entity_prefix, rel.target)

                # Only create edge if both entities exist
                if kg.get_node(src_id) is not None and kg.get_node(tgt_id) is not None:
                    edge = Edge(
                        source=src_id,
                        target=tgt_id,
                        relation=rel.relation,
                        weight=rel.weight,
                        metadata=rel.metadata,
                        layer=GraphLayer.ENRICHMENT,
                    )
                    kg.add_edge(edge)
                    edges_created += 1

            enriched += 1

        return PipelineResult(
            enriched=enriched,
            skipped=skipped,
            entities_created=entities_created,
            topics_created=topics_created,
            edges_created=edges_created,
        )


# ---------------------------------------------------------------------------
# PipelineResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineResult:
    """Result of running the enrichment pipeline.

    Parameters
    ----------
    enriched
        Number of documents that were enriched.
    skipped
        Number of documents skipped (already enriched, unchanged).
    entities_created
        Number of new entity nodes created.
    topics_created
        Number of new topic nodes created.
    edges_created
        Number of edges created (mentions, belongs_to, relationships).

    Examples
    --------
    ```python
    result = pipeline.run(kg)
    result.enriched         # 5
    result.entities_created # 12
    result.topics_created   # 8
    ```
    """

    enriched: int = 0
    skipped: int = 0
    entities_created: int = 0
    topics_created: int = 0
    edges_created: int = 0

    @property
    def total(self) -> int:
        """Total documents processed (enriched + skipped)."""
        return self.enriched + self.skipped

    def __repr__(self) -> str:
        return (
            f"PipelineResult(enriched={self.enriched}, skipped={self.skipped}, "
            f"entities={self.entities_created}, topics={self.topics_created}, "
            f"edges={self.edges_created})"
        )


# ---------------------------------------------------------------------------
# Built-in enrichment functions
# ---------------------------------------------------------------------------


def regex_enricher(title: str, content: str) -> EnrichmentResult:
    """Simple regex-based enrichment (no LLM required).

    Extracts entities using common patterns (capitalized phrases,
    hashtags, @-mentions) and assigns basic topics from keywords.
    Useful as a fallback or for testing.

    Parameters
    ----------
    title
        Document title.
    content
        Document content.

    Returns
    -------
    EnrichmentResult
        Extracted entities and topics.

    Examples
    --------
    ```python
    import talk_box as tb

    result = tb.regex_enricher("Meeting Notes", "Talked to Sarah Chen about Python.")
    result.entity_names  # ["Sarah Chen"]
    result.topics        # []
    ```
    """
    text = f"{title}\n{content}"
    entities: list[ExtractedEntity] = []
    seen_names: set[str] = set()

    # Extract capitalized multi-word phrases (likely proper nouns)
    for match in re.finditer(r"\b([A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+)+)\b", text):
        name = match.group(1)
        if name not in seen_names and len(name) > 3:
            seen_names.add(name)
            # Count mentions
            mentions = len(re.findall(re.escape(name), text))
            entities.append(
                ExtractedEntity(
                    name=name,
                    entity_type=_guess_entity_type(name),
                    mentions=mentions,
                )
            )

    # Extract hashtags as topics
    topics: list[str] = []
    seen_topics: set[str] = set()
    for match in re.finditer(r"#([a-zA-Z][a-zA-Z0-9_-]+)", text):
        topic = match.group(1).lower()
        if topic not in seen_topics:
            seen_topics.add(topic)
            topics.append(topic)

    # Build a simple summary from the first sentence
    summary = _first_sentence(content)

    return EnrichmentResult(
        entities=entities,
        topics=topics,
        summary=summary,
    )


def _guess_entity_type(name: str) -> str:
    """Heuristic guess at entity type from a proper noun."""
    words = name.split()
    # Two or three words that look like a person name
    if 2 <= len(words) <= 3 and all(w[0].isupper() and w[1:].islower() for w in words):
        return "person"
    return "entity"


def _first_sentence(text: str) -> str:
    """Extract the first sentence from text as a summary."""
    text = text.strip()
    if not text:
        return ""
    # Find first sentence-ending punctuation
    match = re.search(r"[.!?]\s", text)
    if match:
        return text[: match.start() + 1].strip()
    # No sentence boundary found — take first 200 chars
    if len(text) > 200:
        return text[:200].rsplit(" ", 1)[0] + "..."
    return text


# ---------------------------------------------------------------------------
# Enrichment config
# ---------------------------------------------------------------------------


@dataclass
class EnrichmentConfig:
    """Configuration for how enrichment results are applied.

    Controls how many related documents to include in context, and
    which enrichment features to enable.

    Parameters
    ----------
    add_related_docs
        Number of most-related documents to include as context
        when building prompts.
    add_entity_context
        Whether to include entity descriptions in prompt context.
    add_temporal_context
        Whether to include temporal annotations in context.

    Examples
    --------
    ```python
    import talk_box as tb

    config = tb.EnrichmentConfig(
        add_related_docs=3,
        add_entity_context=True,
    )
    ```
    """

    add_related_docs: int = 3
    add_entity_context: bool = True
    add_temporal_context: bool = False

    def __repr__(self) -> str:
        return (
            f"EnrichmentConfig(related_docs={self.add_related_docs}, "
            f"entity_context={self.add_entity_context}, "
            f"temporal_context={self.add_temporal_context})"
        )


# ---------------------------------------------------------------------------
# Node ID helpers
# ---------------------------------------------------------------------------


def _entity_node_id(prefix: str, name: str) -> str:
    """Generate a deterministic node ID for an entity."""
    normalized = name.strip().lower()
    digest = hashlib.sha256(normalized.encode()).hexdigest()[:12]
    return f"{prefix}-{digest}"


def _topic_node_id(prefix: str, name: str) -> str:
    """Generate a deterministic node ID for a topic."""
    normalized = name.strip().lower()
    digest = hashlib.sha256(normalized.encode()).hexdigest()[:12]
    return f"{prefix}-{digest}"
