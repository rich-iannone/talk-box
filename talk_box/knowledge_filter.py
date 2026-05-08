"""Per-persona knowledge filtering for the knowledge graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from talk_box.enrichment import EnrichmentConfig
from talk_box.knowledge_graph import KnowledgeGraph, Node, NodeType

# ---------------------------------------------------------------------------
# Filter specification
# ---------------------------------------------------------------------------


@dataclass
class KnowledgeFilter:
    """Filter knowledge graph content by topic, tag, type, and directives.

    A ``KnowledgeFilter`` describes which nodes a persona (or other
    consumer) is allowed to see.  Filters are additive: every non-empty
    include list is an allow-list, every non-empty exclude list is a
    deny-list.  Exclude rules win over include rules when both match.

    Parameters
    ----------
    include_topics
        Only include nodes that belong to (or mention) these topics.
        Empty means "no topic restriction".
    exclude_topics
        Exclude nodes belonging to these topics, even if they match
        an include rule.
    include_tags
        Only include nodes whose ``metadata["tags"]`` overlap with
        these values.  Empty means "no tag restriction".
    exclude_tags
        Exclude nodes whose tags overlap with these values.
    include_node_types
        Restrict to these node types.  Empty means "all types".
    exclude_confidential
        If ``True``, skip nodes whose metadata marks them as
        confidential (``metadata["confidential"] == True``).
    exclude_expired
        If ``True``, skip nodes whose ``@expires`` date has passed
        (``metadata["expires"]`` compared against *now*).
    max_results
        Maximum number of nodes to return from :meth:`apply`.
    enrichment_config
        Optional enrichment config governing how many related docs
        and entities to surface in prompt context.

    Examples
    --------
    Create a filter that only allows "engineering" topics:

    ```python
    import talk_box as tb

    f = tb.KnowledgeFilter(include_topics=["engineering"])
    result = f.apply(kg)
    result.total  # number of nodes returned
    ```

    Exclude confidential content:

    ```python
    f = tb.KnowledgeFilter(exclude_confidential=True)
    result = f.apply(kg)
    ```
    """

    include_topics: list[str] = field(default_factory=list)
    exclude_topics: list[str] = field(default_factory=list)
    include_tags: list[str] = field(default_factory=list)
    exclude_tags: list[str] = field(default_factory=list)
    include_node_types: list[NodeType] = field(default_factory=list)
    exclude_confidential: bool = False
    exclude_expired: bool = False
    max_results: int = 100
    enrichment_config: EnrichmentConfig | None = None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def apply(self, kg: KnowledgeGraph) -> FilterResult:
        """Apply this filter to every node in *kg*.

        Parameters
        ----------
        kg
            The knowledge graph to filter.

        Returns
        -------
        FilterResult
            Matching nodes and exclusion counts.

        Examples
        --------
        ```python
        result = f.apply(kg)
        for node in result.nodes:
            print(node.name)
        ```
        """
        # Fetch all nodes (generous limit — filters trim them down)
        all_nodes = kg.list_nodes(limit=10_000)

        accepted: list[Node] = []
        excluded = 0

        # Pre-compute topic node IDs for include/exclude topic checks
        topic_index = _build_topic_index(kg) if self.include_topics or self.exclude_topics else {}

        for node in all_nodes:
            if not self._accepts(node, topic_index, kg):
                excluded += 1
                continue
            accepted.append(node)
            if len(accepted) >= self.max_results:
                break

        return FilterResult(
            nodes=accepted,
            excluded_count=excluded,
            filter_summary=self._summary(),
        )

    def search(
        self,
        kg: KnowledgeGraph,
        query: str,
        *,
        limit: int | None = None,
    ) -> FilterResult:
        """Search the knowledge graph and filter the results.

        Combines ``kg.search()`` with this filter so only permitted
        nodes are returned.  Useful for RAG-style retrieval scoped to
        a persona.

        Parameters
        ----------
        kg
            The knowledge graph to search.
        query
            Text query passed to ``kg.search()``.
        limit
            Override ``max_results`` for this search.

        Returns
        -------
        FilterResult
            Matching, filtered nodes.

        Examples
        --------
        ```python
        f = tb.KnowledgeFilter(include_topics=["python"])
        result = f.search(kg, "decorators")
        result.nodes  # only nodes about "python" matching "decorators"
        ```
        """
        effective_limit = limit if limit is not None else self.max_results

        # Fetch more than needed so we can filter and still fill the limit
        search_results = kg.search(query, limit=effective_limit * 3)

        topic_index = _build_topic_index(kg) if self.include_topics or self.exclude_topics else {}

        accepted: list[Node] = []
        excluded = 0

        for node in search_results:
            if not self._accepts(node, topic_index, kg):
                excluded += 1
                continue
            accepted.append(node)
            if len(accepted) >= effective_limit:
                break

        return FilterResult(
            nodes=accepted,
            excluded_count=excluded,
            filter_summary=self._summary(),
        )

    # ------------------------------------------------------------------
    # Predicate
    # ------------------------------------------------------------------

    def _accepts(
        self,
        node: Node,
        topic_index: dict[str, set[str]],
        kg: KnowledgeGraph,
    ) -> bool:
        """Return ``True`` if *node* passes all filter rules."""
        meta = node.metadata

        # --- Node type filter ---
        if self.include_node_types and node.node_type not in self.include_node_types:
            return False

        # --- Confidential filter ---
        if self.exclude_confidential and meta.get("confidential") is True:
            return False

        # --- Expired filter ---
        if self.exclude_expired:
            expires_str = meta.get("expires")
            if expires_str is not None:
                if _is_expired(expires_str):
                    return False

        # --- Tag filters ---
        node_tags = _get_tags(meta)

        if self.include_tags:
            if not node_tags & _lower_set(self.include_tags):
                return False

        if self.exclude_tags:
            if node_tags & _lower_set(self.exclude_tags):
                return False

        # --- Topic filters ---
        if self.include_topics or self.exclude_topics:
            node_topics = _node_topic_names(node, topic_index, kg)

            if self.include_topics:
                include_lower = _lower_set(self.include_topics)
                if not node_topics & include_lower:
                    return False

            if self.exclude_topics:
                exclude_lower = _lower_set(self.exclude_topics)
                if node_topics & exclude_lower:
                    return False

        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable description of the active rules."""
        summary: dict[str, Any] = {}
        if self.include_topics:
            summary["include_topics"] = self.include_topics
        if self.exclude_topics:
            summary["exclude_topics"] = self.exclude_topics
        if self.include_tags:
            summary["include_tags"] = self.include_tags
        if self.exclude_tags:
            summary["exclude_tags"] = self.exclude_tags
        if self.include_node_types:
            summary["include_node_types"] = [t.value for t in self.include_node_types]
        if self.exclude_confidential:
            summary["exclude_confidential"] = True
        if self.exclude_expired:
            summary["exclude_expired"] = True
        summary["max_results"] = self.max_results
        return summary

    @property
    def is_empty(self) -> bool:
        """Return ``True`` if no filtering rules are active.

        An empty filter still respects ``max_results``.

        Examples
        --------
        ```python
        tb.KnowledgeFilter().is_empty  # True
        tb.KnowledgeFilter(include_topics=["ml"]).is_empty  # False
        ```
        """
        return (
            not self.include_topics
            and not self.exclude_topics
            and not self.include_tags
            and not self.exclude_tags
            and not self.include_node_types
            and not self.exclude_confidential
            and not self.exclude_expired
        )

    @property
    def rule_count(self) -> int:
        """Count the number of active filter rules.

        Examples
        --------
        ```python
        f = tb.KnowledgeFilter(
            include_topics=["ml"],
            exclude_confidential=True,
        )
        f.rule_count  # 2
        ```
        """
        count = 0
        if self.include_topics:
            count += 1
        if self.exclude_topics:
            count += 1
        if self.include_tags:
            count += 1
        if self.exclude_tags:
            count += 1
        if self.include_node_types:
            count += 1
        if self.exclude_confidential:
            count += 1
        if self.exclude_expired:
            count += 1
        return count

    def __repr__(self) -> str:
        parts: list[str] = []
        if self.include_topics:
            parts.append(f"topics={self.include_topics}")
        if self.exclude_topics:
            parts.append(f"exclude_topics={self.exclude_topics}")
        if self.include_tags:
            parts.append(f"tags={self.include_tags}")
        if self.exclude_tags:
            parts.append(f"exclude_tags={self.exclude_tags}")
        if self.include_node_types:
            parts.append(f"types={[t.value for t in self.include_node_types]}")
        if self.exclude_confidential:
            parts.append("no_confidential")
        if self.exclude_expired:
            parts.append("no_expired")
        inner = ", ".join(parts) if parts else "pass-through"
        return f"KnowledgeFilter({inner})"


# ---------------------------------------------------------------------------
# Filter result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FilterResult:
    """Result of applying a :class:`KnowledgeFilter` to a knowledge graph.

    Parameters
    ----------
    nodes
        Nodes that passed the filter.
    excluded_count
        Number of nodes rejected by the filter.
    filter_summary
        JSON-serialisable dict describing which rules were active.

    Examples
    --------
    ```python
    result = f.apply(kg)
    result.total       # nodes returned
    result.excluded_count  # nodes filtered out
    ```
    """

    nodes: list[Node] = field(default_factory=list)
    excluded_count: int = 0
    filter_summary: dict[str, Any] = field(default_factory=dict)

    @property
    def total(self) -> int:
        """Number of nodes that passed the filter.

        Examples
        --------
        ```python
        result.total  # 12
        ```
        """
        return len(self.nodes)

    @property
    def node_ids(self) -> list[str]:
        """IDs of nodes that passed the filter.

        Examples
        --------
        ```python
        result.node_ids  # ["doc-1", "entity-python"]
        ```
        """
        return [n.id for n in self.nodes]

    @property
    def node_names(self) -> list[str]:
        """Names of nodes that passed the filter.

        Examples
        --------
        ```python
        result.node_names  # ["README.md", "Python"]
        ```
        """
        return [n.name for n in self.nodes]

    def __repr__(self) -> str:
        return f"FilterResult(total={self.total}, excluded={self.excluded_count})"


# ---------------------------------------------------------------------------
# Persona integration
# ---------------------------------------------------------------------------


def filter_for_persona(
    persona: Any,
    *,
    exclude_confidential: bool = True,
    exclude_expired: bool = True,
    extra_include_topics: list[str] | None = None,
    extra_exclude_topics: list[str] | None = None,
    max_results: int = 100,
    enrichment_config: EnrichmentConfig | None = None,
) -> KnowledgeFilter:
    """Build a :class:`KnowledgeFilter` from a ``PersonaDefinition``.

    Maps persona tags to topic and tag filters:

    * Tags prefixed with ``topic:`` become ``include_topics``.
    * Tags prefixed with ``!topic:`` become ``exclude_topics``.
    * All other tags become ``include_tags`` (if any).

    Parameters
    ----------
    persona
        A ``PersonaDefinition`` (or any object with a ``tags`` list and
        an optional ``name`` attribute).
    exclude_confidential
        Skip confidential nodes (default ``True``).
    exclude_expired
        Skip expired nodes (default ``True``).
    extra_include_topics
        Additional topics to include beyond those inferred from tags.
    extra_exclude_topics
        Additional topics to exclude beyond those inferred from tags.
    max_results
        Maximum nodes returned.
    enrichment_config
        Optional enrichment config for prompt context tuning.

    Returns
    -------
    KnowledgeFilter
        A filter tailored to the persona.

    Examples
    --------
    ```python
    import talk_box as tb

    persona = tb.get_persona("python_mentor")
    f = tb.filter_for_persona(persona)
    result = f.apply(kg)
    ```

    Tag conventions:

    ```python
    from talk_box.personas import create_persona

    persona = create_persona(
        name="ml_engineer",
        display_name="ML Engineer",
        category="technical",
        description="ML-focused assistant",
        persona_role="machine learning engineer",
        tags=["topic:machine-learning", "topic:python", "!topic:legal"],
    )
    f = tb.filter_for_persona(persona)
    f.include_topics   # ["machine-learning", "python"]
    f.exclude_topics   # ["legal"]
    ```
    """
    tags: list[str] = getattr(persona, "tags", []) or []

    include_topics: list[str] = []
    exclude_topics: list[str] = []
    include_tags: list[str] = []

    for tag in tags:
        if tag.startswith("topic:"):
            include_topics.append(tag[len("topic:") :])
        elif tag.startswith("!topic:"):
            exclude_topics.append(tag[len("!topic:") :])
        else:
            include_tags.append(tag)

    if extra_include_topics:
        include_topics.extend(extra_include_topics)
    if extra_exclude_topics:
        exclude_topics.extend(extra_exclude_topics)

    return KnowledgeFilter(
        include_topics=include_topics,
        exclude_topics=exclude_topics,
        include_tags=include_tags,
        exclude_confidential=exclude_confidential,
        exclude_expired=exclude_expired,
        max_results=max_results,
        enrichment_config=enrichment_config,
    )


# ---------------------------------------------------------------------------
# Retrieve context (search + filter convenience)
# ---------------------------------------------------------------------------


def retrieve_context(
    kg: KnowledgeGraph,
    query: str,
    *,
    knowledge_filter: KnowledgeFilter | None = None,
    limit: int = 10,
) -> list[Node]:
    """Search the knowledge graph with optional persona filtering.

    Convenience function combining ``kg.search()`` with
    :class:`KnowledgeFilter` for RAG-style context retrieval.

    Parameters
    ----------
    kg
        The knowledge graph to search.
    query
        Text query for search.
    knowledge_filter
        Optional filter to restrict results.  If ``None``, returns raw
        search results.
    limit
        Maximum number of nodes to return.

    Returns
    -------
    list[Node]
        Filtered search results.

    Examples
    --------
    ```python
    import talk_box as tb

    kg = tb.KnowledgeGraph(":memory:")
    # ... populate graph ...
    nodes = tb.retrieve_context(kg, "deployment", limit=5)
    ```

    With persona filtering:

    ```python
    persona = tb.get_persona("python_mentor")
    f = tb.filter_for_persona(persona)
    nodes = tb.retrieve_context(kg, "decorators", knowledge_filter=f, limit=5)
    ```
    """
    if knowledge_filter is None:
        return kg.search(query, limit=limit)

    result = knowledge_filter.search(kg, query, limit=limit)
    return result.nodes


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_topic_index(kg: KnowledgeGraph) -> dict[str, set[str]]:
    """Build a mapping from node ID to the set of topic names it belongs to.

    Walks all ``belongs_to`` edges to topic nodes and inverts the index
    so each document/entity ID maps to a set of topic names (lowered).
    """
    topic_nodes = kg.list_nodes(node_type=NodeType.TOPIC, limit=10_000)
    topic_name_by_id: dict[str, str] = {t.id: t.name.lower() for t in topic_nodes}

    # node_id -> set of topic names
    index: dict[str, set[str]] = {}

    for topic_node in topic_nodes:
        topic_name = topic_name_by_id[topic_node.id]
        # Incoming "belongs_to" edges point from docs/entities to this topic
        edges = kg.get_edges(topic_node.id, direction="incoming", relation="belongs_to")
        for edge in edges:
            index.setdefault(edge.source, set()).add(topic_name)

    return index


def _node_topic_names(
    node: Node,
    topic_index: dict[str, set[str]],
    kg: KnowledgeGraph,
) -> set[str]:
    """Get the set of topic names associated with *node*.

    Checks the pre-computed topic index first, then falls back to the
    ``_topics`` metadata key set by the enrichment pipeline.
    """
    names: set[str] = set()

    # From the edge-based index
    if node.id in topic_index:
        names |= topic_index[node.id]

    # From enrichment metadata
    meta_topics = node.metadata.get("_topics")
    if isinstance(meta_topics, list):
        names |= {t.lower() for t in meta_topics}

    # For topic nodes themselves, include their own name
    if node.node_type == NodeType.TOPIC:
        names.add(node.name.lower())

    # From directive contexts stored in metadata
    contexts = node.metadata.get("contexts")
    if isinstance(contexts, list):
        names |= {c.lower() for c in contexts}

    return names


def _get_tags(meta: dict[str, Any]) -> set[str]:
    """Extract the set of tags from node metadata (lowered)."""
    tags = meta.get("tags")
    if isinstance(tags, list):
        return {t.lower() for t in tags}
    return set()


def _lower_set(items: list[str]) -> set[str]:
    """Return a lowered set from a list of strings."""
    return {s.lower() for s in items}


def _is_expired(date_str: str) -> bool:
    """Check whether an ISO date string is in the past."""
    from datetime import datetime

    try:
        expires_dt = datetime.strptime(date_str, "%Y-%m-%d")
        return datetime.now() >= expires_dt
    except ValueError:
        return False
