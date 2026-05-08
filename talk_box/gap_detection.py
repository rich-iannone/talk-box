"""Gap detection: find structural holes in a knowledge graph."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from talk_box.knowledge_graph import KnowledgeGraph, Node, NodeType

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class GapType(Enum):
    """Category of knowledge graph gap.

    Values
    ------
    ORPHAN
        A node with zero edges.
    THIN_CLUSTER
        A group of nodes weakly connected to the rest of the graph.
    STALE
        A node that hasn't been updated recently.
    MISSING_RELATIONSHIP
        An expected relationship that doesn't exist.
    WEAK_TOPIC
        A topic with very few documents.

    Examples
    --------
    ```python
    import talk_box as tb

    tb.GapType.ORPHAN
    tb.GapType.STALE
    ```
    """

    ORPHAN = "orphan"
    THIN_CLUSTER = "thin_cluster"
    STALE = "stale"
    MISSING_RELATIONSHIP = "missing_relationship"
    WEAK_TOPIC = "weak_topic"


# ---------------------------------------------------------------------------
# Gap dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Gap:
    """A single structural gap detected in the knowledge graph.

    Parameters
    ----------
    gap_type
        Category of gap (orphan, stale, etc.).
    node_ids
        IDs of nodes involved in this gap.
    description
        Human-readable explanation of the gap.
    suggestion
        Suggested action to resolve the gap.
    severity
        Impact score from 0.0 (minor) to 1.0 (critical).

    Examples
    --------
    ```python
    import talk_box as tb

    report = tb.detect_gaps(kg)
    gap = report.gaps[0]
    gap.gap_type    # GapType.ORPHAN
    gap.suggestion  # "Connect node 'Python' to related topics or documents."
    ```
    """

    gap_type: GapType
    node_ids: list[str] = field(default_factory=list)
    description: str = ""
    suggestion: str = ""
    severity: float = 0.5

    def __repr__(self) -> str:
        return (
            f"Gap(type={self.gap_type.value!r}, "
            f"nodes={len(self.node_ids)}, "
            f"severity={self.severity:.2f})"
        )


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GapReport:
    """Aggregated gap detection results for a knowledge graph.

    Parameters
    ----------
    gaps
        All detected gaps, sorted by severity (highest first).

    Examples
    --------
    ```python
    import talk_box as tb

    report = tb.detect_gaps(kg)
    report.total          # 5
    report.by_type(tb.GapType.ORPHAN)  # gaps of type ORPHAN
    report.severity_score # 0.65
    ```
    """

    gaps: list[Gap] = field(default_factory=list)

    @property
    def total(self) -> int:
        """Total number of gaps detected.

        Examples
        --------
        ```python
        report.total  # 5
        ```
        """
        return len(self.gaps)

    @property
    def severity_score(self) -> float:
        """Mean severity across all gaps (0.0 = healthy, 1.0 = critical).

        Returns 0.0 if no gaps are detected.

        Examples
        --------
        ```python
        report.severity_score  # 0.45
        ```
        """
        if not self.gaps:
            return 0.0
        return sum(g.severity for g in self.gaps) / len(self.gaps)

    @property
    def orphan_count(self) -> int:
        """Number of orphan node gaps.

        Examples
        --------
        ```python
        report.orphan_count  # 2
        ```
        """
        return sum(1 for g in self.gaps if g.gap_type == GapType.ORPHAN)

    @property
    def stale_count(self) -> int:
        """Number of stale node gaps.

        Examples
        --------
        ```python
        report.stale_count  # 3
        ```
        """
        return sum(1 for g in self.gaps if g.gap_type == GapType.STALE)

    def by_type(self, gap_type: GapType) -> list[Gap]:
        """Filter gaps by type.

        Parameters
        ----------
        gap_type
            The gap type to filter by.

        Returns
        -------
        list[Gap]
            Gaps matching the specified type.

        Examples
        --------
        ```python
        orphans = report.by_type(tb.GapType.ORPHAN)
        ```
        """
        return [g for g in self.gaps if g.gap_type == gap_type]

    @property
    def suggestions(self) -> list[str]:
        """All suggested actions from detected gaps.

        Examples
        --------
        ```python
        for s in report.suggestions:
            print(s)
        ```
        """
        return [g.suggestion for g in self.gaps if g.suggestion]

    def to_dict(self) -> dict[str, Any]:
        """Serialise the report to a JSON-friendly dict.

        Returns
        -------
        dict[str, Any]
            Summary with counts, severity, and gap details.

        Examples
        --------
        ```python
        report.to_dict()["total"]  # 5
        ```
        """
        return {
            "total": self.total,
            "severity_score": round(self.severity_score, 4),
            "orphan_count": self.orphan_count,
            "stale_count": self.stale_count,
            "thin_cluster_count": len(self.by_type(GapType.THIN_CLUSTER)),
            "missing_relationship_count": len(self.by_type(GapType.MISSING_RELATIONSHIP)),
            "weak_topic_count": len(self.by_type(GapType.WEAK_TOPIC)),
            "gaps": [
                {
                    "type": g.gap_type.value,
                    "node_ids": g.node_ids,
                    "description": g.description,
                    "suggestion": g.suggestion,
                    "severity": round(g.severity, 4),
                }
                for g in self.gaps
            ],
        }

    def __repr__(self) -> str:
        return f"GapReport(total={self.total}, severity={self.severity_score:.2f})"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_gaps(
    kg: KnowledgeGraph,
    *,
    stale_days: float = 30.0,
    min_topic_docs: int = 2,
    thin_cluster_threshold: int = 2,
) -> GapReport:
    """Detect structural gaps in a knowledge graph.

    Analyses the graph for five categories of gaps:

    1. **Orphan nodes** — nodes with zero edges.
    2. **Thin clusters** — entity/topic nodes connected to very few
       others (below ``thin_cluster_threshold``).
    3. **Stale regions** — nodes not updated within ``stale_days``.
    4. **Missing relationships** — entities that co-occur in documents
       but have no direct edge between them.
    5. **Weak topics** — topics with fewer than ``min_topic_docs``
       document associations.

    Parameters
    ----------
    kg
        The knowledge graph to analyse.
    stale_days
        Number of days since last update after which a node is
        considered stale.
    min_topic_docs
        Minimum number of documents a topic should have.  Topics
        below this threshold are flagged.
    thin_cluster_threshold
        Maximum edge count for a node to be considered "thinly
        connected" (exclusive).  Only applies to entities and topics.

    Returns
    -------
    GapReport
        All detected gaps sorted by severity.

    Examples
    --------
    ```python
    import talk_box as tb

    kg = tb.KnowledgeGraph(":memory:")
    # ... populate the graph ...
    report = tb.detect_gaps(kg)
    report.total           # number of gaps
    report.severity_score  # overall health (0 = good)
    report.suggestions     # actionable fixes
    ```

    Customise detection parameters:

    ```python
    report = tb.detect_gaps(kg, stale_days=7, min_topic_docs=3)
    ```
    """
    gaps: list[Gap] = []

    all_nodes = kg.list_nodes(limit=10_000)
    if not all_nodes:
        return GapReport()

    # Pre-compute edge info
    node_edge_counts = _compute_edge_counts(kg, all_nodes)

    # 1. Orphan nodes
    gaps.extend(_find_orphans(all_nodes, node_edge_counts))

    # 2. Thin clusters
    gaps.extend(_find_thin_clusters(all_nodes, node_edge_counts, thin_cluster_threshold))

    # 3. Stale regions
    gaps.extend(_find_stale_nodes(all_nodes, stale_days))

    # 4. Missing relationships
    gaps.extend(_find_missing_relationships(kg, all_nodes))

    # 5. Weak topics
    gaps.extend(_find_weak_topics(kg, all_nodes, min_topic_docs))

    # Sort by severity (highest first)
    gaps.sort(key=lambda g: g.severity, reverse=True)

    return GapReport(gaps=gaps)


# ---------------------------------------------------------------------------
# Detection functions
# ---------------------------------------------------------------------------


def _compute_edge_counts(kg: KnowledgeGraph, nodes: list[Node]) -> dict[str, int]:
    """Count edges per node."""
    counts: dict[str, int] = {}
    for node in nodes:
        edges = kg.get_edges(node.id, direction="both")
        counts[node.id] = len(edges)
    return counts


def _find_orphans(nodes: list[Node], edge_counts: dict[str, int]) -> list[Gap]:
    """Find nodes with zero edges."""
    gaps: list[Gap] = []
    for node in nodes:
        if edge_counts.get(node.id, 0) == 0:
            gaps.append(
                Gap(
                    gap_type=GapType.ORPHAN,
                    node_ids=[node.id],
                    description=(
                        f"Node {node.name!r} ({node.node_type.value}) has no connections."
                    ),
                    suggestion=(f"Connect {node.name!r} to related topics or documents."),
                    severity=0.7,
                )
            )
    return gaps


def _find_thin_clusters(
    nodes: list[Node],
    edge_counts: dict[str, int],
    threshold: int,
) -> list[Gap]:
    """Find entity/topic nodes with very few connections (but not orphans)."""
    gaps: list[Gap] = []
    for node in nodes:
        if node.node_type == NodeType.DOCUMENT:
            continue
        count = edge_counts.get(node.id, 0)
        if 0 < count < threshold:
            gaps.append(
                Gap(
                    gap_type=GapType.THIN_CLUSTER,
                    node_ids=[node.id],
                    description=(
                        f"Node {node.name!r} ({node.node_type.value}) "
                        f"has only {count} connection(s)."
                    ),
                    suggestion=(
                        f"Add more relationships for {node.name!r} to improve discoverability."
                    ),
                    severity=0.4,
                )
            )
    return gaps


def _find_stale_nodes(nodes: list[Node], stale_days: float) -> list[Gap]:
    """Find nodes not updated within the staleness window."""
    gaps: list[Gap] = []
    now = time.time()
    cutoff = now - (stale_days * 86_400)

    for node in nodes:
        if node.updated_at > 0 and node.updated_at < cutoff:
            days_old = int((now - node.updated_at) / 86_400)
            gaps.append(
                Gap(
                    gap_type=GapType.STALE,
                    node_ids=[node.id],
                    description=(f"Node {node.name!r} last updated {days_old} days ago."),
                    suggestion=(f"Review {node.name!r} for accuracy and update or archive it."),
                    severity=min(0.3 + (days_old / 365.0) * 0.5, 0.9),
                )
            )
    return gaps


def _find_missing_relationships(kg: KnowledgeGraph, nodes: list[Node]) -> list[Gap]:
    """Find entity pairs that co-occur in documents but have no direct edge.

    Two entities "co-occur" if they are both mentioned by the same
    document node.  If they share a document but have no direct edge
    between them, that's a potential missing relationship.
    """
    gaps: list[Gap] = []

    # Build doc -> entity index
    doc_entities: dict[str, list[str]] = {}
    entity_names: dict[str, str] = {}

    for node in nodes:
        if node.node_type == NodeType.ENTITY:
            entity_names[node.id] = node.name

    for node in nodes:
        if node.node_type != NodeType.DOCUMENT:
            continue
        edges = kg.get_edges(node.id, direction="outgoing", relation="mentions")
        entity_ids = [e.target for e in edges if e.target in entity_names]
        if len(entity_ids) >= 2:
            doc_entities[node.id] = entity_ids

    # Find co-occurring pairs without direct edges
    seen_pairs: set[tuple[str, str]] = set()

    for entity_ids in doc_entities.values():
        for i, eid_a in enumerate(entity_ids):
            for eid_b in entity_ids[i + 1 :]:
                pair = (min(eid_a, eid_b), max(eid_a, eid_b))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                # Check if a direct edge exists between them
                edges_ab = kg.get_edges(eid_a, direction="outgoing")
                has_direct = any(e.target == eid_b for e in edges_ab)
                if not has_direct:
                    edges_ba = kg.get_edges(eid_b, direction="outgoing")
                    has_direct = any(e.target == eid_a for e in edges_ba)

                if not has_direct:
                    name_a = entity_names[eid_a]
                    name_b = entity_names[eid_b]
                    gaps.append(
                        Gap(
                            gap_type=GapType.MISSING_RELATIONSHIP,
                            node_ids=[eid_a, eid_b],
                            description=(
                                f"Entities {name_a!r} and {name_b!r} "
                                f"co-occur in documents but have no "
                                f"direct relationship."
                            ),
                            suggestion=(
                                f"Consider adding a relationship between {name_a!r} and {name_b!r}."
                            ),
                            severity=0.5,
                        )
                    )

    return gaps


def _find_weak_topics(
    kg: KnowledgeGraph,
    nodes: list[Node],
    min_docs: int,
) -> list[Gap]:
    """Find topics with fewer than min_docs document associations."""
    gaps: list[Gap] = []

    for node in nodes:
        if node.node_type != NodeType.TOPIC:
            continue
        # Count incoming belongs_to edges from documents
        edges = kg.get_edges(node.id, direction="incoming", relation="belongs_to")
        doc_count = len(edges)
        if doc_count < min_docs:
            gaps.append(
                Gap(
                    gap_type=GapType.WEAK_TOPIC,
                    node_ids=[node.id],
                    description=(
                        f"Topic {node.name!r} has only {doc_count} "
                        f"document(s) (minimum: {min_docs})."
                    ),
                    suggestion=(
                        f"Add more documents to topic {node.name!r} "
                        f"or merge it with a related topic."
                    ),
                    severity=0.3,
                )
            )

    return gaps
