"""Confusion metric: per-node ambiguity scores for knowledge graph quality."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from talk_box.knowledge_graph import KnowledgeGraph, Node, NodeType

# ---------------------------------------------------------------------------
# Score dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfusionScore:
    """Ambiguity score for a single knowledge graph node.

    A score of **0.0** means the node is clearly defined with rich
    context.  A score of **1.0** means maximum ambiguity — the node
    is poorly connected, has name collisions, or lacks metadata.

    Parameters
    ----------
    node_id
        The scored node's identifier.
    node_name
        Human-readable name of the node.
    score
        Confusion score in [0.0, 1.0].
    reasons
        Human-readable explanations for the score.

    Examples
    --------
    ```python
    import talk_box as tb

    report = tb.confusion(kg)
    report.scores[0].score    # 0.35
    report.scores[0].reasons  # ["low connectivity (1 edge)", ...]
    ```
    """

    node_id: str
    node_name: str
    score: float
    reasons: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"ConfusionScore(node={self.node_name!r}, "
            f"score={self.score:.2f}, reasons={len(self.reasons)})"
        )


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfusionReport:
    """Aggregated confusion scores for a knowledge graph.

    Parameters
    ----------
    scores
        Individual node scores, sorted by score descending.

    Examples
    --------
    ```python
    import talk_box as tb

    report = tb.confusion(kg)
    report.mean_score      # 0.25
    report.max_score       # 0.72
    report.confused_count  # 3  (nodes above threshold)
    ```
    """

    scores: list[ConfusionScore] = field(default_factory=list)
    threshold: float = 0.3

    @property
    def mean_score(self) -> float:
        """Mean confusion score across all scored nodes.

        Examples
        --------
        ```python
        report.mean_score  # 0.25
        ```
        """
        if not self.scores:
            return 0.0
        return sum(s.score for s in self.scores) / len(self.scores)

    @property
    def max_score(self) -> float:
        """Highest confusion score in the report.

        Examples
        --------
        ```python
        report.max_score  # 0.72
        ```
        """
        if not self.scores:
            return 0.0
        return max(s.score for s in self.scores)

    @property
    def confused_nodes(self) -> list[ConfusionScore]:
        """Nodes whose score exceeds the threshold.

        Examples
        --------
        ```python
        for node in report.confused_nodes:
            print(node.node_name, node.score)
        ```
        """
        return [s for s in self.scores if s.score > self.threshold]

    @property
    def confused_count(self) -> int:
        """Number of nodes above the confusion threshold.

        Examples
        --------
        ```python
        report.confused_count  # 3
        ```
        """
        return len(self.confused_nodes)

    @property
    def clear_nodes(self) -> list[ConfusionScore]:
        """Nodes whose score is at or below the threshold.

        Examples
        --------
        ```python
        report.clear_nodes  # low-ambiguity nodes
        ```
        """
        return [s for s in self.scores if s.score <= self.threshold]

    @property
    def clear_count(self) -> int:
        """Number of nodes at or below the confusion threshold.

        Examples
        --------
        ```python
        report.clear_count  # 7
        ```
        """
        return len(self.clear_nodes)

    @property
    def total(self) -> int:
        """Total number of scored nodes.

        Examples
        --------
        ```python
        report.total  # 10
        ```
        """
        return len(self.scores)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the report to a JSON-friendly dict.

        Returns
        -------
        dict[str, Any]
            Summary with ``mean_score``, ``max_score``, ``confused_count``,
            ``clear_count``, ``total``, ``threshold``, and ``scores`` list.

        Examples
        --------
        ```python
        report.to_dict()["mean_score"]  # 0.25
        ```
        """
        return {
            "mean_score": round(self.mean_score, 4),
            "max_score": round(self.max_score, 4),
            "confused_count": self.confused_count,
            "clear_count": self.clear_count,
            "total": self.total,
            "threshold": self.threshold,
            "scores": [
                {
                    "node_id": s.node_id,
                    "node_name": s.node_name,
                    "score": round(s.score, 4),
                    "reasons": s.reasons,
                }
                for s in self.scores
            ],
        }

    def __repr__(self) -> str:
        return (
            f"ConfusionReport(total={self.total}, "
            f"mean={self.mean_score:.2f}, "
            f"confused={self.confused_count})"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def confusion(
    kg: KnowledgeGraph,
    *,
    threshold: float = 0.3,
    node_type: NodeType | None = None,
) -> ConfusionReport:
    """Compute confusion scores for nodes in a knowledge graph.

    Evaluates every entity and topic node (or a specific type) on five
    ambiguity dimensions:

    1. **Connectivity** — orphan or sparsely connected nodes are harder
       to disambiguate.
    2. **Name ambiguity** — similar names in the graph create confusion
       for retrieval.
    3. **Missing metadata** — nodes without entity type, content, or
       tags lack context.
    4. **Weak edges** — low-weight edges suggest uncertain relationships.
    5. **Type coverage** — entities without an explicit type are harder
       to classify.

    Parameters
    ----------
    kg
        The knowledge graph to analyse.
    threshold
        Score above which a node is considered "confused" (default 0.3).
    node_type
        If provided, only score nodes of this type.  By default scores
        ``ENTITY`` and ``TOPIC`` nodes (not documents).

    Returns
    -------
    ConfusionReport
        Scores sorted by confusion (highest first).

    Examples
    --------
    ```python
    import talk_box as tb

    kg = tb.KnowledgeGraph(":memory:")
    # ... add nodes and edges ...
    report = tb.confusion(kg)
    report.mean_score       # overall ambiguity
    report.confused_nodes   # nodes that need attention
    ```

    Filter to entities only:

    ```python
    report = tb.confusion(kg, node_type=tb.NodeType.ENTITY)
    ```
    """
    # Gather target nodes
    if node_type is not None:
        targets = kg.list_nodes(node_type=node_type, limit=10_000)
    else:
        entities = kg.list_nodes(node_type=NodeType.ENTITY, limit=10_000)
        topics = kg.list_nodes(node_type=NodeType.TOPIC, limit=10_000)
        targets = entities + topics

    if not targets:
        return ConfusionReport(threshold=threshold)

    # Pre-compute shared data structures once
    all_names = _collect_names(kg)
    name_counts = _name_frequency(all_names)

    scores: list[ConfusionScore] = []
    for node in targets:
        score = node_confusion(
            kg,
            node,
            all_names=all_names,
            name_counts=name_counts,
        )
        scores.append(score)

    # Sort by score descending (most confused first)
    scores.sort(key=lambda s: s.score, reverse=True)

    return ConfusionReport(scores=scores, threshold=threshold)


def node_confusion(
    kg: KnowledgeGraph,
    node: Node | str,
    *,
    all_names: list[str] | None = None,
    name_counts: dict[str, int] | None = None,
) -> ConfusionScore:
    """Compute the confusion score for a single node.

    Parameters
    ----------
    kg
        The knowledge graph.
    node
        A ``Node`` object or a node ID string.
    all_names
        Pre-computed list of all node names (for performance when
        scoring many nodes).  Computed on-the-fly if ``None``.
    name_counts
        Pre-computed name frequency map.  Computed on-the-fly if ``None``.

    Returns
    -------
    ConfusionScore
        The node's ambiguity score and reasons.

    Examples
    --------
    ```python
    import talk_box as tb

    score = tb.node_confusion(kg, "entity-abc123")
    score.score    # 0.45
    score.reasons  # ["orphan node (0 edges)", ...]
    ```
    """
    if isinstance(node, str):
        resolved = kg.get_node(node)
        if resolved is None:
            return ConfusionScore(
                node_id=node,
                node_name="<unknown>",
                score=1.0,
                reasons=["node not found in graph"],
            )
        node = resolved

    if all_names is None:
        all_names = _collect_names(kg)
    if name_counts is None:
        name_counts = _name_frequency(all_names)

    factors: list[tuple[float, str]] = []

    # --- 1. Connectivity ---
    edges = kg.get_edges(node.id, direction="both")
    edge_count = len(edges)
    conn_score, conn_reason = _connectivity_factor(edge_count)
    if conn_score > 0:
        factors.append((conn_score, conn_reason))

    # --- 2. Name ambiguity ---
    amb_score, amb_reason = _name_ambiguity_factor(node.name, all_names, name_counts)
    if amb_score > 0:
        factors.append((amb_score, amb_reason))

    # --- 3. Missing metadata ---
    meta_score, meta_reason = _metadata_factor(node)
    if meta_score > 0:
        factors.append((meta_score, meta_reason))

    # --- 4. Weak edges ---
    weak_score, weak_reason = _weak_edges_factor(edges)
    if weak_score > 0:
        factors.append((weak_score, weak_reason))

    # --- 5. Type coverage (entities only) ---
    if node.node_type == NodeType.ENTITY:
        type_score, type_reason = _type_coverage_factor(node)
        if type_score > 0:
            factors.append((type_score, type_reason))

    # Combine factors with weights
    final_score = _combine_factors(factors)
    reasons = [r for _, r in factors]

    return ConfusionScore(
        node_id=node.id,
        node_name=node.name,
        score=round(min(final_score, 1.0), 4),
        reasons=reasons,
    )


# ---------------------------------------------------------------------------
# Factor functions (each returns (score, reason) for its dimension)
# ---------------------------------------------------------------------------

# Weights for combining factors
_WEIGHT_CONNECTIVITY = 0.35
_WEIGHT_NAME_AMBIGUITY = 0.25
_WEIGHT_METADATA = 0.15
_WEIGHT_WEAK_EDGES = 0.15
_WEIGHT_TYPE_COVERAGE = 0.10


def _connectivity_factor(edge_count: int) -> tuple[float, str]:
    """Score based on how many edges connect a node.

    0 edges = 1.0 (orphan), 1 edge = 0.7, 2 edges = 0.4, 3+ = 0.0.
    """
    if edge_count == 0:
        return 1.0, "orphan node (0 edges)"
    if edge_count == 1:
        return 0.7, "low connectivity (1 edge)"
    if edge_count == 2:
        return 0.4, "moderate connectivity (2 edges)"
    return 0.0, ""


def _name_ambiguity_factor(
    name: str,
    all_names: list[str],
    name_counts: dict[str, int],
) -> tuple[float, str]:
    """Score based on name collisions and substring overlaps.

    Exact duplicates are the strongest signal, followed by substring
    containment (e.g., "Python" vs "Python Programming").
    """
    lower = name.lower()
    dup_count = name_counts.get(lower, 0)

    # Exact duplicates (more than 1 node with same lowered name)
    if dup_count > 1:
        return 1.0, f"name collision ({dup_count} nodes named {name!r})"

    # Substring containment: other names that contain this name or vice versa
    # Only check for short-ish names to avoid matching everything
    if len(lower) >= 3:
        similar = 0
        for other in all_names:
            other_lower = other.lower()
            if other_lower == lower:
                continue
            if lower in other_lower or other_lower in lower:
                similar += 1
        if similar >= 3:
            return 0.8, f"name substring overlaps with {similar} other nodes"
        if similar >= 1:
            return 0.5, f"name substring overlap with {similar} other node(s)"

    return 0.0, ""


def _metadata_factor(node: Node) -> tuple[float, str]:
    """Score based on missing metadata fields.

    Checks for content, tags, and enrichment metadata.
    """
    missing: list[str] = []

    if not node.content:
        missing.append("content")

    tags = node.metadata.get("tags")
    if not tags:
        missing.append("tags")

    # Check for enrichment metadata on document nodes
    if node.node_type == NodeType.DOCUMENT:
        if not node.metadata.get("_enriched"):
            missing.append("enrichment")
    elif node.node_type == NodeType.ENTITY:
        if not node.metadata.get("entity_type") or node.metadata.get("entity_type") == "unknown":
            missing.append("entity_type")

    if not missing:
        return 0.0, ""

    score = min(len(missing) / 3.0, 1.0)
    return score, f"missing metadata: {', '.join(missing)}"


def _weak_edges_factor(edges: list[Any]) -> tuple[float, str]:
    """Score based on the proportion of low-weight edges."""
    if not edges:
        return 0.0, ""  # Handled by connectivity factor

    weak = sum(1 for e in edges if e.weight < 0.3)
    if weak == 0:
        return 0.0, ""

    ratio = weak / len(edges)
    if ratio >= 0.5:
        return 0.8, f"many weak edges ({weak}/{len(edges)} below 0.3 weight)"
    return 0.4, f"some weak edges ({weak}/{len(edges)} below 0.3 weight)"


def _type_coverage_factor(node: Node) -> tuple[float, str]:
    """Score for entity nodes without an explicit entity type."""
    entity_type = node.metadata.get("entity_type", "")
    if not entity_type or entity_type == "unknown":
        return 1.0, "no explicit entity type"
    return 0.0, ""


# ---------------------------------------------------------------------------
# Combination
# ---------------------------------------------------------------------------


def _combine_factors(factors: list[tuple[float, str]]) -> float:
    """Combine individual factor scores into a single confusion score.

    Uses a weighted-average approach.  Factors with empty reasons are
    skipped (they scored 0).
    """
    if not factors:
        return 0.0

    # Map factor reasons to weight categories
    total_weight = 0.0
    weighted_sum = 0.0

    for score, reason in factors:
        weight = _weight_for_reason(reason)
        weighted_sum += score * weight
        total_weight += weight

    if total_weight == 0:
        return 0.0

    return weighted_sum / total_weight


def _weight_for_reason(reason: str) -> float:
    """Map a reason string to its category weight."""
    reason_lower = reason.lower()
    if "orphan" in reason_lower or "connectivity" in reason_lower:
        return _WEIGHT_CONNECTIVITY
    if "name" in reason_lower:
        return _WEIGHT_NAME_AMBIGUITY
    if "metadata" in reason_lower:
        return _WEIGHT_METADATA
    if "weak edge" in reason_lower:
        return _WEIGHT_WEAK_EDGES
    if "entity type" in reason_lower:
        return _WEIGHT_TYPE_COVERAGE
    return 0.1  # fallback


# ---------------------------------------------------------------------------
# Name helpers
# ---------------------------------------------------------------------------


def _collect_names(kg: KnowledgeGraph) -> list[str]:
    """Collect all node names from the graph."""
    nodes = kg.list_nodes(limit=10_000)
    return [n.name for n in nodes]


def _name_frequency(names: list[str]) -> dict[str, int]:
    """Count how many times each lowered name appears."""
    freq: dict[str, int] = {}
    for name in names:
        key = name.lower()
        freq[key] = freq.get(key, 0) + 1
    return freq
