"""Freshness reporting: knowledge graph hygiene based on node age."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from talk_box.knowledge_graph import KnowledgeGraph, Node, NodeType

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class FreshnessStatus(Enum):
    """Freshness classification for a node.

    Values
    ------
    FRESH
        Updated within the fresh window.
    AGING
        Updated between the fresh and stale thresholds.
    STALE
        Not updated within the stale threshold.
    UNKNOWN
        Node has no timestamp data.

    Examples
    --------
    ```python
    import talk_box as tb

    tb.FreshnessStatus.FRESH
    tb.FreshnessStatus.STALE
    ```
    """

    FRESH = "fresh"
    AGING = "aging"
    STALE = "stale"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Per-node entry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FreshnessEntry:
    """Freshness assessment for a single node.

    Parameters
    ----------
    node_id
        The node's identifier.
    node_name
        Human-readable name.
    node_type
        The node's type.
    status
        Freshness classification.
    age_days
        Days since last update (``-1`` if unknown).
    updated_at
        Unix timestamp of last update (``0.0`` if unknown).

    Examples
    --------
    ```python
    import talk_box as tb

    report = tb.freshness_report(kg)
    entry = report.entries[0]
    entry.status    # FreshnessStatus.FRESH
    entry.age_days  # 3
    ```
    """

    node_id: str
    node_name: str
    node_type: NodeType
    status: FreshnessStatus
    age_days: int = -1
    updated_at: float = 0.0

    def __repr__(self) -> str:
        return (
            f"FreshnessEntry(node={self.node_name!r}, "
            f"status={self.status.value}, "
            f"age_days={self.age_days})"
        )


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FreshnessReport:
    """Aggregated freshness report for a knowledge graph.

    Parameters
    ----------
    entries
        Per-node freshness entries sorted by age (oldest first).
    fresh_days
        Threshold in days for "fresh" status.
    stale_days
        Threshold in days for "stale" status.

    Examples
    --------
    ```python
    import talk_box as tb

    report = tb.freshness_report(kg)
    report.fresh_count   # 10
    report.stale_count   # 2
    report.coverage      # 0.83
    ```
    """

    entries: list[FreshnessEntry] = field(default_factory=list)
    fresh_days: float = 7.0
    stale_days: float = 30.0

    @property
    def total(self) -> int:
        """Total number of entries in the report.

        Examples
        --------
        ```python
        report.total  # 12
        ```
        """
        return len(self.entries)

    @property
    def fresh_count(self) -> int:
        """Number of nodes classified as fresh.

        Examples
        --------
        ```python
        report.fresh_count  # 8
        ```
        """
        return sum(1 for e in self.entries if e.status == FreshnessStatus.FRESH)

    @property
    def aging_count(self) -> int:
        """Number of nodes classified as aging.

        Examples
        --------
        ```python
        report.aging_count  # 3
        ```
        """
        return sum(1 for e in self.entries if e.status == FreshnessStatus.AGING)

    @property
    def stale_count(self) -> int:
        """Number of nodes classified as stale.

        Examples
        --------
        ```python
        report.stale_count  # 2
        ```
        """
        return sum(1 for e in self.entries if e.status == FreshnessStatus.STALE)

    @property
    def unknown_count(self) -> int:
        """Number of nodes with unknown freshness.

        Examples
        --------
        ```python
        report.unknown_count  # 0
        ```
        """
        return sum(1 for e in self.entries if e.status == FreshnessStatus.UNKNOWN)

    @property
    def coverage(self) -> float:
        """Fraction of nodes that are fresh (0.0–1.0).

        Returns 0.0 if no entries exist.

        Examples
        --------
        ```python
        report.coverage  # 0.75
        ```
        """
        if not self.entries:
            return 0.0
        return self.fresh_count / len(self.entries)

    @property
    def mean_age_days(self) -> float:
        """Mean age in days across nodes with known timestamps.

        Returns 0.0 if no nodes have timestamps.

        Examples
        --------
        ```python
        report.mean_age_days  # 12.5
        ```
        """
        ages = [e.age_days for e in self.entries if e.age_days >= 0]
        if not ages:
            return 0.0
        return sum(ages) / len(ages)

    @property
    def max_age_days(self) -> int:
        """Maximum age in days across all entries.

        Returns 0 if no nodes have timestamps.

        Examples
        --------
        ```python
        report.max_age_days  # 90
        ```
        """
        ages = [e.age_days for e in self.entries if e.age_days >= 0]
        if not ages:
            return 0
        return max(ages)

    @property
    def stale_entries(self) -> list[FreshnessEntry]:
        """Entries classified as stale.

        Examples
        --------
        ```python
        for entry in report.stale_entries:
            print(entry.node_name, entry.age_days)
        ```
        """
        return [e for e in self.entries if e.status == FreshnessStatus.STALE]

    @property
    def fresh_entries(self) -> list[FreshnessEntry]:
        """Entries classified as fresh.

        Examples
        --------
        ```python
        report.fresh_entries
        ```
        """
        return [e for e in self.entries if e.status == FreshnessStatus.FRESH]

    def by_type(self, node_type: NodeType) -> list[FreshnessEntry]:
        """Filter entries by node type.

        Parameters
        ----------
        node_type
            The node type to filter by.

        Returns
        -------
        list[FreshnessEntry]
            Entries matching the specified type.

        Examples
        --------
        ```python
        docs = report.by_type(tb.NodeType.DOCUMENT)
        ```
        """
        return [e for e in self.entries if e.node_type == node_type]

    def by_status(self, status: FreshnessStatus) -> list[FreshnessEntry]:
        """Filter entries by freshness status.

        Parameters
        ----------
        status
            The status to filter by.

        Returns
        -------
        list[FreshnessEntry]
            Entries matching the specified status.

        Examples
        --------
        ```python
        stale = report.by_status(tb.FreshnessStatus.STALE)
        ```
        """
        return [e for e in self.entries if e.status == status]

    def to_dict(self) -> dict[str, Any]:
        """Serialise the report to a JSON-friendly dict.

        Returns
        -------
        dict[str, Any]
            Summary with counts, coverage, and entry details.

        Examples
        --------
        ```python
        report.to_dict()["coverage"]  # 0.75
        ```
        """
        return {
            "total": self.total,
            "fresh_count": self.fresh_count,
            "aging_count": self.aging_count,
            "stale_count": self.stale_count,
            "unknown_count": self.unknown_count,
            "coverage": round(self.coverage, 4),
            "mean_age_days": round(self.mean_age_days, 1),
            "max_age_days": self.max_age_days,
            "fresh_days": self.fresh_days,
            "stale_days": self.stale_days,
            "entries": [
                {
                    "node_id": e.node_id,
                    "node_name": e.node_name,
                    "node_type": e.node_type.value,
                    "status": e.status.value,
                    "age_days": e.age_days,
                }
                for e in self.entries
            ],
        }

    def __repr__(self) -> str:
        return (
            f"FreshnessReport(total={self.total}, "
            f"fresh={self.fresh_count}, "
            f"aging={self.aging_count}, "
            f"stale={self.stale_count})"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def freshness_report(
    kg: KnowledgeGraph,
    *,
    fresh_days: float = 7.0,
    stale_days: float = 30.0,
    node_type: NodeType | None = None,
) -> FreshnessReport:
    """Generate a freshness report for a knowledge graph.

    Classifies every node by how recently it was updated:

    - **Fresh**: updated within ``fresh_days``
    - **Aging**: updated between ``fresh_days`` and ``stale_days``
    - **Stale**: not updated within ``stale_days``
    - **Unknown**: no timestamp data

    Parameters
    ----------
    kg
        The knowledge graph to analyse.
    fresh_days
        Maximum age in days for a node to be "fresh" (default 7).
    stale_days
        Minimum age in days for a node to be "stale" (default 30).
    node_type
        If provided, only report on nodes of this type.

    Returns
    -------
    FreshnessReport
        Report sorted by age (oldest first).

    Examples
    --------
    ```python
    import talk_box as tb

    kg = tb.KnowledgeGraph(":memory:")
    # ... add nodes ...
    report = tb.freshness_report(kg)
    report.coverage      # fraction of nodes that are fresh
    report.stale_count   # nodes needing attention
    report.stale_entries # details on stale nodes
    ```

    Filter to documents only:

    ```python
    report = tb.freshness_report(kg, node_type=tb.NodeType.DOCUMENT)
    ```

    Customise thresholds:

    ```python
    report = tb.freshness_report(kg, fresh_days=3, stale_days=14)
    ```
    """
    if node_type is not None:
        nodes = kg.list_nodes(node_type=node_type, limit=10_000)
    else:
        nodes = kg.list_nodes(limit=10_000)

    if not nodes:
        return FreshnessReport(fresh_days=fresh_days, stale_days=stale_days)

    now = time.time()
    entries: list[FreshnessEntry] = []

    for node in nodes:
        entry = _classify_node(node, now, fresh_days, stale_days)
        entries.append(entry)

    # Sort by age descending (oldest first)
    entries.sort(key=lambda e: e.age_days, reverse=True)

    return FreshnessReport(
        entries=entries,
        fresh_days=fresh_days,
        stale_days=stale_days,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _classify_node(
    node: Node,
    now: float,
    fresh_days: float,
    stale_days: float,
) -> FreshnessEntry:
    """Classify a single node's freshness."""
    if node.updated_at <= 0:
        return FreshnessEntry(
            node_id=node.id,
            node_name=node.name,
            node_type=node.node_type,
            status=FreshnessStatus.UNKNOWN,
            age_days=-1,
            updated_at=0.0,
        )

    age_seconds = now - node.updated_at
    age_days = int(age_seconds / 86_400)

    if age_days <= fresh_days:
        status = FreshnessStatus.FRESH
    elif age_days >= stale_days:
        status = FreshnessStatus.STALE
    else:
        status = FreshnessStatus.AGING

    return FreshnessEntry(
        node_id=node.id,
        node_name=node.name,
        node_type=node.node_type,
        status=status,
        age_days=age_days,
        updated_at=node.updated_at,
    )
