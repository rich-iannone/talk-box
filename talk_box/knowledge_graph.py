"""Knowledge graph core: SQLite-backed graph with nodes, edges, and optional embeddings."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class NodeType(Enum):
    """Type of a knowledge graph node.

    Values
    ------
    DOCUMENT
        A source document (file, note, web page).
    ENTITY
        A named entity extracted from documents (person, org, concept).
    TOPIC
        A topic or category used for classification.

    Examples
    --------
    ```python
    import talk_box as tb

    tb.NodeType.DOCUMENT
    tb.NodeType.ENTITY
    tb.NodeType.TOPIC
    ```
    """

    DOCUMENT = "document"
    ENTITY = "entity"
    TOPIC = "topic"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class Node:
    """A node in the knowledge graph.

    Parameters
    ----------
    id
        Unique identifier for the node.
    node_type
        The type of node (document, entity, or topic).
    name
        Human-readable name or title.
    content
        Full text content (for documents) or description.
    metadata
        Arbitrary key-value metadata.
    embedding
        Optional vector embedding as a list of floats.
    created_at
        Unix timestamp of creation.
    updated_at
        Unix timestamp of last update.

    Examples
    --------
    ```python
    import talk_box as tb

    node = tb.Node(
        id="doc-001",
        node_type=tb.NodeType.DOCUMENT,
        name="README.md",
        content="# My Project...",
    )
    node.name  # "README.md"
    ```
    """

    id: str
    node_type: NodeType
    name: str
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    created_at: float = 0.0
    updated_at: float = 0.0

    def __post_init__(self) -> None:
        now = time.time()
        if self.created_at == 0.0:
            self.created_at = now
        if self.updated_at == 0.0:
            self.updated_at = now


@dataclass(frozen=True)
class Edge:
    """A directed relationship between two nodes.

    Parameters
    ----------
    source
        ID of the source node.
    target
        ID of the target node.
    relation
        Label describing the relationship (e.g., ``"mentions"``,
        ``"belongs_to"``, ``"related_to"``).
    weight
        Numeric weight/strength of the relationship (0.0–1.0).
    metadata
        Arbitrary key-value metadata.

    Examples
    --------
    ```python
    import talk_box as tb

    edge = tb.Edge(
        source="doc-001",
        target="entity-python",
        relation="mentions",
        weight=0.9,
    )
    edge.relation  # "mentions"
    ```
    """

    source: str
    target: str
    relation: str
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# SQL schema
# ---------------------------------------------------------------------------

_CREATE_NODES_SQL = """
CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    node_type TEXT NOT NULL,
    name TEXT NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    metadata TEXT NOT NULL DEFAULT '{}',
    embedding BLOB,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
)
"""

_CREATE_EDGES_SQL = """
CREATE TABLE IF NOT EXISTS edges (
    source TEXT NOT NULL,
    target TEXT NOT NULL,
    relation TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    metadata TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (source, target, relation),
    FOREIGN KEY (source) REFERENCES nodes(id) ON DELETE CASCADE,
    FOREIGN KEY (target) REFERENCES nodes(id) ON DELETE CASCADE
)
"""

_CREATE_NODES_TYPE_IDX = """
CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes (node_type)
"""

_CREATE_NODES_NAME_IDX = """
CREATE INDEX IF NOT EXISTS idx_nodes_name ON nodes (name)
"""

_CREATE_EDGES_SOURCE_IDX = """
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges (source)
"""

_CREATE_EDGES_TARGET_IDX = """
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges (target)
"""

_CREATE_EDGES_RELATION_IDX = """
CREATE INDEX IF NOT EXISTS idx_edges_relation ON edges (relation)
"""


# ---------------------------------------------------------------------------
# KnowledgeGraph
# ---------------------------------------------------------------------------


class KnowledgeGraph:
    """SQLite-backed knowledge graph with nodes, edges, and optional embeddings.

    Stores documents, entities, and topics as nodes connected by typed,
    weighted edges.  Supports text search, neighbor traversal, and
    optional vector embeddings for similarity search.

    Parameters
    ----------
    path
        Path to the SQLite database file.  Use ``":memory:"`` for an
        in-memory graph (useful for testing).

    Examples
    --------
    Create a graph and add nodes:

    ```python
    import talk_box as tb

    kg = tb.KnowledgeGraph(":memory:")

    kg.add_node(tb.Node(
        id="doc-1",
        node_type=tb.NodeType.DOCUMENT,
        name="README.md",
        content="# Talk Box\\nAI assistant framework.",
    ))

    kg.add_node(tb.Node(
        id="entity-tb",
        node_type=tb.NodeType.ENTITY,
        name="Talk Box",
    ))

    kg.add_edge(tb.Edge(
        source="doc-1",
        target="entity-tb",
        relation="mentions",
    ))

    neighbors = kg.neighbors("doc-1")
    neighbors[0].name  # "Talk Box"
    ```

    Search nodes by text:

    ```python
    results = kg.search("Talk Box")
    results[0].name  # "Talk Box"
    ```
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._path = str(path)
        self._conn = sqlite3.connect(self._path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self) -> None:
        """Create tables and indexes if they don't exist."""
        self._conn.execute(_CREATE_NODES_SQL)
        self._conn.execute(_CREATE_EDGES_SQL)
        self._conn.execute(_CREATE_NODES_TYPE_IDX)
        self._conn.execute(_CREATE_NODES_NAME_IDX)
        self._conn.execute(_CREATE_EDGES_SOURCE_IDX)
        self._conn.execute(_CREATE_EDGES_TARGET_IDX)
        self._conn.execute(_CREATE_EDGES_RELATION_IDX)
        self._conn.commit()

    @property
    def path(self) -> str:
        """Path to the SQLite database file."""
        return self._path

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def add_node(self, node: Node) -> None:
        """Add a node to the graph, or update it if the ID already exists.

        Parameters
        ----------
        node
            The node to add or update.

        Examples
        --------
        ```python
        kg.add_node(tb.Node(
            id="topic-ml",
            node_type=tb.NodeType.TOPIC,
            name="Machine Learning",
        ))
        ```
        """
        embedding_blob = _floats_to_blob(node.embedding) if node.embedding is not None else None
        self._conn.execute(
            """
            INSERT INTO nodes (id, node_type, name, content, metadata, embedding, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                node_type=excluded.node_type,
                name=excluded.name,
                content=excluded.content,
                metadata=excluded.metadata,
                embedding=excluded.embedding,
                updated_at=excluded.updated_at
            """,
            (
                node.id,
                node.node_type.value,
                node.name,
                node.content,
                json.dumps(node.metadata),
                embedding_blob,
                node.created_at,
                node.updated_at,
            ),
        )
        self._conn.commit()

    def get_node(self, node_id: str) -> Node | None:
        """Get a node by ID.

        Parameters
        ----------
        node_id
            The node identifier.

        Returns
        -------
        Node | None
            The node, or ``None`` if not found.
        """
        row = self._conn.execute(
            "SELECT id, node_type, name, content, metadata, embedding, created_at, updated_at "
            "FROM nodes WHERE id = ?",
            (node_id,),
        ).fetchone()
        if row is None:
            return None
        return _row_to_node(row)

    def delete_node(self, node_id: str) -> bool:
        """Delete a node and all its edges.

        Parameters
        ----------
        node_id
            The node to remove.

        Returns
        -------
        bool
            ``True`` if the node existed and was deleted.
        """
        cursor = self._conn.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    def list_nodes(
        self,
        *,
        node_type: NodeType | None = None,
        limit: int = 100,
    ) -> list[Node]:
        """List nodes, optionally filtered by type.

        Parameters
        ----------
        node_type
            Filter to a specific node type.
        limit
            Maximum number of nodes to return.

        Returns
        -------
        list[Node]
            Matching nodes ordered by name.
        """
        if node_type is not None:
            rows = self._conn.execute(
                "SELECT id, node_type, name, content, metadata, embedding, created_at, updated_at "
                "FROM nodes WHERE node_type = ? ORDER BY name LIMIT ?",
                (node_type.value, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id, node_type, name, content, metadata, embedding, created_at, updated_at "
                "FROM nodes ORDER BY name LIMIT ?",
                (limit,),
            ).fetchall()
        return [_row_to_node(r) for r in rows]

    def node_count(self, *, node_type: NodeType | None = None) -> int:
        """Count nodes in the graph.

        Parameters
        ----------
        node_type
            If provided, count only nodes of this type.

        Returns
        -------
        int
            Number of matching nodes.
        """
        if node_type is not None:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM nodes WHERE node_type = ?",
                (node_type.value,),
            ).fetchone()
        else:
            row = self._conn.execute("SELECT COUNT(*) FROM nodes").fetchone()
        return row[0] if row else 0

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph, or update it if the key already exists.

        Parameters
        ----------
        edge
            The edge to add or update. The (source, target, relation) triple
            is the primary key.

        Raises
        ------
        KeyError
            If source or target node does not exist.
        """
        # Verify both nodes exist
        for nid in (edge.source, edge.target):
            if self.get_node(nid) is None:
                raise KeyError(f"Node '{nid}' not found in graph")

        self._conn.execute(
            """
            INSERT INTO edges (source, target, relation, weight, metadata)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(source, target, relation) DO UPDATE SET
                weight=excluded.weight,
                metadata=excluded.metadata
            """,
            (
                edge.source,
                edge.target,
                edge.relation,
                edge.weight,
                json.dumps(edge.metadata),
            ),
        )
        self._conn.commit()

    def get_edges(
        self,
        node_id: str,
        *,
        direction: str = "both",
        relation: str | None = None,
    ) -> list[Edge]:
        """Get edges connected to a node.

        Parameters
        ----------
        node_id
            The node to query edges for.
        direction
            ``"outgoing"``, ``"incoming"``, or ``"both"`` (default).
        relation
            Optional filter by relation type.

        Returns
        -------
        list[Edge]
            Matching edges.
        """
        edges: list[Edge] = []
        params: list[Any] = []

        if direction in ("outgoing", "both"):
            sql = "SELECT source, target, relation, weight, metadata FROM edges WHERE source = ?"
            p: list[Any] = [node_id]
            if relation is not None:
                sql += " AND relation = ?"
                p.append(relation)
            rows = self._conn.execute(sql, p).fetchall()
            edges.extend(_row_to_edge(r) for r in rows)

        if direction in ("incoming", "both"):
            sql = "SELECT source, target, relation, weight, metadata FROM edges WHERE target = ?"
            p = [node_id]
            if relation is not None:
                sql += " AND relation = ?"
                p.append(relation)
            rows = self._conn.execute(sql, p).fetchall()
            edges.extend(_row_to_edge(r) for r in rows)

        return edges

    def delete_edge(self, source: str, target: str, relation: str) -> bool:
        """Delete a specific edge.

        Parameters
        ----------
        source
            Source node ID.
        target
            Target node ID.
        relation
            The relation label.

        Returns
        -------
        bool
            ``True`` if the edge existed and was deleted.
        """
        cursor = self._conn.execute(
            "DELETE FROM edges WHERE source = ? AND target = ? AND relation = ?",
            (source, target, relation),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def edge_count(self) -> int:
        """Count edges in the graph.

        Returns
        -------
        int
            Total number of edges.
        """
        row = self._conn.execute("SELECT COUNT(*) FROM edges").fetchone()
        return row[0] if row else 0

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    def neighbors(
        self,
        node_id: str,
        *,
        relation: str | None = None,
        direction: str = "outgoing",
    ) -> list[Node]:
        """Get neighboring nodes connected by edges.

        Parameters
        ----------
        node_id
            The starting node.
        relation
            Optional filter by relation type.
        direction
            ``"outgoing"`` (default), ``"incoming"``, or ``"both"``.

        Returns
        -------
        list[Node]
            Connected nodes.
        """
        edges = self.get_edges(node_id, direction=direction, relation=relation)
        neighbor_ids: list[str] = []
        seen: set[str] = set()

        for edge in edges:
            nid = edge.target if edge.source == node_id else edge.source
            if nid not in seen:
                neighbor_ids.append(nid)
                seen.add(nid)

        nodes: list[Node] = []
        for nid in neighbor_ids:
            node = self.get_node(nid)
            if node is not None:
                nodes.append(node)
        return nodes

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        node_type: NodeType | None = None,
        limit: int = 20,
    ) -> list[Node]:
        """Search nodes by name or content (case-insensitive substring match).

        Parameters
        ----------
        query
            Text to search for in node names and content.
        node_type
            Optional filter by node type.
        limit
            Maximum results to return.

        Returns
        -------
        list[Node]
            Matching nodes, name matches ranked first.

        Examples
        --------
        ```python
        results = kg.search("python", node_type=tb.NodeType.ENTITY)
        ```
        """
        pattern = f"%{query}%"
        params: list[Any] = [pattern, pattern]
        type_clause = ""
        if node_type is not None:
            type_clause = "AND node_type = ?"
            params.append(node_type.value)
        params.append(limit)

        sql = f"""
            SELECT id, node_type, name, content, metadata, embedding, created_at, updated_at
            FROM nodes
            WHERE (name LIKE ? OR content LIKE ?) {type_clause}
            ORDER BY
                CASE WHEN name LIKE ? THEN 0 ELSE 1 END,
                name
            LIMIT ?
        """
        # Need an extra pattern param for the ORDER BY CASE
        params.insert(-1, pattern)

        rows = self._conn.execute(sql, params).fetchall()
        return [_row_to_node(r) for r in rows]

    # ------------------------------------------------------------------
    # Stats / health
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Get summary statistics about the graph.

        Returns
        -------
        dict[str, Any]
            Counts of nodes (total and per type) and edges.

        Examples
        --------
        ```python
        kg.stats()
        # {"nodes": 42, "edges": 87, "documents": 10, "entities": 25, "topics": 7}
        ```
        """
        total_nodes = self.node_count()
        total_edges = self.edge_count()
        docs = self.node_count(node_type=NodeType.DOCUMENT)
        entities = self.node_count(node_type=NodeType.ENTITY)
        topics = self.node_count(node_type=NodeType.TOPIC)
        return {
            "nodes": total_nodes,
            "edges": total_edges,
            "documents": docs,
            "entities": entities,
            "topics": topics,
        }

    def health(self) -> dict[str, Any]:
        """Get health metrics for the graph.

        Reports orphan nodes (no edges), isolated clusters, and
        embedding coverage.

        Returns
        -------
        dict[str, Any]
            Health metrics including orphan count and embedding coverage.

        Examples
        --------
        ```python
        report = kg.health()
        report["orphan_nodes"]       # 3
        report["embedding_coverage"] # 0.85
        ```
        """
        total = self.node_count()
        if total == 0:
            return {
                "orphan_nodes": 0,
                "connected_nodes": 0,
                "embedding_coverage": 0.0,
                "total_nodes": 0,
                "total_edges": 0,
            }

        # Nodes with at least one edge
        row = self._conn.execute(
            """
            SELECT COUNT(DISTINCT id) FROM nodes
            WHERE id IN (
                SELECT source FROM edges
                UNION
                SELECT target FROM edges
            )
            """
        ).fetchone()
        connected = row[0] if row else 0
        orphans = total - connected

        # Embedding coverage
        row = self._conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE embedding IS NOT NULL"
        ).fetchone()
        with_embedding = row[0] if row else 0
        coverage = with_embedding / total if total > 0 else 0.0

        return {
            "orphan_nodes": orphans,
            "connected_nodes": connected,
            "embedding_coverage": round(coverage, 4),
            "total_nodes": total,
            "total_edges": self.edge_count(),
        }

    # ------------------------------------------------------------------
    # Enrichment Q&A
    # ------------------------------------------------------------------

    def pending_questions(
        self,
        *,
        refresh: bool = True,
        sort_by: str = "confusion_impact",
        limit: int | None = None,
    ) -> list[Any]:
        """Get pending enrichment questions for this knowledge graph.

        Detects ambiguities (duplicate names, factual conflicts, weak
        relationships) and returns structured questions sorted by
        confusion impact.

        Parameters
        ----------
        refresh
            If ``True`` (default), run built-in detectors to find new
            questions before returning.
        sort_by
            Sort field: ``"confusion_impact"`` (default, descending) or
            ``"created_at"`` (ascending, oldest first).
        limit
            Maximum questions to return. Defaults to the queue's
            ``max_per_session`` (7).

        Returns
        -------
        list[EnrichmentQuestion]
            Pending questions sorted by priority.

        Examples
        --------
        ```python
        import talk_box as tb

        kg = tb.KnowledgeGraph(":memory:")
        # ... add nodes and edges ...
        questions = kg.pending_questions()
        for q in questions:
            print(f"[{q.confusion_impact:.2f}] {q.text}")
        ```

        Skip detection and return only existing questions:

        ```python
        questions = kg.pending_questions(refresh=False)
        ```
        """
        from talk_box.enrichment_qa import pending_questions as _pending

        return _pending(
            self,
            self._get_question_queue(),
            refresh=refresh,
            sort_by=sort_by,
            limit=limit,
        )

    def answer_question(
        self,
        question_id: str,
        *,
        choice: int | None = None,
        freeform: str | None = None,
    ) -> Any | None:
        """Answer a pending enrichment question.

        At least one of ``choice`` or ``freeform`` must be provided.

        Parameters
        ----------
        question_id
            ID of the question to answer.
        choice
            Index of the selected option.
        freeform
            Freeform text answer.

        Returns
        -------
        EnrichmentQuestion | None
            The updated question, or ``None`` if not found or not
            pending.

        Raises
        ------
        ValueError
            If neither ``choice`` nor ``freeform`` is provided, or
            ``choice`` is out of range.

        Examples
        --------
        ```python
        questions = kg.pending_questions()
        kg.answer_question(questions[0].id, choice=0)
        kg.answer_question(questions[1].id, freeform="It's the API migration")
        ```
        """
        return self._get_question_queue().answer(question_id, choice=choice, freeform=freeform)

    def dismiss_question(self, question_id: str) -> Any | None:
        """Dismiss a pending enrichment question without answering.

        Parameters
        ----------
        question_id
            ID of the question to dismiss.

        Returns
        -------
        EnrichmentQuestion | None
            The dismissed question, or ``None`` if not found.

        Examples
        --------
        ```python
        kg.dismiss_question("eq-a1b2c3d4")
        ```
        """
        return self._get_question_queue().dismiss(question_id)

    def question_stats(self) -> dict[str, int]:
        """Get summary statistics for the enrichment question queue.

        Returns
        -------
        dict[str, int]
            Counts by status (total, pending, answered, dismissed,
            expired).

        Examples
        --------
        ```python
        kg.question_stats()
        # {"total": 5, "pending": 3, "answered": 1, ...}
        ```
        """
        return self._get_question_queue().stats().to_dict()

    def _get_question_queue(self) -> Any:
        """Lazily create the question queue."""
        if not hasattr(self, "_question_queue"):
            from talk_box.enrichment_qa import QuestionQueue

            self._question_queue = QuestionQueue()
        return self._question_queue

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Delete all nodes and edges from the graph."""
        self._conn.execute("DELETE FROM edges")
        self._conn.execute("DELETE FROM nodes")
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __repr__(self) -> str:
        s = self.stats()
        return f"KnowledgeGraph(path={self._path!r}, nodes={s['nodes']}, edges={s['edges']})"


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

import struct  # noqa: E402


def _floats_to_blob(floats: list[float]) -> bytes:
    """Pack a list of floats into a compact binary blob."""
    return struct.pack(f"{len(floats)}f", *floats)


def _blob_to_floats(blob: bytes) -> list[float]:
    """Unpack a binary blob into a list of floats."""
    count = len(blob) // 4
    return list(struct.unpack(f"{count}f", blob))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Parameters
    ----------
    a
        First vector.
    b
        Second vector.

    Returns
    -------
    float
        Cosine similarity in [-1, 1].

    Examples
    --------
    ```python
    import talk_box as tb

    tb.cosine_similarity([1, 0, 0], [1, 0, 0])  # 1.0
    tb.cosine_similarity([1, 0, 0], [0, 1, 0])  # 0.0
    ```
    """
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} vs {len(b)}")

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Row conversion helpers
# ---------------------------------------------------------------------------


def _row_to_node(row: tuple[Any, ...]) -> Node:
    """Convert a database row to a Node."""
    embedding = _blob_to_floats(row[5]) if row[5] is not None else None
    return Node(
        id=row[0],
        node_type=NodeType(row[1]),
        name=row[2],
        content=row[3],
        metadata=json.loads(row[4]),
        embedding=embedding,
        created_at=row[6],
        updated_at=row[7],
    )


def _row_to_edge(row: tuple[Any, ...]) -> Edge:
    """Convert a database row to an Edge."""
    return Edge(
        source=row[0],
        target=row[1],
        relation=row[2],
        weight=row[3],
        metadata=json.loads(row[4]),
    )
