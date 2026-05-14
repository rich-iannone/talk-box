"""Knowledge graph core: SQLite-backed graph with nodes, edges, and optional embeddings."""

from __future__ import annotations

import json
import sqlite3
import time
import warnings
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
    DECISION
        An AI reasoning event: enrichment result, Q&A answer, user correction,
        or chat-derived insight.  Decision nodes form an auditable trail and
        can be reverted.

    Examples
    --------
    ```python
    import talk_box as tb

    tb.NodeType.DOCUMENT
    tb.NodeType.ENTITY
    tb.NodeType.TOPIC
    tb.NodeType.DECISION
    ```
    """

    DOCUMENT = "document"
    ENTITY = "entity"
    TOPIC = "topic"
    DECISION = "decision"


class GraphLayer(Enum):
    """Layer that owns a node or edge.

    Values
    ------
    BASE
        User-curated source material (documents, manual entities/edges).
    ENRICHMENT
        Model-generated content (extracted entities, topics, summaries).
    EXTENDED
        Conversational learnings (decisions, corrections, chat insights).

    Examples
    --------
    ```python
    import talk_box as tb

    tb.GraphLayer.BASE
    tb.GraphLayer.ENRICHMENT
    tb.GraphLayer.EXTENDED
    ```
    """

    BASE = "base"
    ENRICHMENT = "enrichment"
    EXTENDED = "extended"


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
    layer: GraphLayer = GraphLayer.BASE

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
    layer: GraphLayer = GraphLayer.BASE


# ---------------------------------------------------------------------------
# Ontology
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EntityTypeDef:
    """Definition of an entity type within an ontology.

    Parameters
    ----------
    description
        Human-readable description (shown to LLMs for semantic grounding).
    properties
        Allowed metadata keys for entities of this type.
    parent
        Optional parent type for inheritance (e.g. ``"project"`` → ``"initiative"``).

    Examples
    --------
    ```python
    import talk_box as tb

    person = tb.EntityTypeDef(
        description="A human individual",
        properties=["role", "team", "email"],
    )
    ```
    """

    description: str = ""
    properties: list[str] = field(default_factory=list)
    parent: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"description": self.description, "properties": self.properties}
        if self.parent is not None:
            d["parent"] = self.parent
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EntityTypeDef:
        return cls(
            description=d.get("description", ""),
            properties=d.get("properties", []),
            parent=d.get("parent"),
        )


@dataclass(frozen=True)
class RelationTypeDef:
    """Definition of a relation type within an ontology.

    Parameters
    ----------
    description
        Human-readable description.
    source_type
        Required entity type for the source node (``None`` = any).
    target_type
        Required entity type for the target node (``None`` = any).
    properties
        Allowed metadata keys for edges of this type.

    Examples
    --------
    ```python
    import talk_box as tb

    leads = tb.RelationTypeDef(
        description="Person is the lead of a project",
        source_type="person",
        target_type="project",
    )
    ```
    """

    description: str = ""
    source_type: str | None = None
    target_type: str | None = None
    properties: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"description": self.description, "properties": self.properties}
        if self.source_type is not None:
            d["source_type"] = self.source_type
        if self.target_type is not None:
            d["target_type"] = self.target_type
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RelationTypeDef:
        return cls(
            description=d.get("description", ""),
            source_type=d.get("source_type"),
            target_type=d.get("target_type"),
            properties=d.get("properties", []),
        )


@dataclass
class Ontology:
    """Lightweight type system for a knowledge graph.

    Provides semantic typing for entities and relationships. The ontology
    is stored per-graph and used for optional validation in
    ``add_node()``/``add_edge()``.

    Parameters
    ----------
    entity_types
        Mapping of type name → definition.
    relation_types
        Mapping of relation name → definition.

    Examples
    --------
    ```python
    import talk_box as tb

    ontology = tb.Ontology(
        entity_types={
            "person": tb.EntityTypeDef(description="A human individual", properties=["role"]),
        },
        relation_types={
            "leads": tb.RelationTypeDef(
                description="Person leads a project",
                source_type="person",
                target_type="project",
            ),
        },
    )
    ```
    """

    entity_types: dict[str, EntityTypeDef] = field(default_factory=dict)
    relation_types: dict[str, RelationTypeDef] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Ancestry / inheritance helpers
    # ------------------------------------------------------------------

    def ancestors(self, entity_type: str) -> list[str]:
        """Return the parent chain for *entity_type* (excluding itself).

        Examples
        --------
        ```python
        ontology.ancestors("project")  # ["initiative"] if project.parent == "initiative"
        ```
        """
        chain: list[str] = []
        current = entity_type
        visited: set[str] = {current}
        while True:
            etd = self.entity_types.get(current)
            if etd is None or etd.parent is None:
                break
            if etd.parent in visited:
                break  # prevent cycles
            chain.append(etd.parent)
            visited.add(etd.parent)
            current = etd.parent
        return chain

    def is_subtype(self, child: str, parent: str) -> bool:
        """Check whether *child* is the same as or a subtype of *parent*.

        Examples
        --------
        ```python
        ontology.is_subtype("project", "initiative")  # True if project inherits from initiative
        ```
        """
        if child == parent:
            return True
        return parent in self.ancestors(child)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_node(self, node: Node) -> list[str]:
        """Validate a node against the ontology.

        Only ENTITY nodes are validated (documents, topics, and decisions
        are always accepted).

        Returns
        -------
        list[str]
            Warning messages.  Empty list means valid.
        """
        msgs: list[str] = []
        if node.node_type != NodeType.ENTITY:
            return msgs
        entity_type = node.metadata.get("entity_type", "")
        if not entity_type:
            return msgs  # no entity_type metadata → nothing to validate
        if self.entity_types and entity_type not in self.entity_types:
            msgs.append(
                f"Entity type '{entity_type}' not in ontology "
                f"(known: {', '.join(sorted(self.entity_types))})"
            )
        etd = self.entity_types.get(entity_type)
        if etd is not None and etd.properties:
            for key in node.metadata:
                if key.startswith("_") or key == "entity_type":
                    continue
                if key not in etd.properties:
                    msgs.append(
                        f"Metadata key '{key}' not in allowed properties "
                        f"for entity type '{entity_type}'"
                    )
        return msgs

    def validate_edge(
        self, edge: Edge, *, source_node: Node | None = None, target_node: Node | None = None
    ) -> list[str]:
        """Validate an edge against the ontology.

        Parameters
        ----------
        edge
            The edge to validate.
        source_node
            The source node (used to check type constraints).
        target_node
            The target node (used to check type constraints).

        Returns
        -------
        list[str]
            Warning messages.  Empty list means valid.
        """
        msgs: list[str] = []
        if not self.relation_types:
            return msgs
        rtd = self.relation_types.get(edge.relation)
        if rtd is None:
            # Unknown relation — not an error, just not typed
            return msgs
        if rtd.source_type is not None and source_node is not None:
            src_et = source_node.metadata.get("entity_type", "")
            if src_et and not self.is_subtype(src_et, rtd.source_type):
                msgs.append(
                    f"Relation '{edge.relation}' expects source type "
                    f"'{rtd.source_type}', got '{src_et}'"
                )
        if rtd.target_type is not None and target_node is not None:
            tgt_et = target_node.metadata.get("entity_type", "")
            if tgt_et and not self.is_subtype(tgt_et, rtd.target_type):
                msgs.append(
                    f"Relation '{edge.relation}' expects target type "
                    f"'{rtd.target_type}', got '{tgt_et}'"
                )
        return msgs

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the ontology to a dictionary."""
        return {
            "entity_types": {k: v.to_dict() for k, v in self.entity_types.items()},
            "relation_types": {k: v.to_dict() for k, v in self.relation_types.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Ontology:
        """Deserialize an ontology from a dictionary."""
        return cls(
            entity_types={
                k: EntityTypeDef.from_dict(v) for k, v in d.get("entity_types", {}).items()
            },
            relation_types={
                k: RelationTypeDef.from_dict(v) for k, v in d.get("relation_types", {}).items()
            },
        )


def general_ontology() -> Ontology:
    """Return a starter ontology with common entity and relation types.

    Provides a reasonable default for general-purpose knowledge graphs.

    Examples
    --------
    ```python
    import talk_box as tb

    ontology = tb.general_ontology()
    "person" in ontology.entity_types  # True
    ```
    """
    return Ontology(
        entity_types={
            "person": EntityTypeDef(
                description="A human individual",
                properties=["role", "team", "email", "title"],
            ),
            "organization": EntityTypeDef(
                description="A company, team, or institutional body",
                properties=["industry", "size", "location"],
            ),
            "project": EntityTypeDef(
                description="A planned initiative with a timeline and deliverables",
                properties=["status", "start_date", "end_date"],
            ),
            "concept": EntityTypeDef(
                description="An abstract idea, theory, or domain term",
                properties=["domain", "definition"],
            ),
            "event": EntityTypeDef(
                description="A discrete occurrence at a point or range in time",
                properties=["date", "location", "duration"],
            ),
            "location": EntityTypeDef(
                description="A physical or logical place",
                properties=["coordinates", "region", "type"],
            ),
            "technology": EntityTypeDef(
                description="A software tool, language, framework, or platform",
                properties=["category", "version", "url"],
            ),
            "metric": EntityTypeDef(
                description="A measurable business or technical quantity",
                properties=["unit", "direction"],
            ),
        },
        relation_types={
            "leads": RelationTypeDef(
                description="Person is the lead/owner of a project",
                source_type="person",
                target_type="project",
            ),
            "works_at": RelationTypeDef(
                description="Person is employed by an organization",
                source_type="person",
                target_type="organization",
            ),
            "works_on": RelationTypeDef(
                description="Person contributes to a project",
                source_type="person",
                target_type="project",
            ),
            "located_in": RelationTypeDef(
                description="Entity is situated in a location",
                target_type="location",
            ),
            "related_to": RelationTypeDef(
                description="General semantic relationship between any two entities",
            ),
            "drives": RelationTypeDef(
                description="One metric causally influences another",
                source_type="metric",
                target_type="metric",
                properties=["strength", "lag_days"],
            ),
            "part_of": RelationTypeDef(
                description="Entity is a component of another entity",
            ),
        },
    )


# ---------------------------------------------------------------------------
# Subgraph
# ---------------------------------------------------------------------------


@dataclass
class Subgraph:
    """A slice of a knowledge graph returned by :meth:`KnowledgeGraph.extract_subgraph`.

    Contains the matched/expanded nodes, connecting edges, the original
    query, and which node IDs were direct search hits (seeds).

    Parameters
    ----------
    nodes
        Nodes included in the subgraph.
    edges
        Edges whose *both* endpoints are in ``nodes``.
    seed_ids
        IDs of the initial search-hit nodes (before expansion).
    query
        The original search query.

    Examples
    --------
    ```python
    sg = kg.extract_subgraph("revenue")
    sg.to_context()
    ```
    """

    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    seed_ids: list[str] = field(default_factory=list)
    query: str = ""

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_context(
        self,
        *,
        format: str = "typed",
        max_tokens: int | None = None,
        ontology: Ontology | None = None,
    ) -> str:
        """Serialise the subgraph into an LLM-readable text block.

        Parameters
        ----------
        format
            ``"typed"`` (default) produces a structured block with entity
            types and relationships.  ``"plain"`` produces a simpler flat
            list.
        max_tokens
            Approximate token budget (4 chars ≈ 1 token).  ``None`` means
            no limit.
        ontology
            If provided, type descriptions are prepended so the LLM
            understands domain semantics.

        Returns
        -------
        str
            A text block suitable for injection into an LLM prompt.

        Examples
        --------
        ```python
        context = sg.to_context(max_tokens=2000)
        ```
        """
        if format == "plain":
            return self._to_plain(max_tokens=max_tokens)
        return self._to_typed(max_tokens=max_tokens, ontology=ontology)

    def _to_typed(
        self,
        *,
        max_tokens: int | None = None,
        ontology: Ontology | None = None,
    ) -> str:
        sections: list[str] = ["[Knowledge Context]"]

        # Types section (from ontology)
        if ontology is not None:
            entity_types_seen: set[str] = set()
            for node in self.nodes:
                et = node.metadata.get("entity_type", "")
                if et and et in ontology.entity_types:
                    entity_types_seen.add(et)
            if entity_types_seen:
                type_parts = []
                for et in sorted(entity_types_seen):
                    desc = ontology.entity_types[et].description
                    type_parts.append(f"{et} ({desc})" if desc else et)
                sections.append(f"Types: {', '.join(type_parts)}")

        # Entities section (non-document nodes)
        entity_lines: list[str] = []
        doc_lines: list[str] = []
        for node in self.nodes:
            if node.node_type == NodeType.DOCUMENT:
                preview = node.content[:200].replace("\n", " ").strip()
                doc_lines.append(f'- "{node.name}" (doc): "{preview}"')
            elif node.node_type in (NodeType.ENTITY, NodeType.TOPIC):
                et = node.metadata.get("entity_type", node.node_type.value)
                props = {
                    k: v
                    for k, v in node.metadata.items()
                    if not k.startswith("_") and k != "entity_type"
                }
                prop_str = ", ".join(f"{k}={v}" for k, v in props.items())
                label = f"- {node.name} [{et}]"
                if prop_str:
                    label += f": {prop_str}"
                entity_lines.append(label)

        if entity_lines:
            sections.append("")
            sections.append("Entities:")
            sections.extend(entity_lines)

        # Relationships section
        if self.edges:
            rel_lines: list[str] = []
            node_map = {n.id: n for n in self.nodes}
            for edge in self.edges:
                src_name = node_map[edge.source].name if edge.source in node_map else edge.source
                tgt_name = node_map[edge.target].name if edge.target in node_map else edge.target
                meta_parts = {k: v for k, v in edge.metadata.items() if not k.startswith("_")}
                meta_str = (
                    f" ({', '.join(f'{k}={v}' for k, v in meta_parts.items())})"
                    if meta_parts
                    else ""
                )
                rel_lines.append(f"- {src_name} --[{edge.relation}]--> {tgt_name}{meta_str}")
            sections.append("")
            sections.append("Relationships:")
            sections.extend(rel_lines)

        # Sources section (documents)
        if doc_lines:
            sections.append("")
            sections.append("Sources:")
            sections.extend(doc_lines)

        text = "\n".join(sections)

        if max_tokens is not None:
            char_budget = max_tokens * 4
            if len(text) > char_budget:
                text = text[:char_budget].rsplit("\n", 1)[0] + "\n..."

        return text

    def _to_plain(self, *, max_tokens: int | None = None) -> str:
        lines: list[str] = [f"[Knowledge Context: {self.query}]"]
        for node in self.nodes:
            preview = node.content[:200].replace("\n", " ").strip() if node.content else ""
            line = f"- {node.name} ({node.node_type.value})"
            if preview:
                line += f": {preview}"
            lines.append(line)
        text = "\n".join(lines)
        if max_tokens is not None:
            char_budget = max_tokens * 4
            if len(text) > char_budget:
                text = text[:char_budget].rsplit("\n", 1)[0] + "\n..."
        return text


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
    updated_at REAL NOT NULL,
    layer TEXT NOT NULL DEFAULT 'base'
)
"""

_CREATE_EDGES_SQL = """
CREATE TABLE IF NOT EXISTS edges (
    source TEXT NOT NULL,
    target TEXT NOT NULL,
    relation TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    metadata TEXT NOT NULL DEFAULT '{}',
    layer TEXT NOT NULL DEFAULT 'base',
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

_CREATE_NODES_LAYER_IDX = """
CREATE INDEX IF NOT EXISTS idx_nodes_layer ON nodes (layer)
"""

_CREATE_EDGES_LAYER_IDX = """
CREATE INDEX IF NOT EXISTS idx_edges_layer ON edges (layer)
"""

_CREATE_META_SQL = """
CREATE TABLE IF NOT EXISTS _meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL DEFAULT '{}'
)
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

    def __init__(
        self,
        path: str | Path = ":memory:",
        *,
        name: str = "",
        ontology: Ontology | None = None,
        strict: bool = False,
    ) -> None:
        self._path = str(path)
        self._name = name
        self._strict = strict
        self._conn = sqlite3.connect(self._path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()
        # Load or store ontology
        if ontology is not None:
            self._ontology = ontology
            self._save_ontology()
        else:
            self._ontology = self._load_ontology()

    @property
    def name(self) -> str:
        """Human-readable name for this graph (empty string if unnamed)."""
        return self._name

    def _init_schema(self) -> None:
        """Create tables and indexes if they don't exist."""
        self._conn.execute(_CREATE_NODES_SQL)
        self._conn.execute(_CREATE_EDGES_SQL)
        self._conn.execute(_CREATE_META_SQL)
        self._migrate_add_layer()
        self._conn.execute(_CREATE_NODES_TYPE_IDX)
        self._conn.execute(_CREATE_NODES_NAME_IDX)
        self._conn.execute(_CREATE_EDGES_SOURCE_IDX)
        self._conn.execute(_CREATE_EDGES_TARGET_IDX)
        self._conn.execute(_CREATE_EDGES_RELATION_IDX)
        self._conn.execute(_CREATE_NODES_LAYER_IDX)
        self._conn.execute(_CREATE_EDGES_LAYER_IDX)
        self._conn.commit()

    def _migrate_add_layer(self) -> None:
        """Add layer column to existing databases that lack it."""
        cur = self._conn.execute("PRAGMA table_info(nodes)")
        node_cols = {row[1] for row in cur.fetchall()}
        if "layer" not in node_cols:
            self._conn.execute("ALTER TABLE nodes ADD COLUMN layer TEXT NOT NULL DEFAULT 'base'")
        cur = self._conn.execute("PRAGMA table_info(edges)")
        edge_cols = {row[1] for row in cur.fetchall()}
        if "layer" not in edge_cols:
            self._conn.execute("ALTER TABLE edges ADD COLUMN layer TEXT NOT NULL DEFAULT 'base'")

    # ------------------------------------------------------------------
    # Ontology
    # ------------------------------------------------------------------

    @property
    def ontology(self) -> Ontology:
        """The ontology associated with this graph."""
        return self._ontology

    def set_ontology(self, ontology: Ontology) -> None:
        """Replace the graph's ontology.

        Parameters
        ----------
        ontology
            The new ontology to set.

        Examples
        --------
        ```python
        kg.set_ontology(tb.general_ontology())
        ```
        """
        self._ontology = ontology
        self._save_ontology()

    def _save_ontology(self) -> None:
        """Persist the current ontology to the _meta table."""
        data = json.dumps(self._ontology.to_dict())
        self._conn.execute(
            "INSERT INTO _meta (key, value) VALUES ('ontology', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (data,),
        )
        self._conn.commit()

    def _load_ontology(self) -> Ontology:
        """Load ontology from the _meta table, or return empty ontology."""
        row = self._conn.execute("SELECT value FROM _meta WHERE key = 'ontology'").fetchone()
        if row is None:
            return Ontology()
        return Ontology.from_dict(json.loads(row[0]))

    def _validate_node(self, node: Node) -> None:
        """Run ontology validation on a node; warn or raise."""
        if not self._ontology.entity_types and not self._ontology.relation_types:
            return
        msgs = self._ontology.validate_node(node)
        if msgs:
            combined = "; ".join(msgs)
            if self._strict:
                raise ValueError(combined)
            warnings.warn(combined, stacklevel=3)

    def _validate_edge(
        self,
        edge: Edge,
        *,
        source_node: Node | None = None,
        target_node: Node | None = None,
    ) -> None:
        """Run ontology validation on an edge; warn or raise."""
        if not self._ontology.relation_types:
            return
        msgs = self._ontology.validate_edge(edge, source_node=source_node, target_node=target_node)
        if msgs:
            combined = "; ".join(msgs)
            if self._strict:
                raise ValueError(combined)
            warnings.warn(combined, stacklevel=3)

    @property
    def path(self) -> str:
        """Path to the SQLite database file."""
        return self._path

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def add_node(self, node: Node) -> None:
        """Add a node to the graph, or update it if the ID already exists.

        If the graph has an ontology, the node is validated first.
        In strict mode a :class:`ValueError` is raised on violations;
        otherwise a warning is emitted.

        Parameters
        ----------
        node
            The node to add or update.

        Raises
        ------
        ValueError
            If validation fails and ``strict=True``.

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
        self._validate_node(node)
        embedding_blob = _floats_to_blob(node.embedding) if node.embedding is not None else None
        self._conn.execute(
            """
            INSERT INTO nodes (id, node_type, name, content, metadata, embedding, created_at, updated_at, layer)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                node_type=excluded.node_type,
                name=excluded.name,
                content=excluded.content,
                metadata=excluded.metadata,
                embedding=excluded.embedding,
                updated_at=excluded.updated_at,
                layer=excluded.layer
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
                node.layer.value,
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
            "SELECT id, node_type, name, content, metadata, embedding, created_at, updated_at, layer "
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
        layer: GraphLayer | None = None,
        limit: int = 100,
    ) -> list[Node]:
        """List nodes, optionally filtered by type and/or layer.

        Parameters
        ----------
        node_type
            Filter to a specific node type.
        layer
            Filter to a specific graph layer.
        limit
            Maximum number of nodes to return.

        Returns
        -------
        list[Node]
            Matching nodes ordered by name.
        """
        clauses: list[str] = []
        params: list[Any] = []
        if node_type is not None:
            clauses.append("node_type = ?")
            params.append(node_type.value)
        if layer is not None:
            clauses.append("layer = ?")
            params.append(layer.value)
        where = f"WHERE {' AND '.join(clauses)} " if clauses else ""
        params.append(limit)
        rows = self._conn.execute(
            "SELECT id, node_type, name, content, metadata, embedding, created_at, updated_at, layer "
            f"FROM nodes {where}ORDER BY name LIMIT ?",
            params,
        ).fetchall()
        return [_row_to_node(r) for r in rows]

    def node_count(
        self, *, node_type: NodeType | None = None, layer: GraphLayer | None = None
    ) -> int:
        """Count nodes in the graph.

        Parameters
        ----------
        node_type
            If provided, count only nodes of this type.
        layer
            If provided, count only nodes in this layer.

        Returns
        -------
        int
            Number of matching nodes.
        """
        clauses: list[str] = []
        params: list[Any] = []
        if node_type is not None:
            clauses.append("node_type = ?")
            params.append(node_type.value)
        if layer is not None:
            clauses.append("layer = ?")
            params.append(layer.value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        row = self._conn.execute(f"SELECT COUNT(*) FROM nodes {where}", params).fetchone()
        return row[0] if row else 0

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph, or update it if the key already exists.

        If the graph has an ontology, the edge is validated first.
        In strict mode a :class:`ValueError` is raised on violations;
        otherwise a warning is emitted.

        Parameters
        ----------
        edge
            The edge to add or update. The (source, target, relation) triple
            is the primary key.

        Raises
        ------
        KeyError
            If source or target node does not exist.
        ValueError
            If validation fails and ``strict=True``.
        """
        # Verify both nodes exist
        source_node = self.get_node(edge.source)
        target_node = self.get_node(edge.target)
        for nid, n in ((edge.source, source_node), (edge.target, target_node)):
            if n is None:
                raise KeyError(f"Node '{nid}' not found in graph")

        self._validate_edge(edge, source_node=source_node, target_node=target_node)

        self._conn.execute(
            """
            INSERT INTO edges (source, target, relation, weight, metadata, layer)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(source, target, relation) DO UPDATE SET
                weight=excluded.weight,
                metadata=excluded.metadata,
                layer=excluded.layer
            """,
            (
                edge.source,
                edge.target,
                edge.relation,
                edge.weight,
                json.dumps(edge.metadata),
                edge.layer.value,
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
            sql = "SELECT source, target, relation, weight, metadata, layer FROM edges WHERE source = ?"
            p: list[Any] = [node_id]
            if relation is not None:
                sql += " AND relation = ?"
                p.append(relation)
            rows = self._conn.execute(sql, p).fetchall()
            edges.extend(_row_to_edge(r) for r in rows)

        if direction in ("incoming", "both"):
            sql = "SELECT source, target, relation, weight, metadata, layer FROM edges WHERE target = ?"
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
        layer: GraphLayer | None = None,
        limit: int = 20,
    ) -> list[Node]:
        """Search nodes by name or content (case-insensitive substring match).

        Parameters
        ----------
        query
            Text to search for in node names and content.
        node_type
            Optional filter by node type.
        layer
            Optional filter by graph layer.
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
        extra_clauses = ""
        if node_type is not None:
            extra_clauses += "AND node_type = ? "
            params.append(node_type.value)
        if layer is not None:
            extra_clauses += "AND layer = ? "
            params.append(layer.value)
        params.append(limit)

        sql = f"""
            SELECT id, node_type, name, content, metadata, embedding, created_at, updated_at, layer
            FROM nodes
            WHERE (name LIKE ? OR content LIKE ?) {extra_clauses}
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
    # Subgraph extraction
    # ------------------------------------------------------------------

    def extract_subgraph(
        self,
        query: str,
        *,
        max_hops: int = 2,
        max_nodes: int = 30,
        node_type: NodeType | None = None,
        layer: GraphLayer | None = None,
        include_ontology: bool = False,
    ) -> Subgraph:
        """Extract a relevant subgraph for a query.

        Pipeline: **match → expand → rank → prune**.

        Parameters
        ----------
        query
            Text query to find seed nodes.
        max_hops
            How many edge-hops to expand from each seed (default 2).
        max_nodes
            Maximum nodes in the returned subgraph (default 30).
        node_type
            Optional filter applied during the initial search.
        layer
            Optional layer filter applied during the initial search.
        include_ontology
            If ``True``, the graph's ontology is attached to the
            returned :class:`Subgraph` for use by
            :meth:`Subgraph.to_context`.

        Returns
        -------
        Subgraph
            The extracted subgraph with nodes, edges, and seed IDs.

        Examples
        --------
        ```python
        sg = kg.extract_subgraph("revenue", max_hops=2, max_nodes=20)
        context = sg.to_context(ontology=kg.ontology)
        ```
        """
        # 1. Match — find seed nodes via substring search
        seeds = self.search(query, node_type=node_type, layer=layer, limit=max_nodes)
        seed_ids = [n.id for n in seeds]

        if not seeds:
            return Subgraph(query=query, seed_ids=[])

        # 2. Expand — BFS up to max_hops
        collected: dict[str, Node] = {n.id: n for n in seeds}
        frontier: set[str] = set(seed_ids)

        for _hop in range(max_hops):
            next_frontier: set[str] = set()
            for nid in frontier:
                for neighbor in self.neighbors(nid, direction="both"):
                    if neighbor.id not in collected:
                        collected[neighbor.id] = neighbor
                        next_frontier.add(neighbor.id)
            frontier = next_frontier
            if not frontier:
                break

        # 3. Rank — score each node
        query_lower = query.lower()
        scored: list[tuple[float, str]] = []
        for nid, node in collected.items():
            score = 0.0
            # (a) Direct seed bonus
            if nid in seed_ids:
                score += 10.0
            # (b) Name relevance
            if query_lower in node.name.lower():
                score += 5.0
            # (c) Content relevance
            if query_lower in node.content.lower():
                score += 2.0
            # (d) Freshness bonus (newer is better)
            score += min(node.updated_at / 1e10, 1.0)
            scored.append((score, nid))

        scored.sort(key=lambda x: x[0], reverse=True)

        # 4. Prune — keep top max_nodes
        keep_ids = {nid for _, nid in scored[:max_nodes]}
        final_nodes = [collected[nid] for _, nid in scored[:max_nodes]]

        # Collect edges where both endpoints are in the subgraph
        final_edges: list[Edge] = []
        seen_edges: set[tuple[str, str, str]] = set()
        for nid in keep_ids:
            for edge in self.get_edges(nid, direction="both"):
                key = (edge.source, edge.target, edge.relation)
                if edge.source in keep_ids and edge.target in keep_ids and key not in seen_edges:
                    final_edges.append(edge)
                    seen_edges.add(key)

        return Subgraph(
            nodes=final_nodes,
            edges=final_edges,
            seed_ids=seed_ids,
            query=query,
        )

    # ------------------------------------------------------------------
    # Stats / health
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Get summary statistics about the graph.

        Returns
        -------
        dict[str, Any]
            Counts of nodes (total and per type), edges, and per-layer breakdowns.

        Examples
        --------
        ```python
        kg.stats()
        # {"nodes": 42, "edges": 87, "documents": 10, "entities": 25, "topics": 7,
        #  "decisions": 3, "layers": {"base": 30, "enrichment": 10, "extended": 2}}
        ```
        """
        total_nodes = self.node_count()
        total_edges = self.edge_count()
        docs = self.node_count(node_type=NodeType.DOCUMENT)
        entities = self.node_count(node_type=NodeType.ENTITY)
        topics = self.node_count(node_type=NodeType.TOPIC)
        decisions = self.node_count(node_type=NodeType.DECISION)
        layers = {layer.value: self.node_count(layer=layer) for layer in GraphLayer}
        return {
            "nodes": total_nodes,
            "edges": total_edges,
            "documents": docs,
            "entities": entities,
            "topics": topics,
            "decisions": decisions,
            "layers": layers,
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
        result = self._get_question_queue().answer(question_id, choice=choice, freeform=freeform)
        if result is not None:
            self._record_qa_decision(result, choice=choice, freeform=freeform)
        return result

    def _record_qa_decision(
        self,
        question: Any,
        *,
        choice: int | None,
        freeform: str | None,
    ) -> None:
        """Create a DECISION node for an answered enrichment question."""
        import uuid as _uuid

        answer_text = freeform or ""
        if choice is not None and hasattr(question, "options") and question.options:
            answer_text = question.options[choice].label
        if freeform and choice is not None:
            answer_text = f"{question.options[choice].label} — {freeform}"

        decision_id = f"decision_{_uuid.uuid4().hex[:12]}"
        decision_node = Node(
            id=decision_id,
            node_type=NodeType.DECISION,
            name=f"Q&A: {question.text[:80]}",
            content=f"Answer: {answer_text}",
            metadata={
                "decision_type": question.question_type.value
                if hasattr(question.question_type, "value")
                else str(question.question_type),
                "source": "enrichment_qa",
                "question_id": question.id,
                "answer_choice": choice,
                "answer_freeform": freeform,
            },
            layer=GraphLayer.EXTENDED,
        )
        self.add_node(decision_node)

        # Link decision to referenced nodes
        node_ids = getattr(question, "node_ids", []) or []
        for nid in node_ids:
            if self.get_node(nid) is not None:
                self.add_edge(
                    Edge(
                        source=decision_id,
                        target=nid,
                        relation="resolves",
                        layer=GraphLayer.EXTENDED,
                    )
                )

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
    # Decision trail
    # ------------------------------------------------------------------

    def decision_trail(self, node_id: str) -> list[Node]:
        """Return DECISION nodes linked to *node_id*, newest first.

        Follows edges in both directions where either endpoint is
        *node_id* and the other endpoint is a DECISION node.

        Parameters
        ----------
        node_id
            The node whose decision history is requested.

        Returns
        -------
        list[Node]
            Decision nodes ordered by ``created_at`` descending.

        Examples
        --------
        ```python
        trail = kg.decision_trail("entity-alex-torres")
        trail[0].name  # most recent decision
        ```
        """
        edges = self.get_edges(node_id, direction="both")
        decision_ids: list[str] = []
        seen: set[str] = set()
        for edge in edges:
            other = edge.target if edge.source == node_id else edge.source
            if other not in seen:
                seen.add(other)
                decision_ids.append(other)

        decisions: list[Node] = []
        for did in decision_ids:
            node = self.get_node(did)
            if node is not None and node.node_type == NodeType.DECISION:
                decisions.append(node)

        decisions.sort(key=lambda n: n.created_at, reverse=True)
        return decisions

    def revert_decision(self, decision_id: str) -> bool:
        """Revert a decision: delete the DECISION node and its edges.

        Any entity/topic nodes that were *only* reachable through this
        decision (i.e., have no remaining edges after the decision's
        edges are removed) are also deleted.

        Parameters
        ----------
        decision_id
            ID of the DECISION node to revert.

        Returns
        -------
        bool
            ``True`` if the decision existed and was reverted.

        Examples
        --------
        ```python
        kg.revert_decision("decision_a1b2c3d4")
        ```
        """
        node = self.get_node(decision_id)
        if node is None or node.node_type != NodeType.DECISION:
            return False

        # Collect neighbor IDs before removing edges
        edges = self.get_edges(decision_id, direction="both")
        neighbor_ids = set()
        for edge in edges:
            other = edge.target if edge.source == decision_id else edge.source
            neighbor_ids.add(other)

        # Delete the decision node (cascade deletes its edges)
        self.delete_node(decision_id)

        # Clean up orphaned enrichment/extended nodes
        for nid in neighbor_ids:
            n = self.get_node(nid)
            if n is None:
                continue
            # Only auto-remove non-DOCUMENT nodes that lost all edges
            if n.node_type == NodeType.DOCUMENT:
                continue
            remaining = self.get_edges(nid, direction="both")
            if len(remaining) == 0:
                self.delete_node(nid)

        return True

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Delete all nodes and edges from the graph."""
        self._conn.execute("DELETE FROM edges")
        self._conn.execute("DELETE FROM nodes")
        self._conn.commit()

    def clear_layer(self, layer: GraphLayer) -> int:
        """Delete all nodes and edges belonging to a specific layer.

        Edges are deleted first (those whose *source* or *target* is in the
        layer), then the nodes themselves.

        Parameters
        ----------
        layer
            The graph layer to clear.

        Returns
        -------
        int
            Number of nodes removed.
        """
        self._conn.execute("DELETE FROM edges WHERE layer = ?", (layer.value,))
        cursor = self._conn.execute("DELETE FROM nodes WHERE layer = ?", (layer.value,))
        self._conn.commit()
        return cursor.rowcount

    # ------------------------------------------------------------------
    # Export / Import / Compose
    # ------------------------------------------------------------------

    def _write_manifest(
        self,
        conn: sqlite3.Connection,
        *,
        layers: list[str],
        description: str,
        author: str,
    ) -> None:
        """Write a _manifest table into *conn*."""
        conn.execute(
            "CREATE TABLE IF NOT EXISTS _manifest "
            "(key TEXT PRIMARY KEY, value TEXT NOT NULL DEFAULT '{}')"
        )
        now = time.time()
        ontology_json = json.dumps(self._ontology.to_dict())
        # Compute a SHA-256 checksum over node/edge content
        import hashlib

        h = hashlib.sha256()
        for row in conn.execute("SELECT id, name, content FROM nodes ORDER BY id"):
            h.update(f"{row[0]}|{row[1]}|{row[2]}".encode())
        for row in conn.execute(
            "SELECT source, target, relation FROM edges ORDER BY source, target, relation"
        ):
            h.update(f"{row[0]}|{row[1]}|{row[2]}".encode())
        manifest = {
            "name": self._name,
            "description": description,
            "ontology": ontology_json,
            "created_at": now,
            "author": author,
            "layer_filter": layers,
            "version": 1,
            "checksum": h.hexdigest(),
        }
        for k, v in manifest.items():
            conn.execute(
                "INSERT OR REPLACE INTO _manifest (key, value) VALUES (?, ?)",
                (k, json.dumps(v) if not isinstance(v, str) else v),
            )
        conn.commit()

    def export_base(self, path: str | Path, *, description: str = "", author: str = "") -> Path:
        """Export the BASE layer to a `.kg` file.

        Parameters
        ----------
        path
            Destination file path.
        description
            Optional human-readable description embedded in the manifest.
        author
            Optional author name.

        Returns
        -------
        Path
            The resolved output path.
        """
        return self.export_layers([GraphLayer.BASE], path, description=description, author=author)

    def export_layers(
        self,
        layers: list[GraphLayer],
        path: str | Path,
        *,
        description: str = "",
        author: str = "",
    ) -> Path:
        """Export selected layers to a `.kg` file.

        The `.kg` file is a self-contained SQLite database with the same
        schema as a live graph, plus a ``_manifest`` table for metadata.

        Parameters
        ----------
        layers
            Which layers to include.
        path
            Destination file path.
        description
            Optional description embedded in the manifest.
        author
            Optional author name.

        Returns
        -------
        Path
            The resolved output path.
        """
        out = Path(path)
        layer_vals = [l.value for l in layers]
        placeholders = ",".join("?" * len(layer_vals))

        dest = sqlite3.connect(str(out))
        try:
            dest.execute("PRAGMA journal_mode=WAL")
            dest.execute("PRAGMA foreign_keys=ON")
            dest.execute(_CREATE_NODES_SQL)
            dest.execute(_CREATE_EDGES_SQL)
            dest.execute(_CREATE_META_SQL)
            dest.commit()

            # Copy filtered nodes
            for row in self._conn.execute(
                f"SELECT * FROM nodes WHERE layer IN ({placeholders})",  # noqa: S608
                layer_vals,
            ):
                dest.execute("INSERT OR REPLACE INTO nodes VALUES (?,?,?,?,?,?,?,?,?)", row)
            # Copy filtered edges
            for row in self._conn.execute(
                f"SELECT * FROM edges WHERE layer IN ({placeholders})",  # noqa: S608
                layer_vals,
            ):
                dest.execute("INSERT OR REPLACE INTO edges VALUES (?,?,?,?,?,?)", row)
            # Copy ontology
            onto_row = self._conn.execute(
                "SELECT value FROM _meta WHERE key = 'ontology'"
            ).fetchone()
            if onto_row:
                dest.execute(
                    "INSERT OR REPLACE INTO _meta (key, value) VALUES ('ontology', ?)",
                    (onto_row[0],),
                )
            dest.commit()

            self._write_manifest(
                dest,
                layers=[l.value for l in layers],
                description=description,
                author=author,
            )
        finally:
            dest.close()

        return out

    def compose(
        self,
        other: "KnowledgeGraph",
        *,
        namespace: str = "",
        merge_strategy: str = "additive",
        conflict: str = "keep_local",
    ) -> int:
        """Merge another graph into this one.

        Parameters
        ----------
        other
            The source graph to merge from.
        namespace
            If non-empty, node IDs from *other* are prefixed with
            ``namespace:`` to avoid collisions.
        merge_strategy
            ``"additive"`` (default) — add new nodes/edges without removing
            existing ones.
        conflict
            How to handle ID collisions: ``"keep_local"`` (default) skips
            duplicates, ``"keep_remote"`` overwrites with the incoming node.

        Returns
        -------
        int
            Number of nodes added or updated.
        """
        if merge_strategy != "additive":
            raise ValueError(f"Unsupported merge_strategy: {merge_strategy!r}")
        if conflict not in ("keep_local", "keep_remote"):
            raise ValueError(f"Unsupported conflict policy: {conflict!r}")

        added = 0
        # Gather all nodes from other
        for node in other.list_nodes(limit=10_000_000):
            nid = f"{namespace}:{node.id}" if namespace else node.id
            existing = self.get_node(nid)
            if existing is not None and conflict == "keep_local":
                continue
            self.add_node(
                Node(
                    id=nid,
                    node_type=node.node_type,
                    name=node.name,
                    content=node.content,
                    metadata=node.metadata,
                    embedding=node.embedding,
                    created_at=node.created_at,
                    updated_at=node.updated_at,
                    layer=node.layer,
                )
            )
            added += 1

        # Gather all edges from other
        id_set: set[str] = set()
        for node in other.list_nodes(limit=10_000_000):
            nid = f"{namespace}:{node.id}" if namespace else node.id
            id_set.add(nid)

        for node in other.list_nodes(limit=10_000_000):
            orig_id = node.id
            for edge in other.get_edges(orig_id, direction="outgoing"):
                src = f"{namespace}:{edge.source}" if namespace else edge.source
                tgt = f"{namespace}:{edge.target}" if namespace else edge.target
                # Only add edge if both endpoints exist in this graph
                if self.get_node(src) is not None and self.get_node(tgt) is not None:
                    self.add_edge(
                        Edge(
                            source=src,
                            target=tgt,
                            relation=edge.relation,
                            weight=edge.weight,
                            metadata=edge.metadata,
                            layer=edge.layer,
                        )
                    )

        # Merge ontology: union disjoint types, local wins on conflicts
        if other.ontology.entity_types or other.ontology.relation_types:
            merged_et = dict(self._ontology.entity_types)
            for k, v in other.ontology.entity_types.items():
                if k not in merged_et:
                    merged_et[k] = v
                else:
                    warnings.warn(
                        f"Entity type {k!r} exists in both graphs; keeping local definition.",
                        stacklevel=2,
                    )
            merged_rt = dict(self._ontology.relation_types)
            for k, v in other.ontology.relation_types.items():
                if k not in merged_rt:
                    merged_rt[k] = v
                else:
                    warnings.warn(
                        f"Relation type {k!r} exists in both graphs; keeping local definition.",
                        stacklevel=2,
                    )
            self.set_ontology(Ontology(entity_types=merged_et, relation_types=merged_rt))

        return added

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __repr__(self) -> str:
        s = self.stats()
        return f"KnowledgeGraph(path={self._path!r}, nodes={s['nodes']}, edges={s['edges']})"


# ---------------------------------------------------------------------------
# KnowledgeGraphRegistry
# ---------------------------------------------------------------------------

import os  # noqa: E402
import re as _re  # noqa: E402

_DEFAULT_DIR = os.path.join(os.path.expanduser("~/.config/talk-box"), "graphs")

# Restrict graph names to safe filesystem characters
_VALID_NAME_RE = _re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.-]{0,98}[A-Za-z0-9]$|^[A-Za-z0-9]$")


class KnowledgeGraphRegistry:
    """Manage multiple named knowledge graphs stored in a directory.

    Each graph lives in its own SQLite file.  A JSON manifest tracks
    names, descriptions, and which graph is the active default.

    Parameters
    ----------
    directory
        Folder where graph databases are stored. Defaults to
        ``~/.config/talk-box/graphs/``.

    Examples
    --------
    ```python
    import talk_box as tb

    registry = tb.KnowledgeGraphRegistry()
    registry.create("research", description="Papers and notes")
    registry.create("work", description="Project knowledge")

    kg = registry.open("research")
    kg.add_node(tb.Node(
        id="doc-1",
        node_type=tb.NodeType.DOCUMENT,
        name="Paper.pdf",
    ))

    registry.list_graphs()
    # [{"name": "research", ...}, {"name": "work", ...}]

    registry.set_default("research")
    kg = registry.open_default()
    ```
    """

    def __init__(self, directory: str | Path | None = None) -> None:
        self._dir = str(directory or _DEFAULT_DIR)
        os.makedirs(self._dir, exist_ok=True)
        self._manifest_path = os.path.join(self._dir, "manifest.json")
        self._manifest = self._load_manifest()

    # ------------------------------------------------------------------
    # Manifest I/O
    # ------------------------------------------------------------------

    def _load_manifest(self) -> dict[str, Any]:
        """Load the JSON manifest (or return defaults)."""
        if os.path.isfile(self._manifest_path):
            with open(self._manifest_path) as f:
                return json.load(f)
        return {"graphs": {}, "default": None}

    def _save_manifest(self) -> None:
        """Persist the manifest to disk."""
        with open(self._manifest_path, "w") as f:
            json.dump(self._manifest, f, indent=2)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_name(name: str) -> None:
        """Raise ValueError for illegal graph names."""
        if not name or not _VALID_NAME_RE.match(name):
            raise ValueError(
                f"Invalid graph name {name!r}. Must be 1-100 chars, "
                "alphanumeric/space/underscore/dot/hyphen, no leading/trailing specials."
            )

    def _db_path(self, name: str) -> str:
        """Return the SQLite path for a graph name."""
        safe = name.replace(" ", "_").lower()
        return os.path.join(self._dir, f"{safe}.db")

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create(
        self,
        name: str,
        *,
        description: str = "",
        set_default: bool = False,
    ) -> KnowledgeGraph:
        """Create a new named knowledge graph.

        Parameters
        ----------
        name
            Unique human-readable name.
        description
            Optional description of this graph's purpose.
        set_default
            If ``True``, make this the default graph.

        Returns
        -------
        KnowledgeGraph
            The newly-created graph (open and ready to use).

        Raises
        ------
        ValueError
            If the name is invalid or already taken.

        Examples
        --------
        ```python
        kg = registry.create("research", description="Papers and notes")
        ```
        """
        self._validate_name(name)
        if name in self._manifest["graphs"]:
            raise ValueError(f"Graph {name!r} already exists")

        db_path = self._db_path(name)
        self._manifest["graphs"][name] = {
            "description": description,
            "db_file": os.path.basename(db_path),
            "created_at": time.time(),
        }
        if set_default or self._manifest["default"] is None:
            self._manifest["default"] = name
        self._save_manifest()

        return KnowledgeGraph(db_path, name=name)

    def open(self, name: str) -> KnowledgeGraph:
        """Open an existing named knowledge graph.

        Parameters
        ----------
        name
            Name of the graph to open.

        Returns
        -------
        KnowledgeGraph
            The open graph.

        Raises
        ------
        KeyError
            If no graph with that name exists.

        Examples
        --------
        ```python
        kg = registry.open("research")
        ```
        """
        if name not in self._manifest["graphs"]:
            raise KeyError(f"No graph named {name!r}")
        db_path = self._db_path(name)
        return KnowledgeGraph(db_path, name=name)

    def open_default(self) -> KnowledgeGraph | None:
        """Open the default knowledge graph.

        Returns
        -------
        KnowledgeGraph | None
            The default graph, or ``None`` if no default is set.

        Examples
        --------
        ```python
        kg = registry.open_default()
        ```
        """
        default = self._manifest.get("default")
        if default is None or default not in self._manifest["graphs"]:
            return None
        return self.open(default)

    def delete(self, name: str) -> bool:
        """Delete a named knowledge graph and its database file.

        Parameters
        ----------
        name
            Name of the graph to delete.

        Returns
        -------
        bool
            ``True`` if the graph existed and was deleted.

        Examples
        --------
        ```python
        registry.delete("old-project")
        ```
        """
        if name not in self._manifest["graphs"]:
            return False
        db_path = self._db_path(name)
        if os.path.isfile(db_path):
            os.remove(db_path)
        # Also clean up WAL/SHM files
        for suffix in ("-wal", "-shm"):
            wal = db_path + suffix
            if os.path.isfile(wal):
                os.remove(wal)
        del self._manifest["graphs"][name]
        if self._manifest["default"] == name:
            remaining = list(self._manifest["graphs"])
            self._manifest["default"] = remaining[0] if remaining else None
        self._save_manifest()
        return True

    def rename(self, old_name: str, new_name: str) -> None:
        """Rename an existing knowledge graph.

        Parameters
        ----------
        old_name
            Current name of the graph.
        new_name
            New name for the graph.

        Raises
        ------
        KeyError
            If ``old_name`` does not exist.
        ValueError
            If ``new_name`` is invalid or already taken.

        Examples
        --------
        ```python
        registry.rename("old-project", "legacy-project")
        ```
        """
        if old_name not in self._manifest["graphs"]:
            raise KeyError(f"No graph named {old_name!r}")
        self._validate_name(new_name)
        if new_name in self._manifest["graphs"]:
            raise ValueError(f"Graph {new_name!r} already exists")

        old_path = self._db_path(old_name)
        new_path = self._db_path(new_name)

        if os.path.isfile(old_path):
            os.rename(old_path, new_path)
            for suffix in ("-wal", "-shm"):
                old_wal = old_path + suffix
                new_wal = new_path + suffix
                if os.path.isfile(old_wal):
                    os.rename(old_wal, new_wal)

        entry = self._manifest["graphs"].pop(old_name)
        entry["db_file"] = os.path.basename(new_path)
        self._manifest["graphs"][new_name] = entry
        if self._manifest["default"] == old_name:
            self._manifest["default"] = new_name
        self._save_manifest()

    def set_default(self, name: str) -> None:
        """Set a graph as the default.

        Parameters
        ----------
        name
            Name of the graph to make the default.

        Raises
        ------
        KeyError
            If no graph with that name exists.

        Examples
        --------
        ```python
        registry.set_default("work")
        ```
        """
        if name not in self._manifest["graphs"]:
            raise KeyError(f"No graph named {name!r}")
        self._manifest["default"] = name
        self._save_manifest()

    @property
    def default_name(self) -> str | None:
        """Name of the current default graph, or ``None``."""
        return self._manifest.get("default")

    def list_graphs(self) -> list[dict[str, Any]]:
        """List all registered graphs.

        Returns
        -------
        list[dict[str, Any]]
            Each dict has ``name``, ``description``, ``db_file``,
            ``created_at``, and ``is_default`` keys.

        Examples
        --------
        ```python
        for g in registry.list_graphs():
            print(g["name"], g["description"])
        ```
        """
        default = self._manifest.get("default")
        result = []
        for name, info in self._manifest["graphs"].items():
            result.append(
                {
                    "name": name,
                    "description": info.get("description", ""),
                    "db_file": info.get("db_file", ""),
                    "created_at": info.get("created_at", 0),
                    "is_default": name == default,
                }
            )
        return result

    def graph_count(self) -> int:
        """Return the number of registered graphs."""
        return len(self._manifest["graphs"])

    def __contains__(self, name: str) -> bool:
        return name in self._manifest["graphs"]

    def __repr__(self) -> str:
        return (
            f"KnowledgeGraphRegistry(directory={self._dir!r}, "
            f"graphs={self.graph_count()}, default={self.default_name!r})"
        )


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
        layer=GraphLayer(row[8]) if len(row) > 8 and row[8] is not None else GraphLayer.BASE,
    )


def _row_to_edge(row: tuple[Any, ...]) -> Edge:
    """Convert a database row to an Edge."""
    return Edge(
        source=row[0],
        target=row[1],
        relation=row[2],
        weight=row[3],
        metadata=json.loads(row[4]),
        layer=GraphLayer(row[5]) if len(row) > 5 and row[5] is not None else GraphLayer.BASE,
    )


# ---------------------------------------------------------------------------
# Import helper
# ---------------------------------------------------------------------------


def import_kg(path: str | Path) -> KnowledgeGraph:
    """Open a ``.kg`` file (or any Talk Box SQLite graph) as a read-only graph.

    The returned :class:`KnowledgeGraph` can be passed to
    :meth:`KnowledgeGraph.compose` or inspected directly.

    Parameters
    ----------
    path
        Path to a ``.kg`` file exported by :meth:`KnowledgeGraph.export_base`
        or :meth:`KnowledgeGraph.export_layers`.

    Returns
    -------
    KnowledgeGraph
        A graph instance backed by the given file.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No such file: {p}")
    kg = KnowledgeGraph(p)
    # Read manifest name if available
    try:
        row = kg._conn.execute("SELECT value FROM _manifest WHERE key = 'name'").fetchone()
        if row:
            kg._name = row[0]
    except sqlite3.OperationalError:
        pass  # No _manifest table — plain graph file
    return kg
