import math

import pytest

from talk_box.knowledge_graph import (
    Edge,
    EntityTypeDef,
    GraphLayer,
    KnowledgeGraph,
    Node,
    NodeType,
    Ontology,
    RelationTypeDef,
    _blob_to_floats,
    _floats_to_blob,
    cosine_similarity,
    general_ontology,
)


# ---------------------------------------------------------------------------
# NodeType
# ---------------------------------------------------------------------------


class TestNodeType:
    def test_values(self):
        assert NodeType.DOCUMENT.value == "document"
        assert NodeType.ENTITY.value == "entity"
        assert NodeType.TOPIC.value == "topic"

    def test_from_string(self):
        assert NodeType("document") is NodeType.DOCUMENT


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


class TestNode:
    def test_basic_creation(self):
        n = Node(id="n1", node_type=NodeType.DOCUMENT, name="Doc")
        assert n.id == "n1"
        assert n.node_type is NodeType.DOCUMENT
        assert n.name == "Doc"
        assert n.content == ""
        assert n.metadata == {}
        assert n.embedding is None
        assert n.created_at > 0
        assert n.updated_at > 0

    def test_auto_timestamps(self):
        n = Node(id="n1", node_type=NodeType.ENTITY, name="E")
        assert n.created_at == n.updated_at
        assert n.created_at > 0

    def test_with_embedding(self):
        n = Node(
            id="n1",
            node_type=NodeType.ENTITY,
            name="E",
            embedding=[0.1, 0.2, 0.3],
        )
        assert n.embedding == [0.1, 0.2, 0.3]


# ---------------------------------------------------------------------------
# Edge
# ---------------------------------------------------------------------------


class TestEdge:
    def test_basic_creation(self):
        e = Edge(source="a", target="b", relation="mentions")
        assert e.source == "a"
        assert e.target == "b"
        assert e.relation == "mentions"
        assert e.weight == 1.0
        assert e.metadata == {}

    def test_frozen(self):
        e = Edge(source="a", target="b", relation="r")
        with pytest.raises(AttributeError):
            e.source = "c"  # type: ignore[misc]

    def test_with_weight_and_metadata(self):
        e = Edge(source="a", target="b", relation="r", weight=0.5, metadata={"k": "v"})
        assert e.weight == 0.5
        assert e.metadata == {"k": "v"}


# ---------------------------------------------------------------------------
# KnowledgeGraph — node operations
# ---------------------------------------------------------------------------


class TestKGNodes:
    def setup_method(self):
        self.kg = KnowledgeGraph(":memory:")

    def teardown_method(self):
        self.kg.close()

    def test_add_and_get_node(self):
        node = Node(id="d1", node_type=NodeType.DOCUMENT, name="Doc 1", content="Hello")
        self.kg.add_node(node)
        result = self.kg.get_node("d1")
        assert result is not None
        assert result.name == "Doc 1"
        assert result.content == "Hello"
        assert result.node_type is NodeType.DOCUMENT

    def test_get_missing_node(self):
        assert self.kg.get_node("nonexistent") is None

    def test_update_node(self):
        self.kg.add_node(Node(id="d1", node_type=NodeType.DOCUMENT, name="V1"))
        self.kg.add_node(Node(id="d1", node_type=NodeType.DOCUMENT, name="V2"))
        result = self.kg.get_node("d1")
        assert result is not None
        assert result.name == "V2"

    def test_delete_node(self):
        self.kg.add_node(Node(id="d1", node_type=NodeType.DOCUMENT, name="Doc"))
        assert self.kg.delete_node("d1") is True
        assert self.kg.get_node("d1") is None

    def test_delete_missing_node(self):
        assert self.kg.delete_node("nonexistent") is False

    def test_list_nodes(self):
        self.kg.add_node(Node(id="d1", node_type=NodeType.DOCUMENT, name="B"))
        self.kg.add_node(Node(id="d2", node_type=NodeType.DOCUMENT, name="A"))
        self.kg.add_node(Node(id="e1", node_type=NodeType.ENTITY, name="C"))
        nodes = self.kg.list_nodes()
        assert len(nodes) == 3
        # Ordered by name
        assert nodes[0].name == "A"
        assert nodes[1].name == "B"

    def test_list_nodes_by_type(self):
        self.kg.add_node(Node(id="d1", node_type=NodeType.DOCUMENT, name="Doc"))
        self.kg.add_node(Node(id="e1", node_type=NodeType.ENTITY, name="Ent"))
        docs = self.kg.list_nodes(node_type=NodeType.DOCUMENT)
        assert len(docs) == 1
        assert docs[0].id == "d1"

    def test_list_nodes_limit(self):
        for i in range(5):
            self.kg.add_node(Node(id=f"n{i}", node_type=NodeType.TOPIC, name=f"T{i}"))
        nodes = self.kg.list_nodes(limit=3)
        assert len(nodes) == 3

    def test_node_count(self):
        assert self.kg.node_count() == 0
        self.kg.add_node(Node(id="d1", node_type=NodeType.DOCUMENT, name="Doc"))
        self.kg.add_node(Node(id="e1", node_type=NodeType.ENTITY, name="Ent"))
        assert self.kg.node_count() == 2
        assert self.kg.node_count(node_type=NodeType.DOCUMENT) == 1

    def test_node_with_metadata(self):
        self.kg.add_node(
            Node(
                id="d1",
                node_type=NodeType.DOCUMENT,
                name="Doc",
                metadata={"source": "test", "page": 1},
            )
        )
        result = self.kg.get_node("d1")
        assert result is not None
        assert result.metadata == {"source": "test", "page": 1}

    def test_node_with_embedding(self):
        emb = [0.1, 0.2, 0.3, 0.4]
        self.kg.add_node(Node(id="d1", node_type=NodeType.DOCUMENT, name="Doc", embedding=emb))
        result = self.kg.get_node("d1")
        assert result is not None
        assert result.embedding is not None
        assert len(result.embedding) == 4
        for a, b in zip(result.embedding, emb):
            assert abs(a - b) < 1e-5


# ---------------------------------------------------------------------------
# KnowledgeGraph — edge operations
# ---------------------------------------------------------------------------


class TestKGEdges:
    def setup_method(self):
        self.kg = KnowledgeGraph(":memory:")
        self.kg.add_node(Node(id="d1", node_type=NodeType.DOCUMENT, name="Doc"))
        self.kg.add_node(Node(id="e1", node_type=NodeType.ENTITY, name="Entity"))
        self.kg.add_node(Node(id="t1", node_type=NodeType.TOPIC, name="Topic"))

    def teardown_method(self):
        self.kg.close()

    def test_add_and_get_edge(self):
        self.kg.add_edge(Edge(source="d1", target="e1", relation="mentions"))
        edges = self.kg.get_edges("d1", direction="outgoing")
        assert len(edges) == 1
        assert edges[0].relation == "mentions"

    def test_edge_with_missing_node_raises(self):
        with pytest.raises(KeyError, match="not found"):
            self.kg.add_edge(Edge(source="d1", target="missing", relation="r"))

    def test_edge_with_missing_source_raises(self):
        with pytest.raises(KeyError, match="not found"):
            self.kg.add_edge(Edge(source="missing", target="e1", relation="r"))

    def test_get_edges_outgoing(self):
        self.kg.add_edge(Edge(source="d1", target="e1", relation="mentions"))
        self.kg.add_edge(Edge(source="d1", target="t1", relation="about"))
        edges = self.kg.get_edges("d1", direction="outgoing")
        assert len(edges) == 2

    def test_get_edges_incoming(self):
        self.kg.add_edge(Edge(source="d1", target="e1", relation="mentions"))
        edges = self.kg.get_edges("e1", direction="incoming")
        assert len(edges) == 1
        assert edges[0].source == "d1"

    def test_get_edges_both(self):
        self.kg.add_edge(Edge(source="d1", target="e1", relation="mentions"))
        self.kg.add_edge(Edge(source="t1", target="d1", relation="contains"))
        edges = self.kg.get_edges("d1", direction="both")
        assert len(edges) == 2

    def test_get_edges_filter_relation(self):
        self.kg.add_edge(Edge(source="d1", target="e1", relation="mentions"))
        self.kg.add_edge(Edge(source="d1", target="t1", relation="about"))
        edges = self.kg.get_edges("d1", direction="outgoing", relation="mentions")
        assert len(edges) == 1
        assert edges[0].relation == "mentions"

    def test_update_edge(self):
        self.kg.add_edge(Edge(source="d1", target="e1", relation="mentions", weight=0.5))
        self.kg.add_edge(Edge(source="d1", target="e1", relation="mentions", weight=0.9))
        edges = self.kg.get_edges("d1", direction="outgoing")
        assert len(edges) == 1
        assert edges[0].weight == 0.9

    def test_delete_edge(self):
        self.kg.add_edge(Edge(source="d1", target="e1", relation="mentions"))
        assert self.kg.delete_edge("d1", "e1", "mentions") is True
        assert self.kg.get_edges("d1") == []

    def test_delete_missing_edge(self):
        assert self.kg.delete_edge("d1", "e1", "r") is False

    def test_edge_count(self):
        assert self.kg.edge_count() == 0
        self.kg.add_edge(Edge(source="d1", target="e1", relation="mentions"))
        self.kg.add_edge(Edge(source="d1", target="t1", relation="about"))
        assert self.kg.edge_count() == 2

    def test_cascade_delete(self):
        self.kg.add_edge(Edge(source="d1", target="e1", relation="mentions"))
        self.kg.delete_node("d1")
        assert self.kg.edge_count() == 0

    def test_edge_metadata(self):
        self.kg.add_edge(Edge(source="d1", target="e1", relation="mentions", metadata={"line": 42}))
        edges = self.kg.get_edges("d1", direction="outgoing")
        assert edges[0].metadata == {"line": 42}


# ---------------------------------------------------------------------------
# KnowledgeGraph — traversal
# ---------------------------------------------------------------------------


class TestKGTraversal:
    def setup_method(self):
        self.kg = KnowledgeGraph(":memory:")
        self.kg.add_node(Node(id="d1", node_type=NodeType.DOCUMENT, name="Doc"))
        self.kg.add_node(Node(id="e1", node_type=NodeType.ENTITY, name="Entity A"))
        self.kg.add_node(Node(id="e2", node_type=NodeType.ENTITY, name="Entity B"))
        self.kg.add_node(Node(id="t1", node_type=NodeType.TOPIC, name="Topic"))
        self.kg.add_edge(Edge(source="d1", target="e1", relation="mentions"))
        self.kg.add_edge(Edge(source="d1", target="e2", relation="mentions"))
        self.kg.add_edge(Edge(source="d1", target="t1", relation="about"))
        self.kg.add_edge(Edge(source="t1", target="d1", relation="contains"))

    def teardown_method(self):
        self.kg.close()

    def test_neighbors_outgoing(self):
        neighbors = self.kg.neighbors("d1", direction="outgoing")
        names = {n.name for n in neighbors}
        assert names == {"Entity A", "Entity B", "Topic"}

    def test_neighbors_incoming(self):
        neighbors = self.kg.neighbors("d1", direction="incoming")
        names = {n.name for n in neighbors}
        assert names == {"Topic"}

    def test_neighbors_both(self):
        neighbors = self.kg.neighbors("d1", direction="both")
        names = {n.name for n in neighbors}
        assert names == {"Entity A", "Entity B", "Topic"}

    def test_neighbors_filter_relation(self):
        neighbors = self.kg.neighbors("d1", relation="mentions")
        names = {n.name for n in neighbors}
        assert names == {"Entity A", "Entity B"}

    def test_neighbors_deduplicates(self):
        # t1 connects to d1 both ways
        neighbors = self.kg.neighbors("d1", direction="both")
        ids = [n.id for n in neighbors]
        assert ids.count("t1") == 1


# ---------------------------------------------------------------------------
# KnowledgeGraph — search
# ---------------------------------------------------------------------------


class TestKGSearch:
    def setup_method(self):
        self.kg = KnowledgeGraph(":memory:")
        self.kg.add_node(
            Node(id="d1", node_type=NodeType.DOCUMENT, name="Python Guide", content="Learn Python.")
        )
        self.kg.add_node(
            Node(id="d2", node_type=NodeType.DOCUMENT, name="Java Guide", content="Learn Java.")
        )
        self.kg.add_node(
            Node(id="e1", node_type=NodeType.ENTITY, name="Python", content="Programming language.")
        )
        self.kg.add_node(
            Node(
                id="e2",
                node_type=NodeType.ENTITY,
                name="Java",
                content="Another programming language.",
            )
        )

    def teardown_method(self):
        self.kg.close()

    def test_search_by_name(self):
        results = self.kg.search("Python")
        names = [r.name for r in results]
        assert "Python" in names
        assert "Python Guide" in names

    def test_search_by_content(self):
        results = self.kg.search("programming")
        assert len(results) == 2  # Both entity contents mention "programming"

    def test_search_case_insensitive(self):
        results = self.kg.search("python")
        assert len(results) >= 1

    def test_search_filter_type(self):
        results = self.kg.search("Python", node_type=NodeType.ENTITY)
        assert len(results) == 1
        assert results[0].id == "e1"

    def test_search_limit(self):
        results = self.kg.search("a", limit=2)
        assert len(results) <= 2

    def test_search_no_results(self):
        results = self.kg.search("zzzznotfound")
        assert results == []

    def test_search_name_matches_first(self):
        results = self.kg.search("Python")
        # "Python" (exact name match) should come before "Python Guide"
        assert results[0].name == "Python"


# ---------------------------------------------------------------------------
# KnowledgeGraph — stats and health
# ---------------------------------------------------------------------------


class TestKGStats:
    def setup_method(self):
        self.kg = KnowledgeGraph(":memory:")

    def teardown_method(self):
        self.kg.close()

    def test_stats_empty(self):
        s = self.kg.stats()
        assert s == {
            "nodes": 0,
            "edges": 0,
            "documents": 0,
            "entities": 0,
            "topics": 0,
            "decisions": 0,
            "layers": {"base": 0, "enrichment": 0, "extended": 0},
        }

    def test_stats_populated(self):
        self.kg.add_node(Node(id="d1", node_type=NodeType.DOCUMENT, name="D"))
        self.kg.add_node(Node(id="e1", node_type=NodeType.ENTITY, name="E"))
        self.kg.add_node(Node(id="t1", node_type=NodeType.TOPIC, name="T"))
        self.kg.add_edge(Edge(source="d1", target="e1", relation="mentions"))
        s = self.kg.stats()
        assert s["nodes"] == 3
        assert s["edges"] == 1
        assert s["documents"] == 1
        assert s["entities"] == 1
        assert s["topics"] == 1

    def test_health_empty(self):
        h = self.kg.health()
        assert h["orphan_nodes"] == 0
        assert h["total_nodes"] == 0
        assert h["embedding_coverage"] == 0.0

    def test_health_with_orphans(self):
        self.kg.add_node(Node(id="d1", node_type=NodeType.DOCUMENT, name="D1"))
        self.kg.add_node(Node(id="d2", node_type=NodeType.DOCUMENT, name="D2"))
        self.kg.add_node(Node(id="e1", node_type=NodeType.ENTITY, name="E1"))
        self.kg.add_edge(Edge(source="d1", target="e1", relation="mentions"))
        h = self.kg.health()
        assert h["orphan_nodes"] == 1  # d2 has no edges
        assert h["connected_nodes"] == 2

    def test_health_embedding_coverage(self):
        self.kg.add_node(
            Node(id="d1", node_type=NodeType.DOCUMENT, name="D1", embedding=[0.1, 0.2])
        )
        self.kg.add_node(Node(id="d2", node_type=NodeType.DOCUMENT, name="D2"))
        h = self.kg.health()
        assert h["embedding_coverage"] == 0.5


# ---------------------------------------------------------------------------
# KnowledgeGraph — lifecycle
# ---------------------------------------------------------------------------


class TestKGLifecycle:
    def test_clear(self):
        kg = KnowledgeGraph(":memory:")
        kg.add_node(Node(id="d1", node_type=NodeType.DOCUMENT, name="D"))
        kg.add_node(Node(id="e1", node_type=NodeType.ENTITY, name="E"))
        kg.add_edge(Edge(source="d1", target="e1", relation="mentions"))
        kg.clear()
        assert kg.node_count() == 0
        assert kg.edge_count() == 0
        kg.close()

    def test_repr(self):
        kg = KnowledgeGraph(":memory:")
        r = repr(kg)
        assert "KnowledgeGraph" in r
        assert "nodes=0" in r
        assert "edges=0" in r
        kg.close()

    def test_file_persistence(self, tmp_path):
        db_path = tmp_path / "test.db"
        kg = KnowledgeGraph(db_path)
        kg.add_node(Node(id="d1", node_type=NodeType.DOCUMENT, name="Doc"))
        kg.close()

        # Reopen and verify persistence
        kg2 = KnowledgeGraph(db_path)
        assert kg2.get_node("d1") is not None
        assert kg2.get_node("d1").name == "Doc"
        kg2.close()

    def test_path_property(self):
        kg = KnowledgeGraph(":memory:")
        assert kg.path == ":memory:"
        kg.close()


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


class TestEmbeddingHelpers:
    def test_roundtrip(self):
        floats = [0.1, 0.2, 0.3, 0.4, 0.5]
        blob = _floats_to_blob(floats)
        result = _blob_to_floats(blob)
        assert len(result) == len(floats)
        for a, b in zip(result, floats):
            assert abs(a - b) < 1e-5

    def test_empty_list(self):
        blob = _floats_to_blob([])
        result = _blob_to_floats(blob)
        assert result == []


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors(self):
        assert cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_similar_vectors(self):
        sim = cosine_similarity([1, 1, 0], [1, 0, 0])
        assert 0 < sim < 1
        assert sim == pytest.approx(1.0 / math.sqrt(2))

    def test_zero_vector(self):
        assert cosine_similarity([0, 0, 0], [1, 2, 3]) == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="mismatch"):
            cosine_similarity([1, 2], [1, 2, 3])


# ---------------------------------------------------------------------------
# Top-level imports
# ---------------------------------------------------------------------------


class TestTopLevelImport:
    def test_import_from_package(self):
        import talk_box

        for name in [
            "KnowledgeGraph",
            "Node",
            "Edge",
            "NodeType",
            "cosine_similarity",
        ]:
            assert hasattr(talk_box, name), f"talk_box.{name} not found"

    def test_all_contains_exports(self):
        import talk_box

        for name in [
            "KnowledgeGraph",
            "KnowledgeGraphRegistry",
            "Node",
            "Edge",
            "NodeType",
            "cosine_similarity",
        ]:
            assert name in talk_box.__all__, f"{name} not in __all__"


# ---------------------------------------------------------------------------
# KnowledgeGraph.name
# ---------------------------------------------------------------------------


class TestKnowledgeGraphName:
    def test_default_name_is_empty(self):
        kg = KnowledgeGraph(":memory:")
        assert kg.name == ""

    def test_custom_name(self):
        kg = KnowledgeGraph(":memory:", name="research")
        assert kg.name == "research"


# ---------------------------------------------------------------------------
# KnowledgeGraphRegistry
# ---------------------------------------------------------------------------


from talk_box.knowledge_graph import KnowledgeGraphRegistry


class TestKnowledgeGraphRegistry:
    @pytest.fixture()
    def reg_dir(self, tmp_path):
        return str(tmp_path / "graphs")

    @pytest.fixture()
    def registry(self, reg_dir):
        return KnowledgeGraphRegistry(reg_dir)

    def test_create_and_list(self, registry):
        registry.create("research", description="Papers")
        registry.create("work")
        graphs = registry.list_graphs()
        assert len(graphs) == 2
        names = {g["name"] for g in graphs}
        assert names == {"research", "work"}

    def test_first_graph_becomes_default(self, registry):
        registry.create("first")
        assert registry.default_name == "first"

    def test_set_default(self, registry):
        registry.create("a")
        registry.create("b")
        registry.set_default("b")
        assert registry.default_name == "b"

    def test_set_default_nonexistent_raises(self, registry):
        with pytest.raises(KeyError):
            registry.set_default("ghost")

    def test_open_graph(self, registry):
        registry.create("test")
        kg = registry.open("test")
        assert isinstance(kg, KnowledgeGraph)
        assert kg.name == "test"
        # Can add nodes
        kg.add_node(Node(id="n1", node_type=NodeType.DOCUMENT, name="Doc"))
        assert kg.node_count() == 1

    def test_open_nonexistent_raises(self, registry):
        with pytest.raises(KeyError):
            registry.open("nope")

    def test_open_default(self, registry):
        registry.create("main", set_default=True)
        kg = registry.open_default()
        assert kg is not None
        assert kg.name == "main"

    def test_open_default_when_empty(self, registry):
        assert registry.open_default() is None

    def test_delete_graph(self, registry):
        registry.create("temp")
        assert registry.delete("temp")
        assert "temp" not in registry
        assert registry.graph_count() == 0

    def test_delete_nonexistent(self, registry):
        assert not registry.delete("ghost")

    def test_delete_default_promotes_next(self, registry):
        registry.create("a")
        registry.create("b")
        registry.set_default("a")
        registry.delete("a")
        assert registry.default_name == "b"

    def test_rename_graph(self, registry):
        registry.create("old")
        kg = registry.open("old")
        kg.add_node(Node(id="n1", node_type=NodeType.DOCUMENT, name="Doc"))
        kg.close()

        registry.rename("old", "new")
        assert "old" not in registry
        assert "new" in registry

        kg2 = registry.open("new")
        assert kg2.node_count() == 1

    def test_rename_updates_default(self, registry):
        registry.create("original", set_default=True)
        registry.rename("original", "renamed")
        assert registry.default_name == "renamed"

    def test_rename_nonexistent_raises(self, registry):
        with pytest.raises(KeyError):
            registry.rename("nope", "new")

    def test_rename_to_existing_raises(self, registry):
        registry.create("a")
        registry.create("b")
        with pytest.raises(ValueError):
            registry.rename("a", "b")

    def test_duplicate_name_raises(self, registry):
        registry.create("dup")
        with pytest.raises(ValueError):
            registry.create("dup")

    def test_invalid_name_raises(self, registry):
        with pytest.raises(ValueError):
            registry.create("")
        with pytest.raises(ValueError):
            registry.create("../sneaky")

    def test_contains(self, registry):
        registry.create("yes")
        assert "yes" in registry
        assert "no" not in registry

    def test_graph_count(self, registry):
        assert registry.graph_count() == 0
        registry.create("a")
        registry.create("b")
        assert registry.graph_count() == 2

    def test_graphs_are_isolated(self, registry):
        kg1 = registry.create("g1")
        kg2 = registry.create("g2")
        kg1.add_node(Node(id="n1", node_type=NodeType.DOCUMENT, name="D1"))
        kg2.add_node(Node(id="n2", node_type=NodeType.ENTITY, name="E1"))
        assert kg1.node_count() == 1
        assert kg2.node_count() == 1
        assert kg1.get_node("n2") is None
        assert kg2.get_node("n1") is None

    def test_persistence(self, reg_dir):
        r1 = KnowledgeGraphRegistry(reg_dir)
        r1.create("persist", description="test")
        kg = r1.open("persist")
        kg.add_node(Node(id="x", node_type=NodeType.TOPIC, name="T"))
        kg.close()

        r2 = KnowledgeGraphRegistry(reg_dir)
        assert "persist" in r2
        kg2 = r2.open("persist")
        assert kg2.node_count() == 1

    def test_repr(self, registry):
        registry.create("a")
        r = repr(registry)
        assert "KnowledgeGraphRegistry" in r
        assert "graphs=1" in r

    def test_list_graphs_metadata(self, registry):
        registry.create("proj", description="My project", set_default=True)
        graphs = registry.list_graphs()
        assert len(graphs) == 1
        g = graphs[0]
        assert g["name"] == "proj"
        assert g["description"] == "My project"
        assert g["is_default"] is True
        assert g["created_at"] > 0


# ---------------------------------------------------------------------------
# GraphLayer
# ---------------------------------------------------------------------------


class TestGraphLayer:
    def test_values(self):
        assert GraphLayer.BASE.value == "base"
        assert GraphLayer.ENRICHMENT.value == "enrichment"
        assert GraphLayer.EXTENDED.value == "extended"

    def test_from_string(self):
        assert GraphLayer("base") is GraphLayer.BASE
        assert GraphLayer("enrichment") is GraphLayer.ENRICHMENT
        assert GraphLayer("extended") is GraphLayer.EXTENDED


class TestGraphLayerCRUD:
    def test_node_default_layer(self, tmp_path):
        kg = KnowledgeGraph(tmp_path / "test.db")
        node = Node(id="n1", node_type=NodeType.DOCUMENT, name="doc")
        kg.add_node(node)
        got = kg.get_node("n1")
        assert got is not None
        assert got.layer == GraphLayer.BASE

    def test_node_explicit_layer(self, tmp_path):
        kg = KnowledgeGraph(tmp_path / "test.db")
        node = Node(
            id="n1",
            node_type=NodeType.ENTITY,
            name="ent",
            layer=GraphLayer.ENRICHMENT,
        )
        kg.add_node(node)
        got = kg.get_node("n1")
        assert got is not None
        assert got.layer == GraphLayer.ENRICHMENT

    def test_edge_default_layer(self, tmp_path):
        kg = KnowledgeGraph(tmp_path / "test.db")
        kg.add_node(Node(id="a", node_type=NodeType.DOCUMENT, name="a"))
        kg.add_node(Node(id="b", node_type=NodeType.DOCUMENT, name="b"))
        kg.add_edge(Edge(source="a", target="b", relation="related"))
        edges = kg.get_edges("a", direction="outgoing")
        assert len(edges) == 1
        assert edges[0].layer == GraphLayer.BASE

    def test_edge_explicit_layer(self, tmp_path):
        kg = KnowledgeGraph(tmp_path / "test.db")
        kg.add_node(Node(id="a", node_type=NodeType.DOCUMENT, name="a"))
        kg.add_node(Node(id="b", node_type=NodeType.DOCUMENT, name="b"))
        kg.add_edge(
            Edge(
                source="a",
                target="b",
                relation="mentions",
                layer=GraphLayer.ENRICHMENT,
            )
        )
        edges = kg.get_edges("a", direction="outgoing")
        assert edges[0].layer == GraphLayer.ENRICHMENT

    def test_list_nodes_filter_by_layer(self, tmp_path):
        kg = KnowledgeGraph(tmp_path / "test.db")
        kg.add_node(Node(id="n1", node_type=NodeType.DOCUMENT, name="a"))
        kg.add_node(Node(id="n2", node_type=NodeType.ENTITY, name="b", layer=GraphLayer.ENRICHMENT))
        kg.add_node(Node(id="n3", node_type=NodeType.TOPIC, name="c", layer=GraphLayer.ENRICHMENT))
        assert len(kg.list_nodes(layer=GraphLayer.BASE)) == 1
        assert len(kg.list_nodes(layer=GraphLayer.ENRICHMENT)) == 2
        assert len(kg.list_nodes(layer=GraphLayer.EXTENDED)) == 0

    def test_node_count_filter_by_layer(self, tmp_path):
        kg = KnowledgeGraph(tmp_path / "test.db")
        kg.add_node(Node(id="n1", node_type=NodeType.DOCUMENT, name="a"))
        kg.add_node(Node(id="n2", node_type=NodeType.ENTITY, name="b", layer=GraphLayer.ENRICHMENT))
        assert kg.node_count(layer=GraphLayer.BASE) == 1
        assert kg.node_count(layer=GraphLayer.ENRICHMENT) == 1
        assert kg.node_count(layer=GraphLayer.EXTENDED) == 0

    def test_node_count_combined_filters(self, tmp_path):
        kg = KnowledgeGraph(tmp_path / "test.db")
        kg.add_node(Node(id="n1", node_type=NodeType.DOCUMENT, name="a"))
        kg.add_node(Node(id="n2", node_type=NodeType.ENTITY, name="b", layer=GraphLayer.ENRICHMENT))
        kg.add_node(Node(id="n3", node_type=NodeType.ENTITY, name="c", layer=GraphLayer.BASE))
        assert kg.node_count(node_type=NodeType.ENTITY, layer=GraphLayer.ENRICHMENT) == 1
        assert kg.node_count(node_type=NodeType.ENTITY, layer=GraphLayer.BASE) == 1

    def test_search_filter_by_layer(self, tmp_path):
        kg = KnowledgeGraph(tmp_path / "test.db")
        kg.add_node(Node(id="n1", node_type=NodeType.DOCUMENT, name="python guide"))
        kg.add_node(
            Node(
                id="n2",
                node_type=NodeType.ENTITY,
                name="python",
                layer=GraphLayer.ENRICHMENT,
            )
        )
        base_results = kg.search("python", layer=GraphLayer.BASE)
        enrichment_results = kg.search("python", layer=GraphLayer.ENRICHMENT)
        assert len(base_results) == 1
        assert base_results[0].id == "n1"
        assert len(enrichment_results) == 1
        assert enrichment_results[0].id == "n2"

    def test_clear_layer(self, tmp_path):
        kg = KnowledgeGraph(tmp_path / "test.db")
        kg.add_node(Node(id="n1", node_type=NodeType.DOCUMENT, name="doc"))
        kg.add_node(
            Node(id="n2", node_type=NodeType.ENTITY, name="ent", layer=GraphLayer.ENRICHMENT)
        )
        kg.add_node(
            Node(id="n3", node_type=NodeType.TOPIC, name="top", layer=GraphLayer.ENRICHMENT)
        )
        kg.add_edge(
            Edge(source="n1", target="n2", relation="mentions", layer=GraphLayer.ENRICHMENT)
        )
        removed = kg.clear_layer(GraphLayer.ENRICHMENT)
        assert removed == 2
        assert kg.node_count() == 1
        assert kg.get_node("n1") is not None
        assert kg.get_node("n2") is None
        assert kg.get_node("n3") is None

    def test_stats_includes_layers(self, tmp_path):
        kg = KnowledgeGraph(tmp_path / "test.db")
        kg.add_node(Node(id="n1", node_type=NodeType.DOCUMENT, name="doc"))
        kg.add_node(
            Node(id="n2", node_type=NodeType.ENTITY, name="ent", layer=GraphLayer.ENRICHMENT)
        )
        s = kg.stats()
        assert "layers" in s
        assert s["layers"]["base"] == 1
        assert s["layers"]["enrichment"] == 1
        assert s["layers"]["extended"] == 0

    def test_migration_adds_layer_to_existing_db(self, tmp_path):
        """Opening a DB created without the layer column should auto-migrate."""
        import sqlite3

        db_path = tmp_path / "old.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE nodes (id TEXT PRIMARY KEY, node_type TEXT, name TEXT, "
            "content TEXT DEFAULT '', metadata TEXT DEFAULT '{}', embedding BLOB, "
            "created_at REAL, updated_at REAL)"
        )
        conn.execute(
            "CREATE TABLE edges (source TEXT, target TEXT, relation TEXT, "
            "weight REAL DEFAULT 1.0, metadata TEXT DEFAULT '{}', "
            "PRIMARY KEY (source, target, relation))"
        )
        conn.execute("INSERT INTO nodes VALUES ('x', 'document', 'old doc', '', '{}', NULL, 0, 0)")
        conn.commit()
        conn.close()

        # Now open with KnowledgeGraph — should migrate
        kg = KnowledgeGraph(db_path)
        node = kg.get_node("x")
        assert node is not None
        assert node.layer == GraphLayer.BASE
        kg.close()


# ---------------------------------------------------------------------------
# DECISION node type
# ---------------------------------------------------------------------------


class TestNodeTypeDecision:
    def test_decision_value(self):
        assert NodeType.DECISION.value == "decision"

    def test_from_string(self):
        assert NodeType("decision") is NodeType.DECISION


class TestDecisionNodes:
    def test_add_decision_node(self, tmp_path):
        kg = KnowledgeGraph(tmp_path / "test.db")
        node = Node(
            id="dec1",
            node_type=NodeType.DECISION,
            name="Enrichment: meeting notes",
            content="Extracted 3 entities from meeting notes.",
            metadata={"decision_type": "enrichment", "source": "enrichment_pipeline"},
            layer=GraphLayer.ENRICHMENT,
        )
        kg.add_node(node)
        got = kg.get_node("dec1")
        assert got is not None
        assert got.node_type == NodeType.DECISION
        assert got.layer == GraphLayer.ENRICHMENT
        assert got.metadata["decision_type"] == "enrichment"

    def test_stats_includes_decisions(self, tmp_path):
        kg = KnowledgeGraph(tmp_path / "test.db")
        kg.add_node(Node(id="d1", node_type=NodeType.DOCUMENT, name="doc"))
        kg.add_node(
            Node(id="dec1", node_type=NodeType.DECISION, name="dec", layer=GraphLayer.ENRICHMENT)
        )
        s = kg.stats()
        assert s["decisions"] == 1
        assert s["documents"] == 1
        assert s["nodes"] == 2

    def test_list_nodes_filter_decision(self, tmp_path):
        kg = KnowledgeGraph(tmp_path / "test.db")
        kg.add_node(Node(id="d1", node_type=NodeType.DOCUMENT, name="doc"))
        kg.add_node(Node(id="dec1", node_type=NodeType.DECISION, name="dec"))
        decisions = kg.list_nodes(node_type=NodeType.DECISION)
        assert len(decisions) == 1
        assert decisions[0].id == "dec1"

    def test_search_finds_decision(self, tmp_path):
        kg = KnowledgeGraph(tmp_path / "test.db")
        kg.add_node(
            Node(
                id="dec1",
                node_type=NodeType.DECISION,
                name="Entity resolution: Alex",
                content="User confirmed Alex = Alex Torres.",
            )
        )
        results = kg.search("Alex")
        assert len(results) == 1
        assert results[0].node_type == NodeType.DECISION


class TestDecisionTrail:
    def test_decision_trail_returns_linked_decisions(self, tmp_path):
        kg = KnowledgeGraph(tmp_path / "test.db")
        kg.add_node(Node(id="e1", node_type=NodeType.ENTITY, name="Alex Torres"))
        kg.add_node(
            Node(
                id="dec1",
                node_type=NodeType.DECISION,
                name="Enrichment: meeting notes",
                layer=GraphLayer.ENRICHMENT,
            )
        )
        kg.add_node(
            Node(
                id="dec2",
                node_type=NodeType.DECISION,
                name="Q&A: Who is Alex?",
                layer=GraphLayer.EXTENDED,
            )
        )
        kg.add_edge(Edge(source="dec1", target="e1", relation="produced"))
        kg.add_edge(Edge(source="dec2", target="e1", relation="resolves"))

        trail = kg.decision_trail("e1")
        assert len(trail) == 2
        # newest first
        assert trail[0].id == "dec2"
        assert trail[1].id == "dec1"

    def test_decision_trail_empty(self, tmp_path):
        kg = KnowledgeGraph(tmp_path / "test.db")
        kg.add_node(Node(id="e1", node_type=NodeType.ENTITY, name="Alex"))
        trail = kg.decision_trail("e1")
        assert trail == []

    def test_decision_trail_ignores_non_decision_neighbors(self, tmp_path):
        kg = KnowledgeGraph(tmp_path / "test.db")
        kg.add_node(Node(id="d1", node_type=NodeType.DOCUMENT, name="doc"))
        kg.add_node(Node(id="e1", node_type=NodeType.ENTITY, name="ent"))
        kg.add_node(Node(id="dec1", node_type=NodeType.DECISION, name="dec"))
        kg.add_edge(Edge(source="d1", target="e1", relation="mentions"))
        kg.add_edge(Edge(source="dec1", target="e1", relation="produced"))

        trail = kg.decision_trail("e1")
        assert len(trail) == 1
        assert trail[0].id == "dec1"


class TestRevertDecision:
    def test_revert_deletes_decision_and_edges(self, tmp_path):
        kg = KnowledgeGraph(tmp_path / "test.db")
        kg.add_node(Node(id="d1", node_type=NodeType.DOCUMENT, name="doc"))
        kg.add_node(Node(id="e1", node_type=NodeType.ENTITY, name="ent"))
        kg.add_node(Node(id="dec1", node_type=NodeType.DECISION, name="dec"))
        kg.add_edge(Edge(source="dec1", target="d1", relation="derived_from"))
        kg.add_edge(Edge(source="dec1", target="e1", relation="produced"))
        # Also an independent edge keeping e1 alive
        kg.add_edge(Edge(source="d1", target="e1", relation="mentions"))

        result = kg.revert_decision("dec1")
        assert result is True
        assert kg.get_node("dec1") is None
        # e1 still has an edge from d1, so it survives
        assert kg.get_node("e1") is not None
        assert kg.get_node("d1") is not None

    def test_revert_cleans_up_orphaned_entities(self, tmp_path):
        kg = KnowledgeGraph(tmp_path / "test.db")
        kg.add_node(Node(id="d1", node_type=NodeType.DOCUMENT, name="doc"))
        kg.add_node(
            Node(id="e1", node_type=NodeType.ENTITY, name="orphan ent", layer=GraphLayer.ENRICHMENT)
        )
        kg.add_node(Node(id="dec1", node_type=NodeType.DECISION, name="dec"))
        kg.add_edge(Edge(source="dec1", target="d1", relation="derived_from"))
        kg.add_edge(Edge(source="dec1", target="e1", relation="produced"))

        kg.revert_decision("dec1")
        assert kg.get_node("dec1") is None
        # e1 had no other edges, so it's cleaned up
        assert kg.get_node("e1") is None
        # d1 is a DOCUMENT — never auto-removed
        assert kg.get_node("d1") is not None

    def test_revert_nonexistent_returns_false(self, tmp_path):
        kg = KnowledgeGraph(tmp_path / "test.db")
        assert kg.revert_decision("nope") is False

    def test_revert_non_decision_returns_false(self, tmp_path):
        kg = KnowledgeGraph(tmp_path / "test.db")
        kg.add_node(Node(id="e1", node_type=NodeType.ENTITY, name="ent"))
        assert kg.revert_decision("e1") is False


class TestAnswerQuestionDecision:
    def test_answer_question_creates_decision_node(self, tmp_path):
        """Answering an enrichment question records a DECISION node."""
        from talk_box.enrichment_qa import (
            EnrichmentQuestion,
            QuestionOption,
            QuestionType,
        )

        kg = KnowledgeGraph(tmp_path / "test.db")
        # Create an entity referenced by the question
        kg.add_node(Node(id="entity-alex", node_type=NodeType.ENTITY, name="Alex"))

        # Seed a question into the queue
        q = EnrichmentQuestion(
            id="eq-001",
            question_type=QuestionType.ENTITY_AMBIGUITY,
            text='Who is "Alex"?',
            options=[
                QuestionOption(label="Alex Torres", node_ids=["entity-alex"]),
                QuestionOption(label="Alex Kim"),
            ],
            node_ids=["entity-alex"],
            confusion_impact=0.8,
        )
        kg._get_question_queue().add(q)

        # Answer the question
        result = kg.answer_question("eq-001", choice=0)
        assert result is not None

        # A DECISION node should now exist
        decisions = kg.list_nodes(node_type=NodeType.DECISION)
        assert len(decisions) == 1
        dec = decisions[0]
        assert dec.metadata["decision_type"] == "entity_ambiguity"
        assert dec.metadata["source"] == "enrichment_qa"
        assert dec.layer == GraphLayer.EXTENDED

        # Decision should be linked to entity-alex
        trail = kg.decision_trail("entity-alex")
        assert len(trail) == 1
        assert trail[0].id == dec.id


# ---------------------------------------------------------------------------
# Ontology
# ---------------------------------------------------------------------------


class TestEntityTypeDef:
    def test_basic(self):
        etd = EntityTypeDef(description="A person", properties=["role"])
        assert etd.description == "A person"
        assert etd.properties == ["role"]
        assert etd.parent is None

    def test_with_parent(self):
        etd = EntityTypeDef(description="A project", parent="initiative")
        assert etd.parent == "initiative"

    def test_roundtrip(self):
        etd = EntityTypeDef(description="A person", properties=["role", "email"], parent="agent")
        d = etd.to_dict()
        restored = EntityTypeDef.from_dict(d)
        assert restored == etd


class TestRelationTypeDef:
    def test_basic(self):
        rtd = RelationTypeDef(description="Leads", source_type="person", target_type="project")
        assert rtd.source_type == "person"
        assert rtd.target_type == "project"

    def test_roundtrip(self):
        rtd = RelationTypeDef(
            description="Drives",
            source_type="metric",
            target_type="metric",
            properties=["strength"],
        )
        d = rtd.to_dict()
        restored = RelationTypeDef.from_dict(d)
        assert restored == rtd


class TestOntology:
    def test_empty(self):
        o = Ontology()
        assert o.entity_types == {}
        assert o.relation_types == {}

    def test_roundtrip(self):
        o = Ontology(
            entity_types={"person": EntityTypeDef(description="Human", properties=["role"])},
            relation_types={"leads": RelationTypeDef(source_type="person", target_type="project")},
        )
        d = o.to_dict()
        restored = Ontology.from_dict(d)
        assert restored.entity_types["person"].description == "Human"
        assert restored.relation_types["leads"].source_type == "person"

    def test_ancestors_simple(self):
        o = Ontology(
            entity_types={
                "initiative": EntityTypeDef(description="Top-level"),
                "project": EntityTypeDef(description="Sub-initiative", parent="initiative"),
            },
        )
        assert o.ancestors("project") == ["initiative"]
        assert o.ancestors("initiative") == []

    def test_ancestors_chain(self):
        o = Ontology(
            entity_types={
                "thing": EntityTypeDef(description="Root"),
                "initiative": EntityTypeDef(description="Mid", parent="thing"),
                "project": EntityTypeDef(description="Leaf", parent="initiative"),
            },
        )
        assert o.ancestors("project") == ["initiative", "thing"]

    def test_is_subtype(self):
        o = Ontology(
            entity_types={
                "initiative": EntityTypeDef(description="Top"),
                "project": EntityTypeDef(description="Sub", parent="initiative"),
            },
        )
        assert o.is_subtype("project", "initiative") is True
        assert o.is_subtype("project", "project") is True
        assert o.is_subtype("initiative", "project") is False

    def test_validate_node_accepts_non_entity(self):
        o = Ontology(entity_types={"person": EntityTypeDef()})
        node = Node(id="d1", node_type=NodeType.DOCUMENT, name="doc")
        assert o.validate_node(node) == []

    def test_validate_node_accepts_known_type(self):
        o = Ontology(entity_types={"person": EntityTypeDef(properties=["role"])})
        node = Node(
            id="e1",
            node_type=NodeType.ENTITY,
            name="Alice",
            metadata={"entity_type": "person", "role": "engineer"},
        )
        assert o.validate_node(node) == []

    def test_validate_node_warns_unknown_type(self):
        o = Ontology(entity_types={"person": EntityTypeDef()})
        node = Node(
            id="e1",
            node_type=NodeType.ENTITY,
            name="Widget",
            metadata={"entity_type": "widget"},
        )
        msgs = o.validate_node(node)
        assert len(msgs) == 1
        assert "widget" in msgs[0]

    def test_validate_node_warns_unknown_property(self):
        o = Ontology(entity_types={"person": EntityTypeDef(properties=["role"])})
        node = Node(
            id="e1",
            node_type=NodeType.ENTITY,
            name="Alice",
            metadata={"entity_type": "person", "role": "eng", "shoe_size": "10"},
        )
        msgs = o.validate_node(node)
        assert any("shoe_size" in m for m in msgs)

    def test_validate_edge_accepts_correct_types(self):
        o = Ontology(
            relation_types={
                "leads": RelationTypeDef(source_type="person", target_type="project"),
            },
        )
        edge = Edge(source="e1", target="e2", relation="leads")
        src = Node(id="e1", node_type=NodeType.ENTITY, name="A", metadata={"entity_type": "person"})
        tgt = Node(
            id="e2", node_type=NodeType.ENTITY, name="B", metadata={"entity_type": "project"}
        )
        assert o.validate_edge(edge, source_node=src, target_node=tgt) == []

    def test_validate_edge_warns_wrong_source_type(self):
        o = Ontology(
            relation_types={
                "leads": RelationTypeDef(source_type="person", target_type="project"),
            },
        )
        edge = Edge(source="e1", target="e2", relation="leads")
        src = Node(id="e1", node_type=NodeType.ENTITY, name="A", metadata={"entity_type": "metric"})
        tgt = Node(
            id="e2", node_type=NodeType.ENTITY, name="B", metadata={"entity_type": "project"}
        )
        msgs = o.validate_edge(edge, source_node=src, target_node=tgt)
        assert len(msgs) == 1
        assert "source type" in msgs[0]

    def test_validate_edge_respects_inheritance(self):
        o = Ontology(
            entity_types={
                "initiative": EntityTypeDef(),
                "project": EntityTypeDef(parent="initiative"),
            },
            relation_types={
                "reviews": RelationTypeDef(target_type="initiative"),
            },
        )
        edge = Edge(source="e1", target="e2", relation="reviews")
        src = Node(id="e1", node_type=NodeType.ENTITY, name="A", metadata={"entity_type": "person"})
        tgt = Node(
            id="e2", node_type=NodeType.ENTITY, name="B", metadata={"entity_type": "project"}
        )
        # project is a subtype of initiative, so this should be valid
        assert o.validate_edge(edge, source_node=src, target_node=tgt) == []


class TestGeneralOntology:
    def test_has_common_types(self):
        o = general_ontology()
        assert "person" in o.entity_types
        assert "organization" in o.entity_types
        assert "project" in o.entity_types
        assert "concept" in o.entity_types
        assert "technology" in o.entity_types
        assert "metric" in o.entity_types

    def test_has_common_relations(self):
        o = general_ontology()
        assert "leads" in o.relation_types
        assert "works_at" in o.relation_types
        assert "related_to" in o.relation_types
        assert "drives" in o.relation_types


class TestKGOntologyIntegration:
    def test_kg_default_empty_ontology(self, tmp_path):
        kg = KnowledgeGraph(tmp_path / "test.db")
        assert kg.ontology.entity_types == {}
        assert kg.ontology.relation_types == {}

    def test_kg_with_ontology(self, tmp_path):
        o = general_ontology()
        kg = KnowledgeGraph(tmp_path / "test.db", ontology=o)
        assert "person" in kg.ontology.entity_types

    def test_ontology_persists_across_reopen(self, tmp_path):
        db = tmp_path / "test.db"
        o = general_ontology()
        kg = KnowledgeGraph(db, ontology=o)
        kg.close()

        kg2 = KnowledgeGraph(db)
        assert "person" in kg2.ontology.entity_types
        assert "leads" in kg2.ontology.relation_types
        kg2.close()

    def test_set_ontology(self, tmp_path):
        kg = KnowledgeGraph(tmp_path / "test.db")
        assert kg.ontology.entity_types == {}
        kg.set_ontology(general_ontology())
        assert "person" in kg.ontology.entity_types

    def test_strict_mode_rejects_invalid_node(self, tmp_path):
        o = Ontology(entity_types={"person": EntityTypeDef(properties=["role"])})
        kg = KnowledgeGraph(tmp_path / "test.db", ontology=o, strict=True)
        node = Node(
            id="e1",
            node_type=NodeType.ENTITY,
            name="Widget",
            metadata={"entity_type": "widget"},
        )
        with pytest.raises(ValueError, match="widget"):
            kg.add_node(node)

    def test_soft_mode_warns_invalid_node(self, tmp_path):
        o = Ontology(entity_types={"person": EntityTypeDef(properties=["role"])})
        kg = KnowledgeGraph(tmp_path / "test.db", ontology=o)
        node = Node(
            id="e1",
            node_type=NodeType.ENTITY,
            name="Widget",
            metadata={"entity_type": "widget"},
        )
        with pytest.warns(UserWarning, match="widget"):
            kg.add_node(node)
        # Node should still be added in soft mode
        assert kg.get_node("e1") is not None

    def test_strict_mode_rejects_invalid_edge(self, tmp_path):
        o = Ontology(
            relation_types={
                "leads": RelationTypeDef(source_type="person", target_type="project"),
            },
        )
        kg = KnowledgeGraph(tmp_path / "test.db", ontology=o, strict=True)
        kg.add_node(
            Node(
                id="e1",
                node_type=NodeType.ENTITY,
                name="A",
                metadata={"entity_type": "metric"},
            )
        )
        kg.add_node(
            Node(
                id="e2",
                node_type=NodeType.ENTITY,
                name="B",
                metadata={"entity_type": "project"},
            )
        )
        with pytest.raises(ValueError, match="source type"):
            kg.add_edge(Edge(source="e1", target="e2", relation="leads"))

    def test_soft_mode_warns_invalid_edge(self, tmp_path):
        o = Ontology(
            relation_types={
                "leads": RelationTypeDef(source_type="person", target_type="project"),
            },
        )
        kg = KnowledgeGraph(tmp_path / "test.db", ontology=o)
        kg.add_node(
            Node(
                id="e1",
                node_type=NodeType.ENTITY,
                name="A",
                metadata={"entity_type": "metric"},
            )
        )
        kg.add_node(
            Node(
                id="e2",
                node_type=NodeType.ENTITY,
                name="B",
                metadata={"entity_type": "project"},
            )
        )
        with pytest.warns(UserWarning, match="source type"):
            kg.add_edge(Edge(source="e1", target="e2", relation="leads"))
        # Edge should still be added in soft mode
        edges = kg.get_edges("e1", direction="outgoing")
        assert len(edges) == 1

    def test_valid_nodes_and_edges_no_warnings(self, tmp_path):
        o = Ontology(
            entity_types={
                "person": EntityTypeDef(properties=["role"]),
                "project": EntityTypeDef(properties=["status"]),
            },
            relation_types={
                "leads": RelationTypeDef(source_type="person", target_type="project"),
            },
        )
        kg = KnowledgeGraph(tmp_path / "test.db", ontology=o, strict=True)
        kg.add_node(
            Node(
                id="p1",
                node_type=NodeType.ENTITY,
                name="Alice",
                metadata={"entity_type": "person", "role": "eng"},
            )
        )
        kg.add_node(
            Node(
                id="pr1",
                node_type=NodeType.ENTITY,
                name="Alpha",
                metadata={"entity_type": "project", "status": "active"},
            )
        )
        kg.add_edge(Edge(source="p1", target="pr1", relation="leads"))
        assert kg.get_node("p1") is not None
        assert len(kg.get_edges("p1", direction="outgoing")) == 1
