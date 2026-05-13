import math

import pytest

from talk_box.knowledge_graph import (
    Edge,
    KnowledgeGraph,
    Node,
    NodeType,
    _blob_to_floats,
    _floats_to_blob,
    cosine_similarity,
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
        assert s == {"nodes": 0, "edges": 0, "documents": 0, "entities": 0, "topics": 0}

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
