"""Tests for talk_box.kg_visualization module."""

from __future__ import annotations

from pathlib import Path

import pytest

from talk_box import Edge, KnowledgeGraph, Node, NodeType
from talk_box.kg_visualization import (
    VisEdge,
    VisGraph,
    VisNode,
    _build_legend_html,
    _freshness_color,
    _node_size,
    _sanitize_mermaid_id,
    _sanitize_mermaid_label,
    prepare_vis_graph,
    to_html,
    to_mermaid,
    visualize,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def empty_kg() -> KnowledgeGraph:
    """Empty in-memory knowledge graph."""
    return KnowledgeGraph(":memory:")


@pytest.fixture()
def sample_kg() -> KnowledgeGraph:
    """KG with a mix of nodes and edges for visualization testing."""
    kg = KnowledgeGraph(":memory:")
    kg.add_node(
        Node(id="doc-1", node_type=NodeType.DOCUMENT, name="README.md", content="# Talk Box")
    )
    kg.add_node(Node(id="e-python", node_type=NodeType.ENTITY, name="Python"))
    kg.add_node(Node(id="e-chatlas", node_type=NodeType.ENTITY, name="chatlas"))
    kg.add_node(Node(id="t-ai", node_type=NodeType.TOPIC, name="AI"))
    kg.add_edge(Edge(source="doc-1", target="e-python", relation="mentions", weight=0.9))
    kg.add_edge(Edge(source="doc-1", target="e-chatlas", relation="mentions", weight=0.7))
    kg.add_edge(Edge(source="e-python", target="t-ai", relation="related_to", weight=0.5))
    return kg


# ---------------------------------------------------------------------------
# TestSanitizeMermaid
# ---------------------------------------------------------------------------


class TestSanitizeMermaid:
    """Tests for Mermaid sanitization helpers."""

    def test_id_alphanumeric(self):
        assert _sanitize_mermaid_id("simple") == "simple"

    def test_id_with_special_chars(self):
        assert _sanitize_mermaid_id("doc-1.foo/bar") == "doc_1_foo_bar"

    def test_label_quotes(self):
        assert _sanitize_mermaid_label('say "hello"') == "say 'hello'"

    def test_label_newlines(self):
        assert _sanitize_mermaid_label("line1\nline2") == "line1 line2"


# ---------------------------------------------------------------------------
# TestToMermaid
# ---------------------------------------------------------------------------


class TestToMermaid:
    """Tests for to_mermaid function."""

    def test_basic_output(self, sample_kg: KnowledgeGraph):
        result = to_mermaid(sample_kg)
        assert result.startswith("graph LR")
        assert "README" in result
        assert "Python" in result

    def test_direction(self, sample_kg: KnowledgeGraph):
        result = to_mermaid(sample_kg, direction="TB")
        assert result.startswith("graph TB")

    def test_empty_graph(self, empty_kg: KnowledgeGraph):
        result = to_mermaid(empty_kg)
        assert "No nodes" in result

    def test_edges_included(self, sample_kg: KnowledgeGraph):
        result = to_mermaid(sample_kg)
        assert "mentions" in result

    def test_weight_shown(self, sample_kg: KnowledgeGraph):
        result = to_mermaid(sample_kg, show_weights=True)
        assert "0.9" in result

    def test_weight_hidden(self, sample_kg: KnowledgeGraph):
        result = to_mermaid(sample_kg, show_weights=False)
        assert "0.9" not in result

    def test_node_type_filter(self, sample_kg: KnowledgeGraph):
        result = to_mermaid(sample_kg, node_type=NodeType.ENTITY)
        assert "Python" in result
        assert "README" not in result

    def test_max_label_truncation(self, empty_kg: KnowledgeGraph):
        empty_kg.add_node(Node(id="long", node_type=NodeType.ENTITY, name="A" * 50))
        result = to_mermaid(empty_kg, max_label_length=10)
        assert "…" in result

    def test_style_classes(self, sample_kg: KnowledgeGraph):
        result = to_mermaid(sample_kg)
        assert "style" in result
        assert "fill:" in result

    def test_node_shapes(self, sample_kg: KnowledgeGraph):
        result = to_mermaid(sample_kg)
        # Documents use [ ], entities use ( ), topics use {{ }}
        assert '["' in result  # document shape
        assert '("' in result  # entity shape
        assert '{{"' in result  # topic shape


# ---------------------------------------------------------------------------
# TestVisNode
# ---------------------------------------------------------------------------


class TestVisNode:
    """Tests for VisNode frozen dataclass."""

    def test_creation(self):
        vn = VisNode(id="e1", label="Python", node_type="entity", color="#50C878", size=20)
        assert vn.id == "e1"
        assert vn.label == "Python"
        assert vn.node_type == "entity"

    def test_defaults(self):
        vn = VisNode(id="e1", label="X", node_type="entity")
        assert vn.color == "#4A90D9"
        assert vn.size == 20
        assert vn.metadata == {}


# ---------------------------------------------------------------------------
# TestVisEdge
# ---------------------------------------------------------------------------


class TestVisEdge:
    """Tests for VisEdge frozen dataclass."""

    def test_creation(self):
        ve = VisEdge(source="a", target="b", label="rel", weight=0.8)
        assert ve.source == "a"
        assert ve.weight == 0.8

    def test_defaults(self):
        ve = VisEdge(source="a", target="b", label="rel")
        assert ve.weight == 1.0
        assert ve.color == "#888888"


# ---------------------------------------------------------------------------
# TestVisGraph
# ---------------------------------------------------------------------------


class TestVisGraph:
    """Tests for VisGraph frozen dataclass."""

    def test_counts(self):
        vg = VisGraph(
            nodes=[VisNode(id="a", label="A", node_type="entity")],
            edges=[VisEdge(source="a", target="b", label="r")],
        )
        assert vg.node_count == 1
        assert vg.edge_count == 1

    def test_empty(self):
        vg = VisGraph()
        assert vg.node_count == 0
        assert vg.edge_count == 0

    def test_to_dict(self):
        vg = VisGraph(
            nodes=[VisNode(id="a", label="A", node_type="entity")],
            edges=[VisEdge(source="a", target="b", label="r")],
            title="Test",
        )
        d = vg.to_dict()
        assert d["title"] == "Test"
        assert len(d["nodes"]) == 1
        assert len(d["edges"]) == 1
        assert d["nodes"][0]["id"] == "a"
        assert d["edges"][0]["source"] == "a"

    def test_default_title(self):
        vg = VisGraph()
        assert vg.title == "Knowledge Graph"


# ---------------------------------------------------------------------------
# TestFreshnessColor
# ---------------------------------------------------------------------------


class TestFreshnessColor:
    """Tests for _freshness_color helper."""

    def test_fresh_node(self):
        import time

        node = Node(id="n", node_type=NodeType.ENTITY, name="N", updated_at=time.time())
        color = _freshness_color(node)
        assert color == "#50C878"  # green

    def test_stale_node(self):
        import time

        node = Node(id="n", node_type=NodeType.ENTITY, name="N")
        # Manually set old timestamp
        object.__setattr__(node, "updated_at", time.time() - 86_400 * 60)
        color = _freshness_color(node)
        assert color == "#FF4444"  # red

    def test_aging_node(self):
        import time

        node = Node(id="n", node_type=NodeType.ENTITY, name="N")
        object.__setattr__(node, "updated_at", time.time() - 86_400 * 15)
        color = _freshness_color(node)
        assert color == "#FFD700"  # yellow


# ---------------------------------------------------------------------------
# TestNodeSize
# ---------------------------------------------------------------------------


class TestNodeSize:
    """Tests for _node_size helper."""

    def test_no_edges(self, empty_kg: KnowledgeGraph):
        empty_kg.add_node(Node(id="n", node_type=NodeType.ENTITY, name="N"))
        node = empty_kg.get_node("n")
        assert node is not None
        size = _node_size(empty_kg, node)
        assert size == 15  # base only

    def test_with_edges(self, sample_kg: KnowledgeGraph):
        node = sample_kg.get_node("doc-1")
        assert node is not None
        size = _node_size(sample_kg, node)
        assert size > 15  # base + edge bonus


# ---------------------------------------------------------------------------
# TestPrepareVisGraph
# ---------------------------------------------------------------------------


class TestPrepareVisGraph:
    """Tests for prepare_vis_graph function."""

    def test_basic(self, sample_kg: KnowledgeGraph):
        graph = prepare_vis_graph(sample_kg)
        assert graph.node_count == 4
        assert graph.edge_count == 3

    def test_empty_graph(self, empty_kg: KnowledgeGraph):
        graph = prepare_vis_graph(empty_kg)
        assert graph.node_count == 0
        assert graph.edge_count == 0

    def test_node_type_filter(self, sample_kg: KnowledgeGraph):
        graph = prepare_vis_graph(sample_kg, node_type=NodeType.ENTITY)
        assert graph.node_count == 2
        assert all(n.node_type == "entity" for n in graph.nodes)

    def test_color_by_type(self, sample_kg: KnowledgeGraph):
        graph = prepare_vis_graph(sample_kg, color_by="type")
        colors = {n.color for n in graph.nodes}
        assert len(colors) > 1  # Different colors for different types

    def test_color_by_freshness(self, sample_kg: KnowledgeGraph):
        graph = prepare_vis_graph(sample_kg, color_by="freshness")
        # All nodes are fresh (just created)
        assert all(n.color == "#50C878" for n in graph.nodes)

    def test_custom_title(self, sample_kg: KnowledgeGraph):
        graph = prepare_vis_graph(sample_kg, title="My Graph")
        assert graph.title == "My Graph"

    def test_max_nodes(self, sample_kg: KnowledgeGraph):
        graph = prepare_vis_graph(sample_kg, max_nodes=2)
        assert graph.node_count <= 2

    def test_no_duplicate_edges(self, sample_kg: KnowledgeGraph):
        graph = prepare_vis_graph(sample_kg)
        edge_keys = [(e.source, e.target, e.label) for e in graph.edges]
        assert len(edge_keys) == len(set(edge_keys))

    def test_node_sizes_vary(self, sample_kg: KnowledgeGraph):
        graph = prepare_vis_graph(sample_kg)
        sizes = {n.size for n in graph.nodes}
        # doc-1 has 2 outgoing edges, others have fewer
        assert len(sizes) > 1


# ---------------------------------------------------------------------------
# TestBuildLegendHtml
# ---------------------------------------------------------------------------


class TestBuildLegendHtml:
    """Tests for _build_legend_html helper."""

    def test_type_legend(self):
        legend = _build_legend_html("type")
        assert "Document" in legend
        assert "Entity" in legend
        assert "Topic" in legend

    def test_freshness_legend(self):
        legend = _build_legend_html("freshness")
        assert "Fresh" in legend
        assert "Aging" in legend
        assert "Stale" in legend
        assert "Unknown" in legend


# ---------------------------------------------------------------------------
# TestToHtml
# ---------------------------------------------------------------------------


class TestToHtml:
    """Tests for to_html function."""

    def test_produces_html(self, sample_kg: KnowledgeGraph):
        result = to_html(sample_kg)
        assert "<!DOCTYPE html>" in result
        assert "Knowledge Graph" in result
        assert "Python" in result

    def test_custom_title(self, sample_kg: KnowledgeGraph):
        result = to_html(sample_kg, title="My Graph")
        assert "My Graph" in result

    def test_color_by_freshness(self, sample_kg: KnowledgeGraph):
        result = to_html(sample_kg, color_by="freshness")
        assert "Fresh" in result

    def test_empty_graph(self, empty_kg: KnowledgeGraph):
        result = to_html(empty_kg)
        assert "<!DOCTYPE html>" in result

    def test_html_escapes_title(self, sample_kg: KnowledgeGraph):
        result = to_html(sample_kg, title='<script>alert("xss")</script>')
        assert "<script>alert" not in result
        assert "&lt;script&gt;" in result

    def test_contains_nodes_json(self, sample_kg: KnowledgeGraph):
        result = to_html(sample_kg)
        assert '"id"' in result
        assert '"label"' in result

    def test_contains_edges_json(self, sample_kg: KnowledgeGraph):
        result = to_html(sample_kg)
        assert '"source"' in result
        assert '"target"' in result


# ---------------------------------------------------------------------------
# TestVisualize
# ---------------------------------------------------------------------------


class TestVisualize:
    """Tests for visualize function."""

    def test_creates_file(self, sample_kg: KnowledgeGraph, tmp_path: Path):
        out = tmp_path / "graph.html"
        result = visualize(sample_kg, output=out, open_browser=False)
        assert result == out
        assert out.exists()
        content = out.read_text()
        assert "<!DOCTYPE html>" in content

    def test_default_path(self, sample_kg: KnowledgeGraph):
        result = visualize(sample_kg, open_browser=False)
        assert result.exists()
        assert result.suffix == ".html"
        # Clean up
        result.unlink()

    def test_custom_color_by(self, sample_kg: KnowledgeGraph, tmp_path: Path):
        out = tmp_path / "fresh.html"
        visualize(sample_kg, output=out, color_by="freshness", open_browser=False)
        content = out.read_text()
        assert "Fresh" in content

    def test_custom_title(self, sample_kg: KnowledgeGraph, tmp_path: Path):
        out = tmp_path / "titled.html"
        visualize(sample_kg, output=out, title="Test Title", open_browser=False)
        content = out.read_text()
        assert "Test Title" in content


# ---------------------------------------------------------------------------
# TestTopLevelImport
# ---------------------------------------------------------------------------


class TestTopLevelImport:
    """Tests that visualization exports are accessible from talk_box."""

    def test_import_to_mermaid(self):
        from talk_box import to_mermaid as fn

        assert callable(fn)

    def test_import_to_html(self):
        from talk_box import to_html as fn

        assert callable(fn)

    def test_import_visualize(self):
        from talk_box import visualize as fn

        assert callable(fn)

    def test_import_prepare_vis_graph(self):
        from talk_box import prepare_vis_graph as fn

        assert callable(fn)

    def test_import_vis_node(self):
        from talk_box import VisNode as cls

        vn = cls(id="x", label="X", node_type="entity")
        assert vn.id == "x"

    def test_import_vis_edge(self):
        from talk_box import VisEdge as cls

        ve = cls(source="a", target="b", label="r")
        assert ve.source == "a"

    def test_import_vis_graph(self):
        from talk_box import VisGraph as cls

        vg = cls()
        assert vg.node_count == 0
