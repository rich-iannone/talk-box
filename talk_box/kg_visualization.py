"""Knowledge graph visualization: Mermaid and interactive HTML renders."""

from __future__ import annotations

import html
import re
import time
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from talk_box.knowledge_graph import KnowledgeGraph, Node, NodeType

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NODE_SHAPES: dict[NodeType, tuple[str, str]] = {
    NodeType.DOCUMENT: ("[", "]"),
    NodeType.ENTITY: ("(", ")"),
    NodeType.TOPIC: ("{{", "}}"),
}

_NODE_COLORS: dict[NodeType, str] = {
    NodeType.DOCUMENT: "#4A90D9",
    NodeType.ENTITY: "#50C878",
    NodeType.TOPIC: "#FF8C42",
}

_FRESHNESS_COLORS: dict[str, str] = {
    "fresh": "#50C878",
    "aging": "#FFD700",
    "stale": "#FF4444",
    "unknown": "#999999",
}

# ---------------------------------------------------------------------------
# Mermaid rendering
# ---------------------------------------------------------------------------


def _sanitize_mermaid_id(node_id: str) -> str:
    """Make a node ID safe for Mermaid syntax."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", node_id)


def _sanitize_mermaid_label(label: str) -> str:
    """Escape a label for Mermaid."""
    return label.replace('"', "'").replace("\n", " ")


def to_mermaid(
    kg: KnowledgeGraph,
    *,
    direction: str = "LR",
    max_nodes: int = 100,
    max_label_length: int = 30,
    show_weights: bool = True,
    node_type: NodeType | None = None,
) -> str:
    """Render a knowledge graph as a Mermaid diagram string.

    Produces a Mermaid ``graph`` definition suitable for rendering in
    Markdown, Quarto documents, or any Mermaid-compatible viewer.

    Parameters
    ----------
    kg
        The knowledge graph to render.
    direction
        Graph direction: ``"LR"`` (left-right), ``"TB"`` (top-bottom),
        ``"RL"``, or ``"BT"``.
    max_nodes
        Maximum number of nodes to include (default 100).
    max_label_length
        Truncate node labels longer than this.
    show_weights
        If ``True``, annotate edges with their weight.
    node_type
        If provided, only include nodes of this type.

    Returns
    -------
    str
        A valid Mermaid graph definition.

    Examples
    --------
    ```python
    import talk_box as tb

    kg = tb.KnowledgeGraph(":memory:")
    kg.add_node(tb.Node(id="doc-1", node_type=tb.NodeType.DOCUMENT, name="README"))
    kg.add_node(tb.Node(id="e-py", node_type=tb.NodeType.ENTITY, name="Python"))
    kg.add_edge(tb.Edge(source="doc-1", target="e-py", relation="mentions"))

    print(tb.to_mermaid(kg))
    ```
    """
    nodes = kg.list_nodes(node_type=node_type, limit=max_nodes)
    if not nodes:
        return f"graph {direction}\n    empty[No nodes]"

    node_ids = {n.id for n in nodes}
    lines = [f"graph {direction}"]

    # Emit nodes with type-specific shapes
    for node in nodes:
        safe_id = _sanitize_mermaid_id(node.id)
        label = node.name[:max_label_length]
        if len(node.name) > max_label_length:
            label += "…"
        label = _sanitize_mermaid_label(label)
        left, right = _NODE_SHAPES.get(node.node_type, ("(", ")"))
        lines.append(f'    {safe_id}{left}"{label}"{right}')

    # Emit edges
    for node in nodes:
        edges = kg.get_edges(node.id, direction="outgoing")
        for edge in edges:
            if edge.target not in node_ids:
                continue
            src = _sanitize_mermaid_id(edge.source)
            tgt = _sanitize_mermaid_id(edge.target)
            rel = _sanitize_mermaid_label(edge.relation)
            if show_weights and edge.weight != 1.0:
                rel += f" ({edge.weight:.1f})"
            lines.append(f"    {src} -->|{rel}| {tgt}")

    # Style classes by node type
    for ntype, color in _NODE_COLORS.items():
        ids = [_sanitize_mermaid_id(n.id) for n in nodes if n.node_type == ntype]
        if ids:
            lines.append(f"    style {','.join(ids)} fill:{color},color:#fff")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Interactive HTML visualization
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VisNode:
    """A node prepared for visualization.

    Parameters
    ----------
    id
        Node identifier.
    label
        Display label.
    node_type
        Node type string.
    color
        CSS color for the node.
    size
        Relative node size.
    metadata
        Extra display metadata.

    Examples
    --------
    ```python
    import talk_box as tb

    vn = tb.VisNode(id="e1", label="Python", node_type="entity", color="#50C878", size=20)
    vn.label  # "Python"
    ```
    """

    id: str
    label: str
    node_type: str
    color: str = "#4A90D9"
    size: int = 20
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VisEdge:
    """An edge prepared for visualization.

    Parameters
    ----------
    source
        Source node ID.
    target
        Target node ID.
    label
        Edge label (relation).
    weight
        Edge weight (affects thickness).
    color
        CSS color for the edge.

    Examples
    --------
    ```python
    import talk_box as tb

    ve = tb.VisEdge(source="doc-1", target="e1", label="mentions", weight=0.9)
    ve.label  # "mentions"
    ```
    """

    source: str
    target: str
    label: str
    weight: float = 1.0
    color: str = "#888888"


@dataclass(frozen=True)
class VisGraph:
    """A complete graph prepared for visualization.

    Parameters
    ----------
    nodes
        Visualization nodes.
    edges
        Visualization edges.
    title
        Graph title.

    Examples
    --------
    ```python
    import talk_box as tb

    graph = tb.prepare_vis_graph(kg)
    graph.node_count  # 10
    graph.edge_count  # 15
    ```
    """

    nodes: list[VisNode] = field(default_factory=list)
    edges: list[VisEdge] = field(default_factory=list)
    title: str = "Knowledge Graph"

    @property
    def node_count(self) -> int:
        """Number of nodes in the visualization.

        Examples
        --------
        ```python
        graph.node_count  # 10
        ```
        """
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        """Number of edges in the visualization.

        Examples
        --------
        ```python
        graph.edge_count  # 15
        ```
        """
        return len(self.edges)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary.

        Examples
        --------
        ```python
        d = graph.to_dict()
        len(d["nodes"])  # 10
        ```
        """
        return {
            "title": self.title,
            "nodes": [
                {
                    "id": n.id,
                    "label": n.label,
                    "node_type": n.node_type,
                    "color": n.color,
                    "size": n.size,
                    "metadata": n.metadata,
                }
                for n in self.nodes
            ],
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "label": e.label,
                    "weight": e.weight,
                    "color": e.color,
                }
                for e in self.edges
            ],
        }


# ---------------------------------------------------------------------------
# Graph preparation
# ---------------------------------------------------------------------------


def _freshness_color(node: Node) -> str:
    """Determine node color based on age of last update."""
    if node.updated_at <= 0:
        return _FRESHNESS_COLORS["unknown"]
    age_days = (time.time() - node.updated_at) / 86_400
    if age_days <= 7:
        return _FRESHNESS_COLORS["fresh"]
    if age_days <= 30:
        return _FRESHNESS_COLORS["aging"]
    return _FRESHNESS_COLORS["stale"]


def _node_size(kg: KnowledgeGraph, node: Node, base: int = 15) -> int:
    """Compute relative node size based on edge count."""
    edge_count = len(kg.get_edges(node.id))
    return base + min(edge_count * 3, 30)


def prepare_vis_graph(
    kg: KnowledgeGraph,
    *,
    max_nodes: int = 200,
    node_type: NodeType | None = None,
    color_by: str = "type",
    title: str = "Knowledge Graph",
) -> VisGraph:
    """Prepare a knowledge graph for visualization.

    Converts nodes and edges into visualization-ready dataclasses with
    computed colors, sizes, and labels.

    Parameters
    ----------
    kg
        The knowledge graph to visualize.
    max_nodes
        Maximum number of nodes to include.
    node_type
        If provided, only include nodes of this type.
    color_by
        Color scheme: ``"type"`` (by node type) or ``"freshness"``
        (by update recency).
    title
        Title for the visualization.

    Returns
    -------
    VisGraph
        A visualization-ready graph.

    Examples
    --------
    ```python
    import talk_box as tb

    kg = tb.KnowledgeGraph(":memory:")
    # ... add nodes and edges ...
    graph = tb.prepare_vis_graph(kg, color_by="freshness")
    graph.node_count
    ```
    """
    nodes = kg.list_nodes(node_type=node_type, limit=max_nodes)
    node_ids = {n.id for n in nodes}

    vis_nodes: list[VisNode] = []
    for node in nodes:
        if color_by == "freshness":
            color = _freshness_color(node)
        else:
            color = _NODE_COLORS.get(node.node_type, "#4A90D9")

        vis_nodes.append(
            VisNode(
                id=node.id,
                label=node.name,
                node_type=node.node_type.value,
                color=color,
                size=_node_size(kg, node),
                metadata={
                    k: v for k, v in node.metadata.items() if isinstance(v, (str, int, float, bool))
                },
            )
        )

    vis_edges: list[VisEdge] = []
    seen: set[tuple[str, str, str]] = set()
    for node in nodes:
        for edge in kg.get_edges(node.id, direction="outgoing"):
            if edge.target not in node_ids:
                continue
            key = (edge.source, edge.target, edge.relation)
            if key in seen:
                continue
            seen.add(key)
            vis_edges.append(
                VisEdge(
                    source=edge.source,
                    target=edge.target,
                    label=edge.relation,
                    weight=edge.weight,
                )
            )

    return VisGraph(nodes=vis_nodes, edges=vis_edges, title=title)


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #1a1a2e; color: #eee; overflow: hidden; }}
  #graph {{ width: 100vw; height: 100vh; }}
  #info {{ position: fixed; top: 12px; right: 12px; background: rgba(30,30,60,0.92);
           padding: 12px 16px; border-radius: 8px; font-size: 13px; max-width: 280px;
           display: none; border: 1px solid rgba(255,255,255,0.1); }}
  #info h3 {{ margin-bottom: 6px; font-size: 15px; }}
  #info .meta {{ color: #aaa; font-size: 12px; margin-top: 4px; }}
  #legend {{ position: fixed; bottom: 12px; left: 12px; background: rgba(30,30,60,0.92);
             padding: 10px 14px; border-radius: 8px; font-size: 12px;
             border: 1px solid rgba(255,255,255,0.1); }}
  #legend div {{ display: flex; align-items: center; gap: 6px; margin: 3px 0; }}
  #legend .dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
  #title {{ position: fixed; top: 12px; left: 12px; font-size: 16px; font-weight: 600;
            background: rgba(30,30,60,0.92); padding: 8px 14px; border-radius: 8px;
            border: 1px solid rgba(255,255,255,0.1); }}
  svg text {{ font-family: inherit; }}
</style>
</head>
<body>
<div id="title">{title}</div>
<div id="graph"></div>
<div id="info"><h3 id="info-name"></h3><div id="info-type" class="meta"></div><div id="info-meta" class="meta"></div></div>
<div id="legend">{legend_html}</div>
<script>
// --- Data ---
const nodes = {nodes_json};
const links = {edges_json};

// --- Layout ---
const width = window.innerWidth;
const height = window.innerHeight;

const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
svg.setAttribute("width", width);
svg.setAttribute("height", height);
document.getElementById("graph").appendChild(svg);

// Simple force-directed layout (no D3 dependency)
const positions = {{}};
nodes.forEach((n, i) => {{
  const angle = (2 * Math.PI * i) / nodes.length;
  const r = Math.min(width, height) * 0.35;
  positions[n.id] = {{
    x: width / 2 + r * Math.cos(angle),
    y: height / 2 + r * Math.sin(angle),
    vx: 0, vy: 0
  }};
}});

// Build adjacency for forces
const adj = {{}};
nodes.forEach(n => adj[n.id] = []);
links.forEach(l => {{
  adj[l.source].push(l.target);
  adj[l.target].push(l.source);
}});

function simulate(iterations) {{
  for (let iter = 0; iter < iterations; iter++) {{
    // Repulsion
    for (let i = 0; i < nodes.length; i++) {{
      for (let j = i + 1; j < nodes.length; j++) {{
        const a = positions[nodes[i].id], b = positions[nodes[j].id];
        let dx = b.x - a.x, dy = b.y - a.y;
        let dist = Math.sqrt(dx * dx + dy * dy) || 1;
        let force = 8000 / (dist * dist);
        let fx = (dx / dist) * force, fy = (dy / dist) * force;
        a.vx -= fx; a.vy -= fy;
        b.vx += fx; b.vy += fy;
      }}
    }}
    // Attraction (edges)
    links.forEach(l => {{
      const a = positions[l.source], b = positions[l.target];
      let dx = b.x - a.x, dy = b.y - a.y;
      let dist = Math.sqrt(dx * dx + dy * dy) || 1;
      let force = (dist - 120) * 0.01;
      let fx = (dx / dist) * force, fy = (dy / dist) * force;
      a.vx += fx; a.vy += fy;
      b.vx -= fx; b.vy -= fy;
    }});
    // Centering
    nodes.forEach(n => {{
      const p = positions[n.id];
      p.vx += (width / 2 - p.x) * 0.001;
      p.vy += (height / 2 - p.y) * 0.001;
    }});
    // Apply velocities
    nodes.forEach(n => {{
      const p = positions[n.id];
      p.vx *= 0.85; p.vy *= 0.85;
      p.x += p.vx; p.y += p.vy;
      p.x = Math.max(40, Math.min(width - 40, p.x));
      p.y = Math.max(40, Math.min(height - 40, p.y));
    }});
  }}
}}
simulate(200);

// --- Render edges ---
links.forEach(l => {{
  const a = positions[l.source], b = positions[l.target];
  const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
  line.setAttribute("x1", a.x); line.setAttribute("y1", a.y);
  line.setAttribute("x2", b.x); line.setAttribute("y2", b.y);
  line.setAttribute("stroke", l.color || "#555");
  line.setAttribute("stroke-width", Math.max(1, l.weight * 3));
  line.setAttribute("stroke-opacity", "0.5");
  svg.appendChild(line);
  // Edge label
  const midX = (a.x + b.x) / 2, midY = (a.y + b.y) / 2;
  const txt = document.createElementNS("http://www.w3.org/2000/svg", "text");
  txt.setAttribute("x", midX); txt.setAttribute("y", midY - 4);
  txt.setAttribute("text-anchor", "middle"); txt.setAttribute("fill", "#777");
  txt.setAttribute("font-size", "10");
  txt.textContent = l.label;
  svg.appendChild(txt);
}});

// --- Render nodes ---
const info = document.getElementById("info");
const infoName = document.getElementById("info-name");
const infoType = document.getElementById("info-type");
const infoMeta = document.getElementById("info-meta");

nodes.forEach(n => {{
  const p = positions[n.id];
  const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
  g.style.cursor = "pointer";

  const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
  circle.setAttribute("cx", p.x); circle.setAttribute("cy", p.y);
  circle.setAttribute("r", n.size || 15);
  circle.setAttribute("fill", n.color);
  circle.setAttribute("stroke", "#fff"); circle.setAttribute("stroke-width", "1.5");
  circle.setAttribute("fill-opacity", "0.85");
  g.appendChild(circle);

  const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
  label.setAttribute("x", p.x); label.setAttribute("y", p.y + (n.size || 15) + 14);
  label.setAttribute("text-anchor", "middle"); label.setAttribute("fill", "#ddd");
  label.setAttribute("font-size", "11");
  label.textContent = n.label.length > 20 ? n.label.slice(0, 18) + "…" : n.label;
  g.appendChild(label);

  g.addEventListener("click", () => {{
    infoName.textContent = n.label;
    infoType.textContent = "Type: " + n.node_type;
    const meta = n.metadata || {{}};
    infoMeta.textContent = Object.keys(meta).length
      ? Object.entries(meta).map(([k,v]) => k + ": " + v).join("\\n")
      : "";
    info.style.display = "block";
  }});
  svg.appendChild(g);
}});

svg.addEventListener("click", e => {{
  if (e.target === svg) info.style.display = "none";
}});
</script>
</body>
</html>"""


def _build_legend_html(color_by: str) -> str:
    """Build legend HTML based on color scheme."""
    if color_by == "freshness":
        items = [
            ("fresh", "Fresh (≤7 days)", _FRESHNESS_COLORS["fresh"]),
            ("aging", "Aging (8–30 days)", _FRESHNESS_COLORS["aging"]),
            ("stale", "Stale (>30 days)", _FRESHNESS_COLORS["stale"]),
            ("unknown", "Unknown", _FRESHNESS_COLORS["unknown"]),
        ]
    else:
        items = [
            ("document", "Document", _NODE_COLORS[NodeType.DOCUMENT]),
            ("entity", "Entity", _NODE_COLORS[NodeType.ENTITY]),
            ("topic", "Topic", _NODE_COLORS[NodeType.TOPIC]),
        ]
    parts = []
    for _, label, color in items:
        parts.append(f'<div><span class="dot" style="background:{color}"></span>{label}</div>')
    return "\n".join(parts)


def to_html(
    kg: KnowledgeGraph,
    *,
    max_nodes: int = 200,
    node_type: NodeType | None = None,
    color_by: str = "type",
    title: str = "Knowledge Graph",
) -> str:
    """Render a knowledge graph as an interactive HTML page.

    Generates a self-contained HTML file with an SVG-based force-directed
    graph layout.  No external JavaScript dependencies are required.

    Parameters
    ----------
    kg
        The knowledge graph to render.
    max_nodes
        Maximum nodes to include.
    node_type
        If provided, only include nodes of this type.
    color_by
        Color scheme: ``"type"`` or ``"freshness"``.
    title
        Page title.

    Returns
    -------
    str
        Complete HTML document as a string.

    Examples
    --------
    ```python
    import talk_box as tb

    kg = tb.KnowledgeGraph(":memory:")
    # ... add nodes ...
    html_str = tb.to_html(kg, color_by="freshness")
    ```
    """
    import json

    graph = prepare_vis_graph(
        kg,
        max_nodes=max_nodes,
        node_type=node_type,
        color_by=color_by,
        title=title,
    )
    data = graph.to_dict()
    legend = _build_legend_html(color_by)
    safe_title = html.escape(title)

    return _HTML_TEMPLATE.format(
        title=safe_title,
        legend_html=legend,
        nodes_json=json.dumps(data["nodes"]),
        edges_json=json.dumps(data["edges"]),
    )


def visualize(
    kg: KnowledgeGraph,
    *,
    output: str | Path | None = None,
    max_nodes: int = 200,
    node_type: NodeType | None = None,
    color_by: str = "type",
    title: str = "Knowledge Graph",
    open_browser: bool = True,
) -> Path:
    """Render a knowledge graph and open it in the browser.

    Writes an interactive HTML file and optionally opens it in the
    default browser.

    Parameters
    ----------
    kg
        The knowledge graph to visualize.
    output
        Path for the HTML file.  Defaults to a temporary file.
    max_nodes
        Maximum nodes to include.
    node_type
        If provided, only include nodes of this type.
    color_by
        Color scheme: ``"type"`` or ``"freshness"``.
    title
        Page title.
    open_browser
        If ``True`` (default), open the file in the default browser.

    Returns
    -------
    Path
        Path to the generated HTML file.

    Examples
    --------
    ```python
    import talk_box as tb

    kg = tb.KnowledgeGraph(":memory:")
    # ... add nodes ...
    path = tb.visualize(kg, open_browser=False)
    path  # PosixPath('/tmp/talk_box_kg_xxxxx.html')
    ```
    """
    if output is None:
        import tempfile

        fd, tmp = tempfile.mkstemp(prefix="talk_box_kg_", suffix=".html")
        import os

        os.close(fd)
        out_path = Path(tmp)
    else:
        out_path = Path(output)

    content = to_html(
        kg,
        max_nodes=max_nodes,
        node_type=node_type,
        color_by=color_by,
        title=title,
    )
    out_path.write_text(content, encoding="utf-8")

    if open_browser:
        webbrowser.open(f"file://{out_path.resolve()}")

    return out_path
