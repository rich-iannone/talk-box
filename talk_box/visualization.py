import json
import tempfile
import webbrowser
from pathlib import Path
from typing import Any, Dict


def generate_pathway_html(
    pathway_data: Dict[str, Any], title: str = "Pathway Visualization"
) -> str:
    """
    Generate HTML visualization for a Pathway using vis.js network library.

    Parameters
    ----------
    pathway_data
        The pathway data structure from `Pathways._build()`.
    title
        Title for the visualization page.

    Returns
    -------
    str
        Complete HTML document with embedded vis.js visualization
    """

    # Extract states and transitions
    states = pathway_data.get("states", {})
    transitions = pathway_data.get("transitions", [])
    start_state = pathway_data.get("start_state")

    # Convert states to vis.js nodes
    nodes = []
    for state_id, state_info in states.items():
        state_type = state_info.get("type", "chat")

        # Color coding based on state type
        color_map = {
            "chat": "#4CAF50",  # Green for general chat
            "collect": "#2196F3",  # Blue for information collection
            "decision": "#FF9800",  # Orange for decision points
            "tool": "#9C27B0",  # Purple for tool usage
            "summary": "#607D8B",  # Blue-grey for summary
        }

        # Shape based on state type
        shape_map = {
            "chat": "ellipse",
            "collect": "box",
            "decision": "diamond",
            "tool": "hexagon",
            "summary": "star",
        }

        # Build tooltip content with state details
        tooltip_parts = [f"<b>{state_info.get('description', state_id)}</b>"]

        if state_info.get("required_info"):
            tooltip_parts.append(f"<br><b>Required:</b> {', '.join(state_info['required_info'])}")

        if state_info.get("optional_info"):
            tooltip_parts.append(f"<br><b>Optional:</b> {', '.join(state_info['optional_info'])}")

        if state_info.get("tools"):
            tooltip_parts.append(f"<br><b>Tools:</b> {', '.join(state_info['tools'])}")

        if state_info.get("success_conditions"):
            tooltip_parts.append(
                f"<br><b>Success:</b> {', '.join(state_info['success_conditions'])}"
            )

        node = {
            "id": state_id,
            "label": state_id.replace("_", " ").title(),
            "color": {
                "background": color_map.get(state_type, "#4CAF50"),
                "border": "#333333",
                "highlight": {"background": "#FFE0B2", "border": "#FF6F00"},
            },
            "shape": shape_map.get(state_type, "ellipse"),
            "font": {"color": "white", "size": 14},
            "title": "".join(tooltip_parts),
            "borderWidth": 3 if state_id == start_state else 2,
        }

        # Make start state stand out
        if state_id == start_state:
            node["color"]["border"] = "#4CAF50"
            node["borderWidth"] = 5

        nodes.append(node)

    # Convert transitions to vis.js edges
    edges = []
    for i, transition in enumerate(transitions):
        from_state = transition.get("from")
        to_state = transition.get("to")
        condition = transition.get("condition")

        edge = {
            "id": f"edge_{i}",
            "from": from_state,
            "to": to_state,
            "arrows": "to",
            "color": {"color": "#666666", "highlight": "#FF6F00"},
            "width": 2,
        }

        # Add label for conditional transitions
        if condition:
            edge["label"] = condition[:30] + "..." if len(condition) > 30 else condition
            edge["font"] = {"size": 12, "color": "#333333"}
            edge["color"]["color"] = "#FF6F00"  # Orange for conditional edges
            edge["width"] = 3

        edges.append(edge)

    # Create the HTML template
    html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style type="text/css">
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}

        .header {{
            text-align: center;
            margin-bottom: 20px;
        }}

        .header h1 {{
            color: #333;
            margin-bottom: 10px;
        }}

        .header .subtitle {{
            color: #666;
            font-size: 16px;
            margin-bottom: 20px;
        }}

        .info-panel {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}

        .info-panel h3 {{
            margin-top: 0;
            color: #333;
        }}

        .legend {{
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            justify-content: center;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .legend-icon {{
            width: 20px;
            height: 20px;
            border-radius: 4px;
            border: 2px solid #333;
        }}

        #network-container {{
            width: 100%;
            height: 600px;
            border: 2px solid #ddd;
            border-radius: 8px;
            background-color: white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}

        .controls {{
            text-align: center;
            margin-top: 15px;
        }}

        .controls button {{
            background: #2196F3;
            color: white;
            border: none;
            padding: 10px 20px;
            margin: 0 5px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
        }}

        .controls button:hover {{
            background: #1976D2;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
        <div class="subtitle">{pathway_data.get("description", "Interactive pathway flow diagram")}</div>
    </div>

    <div class="info-panel">
        <h3>State Types Legend</h3>
        <div class="legend">
            <div class="legend-item">
                <div class="legend-icon" style="background-color: #4CAF50; border-radius: 50%;"></div>
                <span>Chat - General conversation</span>
            </div>
            <div class="legend-item">
                <div class="legend-icon" style="background-color: #2196F3;"></div>
                <span>Collect - Information gathering</span>
            </div>
            <div class="legend-item">
                <div class="legend-icon" style="background-color: #FF9800; transform: rotate(45deg);"></div>
                <span>Decision - Branching logic</span>
            </div>
            <div class="legend-item">
                <div class="legend-icon" style="background-color: #9C27B0; clip-path: polygon(30% 0%, 70% 0%, 100% 30%, 100% 70%, 70% 100%, 30% 100%, 0% 70%, 0% 30%);"></div>
                <span>Tool - Tool usage</span>
            </div>
            <div class="legend-item">
                <div class="legend-icon" style="background-color: #607D8B; clip-path: polygon(50% 0%, 61% 35%, 98% 35%, 68% 57%, 79% 91%, 50% 70%, 21% 91%, 32% 57%, 2% 35%, 39% 35%);"></div>
                <span>Summary - Completion</span>
            </div>
        </div>
    </div>

    <div id="network-container"></div>

    <div class="controls">
        <button onclick="fitNetwork()">Fit to Screen</button>
        <button onclick="resetZoom()">Reset Zoom</button>
        <button onclick="togglePhysics()">Toggle Physics</button>
    </div>

    <script type="text/javascript">
        // Network data
        const nodes = new vis.DataSet({json_encode(nodes)});
        const edges = new vis.DataSet({json_encode(edges)});
        const data = {{ nodes: nodes, edges: edges }};

        // Network options
        const options = {{
            physics: {{
                enabled: true,
                stabilization: {{
                    enabled: true,
                    iterations: 100,
                    updateInterval: 50
                }},
                barnesHut: {{
                    gravitationalConstant: -2000,
                    centralGravity: 0.1,
                    springLength: 200,
                    springConstant: 0.04,
                    damping: 0.09,
                    avoidOverlap: 0.5
                }}
            }},
            layout: {{
                improvedLayout: true,
                hierarchical: {{
                    enabled: false
                }}
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 200,
                hideEdgesOnDrag: false,
                hideNodesOnDrag: false
            }},
            nodes: {{
                font: {{
                    size: 14,
                    color: 'white'
                }},
                margin: 10,
                widthConstraint: {{
                    minimum: 80,
                    maximum: 150
                }}
            }},
            edges: {{
                smooth: {{
                    type: 'continuous',
                    roundness: 0.5
                }},
                arrows: {{
                    to: {{
                        enabled: true,
                        scaleFactor: 1.2
                    }}
                }}
            }}
        }};

        // Initialize network
        const container = document.getElementById('network-container');
        const network = new vis.Network(container, data, options);

        // Control functions
        let physicsEnabled = true;

        function fitNetwork() {{
            network.fit({{
                animation: {{
                    duration: 1000,
                    easingFunction: 'easeInOutQuad'
                }}
            }});
        }}

        function resetZoom() {{
            network.moveTo({{
                scale: 1.0,
                animation: {{
                    duration: 1000,
                    easingFunction: 'easeInOutQuad'
                }}
            }});
        }}

        function togglePhysics() {{
            physicsEnabled = !physicsEnabled;
            network.setOptions({{physics: {{enabled: physicsEnabled}}}});

            const button = event.target;
            button.textContent = physicsEnabled ? 'Toggle Physics' : 'Enable Physics';
        }}

        // Event handlers
        network.on('click', function(params) {{
            if (params.nodes.length > 0) {{
                const nodeId = params.nodes[0];
                console.log('Clicked node:', nodeId);
            }}
        }});

        network.on('hoverNode', function(params) {{
            container.style.cursor = 'pointer';
        }});

        network.on('blurNode', function(params) {{
            container.style.cursor = 'default';
        }});

        // Initial fit
        network.once('stabilizationIterationsDone', function() {{
            fitNetwork();
        }});

    </script>
</body>
</html>
"""
    return html_template


def json_encode(data):
    """Helper function to encode data as JSON for JavaScript"""
    return json.dumps(data, indent=2)


def visualize_pathway(
    pathway_data: Dict[str, Any], title: str = None, filename: str = None, auto_open: bool = True
) -> str:
    """
    Create and optionally display an HTML visualization of a pathway.

    Parameters
    ----------
    pathway_data
        The pathway data structure from `Pathways._build()`.
    title
        Title for the visualization. If None, uses pathway title.
    filename
        Filename to save HTML to. If None, uses temporary file.
    auto_open
        Whether to automatically open the visualization in a browser.

    Returns
    -------
    str
        Path to the generated HTML file
    """
    if title is None:
        title = f"Pathway Visualization: {pathway_data.get('title', 'Untitled')}"

    html_content = generate_pathway_html(pathway_data, title)

    if filename is None:
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False)
        filename = temp_file.name
        temp_file.write(html_content)
        temp_file.close()
    else:
        # Save to specified filename
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html_content)

    if auto_open:
        webbrowser.open(f"file://{Path(filename).absolute()}")

    return filename
