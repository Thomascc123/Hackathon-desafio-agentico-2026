"""
Interactive knowledge graph explorer for Streamlit.
Uses Pyvis (vis.js) for zoom, pan, drag, and click-to-inspect.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from graphrag.base_graph import BaseNormativaGraph

# ── Node/edge style config ──────────────────────────────────────────

NODE_COLORS = {
    "Documento": "#1f77b4",
    "Articulo": "#2ca02c",
    "Capitulo": "#ff7f0e",
    "Titulo": "#9467bd",
    "Concepto": "#d62728",
    "Autoridad": "#8c564b",
}

EDGE_COLORS = {
    "CONTIENE": "#aaaaaa",
    "MODIFICA": "#d62728",
    "CITA": "#1f77b4",
    "REGLAMENTA": "#2ca02c",
    "ACTUALIZA": "#ff7f0e",
    "EMITE": "#8c564b",
}

EDGE_DASHES = {
    "CONTIENE": False,
    "MODIFICA": True,
    "CITA": False,
    "REGLAMENTA": False,
    "ACTUALIZA": True,
    "EMITE": False,
}


def _node_title(data: dict) -> str:
    """Build a tooltip / title string for a node."""
    parts = [f"<b>{data.get('label', '')}</b>", f"Tipo: {data.get('type', '').value if hasattr(data.get('type', ''), 'value') else data.get('type', '')}"]
    for k in ("numero", "fecha", "asunto", "texto", "resuelve", "descripcion"):
        v = data.get(k)
        if v:
            vstr = str(v)[:120]
            parts.append(f"{k}: {vstr}")
    return "<br>".join(parts)


def _edge_title(data: dict) -> str:
    """Build a tooltip for an edge."""
    etype = data.get("type", "")
    etype_str = etype.value if hasattr(etype, "value") else str(etype)
    parts = [f"<b>{etype_str}</b>"]
    for k in ("fecha", "descripcion", "referencia"):
        v = data.get(k)
        if v:
            parts.append(f"{k}: {v}")
    return "<br>".join(parts)


def render_graph_html(
    graph: BaseNormativaGraph,
    selected_types: list[str] | None = None,
    search: str = "",
    max_nodes: int = 500,
    physics: bool = True,
    highlight_ids: list[str] | None = None,
) -> str:
    """
    Build a Pyvis HTML visualization of the knowledge graph.
    
    Parameters
    ----------
    graph : BaseNormativaGraph
        The knowledge graph to visualize.
    selected_types : list[str] or None
        If provided, only include nodes of these types (e.g. ["Articulo", "Documento"]).
    search : str
        If provided, only include nodes whose label contains this string (case-insensitive).
    max_nodes : int
        Maximum number of nodes to display (to avoid browser slowdown).
    physics : bool
        Enable physics-based layout.

    Returns
    -------
    str
        Full HTML page as a string, suitable for st.components.v1.html().
    """
    from pyvis.network import Network

    net = Network(
        height="700px",
        width="100%",
        directed=True,
        notebook=False,
        bgcolor="#ffffff",
        font_color="#333333",
    )

    # ── Configure options dict directly ──────────────────────────
    opts: dict = {
        "edges": {
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.8}},
            "smooth": {"type": "continuous"},
        },
        "nodes": {
            "font": {"size": 12, "face": "Arial"},
        },
        "interaction": {
            "hover": True,
            "tooltipDelay": 200,
            "navigationButtons": True,
            "keyboard": True,
        },
        "configure": {"enabled": False},
    }

    if physics:
        opts["physics"] = {
            "enabled": True,
            "stabilization": {"iterations": 100},
            "barnesHut": {
                "gravitationalConstant": -3000,
                "springLength": 150,
                "springConstant": 0.04,
            },
            "solver": "barnesHut",
        }
    else:
        opts["physics"] = {"enabled": False}

    net.set_options(json.dumps(opts))

    # ── Collect nodes ───────────────────────────────────────────────
    search_lower = search.lower().strip() if search else ""

    # Gather all nodes of interest
    node_ids = []
    node_map = {}  # id -> data

    for nid, data in graph.graph.nodes(data=True):
        ntype = data.get("type", "")
        ntype_str = ntype.value if hasattr(ntype, "value") else str(ntype)

        # Filter by type
        if selected_types and ntype_str not in selected_types:
            continue

        # Filter by search
        if search_lower and search_lower not in data.get("label", "").lower():
            continue

        node_ids.append(nid)
        node_map[nid] = dict(data)

        if len(node_ids) >= max_nodes:
            break

    # ── Collect edges between selected nodes ──────────────────────
    node_set = set(node_ids)
    edge_list = []
    for u, v, k, data in graph.graph.edges(data=True, keys=True):
        if u in node_set and v in node_set:
            edge_list.append((u, v, k, dict(data)))

    # ── Add nodes to Pyvis ───────────────────────────────────────────
    highlight_set = set(highlight_ids or [])
    for nid in node_ids:
        data = node_map[nid]
        ntype = data.get("type", "")
        ntype_str = ntype.value if hasattr(ntype, "value") else str(ntype)
        label = data.get("label", nid)
        title = _node_title(data)
        is_highlighted = nid in highlight_set
        color = NODE_COLORS.get(ntype_str, "#97c2fc")
        size = 20 if ntype_str == "Documento" else (15 if ntype_str == "Articulo" else 12)
        shape = "box" if ntype_str == "Documento" else "dot"
        if is_highlighted:
            size = int(size * 1.5)
            color = "#ff6600"
            shape = "star"
        net.add_node(
            nid,
            label=label[:40],
            title=title,
            color=color,
            size=size,
            shape=shape,
            borderWidth=4 if is_highlighted else 2,
        )

    # ── Add edges to Pyvis ───────────────────────────────────────────
    for u, v, k, data in edge_list:
        etype = data.get("type", "")
        etype_str = etype.value if hasattr(etype, "value") else str(etype)
        title = _edge_title(data)
        color = EDGE_COLORS.get(etype_str, "#999999")
        dashes = EDGE_DASHES.get(etype_str, False)
        width = 2 if etype_str == "MODIFICA" else 1
        net.add_edge(
            u, v,
            title=title,
            color=color,
            dashes=dashes,
            width=width,
            arrowStrikethrough=True,
        )

    # ── Inject custom click handler for properties ─────────────────
    html = net.generate_html()
    # Expose network instance globally so our custom script can attach handlers
    html = html.replace(
        "network = new vis.Network(container, data, options);",
        "network = new vis.Network(container, data, options); window.__network = network;",
    )
    custom_js = """
    <script type="text/javascript">
    (function() {
        var container = document.getElementById('mynetwork');
        if (!container) return;
        var check = setInterval(function() {
            var net = window.__network;
            if (net) {
                clearInterval(check);
                net.on("click", function(params) {
                    var infoDiv = document.getElementById('node-info');
                    if (!infoDiv) {
                        infoDiv = document.createElement('div');
                        infoDiv.id = 'node-info';
                        infoDiv.style.cssText = 'position:fixed; bottom:10px; left:10px; right:10px; max-height:200px; overflow-y:auto; background:#fff; border:1px solid #ccc; border-radius:8px; padding:12px; z-index:9999; font-family:sans-serif; font-size:13px; box-shadow:0 2px 8px rgba(0,0,0,0.15);';
                        document.body.appendChild(infoDiv);
                    }
                    if (params.nodes.length > 0) {
                        var nid = params.nodes[0];
                        var node = net.body.nodes[nid];
                        if (node && node.options.title) {
                            infoDiv.innerHTML = '<b>Nodo seleccionado</b><br>' + node.options.title;
                        } else {
                            infoDiv.innerHTML = '<b>Nodo:</b> ' + nid;
                        }
                    } else if (params.edges.length > 0) {
                        var eid = params.edges[0];
                        var edge = net.body.edges[eid];
                        if (edge && edge.options.title) {
                            infoDiv.innerHTML = '<b>Arista seleccionada</b><br>' + edge.options.title;
                        } else {
                            infoDiv.innerHTML = '<b>Arista:</b> ' + eid;
                        }
                    } else {
                        infoDiv.innerHTML = '';
                    }
                    var closeBtn = document.createElement('button');
                    closeBtn.innerHTML = '✕';
                    closeBtn.style.cssText = 'float:right; border:none; background:none; cursor:pointer; font-size:16px;';
                    closeBtn.onclick = function() { infoDiv.innerHTML = ''; };
                    infoDiv.prepend(closeBtn);
                });
            }
        }, 500);
    })();
    </script>
    """
    html = html.replace("</body>", custom_js + "</body>")
    return html


def render_graph_page(graph: BaseNormativaGraph, ref_to_node_id: dict | None = None) -> None:
    """Render the graph explorer page in Streamlit.

    Parameters
    ----------
    graph : NormativaGraph
        The knowledge graph to visualize.
    ref_to_node_id : dict or None
        Mapping from source reference strings to graph node IDs for highlighting.
    """

    from graphrag.graph_models import NodeType

    type_options = [t.value for t in NodeType]
    type_labels = {
        "Documento": "📄 Documentos",
        "Articulo": "📋 Artículos",
        "Capitulo": "📂 Capítulos",
        "Titulo": "📑 Títulos",
        "Concepto": "🏷️ Conceptos",
        "Autoridad": "👤 Autoridades",
    }

    # ── Controls ─────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        search = st.text_input("🔍 Buscar nodo por nombre", placeholder="Ej: matrícula, artículo 63...")

    with col2:
        selected = st.multiselect(
            "Tipo de entidad",
            options=type_options,
            default=type_options[:4],
            format_func=lambda t: type_labels.get(t, t),
        )

    with col3:
        max_n = st.selectbox("Máx. nodos", [200, 500, 1000, 2000, 5000, 10000], index=1)

    physics = st.toggle("Física (layout automático)", value=True, key="graph_physics")

    # ── Resolve highlighted refs ─────────────────────────────────────
    highlight_ids: list[str] = []
    if ref_to_node_id:
        highlighted_refs = st.session_state.get("highlighted_refs", [])
        for ref in highlighted_refs:
            nid = ref_to_node_id.get(ref)
            if nid:
                highlight_ids.append(nid)

    # ── Render ───────────────────────────────────────────────────────
    with st.spinner("Generando visualización del grafo..."):
        html = render_graph_html(
            graph,
            selected_types=selected,
            search=search,
            max_nodes=max_n,
            physics=physics,
            highlight_ids=highlight_ids,
        )

    st.components.v1.html(html, height=750, scrolling=False)

    # ── Stats ────────────────────────────────────────────────────────
    node_count = graph.graph.number_of_nodes()
    edge_count = graph.graph.number_of_edges()
    st.caption(f"Grafo: {node_count:,} nodos · {edge_count:,} aristas · Tipos: {', '.join(type_labels.get(t, t) for t in type_options if t in selected)}")
