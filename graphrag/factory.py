"""
Factory for creating graph backend instances based on configuration.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from agent.config import settings
from .base_graph import BaseNormativaGraph

logger = logging.getLogger(__name__)


def create_graph() -> BaseNormativaGraph:
    """Create a graph backend based on config.yaml settings.

    For backend="networkx": loads JSON into a NetworkXNormativaGraph.
    For backend="neo4j": connects to Neo4j via Neo4jNormativaGraph.
    """
    cfg = settings.graph

    if cfg.backend == "neo4j":
        return _create_neo4j_graph(cfg)

    return _create_networkx_graph(cfg)


def _create_networkx_graph(cfg) -> BaseNormativaGraph:
    from .graph_builder import NetworkXNormativaGraph
    from .graph_models import NodeType, EdgeType

    g = NetworkXNormativaGraph()
    json_path = Path(cfg.json_path)

    if not json_path.exists():
        logger.warning("Graph JSON not found at %s", json_path)
        return g

    with open(json_path) as f:
        data = json.load(f)

    for node in data.get("nodes", []):
        props = {"type": NodeType(node["type"]), "label": node["label"], **node.get("properties", {})}
        g.graph.add_node(node["id"], **props)

    for edge in data.get("edges", []):
        props = {"type": EdgeType(edge["type"]), **edge.get("properties", {})}
        g.graph.add_edge(edge["source"], edge["target"], key=edge["type"], **props)

    logger.info("Graph loaded: %d nodes, %d edges", g.graph.number_of_nodes(), g.graph.number_of_edges())
    return g


def _create_neo4j_graph(cfg) -> BaseNormativaGraph:
    from .neo4j_graph import Neo4jNormativaGraph

    g = Neo4jNormativaGraph(
        uri=cfg.neo4j_uri,
        user=cfg.neo4j_user,
        password=cfg.neo4j_password,
        database=cfg.neo4j_database,
    )
    summary = g.summary()
    logger.info("Neo4j graph: %d nodes, %d edges", summary["nodes"], summary["edges"])
    return g
