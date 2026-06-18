"""
Abstract interface for graph backends.
Supports both local NetworkX and production Neo4j.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any


class BaseNormativaGraph(ABC):
    """Abstract interface for the Normativa UdeA knowledge graph."""

    @abstractmethod
    def query_articles_by_keyword(self, keyword: str) -> list[dict]:
        """Find articles whose text contains the given keyword(s)."""
        ...

    @abstractmethod
    def query_evolution_of_article(self, art_numero: str) -> list[dict]:
        """Trace the modification history of a specific article."""
        ...

    @abstractmethod
    def query_document_timeline(self, asunto: str = "") -> list[dict]:
        """Get all documents sorted by date for a given subject."""
        ...

    @abstractmethod
    def query_articles_modified_by(self, doc_numero: str) -> list[dict]:
        """Find all articles modified by a given document number."""
        ...

    @abstractmethod
    def summary(self) -> dict:
        """Return node/edge counts and type breakdowns."""
        ...

    @abstractmethod
    def export_to_json(self) -> dict:
        """Export the full graph as a JSON-serializable dict.

        Returns:
            {"nodes": [{"id": str, "type": str, "label": str, "properties": dict}, ...],
             "edges": [{"source": str, "target": str, "type": str, "properties": dict}, ...]}
        """
        ...

    def iter_nodes(self) -> Iterator[tuple[str, dict]]:
        """Iterate over all nodes as (node_id, data_dict) pairs.

        Default implementation uses export_to_json(). Subclasses may override
        with a more efficient implementation.
        """
        data = self.export_to_json()
        for node in data.get("nodes", []):
            props = dict(node.get("properties", {}))
            props["type"] = self._parse_node_type(node["type"])
            props["label"] = node["label"]
            yield node["id"], props

    def iter_edges(self) -> Iterator[tuple[str, str, str | None, dict]]:
        """Iterate over all edges as (source, target, key, data) tuples.

        Default implementation uses export_to_json(). Subclasses may override.
        """
        data = self.export_to_json()
        for edge in data.get("edges", []):
            props = dict(edge.get("properties", {}))
            props["type"] = self._parse_edge_type(edge["type"])
            yield edge["source"], edge["target"], edge["type"], props

    def _parse_node_type(self, raw: str) -> Any:
        """Parse a node type string into the appropriate enum value."""
        from .graph_models import NodeType
        try:
            return NodeType(raw)
        except ValueError:
            return raw

    def _parse_edge_type(self, raw: str) -> Any:
        """Parse an edge type string into the appropriate enum value."""
        from .graph_models import EdgeType
        try:
            return EdgeType(raw)
        except ValueError:
            return raw
