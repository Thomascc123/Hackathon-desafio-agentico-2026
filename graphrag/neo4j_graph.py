"""
Neo4j backend for the Normativa UdeA knowledge graph.
Implements the BaseNormativaGraph interface using Cypher queries.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import networkx as nx
from neo4j import GraphDatabase, Driver, Session

from .base_graph import BaseNormativaGraph
from .graph_models import NodeType, EdgeType

logger = logging.getLogger(__name__)


class Neo4jNormativaGraph(BaseNormativaGraph):
    """Neo4j-backed implementation of the normative knowledge graph.

    Node types are stored as Neo4j labels. Edge types are stored as
    relationship types.

    Provides a lazy-loaded `.graph` (nx.MultiDiGraph) property for code
    that needs direct node/edge iteration (visualization, source index,
    hierarchy traversal). Query methods use Cypher directly for performance.
    """

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        self._driver: Driver = GraphDatabase.driver(uri, auth=(user, password))
        self._database = database
        self._nx_graph: nx.MultiDiGraph | None = None
        self._verify_connectivity()

    @property
    def graph(self) -> nx.MultiDiGraph:
        """Lazy-loaded NetworkX copy for direct-access consumers."""
        if self._nx_graph is None:
            self._nx_graph = self._load_networkx()
        return self._nx_graph

    def _load_networkx(self) -> nx.MultiDiGraph:
        g = nx.MultiDiGraph()
        data = self.export_to_json()
        for node in data.get("nodes", []):
            props = {
                "type": self._parse_node_type(node["type"]),
                "label": node["label"],
                **node.get("properties", {}),
            }
            g.add_node(node["id"], **props)
        for edge in data.get("edges", []):
            props = {
                "type": self._parse_edge_type(edge["type"]),
                **edge.get("properties", {}),
            }
            g.add_edge(edge["source"], edge["target"], key=edge["type"], **props)
        return g

    def _verify_connectivity(self):
        try:
            self._driver.verify_connectivity()
            logger.info("Connected to Neo4j at %s", self._driver._routing_control_address)
        except Exception as e:
            logger.error("Failed to connect to Neo4j: %s", e)
            raise

    def close(self):
        self._driver.close()

    # ── Session helper ───────────────────────────────────────────────

    def _session(self) -> Session:
        return self._driver.session(database=self._database)

    # ── Query implementations ─────────────────────────────────────────

    def query_articles_by_keyword(self, keyword: str) -> list[dict]:
        """Find articles whose text contains any of the keyword words."""
        import unicodedata

        def _normalize(t: str) -> str:
            t = t.lower()
            return ''.join(
                c for c in unicodedata.normalize('NFKD', t)
                if not unicodedata.combining(c)
            )

        stopwords = {'los', 'las', 'del', 'para', 'con', 'por', 'que', 'una',
                     'puede', 'como', 'mas', 'son', 'sus', 'debe', 'sobre',
                     'entre', 'durante', 'cada', "qué", 'tiene', 'esta'}
        words = [
            w for w in _normalize(keyword).split()
            if len(w) >= 3 and w not in stopwords and w.isalpha()
        ]
        if not words:
            words = [_normalize(keyword)]

        results: list[dict] = []
        with self._session() as session:
            for word in words:
                query = """
                    MATCH (a:Articulo)
                    WHERE a.texto CONTAINS $word OR a.texto_completo CONTAINS $word
                    RETURN a.id AS id, a.numero AS articulo,
                           a.texto AS texto,
                           a.texto_completo AS texto_completo,
                           a.documento_codigo AS documento_codigo,
                           a.num_modificaciones AS modificaciones
                    LIMIT 100
                """
                rows = session.run(query, word=word).data()
                for r in rows:
                    doc_codigo = r.get("documento_codigo", "")
                    doc_label = ""
                    doc_asunto = ""
                    if doc_codigo:
                        doc_row = session.run(
                            "MATCH (d:Documento {codigo: $c}) RETURN d.label AS label, d.asunto AS asunto",
                            c=doc_codigo,
                        ).single()
                        if doc_row:
                            doc_label = doc_row.get("label", "")
                            doc_asunto = doc_row.get("asunto", "")
                    results.append({
                        "articulo": r.get("articulo", ""),
                        "texto": (r.get("texto") or "")[:300],
                        "texto_completo": r.get("texto_completo") or "",
                        "documento_codigo": doc_codigo,
                        "documento": doc_asunto or doc_label,
                        "documento_label": doc_label,
                        "documento_asunto": doc_asunto,
                        "modificaciones": r.get("modificaciones", 0),
                    })
        return results

    def query_evolution_of_article(self, art_numero: str) -> list[dict]:
        with self._session() as session:
            query = """
                MATCH (a:Articulo {numero: $art_numero})
                OPTIONAL MATCH (d:Documento)-[mod:MODIFICA]->(a)
                RETURN a.numero AS articulo,
                       a.texto AS texto_actual,
                       a.num_modificaciones AS num_modificaciones,
                       d.numero AS doc_numero,
                       d.fecha AS fecha,
                       d.anio AS anio,
                       d.resuelve AS accion,
                       d.label AS doc_label
            """
            rows = session.run(query, art_numero=art_numero).data()
            if not rows:
                return []

            # First pass: collect modification events
            results: list[dict] = []
            seen_docs = set()
            for r in rows:
                doc_num = r.get("doc_numero")
                if doc_num and doc_num not in seen_docs:
                    seen_docs.add(doc_num)
                    results.append({
                        "modificado_por": r.get("doc_label") or f"AS {doc_num}",
                        "fecha": r.get("fecha") or "",
                        "anio": r.get("anio") or "",
                        "accion": (r.get("accion") or "")[:100],
                    })
            results.sort(key=lambda x: x.get("anio", ""))

            # Add current text info
            first = rows[0]
            results.append({
                "articulo": art_numero,
                "texto_actual": (first.get("texto_actual") or "")[:200],
                "num_modificaciones": first.get("num_modificaciones", 0),
            })
            return results

    def query_document_timeline(self, asunto: str = "") -> list[dict]:
        with self._session() as session:
            if asunto:
                query = """
                    MATCH (d:Documento)
                    WHERE d.asunto = $asunto
                      AND (d.is_referenced IS NULL OR d.is_referenced = false)
                    RETURN d.id AS id, d.numero AS numero, d.fecha AS fecha,
                           d.anio AS anio, d.resuelve AS resuelve,
                           d.autoridad AS autoridad
                    ORDER BY d.fecha DESC
                """
                rows = session.run(query, asunto=asunto.upper()).data()
            else:
                query = """
                    MATCH (d:Documento)
                    WHERE d.is_referenced IS NULL OR d.is_referenced = false
                    RETURN d.id AS id, d.numero AS numero, d.fecha AS fecha,
                           d.anio AS anio, d.resuelve AS resuelve,
                           d.autoridad AS autoridad
                    ORDER BY d.fecha DESC
                """
                rows = session.run(query).data()

            return [
                {
                    "id": r.get("id", ""),
                    "numero": r.get("numero", ""),
                    "fecha": r.get("fecha", ""),
                    "anio": r.get("anio", ""),
                    "resuelve": (r.get("resuelve") or "")[:100],
                    "autoridad": r.get("autoridad", ""),
                }
                for r in rows
            ]

    def query_articles_modified_by(self, doc_numero: str) -> list[dict]:
        with self._session() as session:
            query = """
                MATCH (d:Documento {numero: $doc_numero})-[mod:MODIFICA]->(a:Articulo)
                RETURN a.numero AS articulo,
                       a.texto AS texto,
                       mod.articulos_referenciados AS relacion
                LIMIT 50
            """
            rows = session.run(query, doc_numero=doc_numero).data()
            return [
                {
                    "articulo": r.get("articulo", ""),
                    "texto": (r.get("texto") or "")[:150],
                    "relacion": r.get("relacion") or "",
                }
                for r in rows
            ]

    # ── Export ────────────────────────────────────────────────────────

    def export_to_json(self) -> dict:
        """Export the full graph as JSON by querying all nodes and relationships."""
        nodes = []
        edges = []

        with self._session() as session:
            # Fetch all nodes
            node_query = """
                MATCH (n)
                RETURN n.id AS id, labels(n) AS labels, properties(n) AS props
            """
            for record in session.run(node_query):
                nid = record["id"]
                labels = record["labels"]
                props = dict(record["props"])
                # Determine the primary type from labels
                ntype = self._resolve_node_type(labels)
                label = props.pop("label", nid)
                # Remove internal Neo4j fields
                props.pop("id", None)
                nodes.append({
                    "id": nid,
                    "type": ntype,
                    "label": label,
                    "properties": props,
                })

            # Fetch all relationships
            edge_query = """
                MATCH (s)-[r]->(t)
                RETURN s.id AS source, t.id AS target,
                       type(r) AS rel_type, properties(r) AS props
            """
            for record in session.run(edge_query):
                props = dict(record["props"])
                edges.append({
                    "source": record["source"],
                    "target": record["target"],
                    "type": record["rel_type"],
                    "properties": props,
                })

        return {"nodes": nodes, "edges": edges}

    def _resolve_node_type(self, labels: list[str]) -> str:
        """Map Neo4j labels to node type strings matching NodeType enum values."""
        for label in labels:
            label_clean = label.strip().lower()
            for nt in NodeType:
                if nt.value.lower() == label_clean:
                    return nt.value
        return labels[0] if labels else "Unknown"

    def summary(self) -> dict:
        with self._session() as session:
            node_counts = session.run(
                "MATCH (n) RETURN labels(n) AS label, count(*) AS cnt"
            ).data()
            edge_counts = session.run(
                "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS cnt"
            ).data()

            node_type_counts = {}
            for row in node_counts:
                resolved = self._resolve_node_type(row["label"])
                node_type_counts[resolved] = node_type_counts.get(resolved, 0) + row["cnt"]

            edge_type_counts = {}
            for row in edge_counts:
                edge_type_counts[row["type"]] = edge_type_counts.get(row["type"], 0) + row["cnt"]

            total_nodes = sum(node_type_counts.values())
            total_edges = sum(edge_type_counts.values())

            return {
                "nodes": total_nodes,
                "edges": total_edges,
                "node_types": node_type_counts,
                "edge_types": edge_type_counts,
            }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
