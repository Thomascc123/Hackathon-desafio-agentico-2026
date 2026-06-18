"""
Build a Knowledge Graph from UdeA normative documents.
Uses NetworkX for in-memory graph + exports for Neo4j import.
"""

import json
import re
from collections.abc import Iterator
import networkx as nx
from collections import defaultdict
from datetime import datetime

from .graph_models import Node, Edge, NodeType, EdgeType
from .base_graph import BaseNormativaGraph


class NetworkXNormativaGraph(BaseNormativaGraph):
    """Knowledge graph for UdeA normative documents backed by NetworkX.

    This is the default local backend. For production, use Neo4jNormativaGraph.
    """

    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self._node_counters = defaultdict(int)

    def _node_id(self, prefix: str, key: str) -> str:
        return f"{prefix}:{key}"

    # --- Document nodes ---

    def add_document(self, codigo: str, numero: str, fecha: str,
                     resuelve: str = "", tipo_doc: str = "",
                     autoridad: str = "", asunto: str = "",
                     normas_rel: str = "", vigencia: str = "") -> str:
        nid = self._node_id("DOC", codigo)
        label = f"{tipo_doc or 'Documento'} {numero} - {fecha}"
        properties = {
            "codigo": codigo,
            "numero": numero,
            "fecha": fecha,
            "vigencia": vigencia,
            "tipo_documento": tipo_doc,
            "autoridad": autoridad,
            "asunto": asunto,
            "resuelve": resuelve,
            "normas_relacionadas": normas_rel,
            "anio": fecha[:4] if fecha else "",
        }
        self.graph.add_node(nid, type=NodeType.DOCUMENTO, label=label, **properties)
        return nid

    def add_articulo(self, doc_codigo: str, art_numero: str, texto: str,
                     paragrafos: list = None, modificaciones: list = None) -> str:
        nid = self._node_id("ART", f"{doc_codigo}_{art_numero}")
        label = f"Art. {art_numero}"
        properties = {
            "numero": art_numero,
            "texto": texto[:500],
            "texto_completo": texto,
            "documento_codigo": doc_codigo,
            "num_paragrafos": len(paragrafos or []),
            "num_modificaciones": len(modificaciones or []),
        }
        self.graph.add_node(nid, type=NodeType.ARTICULO, label=label, **properties)
        return nid

    def add_capitulo(self, doc_codigo: str, cap_numero: str) -> str:
        nid = self._node_id("CAP", f"{doc_codigo}_{cap_numero}")
        label = f"Capítulo {cap_numero}"
        properties = {"numero": cap_numero, "documento_codigo": doc_codigo}
        self.graph.add_node(nid, type=NodeType.CAPITULO, label=label, **properties)
        return nid

    def add_titulo(self, doc_codigo: str, tit_numero: str) -> str:
        nid = self._node_id("TIT", f"{doc_codigo}_{tit_numero}")
        label = f"Título {tit_numero}"
        properties = {"numero": tit_numero, "documento_codigo": doc_codigo}
        self.graph.add_node(nid, type=NodeType.TITULO, label=label, **properties)
        return nid

    def add_autoridad(self, nombre: str) -> str:
        nid = self._node_id("AUT", nombre)
        if not self.graph.has_node(nid):
            self.graph.add_node(nid, type=NodeType.AUTORIDAD, label=nombre, nombre=nombre)
        return nid

    def add_concepto(self, nombre: str, contexto: str = "") -> str:
        key = nombre.upper().strip()
        nid = self._node_id("CON", key)
        if not self.graph.has_node(nid):
            self.graph.add_node(nid, type=NodeType.CONCEPTO, label=nombre,
                                nombre=nombre, contexto=contexto[:200])
        return nid

    # --- Edges ---

    def add_edge(self, source_id: str, target_id: str, etype: EdgeType,
                 properties: dict = None):
        self.graph.add_edge(source_id, target_id, key=etype.value,
                            type=etype, **(properties or {}))

    def add_contiene(self, parent_id: str, child_id: str, orden: int = 0):
        self.add_edge(parent_id, child_id, EdgeType.CONTIENE, {"orden": orden})

    def add_modifica(self, doc_id: str, art_id: str, articulos_ref: str = ""):
        self.add_edge(doc_id, art_id, EdgeType.MODIFICA,
                      {"articulos_referenciados": articulos_ref})

    def add_cita(self, source_id: str, target_id: str):
        self.add_edge(source_id, target_id, EdgeType.CITA)

    def add_deroga(self, source_id: str, target_id: str):
        self.add_edge(source_id, target_id, EdgeType.DEROGA)

    def add_actualiza(self, new_id: str, old_id: str):
        self.add_edge(new_id, old_id, EdgeType.ACTUALIZA)

    def add_emite(self, aut_id: str, doc_id: str):
        self.add_edge(aut_id, doc_id, EdgeType.EMITE)

    def add_reglamenta(self, doc_id: str, concept_id: str):
        self.add_edge(doc_id, concept_id, EdgeType.REGLAMENTA)

    # --- Bulk builders ---

    def build_from_metadata(self, docs_metadata: list[dict]):
        """Add documents from website search results metadata."""
        for d in docs_metadata:
            doc_id = self.add_document(
                codigo=d.get("codigo", ""),
                numero=d.get("numero", ""),
                fecha=d.get("fecha", ""),
                resuelve=d.get("resuelve", ""),
                tipo_doc=d.get("tipo_documento", ""),
                autoridad=d.get("autoridad", ""),
                asunto=d.get("asunto", ""),
                normas_rel=d.get("normas_relacionadas", ""),
                vigencia=d.get("vigencia", ""),
            )

            # Add authority node + EMITE edge
            if d.get("autoridad"):
                aut_id = self.add_autoridad(d["autoridad"])
                self.add_emite(aut_id, doc_id)

            # Extract concepts from resuelve
            conceptos = re.findall(r'(MATRÍCULA|REGLAMENTO|RENDIMIENTO|ADMISIÓN|TRANSFERENCIA|CANCELACIÓN|GRADUACIÓN|BECA|SANCIÓN|DISCIPLINARIO|HOMOLOGACIÓN|VALIDACIÓN|REINGRESO|MOVILIDAD|EXAMEN|CRÉDITO|PENSUM|PLAN DE ESTUDIOS)',
                                   d.get("resuelve", ""), re.IGNORECASE)
            for c in set(c.lower() for c in conceptos):
                concept_id = self.add_concepto(c)
                self.add_reglamenta(doc_id, concept_id)

            # Extract referenced documents
            refs = re.findall(r'(?:ACUERDO\s+SUPERIOR|AS)\s+(\d+(?:[A-Za-z])?)\s*(?:DE\s+(\d{4}))?',
                              d.get("normas_relacionadas", ""), re.IGNORECASE)
            for ref_num, ref_year in refs:
                ref_nid = self._node_id("DOC", f"search:{ref_num}")
                if self.graph.has_node(ref_nid) or True:  # create if not exists
                    if not self.graph.has_node(ref_nid):
                        self.graph.add_node(ref_nid, type=NodeType.DOCUMENTO,
                                            label=f"Acuerdo Superior {ref_num}",
                                            numero=ref_num, anio=ref_year,
                                            is_referenced=True)
                    self.add_cita(doc_id, ref_nid)

    def build_from_parsed_reglamento(self, parsed: dict, doc_codigo: str):
        """Add parsed reglamento content (titles, chapters, articles)."""
        doc_id = self._node_id("DOC", doc_codigo)
        if not self.graph.has_node(doc_id):
            doc_id = self.add_document(codigo=doc_codigo, numero="1",
                                       fecha="1981/02/15",
                                       resuelve="Reglamento Estudiantil de Pregrado")

        orden = 0
        for titulo in parsed.get("titulos", []):
            tit_id = self.add_titulo(doc_codigo, titulo["numero"])
            self.add_contiene(doc_id, tit_id, orden)
            orden += 1

            for cap in titulo.get("capitulos", []):
                cap_id = self.add_capitulo(doc_codigo, cap["numero"])
                self.add_contiene(tit_id, cap_id, 0)

                for art in cap.get("articulos", []):
                    art_id = self._add_articulo_with_relations(art, doc_codigo)
                    self.add_contiene(cap_id, art_id, int(art.get("numero", 0)))

            for art in titulo.get("articulos_directos", []):
                art_id = self._add_articulo_with_relations(art, doc_codigo)
                self.add_contiene(tit_id, art_id, int(art.get("numero", 0)))

    def _add_articulo_with_relations(self, art: dict, doc_codigo: str) -> str:
        art_id = self.add_articulo(
            doc_codigo,
            art["numero"],
            art.get("texto", ""),
            art.get("paragrafos"),
            art.get("modificaciones"),
        )
        # Add MODIFICA edges from modification annotations
        for mod in art.get("modificaciones", []):
            mod_doc_num = mod.get("numero", "")
            mod_anio = mod.get("anio", "")
            if mod_doc_num:
                mod_doc_id = self._node_id("DOC", f"search:{mod_doc_num}")
                if not self.graph.has_node(mod_doc_id):
                    ref_label = f"{mod.get('tipo_norma', 'AS')} {mod_doc_num}"
                    if mod_anio:
                        ref_label += f"/{mod_anio}"
                    self.graph.add_node(mod_doc_id, type=NodeType.DOCUMENTO,
                                        label=ref_label,
                                        numero=mod_doc_num, anio=mod_anio,
                                        is_referenced=True)
                self.add_modifica(mod_doc_id, art_id,
                                  mod.get("articulos_ref", ""))
        return art_id

    # --- Queries ---

    def query_articles_modified_by(self, doc_numero: str) -> list[dict]:
        """Find all articles modified by a given document number."""
        doc_id = self._node_id("DOC", f"search:{doc_numero}")
        if not self.graph.has_node(doc_id):
            doc_id = self._node_id("DOC", doc_numero)
        results = []
        for _, art_id, data in self.graph.out_edges(doc_id, data=True):
            if data.get("type") == EdgeType.MODIFICA:
                node = self.graph.nodes[art_id]
                results.append({
                    "articulo": node.get("numero"),
                    "texto": node.get("texto", "")[:150],
                    "relacion": data.get("articulos_referenciados", ""),
                })
        return results

    def query_document_timeline(self, asunto: str = "") -> list[dict]:
        """Get all documents sorted by date for a given subject."""
        docs = []
        for nid, data in self.graph.nodes(data=True):
            if data.get("type") == NodeType.DOCUMENTO:
                if asunto and data.get("asunto", "").upper() != asunto.upper():
                    continue
                if data.get("is_referenced"):
                    continue
                docs.append({
                    "id": nid,
                    "numero": data.get("numero", ""),
                    "fecha": data.get("fecha", ""),
                    "anio": data.get("anio", ""),
                    "resuelve": data.get("resuelve", "")[:100],
                    "autoridad": data.get("autoridad", ""),
                })
        docs.sort(key=lambda x: x.get("fecha", ""), reverse=True)
        return docs

    def query_articles_by_keyword(self, keyword: str) -> list[dict]:
        """Find articles containing a keyword in their text.
        Splits the query into individual meaningful words and matches any of them."""
        results = []
        # Normalize: lowercase + remove accents
        import unicodedata
        def _normalize(t: str) -> str:
            t = t.lower()
            return ''.join(
                c for c in unicodedata.normalize('NFKD', t)
                if not unicodedata.combining(c)
            )

        # Extract meaningful words (3+ chars, skip stopwords)
        stopwords = {'los', 'las', 'del', 'para', 'con', 'por', 'que', 'una',
                     'puede', 'como', 'mas', 'son', 'sus', 'debe', 'sobre',
                     'entre', 'durante', 'cada', "qué", 'tiene', 'esta'}
        words = [
            w for w in _normalize(keyword).split()
            if len(w) >= 3 and w not in stopwords and w.isalpha()
        ]
        if not words:
            words = [_normalize(keyword)]

        for nid, data in self.graph.nodes(data=True):
            if data.get("type") == NodeType.ARTICULO:
                raw_text = data.get("texto", "") + " " + data.get("texto_completo", "")
                texto = _normalize(raw_text)
                if any(w in texto for w in words):
                    doc_label = ""
                    doc_asunto = ""
                    doc_codigo = data.get("documento_codigo", "")
                    if doc_codigo:
                        doc_nid = self._node_id("DOC", doc_codigo)
                        if self.graph.has_node(doc_nid):
                            doc_label = self.graph.nodes[doc_nid].get("label", doc_codigo)
                            doc_asunto = self.graph.nodes[doc_nid].get("asunto", "")
                    results.append({
                        "articulo": data.get("numero"),
                        "texto": data.get("texto", "")[:300],
                        "texto_completo": data.get("texto_completo", ""),
                        "documento_codigo": doc_codigo,
                        "documento": doc_asunto or doc_label,
                        "documento_label": doc_label,
                        "documento_asunto": doc_asunto,
                        "modificaciones": data.get("num_modificaciones", 0),
                    })
        return results

    def query_evolution_of_article(self, art_numero: str) -> list[dict]:
        """Trace the modification history of a specific article."""
        pattern = self._node_id("ART", f"%_{art_numero}")
        results = []
        for nid, data in self.graph.nodes(data=True):
            if (data.get("type") == NodeType.ARTICULO and
                    data.get("numero") == art_numero):
                texto = data.get("texto", "")
                # Find all docs that modify this article
                for src_id, _, edata in self.graph.in_edges(nid, data=True):
                    if edata.get("type") == EdgeType.MODIFICA:
                        src_node = self.graph.nodes[src_id]
                        results.append({
                            "modificado_por": src_node.get("label", src_id),
                            "fecha": src_node.get("fecha", ""),
                            "anio": src_node.get("anio", ""),
                            "accion": src_node.get("resuelve", "")[:100],
                        })
                results.append({
                    "articulo": art_numero,
                    "texto_actual": texto[:200],
                    "num_modificaciones": data.get("num_modificaciones", 0),
                })
        results.sort(key=lambda x: x.get("anio", ""))
        return results

    # --- BaseGraph interface overrides ---

    def export_to_json(self) -> dict:
        return self.to_json()

    def iter_nodes(self) -> Iterator[tuple[str, dict]]:
        for nid, data in self.graph.nodes(data=True):
            yield nid, dict(data)

    def iter_edges(self) -> Iterator[tuple[str, str, str | None, dict]]:
        for u, v, k, data in self.graph.edges(data=True, keys=True):
            yield u, v, k, dict(data)

    # --- Export ---

    def to_cypher(self) -> str:
        """Generate Cypher queries for Neo4j import."""
        lines = ["// Neo4j Cypher export for Normativa UdeA Knowledge Graph"]
        lines.append(f"// Generated: {datetime.now().isoformat()}")
        lines.append("")

        # Create constraints
        lines.append("CREATE CONSTRAINT doc_codigo IF NOT EXISTS FOR (d:Documento) REQUIRE d.codigo IS UNIQUE;")
        lines.append("CREATE CONSTRAINT art_id IF NOT EXISTS FOR (a:Articulo) REQUIRE a.id IS UNIQUE;")
        lines.append("")

        for nid, data in self.graph.nodes(data=True):
            ntype = data.get("type", "").value if hasattr(data.get("type", ""), "value") else str(data.get("type", ""))
            props = {k: v for k, v in data.items() if k not in ("type",) and v is not None}
            # Escape strings
            for k, v in props.items():
                if isinstance(v, str):
                    v = v.replace("\\", "\\\\").replace("'", "\\'")
                    props[k] = v
            props_str = ", ".join(f"{k}: '{v}'" if isinstance(v, str) else f"{k}: {json.dumps(v)}" for k, v in props.items())
            label = ntype
            lines.append(f"CREATE (:{label} {{id: '{nid}', {props_str}}});")

        for u, v, k, data in self.graph.edges(data=True, keys=True):
            etype = data.get("type", "").value if hasattr(data.get("type", ""), "value") else str(data.get("type", ""))
            props = {k2: v2 for k2, v2 in data.items() if k2 not in ("type",)}
            props_str = ""
            if props:
                for pk, pv in props.items():
                    if isinstance(pv, str):
                        pv = pv.replace("\\", "\\\\").replace("'", "\\'")
                    props_str += f"{pk}: '{pv}', " if isinstance(pv, str) else f"{pk}: {pv}, "
                props_str = " {" + props_str.rstrip(", ") + "}"
            lines.append(f"MATCH (a {{id: '{u}'}}), (b {{id: '{v}'}}) CREATE (a)-[:{etype}{props_str}]->(b);")

        return "\n".join(lines)

    def to_json(self) -> dict:
        """Export graph as JSON-serializable dict."""
        nodes = []
        for nid, data in self.graph.nodes(data=True):
            nodes.append({
                "id": nid,
                "type": data.get("type", "").value if hasattr(data.get("type", ""), "value") else str(data.get("type", "")),
                "label": data.get("label", ""),
                "properties": {k: v for k, v in data.items() if k not in ("type", "label")},
            })
        edges = []
        for u, v, k, data in self.graph.edges(data=True, keys=True):
            edges.append({
                "source": u,
                "target": v,
                "type": data.get("type", "").value if hasattr(data.get("type", ""), "value") else str(data.get("type", "")),
                "properties": {k: v for k, v in data.items() if k not in ("type",)},
            })
        return {"nodes": nodes, "edges": edges}

    def summary(self) -> dict:
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "node_types": dict(
                (k, sum(1 for _, d in self.graph.nodes(data=True) if d.get("type") == k))
                for k in NodeType
            ),
            "edge_types": dict(
                (k, sum(1 for _, _, d in self.graph.edges(data=True) if d.get("type") == k))
                for k in EdgeType
            ),
        }

    def visualize_article_history(self, art_numero: str) -> str:
        """Generate a text-based history visualization."""
        evolution = self.query_evolution_of_article(art_numero)
        if not evolution:
            return f"No data found for Article {art_numero}"

        lines = [f"Historia del Artículo {art_numero}", "=" * 50]
        for item in evolution:
            if "modificado_por" in item:
                lines.append(f"\n  [{item.get('anio', '?')}] Modificado por: {item['modificado_por']}")
                if item.get("accion"):
                    lines.append(f"       Acción: {item['accion']}")
            elif "texto_actual" in item:
                lines.append(f"\n  Texto actual: {item['texto_actual']}...")
                lines.append(f"  Total modificaciones: {item['num_modificaciones']}")
        return "\n".join(lines)


# Backward-compatible alias
NormativaGraph = NetworkXNormativaGraph
