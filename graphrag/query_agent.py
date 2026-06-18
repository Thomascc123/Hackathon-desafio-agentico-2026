"""
GraphRAG Query Agent.
Translates natural language questions into graph traversals over the
Normativa UdeA knowledge graph.

Architecture:
  1. Intent Classification: maps user question to query type
  2. Parameter Extraction: extracts key entities (article numbers, concepts, dates)
  3. Graph Traversal: executes the appropriate graph query
  4. Response Synthesis: formats results with citations
"""

import json
import re
import os
from typing import Callable

from .graph_models import NodeType, EdgeType
from .base_graph import BaseNormativaGraph
from agent.models import (
    ArticleTextResult,
    KeywordSearchResult,
    EvolutionItem,
    DocumentTimelineResult,
    DocumentSearchResult,
    HelpResult,
)


# ── Intent patterns ──────────────────────────────────────────────

INTENT_PATTERNS: list[tuple[str, str, list[str]]] = [
    ("article_text", r'(?:qu[ée]\s+)?(?:dice|muestra|texto|contenido|leer)\s+(?:el\s+)?art[ií]culo\s+(\d+)', ["art_num"]),
    ("article_text", r'art[ií]culo\s+(\d+)\s*(?:del\s+reglamento)?\s*(?:\w+\s+)?(?:dice|establece|se[ñn]ala|indica|define|trata)', ["art_num"]),
    ("article_text", r'^(?:el\s+)?art[ií]culo\s+(\d+)\s*$', ["art_num"]),

    ("evolution", r'(?:historia|evoluci[óo]n|modificaciones|c[óo]mo ha cambiado|l[íi]nea de tiempo)\s+(?:del\s+)?(?:el\s+)?art[ií]culo\s+(\d+)', ["art_num"]),
    ("evolution", r'qu[ée]\s+(?:modificaciones|cambios)\s+(?:tiene|ha tenido|se le han hecho)\s+(?:el\s+)?art[ií]culo\s+(\d+)', ["art_num"]),

    ("modified_by", r'qu[ée]\s+(?:documentos|acuerdos|normas|resoluciones)\s+(?:modifican|cambian|alteran)\s+(?:el\s+)?art[ií]culo\s+(\d+)', ["art_num"]),
    ("modified_by", r'(?:qu[ée]\s+)?modifica(?:n)?\s+(?:el\s+)?art[ií]culo\s+(\d+)', ["art_num"]),
    ("modified_by", r'qu[ée]\s+modifica(?:n|ron)?\s+(?:el\s+)?art[ií]culo\s+(\d+)', ["art_num"]),

    ("keyword_search", r'(?:d[óo]nde|en qu[ée] art[ií]culo|busca|encuentra|art[ií]culos|art[ií]culos sobre)\s+(?:se\s+)?(?:habla|menciona|refiere|dice)\s+(?:de\s+|sobre\s+)?(.+)', ["keyword"]),
    ("keyword_search", r'art[ií]culos?\s+(?:relacionados con|sobre|que hablan de|que mencionan)\s+(.+)', ["keyword"]),
    ("keyword_search", r'^art[ií]culos?\s+(?:de|sobre)\s+(.+)$', ["keyword"]),

    ("document_timeline", r'(?:línea de tiempo|cronología|historial|todos los)\s+(?:documentos|acuerdos|normas)\s+(?:de|sobre)\s+(.+)', ["asunto"]),
    ("document_timeline", r'(?:muestra|lista|cu[áa]les son|todos)\s+(?:los\s+)?(?:documentos|acuerdos)\s+(?:de\s+)?(.+?)(?:\s*(?:del|de|$))', ["asunto"]),
    ("document_timeline", r'^documentos\s+(?:de\s+)?(.+?)$', ["asunto"]),
    ("document_timeline", r'^acuerdos\s+(?:de\s+)?(.+?)$', ["asunto"]),

    ("concept_query", r'(?:qu[ée]\s+es|define|explica|significa)\s+(?:las\s+|la\s+|los\s+)?(.{5,}?)(?:\s*\?|$)', ["concept"]),
    ("concept_query", r'(?:cu[áa]les\s+son|qu[ée]\s+son)\s+(?:las\s+|los\s+|el\s+)?(.{5,}?)(?:\s*\?|$)', ["concept"]),

    ("article_by_concept", r'(?:qu[ée]\s+)?art[ií]culo[s]?\s+(?:reglamenta|trata|habla)\s+(?:de|sobre)\s+(.+)', ["concept"]),
    ("article_by_concept", r'(?:en|por)\s+qu[ée]\s+art[ií]culo\s+(?:se\s+)?(?:reglamenta|establece|dice|encuentra)\s+(.+)', ["concept"]),

    ("document_search", r'(?:busca|encuentra|informaci[óo]n sobre)\s+(?:el\s+)?acuerdo\s+(?:superior\s+)?(\d+)\s*(?:de\s+(\d{4}))?', ["doc_num", "doc_year"]),
    ("document_search", r'acuerdo\s+(?:superior\s+)?(\d+)\s+.*?(\d{4})', ["doc_num", "doc_year"]),

    ("help", r'(?:ayuda|help|qu[ée] puedes hacer|comandos|funciones)', []),
]

# ── Response templates ───────────────────────────────────────────


def _truncate(text: str, max_len: int = 200) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len].rsplit(" ", 1)[0] + "..."


def _find_sentence(text: str, keyword: str) -> str:
    import unicodedata
    def _normalize(t: str) -> str:
        return ''.join(c for c in unicodedata.normalize('NFKD', t.lower())
                       if not unicodedata.combining(c))
    raw_key = _normalize(keyword)
    words = [w for w in raw_key.split() if len(w) >= 3]
    if not words:
        words = [raw_key]
    sentences = re.split(r'(?<=[.!?])\s+', text)
    for s in sentences:
        snorm = _normalize(s)
        if any(w in snorm for w in words):
            return s.strip()[:200]
    return text[:150]


# ── Agent class ──────────────────────────────────────────────────

class GraphRAGAgent:
    """GraphRAG agent that answers questions about UdeA normative documents."""

    def __init__(self, graph: BaseNormativaGraph):
        self.graph = graph
        self._load_concept_index()

    def _load_concept_index(self):
        self.concept_index: dict[str, list[KeywordSearchResult]] = {}
        for nid, data in self.graph.graph.nodes(data=True):
            if data.get("type") == NodeType.ARTICULO:
                texto = (data.get("texto", "") + " " +
                         data.get("texto_completo", "")).lower()
                concepts = self._extract_concepts(texto)
                for c in concepts:
                    self.concept_index.setdefault(c, []).append(
                        KeywordSearchResult(
                            articulo=data.get("numero", ""),
                            texto=data.get("texto", "")[:200],
                            documento=data.get("documento_codigo", ""),
                            modificaciones=data.get("num_modificaciones", 0),
                        )
                    )

    def _extract_concepts(self, text: str) -> set:
        keywords = [
            "matrícula", "admisión", "examen", "crédito", "beca", "sanción",
            "rendimiento académico", "cancelación", "transferencia",
            "homologación", "validación", "reingreso", "movilidad",
            "graduación", "título", "práctica", "investigación",
            "disciplinario", "asistencia", "calificación", "nota",
            "habilitación", "premio", "distinción", "estímulo",
        ]
        found = set()
        for kw in keywords:
            if kw.lower() in text:
                found.add(kw)
        return found

    def classify_intent(self, question: str) -> tuple[str, dict]:
        question_clean = question.strip().lower()
        for intent, pattern, groups in INTENT_PATTERNS:
            m = re.search(pattern, question_clean, re.IGNORECASE | re.UNICODE)
            if m:
                params = {}
                for i, g in enumerate(groups):
                    group_idx = i + 1 + (len(m.groups()) - len(groups))
                    if group_idx < 1:
                        group_idx = len(m.groups())
                    vals = [v for v in m.groups() if v is not None]
                    if i < len(vals):
                        params[g] = vals[i] if len(vals) > i else ""
                    else:
                        params[g] = ""
                return intent, params
        return "unknown", {}

    def answer(self, question: str) -> str:
        intent, params = self.classify_intent(question)
        handler = self._get_handler(intent)
        if handler is None:
            result = self._handle_keyword_search({"keyword": question})
            return self._format_response("keyword_search", {"keyword": question}, result)
        result = handler(params)
        return self._format_response(intent, params, result)

    def _get_handler(self, intent: str) -> Callable | None:
        handlers = {
            "article_text": self._handle_article_text,
            "evolution": self._handle_evolution,
            "modified_by": self._handle_modified_by,
            "keyword_search": self._handle_keyword_search,
            "document_timeline": self._handle_document_timeline,
            "concept_query": self._handle_concept_query,
            "article_by_concept": self._handle_article_by_concept,
            "document_search": self._handle_document_search,
            "help": self._handle_help,
        }
        return handlers.get(intent)

    # ── Handlers ─────────────────────────────────────────────────

    def _handle_article_text(self, params: dict) -> list[ArticleTextResult]:
        art_num = params.get("art_num", "")
        results: list[ArticleTextResult] = []
        for nid, data in self.graph.graph.nodes(data=True):
            if data.get("type") == NodeType.ARTICULO and data.get("numero") == art_num:
                doc_label = ""
                doc_asunto = ""
                seen = set()
                stack = [nid]
                while stack:
                    current = stack.pop()
                    if current in seen:
                        continue
                    seen.add(current)
                    for src, _, edata in self.graph.graph.in_edges(current, data=True):
                        if edata.get("type") == EdgeType.CONTIENE:
                            parent_data = self.graph.graph.nodes[src]
                            pt = parent_data.get("type")
                            if pt == NodeType.DOCUMENTO:
                                doc_label = parent_data.get("label", "")
                                doc_asunto = parent_data.get("asunto", "")
                                stack.clear()
                                break
                            elif pt in (NodeType.CAPITULO, NodeType.TITULO):
                                stack.append(src)
                results.append(ArticleTextResult(
                    numero=data.get("numero", ""),
                    texto=data.get("texto", ""),
                    texto_completo=data.get("texto_completo", ""),
                    modificaciones=data.get("num_modificaciones", 0),
                    documento=doc_asunto or doc_label or "Reglamento Estudiantil de Pregrado",
                    documento_asunto=doc_asunto,
                ))
        return results

    def _handle_evolution(self, params: dict) -> list[EvolutionItem]:
        art_num = params.get("art_num", "")
        raw = self.graph.query_evolution_of_article(art_num)
        return [EvolutionItem(**r) for r in raw]

    def _handle_modified_by(self, params: dict) -> list[EvolutionItem]:
        art_num = params.get("art_num", "")
        raw = self.graph.query_evolution_of_article(art_num)
        return [EvolutionItem(**r) for r in raw if "modificado_por" in r]

    def _handle_keyword_search(self, params: dict) -> list[KeywordSearchResult]:
        keyword = params.get("keyword", "")
        raw = self.graph.query_articles_by_keyword(keyword)
        return [KeywordSearchResult(**r) for r in raw]

    def _handle_document_timeline(self, params: dict) -> list[DocumentTimelineResult]:
        asunto = params.get("asunto", "").upper()
        asunto_map = {
            "pregrado": "REGLAMENTO ESTUDIANTIL DE PREGRADO",
            "posgrado": "REGLAMENTO ESTUDIANTIL DE POSGRADO",
            "matrícula": "MATRICULA",
            "matricula": "MATRICULA",
        }
        mapped = asunto_map.get(asunto.lower(), asunto)
        if mapped == asunto and asunto:
            for nid, data in self.graph.graph.nodes(data=True):
                if data.get("type") == NodeType.DOCUMENTO:
                    a = data.get("asunto", "")
                    if asunto in a.upper():
                        mapped = a
                        break
        raw = self.graph.query_document_timeline(asunto=mapped if mapped != asunto else "")
        return [DocumentTimelineResult(**r) for r in raw]

    def _handle_concept_query(self, params: dict) -> list[KeywordSearchResult]:
        concept = params.get("concept", "").lower()
        arts = self.concept_index.get(concept, [])
        if not arts:
            raw = self.graph.query_articles_by_keyword(concept)
            arts = [KeywordSearchResult(**r) for r in raw]
        return arts

    def _handle_article_by_concept(self, params: dict) -> list[KeywordSearchResult]:
        return self._handle_concept_query(params)

    def _handle_document_search(self, params: dict) -> list[DocumentSearchResult]:
        doc_num = params.get("doc_num", "")
        doc_year = params.get("doc_year", "")
        results: list[DocumentSearchResult] = []
        for nid, data in self.graph.graph.nodes(data=True):
            if data.get("type") == NodeType.DOCUMENTO:
                if data.get("numero", "").lstrip("0") == doc_num.lstrip("0"):
                    if not doc_year or data.get("anio", "") == doc_year:
                        results.append(DocumentSearchResult(
                            id=nid,
                            numero=data.get("numero", ""),
                            fecha=data.get("fecha", ""),
                            resuelve=data.get("resuelve", ""),
                            autoridad=data.get("autoridad", ""),
                        ))
        return results

    def _handle_help(self, params: dict) -> list[HelpResult]:
        return [HelpResult(help_text="""
**Comandos disponibles:**

🔍 `¿Qué dice el artículo X?` — Muestra el texto de un artículo
📜 `Historia del artículo X` — Muestra modificaciones a través del tiempo
📝 `¿Qué acuerdos modifican el artículo X?` — Documentos que lo han modificado
🔎 `Artículos sobre [tema]` — Busca artículos por palabra clave
📋 `Documentos de pregrado` — Línea de tiempo de documentos
📖 `Acuerdo Superior XX de YYYY` — Busca un acuerdo específico
❓ `¿Qué son las matrículas de honor?` — Define un concepto
        """)]

    def _unknown_intent_response(self, question: str) -> str:
        return (
            f"No entendí la pregunta: \"{question}\".\n\n"
            f"Prueba con:\n"
            f"  • \"¿Qué dice el artículo 63?\"\n"
            f"  • \"Historia del artículo 130\"\n"
            f"  • \"Artículos sobre matrícula\"\n"
            f"  • \"Documentos de pregrado\"\n"
            f"  • Escribe 'ayuda' para más opciones."
        )

    # ── Response formatting ──────────────────────────────────────

    def _format_response(self, intent: str, params: dict, result: list) -> str:
        if not result:
            return "No encontré información relacionada con tu consulta."

        if intent == "article_text":
            return self._fmt_article_text(params, result)

        elif intent == "evolution":
            return self._fmt_evolution(params, result)

        elif intent == "modified_by":
            return self._fmt_modified_by(params, result)

        elif intent in ("keyword_search", "concept_query", "article_by_concept"):
            kw = params.get("keyword") or params.get("concept", "")
            return self._fmt_keyword_search(kw, result)

        elif intent == "document_timeline":
            return self._fmt_document_timeline(params, result)

        elif intent == "document_search":
            return self._fmt_document_search(params, result)

        elif intent == "help":
            return result[0].help_text

        else:
            lines = [f"**Resultados** ({len(result)} encontrados):"]
            for r in result[:5]:
                lines.append(str(r))
            return "\n".join(lines)

    def _fmt_article_text(self, params: dict, result: list[ArticleTextResult]) -> str:
        lines = ["**Respuesta rápida:**"]
        for r in result:
            lines.append(f"**Artículo {r.numero}**")
            texto = r.texto or r.texto_completo
            lines.append(texto[:300])
            if r.documento:
                lines.append(f"\n_Fuente: {r.documento}_")
            if r.modificaciones > 0:
                lines.append(f"_Modificado en {r.modificaciones} ocasiones_")
        return "\n".join(lines)

    def _fmt_evolution(self, params: dict, result: list[EvolutionItem]) -> str:
        art = params.get("art_num", "")
        lines = [f"**Evolución del Artículo {art}**\n"]
        mods = [r for r in result if r.modificado_por]
        current = [r for r in result if r.texto_actual]

        if mods:
            lines.append(f"**Modificaciones ({len(mods)}):**")
            for m in mods:
                lines.append(f"  • {m.anio or '?'}: **{m.modificado_por}**"
                             f"\n    → {_truncate(m.accion, 120)}")

        if current:
            lines.append(f"\n**Texto vigente:**")
            lines.append(f"{_truncate(current[0].texto_actual, 300)}")

        return "\n".join(lines)

    def _fmt_modified_by(self, params: dict, result: list[EvolutionItem]) -> str:
        art = params.get("art_num", "")
        lines = [f"**Documentos que modifican el Artículo {art}**\n"]
        for r in result:
            if r.modificado_por:
                lines.append(f"  • {r.anio or '?'}: **{r.modificado_por}**")
                if r.accion:
                    lines[-1] += f"\n    → {_truncate(r.accion, 100)}"
        return "\n".join(lines)

    def _fmt_keyword_search(self, keyword: str, result: list[KeywordSearchResult]) -> str:
        lines = ["**Respuesta rápida:**"]
        lines.append(f"Se encontraron {len(result)} artículos relacionados con \"{keyword}\".")
        lines.append("")

        lines.append("**Fuentes consultadas:**\n")
        for r in result[:5]:
            doc_info = f" — {r.documento_asunto}" if r.documento_asunto else ""
            lines.append(f"📄 **Art. {r.articulo}**{doc_info}")
            body = r.texto_completo or r.texto
            excerpt = _find_sentence(body, keyword)
            lines.append(f"   \"{excerpt}\"")
            lines.append("")

        if len(result) > 5:
            remaining = [r.articulo for r in result[5:10]]
            lines.append(f"... y {len(result) - 5} artículos más (Art. {', '.join(remaining)}{', ...' if len(result) > 10 else ''})")

        lines.append("")
        lines.append(f"**Donde se menciona \"{keyword}\":**")
        for r in result[:8]:
            texto = r.texto_completo or r.texto
            excerpt = _find_sentence(texto, keyword)
            lines.append(f"  • **Art. {r.articulo}**: “{excerpt}”")

        return "\n".join(lines)

    def _fmt_document_timeline(self, params: dict, result: list[DocumentTimelineResult]) -> str:
        lines = [f"**Documentos encontrados** ({len(result)})\n"]
        for r in result[:10]:
            lines.append(f"  • {r.fecha or '?'} | **#{r.numero}** | {_truncate(r.resuelve, 80)}")
        if len(result) > 10:
            lines.append(f"\n... y {len(result) - 10} más.")
        return "\n".join(lines)

    def _fmt_document_search(self, params: dict, result: list[DocumentSearchResult]) -> str:
        lines = [f"**Acuerdo Superior {params.get('doc_num', '')}**\n"]
        for r in result:
            lines.append(f"  • Fecha: {r.fecha or '?'}")
            lines.append(f"  • Autoridad: {r.autoridad or '?'}")
            lines.append(f"  • Resuelve: {r.resuelve or '?'}")
        return "\n".join(lines)


# ── Interactive CLI ──────────────────────────────────────────────

def interactive_cli(graph):
    agent = GraphRAGAgent(graph)

    print("\n" + "=" * 60)
    print("  GraphRAG Agent — Normativa UdeA")
    print("  Escribe 'salir' o 'exit' para terminar")
    print("  Escribe 'ayuda' para ver comandos")
    print("=" * 60)

    while True:
        try:
            q = input("\n❓ ").strip()
            if q.lower() in ("salir", "exit", "quit", "q"):
                break
            if not q:
                continue
            print("\n" + agent.answer(q))
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n⚠️ Error: {e}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from graphrag.pipeline import run_pipeline
    import json

    graph_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "graphrag_output",
        "knowledge_graph.json",
    )

    if os.path.exists(graph_path):
        from graphrag.graph_builder import NormativaGraph
        g = NormativaGraph()
        with open(graph_path) as f:
            data = json.load(f)
        for node in data.get("nodes", []):
            props = {"type": NodeType(node["type"]), "label": node["label"], **node["properties"]}
            g.graph.add_node(node["id"], **props)
        for edge in data.get("edges", []):
            props = {"type": EdgeType(edge["type"]), **edge["properties"]}
            g.graph.add_edge(edge["source"], edge["target"], key=edge["type"], **props)
        print(f"Graph loaded: {g.graph.number_of_nodes()} nodes, {g.graph.number_of_edges()} edges")
    else:
        print("Building graph from pipeline...")
        g = run_pipeline(limit=10)

    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        agent = GraphRAGAgent(g)
        print(agent.answer(q))
    else:
        interactive_cli(g)
