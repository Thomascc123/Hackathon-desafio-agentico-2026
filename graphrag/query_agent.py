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


# ── Intent patterns ──────────────────────────────────────────────

INTENT_PATTERNS: list[tuple[str, str, list[str]]] = [
    # (intent_name, regex_pattern, required_groups)
    ("article_text", r'(?:qué dice|muestra|texto|contenido|leer)\s+(?:el\s+)?art[ií]culo\s+(\d+)', ["art_num"]),
    ("article_text", r'art[ií]culo\s+(\d+)\s*(?:del\s+reglamento)?\s*(?:dice|establece|se[ñn]ala|indica|define)', ["art_num"]),

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

def _fmt_article(art: dict) -> str:
    texto = art.get("texto", "")
    mods = art.get("modificaciones", [])
    mod_str = ""
    if mods:
        mod_str = f"\n  📝 {len(mods)} modificación(es): " + "; ".join(
            f"{m.get('tipo_norma','')} {m.get('numero','')}/{m.get('anio','')}"
            for m in mods[:5]
        )
    return f"**Artículo {art.get('numero')}**\n{_truncate(texto, 300)}.{mod_str}"


def _fmt_modification(item: dict) -> str:
    return (f"  • {item.get('anio', '?')}: Modificado por **{item.get('modificado_por', '?')}**\n"
            f"    Acción: {_truncate(item.get('accion', item.get('texto_actual', '')), 100)}")


def _truncate(text: str, max_len: int = 200) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len].rsplit(" ", 1)[0] + "..."


# ── Agent class ──────────────────────────────────────────────────

class GraphRAGAgent:
    """GraphRAG agent that answers questions about UdeA normative documents."""

    def __init__(self, graph):
        self.graph = graph
        self._load_concept_index()

    def _load_concept_index(self):
        """Build a concept → articles index for fast lookup."""
        self.concept_index = {}
        for nid, data in self.graph.graph.nodes(data=True):
            if data.get("type") == NodeType.ARTICULO:
                texto = (data.get("texto", "") + " " +
                         data.get("texto_completo", "")).lower()
                # Index key concepts
                concepts = self._extract_concepts(texto)
                for c in concepts:
                    self.concept_index.setdefault(c, []).append({
                        "numero": data.get("numero"),
                        "texto": data.get("texto", "")[:200],
                        "documento": data.get("documento_codigo", ""),
                        "modificaciones": data.get("num_modificaciones", 0),
                    })

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
        """Classify user question into an intent and extract parameters."""
        question_clean = question.strip().lower()

        for intent, pattern, groups in INTENT_PATTERNS:
            m = re.search(pattern, question_clean, re.IGNORECASE | re.UNICODE)
            if m:
                params = {}
                # Map groups: use last group by default for the first param
                for i, g in enumerate(groups):
                    # Use group from the end if index exceeds available groups
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
        """Process a natural language question and return an answer."""
        intent, params = self.classify_intent(question)
        handler = self._get_handler(intent)
        if handler is None:
            return self._unknown_intent_response(question)
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

    def _handle_article_text(self, params: dict) -> list:
        art_num = params.get("art_num", "")
        pattern = f"%_{art_num}"
        results = []
        for nid, data in self.graph.graph.nodes(data=True):
            if data.get("type") == NodeType.ARTICULO and data.get("numero") == art_num:
                # Find which document it belongs to
                doc_label = ""
                for src, _, edata in self.graph.graph.in_edges(nid, data=True):
                    if edata.get("type") == EdgeType.CONTIENE:
                        parent = self.graph.graph.nodes[src]
                        doc_label = parent.get("label", "")
                results.append({
                    "numero": data.get("numero"),
                    "texto": data.get("texto", ""),
                    "texto_completo": data.get("texto_completo", ""),
                    "modificaciones": data.get("num_modificaciones", 0),
                    "documento": doc_label,
                })
        return results

    def _handle_evolution(self, params: dict) -> list:
        art_num = params.get("art_num", "")
        return self.graph.query_evolution_of_article(art_num)

    def _handle_modified_by(self, params: dict) -> list:
        art_num = params.get("art_num", "")
        raw = self.graph.query_evolution_of_article(art_num)
        # Filter to only modification events
        return [r for r in raw if "modificado_por" in r]

    def _handle_keyword_search(self, params: dict) -> list:
        keyword = params.get("keyword", "")
        return self.graph.query_articles_by_keyword(keyword)

    def _handle_document_timeline(self, params: dict) -> list:
        asunto = params.get("asunto", "").upper()
        # Map to known asuntos
        asunto_map = {
            "pregrado": "REGLAMENTO ESTUDIANTIL DE PREGRADO",
            "posgrado": "REGLAMENTO ESTUDIANTIL DE POSGRADO",
            "matrícula": "MATRICULA",
            "matricula": "MATRICULA",
        }
        mapped = asunto_map.get(asunto.lower(), asunto)
        if mapped == asunto and asunto:
            # Try partial match
            for nid, data in self.graph.graph.nodes(data=True):
                if data.get("type") == NodeType.DOCUMENTO:
                    a = data.get("asunto", "")
                    if asunto in a.upper():
                        mapped = a
                        break
        return self.graph.query_document_timeline(asunto=mapped if mapped != asunto else "")

    def _handle_concept_query(self, params: dict) -> list:
        concept = params.get("concept", "").lower()
        arts = self.concept_index.get(concept, [])
        # Also try keyword search
        if not arts:
            arts = self.graph.query_articles_by_keyword(concept)
        return arts

    def _handle_article_by_concept(self, params: dict) -> list:
        return self._handle_concept_query(params)

    def _handle_document_search(self, params: dict) -> list:
        doc_num = params.get("doc_num", "")
        doc_year = params.get("doc_year", "")
        results = []
        for nid, data in self.graph.graph.nodes(data=True):
            if data.get("type") == NodeType.DOCUMENTO:
                if data.get("numero", "").lstrip("0") == doc_num.lstrip("0"):
                    if not doc_year or data.get("anio", "") == doc_year:
                        results.append({
                            "id": nid,
                            "numero": data.get("numero", ""),
                            "fecha": data.get("fecha", ""),
                            "resuelve": data.get("resuelve", ""),
                            "autoridad": data.get("autoridad", ""),
                        })
        return results

    def _handle_help(self, params: dict) -> list:
        return [{"help_text": """
**Comandos disponibles:**

🔍 `¿Qué dice el artículo X?` — Muestra el texto de un artículo
📜 `Historia del artículo X` — Muestra modificaciones a través del tiempo
📝 `¿Qué acuerdos modifican el artículo X?` — Documentos que lo han modificado
🔎 `Artículos sobre [tema]` — Busca artículos por palabra clave
📋 `Documentos de pregrado` — Línea de tiempo de documentos
📖 `Acuerdo Superior XX de YYYY` — Busca un acuerdo específico
❓ `¿Qué son las matrículas de honor?` — Define un concepto
        """}]

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

    def _format_response(self, intent: str, params: dict, result: list) -> str:
        if not result:
            return "No encontré información relacionada con tu consulta."

        if intent == "article_text":
            lines = [f"📖 **Artículo {params.get('art_num')}**"]
            for r in result:
                lines.append("")
                lines.append(r.get("texto", ""))
                if r.get("modificaciones", 0) > 0:
                    lines.append(f"\n_Modificado en {r['modificaciones']} ocasiones_")
            return "\n".join(lines)

        elif intent == "evolution":
            lines = [f"📜 **Evolución del Artículo {params.get('art_num')}**\n"]
            for item in result:
                if "modificado_por" in item:
                    lines.append(_fmt_modification(item))
                elif "texto_actual" in item:
                    lines.append(f"\n**Texto actual:** {_truncate(item.get('texto_actual', ''), 200)}")
            return "\n".join(lines)

        elif intent == "modified_by":
            lines = [f"📝 **Documentos que modifican el Artículo {params.get('art_num')}**\n"]
            for r in result:
                if "modificado_por" in r:
                    lines.append(f"  • {r.get('anio', '?')}: **{r.get('modificado_por', '?')}**")
                    if r.get("accion"):
                        lines[-1] += f"\n    → {_truncate(r['accion'], 100)}"
                elif "articulo" in r:
                    lines.append(f"  • {r.get('articulo')} — {_truncate(r.get('texto', ''), 100)}")
            return "\n".join(lines)

        elif intent == "keyword_search":
            kw = params.get("keyword", "")
            lines = [f"🔎 **Artículos que mencionan '{kw}'** ({len(result)} encontrados)\n"]
            for r in result[:10]:
                lines.append(f"  • **Art. {r.get('articulo')}**: {_truncate(r.get('texto'), 120)}")
            if len(result) > 10:
                lines.append(f"\n... y {len(result) - 10} más.")
            return "\n".join(lines)

        elif intent == "document_timeline":
            lines = [f"📋 **Documentos** ({len(result)} encontrados)\n"]
            for r in result[:10]:
                lines.append(f"  • {r.get('fecha', '?')} | **#{r.get('numero')}** | {_truncate(r.get('resuelve', ''), 80)}")
            if len(result) > 10:
                lines.append(f"\n... y {len(result) - 10} más.")
            return "\n".join(lines)

        elif intent in ("concept_query", "article_by_concept"):
            concept = params.get("concept", "")
            lines = [f"📚 **Artículos relacionados con '{concept}'** ({len(result)} encontrados)\n"]
            for r in result[:10]:
                lines.append(f"  • **Art. {r.get('articulo')}**: {_truncate(r.get('texto', ''), 120)}")
            return "\n".join(lines)

        elif intent == "document_search":
            lines = [f"📖 **Acuerdo Superior {params.get('doc_num', '')}**\n"]
            for r in result:
                lines.append(f"  • Fecha: {r.get('fecha', '?')}")
                lines.append(f"  • Autoridad: {r.get('autoridad', '?')}")
                lines.append(f"  • Resuelve: {r.get('resuelve', '?')}")
            return "\n".join(lines)

        elif intent == "help":
            return result[0].get("help_text", "")

        else:
            lines = [f"Resultados ({len(result)} encontrados):"]
            for r in result[:5]:
                lines.append(str(r))
            return "\n".join(lines)


# ── Interactive CLI ──────────────────────────────────────────────

def interactive_cli(graph):
    """Run an interactive Q&A session."""
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

    # Build or load graph
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
        # One-shot query
        q = " ".join(sys.argv[1:])
        agent = GraphRAGAgent(g)
        print(agent.answer(q))
    else:
        interactive_cli(g)
