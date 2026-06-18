"""
Live scraping tools for normativa.udea.edu.co.
Allows the agent to search and download documents from the website on demand,
with graceful fallback on connection failure.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile

from pydantic_ai import RunContext
from agent.models import AgentDeps

logger = logging.getLogger(__name__)

_normativa = None


def _get_normativa():
    global _normativa
    if _normativa is None:
        import download_normativa as _normativa
    return _normativa


def _test_connection() -> tuple[bool, str]:
    """Test if normativa.udea.edu.co is reachable."""
    try:
        import requests
        r = requests.get(
            "https://normativa.udea.edu.co",
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        if r.status_code < 500:
            return True, ""
        return False, f"HTTP {r.status_code}"
    except requests.ConnectionError:
        return False, "ConnectionError"
    except requests.Timeout:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)


def register_scrape_tools(agent):
    """Register live scraping tools with a PydanticAI agent."""

    @agent.tool(retries=0)
    async def check_website_available(
        ctx: RunContext[AgentDeps],
    ) -> str:
        """Verifica si el sitio web normativa.udea.edu.co está accesible.
        Úsalo antes de intentar scraping para saber si hay conexión."""
        ok, err = _test_connection()
        if ok:
            return "disponible"
        return f"no_disponible: {err}"

    @agent.tool(retries=0)
    async def search_normativa_website(
        ctx: RunContext[AgentDeps],
        keyword: str,
    ) -> str:
        """Busca documentos en el sitio web normativa.udea.edu.co por palabra clave.
        Úsalo cuando los resultados del grafo local sean insuficientes.
        Retorna una lista de documentos con código, número, fecha y descripción."""
        nv = _get_normativa()
        try:
            html = nv.fetch_page(keyword=keyword, page=1)
            docs = nv.parse_table(html)
            if not docs:
                return json.dumps([], ensure_ascii=False)
            # Add to graph
            _enrich_and_add_to_graph(ctx.deps.graph, docs, keyword)
            return json.dumps(docs[:20], ensure_ascii=False)
        except Exception as e:
            logger.warning("Error scraping website: %s", e)
            return json.dumps({
                "error": f"No se pudo consultar el sitio web: {e}",
                "hint": "usa search_articles para buscar en los datos locales",
            }, ensure_ascii=False)

    @agent.tool(retries=0)
    async def download_document_from_website(
        ctx: RunContext[AgentDeps],
        codigo: str,
    ) -> str:
        """Descarga un documento desde normativa.udea.edu.co por su código numérico.
        Si el documento contiene artículos del reglamento, los parsea y los agrega al grafo.
        Retorna el texto completo del documento y sus artículos."""
        nv = _get_normativa()
        tmpdir = ctx.deps.pdf_dir
        if not tmpdir or not os.path.exists(tmpdir):
            tmpdir = tempfile.mkdtemp()

        fpath = os.path.join(tmpdir, f"scraped_{codigo}.pdf")

        try:
            ext, img = nv.get_document_info(codigo)
            if ext.lower() not in ("pdf", "-"):
                return json.dumps({
                    "error": f"El documento {codigo} no es PDF (extensión: {ext})",
                }, ensure_ascii=False)

            ok, size = nv.download_document(codigo, img, fpath)
            if not ok or size == 0:
                return json.dumps({
                    "error": f"No se pudo descargar el documento {codigo}",
                }, ensure_ascii=False)

            # Try to extract text
            from graphrag.parser import extract_text_from_pdf, parse_consolidated_reglamento
            ok_text, text = extract_text_from_pdf(fpath)
            if not ok_text or not text.strip():
                return json.dumps({
                    "codigo": codigo,
                    "tamano_kb": size // 1024,
                    "nota": "El PDF no tiene capa de texto (posiblemente escaneado)",
                }, ensure_ascii=False)

            # Try to parse as consolidated reglamento
            parsed = parse_consolidated_reglamento(text)
            if parsed and parsed.get("titulos"):
                graph = ctx.deps.graph
                from graphrag.graph_builder import NetworkXNormativaGraph
                if isinstance(graph, NetworkXNormativaGraph):
                    graph.build_from_parsed_reglamento(parsed, codigo)
                else:
                    logger.warning("Skipping build_from_parsed_reglamento: not a NetworkX backend")
                total_arts = sum(
                    sum(len(c.get("articulos", [])) for c in t.get("capitulos", [])) +
                    len(t.get("articulos_directos", []))
                    for t in parsed["titulos"]
                )
                return json.dumps({
                    "codigo": codigo,
                    "texto_primeros_500": text[:500],
                    "tamano_kb": size // 1024,
                    "articulos_parseados": total_arts,
                    "nota": f"Documento parseado: {total_arts} artículos agregados al grafo",
                }, ensure_ascii=False)

            return json.dumps({
                "codigo": codigo,
                "texto_primeros_500": text[:500],
                "tamano_kb": size // 1024,
            }, ensure_ascii=False)

        except Exception as e:
            logger.warning("Error downloading document %s: %s", codigo, e)
            return json.dumps({
                "error": f"Error al descargar documento {codigo}: {e}",
            }, ensure_ascii=False)

    return agent


def _enrich_and_add_to_graph(graph, docs: list[dict], keyword: str):
    """Add scraped documents to the in-memory knowledge graph.
    Only works with NetworkXNormativaGraph (builder methods not available on Neo4j)."""
    from graphrag.graph_builder import NetworkXNormativaGraph

    if not isinstance(graph, NetworkXNormativaGraph):
        logger.warning("Skipping graph enrichment: not a NetworkX backend")
        return

    from graphrag.graph_models import NodeType, EdgeType
    from graphrag.parser import extract_metadata_from_resuelve

    for doc in docs:
        codigo = doc.get("codigo", "")
        if not codigo:
            continue
        nid = f"DOC:{codigo}"
        if graph.graph.has_node(nid):
            continue

        meta = extract_metadata_from_resuelve(doc.get("resuelve", ""))
        doc["tipo_documento"] = meta.get("tipo_documento", "")
        doc["autoridad"] = meta.get("autoridad", "")
        doc["asunto"] = keyword.upper()

        nid = graph.add_document(
            codigo=codigo,
            numero=doc.get("numero", ""),
            fecha=doc.get("fecha", ""),
            resuelve=doc.get("resuelve", ""),
            tipo_doc=doc["tipo_documento"],
            autoridad=doc["autoridad"],
            asunto=doc["asunto"],
            normas_rel=doc.get("normas_relacionadas", ""),
            vigencia=doc.get("vigencia", ""),
        )

        if doc["autoridad"]:
            aut_id = graph.add_autoridad(doc["autoridad"])
            graph.add_emite(aut_id, nid)

        logger.info("Added scraped document %s to graph (%s)", codigo, doc.get("numero", ""))
