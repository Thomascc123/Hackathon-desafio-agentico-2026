import json
from pydantic_ai import RunContext
from agent.models import AgentDeps, ArticleHistory, ModificationEvent, DocumentSummary
from graphrag.base_graph import BaseNormativaGraph


def register_graph_tools(agent):
    @agent.tool
    async def get_article_text(
        ctx: RunContext[AgentDeps],
        articulo: int,
    ) -> str:
        """Obtiene el texto completo de un artículo del reglamento.
        Úsalo cuando el usuario pregunte qué dice un artículo específico."""
        graph = ctx.deps.graph
        for nid, data in graph.graph.nodes(data=True):
            if data.get("type") == "Articulo" and data.get("numero") == str(articulo):
                texto = data.get("texto", "") or data.get("texto_completo", "")
                mods = data.get("num_modificaciones", 0)
                doc = ""
                for src, _, edata in graph.graph.in_edges(nid, data=True):
                    if edata.get("type") == "CONTIENE":
                        parent = graph.graph.nodes[src]
                        doc = parent.get("label", "")
                        break
                result = {
                    "articulo": str(articulo),
                    "texto": texto[:1000],
                    "documento": doc,
                    "modificaciones": mods,
                }
                return json.dumps(result, ensure_ascii=False)
        return json.dumps({"error": f"No se encontró el artículo {articulo}"})

    @agent.tool
    async def get_article_history(
        ctx: RunContext[AgentDeps],
        articulo: int,
    ) -> str:
        """Obtiene el historial de modificaciones de un artículo a lo largo del tiempo.
        Úsalo cuando el usuario pregunte cómo ha cambiado un artículo o sus modificaciones."""
        graph = ctx.deps.graph
        raw = graph.query_evolution_of_article(str(articulo))
        if not raw:
            return json.dumps({"error": f"No se encontró historial para el artículo {articulo}"})

        mods = []
        texto_actual = ""
        for item in raw:
            if "modificado_por" in item:
                mods.append(ModificationEvent(
                    anio=item.get("anio", ""),
                    modificado_por=item.get("modificado_por", ""),
                    accion=item.get("accion", ""),
                ))
            elif "texto_actual" in item:
                texto_actual = item.get("texto_actual", "")

        history = ArticleHistory(
            articulo=str(articulo),
            texto_actual=texto_actual[:500],
            num_modificaciones=len(mods),
            modificaciones=mods,
        )
        return history.model_dump_json()

    @agent.tool
    async def get_document_timeline(
        ctx: RunContext[AgentDeps],
        asunto: str,
    ) -> str:
        """Obtiene una línea de tiempo de documentos normativos filtrados por asunto.
        Úsalo cuando el usuario pida documentos de pregrado, posgrado, matrícula, etc."""
        graph = ctx.deps.graph
        raw = graph.query_document_timeline(asunto=asunto.upper())
        docs = [DocumentSummary(
            numero=d.get("numero", ""),
            fecha=d.get("fecha", ""),
            resuelve=d.get("resuelve", "")[:200],
            autoridad=d.get("autoridad", ""),
            anio=d.get("anio", ""),
        ) for d in raw[:20]]
        return json.dumps([d.model_dump() for d in docs], ensure_ascii=False)

    @agent.tool
    async def get_document_details(
        ctx: RunContext[AgentDeps],
        doc_numero: str,
        doc_anio: str = "",
    ) -> str:
        """Busca detalles de un documento normativo por su número y año.
        Úsalo cuando el usuario mencione un acuerdo o resolución específica."""
        graph = ctx.deps.graph
        from graphrag.graph_models import NodeType
        results = []
        for nid, data in graph.graph.nodes(data=True):
            if data.get("type") == NodeType.DOCUMENTO:
                if data.get("numero", "").lstrip("0") == doc_numero.lstrip("0"):
                    if not doc_anio or data.get("anio", "") == doc_anio:
                        if not data.get("is_referenced"):
                            results.append({
                                "numero": data.get("numero", ""),
                                "fecha": data.get("fecha", ""),
                                "resuelve": data.get("resuelve", ""),
                                "autoridad": data.get("autoridad", ""),
                                "asunto": data.get("asunto", ""),
                            })
        return json.dumps(results, ensure_ascii=False)

    return agent
