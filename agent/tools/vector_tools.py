import json
from pydantic_ai import RunContext
from agent.models import AgentDeps, ArticleResult


def register_vector_tools(agent):
    @agent.tool
    async def search_articles(
        ctx: RunContext[AgentDeps],
        keyword: str,
        top_k: int = 10,
    ) -> str:
        """Busca artículos del reglamento por similitud semántica usando el texto de búsqueda.
        Úsalo cuando el usuario pregunte sobre un tema, concepto o palabra clave."""
        collection = ctx.deps.chroma_collection
        if collection is None or collection.count() == 0:
            return json.dumps({"error": "El índice vectorial no está disponible"})

        results = collection.query(
            query_texts=[keyword],
            n_results=min(top_k, 20),
        )

        articles = []
        for i, doc in enumerate(results.get("documents", [[]])[0]):
            meta = results.get("metadatas", [[]])[0][i] if results.get("metadatas") else {}
            articles.append(ArticleResult(
                numero=meta.get("numero", "?"),
                texto=doc[:500] if doc else "",
                documento=meta.get("documento", ""),
                score=results.get("distances", [[0]])[0][i] if results.get("distances") else 0,
            ))

        return json.dumps([a.model_dump() for a in articles], ensure_ascii=False)

    @agent.tool
    async def search_documents_by_keyword(
        ctx: RunContext[AgentDeps],
        keyword: str,
    ) -> str:
        """Busca documentos normativos por palabra clave en su texto o resumen.
        Úsalo para encontrar documentos que traten sobre un tema específico."""
        collection = ctx.deps.chroma_collection
        if collection is None or collection.count() == 0:
            return json.dumps({"error": "Índice no disponible"})

        results = collection.query(
            query_texts=[keyword],
            n_results=15,
            where={"type": "documento"},
        )

        docs = []
        for i, doc in enumerate(results.get("documents", [[]])[0]):
            meta = results.get("metadatas", [[]])[0][i] if results.get("metadatas") else {}
            docs.append({
                "titulo": doc[:200] if doc else "",
                "fecha": meta.get("fecha", ""),
                "numero": meta.get("numero", ""),
            })

        return json.dumps(docs, ensure_ascii=False)

    return agent
