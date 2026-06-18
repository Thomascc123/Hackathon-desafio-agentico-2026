import json
import logging
import re
from typing import Any, Optional
import httpx

from agent.config import settings
from agent.models import AgentDeps, AgentResponse
from agent.tools.graph_tools import register_graph_tools
from agent.tools.vector_tools import register_vector_tools
from agent.tools.scrape_tools import register_scrape_tools

logger = logging.getLogger(__name__)


def _check_ollama(base_url: str, model_name: str) -> bool:
    api_base = base_url.rstrip("/v1").rstrip("/")
    try:
        r = httpx.get(f"{api_base}/api/tags", timeout=5.0)
        if r.status_code != 200:
            return False
        models = r.json().get("models", [])
        available = any(
            m.get("name", "").startswith(model_name) or m.get("model", "").startswith(model_name)
            for m in models
        )
        if not available:
            logger.warning("Modelo '%s' no encontrado en Ollama. Modelos disponibles: %s",
                           model_name, [m.get("name") for m in models])
            return False
        return True
    except Exception as e:
        logger.warning("Error conectando a Ollama: %s", e)
        return False


def _create_pydantic_ai_agent(deps: AgentDeps):
    from pydantic_ai import Agent
    from pydantic_ai.models.ollama import OllamaModel
    from pydantic_ai.providers.ollama import OllamaProvider
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

    cfg = settings.model

    if cfg.provider == "ollama":
        model = OllamaModel(
            cfg.name,
            provider=OllamaProvider(base_url=cfg.base_url),
        )
    elif cfg.provider == "openai":
        model = OpenAIChatModel(
            cfg.name,
            provider=OpenAIProvider(api_key=cfg.api_key),
        )
    elif cfg.provider == "anthropic":
        model = AnthropicModel(
            cfg.name,
            provider=AnthropicProvider(api_key=cfg.api_key),
        )
    else:
        raise ValueError(f"Unsupported provider: {cfg.provider}")

    # Los modelos locales (Ollama) no manejan bien structured output.
    # Usamos texto plano y extraemos fuentes con regex.
    use_structured = cfg.provider != "ollama"

    agent = Agent(
        model,
        deps_type=AgentDeps,
        output_type=AgentResponse if use_structured else str,
        retries=3 if use_structured else 1,
        system_prompt=(
            "Eres un asistente experto en el reglamento estudiantil de la Universidad de Antioquia (UdeA).\n\n"
            "REGLAS:\n"
            "1. Usa los tools disponibles para obtener información verificada del grafo de conocimiento y la base vectorial.\n"
            "2. Responde SIEMPRE en español.\n"
            f"{'3. Al final de tu respuesta, lista las fuentes en una línea separada con el formato: Fuentes: Art. XX, AS YYYY/ZZ' if not use_structured else '3. DEBES incluir las fuentes en el campo sources de tu respuesta estructurada.'}\n"
            "4. Menciona explícitamente los artículos y documentos citados en el texto. "
            "Ejemplo: 'Según el Artículo 63 del reglamento, ningún estudiante podrá matricularse simultáneamente en más de un programa.'\n"
            "5. Si no encuentras información en los tools, admite que no lo sabes. NO inventes artículos.\n"
            "6. Proporciona respuestas completas y detalladas.\n"
            "7. Si el usuario pregunta sobre un tema general (ej. matrícula), usa search_articles.\n"
            "8. Para preguntas sobre un artículo específico, usa get_article_text.\n"
            "9. Para preguntas sobre modificaciones o historia de un artículo, usa get_article_history.\n"
            "10. Para preguntas sobre documentos (acuerdos, resoluciones), usa get_document_timeline o get_document_details.\n"
            "11. SI los tools locales no encuentran información, usa search_normativa_website para buscar en el sitio web oficial.\n"
            "12. Si search_normativa_website encuentra un documento relevante, usa download_document_from_website con su código para descargarlo y parsearlo.\n"
            "13. Antes de scrapear, usa check_website_available para verificar conexión. Si no hay conexión, informa al usuario."
        ),
    )

    agent = register_graph_tools(agent)
    agent = register_vector_tools(agent)
    agent = register_scrape_tools(agent)

    return agent


def _create_fallback_agent(deps: AgentDeps):
    from graphrag.query_agent import GraphRAGAgent

    return GraphRAGAgent(deps.graph)


def create_agent(deps: AgentDeps):
    """Create an agent, trying LLM first, falling back to regex-based agent.

    Returns (agent, mode) where mode is 'llm' or 'fallback'.
    Also stores fallback agent inside the llm agent for runtime fallback.
    """
    cfg = settings.model

    if cfg.provider == "ollama":
        if not _check_ollama(cfg.base_url, cfg.name):
            logger.warning("Ollama no responde o modelo no disponible, usando respaldo (regex)")
            return _create_fallback_agent(deps), "fallback"
        agent = _create_pydantic_ai_agent(deps)
        agent._fallback = _create_fallback_agent(deps)
        return agent, "llm"

    elif cfg.provider in ("openai", "anthropic"):
        if not cfg.api_key:
            logger.warning(f"No hay API key para {cfg.provider}, usando respaldo")
            return _create_fallback_agent(deps), "fallback"
        agent = _create_pydantic_ai_agent(deps)
        agent._fallback = _create_fallback_agent(deps)
        return agent, "llm"

    else:
        return _create_fallback_agent(deps), "fallback"


def _convert_history(history: list[dict]) -> list:
    """Convert streamlit message dicts to PydanticAI ModelMessage list."""
    from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart, TextPart

    model_messages = []
    if not history:
        return model_messages

    for msg in history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if not content:
            continue
        if role == "user":
            model_messages.append(
                ModelRequest(parts=[UserPromptPart(content=content)])
            )
        elif role == "assistant":
            model_messages.append(
                ModelResponse(parts=[TextPart(content=content)])
            )

    return model_messages


def run_query(agent: Any, question: str, history: list[dict] | None = None,
              deps: AgentDeps | None = None) -> tuple[str, list[str], str]:
    """Execute a query against the agent.

    Falls back to regex agent if LLM call fails at runtime.

    Returns (answer_text, sources_list, mode).
    mode is 'llm' for PydanticAI, 'fallback' for regex agent.
    """
    from pydantic_ai import Agent as PydanticAgent

    if isinstance(agent, PydanticAgent):
        model_history = _convert_history(history or [])
        try:
            result = agent.run_sync(
                question,
                message_history=model_history,
                deps=deps,
            )
            output = result.output
            if isinstance(output, AgentResponse):
                return output.answer, output.sources, "llm"
            # Plain text output (Ollama mode without structured output)
            answer = str(output)
            sources = _extract_sources_from_text(answer)
            return answer, sources, "llm"
        except Exception as e:
            logger.warning("Error en LLM (%s), usando respaldo regex", e)
            fallback = getattr(agent, "_fallback", None)
            if fallback:
                answer = fallback.answer(question)
                return answer, [], "fallback"
            return f"Error al consultar el LLM: {e}. Intenta de nuevo.", [], "error"
    else:
        answer = agent.answer(question)
        sources = _extract_sources_from_text(answer)
        return answer, sources, "fallback"


def _extract_sources_from_text(text: str) -> list[str]:
    """Extract article and document references from a response text."""
    sources = []
    seen = set()

    arts = re.findall(r'(?:Artículo|Art\.?)\s*(\d+)', text, re.IGNORECASE)
    for a in arts:
        ref = f"Art. {a}"
        if ref not in seen:
            seen.add(ref)
            sources.append(ref)

    docs = re.findall(r'(?:AS|Acuerdo Superior)\s+(\d+(?:[A-Za-z])?)(?:\s*/\s*(\d{4}))?', text, re.IGNORECASE)
    for num, anio in docs:
        ref = f"AS {num}/{anio}" if anio else f"AS {num}"
        if ref not in seen:
            seen.add(ref)
            sources.append(ref)

    return sources
