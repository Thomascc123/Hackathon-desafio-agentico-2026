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


def _create_pydantic_ai_agent(deps: AgentDeps, register_tools: bool = True):
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

    use_structured = cfg.provider != "ollama"

    if register_tools:
        system_prompt = (
            "Eres un asistente experto en el reglamento estudiantil de la "
            "Universidad de Antioquia (UdeA).\n\n"
            "REGLAS:\n"
            "1. Usa los tools para obtener información y responde SIEMPRE en "
            "español con texto natural (párrafos, sin JSON).\n"
            "2. Describe la información con tus palabras. Ejemplo:\n"
            "   \"Según el Artículo 63 del reglamento, ningún estudiante "
            "podrá matricularse simultáneamente en más de un programa.\"\n"
            "3. Al final de tu respuesta añade una línea con las fuentes: "
            "Fuentes: Art. XX, AS YYYY/ZZ\n"
            "4. Si no encuentras información en los tools, admite que no lo "
            "sabes. NO inventes artículos.\n"
            "5. Guía de tools:\n"
            "   - Tema general → search_articles\n"
            "   - Artículo específico → get_article_text\n"
            "   - Modificaciones/historia → get_article_history\n"
            "   - Documentos → get_document_timeline o get_document_details\n"
            "   - Si lo anterior no alcanza → search_normativa_website\n"
            "   - Descargar PDF → download_document_from_website\n"
            "6. Antes de scrapear usa check_website_available."
        )
    else:
        system_prompt = (
            "Eres un asistente experto en el reglamento estudiantil de la "
            "Universidad de Antioquia (UdeA)."
        )

    agent = Agent(
        model,
        deps_type=AgentDeps,
        output_type=AgentResponse if use_structured else str,
        retries=3 if use_structured else 1,
        system_prompt=system_prompt,
    )

    if register_tools:
        agent = register_graph_tools(agent)
        agent = register_vector_tools(agent)
        agent = register_scrape_tools(agent)

    return agent


def _create_fallback_agent(deps: AgentDeps):
    from graphrag.query_agent import GraphRAGAgent

    return GraphRAGAgent(deps.graph)


def create_agent(deps: AgentDeps):
    """Create an agent, trying LLM first, falling back to regex-based agent.

    For Ollama models, returns a tuple (llm_agent, fallback_agent) for
    two-step query execution (local models often don't support native
    function calling via the OpenAI-compatible API).

    For cloud providers (OpenAI/Anthropic), returns a single PydanticAI Agent
    with tools registered.

    For all others, returns the regex-based fallback agent.

    Returns (agent, mode) where mode is 'llm' or 'fallback'.
    """
    cfg = settings.model

    if cfg.provider == "ollama":
        if not _check_ollama(cfg.base_url, cfg.name):
            logger.warning("Ollama no responde o modelo no disponible, usando respaldo (regex)")
            return _create_fallback_agent(deps), "fallback"

        # Dos pasos: LLM sin tools para clasificar/formatear, fallback para ejecutar
        llm_agent = _create_pydantic_ai_agent(deps, register_tools=False)
        fallback_agent = _create_fallback_agent(deps)
        return (llm_agent, fallback_agent), "llm"

    elif cfg.provider in ("openai", "anthropic"):
        if not cfg.api_key:
            logger.warning(f"No hay API key para {cfg.provider}, usando respaldo")
            return _create_fallback_agent(deps), "fallback"
        agent = _create_pydantic_ai_agent(deps, register_tools=True)
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


def _format_data_for_llm(data: Any, intent: str, params: dict | None = None) -> str:
    """Format tool results into a concise text preview for the LLM."""
    if not data:
        return "No se encontraron resultados."

    lines = []
    if isinstance(data, list):
        for i, item in enumerate(data[:8]):
            # Convert Pydantic models to dicts for uniform access
            if hasattr(item, 'model_dump'):
                item = item.model_dump()
            if isinstance(item, dict):
                if intent == "article_text":
                    num = item.get("numero", "?")
                    texto = item.get("texto", "") or item.get("texto_completo", "")
                    lines.append(f"--- Artículo {num} ---")
                    lines.append(texto[:500])
                    doc = item.get("documento", "") or item.get("documento_asunto", "")
                    if doc:
                        lines.append(f"Documento: {doc}")
                    mods = item.get("modificaciones", 0)
                    if mods:
                        lines.append(f"Modificado en {mods} ocasiones")
                elif intent == "keyword_search":
                    art = item.get("articulo") or item.get("numero", "?")
                    doc_asunto = item.get("documento_asunto", "")
                    doc_info = f" ({doc_asunto})" if doc_asunto else ""
                    lines.append(f"--- Art. {art}{doc_info} ---")
                    texto = item.get("texto_completo") or item.get("texto", "")
                    lines.append(texto[:400] if texto else "(sin texto)")
                elif intent in ("evolution", "modified_by"):
                    for k, v in item.items():
                        if v:
                            lines.append(f"{k}: {v}")
                elif intent == "document_search":
                    lines.append(f"Documento #{item.get('numero', '?')} - {item.get('fecha', '')}")
                    lines.append(item.get("resuelve", "")[:200])
                elif intent == "concept_query":
                    lines.append(str(item))
                else:
                    lines.append(str(item))
            else:
                lines.append(str(item))
            lines.append("")
    else:
        lines.append(str(data))

    return "\n".join(lines).strip()


def _run_ollama_query(llm_agent, fallback_agent, question: str,
                      history: list[dict] | None, deps: AgentDeps | None):
    """Two-step query for Ollama models without native function calling.

    Local models (Qwen, etc.) via Ollama's OpenAI-compatible API do not
    return tool_calls in the response — they output JSON tool call text instead.

    This approach:
    1. Uses the LLM (no tools) to classify intent and extract parameters
    2. Executes the appropriate tool programmatically via the fallback agent
    3. Uses the LLM (no tools) to format a natural language response
    """
    # ── Step 1: Classify intent ──────────────────────────────────
    intent_prompt = (
        "Clasifica la siguiente pregunta sobre el reglamento estudiantil "
        "de la Universidad de Antioquia en UNO de estos tipos:\n\n"
        "- article_text: Pregunta por el texto de un ARTÍCULO específico\n"
        "  Ej: 'qué dice el art 63', 'artículo 63', 'y el artículo 50', 'muestra el artículo 10'\n"
        "- evolution: Pregunta sobre la HISTORIA o EVOLUCIÓN de un artículo\n"
        "  Ej: 'historia del artículo 130', 'cómo ha cambiado el art 50'\n"
        "- modified_by: Pregunta sobre qué ACUERDOS modifican un artículo\n"
        "  Ej: 'qué acuerdos modifican el art 63'\n"
        "- keyword_search: Búsqueda por PALABRA CLAVE o tema general\n"
        "  Ej: 'requisitos para matrícula', 'artículos sobre créditos'\n"
        "- concept_query: Pregunta sobre un CONCEPTO\n"
        "  Ej: 'qué son los créditos académicos', 'define matrícula de honor'\n"
        "- document_search: Búsqueda de un DOCUMENTO específico\n"
        "  Ej: 'acuerdo superior 458 de 2019'\n"
        "- unknown: No se puede clasificar\n\n"
        "Pregunta: " + question + "\n\n"
        "NOTA: Si la pregunta menciona 'artículo' o 'articulo' seguido de un número, "
        "la intención SIEMPRE es article_text.\n\n"
        "Responde SOLO con un JSON, sin texto adicional:\n"
        '{"intent": "article_text", "params": {"art_num": "63"}}\n'
        '{"intent": "keyword_search", "params": {"keyword": "matricula"}}\n'
        '{"intent": "unknown", "params": {}}'
    )

    try:
        r = llm_agent.run_sync(intent_prompt, deps=deps)
        raw = r.output.strip()
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        intent_data = json.loads(m.group(0)) if m else json.loads(raw)
    except Exception as e:
        logger.warning("Error clasificando intent con LLM: %s. Usando regex.", e)
        intent, params = fallback_agent.classify_intent(question)
        if intent == "unknown":
            intent = "keyword_search"
            params = {"keyword": question}
        intent_data = {"intent": intent, "params": params}

    intent = intent_data.get("intent", "unknown")
    params = intent_data.get("params", {})

    # If LLM couldn't classify, try regex as fallback
    if intent == "unknown":
        rx_intent, rx_params = fallback_agent.classify_intent(question)
        # Also catch simple "articulo N" patterns the LLM might miss
        if rx_intent == "unknown":
            m = re.search(r'art[ií]culo\s+(\d+)', question, re.IGNORECASE)
            if m:
                rx_intent = "article_text"
                rx_params = {"art_num": m.group(1)}
        if rx_intent != "unknown":
            intent = rx_intent
            params = rx_params
            logger.info("Regex corrigió intent de '%s' a '%s'", intent_data['intent'], intent)

    # ── Step 2: Execute tool programmatically ────────────────────
    handler = fallback_agent._get_handler(intent)
    if handler is None:
        result_data = fallback_agent._handle_keyword_search({"keyword": question})
        intent = "keyword_search"
    else:
        try:
            result_data = handler(params)
        except Exception as e:
            logger.warning("Error ejecutando handler '%s': %s. Usando keyword fallback.", intent, e)
            result_data = fallback_agent._handle_keyword_search({"keyword": question})
            intent = "keyword_search"

    if not result_data:
        # Try with just the key noun phrase if available
        if intent == "article_text":
            result_data = fallback_agent._handle_keyword_search(
                {"keyword": f"artículo {params.get('art_num', '')}"}
            )
        if not result_data:
            result_data = fallback_agent._handle_keyword_search({"keyword": question})
            intent = "keyword_search"

    # ── Step 3: Format response with LLM ─────────────────────────
    data_preview = _format_data_for_llm(result_data, intent)

    formatter_prompt = (
        "Eres un asistente experto en el reglamento estudiantil de la "
        "Universidad de Antioquia (UdeA).\n\n"
        "Instrucciones:\n"
        "- Responde en español con texto natural (párrafos descriptivos).\n"
        "- NO uses JSON, listas numeradas ni viñetas.\n"
        "- Usa los DATOS DE ABAJO como fuente de información.\n"
        "- Responde SOLO la pregunta actual, ignorando el historial.\n"
        "- Al final añade: Fuentes: Art. XX, AS YYYY/ZZ\n"
        "- Si no hay datos, di que no encontraste la información."
    )

    format_prompt = (
        "Pregunta: " + question + "\n\n"
        "Datos:\n" + data_preview + "\n\n"
        "Respuesta:"
    )

    try:
        model_history = None
        r = llm_agent.run_sync(
            formatter_prompt + "\n\n" + format_prompt,
            message_history=model_history,
            deps=deps,
        )
        answer = r.output.strip()
    except Exception as e:
        logger.warning("Error formateando respuesta con LLM: %s. Usando fallback.", e)
        answer = fallback_agent._format_response(intent, params, result_data)

    sources = _extract_sources_from_text(answer)
    if not sources and isinstance(result_data, list):
        for r_item in result_data[:5]:
            item = r_item.model_dump() if hasattr(r_item, 'model_dump') else r_item
            if isinstance(item, dict):
                for key in ("numero", "articulo"):
                    val = item.get(key)
                    if val:
                        ref = f"Art. {val}"
                        if ref not in sources:
                            sources.append(ref)

    return answer, sources, "llm"


def run_query(agent: Any, question: str, history: list[dict] | None = None,
              deps: AgentDeps | None = None) -> tuple[str, list[str], str]:
    """Execute a query against the agent (non-streaming).

    Returns (answer_text, sources_list, mode).
    mode is 'llm' for PydanticAI, 'fallback' for regex agent.
    """
    from pydantic_ai import Agent as PydanticAgent

    if isinstance(agent, tuple) and len(agent) == 2:
        return _run_ollama_query(agent[0], agent[1], question, history, deps)

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


def run_query_stream(agent: Any, question: str, history: list[dict] | None = None,
                     deps: AgentDeps | None = None) -> dict:
    """Execute a query with streaming support.

    Returns a dict:
      {"generator": generator_yielding_tokens, "sources": [], "mode": ""}

    The 'sources' and 'mode' keys are populated after the generator is fully consumed.
    Usage in Streamlit:
        result = run_query_stream(agent, question, history, deps)
        st.write_stream(result["generator"])
        sources = result["sources"]
        mode = result["mode"]
    """
    import queue
    import threading
    import asyncio
    from pydantic_ai import Agent as PydanticAgent

    result_box: dict = {"sources": [], "mode": "fallback"}

    # ── Helper: async generator for PydanticAI streaming ──────────
    async def _pydantic_stream():
        model_history = _convert_history(history or [])
        async with agent.run_stream(question, message_history=model_history, deps=deps) as sr:
            full_text = ""
            async for text in sr.stream(debounce_by=0.01):
                full_text += text
                yield text
            output = sr.data
            if isinstance(output, AgentResponse):
                result_box["sources"] = output.sources
            else:
                result_box["sources"] = _extract_sources_from_text(full_text)
            result_box["mode"] = "llm"

    # ── Helper: bridge async generator → sync generator ──────────
    def _bridge(async_gen_coro):
        q: queue.Queue = queue.Queue(maxsize=10)
        exc_info: list[Exception] = []

        async def _producer():
            try:
                async for item in await async_gen_coro:
                    q.put(item)
                q.put(None)
            except Exception as e:
                exc_info.append(e)
                q.put(None)

        def _run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(_producer())
            finally:
                loop.close()

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        while True:
            item = q.get()
            if item is None:
                break
            yield item
            if exc_info:
                break

        thread.join(timeout=5)
        if exc_info:
            logger.warning("Streaming error: %s", exc_info[0])

    # ── Ollama two-step: run sync, yield full answer ──────────────
    if isinstance(agent, tuple) and len(agent) == 2:
        answer, sources, mode = _run_ollama_query(agent[0], agent[1], question, history, deps)
        result_box["sources"] = sources
        result_box["mode"] = mode

        def _ollama_gen():
            yield answer
        result_box["generator"] = _ollama_gen()
        return result_box

    # ── Cloud PydanticAI agent: true streaming ────────────────────
    if isinstance(agent, PydanticAgent):
        model_history = _convert_history(history or [])
        try:
            gen = _bridge(_pydantic_stream())
            result_box["generator"] = gen
            return result_box
        except Exception as e:
            logger.warning("No se pudo iniciar streaming: %s", e)
            # Fallback to sync
            answer, sources, mode = run_query(agent, question, history, deps)
            result_box["sources"] = sources
            result_box["mode"] = mode
            def _fallback_gen():
                yield answer
            result_box["generator"] = _fallback_gen()
            return result_box

    # ── Fallback regex agent ──────────────────────────────────────
    answer = agent.answer(question)
    result_box["sources"] = _extract_sources_from_text(answer)
    result_box["mode"] = "fallback"
    def _fb_gen():
        yield answer
    result_box["generator"] = _fb_gen()
    return result_box


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
