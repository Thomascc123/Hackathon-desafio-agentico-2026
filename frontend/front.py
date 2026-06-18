import sys
from pathlib import Path

FRONTEND_DIR = Path(__file__).parent
sys.path.insert(0, str(FRONTEND_DIR.parent))

import json
import logging
import re
import uuid
import streamlit as st
import streamlit.components.v1 as components

from agent.models import AgentDeps
from agent.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner="Indexando fuentes del grafo...")
def _build_source_index(_graph):
    """Build lookup indexes for enriching sources with document metadata and local PDF paths.

    Returns (art_by_num, doc_by_codigo, ref_to_node_id) where:
      art_by_num: art_num_str -> dict with doc_codigo
      doc_by_codigo: codigo_str -> dict with numero, anio, asunto, pdf_path
      ref_to_node_id: source_ref_str -> graph node id for highlighting
    """
    from graphrag.graph_models import NodeType

    art_by_num = {}
    doc_by_codigo = {}
    ref_to_node_id = {}
    pdf_dir = Path(settings.directories.pdf_dir)

    for nid, data in _graph.graph.nodes(data=True):
        ntype = data.get("type")
        if ntype == NodeType.ARTICULO:
            art_num = data.get("numero", "")
            if art_num:
                art_by_num[art_num] = {
                    "doc_codigo": data.get("documento_codigo", ""),
                }
                ref_to_node_id[f"Art. {art_num}"] = nid
        elif ntype == NodeType.DOCUMENTO:
            codigo = data.get("codigo", "")
            doc_num = data.get("numero", "")
            anio = data.get("anio", "")
            asunto = data.get("asunto", "")
            tipo = data.get("tipo_documento", "") or "AS"

            pdf_path = None
            candidates = [
                pdf_dir / f"{codigo}.pdf",
                pdf_dir / f"scraped_{codigo}.pdf",
                pdf_dir / "02_reglamento_pregrado_actualizado_2025.pdf",
            ]
            for p in candidates:
                if p.exists():
                    pdf_path = str(p)
                    break

            if doc_num:
                numero_clean = doc_num.lstrip("0")
                label = f"{tipo} {numero_clean}" + (f"/{anio}" if anio else "")
                doc_by_codigo[codigo] = {
                    "numero": doc_num,
                    "numero_clean": numero_clean,
                    "anio": anio,
                    "asunto": asunto,
                    "label": label,
                    "pdf_path": pdf_path,
                    "tipo": tipo,
                }
                ref_to_node_id[label] = nid

    return art_by_num, doc_by_codigo, ref_to_node_id


def _enrich_sources(sources: list[str], art_by_num: dict, doc_by_codigo: dict) -> list[dict]:
    """Convert source strings to enriched dicts with document metadata and PDF paths."""
    enriched = []
    seen_refs = set()

    for s in sources:
        if s in seen_refs:
            continue
        seen_refs.add(s)

        item = {
            "ref": s,
            "doc_label": "",
            "doc_codigo": "",
            "pdf_path": None,
        }

        # Match "Art. 63" or "Artículo 63"
        m = re.match(r'(?:Artículo|Art\.?)\s*(\d+)', s, re.IGNORECASE)
        if m:
            art_num = m.group(1)
            art_info = art_by_num.get(art_num)
            if art_info:
                doc_codigo = art_info["doc_codigo"]
                if doc_codigo:
                    doc_info = doc_by_codigo.get(doc_codigo)
                    if doc_info:
                        item["doc_label"] = doc_info["label"]
                        item["doc_codigo"] = doc_codigo
                        item["pdf_path"] = doc_info["pdf_path"]
            enriched.append(item)
            continue

        # Match "AS 458/2019" or "Acuerdo Superior 458/2019"
        m2 = re.match(r'(?:AS|Acuerdo Superior)\s+(\d+)(?:/(\d{4}))?', s, re.IGNORECASE)
        if m2:
            num = m2.group(1)
            anio = m2.group(2)
            # Find by matching numero_clean + optionally anio
            for codigo, doc_info in doc_by_codigo.items():
                if doc_info["numero_clean"] == num.lstrip("0"):
                    if not anio or doc_info["anio"] == anio:
                        item["doc_label"] = doc_info["label"]
                        item["doc_codigo"] = codigo
                        item["pdf_path"] = doc_info["pdf_path"]
                        break
            enriched.append(item)
            continue

        enriched.append(item)

    return enriched

st.set_page_config(
    page_title="Copiloto Administrativo UdeA",
    page_icon=str(FRONTEND_DIR / "icon.png"),
    layout="wide"
)

if "page" not in st.session_state:
    st.session_state.page = "chat"

# ── Conversations ─────────────────────────────────────────────────

if "conversations" not in st.session_state:
    st.session_state.conversations = {}
if "conversation_order" not in st.session_state:
    st.session_state.conversation_order = []
if "active_conversation" not in st.session_state:
    first_id = str(uuid.uuid4())[:8]
    st.session_state.conversations[first_id] = {
        "title": "Conversación 1",
        "messages": [],
    }
    st.session_state.conversation_order = [first_id]
    st.session_state.active_conversation = first_id


def _active_conv():
    return st.session_state.conversations[st.session_state.active_conversation]

# ── Cached initialization ──────────────────────────────────────────

@st.cache_resource(show_spinner="Cargando grafo de conocimiento...")
def load_graph():
    from graphrag.factory import create_graph

    try:
        g = create_graph()
    except Exception as e:
        st.error(f"Error cargando grafo: {e}")
        st.stop()

    summary = g.summary()
    logger.info("Graph loaded: %d nodes, %d edges", summary["nodes"], summary["edges"])
    return g


@st.cache_resource(show_spinner="Inicializando base vectorial...")
def init_chromadb():
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    from agent.config import settings

    persist_dir = Path(settings.chroma.persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(persist_dir))

    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name=settings.embedding.model,
        device=settings.embedding.device,
    )

    collection = client.get_or_create_collection(
        name="articulos",
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    return client, collection


@st.cache_resource(show_spinner="Indexando artículos...")
def index_articles(_graph, _collection):
    from graphrag.graph_models import NodeType

    count = _collection.count()
    if count > 0:
        logger.info("ChromaDB already has %d documents", count)
        return count

    logger.info("Indexing articles into ChromaDB...")
    articles = []
    metadatas = []
    ids = []

    for nid, data in _graph.graph.nodes(data=True):
        if data.get("type") == NodeType.ARTICULO:
            texto = data.get("texto_completo", "") or data.get("texto", "")
            if not texto:
                continue
            art_num = data.get("numero", "?")
            doc_codigo = data.get("documento_codigo", "")
            articles.append(texto[:5000])
            metadatas.append({
                "numero": art_num,
                "documento": doc_codigo,
                "modificaciones": data.get("num_modificaciones", 0),
            })
            ids.append(f"ART_{doc_codigo}_{art_num}")

    if articles:
        _collection.add(documents=articles, metadatas=metadatas, ids=ids)
        logger.info("Indexed %d articles", len(articles))
    return len(articles)


# ── Initialize ─────────────────────────────────────────────────────

graph = load_graph()
_, chroma_collection = init_chromadb()
index_articles(graph, chroma_collection)

from agent.security.audit import AuditLogger
from agent.security.rate_limiter import RateLimiter

audit = AuditLogger(settings.security.audit_db)
rate_limiter = RateLimiter(max_per_minute=settings.security.rate_limit)

deps = AgentDeps(
    graph=graph,
    chroma_collection=chroma_collection,
    pdf_dir=settings.directories.pdf_dir,
    audit=audit,
)

from agent.agent_setup import create_agent, run_query, run_query_stream

agent, agent_mode = create_agent(deps)

# Build source index for enriching displayed sources with document metadata
_source_art_by_num, _source_doc_by_codigo, _source_ref_to_node = _build_source_index(graph)

# ── Navigation ─────────────────────────────────────────────────────

# ── Shared sidebar ──────────────────────────────────────────────────

with st.sidebar:
    st.image(str(FRONTEND_DIR / "logo.png"), use_container_width=True)

    # ── Graph Explorer toggle ───────────────────────────────────────
    graph_label = "🔙 Volver al Chat" if st.session_state.page == "graph" else "🔗 Explorador de Grafo"
    if st.button(graph_label, use_container_width=True, type="primary" if st.session_state.page != "graph" else "secondary"):
        st.session_state.page = "graph" if st.session_state.page == "chat" else "chat"
        st.rerun()

    st.divider()

    # ── Conversation manager ────────────────────────────────────────
    st.subheader("Conversaciones")

    col_new, _ = st.columns([1, 2])
    with col_new:
        if st.button("+ Nueva", use_container_width=True, type="primary"):
            cid = str(uuid.uuid4())[:8]
            num = len(st.session_state.conversations) + 1
            st.session_state.conversations[cid] = {
                "title": f"Conversación {num}",
                "messages": [],
            }
            st.session_state.conversation_order.append(cid)
            st.session_state.active_conversation = cid
            st.rerun()

    for cid in st.session_state.conversation_order:
        conv = st.session_state.conversations[cid]
        is_active = cid == st.session_state.active_conversation
        col1, col2 = st.columns([4, 1])
        with col1:
            label = conv["title"][:25]
            msg_count = len(conv["messages"])
            btn_label = f"{'▸ ' if is_active else ''}{label} ({msg_count})"
            if st.button(btn_label, use_container_width=True,
                         type="primary" if is_active else "secondary",
                         key=f"conv_{cid}"):
                st.session_state.active_conversation = cid
                st.rerun()
        with col2:
            if len(st.session_state.conversations) > 1:
                if st.button("✕", key=f"del_{cid}",
                             help="Eliminar conversación"):
                    del st.session_state.conversations[cid]
                    st.session_state.conversation_order.remove(cid)
                    if st.session_state.active_conversation == cid:
                        st.session_state.active_conversation = \
                            st.session_state.conversation_order[0]
                    st.rerun()

    st.divider()

    # ── System status ───────────────────────────────────────────────
    st.subheader("Estado del sistema")
    st.success("Base normativa cargada")

    if agent_mode == "llm":
        st.info("Motor RAG activo (IA)")
        model_cfg = settings.model
        st.caption(f"Modelo: {model_cfg.provider}:{model_cfg.name}")
    else:
        st.warning("Modo de respaldo activo (búsqueda por palabras)")
        st.caption("Conecta Ollama o configura una API key para activar IA")

    st.divider()

# ── Main content ────────────────────────────────────────────────────

if st.session_state.page == "chat":
    st.title("Copiloto Administrativo UdeA")
    st.caption(
        "Asistente para consulta de normativa, reglamento estudiantil y procesos académicos."
    )

    st.divider()

    # ── Quick questions ────────────────────────────────────────────────

    if "expanded_category" not in st.session_state:
        st.session_state.expanded_category = None

    preguntas = {
        "matricula": [
            "¿Cómo funciona el proceso de matrícula?",
            "¿Cuáles son los requisitos para matricularse?",
            "¿Qué documentos necesita para matricularse?",
            "¿Cuándo es el período de matrícula?",
        ],
        "cancelaciones": [
            "¿Cómo cancelar una asignatura?",
            "¿Hasta cuándo se puede cancelar?",
            "¿Cancelar materia afecta el promedio?",
            "¿Cómo cancelar toda la matrícula?",
        ],
        "grados": [
            "¿Cuáles son los requisitos para grado?",
            "¿Cómo solicitar el diploma?",
            "¿Cuándo es la ceremonia de grado?",
            "¿Cómo saber si cumple los requisitos de grado?",
        ],
    }

    st.subheader("Consultas frecuentes")

    col1, col2, col3 = st.columns(3)
    etiquetas = [("Matrículas", "matricula"), ("Cancelaciones", "cancelaciones"), ("Grados", "grados")]

    for col, (etiqueta, key) in zip([col1, col2, col3], etiquetas):
        with col:
            if st.button(etiqueta, use_container_width=True):
                st.session_state.expanded_category = (
                    key if st.session_state.expanded_category != key else None
                )
            if st.session_state.expanded_category == key:
                for p in preguntas[key]:
                    if st.button("  " + p):
                        st.session_state.quick = p
                        st.session_state.expanded_category = None
                        st.rerun()

    # ── Chat history ───────────────────────────────────────────────────

    conv = _active_conv()
    messages = conv["messages"]

    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg.get("content", ""))
            if msg["role"] == "assistant":
                sources = msg.get("sources", [])
                enriched = _enrich_sources(sources, _source_art_by_num, _source_doc_by_codigo)
                if enriched:
                    st.session_state.highlighted_refs = [e["ref"] for e in enriched]
                with st.expander("Fuentes consultadas", expanded=bool(enriched)):
                    if enriched:
                        for e in enriched:
                            parts = [f"**{e['ref']}**"]
                            if e["doc_codigo"]:
                                parts.append(f"`{e['doc_codigo']}`")
                            if e["doc_label"]:
                                parts.append(f"({e['doc_label']})")
                            if e["pdf_path"]:
                                parts.append(f"📄 `{e['pdf_path']}`")
                            st.markdown("  " + " ".join(parts))
                    else:
                        st.markdown("_No se encontraron fuentes específicas en el grafo_")
                if msg.get("mode") == "fallback":
                    st.caption("⚠️ Modo de respaldo (búsqueda por palabras)")

    # ── Input ──────────────────────────────────────────────────────────

    question = st.chat_input("Escriba su consulta")

    if "quick" in st.session_state:
        question = st.session_state.quick
        del st.session_state.quick

    if question:
        if not rate_limiter.check():
            st.error("Demasiadas consultas. Espera un momento.")
            st.stop()

        # Auto-title from first user message
        if len(messages) == 0:
            conv["title"] = question[:30] + ("…" if len(question) > 30 else "")

        messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Consultando normativa institucional..."):
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in messages[:-1]
                ]

                should_stream = agent_mode == "llm" and not isinstance(agent, tuple)

                if should_stream:
                    result = run_query_stream(agent, question, history=history, deps=deps)
                    answer_text = st.write_stream(result["generator"])
                    sources = result["sources"]
                    mode = result["mode"]
                else:
                    answer_text, sources, mode = run_query(
                        agent, question, history=history, deps=deps
                    )
                    st.markdown(answer_text)

                enriched = _enrich_sources(sources, _source_art_by_num, _source_doc_by_codigo)
                if enriched:
                    st.session_state.highlighted_refs = [e["ref"] for e in enriched]
                with st.expander("Fuentes consultadas", expanded=bool(enriched)):
                    if enriched:
                        for e in enriched:
                            parts = [f"**{e['ref']}**"]
                            if e["doc_codigo"]:
                                parts.append(f"`{e['doc_codigo']}`")
                            if e["doc_label"]:
                                parts.append(f"({e['doc_label']})")
                            if e["pdf_path"]:
                                parts.append(f"📄 `{e['pdf_path']}`")
                            st.markdown("  " + " ".join(parts))
                    else:
                        st.markdown("_No se encontraron fuentes específicas en el grafo_")

                if mode == "fallback":
                    st.caption("⚠️ Modo de respaldo (búsqueda por palabras)")

        messages.append({
            "role": "assistant",
            "content": answer_text,
            "sources": sources,
            "mode": mode,
        })

        audit.log(
            query=question,
            intent="",
            tools_used=sources if mode == "llm" else None,
            response_summary=answer_text[:100],
            agent_type=mode,
        )

        components.html(
            """
            <script>
                var el = window.parent.document.querySelector(
                    '[data-testid="stAppViewContainer"] .main'
                );
                if (el) el.scrollTop = el.scrollHeight;
            </script>
            """,
            height=0,
        )

else:  # st.session_state.page == "graph"
    from frontend.graph_view import render_graph_page
    render_graph_page(graph, ref_to_node_id=_source_ref_to_node)
