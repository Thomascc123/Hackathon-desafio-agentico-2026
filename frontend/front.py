import sys
from pathlib import Path

FRONTEND_DIR = Path(__file__).parent
sys.path.insert(0, str(FRONTEND_DIR.parent))

import json
import logging
import streamlit as st
import streamlit.components.v1 as components

from agent.models import AgentDeps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Copiloto Administrativo UdeA",
    page_icon=str(FRONTEND_DIR / "icon.png"),
    layout="wide"
)

# ── Tab navigation ─────────────────────────────────────────────────

PAGES = {
    "💬 Chat": "chat",
    "🔗 Explorador de Grafo": "graph",
}

if "page" not in st.session_state:
    st.session_state.page = "chat"

cols = st.columns(len(PAGES))
for col, (label, key) in zip(cols, PAGES.items()):
    with col:
        if st.button(label, use_container_width=True,
                     type="primary" if st.session_state.page == key else "secondary"):
            st.session_state.page = key
            st.rerun()

# ── Cached initialization ──────────────────────────────────────────

@st.cache_resource(show_spinner="Cargando grafo de conocimiento...")
def load_graph():
    from graphrag.graph_builder import NormativaGraph
    from graphrag.graph_models import NodeType, EdgeType

    g = NormativaGraph()
    path = Path(__file__).parent.parent / "graphrag_output" / "knowledge_graph.json"

    if not path.exists():
        st.error(f"Grafo no encontrado en {path}. Ejecuta el pipeline primero.")
        st.stop()

    with open(path) as f:
        data = json.load(f)

    for node in data.get("nodes", []):
        props = {"type": NodeType(node["type"]), "label": node["label"], **node["properties"]}
        g.graph.add_node(node["id"], **props)

    for edge in data.get("edges", []):
        props = {"type": EdgeType(edge["type"]), **edge["properties"]}
        g.graph.add_edge(edge["source"], edge["target"], key=edge["type"], **props)

    logger.info("Graph loaded: %d nodes, %d edges", g.graph.number_of_nodes(), g.graph.number_of_edges())
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
from agent.config import settings

audit = AuditLogger(settings.security.audit_db)
rate_limiter = RateLimiter(max_per_minute=settings.security.rate_limit)

deps = AgentDeps(
    graph=graph,
    chroma_collection=chroma_collection,
    pdf_dir=settings.directories.pdf_dir,
    audit=audit,
)

from agent.agent_setup import create_agent, run_query

agent, agent_mode = create_agent(deps)

# ── Navigation ─────────────────────────────────────────────────────

if st.session_state.page == "chat":
    # ── UI ─────────────────────────────────────────────────────────
    st.title("Copiloto Administrativo UdeA")
    st.caption(
        "Asistente para consulta de normativa, reglamento estudiantil y procesos académicos."
    )

    st.divider()

    with st.sidebar:
        st.image(str(FRONTEND_DIR / "logo.png"), use_container_width=True)

        st.header("Estado del sistema")
        st.success("Base normativa cargada")

        if agent_mode == "llm":
            st.info("Motor RAG activo (IA)")
            model_cfg = settings.model
            st.caption(f"Modelo: {model_cfg.provider}:{model_cfg.name}")
        else:
            st.warning("Modo de respaldo activo (búsqueda por palabras)")
            st.caption("Conecta Ollama o configura una API key para activar IA")

        st.divider()
        st.subheader("Fuentes")
        st.markdown("""
        - Reglamento estudiantil
        - Normativas académicas
        - Procesos de matrícula
        """)
        st.divider()
        st.subheader("Información del sistema")
        st.markdown("""
        - Respuestas basadas en documentos oficiales
        - Sistema con recuperación de contexto (RAG)
        - Trazabilidad de fuentes disponible
        """)

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

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg.get("content", ""))
            if msg["role"] == "assistant":
                sources = msg.get("sources", [])
                with st.expander("Fuentes consultadas", expanded=bool(sources)):
                    if sources:
                        for s in sources:
                            st.markdown(f"- {s}")
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

        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Consultando normativa institucional..."):
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                ]
                answer_text, sources, mode = run_query(
                    agent, question, history=history, deps=deps
                )
                st.markdown(answer_text)

                with st.expander("Fuentes consultadas", expanded=bool(sources)):
                    if sources:
                        for s in sources:
                            st.markdown(f"- {s}")
                    else:
                        st.markdown("_No se encontraron fuentes específicas en el grafo_")

                if mode == "fallback":
                    st.caption("⚠️ Modo de respaldo (búsqueda por palabras)")

        st.session_state.messages.append({
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

elif st.session_state.page == "graph":
    st.title("🔗 Explorador del Grafo de Conocimiento")
    st.caption("Visualización interactiva del reglamento estudiantil y sus relaciones.")

    from frontend.graph_view import render_graph_page
    render_graph_page(graph)
