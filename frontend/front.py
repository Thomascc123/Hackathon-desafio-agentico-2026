import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Copiloto Administrativo UdeA",
    page_icon="icon.png",
    layout="wide"
)

# =========================
# HEADER PRINCIPAL
# =========================
st.title("Copiloto Administrativo UdeA")
st.caption(
    "Asistente para consulta de normativa, reglamento estudiantil y procesos académicos."
)

st.divider()

# =========================
# SIDEBAR
# =========================
with st.sidebar:

    st.image("logo.png", use_container_width=True)

    st.header("Estado del sistema")
    st.success("Base normativa cargada")
    st.info("Motor RAG activo")

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

# =========================
# CONSULTAS RÁPIDAS
# =========================
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

# =========================
# HISTORIAL CHAT
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =========================
# INPUT
# =========================
question = st.chat_input("Escriba su consulta")

# =========================
# PROCESAR CONSULTA
# =========================
if "quick" in st.session_state:
    question = st.session_state.quick
    del st.session_state.quick

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Consultando normativa institucional..."):
            # =========================
            # BACKEND (REEMPLAZAR)
            # =========================
            answer = (
                "Según la normativa institucional, "
                "los procesos académicos deben realizarse "
                "dentro de los plazos establecidos por la institución."
            )

            st.markdown(answer)

            with st.expander("Fuentes consultadas"):
                st.markdown("- Reglamento Estudiantil")
                st.markdown("- Documento normativo correspondiente")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
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
