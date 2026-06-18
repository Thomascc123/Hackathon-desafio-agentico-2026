import streamlit as st

st.set_page_config(
    page_title="Copiloto Administrativo UdeA",
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
st.subheader("Consultas frecuentes")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Matrículas"):
        st.session_state.quick = "¿Cómo funciona el proceso de matrícula?"

with col2:
    if st.button("Cancelaciones"):
        st.session_state.quick = "¿Cómo cancelar una asignatura?"

with col3:
    if st.button("Grados"):
        st.session_state.quick = "¿Cuáles son los requisitos para grado?"

# =========================
# HISTORIAL CHAT
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "quick" in st.session_state:
    st.session_state.messages.append(
        {"role": "user", "content": st.session_state.quick}
    )
    del st.session_state.quick

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =========================
# INPUT
# =========================
question = st.chat_input("Escriba su consulta")

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