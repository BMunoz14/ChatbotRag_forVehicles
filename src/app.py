import streamlit as st
import ollama
from rag import setup_model, ask_question
from vector_db_manager import (
    get_vectorstore,
    append_pdf_to_vectorstore,
    get_temporary_vectorstore
)

from memory_manager import get_memory_for_user
from langgraph.graph import MessagesState, StateGraph, START
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="RAG para PDF", layout="centered")
st.title("🔍 Chat con tu documento PDF")
st.write("📄 Carga un archivo PDF y haz preguntas sobre su contenido")

# --- Inicializar historial ---
if "historial" not in st.session_state:
    st.session_state.historial = []

# --- Selección de modelo ---
def list_models():
    models_running = ollama.list()['models']
    return [model["model"] for model in models_running]

lista = list_models()
if 'model_selection' not in st.session_state:
    st.session_state.model_selection = lista[0] if lista else None

# --- Sidebar ---
with st.sidebar:
    st.title('🤖 Opciones')

    st.session_state.model_selection = st.selectbox(
        'Selecciona el modelo:',
        options=lista,
        index=lista.index(st.session_state.model_selection) if st.session_state.model_selection in lista else 0
    )

    uploaded_pdf = st.file_uploader("📄 Carga un PDF", type=["pdf"])
    if uploaded_pdf:
        st.session_state.pdf_path = f"./temp_{uploaded_pdf.name}"
        with open(st.session_state.pdf_path, "wb") as f:
            f.write(uploaded_pdf.read())

    st.session_state.temperature = st.slider('Temperatura', 0.0, 1.0, 0.7, 0.1)
    st.session_state.top_p = st.slider('Top P', 0.0, 1.0, 0.9, 0.1)
    st.session_state.top_k = st.slider('Top K', 0, 100, 40, 1)
    st.session_state.max_tokens = st.slider('Max Tokens', 1, 10000, 10000, 1)

# Inicializar memoria de usuario
import uuid
from memory_manager import get_memory_for_user

if 'pdf_path' in st.session_state:
    user_id = st.session_state.pdf_path.replace("./", "").replace("/", "_")
else:
    user_id = f"anon_{uuid.uuid4()}"

if 'memory' not in st.session_state:
    st.session_state.memory = get_memory_for_user(user_id=user_id)

# --- Inicializar modelo y base vectorial
llm = setup_model()

# Siempre tener una base cargada en sesión
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = get_vectorstore()

# --- Procesamiento del PDF ---
if 'pdf_path' in st.session_state:
    st.radio(
        "¿Qué deseas hacer con el archivo cargado?",
        ["Solo consultar temporalmente", "Agregar a la base de datos permanente"],
        key="pdf_action"
    )

    if st.button("📄 Procesar PDF"):
        if st.session_state.pdf_action == "Agregar a la base de datos permanente":
            append_pdf_to_vectorstore(
                st.session_state.pdf_path,
                chunk_size=1000,
                chunk_overlap=200
            )
            st.success("✅ PDF agregado a la base vectorial permanente.")
        else:
            st.session_state.vector_store_temp = get_temporary_vectorstore(
                st.session_state.pdf_path,
                chunk_size=st.session_state.chunk_size,
                chunk_overlap=st.session_state.chunk_overlap
            )
            st.success("📘 PDF cargado para consulta temporal.")

# --- Mostrar historial ---
for mensaje in st.session_state.historial:
    if mensaje["role"] == "user":
        with st.chat_message("user"):
            st.markdown(mensaje["content"])
    elif mensaje["role"] == "assistant":
        with st.chat_message("ai", avatar="🤖"):
            st.markdown(mensaje["content"])
            with st.expander("📊 Metadatos de la respuesta"):
                metadata = mensaje.get("metadata", {})
                if metadata:
                    st.write({
                        "Modelo": metadata.get("model", ""),
                        "Tokens del prompt": metadata.get("prompt_eval_count", 0),
                        "Tokens generados": metadata.get("eval_count", 0),
                        "Duración total (s)": round(metadata.get("total_duration", 0) / 1e9, 3),
                        "Razón de finalización": metadata.get("done_reason", ""),
                        "Creado en": metadata.get("created_at", "")
                    })

# --- Entrada del usuario ---
if user_input := st.chat_input("Haz tu pregunta"):
    if 'vector_store_temp' not in st.session_state and st.session_state.get("vector_store") is None:
        st.error("❌ No hay base vectorial disponible. Carga un documento o asegúrate de tener una base precargada.")
    else:
        with st.chat_message("user"):
            st.markdown(user_input)

        st.session_state.historial.append({
            "role": "user",
            "content": user_input
        })

        with st.spinner("🤖 Agente RAG Buscando..."):
            vs = st.session_state.get("vector_store_temp", st.session_state.get("vector_store"))
            result_generator = ask_question(user_input, llm, vs)

            with st.chat_message("ai", avatar="🤖"):
                respuesta_completa = ""
                msg_placeholder = st.empty()

                # --- Streaming de texto y captura de metadatos ---
                for texto, metadata in result_generator:
                    respuesta_completa += texto
                    msg_placeholder.markdown(respuesta_completa + "▌")

                msg_placeholder.markdown(respuesta_completa)  # Respuesta final

                with st.expander("📊 Metadatos de la respuesta"):
                    if metadata:
                        st.write({
                            "Modelo": metadata.get("model", ""),
                            "Tokens del prompt": metadata.get("prompt_eval_count", 0),
                            "Tokens generados": metadata.get("eval_count", 0),
                            "Duración total (s)": round(metadata.get("total_duration", 0) / 1e9, 3),
                            "Razón de finalización": metadata.get("done_reason", ""),
                            "Creado en": metadata.get("created_at", "")
                        })

        st.session_state.historial.append({
            "role": "assistant",
            "content": respuesta_completa,
            "metadata": metadata
        })


