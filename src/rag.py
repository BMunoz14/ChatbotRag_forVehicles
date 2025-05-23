# rag.py
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain import hub
from langgraph.graph import START, StateGraph
from langchain_core.documents import Document
from typing_extensions import TypedDict, List
import streamlit as st

# --- Configuración inicial ---
load_dotenv()
#os.environ["LANGCHAIN_TRACING_V2"]  = "true"
#os.environ["LANGCHAIN_API_KEY"]     = os.getenv("LANGCHAIN_API_KEY")
#os.environ["USER_AGENT"]            = "AgenteUAO"
#os.environ["LANGCHAIN_PROJECT"]     = "OllamaRAG_V2"

# --- Prompt base ---
prompt = hub.pull("rlm/rag-prompt")

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# --- Setup modelo (vectorstore se carga aparte) ---
def setup_model():
    llm = ChatOllama(
        model=st.session_state.model_selection,
        temperature=st.session_state.temperature,
        top_p=st.session_state.top_p,
        top_k=st.session_state.top_k,
        num_predict=st.session_state.max_tokens,
    )
    return llm

# --- Recuperar contexto ---
def retrieve_context(state: State, vector_store):
    question = state["question"]
    return {"context": vector_store.similarity_search(question, k=5)}

# --- Generar respuesta con streaming ---
def generate_answer(state: State, llm):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    history = "\n".join([
        f"Usuario: {msg['content']}" if msg['role'] == 'user' else f"Asistente: {msg['content']}"
        for msg in st.session_state.get("historial", [])
    ])
    full_context = f"{history}\n\n{docs_content}"

    messages = prompt.invoke({"question": state["question"], "context": full_context})

    metadata = {}
    for chunk in llm.stream(messages):
        if hasattr(chunk, "response_metadata") and isinstance(chunk.response_metadata, dict):
            metadata = chunk.response_metadata.copy()

        if hasattr(chunk, "content") and chunk.content:
            yield chunk.content, metadata


# --- Pipeline de respuesta RAG ---
def ask_question(raw_question: str, llm, vector_store):
    # Recuperar contexto solamente usando LangGraph
    graph_builder = StateGraph(State)
    graph_builder.add_node("retrieve_context", lambda state: retrieve_context(state, vector_store))
    graph_builder.set_entry_point("retrieve_context")
    graph = graph_builder.compile()

    context_state = graph.invoke({"question": raw_question})
    docs = context_state["context"]

    # Hacer streaming por fuera del grafo
    docs_content = "\n\n".join(doc.page_content for doc in docs)
    history = "\n".join([
        f"Usuario: {msg['content']}" if msg['role'] == 'user' else f"Asistente: {msg['content']}"
        for msg in st.session_state.get("historial", [])
    ])
    full_context = f"{history}\n\n{docs_content}"

    messages = prompt.invoke({"question": raw_question, "context": full_context})

    metadata = {}
    for chunk in llm.stream(messages):
        if not metadata and hasattr(chunk, "response_metadata"):
            metadata = chunk.response_metadata.copy()
        if hasattr(chunk, "content") and chunk.content:
            yield chunk.content, metadata


