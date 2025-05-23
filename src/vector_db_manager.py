# vector_db_manager.py

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from typing import List

# Configuración
DB_PATH = "./chroma_langchain_db"
COLLECTION_NAME = "pdf_collection"
EMBEDDING_MODEL = "nomic-embed-text:latest"

# Cargar base vectorial existente o crear nueva
def get_vectorstore():
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=DB_PATH
    )

# Cargar y dividir PDF
def load_and_split_pdf(pdf_path: str, chunk_size=1000, chunk_overlap=200) -> List[Document]:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

# Crear base vectorial desde 0 con múltiples PDFs
def create_new_vectorstore_from_pdfs(pdf_paths: List[str], chunk_size=1000, chunk_overlap=200):
    vector_store = get_vectorstore()
    all_chunks = []
    for path in pdf_paths:
        if os.path.exists(path):
            all_chunks.extend(load_and_split_pdf(path, chunk_size, chunk_overlap))
    vector_store.reset_collection()
    vector_store.add_documents(all_chunks)
    return vector_store

# Añadir un nuevo PDF a la base existente
def append_pdf_to_vectorstore(pdf_path: str, chunk_size=1000, chunk_overlap=200):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Archivo no encontrado: {pdf_path}")
    vector_store = get_vectorstore()
    chunks = load_and_split_pdf(pdf_path, chunk_size, chunk_overlap)
    vector_store.add_documents(chunks)

# Solo para consulta temporal (no se guarda)
def get_temporary_vectorstore(pdf_path: str, chunk_size=1000, chunk_overlap=200):
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    temp_vector_store = Chroma(
        collection_name="temp_collection",
        embedding_function=embeddings,
        persist_directory=None  # no se guarda
    )
    chunks = load_and_split_pdf(pdf_path, chunk_size, chunk_overlap)
    temp_vector_store.add_documents(chunks)
    return temp_vector_store
